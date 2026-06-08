"""
SFH Thumbnail Duplicate Query
==============================
Given an image and a level ID, finds visually similar thumbnails among all
songs associated with that level in the SFH database.

Flow:
  1. MongoDB: find all songs where levelID array contains the given level ID
  2. Sidecar: filter to doc IDs that exist in the FAISS index
  3. Precompute all features for the query image (DINOv3 + hashes + DCT + grid + ORB)
  4. FAISS: restricted search over candidate doc IDs only
  5. For each FAISS hit: load pre-cached features, run all math directly
     (no PIL images needed for candidates, no DINOv3 forward pass)
  6. Print ranked similarity report

Usage:
  python query.py --uri "mongodb://localhost:27017" --image thumb.jpg --level 59549363
  python query.py --uri "mongodb://localhost:27017" --image thumb.jpg --level 59549363 --top 10
  python query.py --uri "mongodb://localhost:27017" --image thumb.jpg --level 59549363 --threshold 0.70
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import cv2
import faiss
import imagehash
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pymongo import MongoClient
from scipy.fft import dctn

from image_similarity import (
    PipelineConfig,
    StageScores,
    SimilarityReport,
    print_report,
    _load_image,
    _CV2_AVAILABLE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths — must match build_index.py
# ─────────────────────────────────────────────────────────────────────────────

INDEX_DIR     = Path(os.environ.get("INDEX_DIR", "sfh_index"))
INDEX_PATH    = INDEX_DIR / "index.faiss"
SIDECAR_PATH  = INDEX_DIR / "sidecar.json"
EMB_CACHE_DIR = Path(os.environ.get("EMB_CACHE_DIR", ".embedding_cache"))

MODEL_NAME    = "facebook/dinov3-vitl16-pretrain-lvd1689m"
EMBEDDING_DIM = 1024

CANONICAL_SIZE = (480, 360)  # must match build_index.py — all images normalised to this

# Must match build_index.py constants
HASH_SIZE        = 16
GRID_ROWS        = 8
GRID_COLS        = 8
GRID_HASH_SIZE   = 8
DCT_RESIZE       = (64, 64)
ORB_MAX_FEATURES = 5000


# ─────────────────────────────────────────────────────────────────────────────
# Sidecar
# ─────────────────────────────────────────────────────────────────────────────

class Sidecar:
    def __init__(self, data: dict):
        self.by_faiss: dict[str, str] = data.get("by_faiss", {})
        self.by_mongo: dict[str, int] = {k: int(v) for k, v in data.get("by_mongo", {}).items()}
        self.emb_key:  dict[str, str] = data.get("emb_key", {})

    @classmethod
    def load(cls) -> "Sidecar":
        if not SIDECAR_PATH.exists():
            raise FileNotFoundError(f"Sidecar not found. Run: python build_index.py build --uri ...")
        return cls(json.loads(SIDECAR_PATH.read_text()))

    def faiss_id_for(self, mongo_id: str) -> Optional[int]:
        return self.by_mongo.get(mongo_id)

    def mongo_id_for(self, faiss_id: int) -> Optional[str]:
        return self.by_faiss.get(str(faiss_id))

    def emb_path_for(self, mongo_id: str) -> Optional[Path]:
        key = self.emb_key.get(mongo_id)
        return (EMB_CACHE_DIR / f"{key}.pt") if key else None

    def __contains__(self, mongo_id: str) -> bool:
        return mongo_id in self.by_mongo


# ─────────────────────────────────────────────────────────────────────────────
# Query image feature computation
# Mirrors precompute_image_features() in build_index.py, plus DINOv3
# ─────────────────────────────────────────────────────────────────────────────

def compute_query_features(
    img:       Image.Image,
    processor=None,
    model=None,
    device=None,
) -> dict:
    """
    Compute all features for the query image in one pass.
    Returns the same dict structure as the cached .pt files so the
    comparison math is identical for both query and candidate sides.

    Pass an already-loaded `processor`/`model`/`device` (e.g. an Embedder's)
    to reuse weights resident in memory. Otherwise a fresh copy of MODEL_NAME
    is loaded from scratch — fine for one-shot CLI use, but far too slow to
    do on every request in a long-lived server.
    """
    # Normalise to canonical resolution — must match what build_index.py stores
    # so that ORB descriptors, hash bits, and DINOv3 patch grids are comparable
    if img.size != CANONICAL_SIZE:
        log.info(f"  Resizing query image {img.size} → {CANONICAL_SIZE}")
        img = img.resize(CANONICAL_SIZE, Image.LANCZOS)

    features = {}

    # ── DINOv3 (CLS + patches) ───────────────────────────────────────────────
    owns_model = model is None
    if owns_model:
        from transformers import AutoImageProcessor, AutoModel
        log.info(f"Loading DINOv3: {MODEL_NAME}")
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        model     = AutoModel.from_pretrained(MODEL_NAME, device_map="auto")
        model.eval()
        device = next(model.parameters()).device
        log.info(f"  Device: {device}")

    with torch.no_grad():
        inputs  = processor(images=img, return_tensors="pt")
        inputs  = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        cls = outputs.pooler_output
    else:
        cls = outputs.last_hidden_state[:, 0, :]

    cls     = F.normalize(cls, p=2, dim=-1)
    patches = outputs.last_hidden_state[:, 5:, :]
    patches = F.normalize(patches.squeeze(0), p=2, dim=-1)

    features["cls"]     = cls.cpu()        # (1, D)
    features["patches"] = patches.cpu()   # (196, D)
    features["_device"] = device          # keep for patch math later

    # Free GPU memory — only for a model we loaded ourselves; a passed-in
    # model is owned by the caller (e.g. server's long-lived embedder).
    del outputs
    if owns_model:
        del model, processor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Perceptual hashes ────────────────────────────────────────────────────
    log.info("  Computing hashes + DCT + grid + ORB...")
    features["hash_phash"]    = imagehash.phash(img, hash_size=HASH_SIZE).hash.flatten()
    features["hash_dhash"]    = imagehash.dhash(img, hash_size=HASH_SIZE).hash.flatten()
    features["hash_ahash"]    = imagehash.average_hash(img, hash_size=HASH_SIZE).hash.flatten()
    features["hash_whash"]    = imagehash.whash(img, hash_size=HASH_SIZE).hash.flatten()
    features["hash_size"]     = HASH_SIZE

    # ── DCT band energies ────────────────────────────────────────────────────
    gray  = np.array(img.convert("L").resize(DCT_RESIZE), dtype=np.float64)
    dct   = dctn(gray, norm="ortho")
    n     = DCT_RESIZE[0]
    low   = np.sum(dct[:n//4, :n//4] ** 2)
    high  = np.sum(dct[n//2:,  n//2:] ** 2)
    total = np.sum(dct ** 2) + 1e-9
    mid   = total - low - high
    features["dct_bands"] = np.array([low/total, mid/total, high/total])

    # ── Sector grid hashes ───────────────────────────────────────────────────
    w, h     = img.size
    cell_w   = w // GRID_COLS
    cell_h   = h // GRID_ROWS
    gh_size  = GRID_HASH_SIZE
    grid_arr = np.zeros((GRID_ROWS, GRID_COLS, gh_size * gh_size), dtype=bool)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            box  = (c*cell_w, r*cell_h, (c+1)*cell_w, (r+1)*cell_h)
            cell = img.crop(box)
            grid_arr[r, c] = imagehash.phash(cell, hash_size=gh_size).hash.flatten()
    features["grid_hashes"]    = grid_arr
    features["grid_rows"]      = GRID_ROWS
    features["grid_cols"]      = GRID_COLS
    features["grid_hash_size"] = gh_size

    # ── ORB keypoints + descriptors ──────────────────────────────────────────
    gray_np    = np.array(img.convert("L"))
    orb        = cv2.ORB_create(nfeatures=ORB_MAX_FEATURES)
    kps, descs = orb.detectAndCompute(gray_np, None)
    if kps and descs is not None:
        features["orb_kp_pts"]    = np.array([kp.pt    for kp in kps], dtype=np.float32)
        features["orb_kp_sizes"]  = np.array([kp.size  for kp in kps], dtype=np.float32)
        features["orb_kp_angles"] = np.array([kp.angle for kp in kps], dtype=np.float32)
        features["orb_descs"]     = descs
    else:
        features["orb_kp_pts"]    = np.zeros((0, 2),  dtype=np.float32)
        features["orb_kp_sizes"]  = np.zeros((0,),    dtype=np.float32)
        features["orb_kp_angles"] = np.zeros((0,),    dtype=np.float32)
        features["orb_descs"]     = np.zeros((0, 32), dtype=np.uint8)

    features["img_size"] = img.size
    return features


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise math — all operations on precomputed features, no PIL needed
# ─────────────────────────────────────────────────────────────────────────────

def _hamming_score(bits_a: np.ndarray, bits_b: np.ndarray) -> float:
    """Normalised Hamming similarity [0, 1] from two boolean bit arrays."""
    max_bits = len(bits_a)
    dist     = int(np.sum(bits_a != bits_b))
    return max(0.0, 1.0 - dist / max_bits)


def compare_features(q: dict, c: dict, cfg: PipelineConfig) -> SimilarityReport:
    """
    Run all similarity stages using precomputed feature dicts.
    No PIL images, no DINOv3, no file I/O — pure math.
    """
    import time
    t0         = time.perf_counter()
    scores     = StageScores()
    stages_run = []

    # ── Hash ensemble ────────────────────────────────────────────────────────
    ph = _hamming_score(q["hash_phash"], c["hash_phash"])
    dh = _hamming_score(q["hash_dhash"], c["hash_dhash"])
    ah = _hamming_score(q["hash_ahash"], c["hash_ahash"])
    wh = _hamming_score(q["hash_whash"], c["hash_whash"])
    scores.hash_ensemble = ph*0.35 + dh*0.35 + ah*0.15 + wh*0.15
    stages_run.append("hash_ensemble")

    # ── DCT spectrum ─────────────────────────────────────────────────────────
    e1, e2 = q["dct_bands"], c["dct_bands"]
    scores.dct_spectrum = float(
        np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9)
    )
    stages_run.append("dct_spectrum")

    # ── Sector grid ──────────────────────────────────────────────────────────
    gh1      = q["grid_hashes"]   # (R, C, bits)
    gh2      = c["grid_hashes"]
    max_bits = GRID_HASH_SIZE * GRID_HASH_SIZE
    matrix   = np.zeros((GRID_ROWS, GRID_COLS))
    for r in range(GRID_ROWS):
        for cc in range(GRID_COLS):
            dist         = int(np.sum(gh1[r, cc] != gh2[r, cc]))
            matrix[r,cc] = max(0.0, 1.0 - dist / max_bits)
    scores.sector_grid = float(np.mean(matrix))
    stages_run.append("sector_grid")

    # ── Geometric alignment (ORB match + RANSAC on cached descriptors) ────────
    # Reconstruct cv2.KeyPoint objects from cached arrays for BFMatcher
    q_pts, q_sizes, q_angles, q_descs = (
        q["orb_kp_pts"], q["orb_kp_sizes"], q["orb_kp_angles"], q["orb_descs"]
    )
    c_pts, c_sizes, c_angles, c_descs = (
        c["orb_kp_pts"], c["orb_kp_sizes"], c["orb_kp_angles"], c["orb_descs"]
    )

    scale = rot_deg = None
    align_applied = False

    if (len(q_descs) >= 4 and len(c_descs) >= 4 and _CV2_AVAILABLE):
        q_kps = [cv2.KeyPoint(x=float(q_pts[i,0]), y=float(q_pts[i,1]),
                              size=float(q_sizes[i]), angle=float(q_angles[i]))
                 for i in range(len(q_pts))]
        c_kps = [cv2.KeyPoint(x=float(c_pts[i,0]), y=float(c_pts[i,1]),
                              size=float(c_sizes[i]), angle=float(c_angles[i]))
                 for i in range(len(c_pts))]

        bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw     = bf.knnMatch(c_descs, q_descs, k=2)
        good    = [m for pair in raw if len(pair)==2
                   for m, n in [pair] if m.distance < cfg.align_match_keep_ratio * n.distance]

        if len(good) >= cfg.align_min_inliers:
            pts_c = np.float32([c_kps[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            pts_q = np.float32([q_kps[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            H, mask = cv2.findHomography(pts_c, pts_q, cv2.RANSAC, 4.0)
            if H is not None:
                inliers = int(mask.sum()) if mask is not None else 0
                sx    = np.sqrt(H[0,0]**2 + H[1,0]**2)
                sy    = np.sqrt(H[0,1]**2 + H[1,1]**2)
                scale = float((sx+sy)/2)
                rot_deg = float(np.degrees(np.arctan2(H[1,0], H[0,0])))

                if (inliers >= cfg.align_min_inliers
                        and scale <= cfg.align_max_scale
                        and (1.0/cfg.align_max_scale) <= scale
                        and abs(rot_deg) <= cfg.align_max_rotation_deg):
                    # Alignment is valid — adjust grid matrix by the detected scale
                    # (we can't re-warp without PIL, but we record the transform)
                    align_applied = True
                    scores.align_inliers      = inliers
                    stages_run.append(f"align[scale={scale:.3f}x,rot={rot_deg:.1f}°,inliers={inliers}]")
                else:
                    stages_run.append("align[skipped:implausible_transform]")
            else:
                stages_run.append("align[skipped:ransac_failed]")
        else:
            stages_run.append(f"align[skipped:too_few_matches({len(good)})]")
    else:
        stages_run.append("align[skipped:insufficient_keypoints]")

    scores.align_scale        = scale
    scores.align_rotation_deg = rot_deg
    scores.align_applied      = align_applied

    # ── DINOv3 global ────────────────────────────────────────────────────────
    device  = q.get("_device", torch.device("cpu"))
    q_cls   = q["cls"].to(device)
    c_cls   = c["cls"].to(device)
    scores.global_semantic = float(F.cosine_similarity(q_cls, c_cls).clamp(0, 1))
    stages_run.append("dinov3_global")

    # ── DINOv3 patch spatial (full 196×196) ──────────────────────────────────
    p1         = q["patches"].to(device)   # (196, D)
    p2         = c["patches"].to(device)   # (196, D)
    sim_matrix = p1 @ p2.T                 # (196, 196)
    grid_size  = 14
    positions  = torch.stack(torch.meshgrid(
        torch.arange(grid_size, device=device),
        torch.arange(grid_size, device=device),
        indexing="ij",
    ), dim=-1).reshape(196, 2).float()
    pos_dist    = torch.cdist(positions, positions)
    sigma       = grid_size / 3.0
    pos_penalty = torch.exp(-pos_dist**2 / (2 * sigma**2))
    scores.patch_spatial = float(
        (sim_matrix * pos_penalty).max(dim=1).values.mean().clamp(0, 1)
    )
    stages_run.append("dinov3_patch")

    # ── Composite ────────────────────────────────────────────────────────────
    total_w   = (cfg.weight_hash + cfg.weight_grid + cfg.weight_dct
                 + cfg.weight_global + cfg.weight_patch)
    composite = (
        scores.hash_ensemble   * cfg.weight_hash   +
        scores.sector_grid     * cfg.weight_grid    +
        scores.dct_spectrum    * cfg.weight_dct     +
        scores.global_semantic * cfg.weight_global  +
        scores.patch_spatial   * cfg.weight_patch
    ) / total_w
    # Grid variance penalty
    grid_var = float(np.var(matrix))
    composite = max(0.0, composite - grid_var * cfg.grid_variance_penalty)

    # Grid veto: cap composite if sector grid mean is too low
    if scores.sector_grid < 0.65:
        composite = min(composite, scores.sector_grid * 1.1)

    scores.composite = float(composite)

    # ── Verdict ──────────────────────────────────────────────────────────────
    VERDICTS = {
        "EXACT":            "EXACT DUPLICATE — byte/pixel-level identical or losslessly re-encoded",
        "MICRO_VARIANT":    "MICRO VARIANT — same base image; minor mutation (crop, watermark, text overlay, compression)",
        "MODIFIED_VARIANT": "MODIFIED VARIANT — same scene/subject; significant layout or content change",
        "DISTANT":          "DISTANT RELATIVE — loosely related (same template/style, different content)",
        "DISTINCT":         "DISTINCT IMAGES — no meaningful similarity detected",
    }
    if   composite >= cfg.thresh_exact:    verdict, code = VERDICTS["EXACT"],            "EXACT"
    elif composite >= cfg.thresh_micro:    verdict, code = VERDICTS["MICRO_VARIANT"],    "MICRO_VARIANT"
    elif composite >= cfg.thresh_variant:  verdict, code = VERDICTS["MODIFIED_VARIANT"], "MODIFIED_VARIANT"
    elif composite >= cfg.thresh_distant:  verdict, code = VERDICTS["DISTANT"],          "DISTANT"
    else:                                  verdict, code = VERDICTS["DISTINCT"],          "DISTINCT"

    elapsed = (time.perf_counter() - t0) * 1000
    return SimilarityReport(
        path_a="<query>",
        path_b="<candidate>",
        scores=scores,
        verdict=verdict,
        verdict_code=code,
        grid_matrix=matrix.tolist(),
        grid_min_cell=float(np.min(matrix)),
        grid_variance=float(np.var(matrix)),
        elapsed_ms=elapsed,
        stages_run=stages_run,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Restricted FAISS search
# ─────────────────────────────────────────────────────────────────────────────

def restricted_search(
    index:             faiss.IndexIDMap,
    query_vec:         np.ndarray,
    allowed_faiss_ids: list[int],
    top_k:             int,
) -> list[tuple[int, float]]:
    """
    Search the full index and filter to allowed_faiss_ids after retrieval.
    Simpler and fully compatible with faiss-cpu (SearchParametersIDMap is
    GPU-only and not available in faiss-cpu).
    At 22k vectors a full search is <5ms so no meaningful performance cost.
    """
    if not allowed_faiss_ids:
        return []

    allowed_set = set(allowed_faiss_ids)

    # Search the full index for more candidates than needed so we have
    # enough after filtering — retrieve min(len(index), allowed*4) at most
    k_retrieve = min(index.ntotal, max(len(allowed_faiss_ids), top_k * 4))
    scores, ids = index.search(query_vec, k_retrieve)

    results = [
        (int(fid), float(s))
        for s, fid in zip(scores[0], ids[0])
        if fid != -1 and int(fid) in allowed_set
    ]
    return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# Main query
# ─────────────────────────────────────────────────────────────────────────────

def run_query(
    mongo_uri:        str,
    image_path:       str,
    level_id:         str,
    top_k:            int   = 5,
    faiss_candidates: int   = 20,
    threshold:        float = 0.50,
) -> None:
    t_total = time.perf_counter()

    if not Path(image_path).exists():
        log.error(f"Image not found: {image_path}")
        sys.exit(1)
    if not INDEX_PATH.exists():
        log.error(f"FAISS index not found. Run: python build_index.py build --uri ...")
        sys.exit(1)

    # ── Step 1: MongoDB ───────────────────────────────────────────────────────
    log.info(f"Querying MongoDB for level {level_id}...")
    client     = MongoClient(mongo_uri)
    collection = client["SFH"]["songs"]
    # Accept single string or list — $in on array field matches any element
    level_id_query = level_id if isinstance(level_id, list) else [level_id]
    docs       = list(collection.find(
        {"levelID": {"$in": level_id_query}},
        {"_id": 1, "ytVideoID": 1, "songName": 1, "name": 1},
    ))
    client.close()

    if not docs:
        log.warning(f"No songs found with levelID in {level_id_query}")
        return
    log.info(f"  {len(docs)} songs for level {level_id}")

    # ── Step 2: Filter to indexed docs ───────────────────────────────────────
    sidecar = Sidecar.load()
    index   = faiss.read_index(str(INDEX_PATH))

    indexed = [d for d in docs if str(d["_id"]) in sidecar]
    if len(indexed) < len(docs):
        log.warning(f"  {len(docs)-len(indexed)} docs not in index (run build_index.py to update)")
    if not indexed:
        log.error("None of this level's songs are indexed.")
        return
    log.info(f"  {len(indexed)} indexed candidates")

    allowed_faiss_ids = [sidecar.faiss_id_for(str(d["_id"])) for d in indexed]

    # ── Step 3: Compute all query features ───────────────────────────────────
    log.info("Computing query image features...")
    query_img      = _load_image(image_path)
    t_feat         = time.perf_counter()
    query_features = compute_query_features(query_img)
    log.info(f"  Query features computed in {(time.perf_counter()-t_feat)*1000:.0f}ms")

    # Extract CLS vector for FAISS
    query_vec = query_features["cls"].float().numpy()  # (1, D)

    # ── Step 4: Restricted FAISS search ──────────────────────────────────────
    log.info(f"FAISS search over {len(allowed_faiss_ids)} candidates...")
    faiss_hits = restricted_search(
        index, query_vec, allowed_faiss_ids,
        top_k=min(faiss_candidates, len(allowed_faiss_ids)),
    )
    if not faiss_hits:
        log.warning("FAISS returned no results.")
        return

    doc_by_id = {str(d["_id"]): d for d in indexed}

    log.info("Top FAISS hits (coarse cosine):")
    candidates = []
    for fid, coarse in faiss_hits:
        mid = sidecar.mongo_id_for(fid)
        if mid and mid in doc_by_id:
            doc   = doc_by_id[mid]
            label = doc.get("songName") or doc.get("name") or mid
            log.info(f"  {label[:55]:<55} cosine={coarse:.4f}")
            candidates.append((mid, coarse, doc))

    # ── Step 5: Full pipeline using cached features ───────────────────────────
    log.info(f"\nRunning pipeline on {len(candidates)} candidates (pure math)...")
    cfg     = PipelineConfig(use_ocr=False)
    reports = []

    for mid, coarse, doc in candidates:
        label    = doc.get("songName") or doc.get("name") or mid
        emb_path = sidecar.emb_path_for(mid)

        if emb_path is None or not emb_path.exists():
            log.warning(f"  {label}: no cached features — skipping")
            continue

        cand_features = torch.load(emb_path, map_location="cpu", weights_only=False)
        report        = compare_features(query_features, cand_features, cfg)

        if report.scores.composite >= threshold:
            reports.append((report, doc))
            log.info(f"  {label[:55]:<55} composite={report.scores.composite:.4f}  {report.verdict_code}")
        else:
            log.info(f"  {label[:55]:<55} composite={report.scores.composite:.4f}  [below threshold]")

    # ── Step 6: Output ────────────────────────────────────────────────────────
    reports.sort(key=lambda x: x[0].scores.composite, reverse=True)
    elapsed = time.perf_counter() - t_total

    W = 65
    print("\n" + "═" * W)
    print(f"  QUERY RESULTS — level {level_id}")
    print(f"  Query image  : {Path(image_path).name}")
    print(f"  Candidates   : {len(indexed)} indexed  |  {len(candidates)} passed FAISS")
    print(f"  Results      : {len(reports)} above threshold ({threshold})")
    print(f"  Total time   : {elapsed:.2f}s")
    print("═" * W)

    if not reports:
        print(f"\n  No matches found above threshold {threshold}.\n")
        return

    for report, doc in reports[:top_k]:
        label = doc.get("songName") or doc.get("name") or str(doc["_id"])
        report.path_a = Path(image_path).name
        report.path_b = doc.get("ytVideoID", str(doc["_id"]))
        print(f"\n  ── {label}")
        print(f"     MongoDB _id : {doc['_id']}")
        print(f"     YouTube ID  : {doc.get('ytVideoID', 'N/A')}")
        print_report(report, show_grid=True, show_ocr=False)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Query SFH thumbnail index for duplicates by level ID",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--uri",        required=True,            help="MongoDB connection URI")
    parser.add_argument("--image",      required=True,            help="Path to query thumbnail")
    parser.add_argument("--level",      required=True,            help="GD level ID to search within")
    parser.add_argument("--top",        type=int,   default=5,    help="Max results to display")
    parser.add_argument("--threshold",  type=float, default=0.50, help="Min composite score to show")
    parser.add_argument("--candidates", type=int,   default=20,   help="FAISS results to pass to pipeline")
    args = parser.parse_args()

    run_query(
        mongo_uri        = args.uri,
        image_path       = args.image,
        level_id         = args.level,
        top_k            = args.top,
        faiss_candidates = args.candidates,
        threshold        = args.threshold,
    )


if __name__ == "__main__":
    main()