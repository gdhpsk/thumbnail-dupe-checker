"""
SFH Thumbnail FAISS Index Builder
===================================
Reads ytVideoID from MongoDB SFH.songs, fetches thumbnails from YouTube,
embeds them with DINOv3 (CLS token only), and writes:

  sfh_index/
    index.faiss      — FAISS IndexIDMap wrapping IndexFlatIP (supports deletion)
    sidecar.json     — bidirectional map: faiss_id (int) <-> mongo _id (str)
                       + emb_key: mongo_id -> .pt cache filename (no extension)
    progress.json    — checkpoint of already-processed doc IDs (resumable runs)
    thumb_cache/     — cached raw JPEG bytes keyed by ytVideoID
  .embedding_cache/  — CLS + patch tensors per image as {pixel_hash}.pt

Read-only against MongoDB. Does not write to the database.

Usage:
  python build_index.py --uri "mongodb://localhost:27017"
  python build_index.py --uri "mongodb://localhost:27017" --batch 32   # larger GPU batch
  python build_index.py --delete <mongo_id> [<mongo_id> ...]           # remove entries

Incremental update (re-run anytime, skips already-processed docs):
  python build_index.py --uri "mongodb://localhost:27017"
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import struct
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import aiohttp
import cv2
import faiss
import imagehash
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pymongo import MongoClient
from scipy.fft import dctn
from transformers import AutoImageProcessor, AutoModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

INDEX_DIR       = Path("sfh_index")
INDEX_PATH      = INDEX_DIR / "index.faiss"
SIDECAR_PATH    = INDEX_DIR / "sidecar.json"
PROGRESS_PATH   = INDEX_DIR / "progress.json"
THUMB_CACHE_DIR = INDEX_DIR / "thumb_cache"

MODEL_NAME    = "facebook/dinov3-vitl16-pretrain-lvd1689m"
EMBEDDING_DIM = 1024  # ViT-L hidden size

EMB_CACHE_DIR = Path(".embedding_cache")  # shared with image_similarity.py

# ─────────────────────────────────────────────────────────────────────────────
# Per-image feature precomputation
#
# Everything that depends only on a single image is computed once here and
# saved into the .pt cache file alongside CLS + patches. At query time,
# candidates never need PIL images or DINOv3 — just load and do math.
#
# Stored in each .pt file:
#   cls          : (1, D) tensor   — L2-normalised CLS embedding
#   patches      : (196, D) tensor — L2-normalised patch embeddings
#   hash_phash   : (H*H,) bool ndarray  — pHash bits
#   hash_dhash   : (H*H,) bool ndarray  — dHash bits
#   hash_ahash   : (H*H,) bool ndarray  — aHash bits
#   hash_whash   : (H*H,) bool ndarray  — wHash bits
#   hash_size    : int
#   dct_bands    : (3,) float64 ndarray — low/mid/high band energy ratios
#   grid_hashes  : (R, C, H*H) bool ndarray — per-cell pHash bits
#   grid_rows    : int
#   grid_cols    : int
#   grid_hash_size: int
#   orb_kp_pts   : (N, 2) float32 ndarray — keypoint (x, y) coordinates
#   orb_kp_sizes : (N,) float32 ndarray   — keypoint sizes (for cv2.KeyPoint reconstruction)
#   orb_kp_angles: (N,) float32 ndarray   — keypoint angles
#   orb_descs    : (N, 32) uint8 ndarray  — ORB binary descriptors
#   img_size     : (w, h) tuple           — original image dimensions
# ─────────────────────────────────────────────────────────────────────────────

HASH_SIZE       = 16
GRID_ROWS       = 8
GRID_COLS       = 8
GRID_HASH_SIZE  = 8
DCT_RESIZE      = (64, 64)
ORB_MAX_FEATURES = 5000


def precompute_image_features(img: Image.Image) -> dict:
    """
    Compute all per-image features that can be cached.
    Resizes to CANONICAL_SIZE (320x180) first so all cached features are
    resolution-consistent regardless of which YouTube thumbnail size was fetched.
    Returns a dict suitable for np.savez / torch.save alongside the DINOv3 tensors.
    """
    # Normalise to canonical resolution (480x360) before any feature computation
    if img.size != CANONICAL_SIZE:
        img = img.resize(CANONICAL_SIZE, Image.LANCZOS)

    features = {}

    # ── Perceptual hashes ────────────────────────────────────────────────────
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
    gray_np = np.array(img.convert("L"))
    orb     = cv2.ORB_create(nfeatures=ORB_MAX_FEATURES)
    kps, descs = orb.detectAndCompute(gray_np, None)
    if kps and descs is not None:
        features["orb_kp_pts"]    = np.array([kp.pt    for kp in kps], dtype=np.float32)
        features["orb_kp_sizes"]  = np.array([kp.size  for kp in kps], dtype=np.float32)
        features["orb_kp_angles"] = np.array([kp.angle for kp in kps], dtype=np.float32)
        features["orb_descs"]     = descs
    else:
        features["orb_kp_pts"]    = np.zeros((0, 2),    dtype=np.float32)
        features["orb_kp_sizes"]  = np.zeros((0,),      dtype=np.float32)
        features["orb_kp_angles"] = np.zeros((0,),      dtype=np.float32)
        features["orb_descs"]     = np.zeros((0, 32),   dtype=np.uint8)

    features["img_size"] = img.size  # (w, h)
    return features


# hqdefault (480x360) — guaranteed to exist for every YouTube video.
YT_URL_TEMPLATES = [
    "https://img.youtube.com/vi/{vid}/hqdefault.jpg",
]

CANONICAL_SIZE = (480, 360)  # all images resized to this before feature computation

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=15)
MAX_RETRIES     = 3


# ─────────────────────────────────────────────────────────────────────────────
# ID mapping — MongoDB ObjectID string <-> FAISS int64
#
# FAISS IndexIDMap uses explicit int64 IDs. We derive a stable 63-bit integer
# from each MongoDB _id string via SHA-256, avoiding sign-bit issues.
# Collision probability over 22k IDs is negligible (~3e-11).
# ─────────────────────────────────────────────────────────────────────────────

def mongo_id_to_faiss_id(mongo_id: str) -> int:
    digest = hashlib.sha256(mongo_id.encode()).digest()
    return struct.unpack(">q", digest[:8])[0] & 0x7FFFFFFFFFFFFFFF


# ─────────────────────────────────────────────────────────────────────────────
# Sidecar — bidirectional map stored as JSON
#
# {
#   "by_faiss": {"<faiss_int_str>": "<mongo_id>", ...},
#   "by_mongo": {"<mongo_id>": <faiss_int>, ...}
# }
#
# JSON keys must be strings, so faiss IDs are stored as strings in by_faiss
# and as ints in by_mongo values.
# ─────────────────────────────────────────────────────────────────────────────

class Sidecar:
    def __init__(self):
        self.by_faiss: dict[str, str] = {}   # str(faiss_id) -> mongo_id
        self.by_mongo: dict[str, int] = {}   # mongo_id -> faiss_id
        self.emb_key:  dict[str, str] = {}   # mongo_id -> pixel hash (no ext) for .pt lookup

    @classmethod
    def load(cls) -> "Sidecar":
        s = cls()
        if SIDECAR_PATH.exists():
            data = json.loads(SIDECAR_PATH.read_text())
            s.by_faiss = data.get("by_faiss", {})
            s.by_mongo = {k: int(v) for k, v in data.get("by_mongo", {}).items()}
            s.emb_key  = data.get("emb_key", {})
        return s

    def save(self) -> None:
        SIDECAR_PATH.write_text(json.dumps(
            {"by_faiss": self.by_faiss, "by_mongo": self.by_mongo, "emb_key": self.emb_key},
            indent=2,
        ))

    def add(self, mongo_id: str, faiss_id: int, pixel_hash: str) -> None:
        self.by_faiss[str(faiss_id)] = mongo_id
        self.by_mongo[mongo_id] = faiss_id
        self.emb_key[mongo_id]  = pixel_hash

    def remove(self, mongo_id: str) -> Optional[int]:
        faiss_id = self.by_mongo.pop(mongo_id, None)
        if faiss_id is not None:
            self.by_faiss.pop(str(faiss_id), None)
        self.emb_key.pop(mongo_id, None)
        return faiss_id

    def faiss_id_for(self, mongo_id: str) -> Optional[int]:
        return self.by_mongo.get(mongo_id)

    def mongo_id_for(self, faiss_id: int) -> Optional[str]:
        return self.by_faiss.get(str(faiss_id))

    def emb_path_for(self, mongo_id: str) -> Optional[Path]:
        """Returns the .pt file path for a doc ID, or None if not recorded."""
        key = self.emb_key.get(mongo_id)
        if key is None:
            return None
        return EMB_CACHE_DIR / f"{key}.pt"

    def __contains__(self, mongo_id: str) -> bool:
        return mongo_id in self.by_mongo

    def __len__(self) -> int:
        return len(self.by_mongo)


# ─────────────────────────────────────────────────────────────────────────────
# Progress checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_progress() -> set[str]:
    if PROGRESS_PATH.exists():
        return set(json.loads(PROGRESS_PATH.read_text()).get("done", []))
    return set()

def save_progress(done: set[str]) -> None:
    PROGRESS_PATH.write_text(json.dumps({"done": sorted(done)}, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# FAISS index helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_or_create_index() -> faiss.IndexIDMap:
    if INDEX_PATH.exists():
        log.info("Loading existing FAISS index...")
        return faiss.read_index(str(INDEX_PATH))
    log.info(f"Creating new FAISS IndexIDMap(IndexFlatIP) dim={EMBEDDING_DIM}")
    flat = faiss.IndexFlatIP(EMBEDDING_DIM)
    return faiss.IndexIDMap(flat)

def save_index(index: faiss.IndexIDMap) -> None:
    faiss.write_index(index, str(INDEX_PATH))


# ─────────────────────────────────────────────────────────────────────────────
# Thumbnail fetcher (async)
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_thumbnail(
    session: aiohttp.ClientSession,
    video_id: str,
) -> Optional[Image.Image]:
    cache_path = THUMB_CACHE_DIR / f"{video_id}.jpg"
    if cache_path.exists():
        try:
            return Image.open(cache_path).convert("RGB")
        except Exception:
            cache_path.unlink(missing_ok=True)

    for template in YT_URL_TEMPLATES:
        url = template.format(vid=video_id)
        for attempt in range(MAX_RETRIES):
            try:
                async with session.get(url, timeout=REQUEST_TIMEOUT) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        img = Image.open(BytesIO(data)).convert("RGB")
                        if img.size == (120, 90):
                            break  # YouTube placeholder image, try next resolution
                        cache_path.write_bytes(data)
                        return img
                    elif resp.status == 404:
                        break
                    else:
                        await asyncio.sleep(0.5 * (attempt + 1))
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == MAX_RETRIES - 1:
                    log.warning(f"  Fetch failed {url}: {e}")
                else:
                    await asyncio.sleep(0.5 * (attempt + 1))

    log.warning(f"  No thumbnail available for video_id={video_id}")
    return None


async def fetch_batch_async(
    session: aiohttp.ClientSession,
    batch: list[dict],
    concurrency: int,
) -> list[tuple[str, Optional[Image.Image]]]:
    sem = asyncio.Semaphore(concurrency)
    async def one(doc):
        async with sem:
            img = await fetch_thumbnail(session, doc["ytVideoID"])
            return (str(doc["_id"]), img)
    return await asyncio.gather(*[one(d) for d in batch])


# ─────────────────────────────────────────────────────────────────────────────
# Embedder
# ─────────────────────────────────────────────────────────────────────────────

class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        log.info(f"Loading DINOv3: {model_name}")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map="auto")
        self.model.eval()
        self.device = next(self.model.parameters()).device
        EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        log.info(f"Model on device: {self.device}")

    @staticmethod
    def _pil_hash(img: Image.Image) -> str:
        """SHA-256 of raw pixel bytes — stable cache key for an in-memory image."""
        return hashlib.sha256(img.tobytes()).hexdigest()

    @torch.no_grad()
    def embed_batch(
        self,
        images: list[Image.Image],
    ) -> tuple[np.ndarray, list[str]]:
        """
        Embed a batch of images.

        Returns:
          cls_embs   : (N, D) float32 numpy array of L2-normalised CLS embeddings
                       (used for FAISS indexing)
          pixel_hashes: list of N hex strings — cache keys for the saved .pt files

        Also saves CLS + patch tensors to EMB_CACHE_DIR/{pixel_hash}.pt so that
        query.py can load them directly without re-running DINOv3.
        Skips saving if the .pt file already exists (idempotent).
        """
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        # CLS / global token
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            cls_batch = outputs.pooler_output                        # (N, D)
        else:
            cls_batch = outputs.last_hidden_state[:, 0, :]          # (N, D)
        cls_batch = F.normalize(cls_batch, p=2, dim=-1)

        # Patch tokens — DINOv3: [CLS, reg0..reg3, patch0..patch195] → indices 5:201
        patch_batch = outputs.last_hidden_state[:, 5:, :]           # (N, 196, D)
        patch_batch = F.normalize(patch_batch, p=2, dim=-1)

        cls_np       = cls_batch.cpu().float().numpy()               # (N, D) for FAISS
        pixel_hashes = [self._pil_hash(img) for img in images]

        # Persist per-image cache to disk — DINOv3 tensors + all precomputed features
        # Resize to canonical size before precomputing so cache is resolution-consistent
        images_canonical = [
            img.resize(CANONICAL_SIZE, Image.LANCZOS) if img.size != CANONICAL_SIZE else img
            for img in images
        ]
        for i, (phash, img) in enumerate(zip(pixel_hashes, images_canonical)):
            pt_path = EMB_CACHE_DIR / f"{phash}.pt"
            if not pt_path.exists():
                cache = {
                    "cls":     cls_batch[i].unsqueeze(0).cpu(),  # (1, D)
                    "patches": patch_batch[i].cpu(),             # (196, D)
                }
                cache.update(precompute_image_features(img))
                torch.save(cache, pt_path)

        return cls_np, pixel_hashes


# ─────────────────────────────────────────────────────────────────────────────
# Build / update
# ─────────────────────────────────────────────────────────────────────────────

def build_index(
    mongo_uri: str,
    embed_batch_size: int = 16,
    fetch_concurrency: int = 100,
    mongo_batch_size:  int = 500,
) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Connecting to MongoDB...")
    client     = MongoClient(mongo_uri)
    collection = client["SFH"]["songs"]
    total_docs = collection.count_documents({})
    log.info(f"Found {total_docs} documents in SFH.songs")

    done_ids = load_progress()
    sidecar  = Sidecar.load()
    index    = load_or_create_index()
    log.info(f"Already indexed: {len(sidecar)} vectors | processed: {len(done_ids)}")

    cursor = collection.find(
        {"ytVideoID": {"$exists": True, "$ne": None, "$ne": ""}},
        {"_id": 1, "ytVideoID": 1},
    ).batch_size(mongo_batch_size)

    pending = [
        {"_id": str(doc["_id"]), "ytVideoID": doc["ytVideoID"]}
        for doc in cursor
        if str(doc["_id"]) not in done_ids
    ]
    log.info(f"Pending: {len(pending)} new documents")

    if not pending:
        log.info("Index is up to date. Nothing to do.")
        client.close()
        return

    embedder = Embedder()
    t_start  = time.perf_counter()
    processed = failed = 0
    chunks = [pending[i:i+fetch_concurrency] for i in range(0, len(pending), fetch_concurrency)]

    async def run():
        nonlocal processed, failed
        connector = aiohttp.TCPConnector(limit=fetch_concurrency, limit_per_host=50)
        async with aiohttp.ClientSession(connector=connector) as session:
            for chunk_idx, chunk in enumerate(chunks):
                results = await fetch_batch_async(session, chunk, fetch_concurrency)
                valid   = [(did, img) for did, img in results if img is not None]
                failed += len(results) - len(valid)

                # Mark failed fetches as done so they don't retry on resume
                for did, img in results:
                    if img is None:
                        done_ids.add(did)

                if not valid:
                    continue

                doc_ids = [did for did, _ in valid]
                images  = [img for _, img in valid]
                sub_chunks = [
                    (doc_ids[i:i+embed_batch_size], images[i:i+embed_batch_size])
                    for i in range(0, len(images), embed_batch_size)
                ]

                for sub_ids, sub_imgs in sub_chunks:
                    try:
                        cls_embs, pixel_hashes = embedder.embed_batch(sub_imgs)
                        faiss_ids = np.array(
                            [mongo_id_to_faiss_id(did) for did in sub_ids],
                            dtype=np.int64,
                        )
                        index.add_with_ids(cls_embs, faiss_ids)
                        for did, fid, phash in zip(sub_ids, faiss_ids.tolist(), pixel_hashes):
                            sidecar.add(did, fid, phash)
                            done_ids.add(did)
                        processed += len(sub_ids)
                    except Exception as e:
                        log.error(f"Embedding sub-batch failed: {e}")
                        failed += len(sub_ids)
                        done_ids.update(sub_ids)

                # Checkpoint after every fetch-batch
                save_index(index)
                sidecar.save()
                save_progress(done_ids)

                elapsed   = time.perf_counter() - t_start
                rate      = processed / elapsed if elapsed > 0 else 0
                remaining = len(pending) - processed - failed
                eta_s     = remaining / rate if rate > 0 else 0
                log.info(
                    f"  [{chunk_idx+1}/{len(chunks)}] "
                    f"processed={processed} failed={failed} "
                    f"rate={rate:.1f}/s  ETA={eta_s/60:.1f}min"
                )

    asyncio.run(run())

    save_index(index)
    sidecar.save()
    save_progress(done_ids)

    elapsed = time.perf_counter() - t_start
    log.info(f"Done. {processed} embedded, {failed} failed, in {elapsed:.1f}s")
    log.info(f"Index size: {index.ntotal} vectors  |  {INDEX_PATH}")
    client.close()


# ─────────────────────────────────────────────────────────────────────────────
# Deletion
# ─────────────────────────────────────────────────────────────────────────────

def delete_entries(mongo_ids: list[str]) -> None:
    """Remove one or more MongoDB doc IDs from the FAISS index and sidecar."""
    if not INDEX_PATH.exists():
        log.error("No index found. Run build first.")
        return

    index   = load_or_create_index()
    sidecar = Sidecar.load()
    done    = load_progress()

    removed = skipped = 0
    faiss_ids_to_remove = []

    for mongo_id in mongo_ids:
        fid = sidecar.faiss_id_for(mongo_id)
        if fid is None:
            log.warning(f"  {mongo_id} not found in index — skipping")
            skipped += 1
            continue
        faiss_ids_to_remove.append(fid)
        sidecar.remove(mongo_id)
        done.discard(mongo_id)
        removed += 1
        log.info(f"  Removed {mongo_id} (faiss_id={fid})")

    if faiss_ids_to_remove:
        id_selector = faiss.IDSelectorArray(
            np.array(faiss_ids_to_remove, dtype=np.int64)
        )
        index.remove_ids(id_selector)
        save_index(index)
        sidecar.save()
        save_progress(done)

    log.info(f"Deletion complete: {removed} removed, {skipped} not found")
    log.info(f"Index size now: {index.ntotal} vectors")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SFH thumbnail FAISS index builder with deletion support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # build / update
    p_build = sub.add_parser("build", help="Build or incrementally update the index")
    p_build.add_argument("--uri",     required=True, help="MongoDB connection URI")
    p_build.add_argument("--batch",   type=int, default=16,  help="DINOv3 embed batch size")
    p_build.add_argument("--workers", type=int, default=100, help="Concurrent HTTP fetch workers")

    # delete
    p_del = sub.add_parser("delete", help="Remove entries from the index by MongoDB _id")
    p_del.add_argument("ids", nargs="+", help="MongoDB _id strings to remove")

    # info
    sub.add_parser("info", help="Print index stats")

    args = parser.parse_args()

    if args.cmd == "build":
        build_index(
            mongo_uri=args.uri,
            embed_batch_size=args.batch,
            fetch_concurrency=args.workers,
        )
    elif args.cmd == "delete":
        delete_entries(args.ids)
    elif args.cmd == "info":
        if not INDEX_PATH.exists():
            print("No index found.")
            return
        index   = load_or_create_index()
        sidecar = Sidecar.load()
        done    = load_progress()
        print(f"Index vectors : {index.ntotal}")
        print(f"Sidecar entries: {len(sidecar)}")
        print(f"Processed IDs  : {len(done)}")
        print(f"Index path     : {INDEX_PATH.resolve()}")
        print(f"Sidecar path   : {SIDECAR_PATH.resolve()}")


if __name__ == "__main__":
    main()