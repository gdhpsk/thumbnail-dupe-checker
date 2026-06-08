"""
Multi-Stage Image Similarity Detection Pipeline
================================================
Stages (fast-to-slow, short-circuit on high-confidence early):

  Stage 0 — Metadata fast-exit     : file size, dimensions, mode
  Stage 1 — Hash ensemble           : pHash + dHash + aHash + wHash (4-way vote)
  Stage 2 — DCT frequency spectrum  : low/mid/high band energy ratios (catches compression variants)
  Stage 3 — Sector-grid analysis    : 4x4 grid of perceptual hashes (spatial layout diff)
  Stage 4 — DINOv3 global semantics : pooler / CLS cosine similarity
  Stage 5 — DINOv3 patch-level      : 196-patch spatial feature map comparison (SSIM-style on embedding grid)
  Stage 6 — OCR text delta          : optional tesseract overlay-text extraction + Levenshtein diff

Decision logic uses a weighted score matrix with configurable thresholds.
Supports single-pair, batch/directory, and embedding-cache modes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import imagehash
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# Optional OCR — gracefully disabled if tesseract not installed
try:
    import pytesseract
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False

# Optional geometric alignment — gracefully disabled if opencv not installed
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"

    # Stage enable flags
    use_ocr: bool = True          # requires pytesseract + tesseract binary

    # Hash stage config
    hash_size: int = 16           # larger = more sensitive (standard is 8; 16 catches subtler diffs)

    # Sector grid config
    grid_rows: int = 8
    grid_cols: int = 8

    # DCT config
    dct_resize: tuple = (64, 64)  # image size before DCT analysis

    # Score weights (must sum to 1.0)
    weight_hash:   float = 0.25
    weight_grid:   float = 0.30
    weight_dct:    float = 0.05
    weight_global: float = 0.20
    weight_patch:  float = 0.20

    # Verdict thresholds (composite weighted score)
    thresh_exact:    float = 0.97   # effectively identical
    thresh_micro:    float = 0.88   # same base image, minor mutation
    thresh_variant:  float = 0.70   # recognizably related
    thresh_distant:  float = 0.50   # possibly related

    # OCR text-delta weight override (when OCR is active and text differs significantly)
    ocr_penalty_weight: float = 0.15  # deducted from composite when text δ is large
    grid_variance_penalty: float = 1.20  # multiplier on grid variance subtracted from composite
                                          # 0.40 means variance=0.01 → -0.004 penalty (mild)
                                          # variance=0.05 → -0.02 penalty (significant)

    # Embedding cache directory (None = no caching)
    cache_dir: Optional[str] = ".embedding_cache"

    # Short-circuit thresholds (skip later stages if early stages are decisive)
    shortcircuit_high: float = 0.99   # hash score above this → skip DINOv3 entirely
    shortcircuit_low:  float = 0.05   # hash score below this → skip DINOv3 entirely

    # Geometric alignment pre-pass (requires opencv-python)
    use_alignment:          bool  = True
    align_max_features:     int   = 5000   # ORB keypoints to detect
    align_match_keep_ratio: float = 0.75   # Lowe ratio test threshold
    align_min_inliers:      int   = 12     # RANSAC min inliers to trust homography
    align_max_scale:        float = 2.5    # reject homographies with extreme scale change
    align_max_rotation_deg: float = 30.0   # reject homographies with large rotation


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StageScores:
    hash_ensemble:      float = 0.0
    dct_spectrum:       float = 0.0
    dct_fallback:       bool  = False   # True = scipy missing, score cloned from hash
    sector_grid:        float = 0.0
    global_semantic:    float = 0.0
    patch_spatial:      float = 0.0
    ocr_text_delta:     Optional[float] = None  # None = OCR not run
    composite:          float = 0.0
    align_scale:        Optional[float] = None  # detected scale ratio (None = not run / failed)
    align_rotation_deg: Optional[float] = None  # detected rotation in degrees
    align_inliers:      Optional[int]   = None  # RANSAC inlier count
    align_applied:      bool = False             # True = images were warped before grid/patch

@dataclass
class SimilarityReport:
    path_a: str
    path_b: str
    scores: StageScores
    verdict: str
    verdict_code: str   # EXACT | MICRO_VARIANT | MODIFIED_VARIANT | DISTANT | DISTINCT
    grid_matrix:    Optional[list] = None   # 4x4 list-of-lists of per-cell scores
    grid_min_cell:  Optional[float] = None
    grid_variance:  Optional[float] = None
    ocr_text_a: Optional[str] = None
    ocr_text_b: Optional[str] = None
    elapsed_ms: float = 0.0
    stages_run: list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_image(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def _image_file_hash(path: str) -> str:
    """SHA-256 of raw file bytes — used as embedding cache key."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def _levenshtein_ratio(a: str, b: str) -> float:
    """Normalized Levenshtein similarity [0, 1]."""
    a, b = a.strip(), b.strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    # DP with row compression
    prev = list(range(lb + 1))
    for i, ca in enumerate(a):
        curr = [i + 1] + [0] * lb
        for j, cb in enumerate(b):
            curr[j + 1] = min(
                prev[j + 1] + 1,
                curr[j] + 1,
                prev[j] + (0 if ca == cb else 1)
            )
        prev = curr
    dist = prev[lb]
    return 1.0 - dist / max(la, lb)


def _dct2(block: np.ndarray) -> np.ndarray:
    """2D DCT-II via separable 1D DCTs (scipy-free)."""
    from scipy.fft import dctn
    return dctn(block.astype(np.float64), norm="ortho")


# ─────────────────────────────────────────────────────────────────────────────
# Stage implementations
# ─────────────────────────────────────────────────────────────────────────────

class HashStage:
    """Stage 1: 4-way perceptual hash ensemble → normalized similarity score."""

    def __init__(self, hash_size: int = 16):
        self.hash_size = hash_size
        self.max_bits = hash_size * hash_size  # pHash, aHash, wHash use NxN bits

    def _score_pair(self, h1, h2) -> float:
        dist = h1 - h2
        return max(0.0, 1.0 - dist / self.max_bits)

    def compute(self, img1: Image.Image, img2: Image.Image) -> dict:
        hs = self.hash_size
        p1, p2 = imagehash.phash(img1, hash_size=hs), imagehash.phash(img2, hash_size=hs)
        d1, d2 = imagehash.dhash(img1, hash_size=hs), imagehash.dhash(img2, hash_size=hs)
        a1, a2 = imagehash.average_hash(img1, hash_size=hs), imagehash.average_hash(img2, hash_size=hs)
        w1, w2 = imagehash.whash(img1, hash_size=hs), imagehash.whash(img2, hash_size=hs)

        scores = {
            "phash": self._score_pair(p1, p2),
            "dhash": self._score_pair(d1, d2),
            "ahash": self._score_pair(a1, a2),
            "whash": self._score_pair(w1, w2),
        }
        # Weighted ensemble: pHash and dHash carry more discriminative power
        ensemble = (
            scores["phash"] * 0.35 +
            scores["dhash"] * 0.35 +
            scores["ahash"] * 0.15 +
            scores["whash"] * 0.15
        )
        return {"individual": scores, "ensemble": ensemble}


class DCTStage:
    """Stage 2: DCT frequency band analysis.

    Splits each image's DCT into low/mid/high frequency bands and compares
    the energy distribution. Compression artifacts shift energy from high
    to low bands; this catches re-encoded duplicates that fool pixel-level hashes.
    """

    def __init__(self, resize: tuple = (64, 64)):
        self.resize = resize

    def _band_energies(self, img: Image.Image) -> np.ndarray:
        from scipy.fft import dctn
        gray = np.array(img.convert("L").resize(self.resize), dtype=np.float64)
        dct = dctn(gray, norm="ortho")
        n = self.resize[0]
        # Low: top-left quadrant, mid: middle ring, high: bottom-right quadrant
        low  = np.sum(dct[:n//4, :n//4] ** 2)
        high = np.sum(dct[n//2:, n//2:] ** 2)
        total = np.sum(dct ** 2) + 1e-9
        mid = total - low - high
        return np.array([low / total, mid / total, high / total])

    def compute(self, img1: Image.Image, img2: Image.Image) -> float:
        e1 = self._band_energies(img1)
        e2 = self._band_energies(img2)
        # Cosine similarity of energy distribution vectors
        cos = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9))
        return max(0.0, cos)


class GeometricAlignmentStage:
    """Pre-pass: detect and correct zoom/crop/minor-rotation differences using ORB + RANSAC.

    Strategy:
      1. Detect ORB keypoints in both images.
      2. Match with BFMatcher + Lowe ratio test.
      3. Estimate homography via RANSAC.
      4. Decompose homography to extract scale and rotation.
      5. Reject implausible transforms (large rotation, extreme scale).
      6. Warp img_b to align with img_a using the homography.

    Returns the aligned version of img_b (or the original if alignment fails/is rejected),
    plus metadata about the detected transform.

    Why ORB over SIFT: no patent issues, ships with every opencv build, fast enough
    for thumbnail-sized images. SIFT would give more accurate matches for large
    perspective distortions but thumbnails are almost always pure similarity transforms
    (scale + translate, maybe tiny rotation) so ORB is sufficient.
    """

    def __init__(
        self,
        max_features: int   = 5000,
        match_ratio:  float = 0.75,
        min_inliers:  int   = 12,
        max_scale:    float = 2.5,
        max_rot_deg:  float = 30.0,
    ):
        if not _CV2_AVAILABLE:
            raise RuntimeError("opencv-python not installed. Install with: pip install opencv-python")
        self.max_features = max_features
        self.match_ratio  = match_ratio
        self.min_inliers  = min_inliers
        self.max_scale    = max_scale
        self.max_rot_deg  = max_rot_deg

    @staticmethod
    def _pil_to_gray_np(img: Image.Image) -> np.ndarray:
        return np.array(img.convert("L"))

    @staticmethod
    def _pil_to_np(img: Image.Image) -> np.ndarray:
        return np.array(img)

    @staticmethod
    def _np_to_pil(arr: np.ndarray) -> Image.Image:
        return Image.fromarray(arr)

    @staticmethod
    def _decompose_homography(H: np.ndarray) -> dict:
        """Extract scale and rotation from a 3x3 homography matrix.

        For a similarity transform H ≈ s*R + t, the scale is the norm of the
        first column of the upper-left 2x2 block and rotation is atan2(H[1,0], H[0,0]).
        """
        sx = np.sqrt(H[0, 0] ** 2 + H[1, 0] ** 2)
        sy = np.sqrt(H[0, 1] ** 2 + H[1, 1] ** 2)
        scale = float((sx + sy) / 2.0)
        rotation_rad = float(np.arctan2(H[1, 0], H[0, 0]))
        rotation_deg = float(np.degrees(rotation_rad))
        return {"scale": scale, "rotation_deg": rotation_deg, "sx": sx, "sy": sy}

    def align(self, img_a: Image.Image, img_b: Image.Image) -> dict:
        """Attempt to align img_b onto img_a's frame.

        Returns:
          aligned_b  : Image.Image — warped img_b (or original on failure)
          applied    : bool        — True if warp was actually applied
          scale      : float | None
          rotation_deg: float | None
          inliers    : int | None
          reason     : str         — human-readable outcome description
        """
        gray_a = self._pil_to_gray_np(img_a)
        gray_b = self._pil_to_gray_np(img_b)

        orb = cv2.ORB_create(nfeatures=self.max_features)
        kp_a, des_a = orb.detectAndCompute(gray_a, None)
        kp_b, des_b = orb.detectAndCompute(gray_b, None)

        if des_a is None or des_b is None or len(kp_a) < 4 or len(kp_b) < 4:
            return {
                "aligned_b": img_b, "applied": False,
                "scale": None, "rotation_deg": None, "inliers": None,
                "reason": f"insufficient keypoints (a={len(kp_a)}, b={len(kp_b)})",
            }

        # BFMatcher with Hamming distance (correct for ORB binary descriptors)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw_matches = bf.knnMatch(des_b, des_a, k=2)

        # Lowe ratio test
        good = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.match_ratio * n.distance:
                    good.append(m)

        if len(good) < self.min_inliers:
            return {
                "aligned_b": img_b, "applied": False,
                "scale": None, "rotation_deg": None, "inliers": len(good),
                "reason": f"too few good matches after ratio test ({len(good)} < {self.min_inliers})",
            }

        pts_b = np.float32([kp_b[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_a = np.float32([kp_a[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, ransacReprojThreshold=4.0)

        if H is None:
            return {
                "aligned_b": img_b, "applied": False,
                "scale": None, "rotation_deg": None, "inliers": 0,
                "reason": "RANSAC failed to find homography",
            }

        inliers = int(mask.sum()) if mask is not None else 0
        if inliers < self.min_inliers:
            return {
                "aligned_b": img_b, "applied": False,
                "scale": None, "rotation_deg": None, "inliers": inliers,
                "reason": f"too few RANSAC inliers ({inliers} < {self.min_inliers})",
            }

        decomp = self._decompose_homography(H)
        scale = decomp["scale"]
        rot   = decomp["rotation_deg"]

        # Sanity checks: reject implausible transforms
        if scale > self.max_scale or scale < (1.0 / self.max_scale):
            return {
                "aligned_b": img_b, "applied": False,
                "scale": scale, "rotation_deg": rot, "inliers": inliers,
                "reason": f"implausible scale ({scale:.3f}x exceeds ±{self.max_scale}x limit)",
            }

        if abs(rot) > self.max_rot_deg:
            return {
                "aligned_b": img_b, "applied": False,
                "scale": scale, "rotation_deg": rot, "inliers": inliers,
                "reason": f"implausible rotation ({rot:.1f}° exceeds ±{self.max_rot_deg}° limit)",
            }

        # Always use the forward homography H (maps pts_b → pts_a).
        # H encodes "where does each pixel of B appear in A's coordinate frame",
        # which is exactly the warp we want: transform B so it overlays A.
        # Inverting H would warp A's content into B's frame, giving the wrong result.
        # The scale from decompose_homography is used only for reporting, not routing.
        h_a, w_a = gray_a.shape
        arr_b = self._pil_to_np(img_b)

        warped = cv2.warpPerspective(
            arr_b, H, (w_a, h_a),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,  # replicate edge pixels instead of black fill
        )
        aligned_b = self._np_to_pil(warped)

        return {
            "aligned_b": aligned_b, "applied": True,
            "scale": scale, "rotation_deg": rot, "inliers": inliers,
            "reason": f"aligned: scale={scale:.3f}x, rot={rot:.1f}°, inliers={inliers}",
        }


class SectorGridStage:
    """Stage 3: Divide each image into a grid of cells; compute pHash per cell.

    Returns a spatial similarity matrix and a summary score.
    This is the key stage for catching text-overlay changes in thumbnails:
    a different song title in the top-left cell shows up as a low score for
    that cell even if the rest of the image is identical.
    """

    def __init__(self, rows: int = 4, cols: int = 4, hash_size: int = 8):
        self.rows = rows
        self.cols = cols
        self.hash_size = hash_size

    def _grid_hashes(self, img: Image.Image) -> list[list]:
        w, h = img.size
        cell_w = w // self.cols
        cell_h = h // self.rows
        hashes = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                box = (c * cell_w, r * cell_h, (c + 1) * cell_w, (r + 1) * cell_h)
                cell = img.crop(box)
                row.append(imagehash.phash(cell, hash_size=self.hash_size))
            hashes.append(row)
        return hashes

    def compute(self, img1: Image.Image, img2: Image.Image) -> dict:
        g1 = self._grid_hashes(img1)
        g2 = self._grid_hashes(img2)
        max_bits = self.hash_size * self.hash_size
        matrix = np.zeros((self.rows, self.cols))
        for r in range(self.rows):
            for c in range(self.cols):
                dist = g1[r][c] - g2[r][c]
                matrix[r][c] = max(0.0, 1.0 - dist / max_bits)

        summary = float(np.mean(matrix))
        # Also report the minimum cell score (useful for detecting localized mutations)
        min_score = float(np.min(matrix))
        # Cell variance: high variance means localized difference (e.g., text overlay)
        cell_variance = float(np.var(matrix))
        return {
            "matrix": matrix.tolist(),
            "summary": summary,
            "min_cell": min_score,
            "cell_variance": cell_variance,
        }


class DINOv3Stage:
    """Stages 4 & 5: DINOv3 global (CLS) and patch-level spatial similarity.

    Global: pooler_output cosine similarity — semantic scene-level match.
    Patch:  compare the 196 spatial patch embeddings as a 14x14 feature map.
            Uses top-K patch selection to focus on discriminative regions
            and computes a spatial correlation score.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
    ):
        log.info(f"Loading DINOv3 model: {model_name}")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map="auto")
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"DINOv3 mapped to: {self.device}")

    def _cache_path(self, file_hash: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{file_hash}.pt"

    @staticmethod
    def _pil_hash(img: Image.Image) -> str:
        """SHA-256 of raw pixel bytes — used as cache key for in-memory images."""
        h = hashlib.sha256(img.tobytes())
        return h.hexdigest()

    def _get_embeddings(self, img: Image.Image, file_path: str) -> dict:
        """Returns dict with 'cls' (1, D) and 'patches' (196, D) tensors.

        file_path is used as the cache key source. If it contains ':' (synthetic
        key like 'image.jpg:aligned') or does not exist on disk, the cache key
        is derived from the image pixel data instead.
        """
        if ':' in file_path or not os.path.exists(file_path):
            fhash = self._pil_hash(img)
        else:
            fhash = _image_file_hash(file_path)
        cp = self._cache_path(fhash)
        if cp and cp.exists():
            cached = torch.load(cp, map_location=self.device, weights_only=True)
            return cached

        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # CLS / global token
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            cls_emb = F.normalize(outputs.pooler_output, p=2, dim=-1)  # (1, D)
        else:
            cls_emb = F.normalize(outputs.last_hidden_state[:, 0, :], p=2, dim=-1)

        # Patch tokens — DINOv3 layout: [CLS, reg0, reg1, reg2, reg3, patch0, ..., patch195]
        # Skip the 1 CLS + 4 register tokens → indices 5:201
        last_hidden = outputs.last_hidden_state  # (1, 201, D)
        patch_embs = last_hidden[:, 5:, :]       # (1, 196, D)
        patch_embs = F.normalize(patch_embs.squeeze(0), p=2, dim=-1)  # (196, D)

        result = {"cls": cls_emb.cpu(), "patches": patch_embs.cpu()}
        if cp:
            torch.save(result, cp)
        return result

    def compute_global(self, emb1: dict, emb2: dict) -> float:
        cos = F.cosine_similarity(
            emb1["cls"].to(self.device),
            emb2["cls"].to(self.device)
        ).item()
        return float(max(0.0, cos))

    def compute_patch_spatial(self, emb1: dict, emb2: dict) -> dict:
        """
        Full dense patch-level spatial similarity across all 196 patches.

        Method:
          1. Compute the full 196×196 cosine similarity matrix between every
             patch in img1 and every patch in img2 (38,416 dot products, trivial on GPU).
          2. Apply a per-patch Gaussian spatial penalty that softly constrains
             each match to its expected position — allows for the small shifts
             introduced by zoom/crop variants without permitting wholesale reordering.
          3. Two scores returned:
               spatially_weighted : best spatially-penalised match per patch, averaged
               unweighted         : best unrestricted match per patch, averaged
                                    (useful for detecting content-preserving flips/rotations)
        """
        p1 = emb1["patches"].to(self.device)  # (196, D)
        p2 = emb2["patches"].to(self.device)  # (196, D)

        # Full 196×196 cosine similarity (both already L2-normalised)
        sim_matrix = p1 @ p2.T  # (196, 196)

        # Spatial position grid — patches are arranged in a 14×14 raster
        grid_size = 14
        positions = torch.stack(torch.meshgrid(
            torch.arange(grid_size, device=self.device),
            torch.arange(grid_size, device=self.device),
            indexing="ij"
        ), dim=-1).reshape(196, 2).float()  # (196, 2)

        # Pairwise L2 distance between all patch position pairs
        pos_dist = torch.cdist(positions, positions)  # (196, 196)

        # Gaussian spatial penalty: σ = grid_size / 3 ≈ 4.67 patch units
        # Allows ~1-2 patch shifts (sub-2% zoom) with minimal penalty;
        # penalises cross-image matches more than ~7 patches away.
        sigma = grid_size / 3.0
        pos_penalty = torch.exp(-pos_dist ** 2 / (2 * sigma ** 2))  # (196, 196)

        # Spatially-weighted best match per patch in img1
        weighted_sim = sim_matrix * pos_penalty          # (196, 196)
        best_match_score = weighted_sim.max(dim=1).values  # (196,)
        patch_score = float(best_match_score.mean().clamp(0, 1))

        # Unweighted best match (no spatial constraint)
        full_score = float(sim_matrix.max(dim=1).values.mean().clamp(0, 1))

        return {
            "spatially_weighted": patch_score,
            "unweighted": full_score,
        }


class OCRStage:
    """Stage 6: Extract visible text via tesseract and compute text similarity."""

    def __init__(self):
        if not _OCR_AVAILABLE:
            raise RuntimeError("pytesseract not installed. Install with: pip install pytesseract")

    def extract(self, img: Image.Image) -> str:
        # Upscale small images for better OCR accuracy
        w, h = img.size
        if w < 300 or h < 300:
            scale = max(300 / w, 300 / h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        try:
            text = pytesseract.image_to_string(img, config="--psm 11")
            return text.strip()
        except Exception as e:
            log.warning(f"OCR failed: {e}")
            return ""

    def similarity(self, text_a: str, text_b: str) -> float:
        return _levenshtein_ratio(text_a, text_b)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class ImageSimilarityPipeline:

    VERDICTS = {
        "EXACT":           "EXACT DUPLICATE — byte/pixel-level identical or losslessly re-encoded",
        "MICRO_VARIANT":   "MICRO VARIANT — same base image; minor mutation (crop, watermark, text overlay, compression)",
        "MODIFIED_VARIANT": "MODIFIED VARIANT — same scene/subject; significant layout or content change",
        "DISTANT":         "DISTANT RELATIVE — loosely related (same template/style, different content)",
        "DISTINCT":        "DISTINCT IMAGES — no meaningful similarity detected",
    }

    def __init__(self, config: PipelineConfig = PipelineConfig()):
        self.cfg = config
        self.hash_stage  = HashStage(hash_size=config.hash_size)
        self.dct_stage   = DCTStage(resize=config.dct_resize)
        self.grid_stage  = SectorGridStage(rows=config.grid_rows, cols=config.grid_cols)
        self.dino_stage  = DINOv3Stage(
            model_name=config.model_name,
            cache_dir=config.cache_dir,
        )
        self.ocr_stage = OCRStage() if (config.use_ocr and _OCR_AVAILABLE) else None
        if config.use_alignment and _CV2_AVAILABLE:
            self.align_stage = GeometricAlignmentStage(
                max_features = config.align_max_features,
                match_ratio  = config.align_match_keep_ratio,
                min_inliers  = config.align_min_inliers,
                max_scale    = config.align_max_scale,
                max_rot_deg  = config.align_max_rotation_deg,
            )
        else:
            self.align_stage = None
            if config.use_alignment and not _CV2_AVAILABLE:
                log.warning("use_alignment=True but opencv-python not installed. Install with: pip install opencv-python")

    def _compute_composite(self, scores: StageScores, cfg: PipelineConfig) -> float:
        total_weight = (
            cfg.weight_hash +
            cfg.weight_grid +
            cfg.weight_dct +
            cfg.weight_global +
            cfg.weight_patch
        )
        composite = (
            scores.hash_ensemble   * cfg.weight_hash   +
            scores.sector_grid     * cfg.weight_grid    +
            scores.dct_spectrum    * cfg.weight_dct     +
            scores.global_semantic * cfg.weight_global  +
            scores.patch_spatial   * cfg.weight_patch
        ) / total_weight

        # Grid variance penalty: high variance = localized differences (different art,
        # same style). Penalises pairs where some cells differ significantly even if
        # the mean grid score is decent.
        if scores.grid_variance is not None:
            composite = max(0.0, composite - scores.grid_variance * cfg.grid_variance_penalty)

        # OCR penalty: if text similarity is low, the images likely differ in
        # displayed text even if visually similar (same thumbnail, different song name)
        if scores.ocr_text_delta is not None:
            text_diff = 1.0 - scores.ocr_text_delta
            if text_diff > 0.3:  # significant text difference
                penalty = text_diff * cfg.ocr_penalty_weight
                composite = max(0.0, composite - penalty)

        return composite

    def _verdict(self, composite: float, scores: StageScores) -> tuple[str, str]:
        cfg = self.cfg
        if composite >= cfg.thresh_exact:
            return self.VERDICTS["EXACT"], "EXACT"
        elif composite >= cfg.thresh_micro:
            # Distinguish micro-variant from modified: check cell variance
            # High cell variance → localized change (text, watermark)
            return self.VERDICTS["MICRO_VARIANT"], "MICRO_VARIANT"
        elif composite >= cfg.thresh_variant:
            return self.VERDICTS["MODIFIED_VARIANT"], "MODIFIED_VARIANT"
        elif composite >= cfg.thresh_distant:
            return self.VERDICTS["DISTANT"], "DISTANT"
        else:
            return self.VERDICTS["DISTINCT"], "DISTINCT"

    def analyze_pair(
        self,
        path_a: str,
        path_b: str,
        force_full_pipeline: bool = False,
    ) -> SimilarityReport:
        t0 = time.perf_counter()
        stages_run = []

        if not os.path.exists(path_a):
            raise FileNotFoundError(f"Not found: {path_a}")
        if not os.path.exists(path_b):
            raise FileNotFoundError(f"Not found: {path_b}")

        img_a = _load_image(path_a)
        img_b = _load_image(path_b)
        scores = StageScores()

        # ── Stage 0: Metadata fast-exit ─────────────────────────────────────
        # Exact byte match (same file reposted)
        if _image_file_hash(path_a) == _image_file_hash(path_b):
            scores.hash_ensemble = scores.dct_spectrum = scores.sector_grid = 1.0
            scores.global_semantic = scores.patch_spatial = 1.0
            scores.composite = 1.0
            elapsed = (time.perf_counter() - t0) * 1000
            return SimilarityReport(
                path_a=path_a, path_b=path_b, scores=scores,
                verdict=self.VERDICTS["EXACT"], verdict_code="EXACT",
                elapsed_ms=elapsed, stages_run=["metadata_exact_match"],
            )
        stages_run.append("metadata")

        # ── Stage 1: Hash ensemble ───────────────────────────────────────────
        hash_result = self.hash_stage.compute(img_a, img_b)
        scores.hash_ensemble = hash_result["ensemble"]
        stages_run.append("hash_ensemble")
        log.debug(f"Hash ensemble: {scores.hash_ensemble:.4f} | {hash_result['individual']}")

        # Short-circuit: trivially identical or trivially different
        if not force_full_pipeline:
            if scores.hash_ensemble >= self.cfg.shortcircuit_high:
                scores.composite = scores.hash_ensemble
                verdict, code = self._verdict(scores.composite, scores)
                elapsed = (time.perf_counter() - t0) * 1000
                return SimilarityReport(
                    path_a=path_a, path_b=path_b, scores=scores,
                    verdict=verdict, verdict_code=code,
                    elapsed_ms=elapsed, stages_run=stages_run,
                )

        # ── Stage 1b: Geometric alignment pre-pass ──────────────────────────
        # Warp img_b onto img_a's coordinate frame before spatial stages.
        # This normalises zoom/crop differences so that sector-grid and patch
        # comparisons operate on content-aligned images rather than shifted ones.
        if self.align_stage is not None:
            align_result = self.align_stage.align(img_a, img_b)
            scores.align_scale        = align_result["scale"]
            scores.align_rotation_deg = align_result["rotation_deg"]
            scores.align_inliers      = align_result["inliers"]
            scores.align_applied      = align_result["applied"]
            if align_result["applied"]:
                img_b = align_result["aligned_b"]   # replace for all downstream stages
                stages_run.append(f"align[{align_result['reason']}]")
                log.info(f"Alignment applied: {align_result['reason']}")
            else:
                stages_run.append(f"align[skipped:{align_result['reason']}]")
                log.debug(f"Alignment skipped: {align_result['reason']}")

        # ── Stage 2: DCT frequency spectrum ─────────────────────────────────
        try:
            scores.dct_spectrum = self.dct_stage.compute(img_a, img_b)
            stages_run.append("dct_spectrum")
        except ImportError:
            log.warning(
                "scipy not installed — DCT stage disabled. "
                "Install with: pip install scipy\n"
                "  DCT score is FALLBACK (cloned from hash ensemble — not independent)"
            )
            scores.dct_spectrum = scores.hash_ensemble
            scores.dct_fallback = True
            stages_run.append("dct_spectrum[FALLBACK:no-scipy]")

        # ── Stage 3: Sector grid ─────────────────────────────────────────────
        grid_result = self.grid_stage.compute(img_a, img_b)
        scores.sector_grid = grid_result["summary"]
        stages_run.append("sector_grid")
        log.debug(f"Grid summary: {scores.sector_grid:.4f} | min_cell: {grid_result['min_cell']:.4f} | var: {grid_result['cell_variance']:.4f}")

        # ── Stages 4 & 5: DINOv3 ────────────────────────────────────────────
        if force_full_pipeline or scores.hash_ensemble > self.cfg.shortcircuit_low:
            emb_a = self.dino_stage._get_embeddings(img_a, path_a)
            # If alignment was applied, img_b is now a warped PIL image.
            # Use a derived cache key so the aligned embedding is cached
            # separately from the original (path_b + ":aligned").
            dino_path_b = (path_b + ":aligned") if scores.align_applied else path_b
            emb_b = self.dino_stage._get_embeddings(img_b, dino_path_b)

            scores.global_semantic = self.dino_stage.compute_global(emb_a, emb_b)
            stages_run.append("dinov3_global")

            patch_result = self.dino_stage.compute_patch_spatial(emb_a, emb_b)
            scores.patch_spatial = patch_result["spatially_weighted"]
            stages_run.append("dinov3_patch")

            log.debug(f"DINOv3 global: {scores.global_semantic:.4f} | patch: {scores.patch_spatial:.4f} (unweighted: {patch_result['unweighted']:.4f})")
        else:
            log.info("Short-circuit (low hash score): skipping DINOv3 stages")
            scores.global_semantic = 0.0
            scores.patch_spatial = 0.0

        # ── Stage 6: OCR (optional) ──────────────────────────────────────────
        text_a = text_b = None
        if self.ocr_stage is not None:
            text_a = self.ocr_stage.extract(img_a)
            text_b = self.ocr_stage.extract(img_b)
            scores.ocr_text_delta = self.ocr_stage.similarity(text_a, text_b)
            stages_run.append("ocr_text")
            log.debug(f"OCR similarity: {scores.ocr_text_delta:.4f}")

        # ── Composite score & verdict ────────────────────────────────────────
        scores.composite = self._compute_composite(scores, self.cfg)
        verdict, code = self._verdict(scores.composite, scores)

        elapsed = (time.perf_counter() - t0) * 1000
        return SimilarityReport(
            path_a=path_a, path_b=path_b, scores=scores,
            verdict=verdict, verdict_code=code,
            grid_matrix=grid_result["matrix"],
            grid_min_cell=grid_result["min_cell"],
            grid_variance=grid_result["cell_variance"],
            ocr_text_a=text_a, ocr_text_b=text_b,
            elapsed_ms=elapsed, stages_run=stages_run,
        )

    def analyze_directory(
        self,
        directory: str,
        extensions: tuple = (".jpg", ".jpeg", ".png", ".webp"),
        output_json: Optional[str] = None,
    ) -> list[SimilarityReport]:
        """
        Compare all image pairs in a directory.
        Returns reports for pairs with composite score above thresh_distant.
        """
        paths = [
            str(p) for p in Path(directory).rglob("*")
            if p.suffix.lower() in extensions
        ]
        paths.sort()
        log.info(f"Found {len(paths)} images in {directory}")

        reports = []
        total_pairs = len(paths) * (len(paths) - 1) // 2
        n = 0

        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                n += 1
                if n % 50 == 0:
                    log.info(f"  {n}/{total_pairs} pairs processed…")
                try:
                    report = self.analyze_pair(paths[i], paths[j])
                    if report.scores.composite >= self.cfg.thresh_distant:
                        reports.append(report)
                except Exception as e:
                    log.error(f"Pair ({paths[i]}, {paths[j]}) failed: {e}")

        reports.sort(key=lambda r: r.scores.composite, reverse=True)

        if output_json:
            serializable = []
            for r in reports:
                d = asdict(r)
                # Convert matrix to list-of-lists for JSON
                if "sector_grid_matrix" in d:
                    d["sector_grid_matrix"] = d["sector_grid_matrix"]
                serializable.append(d)
            with open(output_json, "w") as f:
                json.dump(serializable, f, indent=2)
            log.info(f"Results written to {output_json}")

        return reports


# ─────────────────────────────────────────────────────────────────────────────
# Pretty report printer
# ─────────────────────────────────────────────────────────────────────────────

def _heat_char(v: float) -> str:
    """Map a [0,1] cell score to a Unicode block with ANSI 256-color background.

    Color scale (cold → hot = similar → different):
      ≥0.95 : deep blue   (identical)
      ≥0.85 : cyan-blue
      ≥0.70 : green
      ≥0.55 : yellow-green
      ≥0.40 : orange
      < 0.40 : red        (very different)
    """
    # ANSI 256-color escape: \x1b[48;5;<n>m  (background)
    # Pick a colour ramp from blue (high similarity) → red (low similarity)
    if   v >= 0.95: color = 21   # deep blue
    elif v >= 0.85: color = 39   # sky blue
    elif v >= 0.70: color = 48   # teal-green
    elif v >= 0.55: color = 226  # yellow
    elif v >= 0.40: color = 208  # orange
    else:           color = 196  # red
    RESET = "\x1b[0m"
    return f"\x1b[48;5;{color}m  {RESET}"


def _render_grid_heatmap(matrix: list, rows: int = 4, cols: int = 4) -> str:
    """Render the sector-grid similarity matrix as a terminal heatmap.

    Each cell is 2 chars wide for readability. Score printed below each row.
    Falls back to ASCII if the terminal doesn't support ANSI (CI, file redirect).
    """
    import sys
    use_color = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    lines = []
    cell_w = 6  # width of each cell in the ASCII fallback

    for r in range(rows):
        # Top half: colored block
        row_str = "  "
        for c in range(cols):
            v = matrix[r][c]
            if use_color:
                row_str += _heat_char(v)
            else:
                # ASCII fallback: shade characters
                ch = "██" if v >= 0.90 else "▓▓" if v >= 0.75 else "▒▒" if v >= 0.55 else "░░" if v >= 0.35 else ".."
                row_str += ch
        # Append per-cell scores on the same line (after color block row)
        score_str = "  "
        for c in range(cols):
            score_str += f"{matrix[r][c]:.2f} "
        lines.append(row_str + "    " + score_str.rstrip())

    # Column headers (col index)
    header = "  " + "".join(f"C{c} " if not use_color else "   " for c in range(cols))
    return "\n".join(lines)


def print_report(report: SimilarityReport, show_ocr: bool = True, show_grid: bool = True) -> None:
    W = 65
    print("\n" + "═" * W)
    print("  IMAGE SIMILARITY PIPELINE — REPORT")
    print("═" * W)
    print(f"  Image A : {os.path.basename(report.path_a)}")
    print(f"  Image B : {os.path.basename(report.path_b)}")
    print("─" * W)

    s = report.scores
    def bar(v: float, width: int = 30) -> str:
        filled = int(round(v * width))
        return "█" * filled + "░" * (width - filled)

    dct_tag = " [FALLBACK — install scipy]" if s.dct_fallback else ""
    print(f"  Hash Ensemble    {s.hash_ensemble:.4f}  {bar(s.hash_ensemble)}")
    print(f"  DCT Spectrum     {s.dct_spectrum:.4f}  {bar(s.dct_spectrum)}{dct_tag}")
    print(f"  Sector Grid      {s.sector_grid:.4f}  {bar(s.sector_grid)}", end="")
    if report.grid_min_cell is not None:
        print(f"  (min cell: {report.grid_min_cell:.3f}, var: {report.grid_variance:.4f})", end="")
    print()
    print(f"  DINOv3 Global    {s.global_semantic:.4f}  {bar(s.global_semantic)}")
    print(f"  DINOv3 Patch     {s.patch_spatial:.4f}  {bar(s.patch_spatial)}")
    if s.ocr_text_delta is not None:
        print(f"  OCR Text Δ       {s.ocr_text_delta:.4f}  {bar(s.ocr_text_delta)}")
    print("─" * W)
    print(f"  COMPOSITE        {s.composite:.4f}  {bar(s.composite)}")
    print("─" * W)
    print(f"  VERDICT  »  {report.verdict}")
    print(f"  CODE     »  {report.verdict_code}")
    print(f"  STAGES   »  {', '.join(report.stages_run)}")
    print(f"  TIME     »  {report.elapsed_ms:.1f} ms")

    # ── Geometric alignment summary ─────────────────────────────────────────
    if report.scores.align_scale is not None:
        applied_tag = "✓ applied" if report.scores.align_applied else "✗ not applied"
        print("─" * W)
        print(f"  ALIGNMENT [{applied_tag}]")
        sc = report.scores.align_scale
        print(f"    Scale    : {sc:.4f}x  ", end="")
        if sc is not None:
            delta_pct = abs(sc - 1.0) * 100
            if sc > 1.005:
                print(f"(img_b zoomed IN  +{delta_pct:.1f}%)", end="")
            elif sc < 0.995:
                print(f"(img_b zoomed OUT -{delta_pct:.1f}%)", end="")
            else:
                print("(sub-0.5% scale delta, effectively identical)", end="")
        print()
        if report.scores.align_rotation_deg is not None:
            print(f"    Rotation : {report.scores.align_rotation_deg:+.2f}°")
        if report.scores.align_inliers is not None:
            print(f"    Inliers  : {report.scores.align_inliers} RANSAC consensus points")

    # ── Sector grid heatmap ──────────────────────────────────────────────────
    if show_grid and report.grid_matrix is not None:
        print("─" * W)
        print("  SECTOR GRID  (blue=similar → red=different)")
        print()
        heatmap = _render_grid_heatmap(
            report.grid_matrix,
            rows=len(report.grid_matrix),
            cols=len(report.grid_matrix[0]),
        )
        for line in heatmap.splitlines():
            print("  " + line)
        # Flag the coldest cell
        matrix = report.grid_matrix
        min_val = report.grid_min_cell
        min_pos = None
        for r, row in enumerate(matrix):
            for c, v in enumerate(row):
                if abs(v - min_val) < 1e-6:
                    min_pos = (r, c)
                    break
            if min_pos:
                break
        if min_pos and min_val < 0.80:
            print(f"\n  ⚠  Lowest similarity at cell ({min_pos[0]},{min_pos[1]}) = {min_val:.3f}")
            print(f"     → Localized difference detected (text overlay, watermark, or crop)")
        print()

    if show_ocr and report.ocr_text_a is not None:
        print("─" * W)
        def truncate(t, n=100): return t[:n] + "…" if len(t) > n else t
        print(f"  OCR A : {truncate(report.ocr_text_a or '(empty)')}")
        print(f"  OCR B : {truncate(report.ocr_text_b or '(empty)')}")

    print("═" * W + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-stage image similarity detection pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Pair mode
    pair_p = subparsers.add_parser("pair", help="Compare two images")
    pair_p.add_argument("image_a")
    pair_p.add_argument("image_b")
    pair_p.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    pair_p.add_argument("--no-ocr", action="store_true")
    pair_p.add_argument("--no-align", action="store_true", help="Disable geometric alignment pre-pass")
    pair_p.add_argument("--no-cache", action="store_true")
    pair_p.add_argument("--full", action="store_true", help="Disable short-circuit logic")
    pair_p.add_argument("--json", dest="json_out", help="Write JSON output to this path")

    # Directory mode
    dir_p = subparsers.add_parser("dir", help="Find similar pairs in a directory")
    dir_p.add_argument("directory")
    dir_p.add_argument("--model", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    dir_p.add_argument("--no-ocr", action="store_true")
    dir_p.add_argument("--no-cache", action="store_true")
    dir_p.add_argument("--json", dest="json_out", default="similarity_results.json")

    args = parser.parse_args()

    no_align = getattr(args, 'no_align', False)
    cfg = PipelineConfig(
        model_name=args.model,
        use_ocr=not args.no_ocr,
        use_alignment=not no_align,
        cache_dir=None if args.no_cache else ".embedding_cache",
    )
    pipeline = ImageSimilarityPipeline(cfg)

    if args.mode == "pair":
        report = pipeline.analyze_pair(args.image_a, args.image_b, force_full_pipeline=args.full)
        print_report(report)
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(asdict(report), f, indent=2)
    elif args.mode == "dir":
        reports = pipeline.analyze_directory(args.directory, output_json=args.json_out)
        print(f"\nFound {len(reports)} similar pairs (above 'distant' threshold):\n")
        for r in reports[:20]:  # print top 20
            print_report(r, show_ocr=False)


if __name__ == "__main__":
    main()