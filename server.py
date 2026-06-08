"""
SFH Thumbnail Duplicate Detection API
=======================================
FastAPI server wrapping the query pipeline. Loads FAISS index, sidecar,
and DINOv3 model once at startup — no cold start per request.

Endpoints:
  GET  /health           — liveness check
  POST /check-duplicate  — check an image URL against a level's indexed songs

Request body:
  {
    "image_url": "https://...",
    "level_ids": ["59549363"]
  }

Response (match found, composite >= threshold):
  {
    "duplicate": true,
    "mongo_id":  "64f54c6ceba5efcdadf78741",
    "song_name": "Christian Mossuto - Violence...",
    "yt_video_id": "7WDcTPIKcms",
    "composite": 0.9242,
    "verdict":   "MICRO_VARIANT",
    "verdict_description": "MICRO VARIANT — same base image; minor mutation (crop, watermark, text overlay, compression)",
    "scores": {
      "hash_ensemble":   0.9455,
      "dct_spectrum":    1.0000,
      "sector_grid":     0.9570,
      "global_semantic": 0.9244,
      "patch_spatial":   0.8894
    },
    "grid": {
      "matrix":   [[0.97, ...], ...],
      "min_cell": 0.8823,
      "variance": 0.0031
    },
    "alignment": {
      "applied":      false,
      "scale":        null,
      "rotation_deg": null,
      "inliers":      null
    },
    "stages_run": ["hash_ensemble", "dct_spectrum", "sector_grid", "align[...]", "dinov3_global", "dinov3_patch"],
    "elapsed_ms": 312.4
  }

Response (no match):
  {}

Error responses:
  { "error": "ERROR_CODE", "message": "Human-readable message" }

Usage:
  pip install fastapi uvicorn httpx
  python server.py --uri "mongodb://localhost:27017" --host 0.0.0.0 --port 8000

  # With custom threshold:
  python server.py --uri "mongodb://localhost:27017" --threshold 0.80
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()
import argparse
import asyncio
import hashlib
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Optional

import faiss
import httpx
import json
import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image
from pymongo import MongoClient

from image_similarity import (
    PipelineConfig,
    _CV2_AVAILABLE,
)
from query import (
    compute_query_features,
    compare_features,
    restricted_search,
    CANONICAL_SIZE,
    INDEX_PATH,
)
from build import (
    Sidecar,
    Embedder,
    mongo_id_to_faiss_id,
    precompute_image_features,
    save_index,
    SIDECAR_PATH,
    PROGRESS_PATH,
    load_progress,
    save_progress,
    EMBEDDING_DIM,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Error codes
# ─────────────────────────────────────────────────────────────────────────────

class APIError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        self.code    = code
        self.message = message
        self.status  = status
        super().__init__(message)

ERRORS = {
    "MISSING_FIELDS":       (400, "Required fields missing: image_url, level_ids"),
    "INVALID_IMAGE_URL":    (400, "image_url must be a valid http/https URL"),
    "INVALID_LEVEL_ID":     (400, "level_ids must be a non-empty string or array"),
    "IMAGE_FETCH_FAILED":   (502, "Failed to fetch image from the provided URL"),
    "IMAGE_DECODE_FAILED":  (422, "URL did not return a valid image"),
    "IMAGE_TOO_SMALL":      (422, "Image is too small to process (minimum 64x64)"),
    "NO_SONGS_FOR_LEVEL":   (404, "No songs found for the given level_ids"),
    "NO_INDEXED_SONGS":     (404, "No indexed songs found for the given level_ids — rebuild index"),
    "INDEX_NOT_FOUND":      (503, "FAISS index not loaded — server not ready"),
    "MONGO_ERROR":          (503, "MongoDB query failed"),
    "PIPELINE_ERROR":       (500, "Internal pipeline error"),
    "DOC_ALREADY_INDEXED":  (409, "Document is already in the index — use /edit to update"),
    "DOC_NOT_FOUND":        (404, "Document ID not found in the index"),
    "EMBEDDER_NOT_READY":   (503, "DINOv3 embedder not loaded"),
    "INDEX_WRITE_FAILED":   (500, "Failed to persist index changes to disk"),
}


def api_error(code: str, override_message: Optional[str] = None) -> JSONResponse:
    status, default_msg = ERRORS.get(code, (500, "Unknown error"))
    return JSONResponse(
        status_code=status,
        content={"error": code, "message": override_message or default_msg},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Application state — loaded once at startup
# ─────────────────────────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.index:      Optional[faiss.IndexIDMap] = None
        self.sidecar:    Optional[Sidecar]          = None
        self.embedder:   Optional[Embedder]         = None
        self.mongo_uri:  str                        = ""
        self.threshold:  float                      = 0.80
        self.candidates: int                        = 20
        self.ready:      bool                       = False
        # Protects concurrent writes to index + sidecar + disk
        self.write_lock: threading.Lock             = threading.Lock()


state = AppState()


# ─────────────────────────────────────────────────────────────────────────────
# Startup / shutdown
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading FAISS index and sidecar...")
    if not INDEX_PATH.exists():
        log.error(f"FAISS index not found at {INDEX_PATH}. Run build_index.py first.")
        sys.exit(1)

    state.index   = faiss.read_index(str(INDEX_PATH))
    state.sidecar = Sidecar.load()
    log.info(f"  {state.index.ntotal} vectors | {len(state.sidecar.by_mongo)} sidecar entries")

    # Load DINOv3 embedder — used for /add and /edit routes.
    # Also warms up compute_query_features for /check-duplicate.
    log.info("Loading DINOv3 embedder...")
    try:
        state.embedder = Embedder()
        # Warm up query pipeline with a dummy image — reuse the embedder's
        # already-loaded weights rather than loading a second copy.
        dummy = Image.new("RGB", (64, 64), color=(128, 128, 128))
        compute_query_features(
            dummy,
            processor=state.embedder.processor,
            model=state.embedder.model,
            device=state.embedder.device,
        )
        log.info("  DINOv3 warm-up complete")
    except Exception as e:
        log.warning(f"  DINOv3 load failed: {e}")

    state.ready = True
    log.info("Server ready.")
    yield

    log.info("Shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SFH Duplicate Detection API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "PIPELINE_ERROR", "message": str(exc)},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Image fetcher
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_image(url: str) -> Image.Image:
    """Fetch an image from a URL and return as PIL Image."""
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        try:
            resp = await client.get(url, headers={"User-Agent": "SFH-DupeBot/1.0"})
            resp.raise_for_status()
        except httpx.TimeoutException:
            raise APIError("IMAGE_FETCH_FAILED", f"Request timed out fetching {url}", 502)
        except httpx.HTTPStatusError as e:
            raise APIError("IMAGE_FETCH_FAILED",
                           f"HTTP {e.response.status_code} fetching image URL", 502)
        except httpx.RequestError as e:
            raise APIError("IMAGE_FETCH_FAILED", f"Network error: {e}", 502)

    try:
        img = Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception:
        raise APIError("IMAGE_DECODE_FAILED",
                       "Response body is not a valid image", 422)

    if img.size[0] < 64 or img.size[1] < 64:
        raise APIError("IMAGE_TOO_SMALL",
                       f"Image size {img.size} is too small (minimum 64x64)", 422)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# Core check logic (sync, runs in thread pool via asyncio.to_thread)
# ─────────────────────────────────────────────────────────────────────────────

def _check_duplicate_sync(
    img:        Image.Image,
    level_ids:  list[str],
    exclude_id: Optional[str] = None,
) -> dict:
    """
    Runs the full query pipeline synchronously.
    Called via asyncio.to_thread to avoid blocking the event loop.
    """
    t0 = time.perf_counter()

    if not state.ready or state.index is None:
        raise APIError("INDEX_NOT_FOUND", "Server index not loaded", 503)

    # ── MongoDB ───────────────────────────────────────────────────────────────
    # $in on an array field matches documents where levelID contains any of
    # the provided values — correct for both single and multi-level queries.
    try:
        client     = MongoClient(state.mongo_uri, serverSelectionTimeoutMS=5000)
        collection = client["SFH"]["songs"]
        docs       = list(collection.find(
            {"levelID": {"$in": level_ids}},
            {"_id": 1, "ytVideoID": 1, "songName": 1, "name": 1},
        ))
        client.close()
    except Exception as e:
        raise APIError("MONGO_ERROR", f"MongoDB query failed: {e}", 503)

    if not docs:
        raise APIError("NO_SONGS_FOR_LEVEL",
                       f"No songs found with levelID in {level_ids}", 404)

    # ── Filter to indexed docs ────────────────────────────────────────────────
    indexed = [d for d in docs if str(d["_id"]) in state.sidecar and str(d["_id"]) != exclude_id]
    if not indexed:
        raise APIError("NO_INDEXED_SONGS",
                       f"None of the {len(docs)} songs for level IDs {level_ids} are indexed", 404)

    allowed_faiss_ids = [state.sidecar.faiss_id_for(str(d["_id"])) for d in indexed]

    # ── Feature extraction ────────────────────────────────────────────────────
    # Reuse the long-lived embedder's weights (set up at startup) instead of
    # loading a fresh copy of DINOv3 on every single request.
    emb = state.embedder
    query_features = compute_query_features(
        img,
        processor=emb.processor if emb else None,
        model=emb.model if emb else None,
        device=emb.device if emb else None,
    )
    query_vec      = query_features["cls"].float().numpy()

    # ── FAISS search ──────────────────────────────────────────────────────────
    faiss_hits = restricted_search(
        state.index, query_vec, allowed_faiss_ids,
        top_k=min(state.candidates, len(allowed_faiss_ids)),
    )
    if not faiss_hits:
        return {}

    # ── Full pipeline on candidates ───────────────────────────────────────────
    doc_by_id = {str(d["_id"]): d for d in indexed}
    cfg       = PipelineConfig(use_ocr=False)
    best      = None

    for fid, coarse in faiss_hits:
        mid = state.sidecar.mongo_id_for(fid)
        if not mid or mid not in doc_by_id:
            continue

        emb_path = state.sidecar.emb_path_for(mid)
        if emb_path is None or not emb_path.exists():
            continue

        try:
            cand_features = torch.load(emb_path, map_location="cpu", weights_only=False)
        except Exception:
            continue

        report = compare_features(query_features, cand_features, cfg)

        if report.scores.composite >= state.threshold:
            if best is None or report.scores.composite > best[0].scores.composite:
                best = (report, doc_by_id[mid])

    if best is None:
        return {}

    report, doc = best
    s = report.scores
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "duplicate":           True,
        "mongo_id":            str(doc["_id"]),
        "song_name":           doc.get("songName") or doc.get("name") or "",
        "yt_video_id":         doc.get("ytVideoID") or "",
        "composite":           round(s.composite, 4),
        "verdict":             report.verdict_code,
        "verdict_description": report.verdict,
        "scores": {
            "hash_ensemble":   round(s.hash_ensemble,   4),
            "dct_spectrum":    round(s.dct_spectrum,    4),
            "sector_grid":     round(s.sector_grid,     4),
            "global_semantic": round(s.global_semantic, 4),
            "patch_spatial":   round(s.patch_spatial,   4),
        },
        "grid": {
            "matrix":   report.grid_matrix,
            "min_cell": round(report.grid_min_cell, 4) if report.grid_min_cell is not None else None,
            "variance": round(report.grid_variance,  4) if report.grid_variance  is not None else None,
        },
        "alignment": {
            "applied":      s.align_applied,
            "scale":        round(s.align_scale,        4) if s.align_scale        is not None else None,
            "rotation_deg": round(s.align_rotation_deg, 2) if s.align_rotation_deg is not None else None,
            "inliers":      s.align_inliers,
        },
        "stages_run": report.stages_run,
        "elapsed_ms": round(elapsed_ms, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":        "ok" if state.ready else "starting",
        "index_vectors": state.index.ntotal if state.index else 0,
        "threshold":     state.threshold,
    }


@app.post("/check-duplicate")
async def check_duplicate(request: Request):
    # ── Parse body ────────────────────────────────────────────────────────────
    try:
        body = await request.json()
    except Exception:
        return api_error("MISSING_FIELDS", "Request body must be valid JSON")

    image_url = body.get("image_url", "").strip()
    level_id_raw = body.get("level_ids")

    if not image_url or level_id_raw is None:
        return api_error("MISSING_FIELDS")

    if not image_url.startswith(("http://", "https://")):
        return api_error("INVALID_IMAGE_URL")

    # Accept either a single string or a list of strings
    if isinstance(level_id_raw, list):
        level_ids = [str(l).strip() for l in level_id_raw if str(l).strip()]
    else:
        level_ids = [str(level_id_raw).strip()]

    if not level_ids:
        return api_error("INVALID_LEVEL_ID")

    exclude_id = body.get("exclude_id")
    if exclude_id is not None:
        exclude_id = str(exclude_id).strip() or None

    # ── Fetch image ───────────────────────────────────────────────────────────
    try:
        img = await fetch_image(image_url)
    except APIError as e:
        return api_error(e.code, e.message)

    # ── Run pipeline in thread pool (non-blocking) ────────────────────────────
    try:
        result = await asyncio.to_thread(_check_duplicate_sync, img, level_ids, exclude_id)
    except APIError as e:
        return api_error(e.code, e.message)
    except Exception as e:
        log.error(f"Pipeline error: {e}", exc_info=True)
        return api_error("PIPELINE_ERROR", str(e))

    return JSONResponse(content=result)


# ─────────────────────────────────────────────────────────────────────────────
# Index mutation helpers (sync, run in thread pool)
# All three acquire write_lock before touching index/sidecar/disk.
# ─────────────────────────────────────────────────────────────────────────────

def _embed_and_store(img: Image.Image, mongo_id: str) -> str:
    """
    Embed an image, save .pt to disk, return the pixel hash.
    Does NOT touch the FAISS index or sidecar — caller does that under lock.
    """
    if state.embedder is None:
        raise APIError("EMBEDDER_NOT_READY", "DINOv3 embedder not loaded", 503)

    # Resize to canonical resolution
    if img.size != CANONICAL_SIZE:
        img = img.resize(CANONICAL_SIZE, Image.LANCZOS)

    cls_np, pixel_hashes = state.embedder.embed_batch([img])
    pixel_hash = pixel_hashes[0]

    # embed_batch already writes the .pt file; return hash and cls vector
    return pixel_hash, cls_np[0]   # str, (D,) float32


def _index_add_sync(img: Image.Image, mongo_id: str) -> dict:
    """Add a new document to the FAISS index. Raises APIError if already present."""
    with state.write_lock:
        if mongo_id in state.sidecar:
            raise APIError("DOC_ALREADY_INDEXED",
                           f"{mongo_id} is already indexed — use /edit to update", 409)

        pixel_hash, cls_vec = _embed_and_store(img, mongo_id)
        faiss_id = mongo_id_to_faiss_id(mongo_id)

        # Update in-memory state
        state.index.add_with_ids(
            cls_vec.reshape(1, -1).astype(np.float32),
            np.array([faiss_id], dtype=np.int64),
        )
        state.sidecar.add(mongo_id, faiss_id, pixel_hash)

        # Persist to disk
        try:
            save_index(state.index)
            state.sidecar.save()
            done = load_progress()
            done.add(mongo_id)
            save_progress(done)
        except Exception as e:
            raise APIError("INDEX_WRITE_FAILED", f"Disk write failed: {e}", 500)

        log.info(f"[ADD] {mongo_id} → faiss_id={faiss_id}")
        return {
            "action":    "added",
            "mongo_id":  mongo_id,
            "faiss_id":  faiss_id,
            "emb_key":   pixel_hash,
        }


def _index_edit_sync(img: Image.Image, mongo_id: str) -> dict:
    """
    Update an existing document: remove old vector, embed new image, re-add.
    Raises APIError if the document is not currently indexed.
    """
    with state.write_lock:
        if mongo_id not in state.sidecar:
            raise APIError("DOC_NOT_FOUND",
                           f"{mongo_id} not found in index — use /add to insert", 404)

        old_faiss_id = state.sidecar.faiss_id_for(mongo_id)

        # Delete old .pt file if it exists
        old_emb_path = state.sidecar.emb_path_for(mongo_id)
        if old_emb_path and old_emb_path.exists():
            old_emb_path.unlink(missing_ok=True)

        # Remove old vector from FAISS
        sel = faiss.IDSelectorArray(np.array([old_faiss_id], dtype=np.int64))
        state.index.remove_ids(sel)

        # Remove from sidecar so _embed_and_store doesn't conflict
        state.sidecar.remove(mongo_id)

        # Embed new image
        pixel_hash, cls_vec = _embed_and_store(img, mongo_id)
        faiss_id = mongo_id_to_faiss_id(mongo_id)   # stable — same doc, same hash

        # Re-add with same ID
        state.index.add_with_ids(
            cls_vec.reshape(1, -1).astype(np.float32),
            np.array([faiss_id], dtype=np.int64),
        )
        state.sidecar.add(mongo_id, faiss_id, pixel_hash)

        # Persist
        try:
            save_index(state.index)
            state.sidecar.save()
        except Exception as e:
            raise APIError("INDEX_WRITE_FAILED", f"Disk write failed: {e}", 500)

        log.info(f"[EDIT] {mongo_id} → new emb_key={pixel_hash}")
        return {
            "action":    "updated",
            "mongo_id":  mongo_id,
            "faiss_id":  faiss_id,
            "emb_key":   pixel_hash,
        }


def _index_delete_sync(mongo_id: str) -> dict:
    """Remove a document from the FAISS index and sidecar."""
    with state.write_lock:
        if mongo_id not in state.sidecar:
            raise APIError("DOC_NOT_FOUND",
                           f"{mongo_id} not found in index", 404)

        faiss_id = state.sidecar.faiss_id_for(mongo_id)

        # Delete .pt file
        emb_path = state.sidecar.emb_path_for(mongo_id)
        if emb_path and emb_path.exists():
            emb_path.unlink(missing_ok=True)

        # Remove from FAISS
        sel = faiss.IDSelectorArray(np.array([faiss_id], dtype=np.int64))
        state.index.remove_ids(sel)

        # Remove from sidecar + progress
        state.sidecar.remove(mongo_id)

        # Persist
        try:
            save_index(state.index)
            state.sidecar.save()
            done = load_progress()
            done.discard(mongo_id)
            save_progress(done)
        except Exception as e:
            raise APIError("INDEX_WRITE_FAILED", f"Disk write failed: {e}", 500)

        log.info(f"[DELETE] {mongo_id} (faiss_id={faiss_id})")
        return {
            "action":   "deleted",
            "mongo_id": mongo_id,
            "faiss_id": faiss_id,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Index mutation routes
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/add")
async def index_add(request: Request):
    """
    Add a new document to the index.
    Body: {"image_url": "https://...", "doc_id": "<mongo_id>"}
    Returns: {"action": "added", "mongo_id": ..., "faiss_id": ..., "emb_key": ...}
    Error if doc_id already exists: DOC_ALREADY_INDEXED (409)
    """
    try:
        body = await request.json()
    except Exception:
        return api_error("MISSING_FIELDS", "Request body must be valid JSON")

    image_url = body.get("image_url", "").strip()
    doc_id    = str(body.get("doc_id", "")).strip()

    if not image_url or not doc_id:
        return api_error("MISSING_FIELDS", "Required fields: image_url, doc_id")
    if not image_url.startswith(("http://", "https://")):
        return api_error("INVALID_IMAGE_URL")

    try:
        img = await fetch_image(image_url)
    except APIError as e:
        return api_error(e.code, e.message)

    try:
        result = await asyncio.to_thread(_index_add_sync, img, doc_id)
    except APIError as e:
        return api_error(e.code, e.message)
    except Exception as e:
        log.error(f"[ADD] Unexpected error: {e}", exc_info=True)
        return api_error("PIPELINE_ERROR", str(e))

    return JSONResponse(content=result)


@app.post("/edit")
async def index_edit(request: Request):
    """
    Update an existing document's embedding with a new image.
    Body: {"image_url": "https://...", "doc_id": "<mongo_id>"}
    Returns: {"action": "updated", "mongo_id": ..., "faiss_id": ..., "emb_key": ...}
    Error if doc_id not found: DOC_NOT_FOUND (404)
    """
    try:
        body = await request.json()
    except Exception:
        return api_error("MISSING_FIELDS", "Request body must be valid JSON")

    image_url = body.get("image_url", "").strip()
    doc_id    = str(body.get("doc_id", "")).strip()

    if not image_url or not doc_id:
        return api_error("MISSING_FIELDS", "Required fields: image_url, doc_id")
    if not image_url.startswith(("http://", "https://")):
        return api_error("INVALID_IMAGE_URL")

    try:
        img = await fetch_image(image_url)
    except APIError as e:
        return api_error(e.code, e.message)

    try:
        result = await asyncio.to_thread(_index_edit_sync, img, doc_id)
    except APIError as e:
        return api_error(e.code, e.message)
    except Exception as e:
        log.error(f"[EDIT] Unexpected error: {e}", exc_info=True)
        return api_error("PIPELINE_ERROR", str(e))

    return JSONResponse(content=result)


@app.delete("/delete")
async def index_delete(request: Request):
    """
    Remove a document from the index.
    Body: {"doc_id": "<mongo_id>"}
    Returns: {"action": "deleted", "mongo_id": ..., "faiss_id": ...}
    Error if doc_id not found: DOC_NOT_FOUND (404)
    """
    try:
        body = await request.json()
    except Exception:
        return api_error("MISSING_FIELDS", "Request body must be valid JSON")

    doc_id = str(body.get("doc_id", "")).strip()

    if not doc_id:
        return api_error("MISSING_FIELDS", "Required field: doc_id")

    try:
        result = await asyncio.to_thread(_index_delete_sync, doc_id)
    except APIError as e:
        return api_error(e.code, e.message)
    except Exception as e:
        log.error(f"[DELETE] Unexpected error: {e}", exc_info=True)
        return api_error("PIPELINE_ERROR", str(e))

    return JSONResponse(content=result)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
#
# All settings can be provided as environment variables or CLI flags.
# CLI flags take precedence over env vars.
#
# Environment variables:
#   MONGO_URI       MongoDB connection URI (required)
#   HOST            Bind host              (default: 127.0.0.1)
#   PORT            Bind port              (default: 8000)
#   THRESHOLD       Duplicate threshold    (default: 0.80)
#   CANDIDATES      FAISS candidates       (default: 20)
#   WORKERS         Uvicorn workers        (default: 1)
#   INDEX_DIR       FAISS index/sidecar/progress dir (default: sfh_index)
#   EMB_CACHE_DIR   Cached .pt embeddings dir        (default: .embedding_cache)
#
# INDEX_DIR and EMB_CACHE_DIR are read by query.py/build.py at import time —
# set them in .env (or the environment) before starting the server, build, or
# query tools so all three agree on where the index and cache live.
#
# Example .env / docker-compose:
#   MONGO_URI=mongodb://localhost:27017
#   HOST=0.0.0.0
#   PORT=8000
#   THRESHOLD=0.80
#   INDEX_DIR=sfh_index
#   EMB_CACHE_DIR=.embedding_cache
# ─────────────────────────────────────────────────────────────────────────────

def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def main():
    parser = argparse.ArgumentParser(
        description="SFH Thumbnail Duplicate Detection API Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--uri",
        default=_env("MONGO_URI"),
        help="MongoDB connection URI [env: MONGO_URI]",
    )
    parser.add_argument(
        "--host",
        default=_env("HOST", "127.0.0.1"),
        help="Bind host [env: HOST]",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(_env("PORT", "8000")),
        help="Bind port [env: PORT]",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(_env("THRESHOLD", "0.80")),
        help="Duplicate composite score threshold [env: THRESHOLD]",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=int(_env("CANDIDATES", "20")),
        help="FAISS candidates per query [env: CANDIDATES]",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(_env("WORKERS", "1")),
        help="Uvicorn worker processes [env: WORKERS]",
    )
    args = parser.parse_args()

    if not args.uri:
        parser.error(
            "MongoDB URI is required. Set --uri or MONGO_URI environment variable."
        )

    state.mongo_uri  = args.uri
    state.threshold  = args.threshold
    state.candidates = args.candidates

    if args.workers != 1:
        log.warning(
            "Ignoring --workers=%d: in-process state (FAISS index, sidecar, "
            "write lock) is single-process only. Run multiple instances "
            "behind a proxy if you need concurrency.", args.workers
        )

    if sys.platform == "win32":
        # The default ProactorEventLoop on Windows waits on IOCP completion
        # ports and only checks for signals when one of those wakes it up,
        # so Ctrl+C (SIGINT) can sit unhandled while the server is idle.
        # SelectorEventLoop polls with a timeout and reacts to it promptly.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Pass the app object (not "server:app") — passing an import string makes
    # uvicorn re-import this file as a *second* module, creating a fresh
    # AppState with a blank mongo_uri instead of the one configured above.
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()