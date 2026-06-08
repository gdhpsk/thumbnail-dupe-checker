"""
SFH Batch Duplicate Finder
============================
Compares every indexed song against every other indexed song using cached
embeddings. No level ID filtering — full corpus scan.

Flow:
  1. Load all cached .pt files from .embedding_cache/ via sidecar
  2. Use FAISS to find top-K nearest neighbors for every song (batch search)
  3. For each candidate pair above FAISS coarse threshold, run full compare_features()
  4. Collect all pairs with composite >= threshold (default 0.80)
  5. Fetch song metadata from MongoDB for the report
  6. Write a sorted HTML report

This script is CPU-safe — no DINOv3 forward passes needed since all
embeddings are already cached. The only GPU op is the 196x196 patch matmul
which runs fine on CPU too.

Usage:
  python batch_dupes.py --uri "mongodb://localhost:27017"
  python batch_dupes.py --uri "mongodb://localhost:27017" --threshold 0.85
  python batch_dupes.py --uri "mongodb://localhost:27017" --out report.html
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from pymongo import MongoClient

from duplicate_detector.image_similarity import (
    PipelineConfig,
    StageScores,
    SimilarityReport,
)

# Reuse comparison logic from query.py
sys.path.insert(0, str(Path(__file__).parent))
from duplicate_detector.query import Sidecar, compare_features, _hamming_score, CANONICAL_SIZE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

INDEX_DIR     = Path("sfh_index")
INDEX_PATH    = INDEX_DIR / "index.faiss"
SIDECAR_PATH  = INDEX_DIR / "sidecar.json"
EMB_CACHE_DIR = Path(".embedding_cache")


# ─────────────────────────────────────────────────────────────────────────────
# Batch FAISS search — find top-K neighbors for all vectors at once
# ─────────────────────────────────────────────────────────────────────────────

def batch_faiss_search(
    index:   faiss.IndexIDMap,
    vectors: np.ndarray,        # (N, D)
    faiss_ids: list[int],       # corresponding FAISS IDs for each vector
    top_k:   int = 20,
) -> list[list[tuple[int, float]]]:
    """
    Search every vector against the full index.
    Returns list of N result lists, each containing (faiss_id, score) tuples
    excluding self-matches.
    """
    scores, ids = index.search(vectors, top_k + 1)  # +1 to account for self

    id_set = set(faiss_ids)
    results = []
    for i, (row_scores, row_ids) in enumerate(zip(scores, ids)):
        self_id = faiss_ids[i]
        neighbors = [
            (int(fid), float(s))
            for fid, s in zip(row_ids, row_scores)
            if fid != -1 and int(fid) != self_id
        ]
        results.append(neighbors[:top_k])
    return results


# ─────────────────────────────────────────────────────────────────────────────
# HTML report generator
# ─────────────────────────────────────────────────────────────────────────────

def score_to_color(score: float) -> str:
    """Map composite score to a color between orange (0.80) and green (1.0)."""
    t = (score - 0.80) / 0.20  # 0.0 at threshold, 1.0 at perfect
    t = max(0.0, min(1.0, t))
    r = int(255 * (1 - t))
    g = int(200 * t + 55)
    return f"rgb({r},{g},60)"


def verdict_badge(code: str) -> str:
    colors = {
        "EXACT":            "#ff4444",
        "MICRO_VARIANT":    "#ff8800",
        "MODIFIED_VARIANT": "#ffcc00",
        "DISTANT":          "#88aaff",
        "DISTINCT":         "#888888",
    }
    color = colors.get(code, "#888888")
    label = code.replace("_", " ")
    return f'<span class="badge" style="background:{color}">{label}</span>'


def build_html_report(
    pairs: list[dict],
    total_indexed: int,
    total_pairs_checked: int,
    elapsed: float,
    threshold: float,
    generated_at: str,
) -> str:
    rows = ""
    for i, p in enumerate(pairs):
        a  = p["song_a"]
        b  = p["song_b"]
        r  = p["report"]
        s  = r.scores
        yt_a = a.get("ytVideoID", "")
        yt_b = b.get("ytVideoID", "")
        thumb_a = f"https://img.youtube.com/vi/{yt_a}/mqdefault.jpg" if yt_a else ""
        thumb_b = f"https://img.youtube.com/vi/{yt_b}/mqdefault.jpg" if yt_b else ""
        name_a = a.get("songName") or a.get("name") or str(a["_id"])
        name_b = b.get("songName") or b.get("name") or str(b["_id"])
        yt_link_a = f"https://youtube.com/watch?v={yt_a}" if yt_a else "#"
        yt_link_b = f"https://youtube.com/watch?v={yt_b}" if yt_b else "#"
        score_color = score_to_color(s.composite)
        badge = verdict_badge(r.verdict_code)

        # Grid heatmap
        grid_html = ""
        if r.grid_matrix:
            grid_html = '<div class="grid-wrap"><div class="grid">'
            for row in r.grid_matrix:
                for cell in row:
                    t = max(0.0, min(1.0, cell))
                    # blue (similar) to red (different)
                    br = int(255 * (1-t))
                    bg = int(180 * t)
                    bb = int(255 * t)
                    grid_html += f'<div class="cell" style="background:rgb({br},{bg},{bb})" title="{cell:.2f}"></div>'
            grid_html += "</div></div>"

        align_info = ""
        if s.align_scale is not None:
            applied = "✓" if s.align_applied else "✗"
            align_info = f'<div class="align-info">{applied} align: {s.align_scale:.3f}x / {s.align_rotation_deg:.1f}° / {s.align_inliers} inliers</div>'

        rows += f"""
        <tr class="pair-row" data-score="{s.composite:.4f}">
          <td class="rank">#{i+1}</td>
          <td class="score-cell">
            <div class="score-val" style="color:{score_color}">{s.composite:.4f}</div>
            {badge}
          </td>
          <td class="song-cell">
            <a href="{yt_link_a}" target="_blank">
              {"<img class='thumb' src='"+thumb_a+"' onerror=\"this.style.display='none'\">" if thumb_a else ""}
            </a>
            <div class="song-info">
              <div class="song-name">{name_a}</div>
              <div class="song-id">ID: {a['_id']}</div>
              <div class="yt-id">YT: {yt_a or "N/A"}</div>
            </div>
          </td>
          <td class="song-cell">
            <a href="{yt_link_b}" target="_blank">
              {"<img class='thumb' src='"+thumb_b+"' onerror=\"this.style.display='none'\">" if thumb_b else ""}
            </a>
            <div class="song-info">
              <div class="song-name">{name_b}</div>
              <div class="song-id">ID: {b['_id']}</div>
              <div class="yt-id">YT: {yt_b or "N/A"}</div>
            </div>
          </td>
          <td class="stages-cell">
            <div class="stage-scores">
              <div class="stage-row"><span>Hash</span><span class="sv">{s.hash_ensemble:.3f}</span></div>
              <div class="stage-row"><span>DCT</span><span class="sv">{s.dct_spectrum:.3f}</span></div>
              <div class="stage-row"><span>Grid</span><span class="sv">{s.sector_grid:.3f}</span></div>
              <div class="stage-row"><span>Global</span><span class="sv">{s.global_semantic:.3f}</span></div>
              <div class="stage-row"><span>Patch</span><span class="sv">{s.patch_spatial:.3f}</span></div>
            </div>
            {align_info}
            {grid_html}
          </td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SFH Duplicate Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:      #0d0f14;
    --surface: #13161e;
    --border:  #1e2330;
    --text:    #c8cdd8;
    --muted:   #555d72;
    --accent:  #4f8ef7;
    --danger:  #ff4444;
    --warn:    #ff8800;
    --ok:      #37c97d;
    --font-mono: 'Space Mono', monospace;
    --font-body: 'DM Sans', sans-serif;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-body);
    font-size: 14px;
    line-height: 1.5;
  }}

  header {{
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 28px 40px;
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 24px;
  }}

  header h1 {{
    font-family: var(--font-mono);
    font-size: 22px;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.5px;
  }}

  header h1 span {{ color: var(--accent); }}

  .meta {{
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--muted);
    line-height: 1.9;
    text-align: right;
  }}

  .stats {{
    display: flex;
    gap: 1px;
    background: var(--border);
    border-bottom: 1px solid var(--border);
  }}

  .stat {{
    flex: 1;
    background: var(--surface);
    padding: 16px 24px;
    text-align: center;
  }}

  .stat-val {{
    font-family: var(--font-mono);
    font-size: 26px;
    font-weight: 700;
    color: #fff;
  }}

  .stat-val.danger {{ color: var(--danger); }}
  .stat-val.warn   {{ color: var(--warn); }}
  .stat-val.ok     {{ color: var(--ok); }}

  .stat-label {{
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
  }}

  .controls {{
    padding: 16px 40px;
    display: flex;
    gap: 12px;
    align-items: center;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }}

  .controls label {{
    font-size: 12px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
  }}

  select, input[type=range] {{
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 6px 10px;
    font-size: 13px;
    font-family: var(--font-body);
    border-radius: 4px;
    cursor: pointer;
  }}

  .range-val {{
    font-family: var(--font-mono);
    font-size: 13px;
    color: var(--accent);
    min-width: 42px;
  }}

  .count-badge {{
    margin-left: auto;
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--muted);
  }}

  .table-wrap {{
    overflow-x: auto;
    padding: 24px 40px;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
  }}

  thead th {{
    font-family: var(--font-mono);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--muted);
    text-align: left;
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
  }}

  tbody tr {{
    border-bottom: 1px solid var(--border);
    transition: background 0.15s;
  }}

  tbody tr:hover {{ background: rgba(255,255,255,0.02); }}

  td {{
    padding: 14px 16px;
    vertical-align: top;
  }}

  .rank {{
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--muted);
    width: 40px;
    white-space: nowrap;
  }}

  .score-cell {{
    width: 120px;
    text-align: center;
  }}

  .score-val {{
    font-family: var(--font-mono);
    font-size: 22px;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 6px;
  }}

  .badge {{
    display: inline-block;
    font-size: 9px;
    font-family: var(--font-mono);
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 3px 7px;
    border-radius: 3px;
    color: #000;
    white-space: nowrap;
  }}

  .song-cell {{
    min-width: 240px;
    max-width: 300px;
  }}

  .song-cell a {{
    display: block;
    margin-bottom: 8px;
  }}

  .thumb {{
    width: 120px;
    height: 68px;
    object-fit: cover;
    border-radius: 4px;
    border: 1px solid var(--border);
    display: block;
    transition: opacity 0.2s;
  }}

  .thumb:hover {{ opacity: 0.8; }}

  .song-name {{
    font-size: 13px;
    font-weight: 500;
    color: #fff;
    margin-bottom: 3px;
    line-height: 1.3;
  }}

  .song-id, .yt-id {{
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--muted);
  }}

  .stages-cell {{
    width: 200px;
  }}

  .stage-scores {{
    display: flex;
    flex-direction: column;
    gap: 3px;
    margin-bottom: 10px;
  }}

  .stage-row {{
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: var(--muted);
  }}

  .sv {{
    font-family: var(--font-mono);
    color: var(--text);
  }}

  .align-info {{
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--muted);
    margin-bottom: 8px;
    padding: 4px 6px;
    background: rgba(255,255,255,0.03);
    border-radius: 3px;
    border-left: 2px solid var(--border);
  }}

  .grid-wrap {{
    margin-top: 4px;
  }}

  .grid {{
    display: grid;
    grid-template-columns: repeat(4, 18px);
    gap: 2px;
  }}

  .cell {{
    width: 18px;
    height: 18px;
    border-radius: 2px;
    cursor: default;
    transition: transform 0.1s;
  }}

  .cell:hover {{ transform: scale(1.4); }}

  .empty {{
    padding: 60px;
    text-align: center;
    color: var(--muted);
    font-family: var(--font-mono);
    font-size: 13px;
  }}

  footer {{
    padding: 24px 40px;
    border-top: 1px solid var(--border);
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--muted);
    text-align: center;
  }}

  @media (max-width: 900px) {{
    header {{ flex-direction: column; }}
    .meta {{ text-align: left; }}
    .table-wrap {{ padding: 16px; }}
    td {{ padding: 10px 8px; }}
  }}
</style>
</head>
<body>

<header>
  <div>
    <h1>SFH <span>//</span> Duplicate Report</h1>
    <div style="font-size:12px;color:var(--muted);margin-top:6px;">
      Image similarity pipeline — batch corpus scan
    </div>
  </div>
  <div class="meta">
    Generated: {generated_at}<br>
    Threshold: ≥ {threshold}<br>
    Songs indexed: {total_indexed:,}<br>
    Pairs checked: {total_pairs_checked:,}<br>
    Elapsed: {elapsed:.1f}s
  </div>
</header>

<div class="stats">
  <div class="stat">
    <div class="stat-val {'danger' if len(pairs) > 0 else 'ok'}">{len(pairs)}</div>
    <div class="stat-label">Duplicate pairs found</div>
  </div>
  <div class="stat">
    <div class="stat-val">{total_indexed:,}</div>
    <div class="stat-label">Songs indexed</div>
  </div>
  <div class="stat">
    <div class="stat-val">{total_pairs_checked:,}</div>
    <div class="stat-label">Pairs evaluated</div>
  </div>
  <div class="stat">
    <div class="stat-val">{elapsed:.1f}s</div>
    <div class="stat-label">Total runtime</div>
  </div>
  <div class="stat">
    <div class="stat-val">{f"{pairs[0]['report'].scores.composite:.3f}" if pairs else "—"}</div>
    <div class="stat-label">Highest score</div>
  </div>
</div>

<div class="controls">
  <label>Min score</label>
  <input type="range" id="filterRange" min="{threshold}" max="1.0" step="0.01"
         value="{threshold}" oninput="filterRows(this.value)">
  <span class="range-val" id="rangeVal">{threshold}</span>
  <label style="margin-left:16px">Sort</label>
  <select onchange="sortRows(this.value)">
    <option value="score-desc">Score ↓</option>
    <option value="score-asc">Score ↑</option>
  </select>
  <span class="count-badge" id="countBadge">{len(pairs)} pairs shown</span>
</div>

<div class="table-wrap">
  {"<table><thead><tr><th>#</th><th>Score</th><th>Song A</th><th>Song B</th><th>Stage Scores</th></tr></thead><tbody id='tbody'>" + rows + "</tbody></table>" if pairs else '<div class="empty">No duplicate pairs found above threshold ' + str(threshold) + '</div>'}
</div>

<footer>
  SongFileHub · Duplicate Detection Pipeline · {generated_at}
</footer>

<script>
  function filterRows(val) {{
    document.getElementById('rangeVal').textContent = parseFloat(val).toFixed(2);
    const rows = document.querySelectorAll('.pair-row');
    let shown = 0;
    rows.forEach(r => {{
      const score = parseFloat(r.dataset.score);
      const visible = score >= parseFloat(val);
      r.style.display = visible ? '' : 'none';
      if (visible) shown++;
    }});
    document.getElementById('countBadge').textContent = shown + ' pairs shown';
    // Re-number visible rows
    let n = 1;
    rows.forEach(r => {{
      if (r.style.display !== 'none') {{
        r.querySelector('.rank').textContent = '#' + n++;
      }}
    }});
  }}

  function sortRows(val) {{
    const tbody = document.getElementById('tbody');
    if (!tbody) return;
    const rows = Array.from(tbody.querySelectorAll('.pair-row'));
    rows.sort((a, b) => {{
      const sa = parseFloat(a.dataset.score);
      const sb = parseFloat(b.dataset.score);
      return val === 'score-asc' ? sa - sb : sb - sa;
    }});
    rows.forEach(r => tbody.appendChild(r));
  }}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Top-level worker — must be at module level to be picklable for multiprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _worker_compare(args):
    """
    Loads .pt files from disk and runs compare_features.
    Only small primitives (paths, strings, floats) are passed via IPC —
    no large tensors or numpy arrays cross process boundaries.

    args: (mongo_id_a, mongo_id_b, coarse, pt_path_a, pt_path_b, threshold)
    Returns a match dict or None.
    """
    import os
    # Prevent OpenBLAS/OMP spawning sub-threads per worker process.
    # With N_WORKERS processes each spawning their own thread pools,
    # memory pressure causes malloc failures. Single-threaded BLAS is
    # fine here since parallelism comes from the process pool itself.
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)

    mongo_id_a, mongo_id_b, coarse, pt_path_a, pt_path_b, threshold = args

    try:
        feat_a = torch.load(pt_path_a, map_location="cpu", weights_only=False)
        feat_b = torch.load(pt_path_b, map_location="cpu", weights_only=False)
    except Exception:
        return None

    cfg = PipelineConfig(use_ocr=False)

    # Short-circuit: hash ensemble first (cheap bit ops, no tensor math)
    ph = _hamming_score(feat_a["hash_phash"], feat_b["hash_phash"])
    dh = _hamming_score(feat_a["hash_dhash"], feat_b["hash_dhash"])
    ah = _hamming_score(feat_a["hash_ahash"], feat_b["hash_ahash"])
    wh = _hamming_score(feat_a["hash_whash"], feat_b["hash_whash"])
    hash_ensemble = ph*0.35 + dh*0.35 + ah*0.15 + wh*0.15

    max_possible = (
        hash_ensemble * cfg.weight_hash + 1.0 * (
            cfg.weight_grid + cfg.weight_dct +
            cfg.weight_global + cfg.weight_patch
        )
    )
    if max_possible < threshold:
        return None

    report = compare_features(feat_a, feat_b, cfg)
    if report.scores.composite >= threshold:
        return {
            "mongo_id_a": mongo_id_a,
            "mongo_id_b": mongo_id_b,
            "coarse":     coarse,
            "report":     report,
            "song_a":     {"_id": mongo_id_a},
            "song_b":     {"_id": mongo_id_b},
        }
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised batch comparison
# Processes N pairs simultaneously using batched torch ops instead of a loop.
# ─────────────────────────────────────────────────────────────────────────────

def _precompute_patch_penalty(grid_size: int = 14) -> torch.Tensor:
    """
    Gaussian spatial penalty matrix — identical for every pair so computed once.
    Returns (196, 196) float32 tensor.
    """
    positions = torch.stack(torch.meshgrid(
        torch.arange(grid_size),
        torch.arange(grid_size),
        indexing="ij",
    ), dim=-1).reshape(196, 2).float()
    pos_dist    = torch.cdist(positions, positions)      # (196, 196)
    sigma       = grid_size / 3.0
    return torch.exp(-pos_dist**2 / (2 * sigma**2))     # (196, 196)


# Precompute once at import time
_PATCH_PENALTY = _precompute_patch_penalty()


def batch_compare(
    feats_a: list[dict],
    feats_b: list[dict],
    cfg: PipelineConfig,
) -> list[dict]:
    """
    Compare N pairs simultaneously using vectorised ops.

    Returns a list of N dicts, each with keys:
      composite, hash_ensemble, dct_spectrum, sector_grid,
      global_semantic, patch_spatial
    """
    N = len(feats_a)
    assert N == len(feats_b)

    # ── Hash ensemble (numpy, fully vectorised) ───────────────────────────────
    # Stack all hash bits into (N, bits) matrices, XOR, sum across bits axis
    max_bits = feats_a[0]["hash_phash"].shape[0]

    def hamming_batch(key):
        a = np.stack([f[key] for f in feats_a])   # (N, bits)
        b = np.stack([f[key] for f in feats_b])
        dist = np.sum(a != b, axis=1).astype(np.float32)
        return np.clip(1.0 - dist / max_bits, 0, 1)  # (N,)

    ph = hamming_batch("hash_phash")
    dh = hamming_batch("hash_dhash")
    ah = hamming_batch("hash_ahash")
    wh = hamming_batch("hash_whash")
    hash_ens = ph*0.35 + dh*0.35 + ah*0.15 + wh*0.15  # (N,)

    # ── DCT spectrum (numpy) ──────────────────────────────────────────────────
    e1 = np.stack([f["dct_bands"] for f in feats_a])  # (N, 3)
    e2 = np.stack([f["dct_bands"] for f in feats_b])
    dot  = np.sum(e1 * e2, axis=1)
    norm = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) + 1e-9
    dct_spec = np.clip(dot / norm, 0, 1)               # (N,)

    # ── Sector grid (numpy) ───────────────────────────────────────────────────
    # (N, R, C, bits) XOR then mean over cells
    grid_a = np.stack([f["grid_hashes"] for f in feats_a])  # (N, R, C, bits)
    grid_b = np.stack([f["grid_hashes"] for f in feats_b])
    grid_bits = grid_a.shape[-1]
    cell_dist  = np.sum(grid_a != grid_b, axis=-1).astype(np.float32)  # (N, R, C)
    cell_score = np.clip(1.0 - cell_dist / grid_bits, 0, 1)            # (N, R, C)
    grid_mean  = cell_score.mean(axis=(1, 2))                           # (N,)
    grid_mat   = cell_score                                             # (N, R, C) kept for report

    # ── DINOv3 global (torch, vectorised) ────────────────────────────────────
    cls_a = torch.stack([f["cls"].squeeze(0) for f in feats_a])  # (N, D)
    cls_b = torch.stack([f["cls"].squeeze(0) for f in feats_b])
    global_sim = F.cosine_similarity(cls_a, cls_b, dim=1).clamp(0, 1)  # (N,)

    # ── DINOv3 patch spatial (torch bmm) ─────────────────────────────────────
    patches_a = torch.stack([f["patches"] for f in feats_a])  # (N, 196, D)
    patches_b = torch.stack([f["patches"] for f in feats_b])  # (N, 196, D)

    # Batched matmul: (N, 196, D) @ (N, D, 196) = (N, 196, 196)
    sim_mats = torch.bmm(patches_a, patches_b.transpose(1, 2))  # (N, 196, 196)

    # Apply spatial penalty — broadcast (1, 196, 196) over batch
    penalty  = _PATCH_PENALTY.unsqueeze(0)                      # (1, 196, 196)
    weighted = sim_mats * penalty                               # (N, 196, 196)

    # Best spatially-weighted match per patch, averaged
    patch_scores = weighted.max(dim=2).values.mean(dim=1).clamp(0, 1)  # (N,)

    # ── Composite ────────────────────────────────────────────────────────────
    w = cfg
    total_w = w.weight_hash + w.weight_grid + w.weight_dct + w.weight_global + w.weight_patch

    global_np  = global_sim.numpy()
    patch_np   = patch_scores.numpy()
    hash_ens   = hash_ens
    composite  = (
        hash_ens   * w.weight_hash   +
        grid_mean  * w.weight_grid   +
        dct_spec   * w.weight_dct    +
        global_np  * w.weight_global +
        patch_np   * w.weight_patch
    ) / total_w                                                  # (N,)

    # Grid variance penalty — vectorised over batch
    grid_var  = cell_score.var(axis=(1, 2))                      # (N,)
    composite = np.clip(composite - grid_var * w.grid_variance_penalty, 0, 1)

    # ── Build result dicts ────────────────────────────────────────────────────
    results = []
    for i in range(N):
        results.append({
            "composite":       float(composite[i]),
            "hash_ensemble":   float(hash_ens[i]),
            "dct_spectrum":    float(dct_spec[i]),
            "sector_grid":     float(grid_mean[i]),
            "global_semantic": float(global_np[i]),
            "patch_spatial":   float(patch_np[i]),
            "grid_matrix":     grid_mat[i].tolist(),
            "grid_min_cell":   float(cell_score[i].min()),
            "grid_variance":   float(cell_score[i].var()),
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main batch scan
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(
    mongo_uri:  str,
    threshold:  float = 0.80,
    faiss_top_k: int  = 20,    # FAISS neighbors per song before full pipeline
    out_path:   str   = "sfh_duplicates.html",
) -> None:
    t_start = time.perf_counter()

    if not INDEX_PATH.exists():
        log.error("FAISS index not found. Run: python build_index.py build --uri ...")
        sys.exit(1)

    # ── Load index + sidecar ─────────────────────────────────────────────────
    log.info("Loading FAISS index and sidecar...")
    sidecar = Sidecar.load()
    index   = faiss.read_index(str(INDEX_PATH))
    log.info(f"  {index.ntotal} vectors in index | {len(sidecar.by_mongo)} sidecar entries")

    # Build ordered list of (mongo_id, faiss_id) for all indexed songs
    entries = [
        (mid, fid)
        for mid, fid in sidecar.by_mongo.items()
        if sidecar.emb_path_for(mid) and sidecar.emb_path_for(mid).exists()
    ]
    log.info(f"  {len(entries)} entries with cached embeddings")

    if not entries:
        log.error("No cached embeddings found. Run build_index.py first.")
        sys.exit(1)

    # ── Load CLS vectors in parallel using a thread pool ────────────────────────
    # Disk I/O is the bottleneck here, not CPU — threading bypasses the GIL and
    # lets multiple reads happen concurrently. We only extract the small CLS
    # tensor from each .pt file (not the full dict) to keep memory low.
    import concurrent.futures

    mongo_ids = [mid for mid, _ in entries]
    faiss_ids = [fid for _, fid in entries]
    N = len(mongo_ids)

    log.info(f"Loading {N} CLS vectors in parallel...")

    def load_cls(args):
        i, mid = args
        pt_path = sidecar.emb_path_for(mid)
        if pt_path is None or not pt_path.exists():
            return i, None
        try:
            cache = torch.load(pt_path, map_location="cpu", weights_only=False)
            return i, cache["cls"].float().numpy().squeeze(0)
        except Exception:
            return i, None

    all_vecs   = np.zeros((N, 1024), dtype=np.float32)
    valid_mask = np.zeros(N, dtype=bool)

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
        futures = ex.map(load_cls, enumerate(mongo_ids))
        done = 0
        for i, vec in futures:
            if vec is not None:
                all_vecs[i]   = vec
                valid_mask[i] = True
            done += 1
            if done % 5000 == 0:
                log.info(f"  {done}/{N} loaded...")

    # Filter to valid entries only
    mongo_ids = [mid for mid, v in zip(mongo_ids, valid_mask) if v]
    faiss_ids = [fid for fid, v in zip(faiss_ids, valid_mask) if v]
    all_vecs  = all_vecs[valid_mask]
    N = len(mongo_ids)
    log.info(f"  {N} CLS vectors loaded")

    # ── Batch FAISS search ───────────────────────────────────────────────────
    log.info(f"Batch FAISS search (top-{faiss_top_k} neighbors per song)...")
    chunk_size = 512
    all_neighbors: list[list[tuple[int, float]]] = []
    faiss_id_to_idx = {fid: i for i, fid in enumerate(faiss_ids)}

    for start in range(0, N, chunk_size):
        chunk = all_vecs[start:start+chunk_size]
        chunk_faiss_ids = faiss_ids[start:start+chunk_size]
        nbrs = batch_faiss_search(index, chunk, chunk_faiss_ids, top_k=faiss_top_k)
        all_neighbors.extend(nbrs)
        if start % 2048 == 0:
            log.info(f"  FAISS search: {start}/{N}...")

    # ── Deduplicate pairs + apply coarse FAISS threshold ────────────────────────
    log.info("Deduplicating candidate pairs...")
    seen_pairs: set[frozenset] = set()
    candidate_pairs: list[tuple[int, int, float]] = []

    COARSE_THRESHOLD = max(threshold - 0.05, 0.65)  # slightly below final threshold, min 0.65
    skipped_coarse = 0

    for i, neighbors in enumerate(all_neighbors):
        for fid_b, coarse in neighbors:
            j = faiss_id_to_idx.get(fid_b)
            if j is None or j == i:
                continue
            if coarse < COARSE_THRESHOLD:
                skipped_coarse += 1
                continue
            pair_key = frozenset([i, j])
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                candidate_pairs.append((i, j, coarse))

    log.info(f"  {len(candidate_pairs)} unique pairs after dedup (skipped {skipped_coarse} below coarse threshold {COARSE_THRESHOLD:.2f})")

    # ── Stage A: load hash bits only for all candidate songs (parallel) ─────────
    # Hash bits are tiny (~256 bytes each). Load them all into RAM once so we
    # can run the hash short-circuit in the main process without any per-pair
    # disk I/O. This eliminates redundant .pt reads across pairs sharing songs.
    log.info("Loading hash bits for all candidate songs...")

    needed_indices: set[int] = set()
    for idx_a, idx_b, _ in candidate_pairs:
        needed_indices.add(idx_a)
        needed_indices.add(idx_b)

    hash_cache: dict[int, dict] = {}  # idx -> hash arrays only

    def load_hashes(i):
        mid     = mongo_ids[i]
        pt_path = sidecar.emb_path_for(mid)
        if pt_path and pt_path.exists():
            try:
                d = torch.load(pt_path, map_location="cpu", weights_only=False)
                return i, {
                    "hash_phash": d["hash_phash"],
                    "hash_dhash": d["hash_dhash"],
                    "hash_ahash": d["hash_ahash"],
                    "hash_whash": d["hash_whash"],
                    "pt_path":    str(pt_path),
                }
            except Exception:
                pass
        return i, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
        for i, h in ex.map(load_hashes, list(needed_indices)):
            if h is not None:
                hash_cache[i] = h

    log.info(f"  {len(hash_cache)} hash dicts loaded")

    # ── Stage B: build survivor list from hash cache ─────────────────────────
    cfg = PipelineConfig(use_ocr=False)
    survivors = []
    skipped_missing = 0

    for idx_a, idx_b, coarse in candidate_pairs:
        ha = hash_cache.get(idx_a)
        hb = hash_cache.get(idx_b)
        if ha is None or hb is None:
            skipped_missing += 1
            continue
        survivors.append((
            mongo_ids[idx_a], mongo_ids[idx_b], coarse,
            ha["pt_path"], hb["pt_path"],
        ))

    if skipped_missing:
        log.warning(f"  {skipped_missing} pairs skipped (missing hash cache)")
    log.info(f"  {len(survivors)} pairs ready for full pipeline")

    # ── Stage C: batched vectorised comparison ───────────────────────────────
    # Load all survivor .pt files in parallel, then run batch_compare() in
    # chunks — one torch.bmm for N pairs instead of N individual matmuls.
    # Chunk size of 512 keeps memory reasonable (~512 * 196 * 1024 * 4 = ~400MB).
    BATCH_SIZE = 512
    N_LOAD_WORKERS = 32

    log.info(f"Loading {len(survivors)} survivor feature dicts in parallel...")

    # Load all needed .pt files — parallel reads, each file loaded once
    survivor_pt_paths = list({pt for _, _, _, pt, _ in survivors} |
                              {pt for _, _, _, _, pt in survivors})
    feat_store: dict[str, dict] = {}

    def load_pt(path):
        try:
            return path, torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return path, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=N_LOAD_WORKERS) as ex:
        for path, feat in ex.map(load_pt, survivor_pt_paths):
            if feat is not None:
                feat_store[path] = feat

    log.info(f"  {len(feat_store)} feature dicts loaded")

    log.info(f"Running batched pipeline on {len(survivors)} pairs "
             f"(batch_size={BATCH_SIZE})...")

    VERDICTS = {
        "EXACT":            "EXACT DUPLICATE — byte/pixel-level identical or losslessly re-encoded",
        "MICRO_VARIANT":    "MICRO VARIANT — same base image; minor mutation (crop, watermark, text overlay, compression)",
        "MODIFIED_VARIANT": "MODIFIED VARIANT — same scene/subject; significant layout or content change",
        "DISTANT":          "DISTANT RELATIVE — loosely related (same template/style, different content)",
        "DISTINCT":         "DISTINCT IMAGES — no meaningful similarity detected",
    }

    def make_verdict(composite):
        if   composite >= cfg.thresh_exact:    return VERDICTS["EXACT"],            "EXACT"
        elif composite >= cfg.thresh_micro:    return VERDICTS["MICRO_VARIANT"],    "MICRO_VARIANT"
        elif composite >= cfg.thresh_variant:  return VERDICTS["MODIFIED_VARIANT"], "MODIFIED_VARIANT"
        elif composite >= cfg.thresh_distant:  return VERDICTS["DISTANT"],          "DISTANT"
        else:                                  return VERDICTS["DISTINCT"],          "DISTINCT"

    matches = []
    checked = 0

    for batch_start in range(0, len(survivors), BATCH_SIZE):
        batch = survivors[batch_start:batch_start + BATCH_SIZE]

        feats_a, feats_b, mids_a, mids_b, coarses = [], [], [], [], []
        for mid_a, mid_b, coarse, pt_a, pt_b in batch:
            fa = feat_store.get(pt_a)
            fb = feat_store.get(pt_b)
            if fa is None or fb is None:
                continue
            feats_a.append(fa)
            feats_b.append(fb)
            mids_a.append(mid_a)
            mids_b.append(mid_b)
            coarses.append(coarse)

        if not feats_a:
            continue

        results = batch_compare(feats_a, feats_b, cfg)
        checked += len(results)

        for i, res in enumerate(results):
            if res["composite"] >= threshold:
                scores = StageScores(
                    hash_ensemble   = res["hash_ensemble"],
                    dct_spectrum    = res["dct_spectrum"],
                    sector_grid     = res["sector_grid"],
                    global_semantic = res["global_semantic"],
                    patch_spatial   = res["patch_spatial"],
                    composite       = res["composite"],
                )
                verdict, code = make_verdict(res["composite"])
                report = SimilarityReport(
                    path_a      = "<batch_a>",
                    path_b      = "<batch_b>",
                    scores      = scores,
                    verdict     = verdict,
                    verdict_code= code,
                    grid_matrix = res["grid_matrix"],
                    grid_min_cell = res["grid_min_cell"],
                    grid_variance = res["grid_variance"],
                    elapsed_ms  = 0.0,
                    stages_run  = ["hash_ensemble","dct_spectrum","sector_grid",
                                   "dinov3_global","dinov3_patch"],
                )
                matches.append({
                    "mongo_id_a": mids_a[i],
                    "mongo_id_b": mids_b[i],
                    "coarse":     coarses[i],
                    "report":     report,
                    "song_a":     {"_id": mids_a[i]},
                    "song_b":     {"_id": mids_b[i]},
                })

        if checked % 10000 == 0:
            log.info(f"  {checked}/{len(survivors)} pairs checked "
                     f"| {len(matches)} matches so far")

    log.info(f"Pipeline done: {checked} pairs checked, {len(matches)} matches above {threshold}")

    # ── Fetch song metadata from MongoDB ─────────────────────────────────────
    if matches:
        log.info("Fetching song metadata from MongoDB...")
        all_ids = set()
        for m in matches:
            all_ids.add(m["mongo_id_a"])
            all_ids.add(m["mongo_id_b"])

        from bson import ObjectId
        client     = MongoClient(mongo_uri)
        collection = client["SFH"]["songs"]

        # Convert string IDs to ObjectId where valid
        oid_map = {}
        for sid in all_ids:
            try:
                oid_map[sid] = ObjectId(sid)
            except Exception:
                oid_map[sid] = sid

        docs = collection.find(
            {"_id": {"$in": list(oid_map.values())}},
            {"_id": 1, "ytVideoID": 1, "songName": 1, "name": 1},
        )
        doc_by_id = {str(d["_id"]): d for d in docs}
        client.close()

        for m in matches:
            m["song_a"] = doc_by_id.get(m["mongo_id_a"], {"_id": m["mongo_id_a"]})
            m["song_b"] = doc_by_id.get(m["mongo_id_b"], {"_id": m["mongo_id_b"]})

    # ── Sort by composite score descending ───────────────────────────────────
    matches.sort(key=lambda m: m["report"].scores.composite, reverse=True)

    # ── Write HTML report ────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = build_html_report(
        pairs=matches,
        total_indexed=N,
        total_pairs_checked=checked,
        elapsed=elapsed,
        threshold=threshold,
        generated_at=generated_at,
    )

    Path(out_path).write_text(html, encoding="utf-8")
    log.info(f"Report written to {out_path}")
    log.info(f"Total time: {elapsed:.1f}s | {len(matches)} duplicate pairs found")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch duplicate scan across all indexed SFH thumbnails",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--uri",       required=True,              help="MongoDB connection URI")
    parser.add_argument("--threshold", type=float, default=0.80,   help="Minimum composite score to flag as duplicate")
    parser.add_argument("--top-k",     type=int,   default=20,     help="FAISS neighbors per song before full pipeline")
    parser.add_argument("--out",       default="sfh_duplicates.html", help="Output HTML report path")
    args = parser.parse_args()

    run_batch(
        mongo_uri   = args.uri,
        threshold   = args.threshold,
        faiss_top_k = args.top_k,
        out_path    = args.out,
    )


if __name__ == "__main__":
    main()