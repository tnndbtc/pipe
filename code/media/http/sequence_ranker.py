"""
sequence_ranker.py — Shot Continuity Sequence Ranker
=====================================================

Given a completed batch with scored candidates, computes a recommended_sequence:
a per-item selection that minimises visual discontinuity across the episode.

Algorithm
---------
1. For each item, load .meta.json sidecars for its top-N candidates.
2. Group items by continuity_hints.group_id (items without a group_id are
   treated as singletons and return their top-scored candidate).
3. For each group, run a DP sequence ranker:
   - Pairwise discontinuity S_pair(a, b) = w_color*Δ_color + w_light*Δ_light +
     w_motion*Δ_motion + w_sem*Δ_sem
   - DP selects one candidate per item to minimise total pairwise discontinuity
     along the sequence.
4. Returns recommended_sequence: {item_id: {candidate info dict}} mapping.

Meta sidecars (.meta.json files)
---------------------------------
Written by scorer.py alongside each scored file:
  {
    "hue_hist_16bin": [float * 16],   # normalized hue histogram
    "mean_luma": float,
    "clip_embedding": [float * 512]   # L2-normalized CLIP embedding
  }

Usage
-----
  from sequence_ranker import compute_recommended_sequence
  result = compute_recommended_sequence(items_dict, batch_dir)
  # result: {item_id: candidate_dict} or None if not enough data
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

log = logging.getLogger("sequence_ranker")


def load_meta(path: Path) -> dict | None:
    """Load the .meta.json sidecar for a media file.

    Args:
        path: Absolute path to the media file.

    Returns:
        Parsed metadata dict, or None if not found or invalid JSON.
    """
    meta_path = Path(str(path) + ".meta.json")
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        log.debug("Failed to load meta sidecar %s: %s", meta_path, exc)
        return None


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def discontinuity(meta_a: dict, meta_b: dict) -> float:
    """Compute pairwise discontinuity score between two candidates.

    Components:
    - Δ_color: L1 distance between hue_hist_16bin arrays, normalized to [0,1]
      (divide by 2.0 since L1 of two normalized histograms <= 2).
    - Δ_light: absolute difference between mean_luma values.
    - Δ_motion: 0.0 (placeholder for optical flow).
    - Δ_sem: 1 - cosine_similarity(clip_embedding_a, clip_embedding_b).
      If embeddings not present, use 0.0.

    Returns:
        Weighted discontinuity: 0.35*Δ_color + 0.35*Δ_light + 0.15*Δ_motion + 0.15*Δ_sem
    """
    # Δ_color
    hist_a = meta_a.get("hue_hist_16bin")
    hist_b = meta_b.get("hue_hist_16bin")
    if hist_a and hist_b and len(hist_a) == len(hist_b):
        delta_color = sum(abs(x - y) for x, y in zip(hist_a, hist_b)) / 2.0
    else:
        delta_color = 0.0

    # Δ_light
    luma_a = meta_a.get("mean_luma")
    luma_b = meta_b.get("mean_luma")
    if luma_a is not None and luma_b is not None:
        delta_light = abs(luma_a - luma_b)
    else:
        delta_light = 0.0

    # Δ_motion (placeholder)
    delta_motion = 0.0

    # Δ_sem
    emb_a = meta_a.get("clip_embedding")
    emb_b = meta_b.get("clip_embedding")
    if emb_a and emb_b and len(emb_a) == len(emb_b):
        delta_sem = 1.0 - _cosine_sim(emb_a, emb_b)
    else:
        delta_sem = 0.0

    return 0.35 * delta_color + 0.35 * delta_light + 0.15 * delta_motion + 0.15 * delta_sem


def _dp_rank_group(
    group_items: list[str],
    item_candidates: dict[str, list[tuple[dict, dict | None]]],
) -> dict[str, dict | None]:
    """Select one candidate per item in a group to minimise total discontinuity.

    Args:
        group_items: List of item_ids in sequence order.
        item_candidates: Mapping of item_id to list of (cand_dict, meta_dict_or_None).

    Returns:
        Dict mapping item_id to the selected candidate dict.
    """
    n = len(group_items)
    if n == 0:
        return {}
    if n == 1:
        item_id = group_items[0]
        cands = item_candidates.get(item_id, [])
        return {item_id: cands[0][0] if cands else None}

    # Build candidate lists per item position, preferring those with meta
    cand_lists: list[tuple[str, list[tuple[dict, dict | None]]]] = []
    for item_id in group_items:
        cands = item_candidates.get(item_id, [])
        with_meta = [(c, m) for c, m in cands if m is not None]
        if with_meta:
            cand_lists.append((item_id, with_meta))
        elif cands:
            cand_lists.append((item_id, cands))
        else:
            cand_lists.append((item_id, []))

    INF = float("inf")
    dp: list[list[float]] = []
    choice: list[list[int]] = []

    # Initialize first item
    first_id, first_cands = cand_lists[0]
    dp.append([0.0] * len(first_cands))
    choice.append([-1] * len(first_cands))

    for i in range(1, len(cand_lists)):
        curr_id, curr_cands = cand_lists[i]
        prev_id, prev_cands = cand_lists[i - 1]
        dp_row: list[float] = []
        choice_row: list[int] = []
        for j, (curr_cand, curr_meta) in enumerate(curr_cands):
            best_cost = INF
            best_k = 0
            for k, (prev_cand, prev_meta) in enumerate(prev_cands):
                if curr_meta is not None and prev_meta is not None:
                    disc = discontinuity(prev_meta, curr_meta)
                else:
                    disc = 0.0
                cost = dp[i - 1][k] + disc
                if cost < best_cost:
                    best_cost = cost
                    best_k = k
            dp_row.append(best_cost)
            choice_row.append(best_k)
        dp.append(dp_row)
        choice.append(choice_row)

    # Backtrack from the last item
    last_id, last_cands = cand_lists[-1]
    if not last_cands:
        return {item_id: None for item_id, _ in cand_lists}

    best_last = min(range(len(last_cands)), key=lambda j: dp[-1][j])

    result: dict[str, dict | None] = {}
    idx = best_last
    for i in range(len(cand_lists) - 1, -1, -1):
        item_id, cands = cand_lists[i]
        result[item_id] = cands[idx][0] if cands else None
        if i > 0:
            idx = choice[i][idx]

    return result


def compute_recommended_sequence(
    items: dict,
    batch_dir: Path,
    top_n: int = 5,
) -> dict[str, dict] | None:
    """Compute a recommended sequence for background media candidates.

    For each background item in the batch, selects which candidate (image or
    video) best fits into the overall visual flow of the episode by minimising
    pairwise discontinuity across grouped items.

    Args:
        items: state["items"] from BatchStore — mapping of item_id to item dict.
               Each item may contain "images_ranked", "videos_ranked" (sorted
               by score descending), and "continuity_hints" fields.
        batch_dir: Absolute path to the batch directory. Used to resolve
                   candidate paths for loading .meta.json sidecars.
        top_n: Number of top candidates to consider per item (default 5).

    Returns:
        Dict mapping item_id to the recommended candidate dict (with path, url,
        score, score_detail fields), or None if no candidates were found.
    """
    if not items:
        return None

    # Step 1: collect candidates with meta for each item
    item_candidates: dict[str, list[tuple[dict, dict | None]]] = {}
    for item_id, item in items.items():
        all_cands: list[tuple[dict, dict | None]] = []
        for cand in (item.get("images_ranked") or [])[:top_n]:
            meta = load_meta(batch_dir / cand["path"]) if cand.get("path") else None
            all_cands.append((cand, meta))
        for cand in (item.get("videos_ranked") or [])[:top_n]:
            meta = load_meta(batch_dir / cand["path"]) if cand.get("path") else None
            all_cands.append((cand, meta))
        if all_cands:
            item_candidates[item_id] = all_cands

    if len(item_candidates) < 1:
        return None

    # Step 2: group by group_id from continuity_hints
    singletons: dict[str, str] = {}   # item_id → item_id
    groups: dict[str, list[str]] = {} # group_id → ordered list of item_ids
    for item_id, item in items.items():
        if item_id not in item_candidates:
            continue
        ch = item.get("continuity_hints") or {}
        gid = ch.get("group_id")
        if gid:
            groups.setdefault(gid, []).append(item_id)
        else:
            singletons[item_id] = item_id

    result: dict[str, dict] = {}

    # Step 3: singletons — pick first with meta, else first overall
    for item_id in singletons:
        cands = item_candidates[item_id]
        with_meta = [c for c, m in cands if m is not None]
        result[item_id] = with_meta[0] if with_meta else cands[0][0]

    # Step 4: groups — run DP ranker for each group
    for gid, group_items in groups.items():
        group_result = _dp_rank_group(group_items, item_candidates)
        for item_id, cand in group_result.items():
            if cand is not None:
                result[item_id] = cand

    return result if result else None
