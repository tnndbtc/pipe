#!/usr/bin/env python3
# =============================================================================
# manifest_merge.py — Merge shared + locale AssetManifest drafts
# =============================================================================
#
# Runs AFTER post_tts_analysis.py (vo_items have start_sec/end_sec).
# Produces VOPlan.{locale}.json consumed by the renderer.
#
# What it does:
#   1. Loads the shared manifest  (character_packs, backgrounds, sfx, music)
#   2. Loads the locale manifest  (vo_items with start_sec/end_sec)
#   3. Merges them into a single resolved view
#   4. Computes duck_intervals per shot from VO positions + music fade_sec
#   5. Computes a per-locale timing_lock_hash
#   6. Writes VOPlan.{locale}.json (locale_scope: "merged")
#
# Usage:
#   python manifest_merge.py \
#       --shared  projects/slug/ep/AssetManifest.shared.json \
#       --locale  projects/slug/ep/VOPlan.zh-Hans.json
#
#   python manifest_merge.py --shared ... --locale ... --out /custom/out.json
#
# Requirements: stdlib only (json, hashlib, pathlib)
# =============================================================================

import argparse
import hashlib
import json
import sys
from pathlib import Path

PIPE_DIR = Path(__file__).resolve().parent.parent.parent

# Import sentinel verification from vo_utils (same directory)
try:
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.dirname(__file__))
    from vo_utils import verify_sentinel, get_primary_locale
    _VO_UTILS_AVAILABLE = True
except ImportError:
    _VO_UTILS_AVAILABLE = False


# ── Manifest helpers ──────────────────────────────────────────────────────────

def load_manifest(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ── Duck interval computation ─────────────────────────────────────────────────

def merge_overlapping(intervals: list[tuple[float, float]]) -> list[list[float]]:
    """Merge a list of (t0, t1) intervals into non-overlapping sorted ranges."""
    if not intervals:
        return []
    sorted_ivs = sorted(intervals, key=lambda x: x[0])
    merged = [list(sorted_ivs[0])]
    for t0, t1 in sorted_ivs[1:]:
        if t0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], t1)
        else:
            merged.append([t0, t1])
    return [[round(a, 3), round(b, 3)] for a, b in merged]


def compute_duck_intervals(
    vo_items_for_shot: list[dict],
    fade_sec: float,
) -> list[list[float]]:
    """
    Compute music duck intervals from VO line positions.

    For each VO line with start_sec + end_sec, the music is ducked from
    (start_sec - fade_sec) to (end_sec + fade_sec), clamped to >= 0.
    Overlapping intervals are merged.
    """
    raw = []
    for item in vo_items_for_shot:
        start = item.get("start_sec")
        end   = item.get("end_sec")
        if start is None or end is None:
            continue
        t0 = max(0.0, start - fade_sec)
        t1 = end + fade_sec
        raw.append((t0, t1))
    return merge_overlapping(raw)


# ── ShotList helpers ──────────────────────────────────────────────────────────

def load_shotlist(manifest: dict, manifest_path: Path) -> list[dict]:
    """
    Load the ShotList referenced by shotlist_ref.
    Uses the same three-candidate resolution as post_tts_analysis.py.
    Returns [] if not found (non-fatal — duck_intervals will be empty).
    """
    shotlist_ref = manifest.get("shotlist_ref", "")
    if not shotlist_ref:
        return []

    candidates = [
        manifest_path.parent / shotlist_ref,
        manifest_path.parent.parent / shotlist_ref,
        manifest_path.parent / "ShotList.json",  # conventional fallback
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                with open(candidate, encoding="utf-8") as f:
                    return json.load(f).get("shots", [])
            except Exception as exc:
                print(f"[WARN] Could not parse ShotList {candidate}: {exc}")
                return []

    print(f"[WARN] ShotList not found: {shotlist_ref} — duck_intervals will be empty")
    return []


def build_vo_shot_map(shots: list[dict]) -> dict[str, str]:
    """
    Build reverse mapping {vo_item_id → shot_id} from ShotList shots.
    Reads shots[].audio_intent.vo_item_ids for each shot.
    """
    mapping: dict[str, str] = {}
    for shot in shots:
        shot_id = shot.get("shot_id", "")
        if not shot_id:
            continue
        vo_ids = shot.get("audio_intent", {}).get("vo_item_ids", [])
        for vid in vo_ids:
            mapping[vid] = shot_id
    return mapping


# ── timing_lock_hash computation ──────────────────────────────────────────────

def compute_timing_lock_hash(shots: list[dict]) -> str:
    """
    SHA-256 of the canonical JSON array of [shot_id, duration_ms] pairs
    sorted by shot_id, using locale-adjusted duration_ms values.

    duration_ms = round(duration_sec * 1000)

    This is the same algorithm used by the ShotList timing_lock_hash;
    the only difference is the input durations (locale-adjusted vs canonical).
    """
    pairs = sorted(
        [[s["shot_id"], round(s["duration_sec"] * 1000)]
         for s in shots
         if "shot_id" in s and "duration_sec" in s],
        key=lambda x: x[0],
    )
    canonical = json.dumps(pairs, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── Merge logic ───────────────────────────────────────────────────────────────

def merge_manifests(
    shared: dict,
    locale: dict,
    vo_shot_map: dict[str, str] | None = None,
) -> dict:
    """
    Merge shared + locale manifests into a single resolved view.

    - Shared contributes: character_packs, backgrounds
    - Locale contributes: vo_items (with start_sec/end_sec), background_overrides
    - timing_lock_hash computed from locale-adjusted shot durations
    - locale_scope set to "merged"

    vo_shot_map: reverse mapping {vo_item_id → shot_id} built from the ShotList.
                 When provided, overrides the (usually absent) shot_id field on
                 vo_items so that duck_intervals are computed per shot correctly.
    """
    merged = {
        "schema_id":      "VOPlan",
        "schema_version": "1.0.0",
        "manifest_id":    locale.get("manifest_id", shared.get("manifest_id", "")),
        "project_id":     shared.get("project_id", ""),
        "episode_id":     shared.get("episode_id", ""),
        "locale_scope":   "merged",
        "locale":         locale.get("locale", ""),
        "shotlist_ref":   shared.get("shotlist_ref", locale.get("shotlist_ref", "")),
        # Shared assets
        "character_packs":     shared.get("character_packs", []),
        "backgrounds":         shared.get("backgrounds", []),
        # Locale VO
        "vo_items":            locale.get("vo_items", []),
        # Background overrides from locale (post_tts_analysis.py populated these)
        "background_overrides": locale.get("background_overrides", []),
        # ── Sections owned by downstream scripts ──────────────────────────
        # Populated by resolve_assets.py. Empty [] until that script runs.
        "resolved_assets": [],
        # Populated by gen_render_plan.py. None until that script runs.
        "render_plan":     None,
    }

    # Preserve vo_approval block if it exists in the locale manifest.
    # This block is written by write_sentinel() when the user approves VO in the
    # VO tab and must survive manifest_merge in-place re-runs intact.
    if locale.get("vo_approval"):
        merged["vo_approval"] = locale["vo_approval"]

    # Compute per-locale timing_lock_hash
    # Use background_overrides to get locale-adjusted durations.
    shot_durations: dict[str, float] = {}
    for override in merged["background_overrides"]:
        sid = override.get("shot_id")
        dur = override.get("duration_sec")
        if sid and dur is not None:
            shot_durations[sid] = float(dur)

    shots_for_hash = [
        {"shot_id": sid, "duration_sec": dur}
        for sid, dur in sorted(shot_durations.items())
    ]
    merged["timing_lock_hash"] = compute_timing_lock_hash(shots_for_hash)

    return merged


# ── Output path derivation ────────────────────────────────────────────────────

def derive_output_path(locale_manifest_path: Path, locale: str) -> Path:
    """
    Default output: same directory as the locale manifest,
    named VOPlan.{locale}.json.
    """
    return locale_manifest_path.parent / f"VOPlan.{locale}.json"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Merge shared + locale AssetManifest drafts into a single "
                    "resolved manifest with computed duck_intervals and timing_lock_hash.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--shared", required=True, metavar="PATH",
                   help="Path to the shared AssetManifest draft "
                        "(locale_scope='shared').")
    p.add_argument("--locale", required=True, metavar="PATH",
                   help="Path to the locale AssetManifest draft "
                        "(locale_scope='locale'). Must have vo_items with "
                        "start_sec/end_sec populated by post_tts_analysis.py.")
    p.add_argument("--out", default=None, metavar="PATH",
                   help="Output path for merged manifest. "
                        "Default: VOPlan.{locale}.json "
                        "in the same directory as the locale manifest.")
    p.add_argument("--primary", default=None, metavar="PATH",
                   help="Path to the primary-locale VOPlan (e.g. VOPlan.en.json). "
                        "When provided and the locale being merged is NOT the primary, "
                        "scene_heads and scene_tails are copied from this file so "
                        "translated locales inherit the same structural pauses.")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    shared_path = Path(args.shared).resolve()
    locale_path = Path(args.locale).resolve()

    for label, path in [("shared", shared_path), ("locale", locale_path)]:
        if not path.exists():
            print(f"[ERROR] {label} manifest not found: {path}", file=sys.stderr)
            sys.exit(1)

    shared = load_manifest(shared_path)
    locale = load_manifest(locale_path)

    # Guards
    shared_scope = shared.get("locale_scope")
    if shared_scope not in ("shared", None):
        raise SystemExit(
            f"[ERROR] --shared manifest has locale_scope='{shared_scope}'. "
            "Expected 'shared'."
        )

    locale_scope = locale.get("locale_scope")
    if locale_scope not in ("locale", "monolithic", "merged", None):
        raise SystemExit(
            f"[ERROR] --locale manifest has locale_scope='{locale_scope}'. "
            "Expected 'locale', 'monolithic', or 'merged'."
        )

    locale_tag = locale.get("locale", "")
    if not locale_tag:
        raise SystemExit(
            "[ERROR] locale manifest is missing 'locale' field. "
            "Cannot derive output filename."
        )

    # Derive output path
    if args.out:
        out_path = Path(args.out).resolve()
    else:
        out_path = derive_output_path(locale_path, locale_tag)

    print("=" * 60)
    print("  manifest_merge")
    print(f"  Shared   : {shared_path.name}")
    print(f"  Locale   : {locale_path.name}  ({locale_tag})")
    print(f"  Output   : {out_path}")
    if args.primary:
        print(f"  Primary  : {Path(args.primary).name}")
    print("=" * 60)

    # Check vo_items have start_sec/end_sec
    vo_items    = locale.get("vo_items", [])
    missing_timing = [
        v["item_id"] for v in vo_items
        if "start_sec" not in v or "end_sec" not in v
    ]
    if missing_timing:
        print(f"[WARN] {len(missing_timing)} vo_items missing start_sec/end_sec — "
              f"run post_tts_analysis.py first.")
        print(f"       First missing: {missing_timing[:3]}")

    # Load ShotList and build VO → shot reverse mapping for duck_intervals.
    # Try shared path first, then locale path (shared manifest has shotlist_ref).
    shots = load_shotlist(shared, shared_path)
    if not shots:
        shots = load_shotlist(locale, locale_path)
    vo_shot_map = build_vo_shot_map(shots)
    print(f"  ShotList shots    : {len(shots)}  "
          f"(vo→shot mappings: {len(vo_shot_map)})")

    # Merge
    merged = merge_manifests(shared, locale, vo_shot_map=vo_shot_map)

    # Inherit scene_heads / scene_tails from primary locale VOPlan when:
    #   --primary is given  AND  this locale is NOT the primary locale
    # (If this IS the primary locale, its own scene_heads are already in the
    #  locale manifest and must not be overwritten.)
    if args.primary:
        primary_path = Path(args.primary).resolve()
        primary_locale_tag = primary_path.stem.split(".", 1)[-1]  # VOPlan.<tag>.json
        if locale_tag != primary_locale_tag:
            if primary_path.exists():
                try:
                    primary_plan = load_manifest(primary_path)
                    for field in ("scene_heads", "scene_tails"):
                        val = primary_plan.get(field)
                        if val:
                            merged[field] = val
                            print(f"  [{field}] copied from primary ({primary_locale_tag}): {val}")
                        else:
                            print(f"  [{field}] not present in primary — skipped")
                except Exception as exc:
                    print(f"[WARN] Could not read primary VOPlan {primary_path}: {exc}")
            else:
                print(f"[WARN] --primary path not found: {primary_path} — scene_heads/scene_tails not inherited")
        else:
            print(f"  [primary] locale_tag matches primary ({primary_locale_tag}) — scene_heads/scene_tails not overwritten")

    # Print stats
    print(f"\n  character_packs   : {len(merged.get('character_packs', []))}")
    print(f"  backgrounds       : {len(merged.get('backgrounds', []))}")
    print(f"  vo_items          : {len(merged.get('vo_items', []))}")
    print(f"  background_overrides: {len(merged.get('background_overrides', []))}")
    print(f"  timing_lock_hash  : {merged.get('timing_lock_hash', '')[:16]}…")

    # Write
    save_manifest(merged, out_path)
    print(f"\n  [OK] Written: {out_path}")


if __name__ == "__main__":
    main()
