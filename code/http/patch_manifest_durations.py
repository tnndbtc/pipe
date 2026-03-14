#!/usr/bin/env python3
# =============================================================================
# patch_manifest_durations.py — Patch AssetManifest_draft.shared.json duration
#                               fields from authoritative ShotList.json
# =============================================================================
#
# Runs AFTER Stage 5 (AssetManifest_draft produced) and BEFORE Stage 6.
# Called by run.sh immediately after the Stage 5 LLM completes.
#
# Problem it solves:
#   Stage 5 is an LLM.  It must copy duration_sec values from the patched
#   ShotList.json into the manifest, but LLMs are probabilistic and can
#   produce wrong values (rounding errors, swapped shots, hallucinated numbers).
#   This script enforces the correct values deterministically, mirroring the
#   pattern of patch_shotlist_durations.py (which enforces ShotList.duration_sec
#   after Stage 4).
#
# Reads:
#   ShotList.json                        — authoritative duration_sec per shot_id
#   AssetManifest_draft.shared.json      — shared locale-free assets
#
# Writes (in-place, atomic):
#   AssetManifest_draft.shared.json      — three field groups patched:
#
#     sfx_items[].duration_sec           ← ShotList[shot_id].duration_sec
#       Clip length passed to SFX generation model (AudioGen). If wrong, the
#       SFX clip cuts off early or plays into dead silence during the shot.
#
#     music_items[].duration_sec         ← ShotList[shot_id].duration_sec
#       Track length passed to music generation model (MusicGen). If wrong,
#       music clips wrong length → audible looping or abrupt cut.
#
#     backgrounds[].search_filters.min_duration_sec
#       Only patched for backgrounds that have media_type="video" in their
#       search_filters (i.e. they require a video clip, not a photo).
#       Value = max(duration_sec for all shots using this background_id).
#       max_duration_sec is also updated to min_duration_sec + 10.
#       If wrong, the media search returns clips shorter than the shot →
#       background video freezes or loops mid-shot in the final render.
#
# Shots with duration_sec = 0:
#   Skipped. These are shots where patch_shotlist_durations.py found no approved
#   VO timing (e.g. no-VO action beats still have their Stage 4 LLM estimate).
#   The LLM estimate is preserved for sfx/music/backgrounds in those shots.
#   (patch_shotlist_durations.py only writes 0 when the approved file is absent
#    entirely; in normal operation all VO shots get their correct duration.)
#
# =============================================================================

import argparse
import json
import sys
from pathlib import Path

PIPE_DIR = Path(__file__).resolve().parent.parent.parent


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    """Atomic write via temp file → rename."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


# ── Core patcher ──────────────────────────────────────────────────────────────

def patch(
    shared: dict,
    shotlist: dict,
) -> tuple[int, int, int, list[str]]:
    """
    Patch AssetManifest_draft.shared.json duration fields in-place.

    Returns:
        (sfx_patched, music_patched, bg_patched, warnings)
    """
    warnings: list[str] = []

    # Build shot_id → duration_sec map from ShotList
    shot_dur: dict[str, float] = {}
    for shot in shotlist.get("shots", []):
        sid = shot.get("shot_id", "")
        dur = shot.get("duration_sec", 0)
        if sid:
            shot_dur[sid] = float(dur)

    # Build background_id → max(duration_sec) across all shots using it
    bg_dur: dict[str, float] = {}
    for shot in shotlist.get("shots", []):
        bg_id = shot.get("background_id", "")
        dur   = float(shot.get("duration_sec", 0))
        if bg_id and dur > 0:
            bg_dur[bg_id] = max(bg_dur.get(bg_id, 0.0), dur)

    # ── sfx_items ─────────────────────────────────────────────────────────────
    sfx_patched = 0
    for sfx in shared.get("sfx_items", []):
        item_id = sfx.get("item_id", "?")
        shot_id = sfx.get("shot_id", "")
        if not shot_id:
            warnings.append(f"sfx {item_id}: missing shot_id — skipped")
            continue
        if shot_id not in shot_dur:
            warnings.append(f"sfx {item_id}: shot_id {shot_id!r} not found in ShotList — skipped")
            continue
        new_dur = round(shot_dur[shot_id], 3)
        old_dur = sfx.get("duration_sec", 0)
        if new_dur == 0:
            print(f"  [SKIP-SFX] {item_id:30s}  shot={shot_id}  duration=0 (no approved timing)")
            continue
        sfx["duration_sec"] = new_dur
        sfx_patched += 1
        change = "PATCH" if abs(new_dur - old_dur) > 0.01 else "OK   "
        print(f"  [{change}-SFX  ] {item_id:30s}  shot={shot_id:20s}  "
              f"old={old_dur:7.3f}s → new={new_dur:7.3f}s")

    # ── music_items ───────────────────────────────────────────────────────────
    music_patched = 0
    for music in shared.get("music_items", []):
        item_id = music.get("item_id", "?")
        shot_id = music.get("shot_id", "")
        if not shot_id:
            warnings.append(f"music {item_id}: missing shot_id — skipped")
            continue
        if shot_id not in shot_dur:
            warnings.append(f"music {item_id}: shot_id {shot_id!r} not found in ShotList — skipped")
            continue
        new_dur = round(shot_dur[shot_id], 3)
        old_dur = music.get("duration_sec", 0)
        if new_dur == 0:
            print(f"  [SKIP-MUSIC] {item_id:30s}  shot={shot_id}  duration=0 (no approved timing)")
            continue
        music["duration_sec"] = new_dur
        music_patched += 1
        change = "PATCH" if abs(new_dur - old_dur) > 0.01 else "OK   "
        print(f"  [{change}-MUSIC] {item_id:30s}  shot={shot_id:20s}  "
              f"old={old_dur:7.3f}s → new={new_dur:7.3f}s")

    # ── backgrounds — search_filters.min_duration_sec (video only) ────────────
    bg_patched = 0
    for bg in shared.get("backgrounds", []):
        asset_id = bg.get("asset_id", "?")
        sf = bg.get("search_filters", {})
        if sf.get("media_type") != "video":
            continue                # photo backgrounds have no duration constraint
        if asset_id not in bg_dur:
            warnings.append(f"bg {asset_id}: not found in ShotList backgrounds — skipped")
            continue
        new_min = round(bg_dur[asset_id], 3)
        if new_min == 0:
            print(f"  [SKIP-BG ] {asset_id:30s}  duration=0 (no approved timing)")
            continue
        new_max = round(new_min + 10.0, 3)
        old_min = sf.get("min_duration_sec", 0)
        sf["min_duration_sec"] = new_min
        sf["max_duration_sec"] = new_max
        bg_patched += 1
        change = "PATCH" if abs(new_min - old_min) > 0.01 else "OK   "
        print(f"  [{change}-BG  ] {asset_id:30s}  "
              f"min_dur old={old_min:7.3f}s → new={new_min:7.3f}s  (max={new_max:.3f}s)")

    return sfx_patched, music_patched, bg_patched, warnings


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Patch AssetManifest_draft.shared.json duration fields from ShotList.\n"
            "Runs after Stage 5 LLM, before Stage 6."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "ep_dir",
        help="Episode directory (e.g. projects/slug/episodes/s01e01)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Compute and print patches without writing AssetManifest_draft.shared.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ep_dir = Path(args.ep_dir)
    if not ep_dir.is_dir():
        ep_dir = PIPE_DIR / args.ep_dir
    if not ep_dir.is_dir():
        print(f"[ERROR] ep_dir not found: {args.ep_dir}", file=sys.stderr)
        sys.exit(1)
    ep_dir = ep_dir.resolve()

    # ── Load ShotList ─────────────────────────────────────────────────────────
    shotlist_path = ep_dir / "ShotList.json"
    if not shotlist_path.exists():
        print(
            f"[ERROR] ShotList.json not found in {ep_dir}\n"
            "       Run Stage 4 (and patch_shotlist_durations.py) before calling this script.",
            file=sys.stderr,
        )
        sys.exit(1)
    shotlist = load_json(shotlist_path)

    # ── Load shared manifest ──────────────────────────────────────────────────
    shared_path = ep_dir / "AssetManifest_draft.shared.json"
    if not shared_path.exists():
        print(
            f"[ERROR] AssetManifest_draft.shared.json not found in {ep_dir}\n"
            "       Run Stage 5 before calling this script.",
            file=sys.stderr,
        )
        sys.exit(1)
    shared = load_json(shared_path)

    shot_count  = len(shotlist.get("shots", []))
    sfx_count   = len(shared.get("sfx_items", []))
    music_count = len(shared.get("music_items", []))
    bg_count    = sum(
        1 for bg in shared.get("backgrounds", [])
        if bg.get("search_filters", {}).get("media_type") == "video"
    )

    print("=" * 60)
    print("  patch_manifest_durations")
    print(f"  Episode : {ep_dir}")
    print(f"  Shots   : {shot_count}")
    print(f"  SFX     : {sfx_count} items")
    print(f"  Music   : {music_count} items")
    print(f"  BG-video: {bg_count} backgrounds with media_type=video")
    print("=" * 60)

    sfx_p, music_p, bg_p, warnings = patch(shared, shotlist)

    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"  SFX patched   : {sfx_p}")
    print(f"  Music patched : {music_p}")
    print(f"  BG patched    : {bg_p}")
    if warnings:
        print(f"  Warnings      : {len(warnings)}")
        for w in warnings:
            print(f"    ⚠  {w}")

    if args.dry_run:
        print("\n  [DRY-RUN] AssetManifest_draft.shared.json NOT written.")
    else:
        save_json(shared, shared_path)
        print(f"\n  ✓ AssetManifest_draft.shared.json updated: {shared_path}")
    print("=" * 60)

    if warnings:
        sys.exit(1)


if __name__ == "__main__":
    main()
