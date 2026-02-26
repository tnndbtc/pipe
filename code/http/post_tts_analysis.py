#!/usr/bin/env python3
# =============================================================================
# post_tts_analysis.py — Post-TTS VO timeline analysis
# =============================================================================
#
# Runs AFTER gen_tts.py (VO WAVs exist on disk).
# Runs BEFORE manifest_merge.py.
#
# For each VO item in a locale AssetManifest draft:
#   1. Measures the actual WAV duration on disk
#   2. Computes start_sec / end_sec for every VO line within its shot
#      (sequential placement: start[n+1] = end[n] + inter_line_pause)
#   3. Detects shot overflow (total VO > shot duration_sec)
#   4. Writes start_sec, end_sec into vo_items[] in the locale manifest in place
#   5. Writes background_overrides[] for any overflowing shots
#
# ⚠️  Mutates the locale manifest in place.
#     Re-run after every Stage 5 (LLM) regeneration.
#
# Usage:
#   python post_tts_analysis.py \
#       --manifest projects/slug/episodes/ep/AssetManifest_draft.zh-Hans.json
#
#   python post_tts_analysis.py --manifest ... --pause 0.2   # tighter pacing
#   python post_tts_analysis.py --manifest ... --buffer 1.0  # overflow padding
#   python post_tts_analysis.py --manifest ... --vo-dir /custom/vo/path/
#
# Requirements:
#   pip install soundfile   (already required by gen_music_clip.py)
# =============================================================================

import argparse
import json
import sys
from pathlib import Path

PIPE_DIR = Path(__file__).resolve().parent.parent.parent

DEFAULT_PAUSE_SEC  = 0.3   # inter-line pause between consecutive VO lines
DEFAULT_BUFFER_SEC = 0.5   # extra padding added to overflow background duration


# ── Audio helpers ─────────────────────────────────────────────────────────────

def wav_duration(path: Path) -> float:
    """Return the duration in seconds of a WAV file."""
    import soundfile as sf
    info = sf.info(str(path))
    return info.duration


# ── Manifest helpers ──────────────────────────────────────────────────────────

def load_manifest(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")


def derive_vo_dir(manifest: dict, manifest_path: Path) -> Path:
    """
    Derive the VO WAV directory from manifest fields.
    Layout: projects/{project_id}/episodes/{episode_id}/assets/{locale}/audio/vo/
    Falls back to a sibling assets/ directory if fields are missing.
    """
    project_id = manifest.get("project_id", "")
    episode_id = manifest.get("episode_id", "")
    locale     = manifest.get("locale", "en")

    if project_id and episode_id:
        return (PIPE_DIR / "projects" / project_id
                / "episodes" / episode_id / "assets" / locale / "audio" / "vo")

    # Fallback: assets/{locale}/audio/vo/ relative to manifest
    return manifest_path.parent / "assets" / locale / "audio" / "vo"


# ── Core analysis ─────────────────────────────────────────────────────────────

def analyse(
    manifest: dict,
    vo_dir: Path,
    pause_sec: float,
    buffer_sec: float,
) -> tuple[dict, list[str], list[str]]:
    """
    Compute VO timelines and detect overflows.

    Returns:
        updated manifest dict (vo_items mutated in place with start_sec/end_sec;
                               background_overrides[] populated for overflowing shots)
        warnings  list of warning strings (non-fatal)
        errors    list of error strings (missing WAV files)
    """
    warnings = []
    errors   = []

    # Index vo_items by item_id for fast lookup
    vo_index: dict[str, dict] = {
        v["item_id"]: v for v in manifest.get("vo_items", [])
    }

    # Index shot duration_sec from shotlist_ref if available in manifest.
    # The locale manifest itself doesn't carry shot durations, so we read
    # them from the ShotList referenced by shotlist_ref (best effort).
    shot_durations: dict[str, float] = {}
    shotlist_ref = manifest.get("shotlist_ref", "")
    if shotlist_ref:
        # Resolve relative to manifest location
        # (will be set by caller via manifest_path)
        pass  # populated below in main after we have manifest_path

    background_overrides: list[dict] = list(manifest.get("background_overrides", []))
    existing_override_shots = {o["shot_id"] for o in background_overrides}

    # Group vo_items by shot_id (preserving order)
    shots_to_items: dict[str, list[dict]] = {}
    for item in manifest.get("vo_items", []):
        sid = item.get("shot_id", "__no_shot__")
        shots_to_items.setdefault(sid, []).append(item)

    for shot_id, items in shots_to_items.items():
        cursor = 0.0  # running timeline position within the shot

        for item in items:
            item_id  = item["item_id"]
            wav_path = vo_dir / f"{item_id}.wav"

            if not wav_path.exists():
                errors.append(f"WAV not found: {wav_path}")
                # Assign placeholder timings so downstream isn't blocked
                item["start_sec"] = round(cursor, 3)
                item["end_sec"]   = round(cursor, 3)
                continue

            duration = wav_duration(wav_path)
            item["start_sec"] = round(cursor, 3)
            item["end_sec"]   = round(cursor + duration, 3)
            cursor = cursor + duration + pause_sec

        # Total VO span for this shot (cursor is now past the last pause)
        total_vo_sec = cursor - pause_sec if items else 0.0

        # Overflow detection
        shot_dur = shot_durations.get(shot_id)
        if shot_dur is not None and total_vo_sec > shot_dur:
            override_dur = round(total_vo_sec + buffer_sec, 3)
            warnings.append(
                f"[OVERFLOW] {shot_id}: VO total {total_vo_sec:.2f}s "
                f"> shot {shot_dur:.2f}s — override duration {override_dur:.2f}s"
            )
            if shot_id not in existing_override_shots:
                # Derive item_id from shot_id + locale
                locale  = manifest.get("locale", "xx")
                item_id = f"bg-{shot_id}-{locale}"
                uri     = (f"assets/{locale}/backgrounds/overrides/"
                           f"{item_id}.mp4")
                background_overrides.append({
                    "item_id":      item_id,
                    "shot_id":      shot_id,
                    "duration_sec": override_dur,
                    "uri":          uri,
                })
                existing_override_shots.add(shot_id)
            else:
                # Update existing override duration
                for o in background_overrides:
                    if o["shot_id"] == shot_id:
                        o["duration_sec"] = round(total_vo_sec + buffer_sec, 3)

    manifest["background_overrides"] = background_overrides
    return manifest, warnings, errors


# ── ShotList reader ───────────────────────────────────────────────────────────

def load_shot_durations(manifest: dict, manifest_path: Path) -> dict[str, float]:
    """
    Read shot duration_sec values from the referenced ShotList.
    Returns {} if the ShotList cannot be found (non-fatal).
    """
    shotlist_ref = manifest.get("shotlist_ref", "")
    if not shotlist_ref:
        return {}

    # shotlist_ref may be a filename ("ShotList.json") or a logical ID
    # ("the-pharaoh-who-defied-death-s01e02-shotlist"). Always also try
    # the conventional filename in the same directory as the manifest.
    candidates = [
        manifest_path.parent / shotlist_ref,
        manifest_path.parent.parent / shotlist_ref,
        manifest_path.parent / "ShotList.json",       # conventional fallback
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                with open(candidate, encoding="utf-8") as f:
                    shotlist = json.load(f)
                return {
                    s["shot_id"]: float(s["duration_sec"])
                    for s in shotlist.get("shots", [])
                    if "shot_id" in s and "duration_sec" in s
                }
            except Exception as exc:
                print(f"[WARN] Could not parse ShotList {candidate}: {exc}")
                return {}
    print(f"[WARN] ShotList not found: {shotlist_ref} — overflow detection disabled")
    return {}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Post-TTS VO timeline analysis — measures WAVs, computes "
                    "start_sec/end_sec, detects shot overflows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", required=True, metavar="PATH",
                   help="Path to a locale AssetManifest draft JSON "
                        "(locale_scope must be 'locale' or 'monolithic').")
    p.add_argument("--vo-dir", default=None, metavar="DIR",
                   help="Directory containing VO WAV files. "
                        "Default: derived from manifest project_id/episode_id/locale.")
    p.add_argument("--pause", type=float, default=DEFAULT_PAUSE_SEC, metavar="SEC",
                   help=f"Inter-line pause between consecutive VO lines "
                        f"(default: {DEFAULT_PAUSE_SEC}s).")
    p.add_argument("--buffer", type=float, default=DEFAULT_BUFFER_SEC, metavar="SEC",
                   help=f"Extra duration padding added to overflow background clips "
                        f"(default: {DEFAULT_BUFFER_SEC}s).")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    manifest = load_manifest(manifest_path)

    # Guard: reject shared manifests
    locale_scope = manifest.get("locale_scope")
    if locale_scope == "shared":
        raise SystemExit(
            "[ERROR] post_tts_analysis.py requires a locale manifest "
            "(locale_scope='locale'). Shared manifests have no vo_items."
        )

    # Resolve VO directory
    if args.vo_dir:
        vo_dir = Path(args.vo_dir).resolve()
    else:
        vo_dir = derive_vo_dir(manifest, manifest_path)

    if not vo_dir.exists():
        print(f"[WARN] VO directory not found: {vo_dir}", file=sys.stderr)
        print("[WARN] No WAV files will be found — timings will be zero.", file=sys.stderr)

    # Load shot durations for overflow detection
    shot_durations = load_shot_durations(manifest, manifest_path)

    vo_count = len(manifest.get("vo_items", []))
    print("=" * 60)
    print("  post_tts_analysis")
    print(f"  Manifest  : {manifest_path.name}")
    print(f"  Locale    : {manifest.get('locale', 'unknown')}")
    print(f"  VO dir    : {vo_dir}")
    print(f"  VO items  : {vo_count}")
    print(f"  Pause     : {args.pause}s  |  Buffer: {args.buffer}s")
    print(f"  Shots w/ duration: {len(shot_durations)}")
    print("=" * 60)

    if vo_count == 0:
        print("[INFO] No vo_items in manifest — nothing to do.")
        return

    # Inject shot_durations into analyse() via closure trick
    # (analyse uses shot_durations dict directly)
    # We patch it in after load
    shots_to_items: dict[str, list] = {}
    for item in manifest.get("vo_items", []):
        sid = item.get("shot_id", "__no_shot__")
        shots_to_items.setdefault(sid, []).append(item)

    # Run analysis
    background_overrides: list[dict] = list(manifest.get("background_overrides", []))
    existing_override_shots = {o["shot_id"] for o in background_overrides}
    warnings = []
    errors   = []

    total_processed = 0

    for shot_id, items in shots_to_items.items():
        cursor = 0.0

        for item in items:
            item_id  = item["item_id"]
            wav_path = vo_dir / f"{item_id}.wav"

            if not wav_path.exists():
                msg = f"WAV not found: {wav_path}"
                errors.append(msg)
                print(f"  [MISSING] {item_id}")
                item["start_sec"] = round(cursor, 3)
                item["end_sec"]   = round(cursor, 3)
                continue

            duration = wav_duration(wav_path)
            item["start_sec"] = round(cursor, 3)
            item["end_sec"]   = round(cursor + duration, 3)
            print(f"  [OK] {item_id}  {item['start_sec']}s – {item['end_sec']}s  "
                  f"(dur {duration:.3f}s)")
            cursor += duration + args.pause
            total_processed += 1

        total_vo_sec = cursor - args.pause if items else 0.0
        shot_dur     = shot_durations.get(shot_id)

        if shot_dur is not None and total_vo_sec > shot_dur:
            override_dur = round(total_vo_sec + args.buffer, 3)
            msg = (f"[OVERFLOW] {shot_id}: VO {total_vo_sec:.2f}s "
                   f"> shot {shot_dur:.2f}s → override {override_dur:.2f}s")
            warnings.append(msg)
            print(f"  {msg}")

            if shot_id not in existing_override_shots:
                locale  = manifest.get("locale", "xx")
                bg_id   = f"bg-{shot_id}-{locale}"
                uri     = (f"assets/{locale}/backgrounds/overrides/{bg_id}.mp4")
                background_overrides.append({
                    "item_id":      bg_id,
                    "shot_id":      shot_id,
                    "duration_sec": override_dur,
                    "uri":          uri,
                })
                existing_override_shots.add(shot_id)
            else:
                for o in background_overrides:
                    if o["shot_id"] == shot_id:
                        o["duration_sec"] = round(total_vo_sec + args.buffer, 3)

    manifest["background_overrides"] = background_overrides

    # Write manifest back in place
    save_manifest(manifest, manifest_path)

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY — post_tts_analysis")
    print(f"{'='*60}")
    print(f"  VO items processed : {total_processed}/{vo_count}")
    print(f"  Missing WAVs       : {len(errors)}")
    print(f"  Overflow shots     : {len(warnings)}")
    print(f"  Background overrides: {len(background_overrides)}")
    print(f"  Manifest updated   : {manifest_path}")

    if errors:
        print("\n  [ERRORS]")
        for e in errors:
            print(f"    {e}")

    if warnings:
        print("\n  [WARNINGS]")
        for w in warnings:
            print(f"    {w}")


if __name__ == "__main__":
    main()
