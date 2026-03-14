#!/usr/bin/env python3
# =============================================================================
# patch_shotlist_durations.py — Patch ShotList.json duration_sec from approved VO
# =============================================================================
#
# Runs AFTER Stage 4 (ShotList.json produced) and BEFORE Stage 5.
# Called by run.sh immediately after the Stage 4 LLM completes.
#
# Reads:
#   vo_preview_approved.{locale}.json  — approved item timing (end_sec per item)
#   AssetManifest_draft.{locale}.json  — pause_after_ms per item  (optional)
#   ShotList.json                      — shot groupings (audio_intent.vo_item_ids)
#
# Writes:
#   ShotList.json                      — duration_sec, start_sec, end_sec patched
#                                        in-place (atomic write via temp+rename)
#
# Duration formula per shot:
#   shot_vo_span + pause_sec + tail_sec
#
# Where:
#   shot_vo_span           — last_item.end_sec - first_item.start_sec
#                            The approved file stores EPISODE-WIDE cumulative
#                            timestamps (post_tts_analysis runs at Stage 3.5
#                            before ShotList exists; all items share one cursor
#                            under a single "__no_shot__" group).  Subtracting
#                            the first item's start_sec gives the true shot span.
#                            NOTE: voApproveTTS writes end_sec = start + dur
#                            (pause NOT included), so the pause must be added
#                            explicitly here.
#   pause_sec              — last_item.pause_after_ms / 1000 (from manifest).
#                            Default 300 ms; user may raise it to 2000–5000 ms
#                            for an inter-scene or inter-act break.
#                            voApproveTTS embeds this pause in the gap between
#                            last_item.end_sec and next_shot.first_item.start_sec,
#                            so gap = pause_sec + approved_tail_sec.
#   tail_sec               — max(gap - pause_sec, SCENE_TAIL_SEC)
#                            Strips the pause out of the raw gap to isolate the
#                            approved tail, then enforces the 2.0 s minimum.
#                            For the last shot (no next-shot look-ahead):
#                            tail_sec = SCENE_TAIL_SEC always.
#   SCENE_TAIL_SEC = 2.0   — minimum visual tail; matches gen_render_plan.py
#                            VO_TAIL_MS = 2000.
#
# Why separate pause and tail rather than max(pause, SCENE_TAIL_SEC)?
#   voApproveTTS sets next_shot.start_sec = last.end_sec + pause + tail.
#   The VO Timeline display (_voRecalcSceneTimes) shows scene boundaries with
#   the pause included, so its "sc02 starts at 26.1 s" means sc01 lasts 26.1 s.
#   The old max() formula swallowed the 300 ms pause whenever the approved tail
#   was small (e.g. gap = 0.3 + 1.0 = 1.3 → max(1.3, 2.0) = 2.0, losing 0.3 s).
#   The new formula always preserves the pause: 23.806 + 0.300 + 2.000 = 26.106.
#
# Shots with empty vo_item_ids are skipped (their duration_sec is unchanged).
# This handles no-VO shots (action beats, pure music, etc.).
#
# =============================================================================

import argparse
import json
import sys
from pathlib import Path

PIPE_DIR = Path(__file__).resolve().parent.parent.parent

SCENE_TAIL_SEC   = 2.0   # matches gen_render_plan.py VO_TAIL_MS = 2000
DEFAULT_PAUSE_MS = 300   # matches post_tts_analysis DEFAULT_PAUSE_SEC = 0.3


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
    shotlist: dict,
    approved_items: dict[str, dict],
    manifest_pause: dict[str, int],
) -> tuple[int, int, list[str]]:
    """
    Patch shotlist shots[] duration_sec in-place.

    Args:
        shotlist:       Parsed ShotList.json (mutated in-place).
        approved_items: {item_id: {start_sec, end_sec, ...}} from vo_preview_approved.
                        start_sec/end_sec are episode-wide cumulative and include
                        scene tails (computed by voApproveTTS on the client).
        manifest_pause: {item_id: pause_after_ms} from AssetManifest_draft.
                        Used for ALL shots — pause_sec is always added to shot_vo_span
                        on top of tail_sec.  For the last shot it also determines
                        tail_sec (= SCENE_TAIL_SEC, pause stacks on top).

    Returns:
        (patched_count, skipped_count, warnings)

    Duration formula:
        shot_vo_span = last_item.end_sec − first_item.start_sec
        pause_sec    = last_item.pause_after_ms / 1000  (from manifest, default 300 ms)
        tail_sec     = max(gap − pause_sec, SCENE_TAIL_SEC)
                       where gap = next_first.start_sec − last_item.end_sec
                                 = pause_sec + approved_tail (set by voApproveTTS)
                       OR just SCENE_TAIL_SEC for the last shot (no look-ahead)
        duration_sec = shot_vo_span + pause_sec + tail_sec
    """
    patched  = 0
    skipped  = 0
    warnings = []
    shots    = shotlist.get("shots", [])

    # Pre-collect (first_item, last_item) per shot for look-ahead tail computation
    shot_bounds: list[tuple[dict, dict | None, dict | None]] = []
    for shot in shots:
        vo_ids = shot.get("audio_intent", {}).get("vo_item_ids", [])
        items  = [approved_items[iid] for iid in vo_ids if iid in approved_items]
        shot_bounds.append((shot, items[0] if items else None, items[-1] if items else None))

    for idx, (shot, first_item, last_item) in enumerate(shot_bounds):
        shot_id = shot.get("shot_id", "?")
        vo_ids  = shot.get("audio_intent", {}).get("vo_item_ids", [])

        if not vo_ids:
            skipped += 1
            print(f"  [SKIP] {shot_id}  — no vo_item_ids (duration unchanged: "
                  f"{shot.get('duration_sec', 0)}s)")
            continue

        if first_item is None:
            msg = (f"{shot_id}: vo_item_ids {vo_ids} not found in approved file — "
                   "duration_sec unchanged")
            warnings.append(msg)
            print(f"  [WARN] {msg}")
            skipped += 1
            continue

        missing = [iid for iid in vo_ids if iid not in approved_items]
        if missing:
            msg = f"{shot_id}: items not in approved file: {missing}"
            warnings.append(msg)
            print(f"  [WARN] {msg} — using available items")

        shot_vo_span  = last_item["end_sec"] - first_item["start_sec"]

        # Pause after the last VO item in this shot (inter-sentence silence).
        # voApproveTTS writes end_sec = start + dur (no pause), then advances
        # its cursor by dur + pauseMs — so the pause lives in the gap between
        # last_item.end_sec and next_shot.first_item.start_sec, not in end_sec.
        # We must add it explicitly so ShotList matches the VO Timeline display.
        last_pause_ms = manifest_pause.get(last_item["item_id"], DEFAULT_PAUSE_MS)
        pause_sec     = last_pause_ms / 1000.0

        # Tail: gap from this shot's last item end to the next VO shot's first item start.
        # gap = pause_sec + approved_tail_sec  (both set by voApproveTTS).
        # Strip the pause to isolate the raw approved tail, then enforce SCENE_TAIL_SEC.
        tail_sec     = SCENE_TAIL_SEC
        tail_source  = "default"
        for j in range(idx + 1, len(shot_bounds)):
            next_first = shot_bounds[j][1]
            if next_first is not None:
                gap      = next_first["start_sec"] - last_item["end_sec"]
                raw_tail = gap - pause_sec          # gap minus the trailing pause
                tail_sec = max(raw_tail, SCENE_TAIL_SEC)
                tail_source = f"gap-to-{shot_bounds[j][0].get('shot_id','?')}"
                break
        else:
            # Last VO shot — no next-shot look-ahead; always use SCENE_TAIL_SEC.
            # pause_sec is still stacked on top (already computed above).
            tail_sec    = SCENE_TAIL_SEC
            tail_source = "manifest-pause"

        new_dur = round(shot_vo_span + pause_sec + tail_sec, 3)
        old_dur = shot.get("duration_sec", 0)
        shot["duration_sec"] = new_dur
        # Carry episode-cumulative timestamps so downstream tools (gen_render_plan,
        # SRT validators) can derive absolute timeline positions without re-reading
        # vo_preview_approved.  These are the hard truth from the approved file.
        shot["start_sec"] = round(first_item["start_sec"], 3)
        shot["end_sec"]   = round(last_item["end_sec"], 3)
        patched += 1

        change = "PATCH" if abs(new_dur - old_dur) > 0.01 else "OK   "
        print(f"  [{change}] {shot_id:20s}  "
              f"old={old_dur:7.3f}s → new={new_dur:7.3f}s  "
              f"(span={shot_vo_span:.3f}s + pause={pause_sec:.3f}s + tail={tail_sec:.3f}s  [{tail_source}])")

    return patched, skipped, warnings


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Patch ShotList.json duration_sec from approved VO timing.\n"
            "Runs after Stage 4 LLM, before Stage 5."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "ep_dir",
        help="Episode directory (e.g. projects/slug/episodes/s01e01)",
    )
    p.add_argument(
        "--locale", default="en",
        help="Primary locale whose approved timing drives the patch (default: en)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Compute and print durations without writing ShotList.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ep_dir = Path(args.ep_dir)
    if not ep_dir.is_dir():
        # Try resolving relative to PIPE_DIR (matches run.sh convention)
        ep_dir = PIPE_DIR / args.ep_dir
    if not ep_dir.is_dir():
        print(f"[ERROR] ep_dir not found: {args.ep_dir}", file=sys.stderr)
        sys.exit(1)
    ep_dir = ep_dir.resolve()
    locale = args.locale

    # ── Load vo_preview_approved ──────────────────────────────────────────────
    approved_path = ep_dir / f"vo_preview_approved.{locale}.json"
    if not approved_path.exists():
        print(
            f"[ERROR] vo_preview_approved.{locale}.json not found in {ep_dir}\n"
            "       Stage 3.5 approval is required before Stage 4 runs.",
            file=sys.stderr,
        )
        sys.exit(1)

    approved_data  = load_json(approved_path)
    approved_items = {item["item_id"]: item for item in approved_data.get("items", [])}

    # ── Load AssetManifest_draft for pause_after_ms (optional) ───────────────
    manifest_pause: dict[str, int] = {}
    manifest_path = ep_dir / f"AssetManifest_draft.{locale}.json"
    if manifest_path.exists():
        try:
            mani = load_json(manifest_path)
            for v in mani.get("vo_items", []):
                iid = v.get("item_id")
                if iid and "pause_after_ms" in v:
                    manifest_pause[iid] = int(v["pause_after_ms"])
        except Exception as exc:
            print(f"[WARN] Could not read pause_after_ms from manifest: {exc}")
    else:
        print(
            f"[WARN] AssetManifest_draft.{locale}.json not found — "
            f"using default {DEFAULT_PAUSE_MS} ms inter-item pause for all shots."
        )

    # ── Load ShotList ─────────────────────────────────────────────────────────
    shotlist_path = ep_dir / "ShotList.json"
    if not shotlist_path.exists():
        print(
            f"[ERROR] ShotList.json not found in {ep_dir}\n"
            "       Run Stage 4 before calling patch_shotlist_durations.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    shotlist = load_json(shotlist_path)

    # ── Patch ─────────────────────────────────────────────────────────────────
    shot_count = len(shotlist.get("shots", []))
    print("=" * 60)
    print("  patch_shotlist_durations")
    print(f"  Episode : {ep_dir}")
    print(f"  Locale  : {locale}")
    print(f"  Shots   : {shot_count}")
    print(f"  Approved: {len(approved_items)} items")
    print(f"  Pauses  : {len(manifest_pause)} custom pause_after_ms overrides")
    print("=" * 60)

    patched, skipped, warnings = patch(shotlist, approved_items, manifest_pause)

    # ── Update top-level total_duration_sec ───────────────────────────────────
    # The LLM writes its own estimate; patch() fixes each shot but never updates
    # the top-level field.  Recompute from the now-patched shots[] array so the
    # field reflects the approved VO truth rather than the LLM's stale guess.
    if "total_duration_sec" in shotlist:
        old_total = shotlist["total_duration_sec"]
        new_total = round(sum(s.get("duration_sec", 0.0) for s in shotlist.get("shots", [])), 3)
        shotlist["total_duration_sec"] = new_total
        if abs(new_total - old_total) > 0.01:
            print(f"\n  [PATCH] total_duration_sec: {old_total} → {new_total}s")

    # ── Write ─────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"  Patched : {patched} shots")
    print(f"  Skipped : {skipped} shots (no VO)")
    if warnings:
        print(f"  Warnings: {len(warnings)}")
        for w in warnings:
            print(f"    ⚠  {w}")

    if args.dry_run:
        print("\n  [DRY-RUN] ShotList.json NOT written.")
    else:
        save_json(shotlist, shotlist_path)
        print(f"\n  ✓ ShotList.json updated: {shotlist_path}")
    print("=" * 60)

    if warnings:
        sys.exit(1)  # exit code 1 so run.sh can detect data integrity issues


if __name__ == "__main__":
    main()
