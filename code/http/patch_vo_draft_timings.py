#!/usr/bin/env python3
# =============================================================================
# patch_vo_draft_timings.py — Patch AssetManifest locale vo_items from approved VO
# =============================================================================
#
# Runs AFTER Stage 5 (gen_vo_manifest.py) and BEFORE Stage 9 (manifest_merge).
# Called by run.sh immediately after each per-locale VO manifest is generated.
#
# Problem: gen_vo_manifest.py builds vo_items with only estimated_duration_sec.
# It does NOT read vo_preview_approved.{locale}.json, so start_sec and end_sec
# are absent from the draft.  manifest_merge.py then computes duck_intervals
# from missing timing → all duck_intervals are empty.
#
# Fix: this script patches each vo_item in VOPlan.{locale}.json
# with the exact start_sec, end_sec, and duration_sec from the approved file.
# The approved file is the single source of truth for all VO timing.
#
# Reads:
#   vo_preview_approved.{locale}.json  — approved item timing (start/end/duration)
#   VOPlan.{locale}.json        — locale manifest to patch
#
# Writes:
#   VOPlan.{locale}.json        — vo_items patched in-place (atomic)
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
    draft: dict,
    approved_items: dict[str, dict],
) -> tuple[int, int, list[str]]:
    """
    Patch AssetManifest vo_items[] start_sec/end_sec/duration_sec in-place.

    Args:
        draft:          Parsed AssetManifest locale manifest (mutated in-place).
        approved_items: {item_id: {start_sec, end_sec, duration_sec, ...}}
                        from vo_preview_approved.{locale}.json.
                        These are episode-wide cumulative timestamps.

    Returns:
        (patched_count, skipped_count, warnings)
    """
    patched  = 0
    skipped  = 0
    warnings = []
    vo_items = draft.get("vo_items", [])

    for item in vo_items:
        item_id = item.get("item_id", "")
        if not item_id:
            skipped += 1
            continue

        if item_id not in approved_items:
            msg = f"{item_id}: not found in approved file — timing unchanged"
            warnings.append(msg)
            print(f"  [WARN] {msg}")
            skipped += 1
            continue

        ap = approved_items[item_id]
        old_dur = item.get("estimated_duration_sec") or item.get("duration_sec")

        item["start_sec"]    = float(ap["start_sec"])
        item["end_sec"]      = float(ap["end_sec"])
        item["duration_sec"] = float(ap["duration_sec"])
        patched += 1

        change = "PATCH" if abs(float(ap["duration_sec"]) - (old_dur or 0)) > 0.01 else "OK   "
        print(f"  [{change}] {item_id:30s}  "
              f"start={ap['start_sec']:.3f}s  "
              f"end={ap['end_sec']:.3f}s  "
              f"dur={ap['duration_sec']:.3f}s")

    return patched, skipped, warnings


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Patch VOPlan.{locale}.json vo_items with approved VO timing.\n"
            "Runs after Stage 5 gen_vo_manifest.py, before Stage 9 manifest_merge."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "ep_dir",
        help="Episode directory (e.g. projects/slug/episodes/s01e01)",
    )
    p.add_argument(
        "--locale", default="en",
        help="Locale whose draft manifest to patch (default: en)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Compute and print changes without writing the draft manifest",
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
    locale = args.locale

    # ── Load timing from AssetManifest (requires vo_approval block) ─────────
    manifest_path_check = ep_dir / f"VOPlan.{locale}.json"
    if not manifest_path_check.exists():
        print(
            f"[SKIP] VOPlan.{locale}.json not found in {ep_dir}\n"
            "       No timing patch applied (Stage 3.5 approval not yet done).",
        )
        sys.exit(0)  # non-fatal: patch runs opportunistically

    _manifest_check = load_json(manifest_path_check)
    if not _manifest_check.get("vo_approval", {}).get("approved_at"):
        print(
            f"[SKIP] VOPlan.{locale}.json has no vo_approval block in {ep_dir}\n"
            "       No timing patch applied (Stage 3.5 approval not yet done).",
        )
        sys.exit(0)  # non-fatal: patch runs opportunistically

    approved_items = {item["item_id"]: item for item in _manifest_check.get("vo_items", [])}

    # ── Load AssetManifest locale manifest ───────────────────────────────────
    manifest_path = ep_dir / f"VOPlan.{locale}.json"
    if not manifest_path.exists():
        print(
            f"[ERROR] VOPlan.{locale}.json not found in {ep_dir}\n"
            "       Run Stage 5 before calling patch_vo_draft_timings.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    draft = load_json(manifest_path)

    # ── Patch ─────────────────────────────────────────────────────────────────
    vo_count = len(draft.get("vo_items", []))
    print("=" * 60)
    print("  patch_vo_draft_timings")
    print(f"  Episode  : {ep_dir}")
    print(f"  Locale   : {locale}")
    print(f"  VO items : {vo_count}")
    print(f"  Approved : {len(approved_items)} items")
    print("=" * 60)

    patched, skipped, warnings = patch(draft, approved_items)

    # ── Write ─────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"  Patched : {patched} vo_items")
    print(f"  Skipped : {skipped} vo_items")
    if warnings:
        print(f"  Warnings: {len(warnings)}")
        for w in warnings:
            print(f"    ⚠  {w}")

    if args.dry_run:
        print("\n  [DRY-RUN] AssetManifest NOT written.")
    else:
        save_json(draft, manifest_path)
        print(f"\n  ✓ VOPlan.{locale}.json updated: {manifest_path}")
    print("=" * 60)

    if warnings:
        sys.exit(1)  # exit code 1 so run.sh can detect data integrity issues


if __name__ == "__main__":
    main()
