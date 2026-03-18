#!/usr/bin/env python3
"""retune_vo.py — CLI for selective VO re-tuning.

Thin wrapper around code/http/vo_retune.py.
All core logic lives in vo_retune.py; this script handles arg parsing and output.

Usage examples:

  # Retune one item: change text + style
  python retune_vo.py \\
    --manifest projects/foo/episodes/s01e02/VOPlan.en.json \\
    --locale en \\
    --item vo-sc02-003 \\
    --text "New wording for this line." \\
    --style "excited" \\
    --rate "+10%"

  # Retune multiple items: change style on all
  python retune_vo.py \\
    --manifest projects/foo/episodes/s01e02/VOPlan.en.json \\
    --locale en \\
    --items vo-sc02-001 vo-sc02-002 vo-sc02-003 \\
    --style "calm"

  # Re-synthesize all items in a scene (current parameters, no text/style change)
  python retune_vo.py \\
    --manifest projects/foo/episodes/s01e02/VOPlan.en.json \\
    --locale en \\
    --scene sc02

  # Dry-run: preview what would change without writing anything
  python retune_vo.py \\
    --manifest projects/foo/episodes/s01e02/VOPlan.en.json \\
    --locale en \\
    --scene sc02 --style "calm" --dry-run

  # Retune with WAV backup
  python retune_vo.py \\
    --manifest projects/foo/episodes/s01e02/VOPlan.en.json \\
    --locale en \\
    --item vo-sc02-003 --text "New text." --backup
"""

import argparse
import sys
from pathlib import Path

# vo_retune.py is co-located in the same directory (code/http/)
# No sys.path manipulation required.
from vo_retune import retune_vo_items


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Selective VO retune — patch and re-synthesize one or more VO items.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    p.add_argument(
        "--manifest", required=True,
        help="Path to VOPlan.{locale}.json",
    )
    p.add_argument(
        "--locale", required=True,
        help="Locale code, e.g. 'en', 'zh-Hans'",
    )

    # Target selection — mutually exclusive, exactly one required
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument(
        "--item", metavar="ITEM_ID",
        help="Single item ID to retune",
    )
    sel.add_argument(
        "--items", metavar="ITEM_ID", nargs="+",
        help="Explicit list of item IDs to retune",
    )
    sel.add_argument(
        "--scene", metavar="SCENE_ID",
        help="Re-tune all items in this scene (e.g. sc02)",
    )

    # Patch fields — all optional; omitting all → zero-patch re-synthesis
    p.add_argument("--text",         metavar="TEXT",
                   help="New VO text")
    p.add_argument("--style",        metavar="STYLE",
                   help="azure_style (e.g. 'excited', 'calm', 'narration-professional')")
    p.add_argument("--rate",         metavar="RATE",
                   help="azure_rate (e.g. '+10%%', '-5%%', '0%%')")
    p.add_argument("--pitch",        metavar="PITCH",
                   help="azure_pitch (e.g. '+5%%', '-3%%')")
    p.add_argument("--style-degree", metavar="DEGREE", type=float,
                   dest="style_degree",
                   help="azure_style_degree, positive float (e.g. 1.8)")
    p.add_argument("--break-ms",     metavar="MS", type=int,
                   dest="break_ms",
                   help="azure_break_ms, non-negative integer (e.g. 400)")
    p.add_argument("--voice",        metavar="VOICE",
                   help="azure_voice name (e.g. 'zh-CN-YunjianNeural', 'en-US-AndrewNeural')")

    # Modes
    p.add_argument(
        "--dry-run", action="store_true",
        help="Preview changes without writing manifests or calling Azure TTS",
    )
    p.add_argument(
        "--backup", action="store_true",
        help="Backup manifests and WAV files with a timestamp suffix before writing",
    )

    return p


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_STATUS_ICON = {"ok": "✓", "error": "✗", "skipped": "–", "dry_run": "?"}


def _fmt_dur(v) -> str:
    return f"{v:.2f}s" if v is not None else "—"


def _print_results(results: list, dry_run: bool) -> bool:
    """Print per-item result rows. Returns True if any item errored."""
    any_error = False

    for r in results:
        icon   = _STATUS_ICON.get(r["status"], "?")
        before = _fmt_dur(r.get("before_duration_sec"))
        after  = _fmt_dur(r.get("after_duration_sec"))
        delta  = r.get("duration_delta_sec")
        warn   = "  ⚠️  duration drift" if r.get("duration_warn") else ""

        delta_str = ""
        if delta is not None and r["status"] == "ok":
            sign = "+" if delta >= 0 else ""
            delta_str = f"  ({sign}{delta:.2f}s)"

        print(f"  {icon} {r['item_id']}  {before} → {after}{delta_str}{warn}")

        if r["status"] == "error":
            print(f"      ERROR: {r['error']}", file=sys.stderr)
            any_error = True

        if r["status"] == "dry_run" and r.get("patch"):
            for field, val in r["patch"].items():
                print(f"      would set  {field} = {val!r}")

    return any_error


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    # Build patches dict from CLI flags
    patches: dict = {}
    if args.text         is not None: patches["text"]               = args.text
    if args.style        is not None: patches["azure_style"]        = args.style
    if args.rate         is not None: patches["azure_rate"]         = args.rate
    if args.pitch        is not None: patches["azure_pitch"]        = args.pitch
    if args.style_degree is not None: patches["azure_style_degree"] = args.style_degree
    if args.break_ms     is not None: patches["azure_break_ms"]     = args.break_ms
    if args.voice        is not None: patches["azure_voice"]        = args.voice

    # Resolve target selector
    item_ids = None
    scene    = None
    if   args.item:  item_ids = [args.item]
    elif args.items: item_ids = args.items
    else:            scene    = args.scene

    # Run
    try:
        results = retune_vo_items(
            manifest_path = args.manifest,
            locale        = args.locale,
            item_ids      = item_ids,
            scene         = scene,
            patches       = patches or None,
            dry_run       = args.dry_run,
            backup        = args.backup,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # Print results
    any_error = _print_results(results, dry_run=args.dry_run)

    # Summary line
    total   = len(results)
    ok      = sum(1 for r in results if r["status"] == "ok")
    errors  = sum(1 for r in results if r["status"] == "error")

    if args.dry_run:
        print(f"\nDry run complete: {total} item(s) would be processed.")
    else:
        suffix = f", {errors} failed" if errors else ""
        print(f"\nDone: {ok}/{total} succeeded{suffix}.")
        if ok:
            print("Re-run Stage 9 (render) to pick up the updated WAV files.")

    sys.exit(1 if any_error else 0)


if __name__ == "__main__":
    main()
