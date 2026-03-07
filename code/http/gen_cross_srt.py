#!/usr/bin/env python3
# =============================================================================
# gen_cross_srt.py — Build cross-language SRT files from locale sidecar JSONs
# =============================================================================
#
# For a multi-locale episode each render folder gets two SRT files:
#   renders/{locale_A}/output.{locale_A}.srt  — native (written by render_video.py)
#   renders/{locale_A}/output.{locale_B}.srt  — cross: A timecodes + B text
#
# Entries are matched by line_id (NOT by list position) so a missing or empty
# subtitle in one locale never causes silent drift in the cross file.
#
# Usage:
#   python3 gen_cross_srt.py \
#       --ep-dir  projects/tennis/episodes/s01e01 \
#       --locales en,zh-Hans \
#       --primary en
#
# Requires: stdlib only
# =============================================================================

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate cross-language SRT files from locale subtitle sidecars."
    )
    p.add_argument("--ep-dir",  required=True, type=Path,
                   help="Episode root directory (contains renders/)")
    p.add_argument("--locales", required=True,
                   help="Comma-separated locale list, e.g. 'en,zh-Hans'")
    p.add_argument("--primary", default="en",
                   help="Primary locale (default: en) — informational only")
    return p.parse_args()


def load_subs(subs_path: Path) -> list[dict] | None:
    """Load output.subs.json; return None if missing or malformed."""
    if not subs_path.exists():
        print(f"  [SKIP] subs not found: {subs_path}")
        return None
    try:
        data = json.loads(subs_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            print(f"  [WARN] unexpected format in {subs_path} — skipping")
            return None
        return data
    except Exception as exc:
        print(f"  [WARN] failed to read {subs_path}: {exc} — skipping")
        return None


def write_cross_srt(out_path: Path, entries: list[tuple[str, str]]) -> None:
    """Write a cross-language SRT from (timecode, text) pairs."""
    blocks: list[str] = []
    for seq, (timecode, text) in enumerate(entries, 1):
        blocks.append(f"{seq}\n{timecode}\n{text}\n")
    out_path.write_text("\n".join(blocks), encoding="utf-8")
    print(f"  [OK]  {out_path}  ({len(entries)} entries)")


def build_cross(
    renders_dir: Path,
    locale_a: str,
    locale_b: str,
) -> None:
    """
    Create renders/A/output.B.srt  (A timecodes + B text)
    and   renders/B/output.A.srt  (B timecodes + A text).
    """
    subs_a = load_subs(renders_dir / locale_a / "output.subs.json")
    subs_b = load_subs(renders_dir / locale_b / "output.subs.json")

    if subs_a is None or subs_b is None:
        print(f"  [SKIP] cross SRT for {locale_a} ↔ {locale_b} — sidecar(s) missing")
        return

    # Build lookup: line_id → {timecode, text} for each locale
    idx_a = {e["line_id"]: e for e in subs_a if e.get("line_id")}
    idx_b = {e["line_id"]: e for e in subs_b if e.get("line_id")}

    # renders/A/output.B.srt — A timecodes + B text (in A's order)
    entries_ab: list[tuple[str, str]] = []
    for entry in subs_a:
        lid = entry.get("line_id", "")
        if lid in idx_b:
            entries_ab.append((entry["timecode"], idx_b[lid]["text"]))
        else:
            print(f"  [WARN] line_id '{lid}' missing from {locale_b} — skipped")

    # renders/B/output.A.srt — B timecodes + A text (in B's order)
    entries_ba: list[tuple[str, str]] = []
    for entry in subs_b:
        lid = entry.get("line_id", "")
        if lid in idx_a:
            entries_ba.append((entry["timecode"], idx_a[lid]["text"]))
        else:
            print(f"  [WARN] line_id '{lid}' missing from {locale_a} — skipped")

    write_cross_srt(renders_dir / locale_a / f"output.{locale_b}.srt", entries_ab)
    write_cross_srt(renders_dir / locale_b / f"output.{locale_a}.srt", entries_ba)


def main() -> None:
    args = parse_args()

    locales = [l.strip() for l in args.locales.split(",") if l.strip()]
    if len(locales) < 2:
        print(f"Only one locale configured ({locales}) — no cross SRT needed.")
        sys.exit(0)

    renders_dir = args.ep_dir / "renders"
    if not renders_dir.is_dir():
        print(f"[SKIP] renders dir not found: {renders_dir}")
        sys.exit(0)

    print(f"\n── Cross-language SRT generation ({'  ↔  '.join(locales)}) ──")

    # Build cross files for every ordered pair (A, B)
    seen: set[frozenset] = set()
    for a in locales:
        for b in locales:
            if a == b:
                continue
            pair = frozenset({a, b})
            if pair in seen:
                continue
            seen.add(pair)
            print(f"\n  {a} ↔ {b}")
            build_cross(renders_dir, a, b)

    print("\n── Cross-language SRT done ──")


if __name__ == "__main__":
    main()
