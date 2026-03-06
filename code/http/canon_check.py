#!/usr/bin/env python3
# =============================================================================
# canon_check.py — Stage 1: Canon consistency check (deterministic)
# =============================================================================
#
# Replaces the LLM-based Stage 1. Reads canon.json for the project (if it
# exists) and prints a structured report of all tracked canon state.
#
# Usage:
#   python canon_check.py <ep_dir>
#
#   ep_dir — path to the episode directory (e.g. projects/slug/episodes/s01e01)
#
# Requirements: stdlib only (json, pathlib, argparse, re, sys)
# =============================================================================

import argparse
import json
import re
import sys
from pathlib import Path

STAGE_LABEL = "[1/9] Stage 1 — Canon Check"


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_pipeline_vars(ep_dir: Path) -> dict:
    """Parse pipeline_vars.sh and return dict of exported variables."""
    vars_file = ep_dir / "pipeline_vars.sh"
    if not vars_file.exists():
        raise FileNotFoundError(f"pipeline_vars.sh not found: {vars_file}")
    result = {}
    for line in vars_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        # Match: export KEY="value" or export KEY=value or KEY="value"
        m = re.match(r'^(?:export\s+)?([A-Z_][A-Z0-9_]*)=["\']?(.*?)["\']?\s*$', line)
        if m:
            result[m.group(1)] = m.group(2)
    return result


# ── Canon report ──────────────────────────────────────────────────────────────

def print_canon_report(canon: dict) -> None:
    """Print a structured human-readable report of canon contents."""
    characters = canon.get("characters", {})
    world_facts = canon.get("world_facts", [])
    unresolved_threads = canon.get("unresolved_threads", [])
    episode_summaries = canon.get("episode_summaries", [])

    # Characters
    char_count = len(characters)
    char_ids = list(characters.keys())
    print(f"\n  Characters ({char_count}):")
    if char_ids:
        for cid in char_ids:
            entry = characters[cid]
            status   = entry.get("status", "unknown")
            location = entry.get("location", "unknown")
            print(f"    • {cid}  [status: {status}, location: {location}]")
    else:
        print("    (none)")

    # World facts
    wf_count = len(world_facts)
    print(f"\n  World Facts ({wf_count}):")
    if world_facts:
        for i, fact in enumerate(world_facts, 1):
            print(f"    {i:2d}. {fact}")
    else:
        print("    (none)")

    # Unresolved threads
    ut_count = len(unresolved_threads)
    print(f"\n  Unresolved Threads ({ut_count}):")
    if unresolved_threads:
        for thread in unresolved_threads:
            print(f"    - {thread}")
    else:
        print("    (none)")

    # Episode summaries — show count + last 2 entries
    es_count = len(episode_summaries)
    print(f"\n  Episode Summaries ({es_count} total):")
    if episode_summaries:
        recent = episode_summaries[-2:]
        if es_count > 2:
            print(f"    (showing last 2 of {es_count})")
        for entry in recent:
            ep      = entry.get("ep", "?")
            summary = entry.get("summary", "")
            print(f"    [{ep}] {summary}")
    else:
        print("    (none)")

    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 1: Canon consistency check. "
                    "Reads canon.json for the project and prints a structured report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("ep_dir", metavar="EP_DIR",
                   help="Episode directory (contains pipeline_vars.sh).")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    ep_dir = Path(args.ep_dir).resolve()

    print(f"▶ {STAGE_LABEL}")

    # Parse pipeline_vars.sh to find PROJECT_SLUG
    try:
        pipeline_vars = load_pipeline_vars(ep_dir)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    project_slug = pipeline_vars.get("PROJECT_SLUG", "").strip()
    if not project_slug:
        print("[ERROR] PROJECT_SLUG not found in pipeline_vars.sh", file=sys.stderr)
        sys.exit(1)

    # Resolve canon path relative to repo root (two levels up from ep_dir's
    # episodes/<id> → project/<slug> → projects/<slug>/canon.json).
    # pipeline_vars.sh lives in the episode dir; canon.json is at the project root.
    # Walk up from ep_dir to find the projects/ directory by looking for the slug.
    canon_rel = Path("projects") / project_slug / "canon.json"

    # Try ep_dir-relative first, then walk up to repo root candidates
    search_roots = [ep_dir]
    # Add parent directories up to 6 levels deep (covers typical repo layouts)
    parent = ep_dir
    for _ in range(6):
        parent = parent.parent
        search_roots.append(parent)

    canon_path: Path | None = None
    for root in search_roots:
        candidate = root / canon_rel
        if candidate.exists():
            canon_path = candidate
            break

    if canon_path is None:
        print(f"  No canon constraints found for this project.")
        print(f"  (Searched for: {canon_rel})")
        print(f"\n✓ {STAGE_LABEL} done")
        sys.exit(0)

    # Canon found — load and report
    try:
        canon = load_json(canon_path)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[ERROR] Failed to load canon.json: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  Canon file : {canon_path}")
    print(f"  Project    : {project_slug}")
    print_canon_report(canon)

    print(f"✓ {STAGE_LABEL} done")


if __name__ == "__main__":
    main()
