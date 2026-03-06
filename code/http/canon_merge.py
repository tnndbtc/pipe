#!/usr/bin/env python3
# =============================================================================
# canon_merge.py — Stage 7: Canon update (deterministic)
# =============================================================================
#
# Replaces the LLM-based Stage 7. Merges canon_diff.json into canon.json,
# applying character additions, state updates, world fact and thread
# accumulation, and episode summary recording.
#
# Usage:
#   python canon_merge.py <ep_dir>
#
#   ep_dir — path to the episode directory (contains pipeline_vars.sh and
#             canon_diff.json produced by Stage 6).
#
# Inputs:
#   ep_dir/pipeline_vars.sh        — for PROJECT_SLUG
#   ep_dir/canon_diff.json         — required; CanonDiff.v1 document
#   projects/{slug}/canon.json     — optional; Canon.v1 document (absent for ep001)
#
# Output:
#   projects/{slug}/canon.json     — updated Canon.v1 document (written in-place)
#
# Requirements: stdlib only (json, pathlib, argparse, re, sys)
# =============================================================================

import argparse
import json
import re
import sys
from pathlib import Path

STAGE_LABEL = "[7/9] Stage 7 — Canon Update"


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(doc: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")


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


# ── Canon search ──────────────────────────────────────────────────────────────

def find_canon_path(ep_dir: Path, project_slug: str) -> Path:
    """
    Resolve path to projects/{slug}/canon.json.

    Searches upward from ep_dir to locate the projects/ directory.
    Returns the resolved path (may not exist yet — caller checks).
    """
    canon_rel = Path("projects") / project_slug / "canon.json"

    search_roots = [ep_dir]
    parent = ep_dir
    for _ in range(6):
        parent = parent.parent
        search_roots.append(parent)

    # Return the first root that contains a projects/ directory
    for root in search_roots:
        if (root / "projects").is_dir():
            return root / canon_rel

    # Fallback: place canon next to ep_dir's project directory
    return ep_dir.parent.parent / canon_rel


def empty_canon(project_id: str) -> dict:
    """Return a fresh Canon.v1 document with no content."""
    return {
        "schema_id":          "Canon",
        "schema_version":     "1.0.0",
        "project_id":         project_id,
        "characters":         {},
        "world_facts":        [],
        "unresolved_threads": [],
        "episode_summaries":  [],
    }


# ── Merge logic ───────────────────────────────────────────────────────────────

def merge(canon: dict, diff: dict) -> dict:
    """
    Apply diff to canon in-place and return the modified canon.

    Steps (in order):
      1. Add new characters (skip if id already present)
      2. Apply state updates (skip if character not in canon)
      3. Append new world facts (skip exact duplicates)
      4. Append new unresolved threads (skip exact duplicates)
      5. Remove resolved threads
      6. Append episode summary entry
    """
    characters         = canon.setdefault("characters", {})
    world_facts        = canon.setdefault("world_facts", [])
    unresolved_threads = canon.setdefault("unresolved_threads", [])
    episode_summaries  = canon.setdefault("episode_summaries", [])

    stats = {
        "chars_added":     0,
        "chars_skipped":   0,
        "states_applied":  0,
        "states_skipped":  0,
        "facts_added":     0,
        "facts_skipped":   0,
        "threads_added":   0,
        "threads_skipped": 0,
        "threads_resolved":0,
    }

    # ── Step 1: Add characters ─────────────────────────────────────────────
    for char in diff.get("added_characters", []):
        cid = char["id"]
        if cid in characters:
            print(f"    [SKIP] Character '{cid}' already in canon — not overwriting.")
            stats["chars_skipped"] += 1
            continue
        characters[cid] = {
            "status":        char.get("status", ""),
            "location":      char.get("location", ""),
            "knows":         list(char.get("knows", [])),
            "relationships": dict(char.get("relationships", {})),
        }
        print(f"    [ADD]  Character '{cid}' (status={char.get('status')!r}, "
              f"location={char.get('location')!r})")
        stats["chars_added"] += 1

    # ── Step 2: Apply state updates ────────────────────────────────────────
    for update in diff.get("updated_states", []):
        cid   = update["character_id"]
        field = update["field"]
        value = update["new_value"]
        if cid not in characters:
            print(f"    [SKIP] updated_states: character '{cid}' not in canon.")
            stats["states_skipped"] += 1
            continue
        old_val = characters[cid].get(field, "<unset>")
        characters[cid][field] = value
        print(f"    [UPDATE] {cid}.{field}: {old_val!r} → {value!r}")
        stats["states_applied"] += 1

    # ── Step 3: Append new world facts (deduplicate) ───────────────────────
    existing_facts = set(world_facts)
    for fact in diff.get("new_world_facts", []):
        if fact in existing_facts:
            stats["facts_skipped"] += 1
            continue
        world_facts.append(fact)
        existing_facts.add(fact)
        stats["facts_added"] += 1

    # ── Step 4: Append new unresolved threads (deduplicate) ────────────────
    existing_threads = set(unresolved_threads)
    for thread in diff.get("new_unresolved_threads", []):
        if thread in existing_threads:
            stats["threads_skipped"] += 1
            continue
        unresolved_threads.append(thread)
        existing_threads.add(thread)
        stats["threads_added"] += 1

    # ── Step 5: Remove resolved threads ────────────────────────────────────
    for resolved in diff.get("resolved_threads", []):
        if resolved in existing_threads:
            unresolved_threads.remove(resolved)
            existing_threads.discard(resolved)
            stats["threads_resolved"] += 1

    # ── Step 6: Append episode summary ────────────────────────────────────
    episode_id = diff.get("episode", "unknown")

    # Use episode_summary field if the LLM added it (schema allows no extra
    # fields, but LLMs often include it anyway — honour it when present).
    summary_str = diff.get("episode_summary", "").strip()
    if not summary_str:
        resolved_list = diff.get("resolved_threads", [])
        new_thread_list = diff.get("new_unresolved_threads", [])
        resolved_str    = ", ".join(resolved_list) if resolved_list else "none"
        new_thread_str  = ", ".join(new_thread_list) if new_thread_list else "none"
        summary_str = (
            f"{episode_id}: Resolved: {resolved_str}. "
            f"New threads: {new_thread_str}."
        )

    episode_summaries.append({"ep": episode_id, "summary": summary_str})
    print(f"    [SUMMARY] {episode_id}: {summary_str[:80]}"
          f"{'…' if len(summary_str) > 80 else ''}")

    return canon, stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 7: Merge canon_diff.json into canon.json deterministically.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("ep_dir", metavar="EP_DIR",
                   help="Episode directory (contains pipeline_vars.sh and canon_diff.json).")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    ep_dir = Path(args.ep_dir).resolve()

    print(f"▶ {STAGE_LABEL}")

    # Parse pipeline_vars.sh
    try:
        pipeline_vars = load_pipeline_vars(ep_dir)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    project_slug = pipeline_vars.get("PROJECT_SLUG", "").strip()
    if not project_slug:
        print("[ERROR] PROJECT_SLUG not found in pipeline_vars.sh", file=sys.stderr)
        sys.exit(1)

    # Load canon_diff.json (required)
    diff_path = ep_dir / "canon_diff.json"
    if not diff_path.exists():
        print(f"[ERROR] canon_diff.json not found: {diff_path}", file=sys.stderr)
        sys.exit(1)
    try:
        diff = load_json(diff_path)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[ERROR] Failed to load canon_diff.json: {exc}", file=sys.stderr)
        sys.exit(1)

    # Resolve canon.json path
    canon_path = find_canon_path(ep_dir, project_slug)

    # Load existing canon or start fresh
    if canon_path.exists():
        try:
            canon = load_json(canon_path)
            print(f"  Canon file   : {canon_path}  (existing)")
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[ERROR] Failed to load canon.json: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        canon = empty_canon(project_slug)
        print(f"  Canon file   : {canon_path}  (new — will be created)")

    print(f"  Diff file    : {diff_path}")
    print(f"  Episode      : {diff.get('episode', '?')}")
    print()
    print("  Applying diff:")

    # Apply merge
    canon, stats = merge(canon, diff)

    # Ensure schema fields are present
    if "schema_id" not in canon:
        canon["schema_id"] = "Canon"
    if "schema_version" not in canon:
        canon["schema_version"] = "1.0.0"
    if "project_id" not in canon or not canon["project_id"]:
        canon["project_id"] = project_slug

    # Write updated canon
    save_json(canon, canon_path)

    # Summary
    print()
    print("  Changes applied:")
    print(f"    Characters added      : {stats['chars_added']}")
    if stats["chars_skipped"]:
        print(f"    Characters skipped    : {stats['chars_skipped']} (already in canon)")
    print(f"    State updates applied : {stats['states_applied']}")
    if stats["states_skipped"]:
        print(f"    State updates skipped : {stats['states_skipped']} (character not found)")
    print(f"    World facts added     : {stats['facts_added']}")
    if stats["facts_skipped"]:
        print(f"    World facts skipped   : {stats['facts_skipped']} (duplicate)")
    print(f"    Threads added         : {stats['threads_added']}")
    print(f"    Threads resolved      : {stats['threads_resolved']}")
    if stats["threads_skipped"]:
        print(f"    Threads skipped       : {stats['threads_skipped']} (duplicate)")
    print()
    print("  Canon totals after merge:")
    print(f"    Characters            : {len(canon.get('characters', {}))}")
    print(f"    World facts           : {len(canon.get('world_facts', []))}")
    print(f"    Unresolved threads    : {len(canon.get('unresolved_threads', []))}")
    print(f"    Episode summaries     : {len(canon.get('episode_summaries', []))}")
    print()
    print(f"  [OK] {canon_path}")
    print()
    print(f"✓ {STAGE_LABEL} done")


if __name__ == "__main__":
    main()
