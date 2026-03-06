#!/usr/bin/env python3
# =============================================================================
# canon_diff_chars.py — P9 Phase A: Deterministic character extraction
# =============================================================================
#
# Reads Script.json and the project canon (if it exists) and produces a
# canon_diff_partial.json with deterministic character fields pre-filled and
# __FILL__ markers for the narrative fields the LLM will complete.
#
# If canon_diff.json already exists in the episode directory AND contains no
# __FILL__ markers, the script exits 0 (already completed by LLM — idempotent).
#
# Usage:
#   python3 canon_diff_chars.py <ep_dir>
#
#   ep_dir — path to the episode directory (e.g. projects/slug/episodes/s01e02)
#
# Requirements: stdlib only (json, pathlib, argparse, re, sys)
# =============================================================================

import argparse
import json
import re
import sys
from pathlib import Path

STAGE_LABEL = "canon_diff_chars.py"


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


# ── __FILL__ detection ────────────────────────────────────────────────────────

def has_fill_markers(obj) -> bool:
    """Return True if obj contains any __FILL__ string value anywhere."""
    if isinstance(obj, dict):
        return any(has_fill_markers(v) for v in obj.values())
    if isinstance(obj, list):
        return any(has_fill_markers(v) for v in obj)
    if isinstance(obj, str):
        return obj.startswith("__FILL__")
    return False


# ── Canon path resolution (mirrors canon_check.py convention) ─────────────────

def find_canon_path(ep_dir: Path, project_slug: str) -> Path | None:
    """Search up the directory tree for projects/{slug}/canon.json."""
    canon_rel = Path("projects") / project_slug / "canon.json"
    search_roots = [ep_dir]
    parent = ep_dir
    for _ in range(6):
        parent = parent.parent
        search_roots.append(parent)
    for root in search_roots:
        candidate = root / canon_rel
        if candidate.exists():
            return candidate
    return None


# ── Character extraction ──────────────────────────────────────────────────────

def extract_canon_entry(cast_member: dict) -> dict:
    """
    Map a Script.cast entry to the CanonDiff added_characters item format.

    CanonDiff.v1.json required fields: id, role, status, location, knows, relationships
    Script.v1.json cast fields: character_id (required), gender (required), role (optional),
                                plus additionalProperties (status, location, knows,
                                relationships, traits, etc.)
    """
    traits = cast_member.get("traits", {}) if isinstance(cast_member.get("traits"), dict) else {}

    # status: try traits.status first, then top-level status, fall back to "unknown"
    status = traits.get("status") or cast_member.get("status") or "unknown"

    # location: try traits.location first, then top-level location, fall back to "unspecified"
    location = traits.get("location") or cast_member.get("location") or "unspecified"

    # knows: try top-level knows, then traits.knows, fall back to []
    knows_raw = cast_member.get("knows") or traits.get("knows") or []
    knows = knows_raw if isinstance(knows_raw, list) else []

    # relationships: try top-level, then traits, fall back to {}
    rels_raw = cast_member.get("relationships") or traits.get("relationships") or {}
    relationships = rels_raw if isinstance(rels_raw, dict) else {}

    return {
        "id":            cast_member["character_id"],
        "role":          cast_member.get("role", "unknown"),
        "status":        str(status),
        "location":      str(location),
        "knows":         knows,
        "relationships": relationships,
    }


def compute_updated_states(
    cast_member: dict,
    canon_entry: dict,
) -> list[dict]:
    """
    Compare Script cast fields against a canon character entry.
    Record updated_states items for status or location that have an explicit
    non-default value in Script that differs from canon.
    """
    updates = []
    traits = cast_member.get("traits", {}) if isinstance(cast_member.get("traits"), dict) else {}

    # status comparison
    script_status = traits.get("status") or cast_member.get("status")
    if script_status and script_status != "unknown":
        canon_status = canon_entry.get("status", "unknown")
        if script_status != canon_status:
            updates.append({
                "character_id": cast_member["character_id"],
                "field":        "status",
                "new_value":    str(script_status),
            })

    # location comparison
    script_location = traits.get("location") or cast_member.get("location")
    if script_location and script_location != "unspecified":
        canon_location = canon_entry.get("location", "unspecified")
        if script_location != canon_location:
            updates.append({
                "character_id": cast_member["character_id"],
                "field":        "location",
                "new_value":    str(script_location),
            })

    return updates


def build_character_diffs(
    script: dict,
    canon_characters: dict,
) -> tuple[list[dict], list[dict]]:
    """
    Compare Script.cast against canon.characters.

    Returns:
        added_characters  — CanonDiff entries for cast members not in canon
        updated_states    — CanonDiff state-change entries for existing characters
    """
    added_characters: list[dict] = []
    updated_states:   list[dict] = []

    for cast_member in script.get("cast", []):
        char_id = cast_member.get("character_id")
        if not char_id:
            continue

        if char_id not in canon_characters:
            # New character — add full entry
            added_characters.append(extract_canon_entry(cast_member))
        else:
            # Existing character — check for state changes
            canon_entry = canon_characters[char_id]
            updates = compute_updated_states(cast_member, canon_entry)
            updated_states.extend(updates)

    return added_characters, updated_states


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "P9 Phase A — Deterministic character extraction from Script.json. "
            "Produces canon_diff_partial.json with __FILL__ markers for narrative fields."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("ep_dir", metavar="EP_DIR",
                   help="Episode directory (contains pipeline_vars.sh, Script.json).")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    ep_dir = Path(args.ep_dir).resolve()

    # ── Load pipeline_vars.sh ─────────────────────────────────────────────────
    try:
        pipeline_vars = load_pipeline_vars(ep_dir)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    project_slug = pipeline_vars.get("PROJECT_SLUG", "").strip()
    episode_id   = pipeline_vars.get("EPISODE_ID",   "").strip()

    if not project_slug:
        print("[ERROR] PROJECT_SLUG not found in pipeline_vars.sh", file=sys.stderr)
        sys.exit(1)
    if not episode_id:
        print("[ERROR] EPISODE_ID not found in pipeline_vars.sh", file=sys.stderr)
        sys.exit(1)

    print(f"▶ {STAGE_LABEL} — {episode_id}")

    # ── Guard: canon_diff.json already completed? ─────────────────────────────
    canon_diff_path = ep_dir / "canon_diff.json"
    if canon_diff_path.exists():
        try:
            existing = load_json(canon_diff_path)
            if not has_fill_markers(existing):
                print(f"  canon_diff.json already completed (no __FILL__ markers) — skipping")
                print(f"✓ {STAGE_LABEL} done (idempotent exit)")
                sys.exit(0)
        except (json.JSONDecodeError, OSError):
            # Unreadable — proceed and overwrite partial
            pass

    # ── Load Script.json ──────────────────────────────────────────────────────
    script_path = ep_dir / "Script.json"
    if not script_path.exists():
        print(f"[ERROR] Script.json not found: {script_path}", file=sys.stderr)
        sys.exit(1)

    try:
        script = load_json(script_path)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[ERROR] Failed to load Script.json: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Load canon.json (optional) ────────────────────────────────────────────
    canon_path = find_canon_path(ep_dir, project_slug)
    canon_characters: dict = {}

    if canon_path is not None:
        try:
            canon = load_json(canon_path)
            canon_characters = canon.get("characters", {})
            if not isinstance(canon_characters, dict):
                canon_characters = {}
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [WARN] Failed to load canon.json ({exc}) — treating as empty",
                  file=sys.stderr)
            canon_characters = {}
    else:
        print(f"  No canon.json found for project '{project_slug}' — all cast are new")

    # ── Compute diffs ─────────────────────────────────────────────────────────
    added_characters, updated_states = build_character_diffs(script, canon_characters)

    print(f"  added_characters: {len(added_characters)} new, "
          f"updated_states: {len(updated_states)} updates")

    # ── Build partial canon_diff ──────────────────────────────────────────────
    partial: dict = {
        "schema_id":      "CanonDiff",
        "schema_version": "1.0.0",
        "episode":        episode_id,
        "added_characters":        added_characters,
        "updated_states":          updated_states,
        "new_world_facts":         "__FILL__: list of new immutable world facts revealed in this episode",
        "new_unresolved_threads":  "__FILL__: list of new open plot threads introduced",
        "resolved_threads":        "__FILL__: list of previously open threads resolved in this episode",
    }

    # ── Write output ──────────────────────────────────────────────────────────
    out_path = ep_dir / "canon_diff_partial.json"
    try:
        save_json(partial, out_path)
    except OSError as exc:
        print(f"[ERROR] Failed to write canon_diff_partial.json: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"✓ canon_diff_partial.json written — LLM will complete narrative fields")


if __name__ == "__main__":
    main()
