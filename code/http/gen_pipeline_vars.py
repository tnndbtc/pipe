#!/usr/bin/env python3
# =============================================================================
# gen_pipeline_vars.py — Stage 0 Phase A: read meta.json → write pipeline_vars.sh
# =============================================================================
#
# Deterministic replacement for the pipeline_vars.sh generation part of the
# LLM-driven Stage 0 prompt (p_0.txt).  No AI calls required.
#
# Reads ep_dir/meta.json and writes ep_dir/pipeline_vars.sh with the same
# format and variable set produced by ssml_preprocess.py:build_pipeline_vars().
#
# Usage:
#   python gen_pipeline_vars.py <ep_dir>
#
#   ep_dir — path to the episode directory
#             e.g. projects/the-pharaoh-who-defied-death/episodes/s01e02
#
# Output:
#   ep_dir/pipeline_vars.sh
#
# Requirements: stdlib only (json, pathlib, sys)
# =============================================================================

import json
import sys
from pathlib import Path

STAGE_LABEL_START = "Stage 0a — gen_pipeline_vars.py"
STAGE_LABEL_DONE  = "Stage 0a done — pipeline_vars.sh written"


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_text(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ── pipeline_vars.sh builder ──────────────────────────────────────────────────

def build_pipeline_vars(meta: dict) -> str:
    """Build pipeline_vars.sh content from meta.json fields.

    Matches the exact output format of ssml_preprocess.py:build_pipeline_vars(),
    including variable order and quoting style.

    The PRIMARY_LOCALE is derived from the 'primary_locale' field in meta.json
    if present; otherwise it defaults to the first locale in the LOCALES list.
    ssml_preprocess sets it from the parsed SSML xml:lang — for episodic content
    (no SSML), we use the same fallback path: meta.get('primary_locale', first).
    """
    slug    = meta["project_slug"]
    ep_id   = meta["episode_id"]
    locales = str(meta.get("locales", "en"))

    # Derive PRIMARY_LOCALE: explicit field first, then first token of locales.
    primary = meta.get("primary_locale", "")
    if not primary:
        primary = locales.split(",")[0].strip()

    lines = [
        f'export STORY_TITLE="{meta.get("story_title", "")}"',
        f'export EPISODE_NUMBER="{meta.get("episode_number", "")}"',
        f'export EPISODE_ID="{ep_id}"',
        f'export PRIMARY_LOCALE="{primary}"',
        f'export LOCALES="{locales}"',
        f'export PROJECT_SLUG="{slug}"',
        f'export SERIES_GENRE="{meta.get("series_genre", "")}"',
        f'export GENERATION_SEED="{meta.get("generation_seed", "")}"',
        f'export RENDER_PROFILE="{meta.get("render_profile", "")}"',
        f'export STORY_FORMAT="{meta.get("story_format", "")}"',
        f'export PROJECT_DIR="projects/{slug}"',
        f'export EPISODE_DIR="projects/{slug}/episodes/{ep_id}"',
        f'export VOICE_CAST_FILE="projects/{slug}/VoiceCast.json"',
    ]
    return "\n".join(lines) + "\n"


# ── Confirmation block ────────────────────────────────────────────────────────

def print_confirmation(meta: dict, vars_path: Path) -> None:
    """Print the Stage 0 confirmation block to stdout (matches p_0.txt format)."""
    slug    = meta["project_slug"]
    ep_id   = meta["episode_id"]
    locales = str(meta.get("locales", "en"))
    primary = meta.get("primary_locale", "")
    if not primary:
        primary = locales.split(",")[0].strip()

    print("── Stage 0a complete ─────────────────────────────────────────────────")
    print(f"STORY_TITLE      : {meta.get('story_title', '')}")
    print(f"EPISODE_NUMBER   : {meta.get('episode_number', '')}")
    print(f"EPISODE_ID       : {ep_id}")
    print(f"PRIMARY_LOCALE   : {primary}")
    print(f"LOCALES          : {locales}")
    print(f"PROJECT_SLUG     : {slug}")
    print(f"SERIES_GENRE     : {meta.get('series_genre', '')}")
    print(f"GENERATION_SEED  : {meta.get('generation_seed', '')}")
    print(f"RENDER_PROFILE   : {meta.get('render_profile', '')}")
    print(f"STORY_FORMAT     : {meta.get('story_format', '')}")
    print(f"VOICE_CAST_FILE  : projects/{slug}/VoiceCast.json")
    print("─────────────────────────────────────────────────────────────────────")
    print(f"  Written: {vars_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"▶ {STAGE_LABEL_START}")

    if len(sys.argv) < 2:
        print("Usage: python gen_pipeline_vars.py <ep_dir>", file=sys.stderr)
        sys.exit(1)

    ep_dir = Path(sys.argv[1]).resolve()

    if not ep_dir.is_dir():
        print(f"[ERROR] ep_dir does not exist: {ep_dir}", file=sys.stderr)
        sys.exit(1)

    meta_path = ep_dir / "meta.json"
    if not meta_path.is_file():
        print(f"[ERROR] meta.json not found: {meta_path}", file=sys.stderr)
        sys.exit(1)

    # Load meta.json
    try:
        meta = load_json(meta_path)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[ERROR] Cannot read meta.json: {exc}", file=sys.stderr)
        sys.exit(1)

    # Validate required fields
    for field in ("project_slug", "episode_id"):
        if not meta.get(field):
            print(f"[ERROR] meta.json missing required field: '{field}'", file=sys.stderr)
            sys.exit(1)

    # Build and write pipeline_vars.sh
    content   = build_pipeline_vars(meta)
    vars_path = ep_dir / "pipeline_vars.sh"
    save_text(content, vars_path)

    print_confirmation(meta, vars_path)
    print(f"✓ {STAGE_LABEL_DONE}")


if __name__ == "__main__":
    main()
