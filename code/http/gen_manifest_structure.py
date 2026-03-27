#!/usr/bin/env python3
# =============================================================================
# gen_manifest_structure.py — Build AssetManifest.shared.json scaffold
# =============================================================================
#
# Generates a deterministic scaffold for AssetManifest.shared.json from
# ShotList.json + Script.json + VoiceCast.json.  All creative fields are left
# as __FILL__ strings so the LLM (p_5_c.txt) can complete them.
#
# Usage:
#   python3 code/http/gen_manifest_structure.py <ep_dir>
#
# Reads:
#   <ep_dir>/pipeline_vars.sh        → PROJECT_SLUG, EPISODE_ID, STORY_FORMAT
#   <ep_dir>/ShotList.json
#   <ep_dir>/Script.json
#   projects/<PROJECT_SLUG>/VoiceCast.json
#
# Writes:
#   <ep_dir>/AssetManifest.shared.json
#
# Requirements: stdlib only + jsonschema
# =============================================================================

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

try:
    import jsonschema
    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False

PIPE_DIR = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Camera movement → motion_level mapping
# ---------------------------------------------------------------------------
_MOTION_LEVEL: dict[str, str] = {
    "static":    "none",
    "slow_push": "low",
    "slow_zoom": "low",
    "tracking":  "medium",
    "pan":       "medium",
    "tilt":      "medium",
    "handheld":  "high",
    "dynamic":   "high",
}


def camera_to_motion_level(camera_movement: str) -> str:
    """Map camera_movement string to motion_level enum value."""
    if not camera_movement:
        return "none"
    lower = camera_movement.lower().strip()
    # Direct key lookup first
    if lower in _MOTION_LEVEL:
        return _MOTION_LEVEL[lower]
    # Substring scan for partial matches (e.g. "slow pan left to right")
    for key, level in _MOTION_LEVEL.items():
        if key in lower:
            return level
    return "none"


# ---------------------------------------------------------------------------
# Shot position → cinematic_role mapping
# ---------------------------------------------------------------------------
def compute_cinematic_roles(shots: list[dict]) -> dict[str, str]:
    """
    Return {background_id → cinematic_role} based on each bg_id's first
    appearance in the shot list.

    Rules (applied per-scene for each unique background_id):
      - first shot of a scene  → "establish"
      - only shot in a scene   → "hold"
      - last shot of scene AND scene has > 1 shot → "transition"
      - others                 → "atmosphere"

    When the same background_id appears in multiple scenes, the role assigned
    is the role from its *first* occurrence (earliest shot index).
    """
    # Group shots by scene_id while preserving order
    from collections import OrderedDict
    scenes: dict[str, list[dict]] = OrderedDict()
    for shot in shots:
        sid = shot.get("scene_id", "")
        scenes.setdefault(sid, []).append(shot)

    # bg_id → role (first assignment wins)
    roles: dict[str, str] = {}
    for scene_shots in scenes.values():
        n = len(scene_shots)
        for idx, shot in enumerate(scene_shots):
            bg_id = shot.get("background_id")
            if not bg_id or bg_id in roles:
                continue
            if n == 1:
                role = "hold"
            elif idx == 0:
                role = "establish"
            elif idx == n - 1:
                role = "transition"
            else:
                role = "atmosphere"
            roles[bg_id] = role

    return roles


# ---------------------------------------------------------------------------
# pipeline_vars.sh parser
# ---------------------------------------------------------------------------
def load_pipeline_vars(ep_dir: Path) -> dict[str, str]:
    """Parse pipeline_vars.sh and return a dict of key → value."""
    path = ep_dir / "pipeline_vars.sh"
    if not path.is_file():
        raise FileNotFoundError(f"pipeline_vars.sh not found: {path}")
    vars_: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        m = re.match(r'^export\s+([A-Z_]+)="(.*)"$', line)
        if m:
            vars_[m.group(1)] = m.group(2)
    return vars_


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(doc: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ---------------------------------------------------------------------------
# VoiceCast helpers
# ---------------------------------------------------------------------------
def load_cast_genders(script: dict) -> dict[str, str]:
    """Return {character_id → gender} from Script.json cast[]."""
    return {c["character_id"]: c.get("gender", "neutral")
            for c in script.get("cast", [])}


# ---------------------------------------------------------------------------
# Backgrounds builder
# ---------------------------------------------------------------------------
def build_backgrounds(shots: list[dict]) -> list[dict]:
    """
    Build the backgrounds[] array.

    One entry per unique background_id encountered in shots[].  Deterministic
    fields are filled; creative fields are left as __FILL__ strings.
    """
    seen: dict[str, dict] = {}  # bg_id → entry (preserves insertion order)
    cinematic_roles = compute_cinematic_roles(shots)

    for shot in shots:
        bg_id = shot.get("background_id")
        if not bg_id or bg_id in seen:
            continue

        camera_movement = shot.get("camera_movement", "")
        motion_level = camera_to_motion_level(camera_movement)
        cinematic_role = cinematic_roles.get(bg_id, "atmosphere")

        entry = {
            "asset_id":        bg_id,
            "type":            "background",
            "license_type":    "proprietary_cleared",
            "motion_level":    motion_level,
            "cinematic_role":  cinematic_role,
            "search_filters":  {
                "orientation": "landscape",
                "media_type":  "photo",
                "min_width":   1920,
                "min_height":  1080,
            },
            # Creative __FILL__ fields — LLM completes these
            "ai_prompt":            "__FILL__: cinematic visual description for AI image generation",
            "ai_prompt_variations": "__FILL__: list of exactly 2 variation prompt strings",
            "search_prompt":     "__FILL__: brief keyword summary for stock photo search",
            "search_queries":    "__FILL__: list of 2-4 diverse search phrase strings",
            "scoring_hints":     "__FILL__: dict with subjects, environment, style, motion keys",
            "include_keywords":  "__FILL__: list of hard-include search terms",
            "exclude_keywords":  "__FILL__: list of hard-exclude search terms",
            "continuity_hints":  "__FILL__: dict with group_id and palette_target",
            "lighting":          "__FILL__: one of golden_hour/overcast/harsh_noon/dim/night/dramatic",
        }
        seen[bg_id] = entry

    return list(seen.values())


# ---------------------------------------------------------------------------
# Character packs builder
# ---------------------------------------------------------------------------
def build_character_packs(
    shots: list[dict],
    cast_genders: dict[str, str],
) -> list[dict]:
    """
    Build character_packs[] for non-narrator characters.

    For narration formats this list is typically empty because shots have
    characters: [].  For episodic/illustrated_narration formats the character
    ids are collected from all shots.
    """
    seen: set[str] = set()
    packs: list[dict] = []

    for shot in shots:
        for char in shot.get("characters", []):
            cid = char.get("character_id", "")
            if not cid or cid == "narrator" or cid in seen:
                continue
            seen.add(cid)
            gender = cast_genders.get(cid, "neutral")
            packs.append({
                "pack_id":      f"pack-{cid}",
                "type":         "character_pack",
                "gender":       gender,
                "license_type": "proprietary_cleared",
                "search_filters": {
                    "orientation": "portrait",
                    "media_type":  "photo",
                    "min_width":   512,
                    "min_height":  512,
                },
            })

    return packs


# ---------------------------------------------------------------------------
# Manifest assembler
# ---------------------------------------------------------------------------
def build_manifest(
    project_slug: str,
    episode_id:   str,
    shots:        list[dict],
    cast_genders: dict[str, str],
) -> dict:
    """Assemble the full AssetManifest.shared.json scaffold."""
    backgrounds     = build_backgrounds(shots)
    character_packs = build_character_packs(shots, cast_genders)

    manifest: dict = {
        "schema_id":      "AssetManifest",
        "schema_version": "1.0.0",
        "manifest_id":    f"{project_slug}-{episode_id}-shared",
        "project_id":     project_slug,
        "episode_id":     episode_id,
        "shotlist_ref":   "ShotList.json",
        "locale_scope":   "shared",
        "vo_items":       [],
        "character_packs": character_packs,
        "backgrounds":    backgrounds,
        "element_images":  "__FILL__: array of per-shot element image objects — Stage 5-C creates from scratch",
    }

    return manifest


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------
def validate_manifest(manifest: dict) -> list[str]:
    """Validate manifest against VOPlan.v1.json schema.

    Returns a list of error strings (empty = valid).
    Only validates the deterministic / non-__FILL__ portions to avoid false
    failures on the intentional placeholder strings.
    """
    if not _HAS_JSONSCHEMA:
        return ["jsonschema not installed — skipping validation"]

    schema_path = PIPE_DIR / "contracts" / "schemas" / "VOPlan.v1.json"
    if not schema_path.is_file():
        return [f"Schema not found: {schema_path}"]

    schema = load_json(schema_path)

    # Validate only required top-level structure; skip items that contain
    # __FILL__ strings (those are intentional scaffolding placeholders).
    probe = {
        "schema_id":      manifest["schema_id"],
        "schema_version": manifest["schema_version"],
        "manifest_id":    manifest["manifest_id"],
        "project_id":     manifest["project_id"],
        "episode_id":     manifest["episode_id"],
        "shotlist_ref":   manifest["shotlist_ref"],
        "character_packs": [],
        "backgrounds":    [],
        "vo_items":       [],
    }

    errors: list[str] = []
    validator = jsonschema.Draft7Validator(schema)
    for err in validator.iter_errors(probe):
        errors.append(err.message)
    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python3 code/http/gen_manifest_structure.py <ep_dir>",
              file=sys.stderr)
        return 1

    ep_dir = Path(sys.argv[1]).resolve()
    if not ep_dir.is_dir():
        print(f"[ERROR] ep_dir does not exist: {ep_dir}", file=sys.stderr)
        return 1

    # Load pipeline vars
    try:
        pvars = load_pipeline_vars(ep_dir)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    project_slug = pvars.get("PROJECT_SLUG", "")
    episode_id   = pvars.get("EPISODE_ID", "")
    story_format = pvars.get("STORY_FORMAT", "episodic")

    if not project_slug or not episode_id:
        print("[ERROR] pipeline_vars.sh must export PROJECT_SLUG and EPISODE_ID",
              file=sys.stderr)
        return 1

    print(f"▶ gen_manifest_structure.py — {episode_id}")
    print(f"  project  : {project_slug}")
    print(f"  format   : {story_format}")

    # Load ShotList
    shotlist_path = ep_dir / "ShotList.json"
    if not shotlist_path.is_file():
        print(f"[ERROR] ShotList.json not found: {shotlist_path}", file=sys.stderr)
        return 1
    shotlist = load_json(shotlist_path)
    shots = shotlist.get("shots", [])

    # Load Script
    script_path = ep_dir / "Script.json"
    if not script_path.is_file():
        print(f"[ERROR] Script.json not found: {script_path}", file=sys.stderr)
        return 1
    script = load_json(script_path)
    cast_genders = load_cast_genders(script)

    # Load VoiceCast (optional — used for gender fallback in edge cases)
    vc_path = PIPE_DIR / "projects" / project_slug / "VoiceCast.json"
    if vc_path.is_file():
        voicecast = load_json(vc_path)
        for char in voicecast.get("characters", []):
            cid = char.get("character_id", "")
            if cid and cid not in cast_genders:
                cast_genders[cid] = char.get("gender", "neutral")
    else:
        print(f"  [WARN] VoiceCast.json not found at {vc_path}")

    # Build manifest scaffold
    manifest = build_manifest(project_slug, episode_id, shots, cast_genders)

    # Validate (structural check only)
    errors = validate_manifest(manifest)
    if errors:
        print("[WARN] Validation issues (may be expected for scaffold):")
        for e in errors:
            print(f"  - {e}")

    # Write output
    out_path = ep_dir / "AssetManifest.shared.json"
    save_json(manifest, out_path)

    n_bg       = len(manifest["backgrounds"])
    n_packs    = len(manifest["character_packs"])
    n_elements = len(manifest.get("element_images", [])) if isinstance(manifest.get("element_images"), list) else "FILL"

    print(f"  backgrounds     : {n_bg}")
    print(f"  character_packs : {n_packs}")
    print(f"  element_images  : {n_elements}")
    print(f"✓ AssetManifest.shared.json scaffold written ({n_bg} backgrounds)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
