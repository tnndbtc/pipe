#!/usr/bin/env python3
# =============================================================================
# gen_shotlist_scaffold.py — Build ShotList_scaffold.json for narration formats
# =============================================================================
#
# Generates a deterministic ShotList scaffold from Script.json.  Shot
# boundaries align with Script.json scene boundaries; the LLM (p_4_c.txt)
# fills all visual and audio creative fields and writes the final ShotList.json.
#
# Duration formula (from p_4.txt):
#   effective_wpm = 130 × (1 + azure_rate_decimal)
#   base_secs     = word_count / effective_wpm × 60
#   break_secs    = sum(pause_after_ms / 1000) if present, else n_lines × 0.6
#   duration_sec  = base_secs + break_secs + 2.0
#
# Hard limits applied after initial computation:
#   max 45 s → split shot at midpoint of vo_item_ids, new shot id suffixed "-b"
#   min  5 s → merge with next shot (or prev if last)
#
# Usage:
#   python3 code/http/gen_shotlist_scaffold.py <ep_dir>
#
# Reads:
#   <ep_dir>/pipeline_vars.sh        → PROJECT_SLUG, EPISODE_ID, STORY_FORMAT
#   <ep_dir>/Script.json
#   projects/<PROJECT_SLUG>/VoiceCast.json   (for azure_rate → effective_wpm)
#
# Writes:
#   <ep_dir>/ShotList_scaffold.json
#
# Requirements: stdlib only + jsonschema
# =============================================================================

from __future__ import annotations

import hashlib
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

DEFAULT_EFFECTIVE_WPM = 110.0
MAX_SHOT_DURATION_SEC = 45.0
MIN_SHOT_DURATION_SEC = 5.0


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
# azure_rate → effective_wpm
# ---------------------------------------------------------------------------
def parse_rate(rate_str: str) -> float:
    """Convert azure_rate string to decimal.  '0%'→0.0, '-35%'→-0.35"""
    s = rate_str.strip().replace("%", "")
    try:
        return float(s) / 100.0
    except ValueError:
        return 0.0


def get_effective_wpm(voicecast: dict | None, locale: str = "en") -> float:
    """
    Derive effective_wpm from VoiceCast.json narrator entry for the given locale.

    Formula: 130 × (1 + azure_rate_decimal)
    Fallback: DEFAULT_EFFECTIVE_WPM (110) when VoiceCast is unavailable or
    the narrator entry has no azure_rate.
    """
    if voicecast is None:
        return DEFAULT_EFFECTIVE_WPM

    for char in voicecast.get("characters", []):
        if char.get("character_id") == "narrator":
            locale_block = char.get(locale) or char.get("en", {})
            if not locale_block:
                break
            azure_rate = locale_block.get("azure_rate", "")
            if azure_rate:
                rate_dec = parse_rate(azure_rate)
                return round(130.0 * (1.0 + rate_dec), 4)
            break

    return DEFAULT_EFFECTIVE_WPM


# ---------------------------------------------------------------------------
# Duration formula
# ---------------------------------------------------------------------------
def compute_duration(
    vo_lines: list[str],
    effective_wpm: float,
    break_ms_list: list[int] | None = None,
) -> float:
    """
    Compute shot duration in seconds from VO lines.

    vo_lines      : list of text strings for all VO items in this shot
    effective_wpm : speaking rate in words per minute
    break_ms_list : list of pause_after_ms values per line (or None)

    If break_ms_list is provided and non-empty, break_secs = sum / 1000.
    Otherwise break_secs = n_lines × 0.6 (default inter-sentence pause).
    """
    if not vo_lines:
        return 2.0  # minimum shot with no VO

    word_count = sum(len(line.split()) for line in vo_lines)
    base_secs  = (word_count / effective_wpm) * 60.0

    if break_ms_list:
        break_secs = sum(ms / 1000.0 for ms in break_ms_list)
    else:
        break_secs = len(vo_lines) * 0.6

    return round(base_secs + break_secs + 2.0, 2)


# ---------------------------------------------------------------------------
# VO item ID assignment
# ---------------------------------------------------------------------------
def assign_vo_item_ids(scenes: list[dict]) -> dict[str, list[dict]]:
    """
    Assign sequential vo_item_ids to every dialogue action in Script.json.

    Returns {scene_id → [{"vo_id": str, "line": str, "pause_ms": int}, ...]}

    ID format: "vo-{scene_id}-{NNN}" (NNN zero-padded, starting at 001).
    The scene_id is taken verbatim from Script.json scenes[].scene_id.
    """
    result: dict[str, list[dict]] = {}
    for scene in scenes:
        scene_id = scene.get("scene_id", "")
        actions = scene.get("actions", [])
        vo_entries: list[dict] = []
        counter = 0
        for action in actions:
            if action.get("type") != "dialogue":
                continue
            counter += 1
            vo_id = f"vo-{scene_id}-{counter:03d}"
            vo_entries.append({
                "vo_id":    vo_id,
                "line":     action.get("line", ""),
                "pause_ms": action.get("pause_after_ms", 0),
            })
        result[scene_id] = vo_entries
    return result


# ---------------------------------------------------------------------------
# Shot factory
# ---------------------------------------------------------------------------
def make_shot(
    shot_id:       str,
    scene_id:      str,
    vo_entries:    list[dict],
    effective_wpm: float,
) -> dict:
    """
    Build one shot scaffold dict.

    All creative fields are left as __FILL__ strings for the LLM.
    duration_sec is computed deterministically from vo content.
    """
    vo_lines    = [e["line"] for e in vo_entries]
    pause_ms_list = [e["pause_ms"] for e in vo_entries if e.get("pause_ms")]
    vo_item_ids = [e["vo_id"] for e in vo_entries]
    vo_text     = " ".join(vo_lines) if vo_lines else None

    duration_sec = compute_duration(
        vo_lines,
        effective_wpm,
        pause_ms_list if pause_ms_list else None,
    )

    return {
        "shot_id":         shot_id,
        "scene_id":        scene_id,
        "duration_sec":    duration_sec,
        "characters":      [],
        "background_id":   "__FILL__: bg-<descriptive-location-slug>",
        "camera_framing":  "__FILL__: one of close-up/medium/wide/extreme-wide",
        "camera_movement": "__FILL__: one of static/slow_push/tracking/pan/tilt",
        "audio_intent": {
            "sfx_tags":     "__FILL__: list of sound effect tag strings",
            "music_mood":   "__FILL__: one-phrase music direction string or null",
            "vo_speaker_id": "narrator",
            "vo_text":      vo_text,
            "vo_item_ids":  vo_item_ids,
            "sfx_item_ids": "__FILL__: generate sfx-{shot_id}-001 etc based on your sfx_tags count",
            "music_item_id": "__FILL__: music-{shot_id} if music_mood is not null, else null",
        },
        # Store vo_entries on the shot temporarily for split/merge operations;
        # removed before final output
        "_vo_entries": vo_entries,
    }


# ---------------------------------------------------------------------------
# Split and merge helpers
# ---------------------------------------------------------------------------
def split_shot(shot: dict, effective_wpm: float) -> list[dict]:
    """
    Split a shot that exceeds MAX_SHOT_DURATION_SEC at the midpoint of
    vo_item_ids.  Returns two shots: the original (trimmed) and a new "-b" shot.
    """
    vo_entries = shot.get("_vo_entries", [])
    n = len(vo_entries)

    if n < 2:
        # Cannot split a single-line shot; return as-is
        return [shot]

    mid = n // 2
    entries_a = vo_entries[:mid]
    entries_b = vo_entries[mid:]

    shot_id_b = shot["shot_id"] + "-b"
    scene_id  = shot["scene_id"]

    shot_a = make_shot(shot["shot_id"], scene_id, entries_a, effective_wpm)
    shot_b = make_shot(shot_id_b,       scene_id, entries_b, effective_wpm)

    return [shot_a, shot_b]


def merge_shots(shot_a: dict, shot_b: dict, effective_wpm: float) -> dict:
    """
    Merge two shots into one, combining their vo_entries and recomputing
    duration.  The merged shot keeps shot_a's shot_id and scene_id.
    """
    combined_entries = shot_a.get("_vo_entries", []) + shot_b.get("_vo_entries", [])
    return make_shot(shot_a["shot_id"], shot_a["scene_id"], combined_entries, effective_wpm)


# ---------------------------------------------------------------------------
# Apply duration rules (split / merge)
# ---------------------------------------------------------------------------
def apply_duration_rules(shots: list[dict], effective_wpm: float) -> list[dict]:
    """
    Enforce MAX_SHOT_DURATION_SEC and MIN_SHOT_DURATION_SEC on a list of shots.

    Pass 1: split any shot > 45 s at its vo_item midpoint.
    Pass 2: merge any shot < 5 s forward (or backward if it is the last shot).
    Both passes are iterative until stable.
    """
    # Pass 1: split
    changed = True
    while changed:
        changed = False
        new_shots: list[dict] = []
        for shot in shots:
            if shot["duration_sec"] > MAX_SHOT_DURATION_SEC:
                parts = split_shot(shot, effective_wpm)
                new_shots.extend(parts)
                if len(parts) > 1:
                    changed = True
            else:
                new_shots.append(shot)
        shots = new_shots

    # Pass 2: merge short shots
    changed = True
    while changed:
        changed = False
        if len(shots) < 2:
            break
        new_shots = []
        i = 0
        while i < len(shots):
            shot = shots[i]
            if shot["duration_sec"] < MIN_SHOT_DURATION_SEC:
                if i + 1 < len(shots):
                    # Merge forward
                    merged = merge_shots(shot, shots[i + 1], effective_wpm)
                    new_shots.append(merged)
                    i += 2
                elif new_shots:
                    # Last shot: merge backward
                    prev = new_shots.pop()
                    merged = merge_shots(prev, shot, effective_wpm)
                    new_shots.append(merged)
                    i += 1
                else:
                    new_shots.append(shot)
                    i += 1
                changed = True
            else:
                new_shots.append(shot)
                i += 1
        shots = new_shots

    return shots


# ---------------------------------------------------------------------------
# timing_lock_hash
# ---------------------------------------------------------------------------
def compute_timing_lock_hash(shots: list[dict]) -> str:
    shot_ids = sorted(s["shot_id"] for s in shots)
    return hashlib.sha256(json.dumps(shot_ids).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Scaffold assembler
# ---------------------------------------------------------------------------
def build_scaffold(
    project_slug:  str,
    episode_id:    str,
    script:        dict,
    effective_wpm: float,
) -> dict:
    """
    Build the ShotList_scaffold.json document.

    One initial shot per Script.json scene; then split/merge rules applied.
    """
    scenes = script.get("scenes", [])
    vo_map = assign_vo_item_ids(scenes)

    # Initial shots: one per scene
    raw_shots: list[dict] = []
    for scene_idx, scene in enumerate(scenes):
        scene_id  = scene.get("scene_id", f"sc{scene_idx + 1:02d}")
        shot_num  = len(raw_shots) + 1
        shot_id   = f"{scene_id}-sh{shot_num:02d}"
        vo_entries = vo_map.get(scene_id, [])
        shot = make_shot(shot_id, scene_id, vo_entries, effective_wpm)
        raw_shots.append(shot)

    # Apply split/merge rules
    shots = apply_duration_rules(raw_shots, effective_wpm)

    # Reindex shot ids sequentially within each scene after splits/merges
    shots = _reindex_shot_ids(shots, scenes)

    # Compute timing lock hash
    timing_lock_hash = compute_timing_lock_hash(shots)

    # Compute total duration
    total_duration_sec = round(sum(s["duration_sec"] for s in shots), 2)

    # Strip internal _vo_entries helper key before output
    clean_shots = [_clean_shot(s) for s in shots]

    return {
        "schema_id":         "ShotList",
        "schema_version":    "1.0.0",
        "shotlist_id":       f"{project_slug}-{episode_id}",
        "script_ref":        f"{project_slug}-{episode_id}",
        "timing_lock_hash":  timing_lock_hash,
        "total_duration_sec": total_duration_sec,
        "shots":             clean_shots,
    }


def _reindex_shot_ids(shots: list[dict], scenes: list[dict]) -> list[dict]:
    """
    Re-assign shot_id values using a global sequential counter across all shots,
    reset per scene_id.  This ensures splits produce coherent ids like
    sc01-sh01, sc01-sh01-b (from split), sc02-sh02.

    Strategy: maintain a per-scene counter; use the raw shot_id but normalise
    any "-b" suffixed split shots so they are simply left with their existing
    ids (the split already added "-b").  This function only ensures the initial
    shots are numbered sequentially without gaps.
    """
    # Build ordered list of scene_ids from script for ordering reference
    scene_order = [s.get("scene_id", "") for s in scenes]

    # Group shots by scene preserving global order
    from collections import OrderedDict
    by_scene: dict[str, list[dict]] = OrderedDict()
    for shot in shots:
        sid = shot.get("scene_id", "")
        by_scene.setdefault(sid, []).append(shot)

    result: list[dict] = []
    global_idx = 0
    for scene_id in scene_order:
        scene_shots = by_scene.get(scene_id, [])
        for local_idx, shot in enumerate(scene_shots):
            global_idx += 1
            # Only rename the primary shot (index 0) within the scene.
            # Split shots already have a deterministic suffix from split_shot().
            if local_idx == 0:
                new_id = f"{scene_id}-sh{global_idx:02d}"
                shot = dict(shot)
                shot["shot_id"] = new_id
                # Update vo_item_ids that reference the old shot_id in their
                # audio_intent (they don't — vo ids use scene_id not shot_id,
                # so this is safe to skip)
            result.append(shot)

    return result


def _clean_shot(shot: dict) -> dict:
    """Remove internal helper keys before writing output."""
    return {k: v for k, v in shot.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Schema validation (best-effort — scaffolds contain __FILL__ strings)
# ---------------------------------------------------------------------------
def validate_scaffold_structure(scaffold: dict) -> list[str]:
    """
    Validate required top-level fields only.  __FILL__ strings inside shots
    are intentional and not validated against the full schema.
    """
    errors: list[str] = []
    required = ["schema_id", "schema_version", "shotlist_id", "script_ref",
                 "timing_lock_hash", "shots"]
    for field in required:
        if field not in scaffold:
            errors.append(f"Missing required field: {field}")
    if scaffold.get("schema_version") != "1.0.0":
        errors.append(f"schema_version mismatch: {scaffold.get('schema_version')}")
    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python3 code/http/gen_shotlist_scaffold.py <ep_dir>",
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
    story_format = pvars.get("STORY_FORMAT", "continuous_narration")

    if not project_slug or not episode_id:
        print("[ERROR] pipeline_vars.sh must export PROJECT_SLUG and EPISODE_ID",
              file=sys.stderr)
        return 1

    print(f"▶ gen_shotlist_scaffold.py — {episode_id}")
    print(f"  project  : {project_slug}")
    print(f"  format   : {story_format}")

    # Load Script.json
    script_path = ep_dir / "Script.json"
    if not script_path.is_file():
        print(f"[ERROR] Script.json not found: {script_path}", file=sys.stderr)
        return 1
    script = load_json(script_path)

    # Load VoiceCast.json (optional)
    vc_path = PIPE_DIR / "projects" / project_slug / "VoiceCast.json"
    voicecast: dict | None = None
    if vc_path.is_file():
        voicecast = load_json(vc_path)
    else:
        print(f"  [WARN] VoiceCast.json not found at {vc_path} — using default wpm={DEFAULT_EFFECTIVE_WPM}")

    # Compute effective_wpm from narrator azure_rate
    effective_wpm = get_effective_wpm(voicecast, locale="en")
    print(f"  effective_wpm : {effective_wpm:.1f}  "
          f"(narrator azure_rate from VoiceCast.json)")

    # Count total dialogue actions for completeness verification
    total_dialogue_actions = sum(
        1
        for scene in script.get("scenes", [])
        for action in scene.get("actions", [])
        if action.get("type") == "dialogue"
    )

    # Build scaffold
    scaffold = build_scaffold(project_slug, episode_id, script, effective_wpm)

    # Verify VO completeness: every dialogue action must appear in exactly one shot
    all_vo_ids: list[str] = []
    for shot in scaffold["shots"]:
        all_vo_ids.extend(shot["audio_intent"].get("vo_item_ids", []))

    n_vo_ids  = len(all_vo_ids)
    n_unique  = len(set(all_vo_ids))
    if n_vo_ids != total_dialogue_actions:
        print(f"  [WARN] VO count mismatch: scaffold has {n_vo_ids} IDs, "
              f"Script has {total_dialogue_actions} dialogue lines",
              file=sys.stderr)
    if n_unique != n_vo_ids:
        print(f"  [WARN] {n_vo_ids - n_unique} duplicate vo_item_ids detected",
              file=sys.stderr)

    # Validate structure
    errors = validate_scaffold_structure(scaffold)
    if errors:
        for e in errors:
            print(f"  [ERROR] {e}", file=sys.stderr)
        return 1

    # Write output
    out_path = ep_dir / "ShotList_scaffold.json"
    save_json(scaffold, out_path)

    n_scenes = len(script.get("scenes", []))
    n_shots  = len(scaffold["shots"])
    total_dur = scaffold["total_duration_sec"]

    print(f"  Scenes: {n_scenes}, Shots (after split/merge): {n_shots}, "
          f"Total duration: {total_dur:.1f}s")
    print(f"  VO items assigned: {n_vo_ids}/{total_dialogue_actions}")
    print(f"✓ ShotList_scaffold.json written — LLM will fill visual/audio fields")

    return 0


if __name__ == "__main__":
    sys.exit(main())
