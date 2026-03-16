#!/usr/bin/env python3
# =============================================================================
# gen_vo_manifest.py — Generate AssetManifest.{locale}.json (VO only)
#
# Deterministically builds the locale manifest (locale_scope="locale") that
# contains only vo_items[], derived from Script.json + ShotList.json +
# VoiceCast.json.  Does NOT touch the shared manifest (character_packs,
# backgrounds, sfx_items, music_items).
#
# Usage:
#   python3 gen_vo_manifest.py \
#       --script     projects/slug/ep/Script.json \
#       --shotlist   projects/slug/ep/ShotList.json \
#       --voice-cast projects/slug/VoiceCast.json \
#       --locale     en \
#       --out        projects/slug/ep/AssetManifest.en.json
#
#   # Positional ep_dir — derive Script/ShotList from it, --locale required:
#   python3 gen_vo_manifest.py projects/slug/ep \
#       --voice-cast projects/slug/VoiceCast.json \
#       --locale en
#
# Requirements: stdlib + jsonschema
# =============================================================================

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

try:
    import jsonschema
except ImportError:
    print("[ERROR] jsonschema not installed. Run: pip install jsonschema", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Schema path
# ---------------------------------------------------------------------------
PIPE_DIR = Path(__file__).resolve().parent.parent.parent
SCHEMA_PATH = PIPE_DIR / "contracts" / "schemas" / "AssetManifest.v1.json"

# ---------------------------------------------------------------------------
# Style fallback chains (from p_5.txt)
# ---------------------------------------------------------------------------
STYLE_FALLBACK: dict[str, list[str]] = {
    "angry":    ["angry",    "shouting",   "unfriendly", "serious"],
    "fearful":  ["fearful",  "terrified",  "whispering", "sad"],
    "cheerful": ["cheerful", "excited",    "friendly",   "hopeful"],
    "sad":      ["sad",      "disgruntled"],
    "excited":  ["excited",  "cheerful",   "friendly"],
    "serious":  ["serious",  "calm"],
}

# Narrator-preferred styles in priority order
NARRATOR_PREFERRED = ["narration-professional", "newscast"]

# ---------------------------------------------------------------------------
# CJK unicode ranges for word-count estimation
# ---------------------------------------------------------------------------
CJK_RANGES = [
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Extension A
    (0x20000, 0x2A6DF), # CJK Extension B
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0x2E80, 0x2EFF),   # CJK Radicals Supplement
    (0x3000, 0x303F),   # CJK Symbols and Punctuation
    (0x31C0, 0x31EF),   # CJK Strokes
    (0xFE30, 0xFE4F),   # CJK Compatibility Forms
    (0x3040, 0x309F),   # Hiragana
    (0x30A0, 0x30FF),   # Katakana
    (0xAC00, 0xD7AF),   # Hangul Syllables
]


def _is_cjk(text: str) -> bool:
    """Return True if the majority of non-space characters are CJK/kana/hangul."""
    count = 0
    total = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        cp = ord(ch)
        for lo, hi in CJK_RANGES:
            if lo <= cp <= hi:
                count += 1
                break
    return total > 0 and (count / total) >= 0.4


# ---------------------------------------------------------------------------
# I/O helpers
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

def build_cast_map(voicecast: dict) -> dict[str, dict]:
    """Return {character_id → character_dict} for quick lookup."""
    return {c["character_id"]: c for c in voicecast.get("characters", [])}


def get_locale_entry(character: dict, locale: str) -> dict | None:
    """
    Look up the locale block from a character dict.

    VoiceCast stores locale blocks as flat keys on the character:
        { "character_id": "...", "en": {...}, "zh-Hans": {...} }
    """
    return character.get(locale)


# ---------------------------------------------------------------------------
# Style selection
# ---------------------------------------------------------------------------

def _resolve_style(
    desired_style: str | None,
    available_styles: list[str],
    speaker_id: str,
) -> str | None:
    """
    Select the best azure_style for a VO line.

    Priority:
      1. desired_style (from action field) if in available_styles
      2. Walk desired_style through fallback chains
      3. For "narrator" speaker: prefer narration-professional / newscast
      4. None if nothing matches
    """
    if not available_styles:
        return None

    avail = set(available_styles)

    # Narrator preference (applied regardless of desired_style)
    if speaker_id == "narrator":
        for pref in NARRATOR_PREFERRED:
            if pref in avail:
                return pref

    # If a specific style was requested, try it and its fallback chain
    if desired_style:
        # Direct match
        if desired_style in avail:
            return desired_style
        # Walk the fallback chain for the desired emotion
        chain = STYLE_FALLBACK.get(desired_style, [desired_style])
        for candidate in chain:
            if candidate in avail:
                return candidate

    # No match
    return None


# ---------------------------------------------------------------------------
# Timing estimation
# ---------------------------------------------------------------------------

def _parse_rate_decimal(azure_rate: str | None) -> float:
    """Parse azure_rate like '-35%' or '+15%' into a decimal (-0.35 or +0.15)."""
    if not azure_rate:
        return 0.0
    s = azure_rate.strip()
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except ValueError:
            return 0.0
    # Numeric (shouldn't appear here but handle gracefully)
    try:
        val = float(s)
        # If it looks like a multiplier (e.g. 0.86) convert to delta
        if 0.1 <= val <= 3.0:
            return val - 1.0
        return val / 100.0
    except ValueError:
        return 0.0


def estimate_duration_sec(
    text: str,
    azure_rate: str | None,
    azure_break_ms: int,
) -> float:
    """
    Estimate VO duration in seconds.

    Formula:
        effective_wpm = 130 × (1 + rate_decimal)
        word_count:   len(text.split()) for Latin; len(text) // 2 for CJK
        duration_sec = (word_count / effective_wpm) × 60 + (azure_break_ms / 1000)
    """
    rate_decimal = _parse_rate_decimal(azure_rate)
    effective_wpm = 130.0 * (1.0 + rate_decimal)
    if effective_wpm <= 0:
        effective_wpm = 65.0  # safety floor

    if _is_cjk(text):
        word_count = len(text) // 2
    else:
        word_count = len(text.split())

    speaking_sec = (word_count / effective_wpm) * 60.0
    break_sec = azure_break_ms / 1000.0
    return round(speaking_sec + break_sec, 3)


# ---------------------------------------------------------------------------
# Step 1 — Build vo_item_id → shot_id mapping
# ---------------------------------------------------------------------------

def build_vo_to_shot_map(shotlist: dict) -> dict[str, str]:
    """
    Return {vo_item_id → shot_id} by iterating all shots.

    Preserves first-seen assignment if an id appears in multiple shots
    (should not happen in valid ShotList, but be defensive).
    """
    mapping: dict[str, str] = {}
    for shot in shotlist.get("shots", []):
        shot_id = shot["shot_id"]
        for vid in shot.get("audio_intent", {}).get("vo_item_ids", []):
            if vid not in mapping:
                mapping[vid] = shot_id
    return mapping


# ---------------------------------------------------------------------------
# Step 2 — Collect dialogue lines from Script.json
# ---------------------------------------------------------------------------

def collect_dialogue_lines(script: dict) -> list[dict]:
    """
    Return an ordered flat list of all dialogue lines across all scenes.

    Each entry:
        {
            "scene_id": str,
            "speaker_id": str,
            "line": str,
            "style": str | None,       # azure style hint from action (if present)
            "sentence_id": str | None,
        }

    Handles two Script.json formats:
      A) scene.actions[] where action.type == "dialogue"  (trend / rivers-whisper style)
      B) scene.dialogue[] — separate array at scene level  (pharaoh style)
    """
    lines: list[dict] = []
    for scene in script.get("scenes", []):
        scene_id = scene.get("scene_id", "")

        # Format A: actions array with type field
        actions = scene.get("actions", [])
        has_typed_actions = any("type" in a for a in actions)

        if has_typed_actions:
            for action in actions:
                if action.get("type") == "dialogue":
                    lines.append({
                        "scene_id": scene_id,
                        "speaker_id": action.get("speaker_id", ""),
                        "line": action.get("line", ""),
                        "style": action.get("azure_style") or action.get("style"),
                        "sentence_id": action.get("sentence_id"),
                    })
        else:
            # Format B: separate dialogue array (no type field on actions)
            for dlg in scene.get("dialogue", []):
                lines.append({
                    "scene_id": scene_id,
                    "speaker_id": dlg.get("speaker_id", ""),
                    "line": dlg.get("line", ""),
                    "style": dlg.get("azure_style") or dlg.get("style"),
                    "sentence_id": dlg.get("sentence_id"),
                })

    return lines


# ---------------------------------------------------------------------------
# Step 3 — Zip vo_item_ids with dialogue lines
# ---------------------------------------------------------------------------

def build_vo_tuples(
    shotlist: dict,
    dialogue_lines: list[dict],
) -> list[dict]:
    """
    Flatten vo_item_ids from all shots in order, then zip 1:1 with
    dialogue_lines collected in Script order.

    Returns a list of dicts:
        {
            "vo_item_id": str,
            "shot_id": str,
            "scene_id": str,
            "speaker_id": str,
            "text": str,
            "style": str | None,
        }
    """
    # Collect ordered vo_item_ids from ShotList
    ordered_vo_ids: list[tuple[str, str]] = []  # (vo_item_id, shot_id)
    for shot in shotlist.get("shots", []):
        shot_id = shot["shot_id"]
        for vid in shot.get("audio_intent", {}).get("vo_item_ids", []):
            ordered_vo_ids.append((vid, shot_id))

    n_ids = len(ordered_vo_ids)
    n_lines = len(dialogue_lines)

    if n_ids != n_lines:
        print(
            f"  [WARN] vo_item_ids count ({n_ids}) != dialogue lines count ({n_lines}). "
            "Zipping up to min — output may be incomplete.",
            file=sys.stderr,
        )

    tuples: list[dict] = []
    for i, ((vid, shot_id), dlg) in enumerate(
        zip(ordered_vo_ids, dialogue_lines)
    ):
        tuples.append({
            "vo_item_id": vid,
            "shot_id": shot_id,
            "scene_id": dlg["scene_id"],
            "speaker_id": dlg["speaker_id"],
            "text": dlg["line"],
            "style": dlg.get("style"),
        })

    return tuples


# ---------------------------------------------------------------------------
# Step 4-5 — Build a single vo_item entry
# ---------------------------------------------------------------------------

def build_vo_item(
    vo_tuple: dict,
    cast_map: dict[str, dict],
    locale: str,
) -> dict:
    """
    Build one vo_item dict ready for the manifest.

    Applies style selection, timing estimation, and VoiceCast lookups.
    """
    vid = vo_tuple["vo_item_id"]
    speaker_id = vo_tuple["speaker_id"]
    text = vo_tuple["text"]
    desired_style = vo_tuple.get("style")

    # --- VoiceCast lookup ---
    character = cast_map.get(speaker_id)
    locale_entry: dict | None = None
    has_voicecast = False

    if character is not None:
        locale_entry = get_locale_entry(character, locale)
        if locale_entry is not None:
            has_voicecast = True

    if not has_voicecast:
        print(
            f"  [WARN] No VoiceCast entry for '{speaker_id}' locale='{locale}' — "
            "azure_* fields will be omitted",
            file=sys.stderr,
        )

    # --- Style selection ---
    available_styles: list[str] = []
    default_style: str | None = None
    azure_style_degree: float | None = None
    azure_rate: str | None = None
    azure_pitch: str | None = None
    azure_break_ms: int = 0
    azure_voice: str | None = None

    if locale_entry:
        available_styles = locale_entry.get("available_styles") or []
        default_style = locale_entry.get("azure_style")
        azure_voice = locale_entry.get("azure_voice")
        # Read numeric/string values with type-safe fallbacks
        _degree = locale_entry.get("azure_style_degree")
        azure_style_degree = float(_degree) if _degree is not None else 1.5
        azure_rate = locale_entry.get("azure_rate")
        azure_pitch = locale_entry.get("azure_pitch")
        _brk = locale_entry.get("azure_break_ms")
        azure_break_ms = int(_brk) if _brk is not None else 0

    # Determine the style to use:
    # 1. Try desired_style (from action) if in available_styles
    # 2. Try default_style from VoiceCast
    # 3. Run fallback chains / narrator preference
    resolved_style: str | None = None
    if has_voicecast:
        # Attempt desired_style first
        if desired_style and desired_style in set(available_styles):
            resolved_style = desired_style
        else:
            # Merge desired and default for fallback resolution
            search_style = desired_style or default_style
            resolved_style = _resolve_style(search_style, available_styles, speaker_id)
            # If still None and default_style is available, accept default
            if resolved_style is None and default_style and default_style in set(available_styles):
                resolved_style = default_style

    # Step 6 — Whisper special case: zero break_ms
    effective_break_ms = azure_break_ms
    if resolved_style == "whispering":
        effective_break_ms = 0

    # --- pace label derived from azure_rate ---
    def _rate_to_pace(rate: str | None) -> str:
        if not rate:
            return "normal"
        r = _parse_rate_decimal(rate)
        if r <= -0.15:
            return "slow"
        if r >= 0.15:
            return "fast"
        return "normal"

    pace = _rate_to_pace(azure_rate)

    # --- voice_style label: gender + role ---
    voice_style = "neutral"
    if character is not None:
        gender = character.get("gender", "neutral")
        personality = character.get("personality") or character.get("role") or ""
        if personality:
            voice_style = f"{gender} {personality}"
        else:
            voice_style = gender

    # --- Build tts_prompt ---
    tts_prompt: dict = {
        "voice_style": voice_style,
        "emotion": resolved_style or "neutral",
        "pace": pace,
        "locale": locale,
    }
    if azure_voice:
        tts_prompt["azure_voice"] = azure_voice
    if resolved_style is not None:
        tts_prompt["azure_style"] = resolved_style
    if azure_style_degree is not None and has_voicecast:
        tts_prompt["azure_style_degree"] = azure_style_degree
    if azure_rate is not None and has_voicecast:
        tts_prompt["azure_rate"] = azure_rate
    if azure_pitch is not None and has_voicecast:
        tts_prompt["azure_pitch"] = azure_pitch
    if has_voicecast:
        tts_prompt["azure_break_ms"] = effective_break_ms

    # --- Estimate duration ---
    estimated_duration_sec = estimate_duration_sec(
        text, azure_rate, effective_break_ms
    )

    # --- Locale/script language mismatch warning ---
    # CJK text in a non-CJK locale (e.g. Chinese in "en") means the Script.json
    # line was never translated.  Warn loudly so the user can fix it in the VO tab.
    _CJK_LOCALE_PREFIXES = {"zh", "ja", "ko"}
    locale_root = locale.lower().split("-")[0]
    if locale_root not in _CJK_LOCALE_PREFIXES and _is_cjk(text):
        print(
            f"  [WARN] {vid}: CJK text in non-CJK locale '{locale}' — "
            f"translate this item in the VO tab before synthesis: "
            f"{text[:50]!r}",
            file=sys.stderr,
        )

    # --- Assemble item ---
    item: dict = {
        "item_id": vid,
        "speaker_id": speaker_id,
        "text": text,
        "license_type": "proprietary_cleared",
        "tts_prompt": tts_prompt,
        "estimated_duration_sec": estimated_duration_sec,
    }

    return item


# ---------------------------------------------------------------------------
# Step 6 — Build the manifest document
# ---------------------------------------------------------------------------

def derive_ids(script: dict, shotlist: dict) -> tuple[str, str, str]:
    """Return (project_id, episode_id, shotlist_ref) from source files."""
    project_id = script.get("project_id", shotlist.get("episode_id", ""))
    episode_id = script.get("episode", {}).get("id") or shotlist.get("episode_id", "")
    # Fallback: derive episode_id from script_id
    if not episode_id:
        script_id = script.get("script_id", "")
        # e.g. "the-pharaoh-who-defied-death-s01e02" → "s01e02"
        import re
        m = re.search(r"(s\d{2}e\d{2})", script_id)
        if m:
            episode_id = m.group(1)
    shotlist_ref = shotlist.get("shotlist_id") or "ShotList.json"
    return project_id, episode_id, shotlist_ref


def build_manifest(
    script: dict,
    shotlist: dict,
    vo_items: list[dict],
    locale: str,
) -> dict:
    """Assemble the top-level locale manifest dict."""
    project_id, episode_id, shotlist_ref = derive_ids(script, shotlist)

    manifest_id = f"{project_id}-{episode_id}-{locale}-manifest"

    return {
        "schema_id": "AssetManifest",
        "schema_version": "1.0.0",
        "manifest_id": manifest_id,
        "project_id": project_id,
        "episode_id": episode_id,
        "locale": locale,
        "locale_scope": "locale",
        "shared_ref": "AssetManifest.shared.json",
        "shotlist_ref": shotlist_ref,
        "character_packs": [],
        "backgrounds": [],
        "background_overrides": [],
        "vo_items": vo_items,
    }


# ---------------------------------------------------------------------------
# Step 7 — Validate
# ---------------------------------------------------------------------------

def validate_manifest(manifest: dict, schema: dict) -> None:
    """Raise jsonschema.ValidationError if invalid."""
    jsonschema.validate(instance=manifest, schema=schema)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate AssetManifest.{locale}.json (VO items) from "
            "Script.json + ShotList.json + VoiceCast.json."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explicit paths
  python3 gen_vo_manifest.py \\
      --script    projects/slug/ep/Script.json \\
      --shotlist  projects/slug/ep/ShotList.json \\
      --voice-cast projects/slug/VoiceCast.json \\
      --locale    en \\
      --out       projects/slug/ep/AssetManifest.en.json

  # Positional ep_dir (Script.json and ShotList.json auto-derived)
  python3 gen_vo_manifest.py projects/slug/ep \\
      --voice-cast projects/slug/VoiceCast.json \\
      --locale en
""",
    )
    p.add_argument(
        "ep_dir",
        nargs="?",
        default=None,
        metavar="EP_DIR",
        help=(
            "Episode directory. When provided, --script and --shotlist default "
            "to EP_DIR/Script.json and EP_DIR/ShotList.json respectively."
        ),
    )
    p.add_argument(
        "--script",
        default=None,
        metavar="PATH",
        help="Path to Script.json.",
    )
    p.add_argument(
        "--shotlist",
        default=None,
        metavar="PATH",
        help="Path to ShotList.json.",
    )
    p.add_argument(
        "--voice-cast",
        default=None,
        metavar="PATH",
        help=(
            "Path to VoiceCast.json. When omitted, searched as "
            "../../VoiceCast.json relative to the episode directory."
        ),
    )
    p.add_argument(
        "--locale",
        default="en",
        metavar="LOCALE",
        help="BCP-47 locale tag (default: en).",
    )
    p.add_argument(
        "--out",
        default=None,
        metavar="PATH",
        help=(
            "Output path. Defaults to EP_DIR/AssetManifest.{locale}.json "
            "when EP_DIR is supplied."
        ),
    )
    return p.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    """Resolve and validate all input/output paths. Returns (script, shotlist, voicecast, out)."""
    ep_dir: Path | None = Path(args.ep_dir).resolve() if args.ep_dir else None

    # Script
    if args.script:
        script_path = Path(args.script).resolve()
    elif ep_dir:
        script_path = ep_dir / "Script.json"
    else:
        print("[ERROR] Provide --script or a positional EP_DIR.", file=sys.stderr)
        sys.exit(1)

    # ShotList
    if args.shotlist:
        shotlist_path = Path(args.shotlist).resolve()
    elif ep_dir:
        shotlist_path = ep_dir / "ShotList.json"
    else:
        print("[ERROR] Provide --shotlist or a positional EP_DIR.", file=sys.stderr)
        sys.exit(1)

    # VoiceCast
    if args.voice_cast:
        voicecast_path = Path(args.voice_cast).resolve()
    elif ep_dir:
        # Try projects/{slug}/VoiceCast.json — two levels up from ep dir
        candidate = ep_dir.parent.parent / "VoiceCast.json"
        if candidate.exists():
            voicecast_path = candidate
        else:
            print(
                f"[ERROR] VoiceCast.json not found at {candidate}. "
                "Pass --voice-cast explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        print("[ERROR] Provide --voice-cast or a positional EP_DIR.", file=sys.stderr)
        sys.exit(1)

    # Output
    locale = args.locale
    if args.out:
        out_path = Path(args.out).resolve()
    elif ep_dir:
        out_path = ep_dir / f"AssetManifest.{locale}.json"
    else:
        print("[ERROR] Provide --out or a positional EP_DIR.", file=sys.stderr)
        sys.exit(1)

    # Validate inputs exist
    for label, path in [
        ("--script", script_path),
        ("--shotlist", shotlist_path),
        ("--voice-cast", voicecast_path),
    ]:
        if not path.exists():
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    return script_path, shotlist_path, voicecast_path, out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    locale = args.locale

    print(f"▶ gen_vo_manifest.py — {locale}")

    script_path, shotlist_path, voicecast_path, out_path = resolve_paths(args)

    print(f"  Script    : {script_path}")
    print(f"  ShotList  : {shotlist_path}")
    print(f"  VoiceCast : {voicecast_path}")
    print(f"  Locale    : {locale}")
    print(f"  Output    : {out_path}")

    # Load inputs
    script    = load_json(script_path)
    shotlist  = load_json(shotlist_path)
    voicecast = load_json(voicecast_path)

    # Load schema
    if not SCHEMA_PATH.exists():
        print(f"[ERROR] Schema not found: {SCHEMA_PATH}", file=sys.stderr)
        sys.exit(1)
    schema = load_json(SCHEMA_PATH)

    # Build cast map
    cast_map = build_cast_map(voicecast)
    print(f"  Cast      : {len(cast_map)} characters in VoiceCast")

    # Step 2: collect dialogue lines
    dialogue_lines = collect_dialogue_lines(script)
    print(f"  Dialogue  : {len(dialogue_lines)} lines in Script.json")

    # Step 3: zip with vo_item_ids
    vo_tuples = build_vo_tuples(shotlist, dialogue_lines)
    print(f"  VO IDs    : {len(vo_tuples)} items to generate")

    # Steps 4-5: build each vo_item
    vo_items: list[dict] = []
    for tup in vo_tuples:
        item = build_vo_item(tup, cast_map, locale)
        vo_items.append(item)

    # Step 6: assemble manifest
    manifest = build_manifest(script, shotlist, vo_items, locale)

    # Step 7: validate
    try:
        validate_manifest(manifest, schema)
        print("  Validation: PASS")
    except jsonschema.ValidationError as exc:
        print(f"  [ERROR] Schema validation failed: {exc.message}", file=sys.stderr)
        print(f"  Path: {list(exc.absolute_path)}", file=sys.stderr)
        # Write the (invalid) manifest anyway so the user can inspect it
        save_json(manifest, out_path)
        print(f"  (Invalid manifest written to {out_path} for inspection)", file=sys.stderr)
        sys.exit(1)

    # Write output
    save_json(manifest, out_path)

    # Summary
    n = len(vo_items)
    speakers: dict[str, int] = {}
    for item in vo_items:
        sid = item["speaker_id"]
        speakers[sid] = speakers.get(sid, 0) + 1

    print()
    print(f"  VO items by speaker:")
    for sid, count in sorted(speakers.items()):
        print(f"    {sid:30s} {count:3d} line(s)")
    print()
    print(f"✓ AssetManifest.{locale}.json written ({n} VO items)")
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
