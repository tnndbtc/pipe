#!/usr/bin/env python3
# =============================================================================
# voice_cast_narrator.py — Stage 0 Phase B: pick narrator voice → VoiceCast.json
# =============================================================================
#
# Deterministic Stage 0 for non-episodic story formats.  Selects the best
# narrator Azure TTS voice for each locale and writes (or updates) the
# project-level VoiceCast.json.
#
# Handles formats: continuous_narration, illustrated_narration, documentary,
# monologue.  Episodic format still requires the LLM (multi-character casting).
#
# Voice selection reads the azure_tts_styles.txt reference shipped with the
# pipeline and applies the same rules as p_0.txt:
#   • narrator → prefer "narration-professional" or "newscast" style
#   • NEVER assign [MULTITALKER] voices
#   • Match locale strictly (locale field or supports() list)
#   • Gender: "neutral" for narrator
#
# Usage:
#   python voice_cast_narrator.py <ep_dir>
#
#   ep_dir — path to the episode directory
#             e.g. projects/the-pharaoh-who-defied-death/episodes/s01e02
#
# Requirements: stdlib only (json, pathlib, re, sys)
# =============================================================================

import json
import re
import sys
from pathlib import Path

STAGE_LABEL_START = "Stage 0b — voice_cast_narrator.py"
STAGE_LABEL_DONE  = "Stage 0b done — VoiceCast.json written"

# Formats handled by this script (narrator-only casting).
# Episodic is excluded — it requires full LLM-based multi-character casting.
NARRATOR_FORMATS = {
    "continuous_narration",
    "illustrated_narration",
    "documentary",
    "monologue",
    "ssml_narration",
}

# Narrator archetype defaults (from p_0.txt Step 2d)
NARRATOR_PITCH    = "-5%"
NARRATOR_BREAK_MS = 600
NARRATOR_RATE     = "0%"
NARRATOR_STYLE_DEGREE = "1.0"

# Safe-default narrator voices when no tts_styles.txt is available.
# Keys are pipeline locale codes (not Azure xml:lang codes).
SAFE_DEFAULTS: dict[str, dict] = {
    "zh-Hans": {
        "azure_voice":  "zh-CN-YunyangNeural",
        "available_styles": ["narration-professional", "newscast-casual", "customerservice"],
    },
    "en": {
        "azure_voice":  "en-US-GuyNeural",
        "available_styles": ["newscast", "narration-professional"],
    },
    "es": {
        "azure_voice":  "es-ES-AlvaroNeural",
        "available_styles": [],
    },
    "ja": {
        "azure_voice":  "ja-JP-NanamiNeural",
        "available_styles": ["chat", "customerservice", "cheerful"],
    },
    "ko": {
        "azure_voice":  "ko-KR-InJoonNeural",
        "available_styles": [],
    },
    "fr": {
        "azure_voice":  "fr-FR-HenriNeural",
        "available_styles": [],
    },
    "de": {
        "azure_voice":  "de-DE-ConradNeural",
        "available_styles": ["cheerful", "sad"],
    },
    "pt": {
        "azure_voice":  "pt-BR-FranciscaNeural",
        "available_styles": ["calm"],
    },
}

# Mapping from pipeline locale code → Azure xml:lang prefix(es) used in voices.
# A voice is eligible for locale L if its `locale` field or `supports()` list
# contains any entry that starts with an Azure prefix for L.
LOCALE_TO_AZURE_PREFIXES: dict[str, list[str]] = {
    "zh-Hans": ["zh-CN"],
    "zh-Hant": ["zh-TW", "zh-HK"],
    "en":      ["en-US", "en-GB", "en-AU", "en-IN", "en-CA", "en-IE"],
    "es":      ["es-ES", "es-MX"],
    "ja":      ["ja-JP"],
    "ko":      ["ko-KR"],
    "fr":      ["fr-FR"],
    "de":      ["de-DE"],
    "pt":      ["pt-BR", "pt-PT"],
    "it":      ["it-IT"],
    "hi":      ["hi-IN"],
}


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(doc: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ── pipeline_vars.sh parser ───────────────────────────────────────────────────

def load_pipeline_vars(ep_dir: Path) -> dict:
    """Parse pipeline_vars.sh and return dict of exported variables.

    Matches the implementation in canon_check.py:load_pipeline_vars().
    """
    vars_file = ep_dir / "pipeline_vars.sh"
    if not vars_file.exists():
        raise FileNotFoundError(f"pipeline_vars.sh not found: {vars_file}")
    result: dict[str, str] = {}
    for line in vars_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        # Match: export KEY="value" or export KEY=value or KEY="value"
        m = re.match(r'^(?:export\s+)?([A-Z_][A-Z0-9_]*)=["\']?(.*?)["\']?\s*$', line)
        if m:
            result[m.group(1)] = m.group(2)
    return result


# ── azure_tts_styles.txt parser ───────────────────────────────────────────────

def parse_tts_styles(styles_path: Path) -> list[dict]:
    """Parse azure_tts_styles.txt into a list of voice dicts.

    Each dict has keys: name, locale, gender, styles (list), supports (list).
    MULTITALKER voices are excluded entirely.

    Format parsed:
        VoiceName  [STANDARD|DRAGON]
          locale=xx-XX  gender=Male
          styles(N): ['style1', ...]
          supports(N): ['loc1', ...]
        --------
    """
    txt = styles_path.read_text(encoding="utf-8")

    # Truncate at the MULTITALKER section — those voices must never be cast.
    multitalker_marker = "MULTITALKER VOICES"
    if multitalker_marker in txt:
        txt = txt[: txt.index(multitalker_marker)]

    voices: list[dict] = []
    blocks = re.split(r"^-{40,}\s*$", txt, flags=re.MULTILINE)

    for block in blocks:
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        if not lines:
            continue
        # First non-blank line must be: VoiceName [STANDARD|DRAGON]
        m = re.match(r"^(\S+)\s+\[(STANDARD|DRAGON)\]$", lines[0])
        if not m:
            continue
        voice_name = m.group(1)

        locale  = ""
        gender  = ""
        styles: list[str]   = []
        supports: list[str] = []

        for line in lines[1:]:
            lm = re.match(r"locale=(\S+)\s+gender=(\S+)", line)
            if lm:
                locale = lm.group(1)
                gender = lm.group(2)
                continue
            sm = re.match(r"styles\(\d+\):\s*\[(.*?)\]", line)
            if sm:
                raw = sm.group(1).strip()
                if raw:
                    styles = [s.strip().strip("'") for s in raw.split(",")]
                continue
            pm = re.match(r"supports\(\d+\):\s*\[(.*?)\]", line)
            if pm:
                raw = pm.group(1).strip()
                if raw:
                    supports = [s.strip().strip("'") for s in raw.split(",")]

        if voice_name and locale:
            voices.append({
                "name":     voice_name,
                "locale":   locale,
                "gender":   gender,
                "styles":   styles,
                "supports": supports,
            })

    return voices


def voices_for_pipeline_locale(all_voices: list[dict], pipeline_locale: str) -> list[dict]:
    """Return voices eligible for a pipeline locale code (e.g. 'zh-Hans', 'en').

    A voice is eligible if its `locale` field or any entry in `supports` starts
    with one of the Azure locale prefixes mapped to the given pipeline locale.
    """
    prefixes = LOCALE_TO_AZURE_PREFIXES.get(pipeline_locale, [])
    if not prefixes:
        # Unknown locale: try treating the pipeline locale as an Azure prefix directly.
        # e.g. pipeline locale "en-US" would match voices whose locale == "en-US".
        prefixes = [pipeline_locale]

    eligible: list[dict] = []
    for v in all_voices:
        voice_locs = [v["locale"]] + v["supports"]
        if any(vl.startswith(p) for vl in voice_locs for p in prefixes):
            eligible.append(v)
    return eligible


def pick_narrator_voice(all_voices: list[dict], pipeline_locale: str) -> dict | None:
    """Select the best narrator voice for a pipeline locale.

    Selection priority (from p_0.txt Step 2b):
      1. Voice supports locale AND has "narration-professional" style.
      2. Voice supports locale AND has "newscast" (exact) or "newscast-casual" style.
      3. Any eligible voice with at least one style (avoids flat no-style voices).
      4. Any eligible voice (last resort).

    Returns None if no eligible voices found (caller falls back to safe defaults).
    """
    eligible = voices_for_pipeline_locale(all_voices, pipeline_locale)
    if not eligible:
        return None

    def score(v: dict) -> int:
        s = set(v["styles"])
        if "narration-professional" in s:
            return 4
        if "newscast" in s or "newscast-casual" in s:
            return 3
        if s:  # has some styles
            return 2
        return 1  # no styles (flat voice)

    best = max(eligible, key=score)
    return best


# ── VoiceCast helpers ─────────────────────────────────────────────────────────

def build_narrator_locale_block(voice_dict: dict) -> dict:
    """Build the per-locale block for the narrator character entry.

    Field names and structure match ssml_preprocess.py:build_voicecast_entry()
    and the VoiceCast.json examples in projects/.

    The `azure_style` field from ssml_preprocess is SSML-source-specific and is
    not included here — we set the recommended style via `available_styles` so
    the TTS stage can choose appropriately.
    """
    styles = voice_dict["styles"] if voice_dict else []

    # Prefer narration-professional; fall back to newscast; keep all styles.
    available_styles: list[str] = []
    for preferred in ("narration-professional", "newscast", "newscast-casual"):
        if preferred in styles and preferred not in available_styles:
            available_styles.append(preferred)
    for s in styles:
        if s not in available_styles:
            available_styles.append(s)

    return {
        "azure_voice":        voice_dict["name"],
        "available_styles":   available_styles,
        "azure_pitch":        NARRATOR_PITCH,
        "azure_break_ms":     NARRATOR_BREAK_MS,
        "azure_style_degree": float(NARRATOR_STYLE_DEGREE),
        "azure_rate":         NARRATOR_RATE,
    }


def build_narrator_locale_block_from_default(default: dict) -> dict:
    """Build narrator locale block from the SAFE_DEFAULTS table."""
    styles = default.get("available_styles", [])
    available_styles: list[str] = []
    for preferred in ("narration-professional", "newscast", "newscast-casual"):
        if preferred in styles and preferred not in available_styles:
            available_styles.append(preferred)
    for s in styles:
        if s not in available_styles:
            available_styles.append(s)

    return {
        "azure_voice":        default["azure_voice"],
        "available_styles":   available_styles,
        "azure_pitch":        NARRATOR_PITCH,
        "azure_break_ms":     NARRATOR_BREAK_MS,
        "azure_style_degree": float(NARRATOR_STYLE_DEGREE),
        "azure_rate":         NARRATOR_RATE,
    }


# ── Main logic ────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"▶ {STAGE_LABEL_START}")

    if len(sys.argv) < 2:
        print("Usage: python voice_cast_narrator.py <ep_dir>", file=sys.stderr)
        sys.exit(1)

    ep_dir = Path(sys.argv[1]).resolve()

    if not ep_dir.is_dir():
        print(f"[ERROR] ep_dir does not exist: {ep_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Step 1: read pipeline_vars.sh ────────────────────────────────────────
    try:
        pvars = load_pipeline_vars(ep_dir)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    project_slug   = pvars.get("PROJECT_SLUG", "").strip()
    story_format   = pvars.get("STORY_FORMAT", "").strip()
    locales_raw    = pvars.get("LOCALES", "").strip()
    primary_locale = pvars.get("PRIMARY_LOCALE", "").strip()

    if not project_slug:
        print("[ERROR] PROJECT_SLUG not found in pipeline_vars.sh", file=sys.stderr)
        sys.exit(1)

    if not story_format:
        print("[ERROR] STORY_FORMAT not found in pipeline_vars.sh", file=sys.stderr)
        sys.exit(1)

    if story_format not in NARRATOR_FORMATS:
        print(
            f"[INFO] story_format='{story_format}' is not a narrator-only format "
            f"(handled: {', '.join(sorted(NARRATOR_FORMATS))})."
        )
        print("[INFO] Episodic casting requires the LLM stage.  Nothing written.")
        print(f"✓ {STAGE_LABEL_DONE} (no-op for episodic format)")
        return

    # Parse locale list: "en,zh-Hans" or "en, zh-Hans"
    locales: list[str] = [lc.strip() for lc in locales_raw.split(",") if lc.strip()]
    if not locales:
        print("[ERROR] LOCALES is empty in pipeline_vars.sh", file=sys.stderr)
        sys.exit(1)

    # ── Resolve paths ─────────────────────────────────────────────────────────
    # PIPE_DIR: three levels up from this script (code/http/voice_cast_narrator.py)
    pipe_dir = Path(__file__).resolve().parent.parent.parent

    voicecast_path = pipe_dir / "projects" / project_slug / "VoiceCast.json"

    # ── Step 2: load existing VoiceCast.json ──────────────────────────────────
    if voicecast_path.exists():
        try:
            vc = load_json(voicecast_path)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [WARN] Could not read VoiceCast.json ({exc}); creating fresh.")
            vc = None
    else:
        vc = None

    if vc is None:
        vc = {
            "schema_id":      "VoiceCast",
            "schema_version": "1.0.0",
            "project_id":     project_slug,
            "characters":     [],
        }

    characters: list[dict] = vc.setdefault("characters", [])

    # Build lookup: character_id → index in characters list
    char_index: dict[str, int] = {c["character_id"]: i for i, c in enumerate(characters)}

    # ── Load tts_styles reference ─────────────────────────────────────────────
    # Check canonical location first; fall back gracefully.
    tts_styles_candidates = [
        pipe_dir / "prompts" / "azure_tts_styles.txt",
        pipe_dir / "projects" / project_slug / f"azure_tts_styles.txt",
    ]
    # Also check locale-specific files (not standard but handle if present)
    for lc in locales:
        tts_styles_candidates.append(
            pipe_dir / "projects" / project_slug / f"azure_tts_styles_{lc}.txt"
        )

    all_voices: list[dict] = []
    for candidate in tts_styles_candidates:
        if candidate.exists():
            try:
                all_voices = parse_tts_styles(candidate)
                print(f"  TTS styles ref : {candidate.relative_to(pipe_dir)}  "
                      f"({len(all_voices)} voices parsed)")
            except Exception as exc:
                print(f"  [WARN] Failed to parse {candidate}: {exc}")
                all_voices = []
            break  # Use first found

    if not all_voices:
        print("  [WARN] No azure_tts_styles.txt found — using safe defaults.")

    # ── Step 3: for each locale, add narrator if not already present ──────────
    narrator_idx = char_index.get("narrator")
    if narrator_idx is not None:
        narrator_entry = characters[narrator_idx]
    else:
        # Create a new narrator entry skeleton
        narrator_entry = {
            "character_id": "narrator",
            "role":         "narrator",
            "gender":       "neutral",
            "personality":  "authoritative, cinematic",
        }

    added_locales:   list[str] = []
    skipped_locales: list[str] = []

    for locale in locales:
        # Check if this locale already has a voice block on the narrator entry.
        if locale in narrator_entry:
            print(f"  Locale {locale!r}: narrator already cast — skipping.")
            skipped_locales.append(locale)
            continue

        # Select voice from parsed tts_styles, or fall back to safe defaults.
        if all_voices:
            best = pick_narrator_voice(all_voices, locale)
            if best:
                locale_block = build_narrator_locale_block(best)
                print(
                    f"  Locale {locale!r}: selected {best['name']!r}  "
                    f"styles={best['styles'][:3]}{'...' if len(best['styles']) > 3 else ''}"
                )
            else:
                # No voice found in tts_styles for this locale — use safe default.
                default = SAFE_DEFAULTS.get(locale)
                if not default:
                    # Ultra-fallback: pick any English neutral default.
                    default = SAFE_DEFAULTS["en"]
                    print(f"  [WARN] No voice found for locale {locale!r}; "
                          f"using en fallback: {default['azure_voice']!r}")
                else:
                    print(f"  [WARN] No voice found in tts_styles for locale {locale!r}; "
                          f"using safe default: {default['azure_voice']!r}")
                locale_block = build_narrator_locale_block_from_default(default)
        else:
            # No tts_styles parsed — use safe defaults.
            default = SAFE_DEFAULTS.get(locale)
            if not default:
                default = SAFE_DEFAULTS["en"]
                print(f"  [WARN] No safe default for locale {locale!r}; "
                      f"using en fallback: {default['azure_voice']!r}")
            else:
                print(f"  Locale {locale!r}: safe default {default['azure_voice']!r}")
            locale_block = build_narrator_locale_block_from_default(default)

        narrator_entry[locale] = locale_block
        added_locales.append(locale)

    # ── Step 4: write updated VoiceCast.json ──────────────────────────────────
    if narrator_idx is not None:
        # Update existing entry in-place
        characters[narrator_idx] = narrator_entry
    else:
        characters.append(narrator_entry)

    save_json(vc, voicecast_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("── Stage 0b complete ─────────────────────────────────────────────────")
    print(f"PROJECT_SLUG     : {project_slug}")
    print(f"STORY_FORMAT     : {story_format}")
    print(f"LOCALES          : {', '.join(locales)}")
    print(f"PRIMARY_LOCALE   : {primary_locale}")
    print(f"Added locales    : {', '.join(added_locales) or '(none)'}")
    print(f"Skipped locales  : {', '.join(skipped_locales) or '(none)'}")
    print(f"VoiceCast.json   : {voicecast_path}")
    print("─────────────────────────────────────────────────────────────────────")
    print(f"✓ {STAGE_LABEL_DONE}")


if __name__ == "__main__":
    main()
