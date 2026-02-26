# =============================================================================
# gen_tts.py
# Generate spoken voice-over audio for every dialogue line in s01e01.
# Supports multiple TTS backends: Kokoro (default), XTTS v2, MeloTTS.
# Use --tts_model [kokoro|xtts|melo|all] to select or compare.
# =============================================================================
#
# requirements.txt (pip install before running):
#   kokoro>=0.9.4
#   soundfile>=0.12.0
#   numpy>=1.24.0
#   huggingface_hub>=0.21.0
#
# Optional backends (install the ones you want to compare):
#   XTTS v2 (Coqui):  pip install TTS>=0.22.0
#   MeloTTS:          pip install melo-tts
#
# Language-specific extras for Kokoro:
#   Chinese (zh-Hans):  pip install "misaki[zh]"
#   Japanese (ja):      pip install "misaki[ja]"
#
# ---------------------------------------------------------------------------
# Hardware Target: NVIDIA RTX 4060 8 GB VRAM
# ---------------------------------------------------------------------------
# Backend comparison:
#
#   kokoro  — 82 M params, CPU-only, very fast, good English/Chinese.
#             No VRAM required.
#
#   xtts    — XTTS v2 (~1.8 GB VRAM on GPU, runs on CPU too).
#             Multilingual, voice-cloning from a short reference WAV.
#             Reference WAVs are auto-generated from Kokoro on first run
#             and cached in code/models/reference_voices/.
#             Place your own {speaker}.wav there for custom voices.
#             NOTE: XTTS v2 does not support speed control; prosody is
#             model-driven and typically more expressive than Kokoro.
#
#   melo    — MeloTTS (~200 MB), lightweight, good Chinese/English,
#             multiple accent options (EN-US, EN-BR, EN-AU, ZH, JP, KR …).
#             Supports speed control. ~200 MB — trivially fits in VRAM.
#
# Output structure:
#   assets/{locale}/{backend}/vo-*.wav
#   e.g. assets/en/kokoro/vo-s01-001.wav
#        assets/en/xtts/vo-s01-001.wav
#        assets/en/melo/vo-s01-001.wav
#        assets/zh-Hans/kokoro/vo-s01-001.wav
#
# Model cache locations:
#   XTTS v2  — default Coqui TTS cache  (Windows: %LOCALAPPDATA%\tts\)
#   MeloTTS  — default HuggingFace cache  (~/.cache/huggingface/)
#   Kokoro   — default HuggingFace cache  (~/.cache/huggingface/)
#   XTTS reference WAVs — code/models/reference_voices/
# ---------------------------------------------------------------------------

import argparse
import gc
import json
import os
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# ---------------------------------------------------------------------------
# DEFAULTS — fully populated; script runs with no CLI flags.
# ---------------------------------------------------------------------------
PROJECTS_ROOT = Path(__file__).resolve().parent.parent.parent / "projects"
MODELS_DIR    = Path(__file__).resolve().parent.parent / "models"
SCRIPT_NAME   = "gen_tts"

# Voice mapping:
#   amunhotep    -> bm_george  (deep British male, elderly, gravelly)
#   ramesses_ka  -> bm_lewis   (commanding resonant male)
#   neferet      -> bf_emma    (young, precise female)
#   khamun       -> bm_daniel  (deep military male)
#   voice_of_gate -> bm_george at speed=0.75 (slower, supernatural weight)
#
# Speed key: slow=0.8, normal=1.0, fast=1.2
VO_ITEMS = [
    {
        "item_id": "vo-s01-001", "speaker": "amunhotep", "voice": "bm_george",
        "speed": 0.8,
        "text": "This was not carved by our people. It predates the Old Kingdom. Perhaps everything.",
        "emotion_note": "ominous, reverent",
    },
    {
        "item_id": "vo-s01-002", "speaker": "ramesses_ka", "voice": "bm_lewis",
        "speed": 1.0,
        "text": "What does it say?",
        "emotion_note": "cold curiosity",
    },
    {
        "item_id": "vo-s01-003", "speaker": "amunhotep", "voice": "bm_george",
        "speed": 0.8,
        "text": (
            "It does not speak of prayers, my king. It describes a mechanism. "
            "A gate beneath the sand — capable of holding the Ka and Ba beyond the reach of death."
        ),
        "emotion_note": "grave, deeply troubled",
    },
    {
        "item_id": "vo-s01-004", "speaker": "ramesses_ka", "voice": "bm_lewis",
        "speed": 0.8,
        "text": "Not resurrection.",
        "emotion_note": "cold, certain",
    },
    {
        "item_id": "vo-s01-005", "speaker": "amunhotep", "voice": "bm_george",
        "speed": 0.8,
        "text": "Not heaven. Containment. The soul, preserved. Conscious. Permanent.",
        "emotion_note": "haunted, reluctant",
    },
    {
        "item_id": "vo-s01-006", "speaker": "ramesses_ka", "voice": "bm_lewis",
        "speed": 1.0,
        "text": "Then why are you afraid?",
        "emotion_note": "contemptuous, challenging",
    },
    {
        "item_id": "vo-s01-007", "speaker": "amunhotep", "voice": "bm_george",
        "speed": 0.8,
        "text": "Because it has been opened before.",
        "emotion_note": "terrified whisper",
    },
    {
        "item_id": "vo-s02-001", "speaker": "neferet", "voice": "bf_emma",
        "speed": 0.8,
        "text": (
            "The Great Silence... temples across the Two Lands ceased all ritual "
            "simultaneously. Entire priesthoods... vanished."
        ),
        "emotion_note": "disturbed, disbelieving",
    },
    {
        "item_id": "vo-s02-002", "speaker": "neferet", "voice": "bf_emma",
        "speed": 0.8,
        "text": "The last line.",
        "emotion_note": "tense, bracing",
    },
    {
        "item_id": "vo-s02-003", "speaker": "neferet", "voice": "bf_emma",
        "speed": 0.8,
        "text": "The First Opener brought judgment upon the Two Lands.",
        "emotion_note": "horrified whisper",
    },
    {
        "item_id": "vo-s03-001", "speaker": "khamun", "voice": "bm_daniel",
        "speed": 1.0,
        "text": "These men have families.",
        "emotion_note": "suppressed anger, flat",
    },
    {
        "item_id": "vo-s03-002", "speaker": "khamun", "voice": "bm_daniel",
        "speed": 0.8,
        "text": "What kind of tomb requires silence enforced by execution?",
        "emotion_note": "low, conflicted, bitter",
    },
    {
        "item_id": "vo-s04-001", "speaker": "amunhotep", "voice": "bm_george",
        "speed": 0.8,
        "text": (
            "We call upon Anubis, Opener of the Way. We call upon Osiris, Lord of Eternity. "
            "The Gate is present. The Gate is listening."
        ),
        "emotion_note": "trance-like, dread",
    },
    {
        "item_id": "vo-s04-002", "speaker": "voice_of_gate", "voice": "bm_george",
        "speed": 0.75,
        # speed=0.75 gives a heavier, more supernatural cadence
        "text": "Who dares open the Gate again?",
        "emotion_note": "ancient, commanding, cold — visual=false, VO only",
    },
    {
        "item_id": "vo-s04-003", "speaker": "amunhotep", "voice": "bm_george",
        "speed": 1.2,
        "text": "My king — do not—",
        "emotion_note": "desperate terror",
    },
    {
        "item_id": "vo-s04-004", "speaker": "ramesses_ka", "voice": "bm_lewis",
        "speed": 0.8,
        "text": "Your Pharaoh.",
        "emotion_note": "sovereign, fearless",
    },
    {
        "item_id": "vo-s05-001", "speaker": "neferet", "voice": "bf_emma",
        "speed": 0.8,
        "text": "Seal it.",
        "emotion_note": "breathless horror",
    },
    {
        "item_id": "vo-s05-002", "speaker": "neferet", "voice": "bf_emma",
        "speed": 0.8,
        "text": "Before the stars return.",
        "emotion_note": "barely audible dread",
    },
]

SAMPLE_RATE = 24000   # Kokoro native sample rate

# ---------------------------------------------------------------------------
# Built-in voice resolution tables  (used when no --voices file is given,
# and as the base that --voices overrides layer on top of)
# ---------------------------------------------------------------------------

# speaker_id -> Kokoro voice ID  (English defaults)
SPEAKER_TO_VOICE = {
    "amunhotep":     "bm_george",   # deep British male, elderly, gravelly
    "ramesses_ka":   "bm_lewis",    # commanding resonant male
    "neferet":       "bf_emma",     # young, precise female
    "khamun":        "bm_daniel",   # deep military male
    "voice_of_gate": "bm_george",   # closest available for supernatural voice
}

# speaker_id -> forced speed (applied after pace; overrides pace entirely)
SPEAKER_SPEED_OVERRIDES = {
    "voice_of_gate": 0.75,          # extra-slow supernatural cadence
}

# voice_style keyword rules for speakers NOT in SPEAKER_TO_VOICE.
# Each entry: (set_of_keywords, kokoro_voice_id).
# Matched against voice_style.lower(). First match wins.
VOICE_STYLE_RULES = [
    ({"young female", "female", "woman", "girl"},           "bf_emma"),
    ({"elderly", "aged", "old", "gravelly", "ceremonial"},  "bm_george"),
    ({"military", "soldier", "warrior", "bearing"},         "bm_daniel"),
    ({"commanding", "resonant", "powerful", "sovereign"},   "bm_lewis"),
    ({"supernatural", "distorted", "ancient", "deep"},      "bm_george"),
]

# manifest locale string -> Kokoro KPipeline lang_code
LOCALE_TO_LANG_CODE = {
    "en":       "en-us",
    "en-us":    "en-us",
    "en-gb":    "en-gb",
    "zh":       "z",        # Mandarin Chinese
    "zh-hans":  "z",        # Simplified Chinese  (case-insensitive match applied below)
    "zh-hant":  "z",        # Traditional Chinese
    "zh-cn":    "z",
    "zh-tw":    "z",
    "ja":       "j",        # Japanese
    "ko":       "k",        # Korean
    "fr":       "f",        # French
    "es":       "e",        # Spanish
    "pt":       "p",        # Portuguese
    "hi":       "h",        # Hindi
    "it":       "i",        # Italian
}

PACE_TO_SPEED = {"slow": 0.8, "normal": 1.0, "fast": 1.2}

# Built-in voice maps for non-English lang_codes.
# Priority: voices file > these lang defaults > English SPEAKER_TO_VOICE
LANG_CODE_DEFAULT_VOICES = {
    "z": {                              # Mandarin Chinese
        "amunhotep":     "zm_yunxi",    # male, mid-aged
        "ramesses_ka":   "zm_yunjian",  # male, commanding
        "neferet":       "zf_xiaobei",  # female, young
        "khamun":        "zm_yunxia",   # male, deep
        "voice_of_gate": "zm_yunxi",    # closest available for supernatural
    },
}

# =============================================================================
# XTTS v2 backend configuration
# =============================================================================
XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_SAMPLE_RATE = 24000

# Kokoro lang_code -> XTTS language string
XTTS_LANG_MAP = {
    "en-us": "en", "en-gb": "en",
    "z":     "zh-cn",
    "j":     "ja",
    "k":     "ko",
    "f":     "fr",
    "e":     "es",
    "p":     "pt",
    "h":     "hi",
    "i":     "it",
}

# Speaker gender — used to pick the gender-default reference WAV
XTTS_SPEAKER_GENDER = {
    "amunhotep":     "male",
    "ramesses_ka":   "male",
    "neferet":       "female",
    "khamun":        "male",
    "voice_of_gate": "male",
}

# Kokoro voice used when auto-generating a gender-default reference WAV.
# Nested by xtts_lang so Chinese/Japanese refs use native voices.
XTTS_REF_KOKORO_VOICE = {
    "en":    {"male": "bm_george",  "female": "bf_emma"},
    "zh-cn": {"male": "zm_yunxi",   "female": "zf_xiaobei"},
    "ja":    {"male": "jm_kumo",    "female": "jf_alpha"},
}

# Kokoro lang_code to use when generating reference WAVs, keyed by xtts_lang
XTTS_REF_KOKORO_LANG = {
    "en":    "en-us",
    "zh-cn": "z",
    "ja":    "j",
}

# Text to synthesise when creating a reference WAV — should be ~6–10 seconds
XTTS_REF_SAMPLE_TEXT = {
    "en": (
        "This is a reference voice sample for speech synthesis. "
        "The tone should be clear and natural, with appropriate pacing and rhythm. "
        "A longer sample produces better voice cloning quality."
    ),
    "zh-cn": (
        "这是用于语音合成的参考音频样本。语调应该清晰自然，节奏合适，表达流畅。"
        "较长的样本能够产生更好的声音克隆质量。"
    ),
    "ja": (
        "これは音声合成用の参照音声サンプルです。"
        "トーンは明瞭で自然であるべきです。より長いサンプルを使用すると品質が向上します。"
    ),
}

# =============================================================================
# MeloTTS backend configuration
# =============================================================================
MELO_SAMPLE_RATE = 44100  # MeloTTS native output sample rate

# Kokoro lang_code -> MeloTTS language string
MELO_LANG_MAP = {
    "en-us": "EN", "en-gb": "EN",
    "z":     "ZH",
    "j":     "JP",
    "k":     "KR",
    "f":     "FR",
    "e":     "ES",
}

# speaker_id -> MeloTTS English accent string (used when melo_lang == "EN")
# NOTE: MeloTTS EN is single-speaker — accent variants differ in accent only, not gender or voice.
# All EN speakers will sound like the same female voice. Use Kokoro/XTTS for gender differentiation.
MELO_ACCENT_MAP = {
    "amunhotep":     "EN-US",
    "ramesses_ka":   "EN-US",
    "neferet":       "EN-US",
    "khamun":        "EN-US",
    "voice_of_gate": "EN-US",
}



# ---------------------------------------------------------------------------
# Shared utility functions
# ---------------------------------------------------------------------------

def resolve_voice_from_style(voice_style: str, rules=None) -> tuple:
    """
    Match a voice_style description against keyword rules.
    Returns (kokoro_voice_id, match_reason).
    Falls back to 'bm_george' if nothing matches.

    rules: list of (set_of_keywords, voice_id) — uses VOICE_STYLE_RULES if None.
    """
    effective_rules = rules if rules is not None else VOICE_STYLE_RULES
    lower = voice_style.lower()
    for keywords, voice_id in effective_rules:
        matched = [kw for kw in keywords if kw in lower]
        if matched:
            return voice_id, f"style keyword '{matched[0]}'"
    return "bm_george", "default fallback"


# ---------------------------------------------------------------------------
# Voice profiles loader  (--voices <file.json>)
# ---------------------------------------------------------------------------

def load_voice_profiles(path: str) -> dict:
    """
    Load a voice-profiles JSON file and return a normalised profiles dict.

    Supported JSON keys (all optional):
      description          — human-readable label, logged only
      speaker_voice_map    — { speaker_id: kokoro_voice_id, ... }
      speaker_speed_overrides — { speaker_id: float, ... }
      locale_lang_map      — { locale_string: kokoro_lang_code, ... }
      voice_style_rules    — [ { "keywords": [...], "voice": "..." }, ... ]

    The returned dict is merged on top of the built-in defaults at resolution
    time; keys present in the file win, missing keys fall back to built-ins.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    profiles = {
        "description":             raw.get("description", ""),
        "speaker_voice_map":       raw.get("speaker_voice_map", {}),
        "speaker_speed_overrides": raw.get("speaker_speed_overrides", {}),
        "locale_lang_map":         raw.get("locale_lang_map", {}),
    }

    # Convert voice_style_rules from JSON list-of-objects to list-of-tuples
    if "voice_style_rules" in raw:
        profiles["voice_style_rules"] = [
            (set(r["keywords"]), r["voice"])
            for r in raw["voice_style_rules"]
        ]

    return profiles


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate voice-over WAV files using selectable TTS backends.\n"
            "Backends: kokoro (default), xtts, melo, all\n\n"
            "Examples:\n"
            "  python gen_tts.py --manifest AssetManifest_draft.json\n"
            "  python gen_tts.py --manifest AssetManifest_draft.zh-Hans.json\n"
            "  python gen_tts.py --manifest AssetManifest_draft.json --tts_model all\n"
            "  python gen_tts.py --manifest AssetManifest_draft.json --tts_model xtts\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--item_id", type=str, default=None,
        help="Generate only a single VO item by item_id (default: generate all).",
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to AssetManifest JSON. Overrides the hardcoded VO_ITEMS list.",
    )
    parser.add_argument(
        "--asset-id", type=str, default=None, dest="asset_id",
        help="Process only this item_id (requires --manifest).",
    )
    parser.add_argument(
        "--voices", type=str, default=None, metavar="PROFILE_JSON",
        help=(
            "Path to a voice-profiles JSON file that overrides speaker->voice mapping, "
            "locale->lang_code mapping, speed overrides, and style-matching rules. "
            "Use voices_en.json for English, voices_zh-Hans.json for Chinese, etc."
        ),
    )
    parser.add_argument(
        "--tts_model",
        choices=["kokoro", "xtts", "melo", "azure", "all"],
        default="kokoro",
        help=(
            "TTS backend to use. "
            "'kokoro' = Kokoro-82M (default, CPU-only, fast). "
            "'xtts'   = XTTS v2 (voice-cloning, GPU recommended, pip install TTS). "
            "'melo'   = MeloTTS (lightweight, accent-aware, pip install melo-tts). "
            "'azure'  = Azure Neural TTS (cloud, requires AZURE_SPEECH_KEY + AZURE_SPEECH_REGION). "
            "'all'    = run all four backends for comparison."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Manifest loader
# ---------------------------------------------------------------------------
def load_from_manifest(manifest_path: str, asset_id_filter, voice_profiles: dict = None):
    """
    Load VO item list from AssetManifest JSON (section: vo_items).

    All four tts_prompt fields drive generation:
      voice_style -> Kokoro voice ID  (fallback for unknown speakers)
      locale      -> KPipeline lang_code
      pace        -> synthesis speed  (slow=0.8 / normal=1.0 / fast=1.2)
      emotion     -> logged at generation time

    voice_profiles: dict from load_voice_profiles(), or None to use built-ins.
    """
    p = voice_profiles or {}

    # Build effective lookup tables (profiles layer on top of built-ins)
    eff_speed_override = {**SPEAKER_SPEED_OVERRIDES, **p.get("speaker_speed_overrides", {})}
    eff_locale_map     = {**LOCALE_TO_LANG_CODE,     **p.get("locale_lang_map", {})}
    eff_style_rules    = p.get("voice_style_rules")   # None = use built-in VOICE_STYLE_RULES

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    items = []
    for vo in manifest.get("vo_items", []):
        if asset_id_filter and vo["item_id"] != asset_id_filter:
            continue

        speaker     = vo["speaker_id"]
        tts         = vo.get("tts_prompt", {})
        voice_style = tts.get("voice_style", "")
        emotion     = tts.get("emotion", "")
        pace        = tts.get("pace", "normal")
        locale      = tts.get("locale", "en")

        # --- locale -> lang_code (resolve first — needed for voice defaults) --
        locale_key = locale.lower()
        lang_code  = eff_locale_map.get(locale_key) or eff_locale_map.get(locale, "en-us")

        # --- voice resolution ---
        # Priority: explicit voices file > lang-code defaults > English built-ins > style rules
        locale_voice_defaults = LANG_CODE_DEFAULT_VOICES.get(lang_code, {})
        effective_speaker_map = {
            **SPEAKER_TO_VOICE,                  # English base (lowest priority)
            **locale_voice_defaults,             # lang-code defaults (e.g. Chinese voices)
            **p.get("speaker_voice_map", {}),    # explicit --voices file (highest priority)
        }
        if speaker in effective_speaker_map:
            voice = effective_speaker_map[speaker]
            if speaker in p.get("speaker_voice_map", {}):
                voice_src = "voices file"
            elif speaker in locale_voice_defaults:
                voice_src = f"lang default ({lang_code})"
            else:
                voice_src = "speaker map (en)"
        else:
            voice, match_reason = resolve_voice_from_style(voice_style, rules=eff_style_rules)
            voice_src = f"voice_style ({match_reason})"

        # --- speed  (explicit override beats pace) -----------------------
        if speaker in eff_speed_override:
            speed     = eff_speed_override[speaker]
            src_label = "voices file" if speaker in p.get("speaker_speed_overrides", {}) else "built-in override"
            speed_src = f"{src_label} -> {speed}"
        else:
            speed     = PACE_TO_SPEED.get(pace, 1.0)
            speed_src = f"pace='{pace}' -> {speed}"

        items.append({
            "item_id":     vo["item_id"],
            "speaker":     speaker,
            "voice":       voice,
            "voice_src":   voice_src,
            "voice_style": voice_style,
            "speed":       speed,
            "speed_src":   speed_src,
            "lang_code":   lang_code,
            "locale":      locale,
            "text":        vo["text"],
            "emotion":     emotion,
        })
    return items


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def locale_from_manifest_path(path: str) -> str:
    """Extract locale from manifest filename.
    'AssetManifest_draft.zh-Hans.json' -> 'zh-Hans'
    'AssetManifest_draft.json'          -> 'en'
    """
    stem = Path(path).stem
    parts = stem.split('.')
    return parts[-1] if len(parts) > 1 else 'en'


def assets_dir_from_manifest(path: str) -> Path:
    """
    Derive the assets output directory from manifest fields.

    Prefers the direct episode_id field (schema v2+).
    Falls back to parsing episode_id from manifest_id for legacy manifests.

    Returns: PROJECTS_ROOT/{project_id}/episodes/{episode_id}/assets/
    """
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)

    project_id = manifest.get("project_id", "")
    if not project_id:
        raise ValueError(f"Manifest {path!r} is missing 'project_id'.")

    # Prefer direct field; fall back to parsing from manifest_id
    episode_id = manifest.get("episode_id", "")
    if not episode_id:
        manifest_id = manifest.get("manifest_id", "")
        episode_id = manifest_id
        if episode_id.startswith(project_id + "-"):
            episode_id = episode_id[len(project_id) + 1:]
        if episode_id.endswith("-manifest"):
            episode_id = episode_id[: -len("-manifest")]

    if not episode_id:
        raise ValueError(
            f"Manifest {path!r} is missing 'episode_id' and episode_id could "
            f"not be parsed from manifest_id. Add an 'episode_id' field."
        )

    return PROJECTS_ROOT / project_id / "episodes" / episode_id / "assets"


def write_license_sidecar(wav_path: Path, model_name: str) -> None:
    """Write a CC0 license sidecar for a generated WAV file.

    Saved at: {wav_path.parent}/licenses/{wav_path.stem}.license.json

    Fields:
      spdx_id             — "CC0" (AI-generated audio has no copyright owner)
      attribution_required — false (CC0 requires none)
      text                — human-readable provenance string
      generator_model     — structured model/voice identifier for future resolvers
    """
    licenses_dir = wav_path.parent / "licenses"
    licenses_dir.mkdir(parents=True, exist_ok=True)
    sidecar = {
        "spdx_id": "CC0",
        "attribution_required": False,
        "text": f"AI-generated voice audio. No copyright claimed. Produced locally by {model_name}.",
        "generator_model": model_name,
    }
    sidecar_path = licenses_dir / f"{wav_path.stem}.license.json"
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2, ensure_ascii=False)


# =============================================================================
# Kokoro backend
# =============================================================================

def load_kokoro(lang_code: str = "en-us"):
    """Load the Kokoro KPipeline for the given lang_code. Downloads on first run."""
    from kokoro import KPipeline
    print(f"[MODEL] Loading Kokoro-82M  lang_code={lang_code}  (hexgrad/Kokoro-82M)...")
    pipe = KPipeline(lang_code=lang_code)
    print("[MODEL] Kokoro ready.")
    return pipe


def synthesise_kokoro(pipe, text: str, voice: str, speed: float) -> np.ndarray:
    """
    Run Kokoro inference and concatenate all audio chunks into one array.
    Kokoro splits long texts internally; we join the chunks back together.
    """
    chunks = []
    for _gs, _ps, audio_chunk in pipe(text, voice=voice, speed=speed):
        if audio_chunk is not None and len(audio_chunk) > 0:
            if hasattr(audio_chunk, "numpy"):
                audio_chunk = audio_chunk.numpy()
            chunks.append(np.array(audio_chunk, dtype=np.float32))
    if not chunks:
        raise RuntimeError("Kokoro returned no audio chunks.")
    return np.concatenate(chunks)


# Keep old name as alias so existing callers work
synthesise = synthesise_kokoro


def run_kokoro_backend(items: list, out_dir: Path, args) -> list:
    """Run Kokoro TTS for all items. Returns result list."""
    lang_groups: dict[str, list] = {}
    for item in items:
        lang_groups.setdefault(item.get("lang_code", "en-us"), []).append(item)

    results = []
    total = len(items)
    item_index = 0

    for lang_code, lang_items in lang_groups.items():
        pipe = load_kokoro(lang_code)

        for vo in lang_items:
            item_index += 1
            out_filename = f"{vo['item_id']}.wav"
            out_path = out_dir / out_filename

            print(f"\n[{item_index}/{total}] {vo['item_id']}")
            print(f"  Speaker    : {vo['speaker']}")
            print(f"  Voice style: {vo.get('voice_style') or '(not set)'}")
            print(f"  Voice ID   : {vo['voice']}  (source: {vo.get('voice_src', 'unknown')})")
            print(f"  Emotion    : {vo.get('emotion') or '(not set)'}")
            print(f"  Speed      : {vo['speed']}  ({vo.get('speed_src', '')})")
            print(f"  Locale     : {vo.get('locale', 'en')}  ->  lang_code={lang_code}")
            print(f"  Text       : \"{vo['text'][:70]}{'...' if len(vo['text']) > 70 else ''}\"")


            try:
                audio = synthesise_kokoro(pipe, vo["text"], voice=vo["voice"], speed=vo["speed"])
                sf.write(str(out_path), audio, SAMPLE_RATE, subtype="PCM_16")
                size = out_path.stat().st_size
                duration_s = len(audio) / SAMPLE_RATE
                model_name = f"Kokoro-82M (voice={vo['voice']})"
                write_license_sidecar(out_path, model_name)
                print(f"  [OK] {out_path}  ({duration_s:.2f}s, {size:,} bytes)")
                results.append({
                    "item_id":      vo["item_id"],
                    "speaker":      vo["speaker"],
                    "voice":        vo["voice"],
                    "voice_style":  vo.get("voice_style", ""),
                    "emotion":      vo.get("emotion", ""),
                    "speed":        vo["speed"],
                    "lang_code":    lang_code,
                    "output":       str(out_path),
                    "size_bytes":   size,
                    "duration_sec": round(duration_s, 3),
                    "status":       "success",
                })
            except Exception as exc:
                print(f"  [ERROR] {vo['item_id']}: {exc}")
                results.append({
                    "item_id": vo["item_id"], "speaker": vo["speaker"],
                    "voice": vo["voice"], "output": str(out_path),
                    "size_bytes": 0, "status": "failed", "error": str(exc),
                })

        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    return results


# =============================================================================
# XTTS v2 backend
# =============================================================================

def load_xtts():
    """
    Load XTTS v2.  Downloads model to MODELS_DIR/tts_home/ on first run
    (~1.8 GB; requires pip install TTS>=0.22.0).
    """
    from TTS.api import TTS  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MODEL] Loading XTTS v2 on {device}  ({XTTS_MODEL_NAME})...")
    tts = TTS(model_name=XTTS_MODEL_NAME, progress_bar=True).to(device)
    print("[MODEL] XTTS v2 ready.")
    return tts


def ensure_xtts_ref_wav(speaker: str, xtts_lang: str) -> "Path | None":
    """
    Return path to a reference WAV for XTTS voice cloning.

    Search priority:
      1. code/models/reference_voices/{speaker}_{xtts_lang}.wav  — language-specific, user-placed
      2. code/models/reference_voices/{speaker}.wav              — speaker default, user-placed
      3. code/models/reference_voices/default_{gender}_{xtts_lang}.wav  — auto-generated lang default
      4. code/models/reference_voices/default_{gender}.wav              — legacy fallback
      5. Auto-generate default_{gender}_{xtts_lang}.wav from Kokoro using native voices.
      6. Return None if auto-generation fails (XTTS will skip that item).

    For Chinese: auto-generates from zm_yunxi (male) / zf_xiaobei (female).
    To provide custom voices, place 6–12 second mono WAVs in code/models/reference_voices/.
    """
    ref_dir = MODELS_DIR / "reference_voices"
    ref_dir.mkdir(parents=True, exist_ok=True)

    gender = XTTS_SPEAKER_GENDER.get(speaker, "male")

    # 1. Speaker + language specific
    speaker_lang_ref = ref_dir / f"{speaker}_{xtts_lang}.wav"
    if speaker_lang_ref.exists():
        return speaker_lang_ref

    # 2. Speaker-specific (language-agnostic)
    speaker_ref = ref_dir / f"{speaker}.wav"
    if speaker_ref.exists():
        return speaker_ref

    # 3. Gender + language default (cached auto-generated)
    gender_lang_ref = ref_dir / f"default_{gender}_{xtts_lang}.wav"
    if gender_lang_ref.exists():
        return gender_lang_ref

    # 4. Legacy gender default (language-agnostic)
    gender_ref = ref_dir / f"default_{gender}.wav"
    if gender_ref.exists():
        return gender_ref

    # 5. Auto-generate language-specific gender default from Kokoro
    voice_map = XTTS_REF_KOKORO_VOICE.get(xtts_lang) or XTTS_REF_KOKORO_VOICE.get("en", {})
    kokoro_voice = voice_map.get(gender, "bm_george")
    kokoro_lang = XTTS_REF_KOKORO_LANG.get(xtts_lang, "en-us")
    sample_text = XTTS_REF_SAMPLE_TEXT.get(xtts_lang, XTTS_REF_SAMPLE_TEXT["en"])
    print(f"  [REF] Auto-generating {gender}/{xtts_lang} reference WAV from Kokoro "
          f"(voice={kokoro_voice}, lang={kokoro_lang})...")
    try:
        kpipe = load_kokoro(kokoro_lang)
        audio = synthesise_kokoro(kpipe, sample_text, voice=kokoro_voice, speed=1.0)
        sf.write(str(gender_lang_ref), audio, SAMPLE_RATE, subtype="PCM_16")
        del kpipe
        gc.collect()
        torch.cuda.empty_cache()
        duration_s = len(audio) / SAMPLE_RATE
        print(f"  [REF] Saved {gender_lang_ref.name}  ({duration_s:.1f}s)")
        return gender_lang_ref
    except Exception as exc:
        print(f"  [WARN] Reference WAV auto-generation failed: {exc}")
        print(f"  [HINT] Place a 6–12 s WAV at: {gender_lang_ref}")
        return None


def run_xtts_backend(items: list, out_dir: Path) -> list:
    """Run XTTS v2 for all items. Returns result list."""
    try:
        tts_model = load_xtts()
    except ImportError:
        print("[ERROR] XTTS backend requires: pip install TTS>=0.22.0")
        return [{"item_id": v["item_id"], "speaker": v["speaker"],
                 "output": str(out_dir / f"{v['item_id']}.wav"),
                 "size_bytes": 0, "status": "failed",
                 "error": "pip install TTS>=0.22.0"} for v in items]
    except Exception as exc:
        print(f"[ERROR] XTTS model load failed: {exc}")
        return [{"item_id": v["item_id"], "speaker": v["speaker"],
                 "output": str(out_dir / f"{v['item_id']}.wav"),
                 "size_bytes": 0, "status": "failed", "error": str(exc)} for v in items]

    results = []
    total = len(items)

    for idx, vo in enumerate(items, start=1):
        out_filename = f"{vo['item_id']}.wav"
        out_path = out_dir / out_filename

        lang_code = vo.get("lang_code", "en-us")
        xtts_lang = XTTS_LANG_MAP.get(lang_code, "en")

        print(f"\n[{idx}/{total}] {vo['item_id']}")
        print(f"  Speaker    : {vo['speaker']}")
        print(f"  XTTS lang  : {xtts_lang}  (from lang_code={lang_code})")
        print(f"  Emotion    : {vo.get('emotion') or '(not set)'}")
        print(f"  Speed note : not supported by XTTS v2 — uses model prosody")
        print(f"  Text       : \"{vo['text'][:70]}{'...' if len(vo['text']) > 70 else ''}\"")


        try:
            ref_wav = ensure_xtts_ref_wav(vo["speaker"], xtts_lang)
            if ref_wav is None:
                raise RuntimeError(
                    "No reference WAV available. Place a WAV in "
                    f"{MODELS_DIR / 'reference_voices'} and retry."
                )

            print(f"  Ref WAV    : {ref_wav.name}")
            wav_list = tts_model.tts(
                text=vo["text"],
                speaker_wav=str(ref_wav),
                language=xtts_lang,
            )
            audio = np.array(wav_list, dtype=np.float32)
            sf.write(str(out_path), audio, XTTS_SAMPLE_RATE, subtype="PCM_16")
            size = out_path.stat().st_size
            duration_s = len(audio) / XTTS_SAMPLE_RATE
            model_name = f"XTTS v2 (coqui-ai/TTS, lang={xtts_lang}, ref={ref_wav.name})"
            write_license_sidecar(out_path, model_name)
            print(f"  [OK] {out_path}  ({duration_s:.2f}s, {size:,} bytes)")
            results.append({
                "item_id":      vo["item_id"],
                "speaker":      vo["speaker"],
                "xtts_lang":    xtts_lang,
                "ref_wav":      str(ref_wav),
                "output":       str(out_path),
                "size_bytes":   size,
                "duration_sec": round(duration_s, 3),
                "status":       "success",
            })
        except Exception as exc:
            print(f"  [ERROR] {vo['item_id']}: {exc}")
            results.append({
                "item_id": vo["item_id"], "speaker": vo["speaker"],
                "output": str(out_path), "size_bytes": 0,
                "status": "failed", "error": str(exc),
            })
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    del tts_model
    torch.cuda.empty_cache()
    gc.collect()
    return results


# =============================================================================
# MeloTTS backend
# =============================================================================

def load_melo(melo_lang: str):
    """
    Load MeloTTS for the given language.
    Requires: pip install melo-tts
    """
    from melo.api import TTS as MeloTTS  # noqa: PLC0415
    print(f"[MODEL] Loading MeloTTS  language={melo_lang}...")
    model = MeloTTS(language=melo_lang, device="auto")
    print("[MODEL] MeloTTS ready.")
    return model


def get_melo_speaker_id(model, speaker: str, melo_lang: str) -> int:
    """Resolve MeloTTS integer speaker ID from speaker name and language."""
    speaker_ids = model.hps.data.spk2id
    if melo_lang == "EN":
        accent = MELO_ACCENT_MAP.get(speaker, "EN-US")
        if accent in speaker_ids:
            return speaker_ids[accent]
    # Non-English or accent not found: use first available speaker
    return next(iter(speaker_ids.values()))


def run_melo_backend(items: list, out_dir: Path) -> list:
    """Run MeloTTS for all items. Returns result list."""
    # Group by melo_lang so we load one model per language
    lang_groups: dict[str, list] = {}
    unsupported = []
    for item in items:
        lang_code = item.get("lang_code", "en-us")
        melo_lang = MELO_LANG_MAP.get(lang_code)
        if melo_lang is None:
            print(f"  [WARN] {item['item_id']}: lang_code={lang_code} not supported by MeloTTS — skipping.")
            unsupported.append({
                "item_id": item["item_id"], "speaker": item["speaker"],
                "output": str(out_dir / f"{item['item_id']}.wav"),
                "size_bytes": 0, "status": "failed",
                "error": f"lang_code={lang_code} not supported by MeloTTS",
            })
        else:
            lang_groups.setdefault(melo_lang, []).append(item)

    # Warn once per language if the MeloTTS model is single-speaker (EN, JP, KR, FR, ES).
    # Only ZH is multi-speaker; all others produce the same voice regardless of speaker/gender.
    MELO_SINGLE_SPEAKER_LANGS = {"EN", "ZH", "JP", "KR", "FR", "ES"}
    for lang in set(lang_groups.keys()) & MELO_SINGLE_SPEAKER_LANGS:
        print(f"  [WARN] MeloTTS {lang} is single-speaker — all characters will sound identical "
              f"(same female voice). Use Kokoro or XTTS for gender/voice differentiation.")

    results = list(unsupported)
    total = len(items)
    item_index = 0

    for melo_lang, lang_items in lang_groups.items():
        try:
            model = load_melo(melo_lang)
        except ImportError:
            print("[ERROR] MeloTTS backend requires: pip install melo-tts")
            for vo in lang_items:
                results.append({
                    "item_id": vo["item_id"], "speaker": vo["speaker"],
                    "output": str(out_dir / f"{vo['item_id']}.wav"),
                    "size_bytes": 0, "status": "failed",
                    "error": "pip install melo-tts",
                })
            continue
        except Exception as exc:
            print(f"[ERROR] MeloTTS model load failed ({melo_lang}): {exc}")
            for vo in lang_items:
                results.append({
                    "item_id": vo["item_id"], "speaker": vo["speaker"],
                    "output": str(out_dir / f"{vo['item_id']}.wav"),
                    "size_bytes": 0, "status": "failed", "error": str(exc),
                })
            continue

        for vo in lang_items:
            item_index += 1
            out_filename = f"{vo['item_id']}.wav"
            out_path = out_dir / out_filename

            speaker_id = get_melo_speaker_id(model, vo["speaker"], melo_lang)
            # Reverse lookup: int -> accent label for logging
            spk2id = model.hps.data.spk2id
            accent = next((k for k, v in spk2id.items() if v == speaker_id), str(speaker_id))

            print(f"\n[{item_index}/{total}] {vo['item_id']}")
            print(f"  Speaker    : {vo['speaker']}")
            print(f"  MeloTTS    : language={melo_lang}, accent={accent}")
            print(f"  Emotion    : {vo.get('emotion') or '(not set)'}")
            print(f"  Speed      : {vo['speed']}  ({vo.get('speed_src', '')})")
            print(f"  Text       : \"{vo['text'][:70]}{'...' if len(vo['text']) > 70 else ''}\"")


            try:
                model.tts_to_file(vo["text"], speaker_id, str(out_path), speed=vo["speed"])
                size = out_path.stat().st_size
                model_name = f"MeloTTS (language={melo_lang}, accent={accent})"
                write_license_sidecar(out_path, model_name)
                print(f"  [OK] {out_path}  ({size:,} bytes)")
                results.append({
                    "item_id":   vo["item_id"],
                    "speaker":   vo["speaker"],
                    "melo_lang": melo_lang,
                    "accent":    accent,
                    "speed":     vo["speed"],
                    "output":    str(out_path),
                    "size_bytes": size,
                    "status":    "success",
                })
            except Exception as exc:
                print(f"  [ERROR] {vo['item_id']}: {exc}")
                results.append({
                    "item_id": vo["item_id"], "speaker": vo["speaker"],
                    "output": str(out_path), "size_bytes": 0,
                    "status": "failed", "error": str(exc),
                })
            finally:
                torch.cuda.empty_cache()
                gc.collect()

        del model
        torch.cuda.empty_cache()
        gc.collect()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    if args.manifest:
        with open(args.manifest, encoding="utf-8") as _f:
            _m = json.load(_f)
        locale = _m.get("locale") or locale_from_manifest_path(args.manifest)
        _locale_scope = _m.get("locale_scope")
    else:
        locale = 'en'
        _locale_scope = None

    if _locale_scope == "shared":
        raise SystemExit(
            "[ERROR] gen_tts.py received a shared manifest (locale_scope='shared'). "
            "Pass a locale manifest (locale_scope='locale') instead."
        )

    if not args.manifest:
        raise SystemExit(
            "[ERROR] --manifest is required. "
            "Pass --manifest <file> so the output path is derived from project_id/episode_id."
        )
    assets_dir = assets_dir_from_manifest(args.manifest)

    print(f"[OUTPUT] {assets_dir}")
    base_out_dir = assets_dir / locale   # WAVs:    assets/{locale}/audio/vo/
    meta_dir     = assets_dir / "meta"   # results: assets/meta/

    # Auto-detect voices file when --voices is not given and manifest locale is non-English.
    if not args.voices and locale.lower() not in ('en', 'en-us', 'en-gb'):
        auto_voices = Path(__file__).resolve().parent / f"voices_{locale}.json"
        if auto_voices.exists():
            args.voices = str(auto_voices)
            print(f"[VOICES] Auto-detected: {auto_voices.name}  (locale={locale})")
        else:
            print(f"[VOICES] No voices_{locale}.json found — using built-in lang defaults.")

    # Load voice profiles from --voices file (optional)
    voice_profiles = None
    if args.voices:
        if not Path(args.voices).exists():
            print(f"[ERROR] Voice profiles file not found: {args.voices}")
            return
        voice_profiles = load_voice_profiles(args.voices)
        desc = voice_profiles.get("description", args.voices)
        print(f"[VOICES] Loaded profiles: {desc}")
        if voice_profiles.get("speaker_voice_map"):
            print(f"         Speaker overrides : {voice_profiles['speaker_voice_map']}")
        if voice_profiles.get("locale_lang_map"):
            print(f"         Locale overrides  : {voice_profiles['locale_lang_map']}")
        if voice_profiles.get("speaker_speed_overrides"):
            print(f"         Speed overrides   : {voice_profiles['speaker_speed_overrides']}")
        print()

    # Load item list: manifest overrides hardcoded; --asset-id/--item_id both filter
    if args.manifest:
        items = load_from_manifest(args.manifest, args.asset_id or args.item_id, voice_profiles)
        if not items:
            print("[WARN] No matching vo_items in manifest. Nothing to do.")
            return
    else:
        items = VO_ITEMS
        filter_id = args.asset_id or args.item_id
        if filter_id:
            items = [v for v in VO_ITEMS if v["item_id"] == filter_id]
            if not items:
                print(f"[ERROR] item_id '{filter_id}' not found in VO_ITEMS.")
                return

    # Determine which backends to run
    if args.tts_model == "all":
        backends = ["kokoro", "xtts", "melo", "azure"]
    else:
        backends = [args.tts_model]

    all_results: dict[str, list] = {}

    for backend in backends:
        out_dir = base_out_dir / "audio" / "vo"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"BACKEND : {backend.upper()}")
        print(f"Output  : {out_dir}")
        print(f"{'='*60}")

        if backend == "kokoro":
            results = run_kokoro_backend(items, out_dir, args)
        elif backend == "xtts":
            results = run_xtts_backend(items, out_dir)
        elif backend == "melo":
            results = run_melo_backend(items, out_dir)
        elif backend == "azure":
            results = run_azure_backend(items, out_dir)
        else:
            results = []

        # Write per-backend results manifest to assets/meta/
        meta_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = meta_dir / f"{SCRIPT_NAME}_{backend}_results.json"
        with open(manifest_path, "w") as fh:
            json.dump(results, fh, indent=2)

        all_results[backend] = results

        ok_count   = sum(1 for r in results if r.get("status") == "success")
        total_bytes = sum(r.get("size_bytes", 0) for r in results)
        print(f"\n{ok_count}/{len(results)} completed | {total_bytes:,} bytes total")
        print(f"Manifest: {manifest_path}")

    # Cross-backend comparison table (only when running multiple backends)
    if len(backends) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        for backend, results in all_results.items():
            ok         = sum(1 for r in results if r.get("status") == "success")
            tot_bytes  = sum(r.get("size_bytes", 0) for r in results)
            print(f"  {backend.upper():8s}: {ok}/{len(results)} files | {tot_bytes:,} bytes")
        print(f"\nListen and compare in: {assets_dir}")


if __name__ == "__main__":
    main()
