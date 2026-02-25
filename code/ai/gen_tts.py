# =============================================================================
# gen_tts.py
# Generate spoken voice-over audio for every dialogue line in s01e01.
# =============================================================================
#
# requirements.txt (pip install before running):
#   kokoro>=0.9.4
#   soundfile>=0.12.0
#   numpy>=1.24.0
#   huggingface_hub>=0.21.0
#
# Language-specific extras (install the ones you need):
#   Chinese  (zh-Hans / zh-Hant):  pip install "misaki[zh]"
#             pulls in: pypinyin, pypinyin-dict, cn2an, jieba, ordered-set
#   Japanese (ja):                  pip install "misaki[ja]"
#   Korean   (ko):                  pip install "misaki[ko]"  (if supported)
#
# ---------------------------------------------------------------------------
# Hardware Target: NVIDIA RTX 4060 8 GB VRAM
# ---------------------------------------------------------------------------
# Memory-saving techniques:
#   Kokoro-82M is a tiny 82-million-parameter TTS model.  It runs entirely
#   on CPU at real-time speed and consumes well under 1 GB RAM.  No GPU is
#   required.  No VRAM techniques are needed — the entire model comfortably
#   fits in system RAM.
#
#   torch.cuda.empty_cache() is still called after the loop as good practice
#   in case the user is running this script alongside GPU-heavy processes.
#
# NOTE: Kokoro downloads model weights from HuggingFace (hexgrad/Kokoro-82M)
#   automatically on first run.  No licence agreement is required.
# ---------------------------------------------------------------------------

import argparse
import gc
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# ---------------------------------------------------------------------------
# DEFAULTS — fully populated; script runs with no CLI flags.
# ---------------------------------------------------------------------------
# Resolve output dir relative to this script's location so it works regardless
# of the working directory the script is launched from.
# code/ai/gen_tts.py -> ../../projects/...  (repo root / projects / ...)
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "projects" / "the-pharaoh-who-defied-death" / "episodes" / "s01e01" / "assets"
SCRIPT_NAME = "gen_tts"

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
    "ko":       "k",        # Korean  (if supported by installed Kokoro version)
    "fr":       "f",        # French
    "es":       "e",        # Spanish
    "pt":       "p",        # Portuguese
    "hi":       "h",        # Hindi
    "it":       "i",        # Italian
}

PACE_TO_SPEED = {"slow": 0.8, "normal": 1.0, "fast": 1.2}

# Built-in voice maps for non-English lang_codes.
# Used automatically when the manifest locale maps to a non-English lang_code
# and no --voices file is provided.  Keys are Kokoro lang_codes.
LANG_CODE_DEFAULT_VOICES = {
    "z": {                              # Mandarin Chinese
        "amunhotep":     "zm_yunxi",    # male, mid-aged
        "ramesses_ka":   "zm_yunjian",  # male, commanding
        "neferet":       "zf_xiaobei",  # female, young
        "khamun":        "zm_yunxia",   # male, deep
        "voice_of_gate": "zm_yunxi",    # closest available for supernatural
    },
}


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
        "description":           raw.get("description", ""),
        "speaker_voice_map":     raw.get("speaker_voice_map", {}),
        "speaker_speed_overrides": raw.get("speaker_speed_overrides", {}),
        "locale_lang_map":       raw.get("locale_lang_map", {}),
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
            "Generate voice-over WAV files using Kokoro-82M.\n"
            "Supports any language via --voices <profile.json>.\n\n"
            "Examples:\n"
            "  python gen_tts.py --manifest AssetManifest_draft.json\n"
            "  python gen_tts.py --manifest AssetManifest_draft.zh-Hans.json "
            "--voices voices_zh-Hans.json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output_dir", type=str, default=None)
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
      Keys that can be overridden:
        speaker_voice_map       — wins over built-in SPEAKER_TO_VOICE
        speaker_speed_overrides — wins over built-in SPEAKER_SPEED_OVERRIDES
        locale_lang_map         — wins over built-in LOCALE_TO_LANG_CODE
        voice_style_rules       — replaces built-in VOICE_STYLE_RULES entirely
    """
    p = voice_profiles or {}

    # Build effective lookup tables (profiles layer on top of built-ins)
    eff_speaker_map    = {**SPEAKER_TO_VOICE,         **p.get("speaker_voice_map", {})}
    eff_speed_override = {**SPEAKER_SPEED_OVERRIDES,  **p.get("speaker_speed_overrides", {})}
    eff_locale_map     = {**LOCALE_TO_LANG_CODE,      **p.get("locale_lang_map", {})}
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
            **SPEAKER_TO_VOICE,          # English base (lowest priority)
            **locale_voice_defaults,     # lang-code defaults (e.g. Chinese voices)
            **p.get("speaker_voice_map", {}),  # explicit --voices file (highest priority)
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
            # Fall back to keyword matching against voice_style description
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
# Kokoro pipeline loader
# ---------------------------------------------------------------------------
def load_kokoro(lang_code: str = "en-us"):
    """Load the Kokoro KPipeline for the given lang_code.  Downloads on first run."""
    from kokoro import KPipeline
    print(f"[MODEL] Loading Kokoro-82M  lang_code={lang_code}  (hexgrad/Kokoro-82M)...")
    pipe = KPipeline(lang_code=lang_code)
    print("[MODEL] Kokoro ready.")
    return pipe


# ---------------------------------------------------------------------------
# Audio generation helpers
# ---------------------------------------------------------------------------
def synthesise(pipe, text: str, voice: str, speed: float) -> np.ndarray:
    """
    Run Kokoro inference and concatenate all audio chunks into one array.
    Kokoro splits long texts internally; we join the chunks back together.
    """
    chunks = []
    # split_pattern=None lets Kokoro decide sentence boundaries
    for _gs, _ps, audio_chunk in pipe(text, voice=voice, speed=speed):
        if audio_chunk is not None and len(audio_chunk) > 0:
            # audio_chunk is a numpy array or torch tensor at SAMPLE_RATE Hz
            if hasattr(audio_chunk, "numpy"):
                audio_chunk = audio_chunk.numpy()
            chunks.append(np.array(audio_chunk, dtype=np.float32))
    if not chunks:
        raise RuntimeError("Kokoro returned no audio chunks.")
    return np.concatenate(chunks)


def locale_from_manifest_path(path: str) -> str:
    """Extract locale from manifest filename.
    'AssetManifest_draft.zh-Hans.json' -> 'zh-Hans'
    'AssetManifest_draft.json'          -> 'en'
    """
    stem = Path(path).stem
    parts = stem.split('.')
    return parts[-1] if len(parts) > 1 else 'en'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    locale = locale_from_manifest_path(args.manifest) if args.manifest else 'en'
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR / locale
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect voices file when --voices is not given and manifest locale is non-English.
    # e.g. AssetManifest_draft.zh-Hans.json -> looks for voices_zh-Hans.json next to the script.
    # Falls back gracefully to LANG_CODE_DEFAULT_VOICES built-ins if the file is absent.
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

    # Load item list: manifest overrides hardcoded, --asset-id/--item_id both filter
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

    # Group items by lang_code so we load one Kokoro pipeline per locale.
    # For this manifest all items are "en-us", but this handles mixed-locale
    # manifests correctly without extra pipeline loads.
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

            # --- log how tts_prompt drove this generation ---
            print(f"\n[{item_index}/{total}] {vo['item_id']}")
            print(f"  Speaker    : {vo['speaker']}")
            print(f"  Voice style: {vo.get('voice_style') or '(not set)'}")
            print(f"  Voice ID   : {vo['voice']}  (source: {vo.get('voice_src', 'unknown')})")
            print(f"  Emotion    : {vo.get('emotion') or '(not set)'}")
            print(f"  Speed      : {vo['speed']}  ({vo.get('speed_src', '')})")
            print(f"  Locale     : {vo.get('locale', 'en')}  ->  lang_code={lang_code}")
            print(f"  Text       : \"{vo['text'][:70]}{'...' if len(vo['text']) > 70 else ''}\"")

            if out_path.exists():
                print(f"  [SKIP] {out_filename} already exists")
                results.append({
                    "item_id":   vo["item_id"],
                    "speaker":   vo["speaker"],
                    "voice":     vo["voice"],
                    "output":    str(out_path),
                    "size_bytes": out_path.stat().st_size,
                    "status":    "skipped",
                })
                continue

            try:
                audio = synthesise(pipe, vo["text"], voice=vo["voice"], speed=vo["speed"])
                sf.write(str(out_path), audio, SAMPLE_RATE, subtype="PCM_16")
                size = out_path.stat().st_size
                duration_s = len(audio) / SAMPLE_RATE
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
                    "item_id":   vo["item_id"],
                    "speaker":   vo["speaker"],
                    "voice":     vo["voice"],
                    "output":    str(out_path),
                    "size_bytes": 0,
                    "status":    "failed",
                    "error":     str(exc),
                })

        # Release pipeline before loading the next lang_code's pipeline
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    # Write manifest
    manifest_path = out_dir / f"{SCRIPT_NAME}_results.json"
    with open(manifest_path, "w") as fh:
        json.dump(results, fh, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — gen_tts")
    print("=" * 60)
    for r in results:
        tag = "OK" if r["status"] == "success" else r["status"].upper()
        dur = f"  {r.get('duration_sec', 0):.2f}s" if "duration_sec" in r else ""
        print(f"  [{tag}]  {r['output']}{dur}  ({r['size_bytes']:,} bytes)")
    ok_count = sum(1 for r in results if r["status"] in ("success", "skipped"))
    total_bytes = sum(r["size_bytes"] for r in results)
    print(f"\n{ok_count}/{total} completed | {total_bytes:,} bytes total")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
