#!/usr/bin/env python3
# =============================================================================
# gen_tts_cloud.py
# Azure Neural TTS backend for the AI Asset Generation Pipeline.
# CPU-only; no GPU or local model required.
#
# Reads vo_items from an AssetManifest locale manifest and synthesises WAV
# files using Azure Cognitive Services Speech SDK.
#
# Install:
#   pip install azure-cognitiveservices-speech>=1.38.0
#
# Configuration (environment variables — never hardcode credentials):
#   AZURE_SPEECH_KEY     Azure Cognitive Services subscription key (required)
#   AZURE_SPEECH_REGION  Azure region, e.g. "eastus" (required)
#   AZURE_ENDPOINT       Custom endpoint URL (optional; overrides region routing)
#
# Usage:
#   python gen_tts_cloud.py --manifest AssetManifest_draft.en.json
#   python gen_tts_cloud.py --manifest AssetManifest_draft.zh-Hans.json
#   python gen_tts_cloud.py --manifest AssetManifest_draft.en.json --asset-id vo-s01e02-sc01-001
#
# Output:
#   projects/{project_id}/episodes/{episode_id}/assets/{locale}/audio/vo/{item_id}.wav
#   projects/{project_id}/episodes/{episode_id}/assets/{locale}/audio/vo/licenses/{item_id}.license.json
#   projects/{project_id}/episodes/{episode_id}/assets/meta/gen_tts_cloud_results.json
#
# tts_prompt fields used (in priority order):
#   azure_voice        → <voice name='...'>                (explicit; overrides speaker lookup)
#   azure_style        → <mstts:express-as style='...'>    (explicit; overrides emotion mapping)
#   azure_style_degree → styledegree='...'                 (explicit; overrides default 1.5)
#   azure_rate         → <prosody rate='...'>              (explicit; overrides pace mapping)
#   voice_style        → gender fallback for speaker lookup
#   emotion            → auto-mapped to Azure style via keyword rules
#   pace               → auto-mapped to prosody rate (slow=-25%, normal=0%, fast=+25%)
#   locale             → Azure xml:lang derivation
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# =============================================================================
# Paths
# =============================================================================
PIPE_DIR      = Path(__file__).resolve().parent.parent.parent
PROJECTS_ROOT = PIPE_DIR / "projects"
SCRIPT_NAME   = "gen_tts_cloud"

# =============================================================================
# Output format — Riff24Khz16BitMonoPcm matches Kokoro/XTTS for easy A/B
# =============================================================================
AZURE_SAMPLE_RATE = 24000   # Hz

# =============================================================================
# Locale → Azure xml:lang
# =============================================================================
LOCALE_TO_AZURE_LANG: dict[str, str] = {
    "en":       "en-US",
    "en-us":    "en-US",
    "en-gb":    "en-GB",
    "zh":       "zh-CN",
    "zh-hans":  "zh-CN",
    "zh-hant":  "zh-TW",
    "zh-cn":    "zh-CN",
    "zh-tw":    "zh-TW",
    "ja":       "ja-JP",
    "ko":       "ko-KR",
    "fr":       "fr-FR",
    "es":       "es-ES",
    "pt":       "pt-BR",
    "hi":       "hi-IN",
    "it":       "it-IT",
    "de":       "de-DE",
    "ar":       "ar-EG",
}

# =============================================================================
# Speaker → Azure voice name, keyed by language prefix
#
# Populated with voices known to work well for this project's characters.
# Override at runtime via tts_prompt.azure_voice in the manifest.
# =============================================================================
AZURE_SPEAKER_VOICE: dict[str, dict[str, str]] = {
    "en": {
        "amunhotep":     "en-US-DavisNeural",    # deep older male, haunted
        "ramesses_ka":   "en-US-TonyNeural",     # commanding, imperious
        "neferet":       "en-US-AriaNeural",     # young intelligent female
        "khamun":        "en-US-GuyNeural",      # military, stern
        "voice_of_gate": "en-US-DavisNeural",   # deep, supernatural cadence
    },
    "zh": {
        "amunhotep":     "zh-CN-YunxiNeural",
        "ramesses_ka":   "zh-CN-Yunyi:DragonHDFlashLatestNeural",
        "neferet":       "zh-CN-Xiaoxiao:DragonHDFlashLatestNeural",
        "khamun":        "zh-CN-Yunxia:DragonHDFlashLatestNeural",
        "voice_of_gate": "zh-CN-Yunyi:DragonHDFlashLatestNeural",
    },
    "ja": {
        "amunhotep":     "ja-JP-KeitaNeural",
        "ramesses_ka":   "ja-JP-DaichiNeural",
        "neferet":       "ja-JP-NanamiNeural",
        "khamun":        "ja-JP-KeitaNeural",
        "voice_of_gate": "ja-JP-KeitaNeural",
    },
}

# Default voice per gender when the speaker is not in AZURE_SPEAKER_VOICE
AZURE_DEFAULT_VOICE: dict[str, dict[str, str]] = {
    "en": {"male": "en-US-DavisNeural",      "female": "en-US-AriaNeural"},
    "zh": {"male": "zh-CN-Yunyi:DragonHDFlashLatestNeural",
           "female": "zh-CN-Xiaoxiao:DragonHDFlashLatestNeural"},
    "ja": {"male": "ja-JP-KeitaNeural",      "female": "ja-JP-NanamiNeural"},
    "ko": {"male": "ko-KR-InJoonNeural",     "female": "ko-KR-SunHiNeural"},
    "fr": {"male": "fr-FR-HenriNeural",      "female": "fr-FR-DeniseNeural"},
    "es": {"male": "es-ES-AlvaroNeural",     "female": "es-ES-ElviraNeural"},
    "de": {"male": "de-DE-ConradNeural",     "female": "de-DE-KatjaNeural"},
    "it": {"male": "it-IT-DiegoNeural",      "female": "it-IT-ElsaNeural"},
    "pt": {"male": "pt-BR-AntonioNeural",    "female": "pt-BR-FranciscaNeural"},
    "hi": {"male": "hi-IN-MadhurNeural",     "female": "hi-IN-SwaraNeural"},
}

# =============================================================================
# pace → Azure prosody rate
# =============================================================================
AZURE_PACE_RATE: dict[str, str] = {
    "slow":   "-25%",
    "normal": "0%",
    "fast":   "+25%",
}

# =============================================================================
# Emotion keyword → Azure express-as style
#
# Matched by substring against emotion.lower(). First match wins.
# Styles not supported by a given voice are silently ignored by Azure.
# =============================================================================
AZURE_EMOTION_RULES: list[tuple[str, str]] = [
    ("whisper",    "whispering"),
    ("terrif",     "terrified"),
    ("desperate",  "terrified"),
    ("horror",     "fearful"),
    ("dread",      "fearful"),
    ("fear",       "fearful"),
    ("urgent",     "excited"),
    ("excit",      "excited"),
    ("anger",      "angry"),
    ("angry",      "angry"),
    ("fury",       "angry"),
    ("suppress",   "angry"),
    ("shout",      "shouting"),
    ("cold",       "unfriendly"),
    ("contempt",   "disgruntled"),
    ("dismissiv",  "disgruntled"),
    ("imperiou",   "disgruntled"),
    ("bitter",     "disgruntled"),
    ("disgrun",    "disgruntled"),
    ("sovereign",  "serious"),
    ("command",    "serious"),
    ("authorit",   "serious"),
    ("grave",      "serious"),
    ("ominous",    "serious"),
    ("solemn",     "serious"),
    ("trance",     "serious"),
    ("warning",    "serious"),
    ("ancient",    "serious"),
    ("grief",      "sad"),
    ("haunt",      "sad"),
    ("reluct",     "sad"),
    ("depress",    "sad"),
    ("trouble",    "sad"),
    ("resolv",     "sad"),
    ("sad",        "sad"),
    ("breathless", "fearful"),
    ("hope",       "hopeful"),
    ("cheerful",   "cheerful"),
    ("calm",       "calm"),
    ("flat",       "calm"),
    ("curiosit",   "calm"),
    ("fearless",   "calm"),
    ("certain",    "unfriendly"),
    ("conflict",   "disgruntled"),
    ("challeng",   "disgruntled"),
]

# Default style degree when not specified in tts_prompt
DEFAULT_STYLE_DEGREE = 1.5

# =============================================================================
# voice_style keyword → inferred gender (for speaker lookup fallback)
# =============================================================================
_GENDER_RULES: list[tuple[str, str]] = [
    ("female", "female"), ("woman", "female"), ("girl", "female"),
    ("male",   "male"),   ("man",   "male"),   ("boy",  "male"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_azure_voice(speaker: str, voice_style: str, azure_lang: str) -> str:
    """Return an Azure voice name for speaker + locale.

    Priority:
      1. AZURE_SPEAKER_VOICE[lang_prefix][speaker]
      2. AZURE_DEFAULT_VOICE[lang_prefix][gender]  (gender from voice_style keywords)
      3. First entry in AZURE_DEFAULT_VOICE[lang_prefix]
      4. en-US-DavisNeural  (absolute fallback)
    """
    lang_prefix = azure_lang.split("-")[0].lower()

    speaker_map = AZURE_SPEAKER_VOICE.get(lang_prefix, {})
    if speaker in speaker_map:
        return speaker_map[speaker]

    gender = "male"
    for kw, g in _GENDER_RULES:
        if kw in voice_style.lower():
            gender = g
            break

    defaults = AZURE_DEFAULT_VOICE.get(lang_prefix, {})
    return defaults.get(gender) or next(iter(defaults.values()), "en-US-DavisNeural")


def resolve_azure_style(emotion: str) -> str | None:
    """Map an emotion string to an Azure express-as style via keyword matching.

    Returns None when no keyword matches (style element omitted from SSML).
    """
    if not emotion:
        return None
    lower = emotion.lower()
    for keyword, style in AZURE_EMOTION_RULES:
        if keyword in lower:
            return style
    return None


def build_ssml(
    text: str,
    voice: str,
    azure_lang: str,
    rate: str,
    style: str | None,
    style_degree: float,
    duration_sec: float | None = None,
) -> str:
    """Build Azure SSML for a single utterance.

    Layer order (outermost → innermost):
      <voice> → <mstts:express-as> → <prosody> → text

    Rules:
    - duration_sec overrides rate when both are set (used for timed-shot fitting)
    - express-as element is omitted entirely when style is None
    - prosody element is omitted when rate is "0%" and duration_sec is None
    """
    escaped = (text
               .replace("&", "&amp;")
               .replace("<", "&lt;")
               .replace(">", "&gt;")
               .replace('"', "&quot;")
               .replace("'", "&apos;"))

    # Prosody
    if duration_sec is not None:
        prosody_attr = f'duration="{duration_sec:.3f}s"'
    elif rate and rate != "0%":
        prosody_attr = f'rate="{rate}"'
    else:
        prosody_attr = ""

    spoken = f"<prosody {prosody_attr}>{escaped}</prosody>" if prosody_attr else escaped

    # Style
    if style:
        spoken = (
            f'<mstts:express-as style="{style}" styledegree="{style_degree}">'
            f"{spoken}</mstts:express-as>"
        )

    return (
        f"<speak version='1.0' xml:lang='{azure_lang}' "
        f"xmlns='http://www.w3.org/2001/10/synthesis' "
        f"xmlns:mstts='http://www.w3.org/2001/mstts'>"
        f"<voice name='{voice}'>{spoken}</voice>"
        f"</speak>"
    )


def write_license_sidecar(wav_path: Path, voice_name: str, style: str | None) -> None:
    """Write a CC0 license sidecar JSON alongside the WAV file."""
    licenses_dir = wav_path.parent / "licenses"
    licenses_dir.mkdir(parents=True, exist_ok=True)
    model_str = f"Azure Neural TTS (voice={voice_name}"
    if style:
        model_str += f", style={style}"
    model_str += ")"
    sidecar = {
        "spdx_id":              "CC0",
        "attribution_required": False,
        "text":                 f"AI-generated voice audio. No copyright claimed. Produced by {model_str}.",
        "generator_model":      model_str,
    }
    sidecar_path = licenses_dir / f"{wav_path.stem}.license.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def locale_from_manifest(manifest: dict, path: str) -> str:
    """Read locale from manifest field first; fall back to filename parsing.

    Manifest field (locale_scope='locale' manifests always have locale set):
      manifest["locale"] = "zh-Hans"  →  "zh-Hans"

    Filename fallback:
      AssetManifest_draft.zh-Hans.json  →  "zh-Hans"
      AssetManifest_draft.en.json       →  "en"
      AssetManifest_draft.json          →  "en"
    """
    if manifest.get("locale"):
        return manifest["locale"]
    stem = Path(path).stem          # e.g. "AssetManifest_draft.zh-Hans"
    parts = stem.split(".")
    return parts[-1] if len(parts) > 1 else "en"


def assets_dir_from_manifest(manifest: dict) -> Path:
    """Derive the assets output directory from manifest project_id / episode_id.

    Prefers the direct episode_id field; falls back to parsing from manifest_id
    for legacy manifests that predate the episode_id field.
    """
    project_id = manifest.get("project_id", "")
    episode_id = manifest.get("episode_id", "")

    # Legacy fallback: parse episode_id from manifest_id string
    if not episode_id:
        manifest_id = manifest.get("manifest_id", "")
        episode_id = manifest_id
        if episode_id.startswith(project_id + "-"):
            episode_id = episode_id[len(project_id) + 1:]
        if episode_id.endswith("-manifest"):
            episode_id = episode_id[: -len("-manifest")]
        # Handle locale suffix: "s01e02-en" → "s01e02"
        if episode_id and not episode_id[:6].replace("s", "").replace("e", "").isdigit():
            episode_id = ""

    if not project_id or not episode_id:
        raise SystemExit(
            f"[ERROR] Cannot derive output path — manifest is missing "
            f"'project_id' and/or 'episode_id'."
        )

    return PROJECTS_ROOT / project_id / "episodes" / episode_id / "assets"


def load_items_from_manifest(manifest: dict, path: str, asset_id_filter: str | None) -> list[dict]:
    """Load and resolve VO items from a locale AssetManifest.

    For each vo_item, resolves Azure voice/style/rate in this priority order:
      1. Explicit azure_* fields in tts_prompt  (authored by LLM / Stage 5)
      2. Auto-derived from voice_style / emotion / pace  (fallback)
    """
    locale     = locale_from_manifest(manifest, path)
    azure_lang = LOCALE_TO_AZURE_LANG.get(locale.lower(), "en-US")

    items = []
    for vo in manifest.get("vo_items", []):
        if asset_id_filter and vo["item_id"] != asset_id_filter:
            continue

        tts          = vo.get("tts_prompt", {})
        voice_style  = tts.get("voice_style", "")
        emotion      = tts.get("emotion", "")
        pace         = tts.get("pace", "normal")

        # ── Voice resolution (explicit > speaker lookup > gender default) ──
        voice = (
            tts.get("azure_voice")
            or resolve_azure_voice(vo["speaker_id"], voice_style, azure_lang)
        )

        # ── Style resolution (explicit > emotion mapping) ──
        style = (
            tts.get("azure_style")
            or resolve_azure_style(emotion)
        )

        # ── Style degree (explicit > default 1.5) ──
        style_degree = tts.get("azure_style_degree") or DEFAULT_STYLE_DEGREE

        # ── Rate resolution (explicit > pace mapping) ──
        rate = tts.get("azure_rate") or AZURE_PACE_RATE.get(pace, "0%")

        items.append({
            "item_id":      vo["item_id"],
            "speaker":      vo["speaker_id"],
            "text":         vo["text"],
            "locale":       locale,
            "azure_lang":   azure_lang,
            "voice":        voice,
            "voice_style":  voice_style,
            "emotion":      emotion,
            "style":        style,
            "style_degree": style_degree,
            "rate":         rate,
        })

    return items


# ---------------------------------------------------------------------------
# Azure synthesizer
# ---------------------------------------------------------------------------

def load_azure_synthesizer():
    """Create an Azure SpeechSynthesizer configured for in-memory WAV output."""
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError:
        raise ImportError(
            "Azure Speech SDK not installed. "
            "Run: pip install azure-cognitiveservices-speech>=1.38.0"
        )

    key    = os.environ.get("AZURE_SPEECH_KEY", "")
    region = os.environ.get("AZURE_SPEECH_REGION", "")
    endpoint = os.environ.get("AZURE_ENDPOINT", "")

    if not key:
        raise SystemExit(
            "[ERROR] AZURE_SPEECH_KEY environment variable is not set.\n"
            "        export AZURE_SPEECH_KEY='your-key-here'"
        )
    if not region and not endpoint:
        raise SystemExit(
            "[ERROR] AZURE_SPEECH_REGION environment variable is not set.\n"
            "        export AZURE_SPEECH_REGION='eastus'"
        )

    if endpoint:
        speech_config = speechsdk.SpeechConfig(endpoint=endpoint, subscription=key)
    else:
        speech_config = speechsdk.SpeechConfig(subscription=key, region=region)

    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
    )

    # audio_config=None → WAV bytes returned in memory, no audio device needed
    return speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)


def synthesise(synthesizer, ssml: str) -> bytes:
    """Submit SSML to Azure and return raw WAV bytes.

    Raises RuntimeError on API failure with the cancellation error details.
    """
    import azure.cognitiveservices.speech as speechsdk

    result = synthesizer.speak_ssml_async(ssml).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return result.audio_data
    details = result.cancellation_details
    raise RuntimeError(
        f"Azure TTS cancelled: reason={details.reason}  detail={details.error_details}"
    )


# ---------------------------------------------------------------------------
# Backend runner
# ---------------------------------------------------------------------------

def run(items: list[dict], out_dir: Path) -> list[dict]:
    """Synthesise all items with Azure TTS. Returns a list of result dicts."""
    synthesizer = load_azure_synthesizer()

    results = []
    total   = len(items)

    for idx, vo in enumerate(items, start=1):
        out_path = out_dir / f"{vo['item_id']}.wav"

        print(f"\n[{idx}/{total}] {vo['item_id']}")
        print(f"  Speaker      : {vo['speaker']}")
        print(f"  Voice        : {vo['voice']}  (lang={vo['azure_lang']})")
        print(f"  Voice style  : {vo['voice_style'] or '(not set)'}")
        print(f"  Emotion      : {vo['emotion'] or '(not set)'}  →  style={vo['style'] or 'none'}  degree={vo['style_degree']}")
        print(f"  Rate         : {vo['rate']}")
        print(f"  Text         : \"{vo['text'][:80]}{'...' if len(vo['text']) > 80 else ''}\"")

        ssml = build_ssml(
            text=vo["text"],
            voice=vo["voice"],
            azure_lang=vo["azure_lang"],
            rate=vo["rate"],
            style=vo["style"],
            style_degree=vo["style_degree"],
        )

        try:
            wav_bytes = synthesise(synthesizer, ssml)

            if out_path.exists():
                out_path.unlink()
            out_path.write_bytes(wav_bytes)

            size       = out_path.stat().st_size
            pcm_bytes  = max(0, size - 44)          # subtract 44-byte WAV header
            duration_s = pcm_bytes / (AZURE_SAMPLE_RATE * 2)   # 16-bit mono

            write_license_sidecar(out_path, vo["voice"], vo["style"])
            print(f"  [OK] {out_path}  ({duration_s:.2f}s, {size:,} bytes)")

            results.append({
                "item_id":      vo["item_id"],
                "speaker":      vo["speaker"],
                "voice":        vo["voice"],
                "azure_lang":   vo["azure_lang"],
                "style":        vo["style"],
                "style_degree": vo["style_degree"],
                "rate":         vo["rate"],
                "output":       str(out_path),
                "size_bytes":   size,
                "duration_sec": round(duration_s, 3),
                "status":       "success",
            })

        except Exception as exc:
            print(f"  [ERROR] {vo['item_id']}: {exc}")
            results.append({
                "item_id":    vo["item_id"],
                "speaker":    vo["speaker"],
                "voice":      vo["voice"],
                "output":     str(out_path),
                "size_bytes": 0,
                "status":     "failed",
                "error":      str(exc),
            })

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Azure Neural TTS voice-over generator (CPU-only, no GPU required).\n\n"
            "Reads vo_items from a locale AssetManifest and synthesises WAV files.\n"
            "Maps tts_prompt fields to Azure SSML: voice, express-as style, prosody rate.\n\n"
            "Credentials via environment variables:\n"
            "  export AZURE_SPEECH_KEY='your-key'\n"
            "  export AZURE_SPEECH_REGION='eastus'\n\n"
            "Examples:\n"
            "  python gen_tts_cloud.py --manifest AssetManifest_draft.en.json\n"
            "  python gen_tts_cloud.py --manifest AssetManifest_draft.zh-Hans.json\n"
            "  python gen_tts_cloud.py --manifest AssetManifest_draft.en.json "
            "--asset-id vo-s01e02-sc01-001\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest", type=str, required=True,
        help="Path to a locale AssetManifest JSON (locale_scope='locale' or 'monolithic').",
    )
    parser.add_argument(
        "--asset-id", type=str, default=None, dest="asset_id",
        help="Process only the item with this item_id (default: all vo_items).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)

    # Guard: reject shared manifests early
    if manifest.get("locale_scope") == "shared":
        raise SystemExit(
            "[ERROR] Passed a shared manifest to gen_tts_cloud.py — "
            "use a locale manifest (locale_scope='locale') instead."
        )

    locale     = locale_from_manifest(manifest, args.manifest)
    assets_dir = assets_dir_from_manifest(manifest)
    out_dir    = assets_dir / locale / "audio" / "vo"
    meta_dir   = assets_dir / "meta"

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ASSETS] {assets_dir}")
    print(f"[OUTPUT] {out_dir}")

    items = load_items_from_manifest(manifest, args.manifest, args.asset_id)
    if not items:
        print("[WARN] No matching vo_items found in manifest. Nothing to do.")
        return

    print(f"\n{'='*60}")
    print(f"BACKEND  : AZURE NEURAL TTS (cloud, no GPU)")
    print(f"Locale   : {locale}  →  {LOCALE_TO_AZURE_LANG.get(locale.lower(), 'en-US')}")
    print(f"Items    : {len(items)}")
    print(f"Output   : {out_dir}")
    print(f"{'='*60}")

    results = run(items, out_dir)

    results_path = meta_dir / f"{SCRIPT_NAME}_results.json"
    if results_path.exists():
        results_path.unlink()
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    ok_count    = sum(1 for r in results if r.get("status") == "success")
    total_bytes = sum(r.get("size_bytes", 0) for r in results)
    failed      = [r for r in results if r.get("status") != "success"]

    print(f"\n{'='*60}")
    print(f"{ok_count}/{len(results)} completed | {total_bytes:,} bytes total")
    if failed:
        print(f"FAILED ({len(failed)}):")
        for r in failed:
            print(f"  {r['item_id']}: {r.get('error', '?')}")
    print(f"Results  : {results_path}")


if __name__ == "__main__":
    main()
