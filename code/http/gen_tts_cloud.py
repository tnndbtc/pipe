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
# Voice selection priority:
#   1. tts_prompt.azure_voice          explicit voice name — highest priority
#   2. character_packs[].gender        set by Stage 5 from Script.json cast[].gender
#   3. tts_prompt.voice_style keywords fallback for manifests without gender field
#   4. AZURE_DEFAULT_VOICE[lang][male] hard default
#
# tts_prompt fields used:
#   azure_voice        → <voice name='...'>                (explicit; overrides all)
#   azure_style        → <mstts:express-as style='...'>    (explicit; overrides emotion mapping)
#   azure_style_degree → styledegree='...'                 (explicit; overrides default 1.5)
#   azure_rate         → <prosody rate='...'>              (explicit; overrides pace mapping)
#   azure_pitch        → <prosody pitch='...'>             (explicit; e.g. '-10%', '+5%')
#   azure_break_ms     → <break time='Nms'/> after each sentence terminator (0 = disabled)
#   voice_style        → secondary gender hint + vocal quality description
#   emotion            → auto-mapped to Azure style via keyword rules
#   pace               → auto-mapped to prosody rate (slow=-25%, normal=0%, fast=+25%)
#   locale             → Azure xml:lang derivation
# =============================================================================

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import struct
import sys
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

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

# Default voice per gender — resolved from character_packs[].gender (set by Stage 5
# from Script.json cast[].gender).  Use tts_prompt.azure_voice in the manifest to
# pin a specific voice for a character.
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

# =============================================================================
# Gender detection from character_pack descriptions (ai_prompt / search_prompt)
# Word-boundary patterns — avoids false matches in words like "commanding".
# =============================================================================
_GENDER_DETECT_RE: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\b(?:woman|female|girl)\b', re.IGNORECASE), "female"),
    (re.compile(r'\b(?:male|man|boy)\b',      re.IGNORECASE), "male"),
]


def _norm_speaker(name: str) -> str:
    """Normalise a speaker name to a lowercase alphanumeric key.

    Strips spaces, underscores, dots and hyphens so that 'Dr. Hale', 'Dr_Hale'
    and 'drhale' all map to the same canonical key 'drhale'.
    """
    return re.sub(r'[\s._\-]', '', name).lower()


def build_gender_map_from_character_packs(character_packs: list) -> dict[str, str]:
    """Return a normalised speaker_id → 'male'|'female' map.

    Keys are produced by _norm_speaker() so they match regardless of whether
    the manifest uses underscores (asset_id: 'Dr_Hale') or spaces/dots
    (speaker_id: 'Dr. Hale').

    Priority per character_pack:
      1. Explicit 'gender' field  ("male"|"female"|"neutral") — set by Stage 5 LLM
      2. Text scan of search_prompt + ai_prompt for gender keywords (legacy fallback)

    'neutral' is intentionally omitted from the map so the voice_style keyword
    check in resolve_azure_voice() can still apply (e.g. a narrator voice).
    """
    gender_map: dict[str, str] = {}
    for cp in character_packs:
        asset_id = cp.get("asset_id", "")
        if not asset_id:
            continue

        key = _norm_speaker(asset_id)

        # 1. Explicit gender field (reliable — always use this when present)
        explicit = cp.get("gender", "")
        if explicit in ("male", "female"):
            gender_map[key] = explicit
            continue
        # "neutral" → skip (leave out of map; voice_style keywords can still match)

        # 2. Legacy fallback: scan description text for gender keywords
        text = " ".join(filter(None, [
            cp.get("search_prompt", ""),
            cp.get("ai_prompt", ""),
        ]))
        for pattern, gender in _GENDER_DETECT_RE:
            if pattern.search(text):
                gender_map[key] = gender
                break   # first match wins (female checked before male)

    return gender_map


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_azure_voice(
    speaker: str,
    voice_style: str,
    azure_lang: str,
    speaker_gender_map: dict[str, str] | None = None,
) -> str:
    """Return an Azure voice name for speaker + locale.

    Priority:
      1. speaker_gender_map[speaker]   gender from character_packs[].gender
                                       (set by Stage 5 from Script.json cast[].gender)
      2. voice_style keywords          fallback for manifests missing the gender field
      3. "male" hard default
    """
    lang_prefix = azure_lang.split("-")[0].lower()

    # 1. Gender from character_packs[].gender (the authoritative pipeline source)
    gender: str | None = None
    if speaker_gender_map:
        gender = speaker_gender_map.get(_norm_speaker(speaker))

    # 2. Gender from voice_style keywords (fallback for old manifests)
    if gender is None:
        for kw, g in _GENDER_RULES:
            if kw in voice_style.lower():
                gender = g
                break

    # 3. Hard default
    if gender is None:
        gender = "male"

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
    pitch: str | None = None,
    break_ms: int = 0,
) -> str:
    """Build Azure SSML for a single utterance.

    Layer order (outermost → innermost):
      <voice> → <mstts:express-as> → <prosody> → text [<break/>text ...]

    Rules:
    - duration_sec overrides rate when both are set (used for timed-shot fitting)
    - express-as element is omitted entirely when style is None
    - prosody element is always emitted when pitch is set, even if rate is "0%"
      and duration_sec is None (pitch is a prosody attribute)
    - break_ms > 0: inserts <break time='Nms'/> after each sentence terminator
      (period, exclamation mark, question mark) in the text.  Set to 0 for
      whispering style — breaks inside whispers sound robotic.
    """
    escaped = (text
               .replace("&", "&amp;")
               .replace("<", "&lt;")
               .replace(">", "&gt;")
               .replace('"', "&quot;")
               .replace("'", "&apos;"))

    # Optionally inject sentence-end breaks
    if break_ms > 0:
        escaped = re.sub(
            r'([.!?](?:\s|$))',
            lambda m: m.group(1).rstrip() + f'<break time="{break_ms}ms"/>' + (m.group(1)[len(m.group(1).rstrip()):] or ' '),
            escaped,
        )

    # Prosody — emit whenever rate, pitch, or duration_sec is meaningful
    prosody_parts: list[str] = []
    if duration_sec is not None:
        prosody_parts.append(f'duration="{duration_sec:.3f}s"')
    elif rate and rate != "0%":
        prosody_parts.append(f'rate="{rate}"')
    if pitch:
        prosody_parts.append(f'pitch="{pitch}"')

    spoken = (
        f"<prosody {' '.join(prosody_parts)}>{escaped}</prosody>"
        if prosody_parts else escaped
    )

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
# WAV writer helper
# ---------------------------------------------------------------------------

def _write_wav(path: str, pcm_bytes: bytes, sample_rate: int, channels: int, bits: int) -> None:
    """Write a standard RIFF PCM WAV file from raw PCM bytes.

    Constructs a minimal 44-byte RIFF/WAVE/fmt /data header and writes it
    together with the PCM payload to *path*.  Overwrites any existing file.
    """
    byte_rate   = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    data_size   = len(pcm_bytes)
    riff_size   = 36 + data_size   # total file size minus 8 bytes for "RIFF" + size field

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_size,
        b"WAVE",
        b"fmt ",
        16,             # PCM sub-chunk size
        1,              # AudioFormat: PCM
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits,
        b"data",
        data_size,
    )
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(pcm_bytes)


# ---------------------------------------------------------------------------
# Batch (episode-level) SSML helpers
# ---------------------------------------------------------------------------

def build_episode_ssml(items: list[dict], locale: str) -> str:
    """Build a single SSML document containing all vo_items for *locale*.

    A ``<bookmark mark="{item_id}"/>`` element is inserted immediately before
    each ``<voice>`` block so that Azure's bookmark_reached event returns the
    audio-stream offset of the start of that item's speech.

    The inner ``<voice>/<mstts:express-as>/<prosody>`` structure reuses the
    same logic as ``build_ssml()`` — only the outer ``<speak>`` wrapper is
    shared across all items.
    """
    azure_lang = items[0]["azure_lang"] if items else LOCALE_TO_AZURE_LANG.get(locale.lower(), "en-US")

    parts: list[str] = []
    for item in items:
        text        = item["text"]
        voice       = item["voice"]
        rate        = item["rate"]
        style       = item["style"]
        style_degree = item["style_degree"]
        pitch       = item.get("pitch")
        break_ms    = item.get("break_ms", 0)
        item_id     = item["item_id"]

        # XML-escape the text (same logic as build_ssml)
        escaped = (text
                   .replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   .replace('"', "&quot;")
                   .replace("'", "&apos;"))

        # Optionally inject sentence-end breaks
        if break_ms > 0:
            escaped = re.sub(
                r'([.!?](?:\s|$))',
                lambda m: m.group(1).rstrip() + f'<break time="{break_ms}ms"/>' + (m.group(1)[len(m.group(1).rstrip()):] or ' '),
                escaped,
            )

        # Prosody attributes
        prosody_parts: list[str] = []
        if rate and rate != "0%":
            prosody_parts.append(f'rate="{rate}"')
        if pitch:
            prosody_parts.append(f'pitch="{pitch}"')

        spoken = (
            f"<prosody {' '.join(prosody_parts)}>{escaped}</prosody>"
            if prosody_parts else escaped
        )

        # Style wrapper
        if style:
            spoken = (
                f'<mstts:express-as style="{style}" styledegree="{style_degree}">'
                f"{spoken}</mstts:express-as>"
            )

        # Bookmark as first child inside voice block (Azure rejects bookmarks
        # as direct children of <speak> / RootSpeak — error 1007).
        parts.append(f"<voice name='{voice}'><bookmark mark='{item_id}'/>{spoken}</voice>")

    inner = "\n  ".join(parts)
    return (
        f"<speak version='1.0' xml:lang='{azure_lang}' "
        f"xmlns='http://www.w3.org/2001/10/synthesis' "
        f"xmlns:mstts='http://www.w3.org/2001/mstts'>\n"
        f"  {inner}\n"
        f"</speak>"
    )


def synthesise_with_bookmarks(
    synthesizer,
    ssml: str,
    expected_ids: list[str],
) -> tuple[str, dict[str, int]]:
    """Submit combined episode SSML and return (tmp_wav_path, offsets).

    Subscribes to ``synthesizer.bookmark_reached`` *before* calling
    ``speak_ssml_async`` so that no events are missed.  Each bookmark event
    records the item_id → audio_offset (100-nanosecond ticks) mapping.

    Saves the combined WAV to a temporary file and validates that every
    expected bookmark was received.

    Returns:
        tmp_path : str   — path to the combined WAV tempfile (caller must unlink)
        offsets  : dict  — {item_id: audio_offset_ticks}

    Raises:
        RuntimeError  — if Azure synthesis fails
        ValueError    — if any expected bookmark IDs are missing from the result
    """
    import azure.cognitiveservices.speech as speechsdk

    offsets: dict[str, int] = {}

    def _on_bookmark(evt) -> None:
        offsets[evt.text] = evt.audio_offset

    synthesizer.bookmark_reached.connect(_on_bookmark)

    try:
        result = synthesizer.speak_ssml_async(ssml).get()

        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            details = result.cancellation_details
            raise RuntimeError(
                f"Azure TTS cancelled: reason={details.reason}  "
                f"detail={details.error_details}"
            )

        # Save combined WAV to a temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()

        stream = speechsdk.AudioDataStream(result)
        stream.save_to_wav_file(tmp_path)

    finally:
        # Always disconnect to avoid stale handlers if synthesizer is reused.
        # EventSignal has no disconnect(cb) — use disconnect_all() instead.
        synthesizer.bookmark_reached.disconnect_all()

    # Validate that every expected bookmark was received
    missing = [id_ for id_ in expected_ids if id_ not in offsets]
    if missing:
        raise ValueError(
            f"Batch synthesis: {len(missing)} bookmark(s) not received — "
            f"missing IDs: {missing}"
        )

    return tmp_path, offsets


def split_and_write_wavs(
    tmp_path: str,
    offsets: dict[str, int],
    items: list[dict],
    out_dir: Path,
) -> None:
    """Split the combined batch WAV into per-item WAV files.

    Locates the raw PCM payload using ``raw.index(b'data') + 8`` (robust
    against variable-length RIFF headers — never hardcodes 44).  Converts
    100-ns tick offsets to byte positions, slices the PCM, and writes a
    fresh WAV header per item via ``_write_wav()``.

    Args:
        tmp_path : str        — path to combined WAV produced by synthesise_with_bookmarks()
        offsets  : dict       — {item_id: audio_offset_ticks}
        items    : list[dict] — vo_items in any order (sorted internally by offset)
        out_dir  : Path       — directory to write {item_id}.wav files into
    """
    with open(tmp_path, "rb") as fh:
        raw = fh.read()

    # Locate PCM data — skip 4-byte 'data' tag + 4-byte chunk-size field.
    # DO NOT hardcode 44 — header size varies by SDK version.
    data_tag_pos = raw.index(b"data")
    pcm = raw[data_tag_pos + 8:]

    # Convert tick offsets to PCM byte positions.
    # Formula: byte_offset = round(ticks / 10_000_000 * 24000 * 2)
    # (24 kHz * 16-bit mono = 48000 bytes/s; 10_000_000 ticks/s)
    sorted_items = sorted(items, key=lambda it: offsets[it["item_id"]])
    byte_offsets = {
        it["item_id"]: round(offsets[it["item_id"]] / 10_000_000 * 24000 * 2)
        for it in sorted_items
    }

    for i, item in enumerate(sorted_items):
        item_id = item["item_id"]
        start   = byte_offsets[item_id]
        end     = (
            byte_offsets[sorted_items[i + 1]["item_id"]]
            if i + 1 < len(sorted_items)
            else len(pcm)
        )
        pcm_slice = pcm[start:end]

        out_path = out_dir / f"{item_id}.wav"
        _write_wav(str(out_path), pcm_slice, sample_rate=24000, channels=1, bits=16)

        assert os.path.getsize(str(out_path)) > 44, f"Output WAV too small: {out_path}"

        # Write license sidecar (same as per-item path)
        write_license_sidecar(out_path, item["voice"], item["style"])

        size      = out_path.stat().st_size
        pcm_len   = max(0, len(pcm_slice))
        duration_s = pcm_len / (AZURE_SAMPLE_RATE * 2)
        log.info(f"  [OK] {out_path}  ({duration_s:.2f}s, {size:,} bytes)")
        print(f"  [batch] {item_id}  ({duration_s:.2f}s, {size:,} bytes)")


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

    # Build a speaker → gender map from character_packs[].gender
    # (written by Stage 5 from Script.json cast[].gender).
    speaker_gender_map = build_gender_map_from_character_packs(
        manifest.get("character_packs", [])
    )
    if speaker_gender_map:
        print(f"  [TTS] Character gender map: {speaker_gender_map}")

    # Load VoiceCast.json for this project — used as fallback when
    # tts_prompt.azure_voice is missing (e.g. locale manifests produced by
    # Stage 8 that omitted it due to LLM truncation).
    project_id = manifest.get("project_id", "")
    voice_cast_map: dict[str, dict] = {}
    if project_id:
        vc_path = PROJECTS_ROOT / project_id / "VoiceCast.json"
        if vc_path.exists():
            try:
                vc_data = json.loads(vc_path.read_text(encoding="utf-8"))
                for ch in vc_data.get("characters", []):
                    cid = ch.get("character_id", "")
                    if cid:
                        voice_cast_map[cid] = ch
                print(f"  [TTS] Loaded VoiceCast.json ({len(voice_cast_map)} characters)")
            except Exception as exc:
                print(f"  [WARN] Could not load VoiceCast.json: {exc}")

    items = []
    for vo in manifest.get("vo_items", []):
        if asset_id_filter and vo["item_id"] != asset_id_filter:
            continue

        tts          = vo.get("tts_prompt", {})
        voice_style  = tts.get("voice_style", "")
        emotion      = tts.get("emotion", "")
        pace         = tts.get("pace", "normal")

        # ── Voice resolution (explicit > VoiceCast > speaker/gender default) ──
        speaker_id = vo.get("speaker_id") or "narrator"   # locale manifests may omit speaker_id
        vc_locale  = voice_cast_map.get(speaker_id, {}).get(locale, {})
        voice = (
            tts.get("azure_voice")
            or vc_locale.get("azure_voice")   # VoiceCast.json fallback (Stage 8 may omit azure_voice)
            or resolve_azure_voice(speaker_id, voice_style, azure_lang, speaker_gender_map)
        )

        # ── Style resolution (explicit > VoiceCast base > emotion mapping) ──
        style = (
            tts.get("azure_style")
            or vc_locale.get("azure_style")   # VoiceCast fallback (Stage 8 may omit azure_style)
            or resolve_azure_style(emotion)
        )

        # ── Style degree (explicit > VoiceCast base > default 1.5) ──
        style_degree = (
            tts.get("azure_style_degree")
            or vc_locale.get("azure_style_degree")
            or DEFAULT_STYLE_DEGREE
        )

        # ── Rate resolution (explicit > VoiceCast base > pace mapping) ──
        rate = (
            tts.get("azure_rate")
            or vc_locale.get("azure_rate")
            or AZURE_PACE_RATE.get(pace, "0%")
        )

        # ── Pitch (explicit > VoiceCast base; no auto-derivation) ──
        pitch = tts.get("azure_pitch") or vc_locale.get("azure_pitch") or None

        # ── Sentence-end break (explicit > VoiceCast base; default 0 = disabled) ──
        break_ms = int(tts.get("azure_break_ms") or vc_locale.get("azure_break_ms") or 0)

        items.append({
            "item_id":      vo["item_id"],
            "speaker":      speaker_id,
            "text":         vo["text"],
            "locale":       locale,
            "azure_lang":   azure_lang,
            "voice":        voice,
            "voice_style":  voice_style,
            "emotion":      emotion,
            "style":        style,
            "style_degree": style_degree,
            "rate":         rate,
            "pitch":        pitch,
            "break_ms":     break_ms,
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

def _synthesise_per_item(synthesizer, items: list[dict], out_dir: Path) -> list[dict]:
    """Per-item synthesis loop (original implementation, preserved as fallback).

    Called by run() when:
      - --asset-id is set (single-item path), OR
      - batch synthesis raises an exception (fallback path)
    """
    results = []
    total   = len(items)

    for idx, vo in enumerate(items, start=1):
        out_path = out_dir / f"{vo['item_id']}.wav"

        print(f"\n[{idx}/{total}] {vo['item_id']}")
        print(f"  Speaker      : {vo['speaker']}")
        print(f"  Voice        : {vo['voice']}  (lang={vo['azure_lang']})")
        print(f"  Voice style  : {vo['voice_style'] or '(not set)'}")
        print(f"  Emotion      : {vo['emotion'] or '(not set)'}  →  style={vo['style'] or 'none'}  degree={vo['style_degree']}")
        print(f"  Rate         : {vo['rate']}"
              + (f"  pitch={vo['pitch']}" if vo.get('pitch') else "")
              + (f"  break={vo['break_ms']}ms" if vo.get('break_ms') else ""))
        print(f"  Text         : \"{vo['text'][:80]}{'...' if len(vo['text']) > 80 else ''}\"")

        ssml = build_ssml(
            text=vo["text"],
            voice=vo["voice"],
            azure_lang=vo["azure_lang"],
            rate=vo["rate"],
            style=vo["style"],
            style_degree=vo["style_degree"],
            pitch=vo.get("pitch"),
            break_ms=vo.get("break_ms", 0),
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


def run(items: list[dict], out_dir: Path, asset_id: str | None = None) -> list[dict]:
    """Synthesise all items with Azure TTS. Returns a list of result dicts.

    When *asset_id* is None (full episode synthesis), attempts a single batched
    API call via ``build_episode_ssml`` / ``synthesise_with_bookmarks`` /
    ``split_and_write_wavs``.  On any failure, falls back transparently to the
    original per-item loop.

    When *asset_id* is set (single-item re-synthesis), uses the per-item path
    directly — batch synthesis for a single item would be wasteful and the
    caller already filtered ``items`` to the one matching item.
    """
    synthesizer = load_azure_synthesizer()

    # ------------------------------------------------------------------
    # Single-item path (--asset-id): per-item synthesis, unchanged.
    # ------------------------------------------------------------------
    if asset_id is not None:
        return _synthesise_per_item(synthesizer, items, out_dir)

    # ------------------------------------------------------------------
    # Full-episode batch path: one API call for the entire locale.
    # Falls back to per-item on any exception.
    # ------------------------------------------------------------------
    locale = items[0]["locale"] if items else "en"

    try:
        log.info("Batch synthesis: building episode SSML...")
        print(f"\n[BATCH] Building episode SSML for {len(items)} items...")
        ssml         = build_episode_ssml(items, locale)
        expected_ids = [it["item_id"] for it in items]

        tmp_path, offsets = synthesise_with_bookmarks(synthesizer, ssml, expected_ids)
        try:
            split_and_write_wavs(tmp_path, offsets, items, out_dir)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        log.info(f"Batch synthesis complete: {len(items)} items written")
        print(f"[BATCH] Complete: {len(items)} items written to {out_dir}")

        # Build results list from the written files (mirrors per-item result schema)
        results = []
        for vo in items:
            out_path = out_dir / f"{vo['item_id']}.wav"
            if out_path.exists():
                size      = out_path.stat().st_size
                # Use raw.index approach for accurate duration — same as split_and_write_wavs
                with open(str(out_path), "rb") as fh:
                    raw = fh.read()
                data_tag_pos = raw.index(b"data")
                pcm_len   = max(0, len(raw) - (data_tag_pos + 8))
                duration_s = pcm_len / (AZURE_SAMPLE_RATE * 2)
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
            else:
                results.append({
                    "item_id":    vo["item_id"],
                    "speaker":    vo["speaker"],
                    "voice":      vo["voice"],
                    "output":     str(out_path),
                    "size_bytes": 0,
                    "status":     "failed",
                    "error":      "WAV file not found after batch split",
                })
        return results

    except Exception as e:
        log.warning(f"Batch synthesis failed ({e}); falling back to per-item synthesis")
        print(f"\n[BATCH] Failed ({e}); falling back to per-item synthesis...")
        return _synthesise_per_item(synthesizer, items, out_dir)


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

    # ── Stale WAV cleanup ─────────────────────────────────────────────────
    # On re-runs the LLM may produce different item_id names (different scene
    # slugs, different numbering).  WAVs from previous runs are never deleted
    # automatically, so the folder accumulates orphaned files.  Delete any
    # .wav in out_dir that is NOT referenced by the current manifest items.
    # NOTE: always use ALL vo_items from the manifest (not the asset_id-filtered
    # subset) so that single-item re-synthesis calls don't delete WAVs for
    # other items that are not being re-synthesized right now.
    all_manifest_ids: set[str] = {
        it["item_id"] for it in manifest.get("vo_items", [])}
    for wav in out_dir.glob("*.wav"):
        if wav.stem not in all_manifest_ids:
            wav.unlink()
            print(f"  [STALE] Deleted orphaned WAV: {wav.name}")
    # ── end stale cleanup ─────────────────────────────────────────────────
    if not items:
        print("[WARN] No matching vo_items found in manifest. Nothing to do.")
        return

    print(f"\n{'='*60}")
    print(f"BACKEND  : AZURE NEURAL TTS (cloud, no GPU)")
    print(f"Locale   : {locale}  →  {LOCALE_TO_AZURE_LANG.get(locale.lower(), 'en-US')}")
    print(f"Items    : {len(items)}")
    print(f"Output   : {out_dir}")
    print(f"{'='*60}")

    results = run(items, out_dir, asset_id=args.asset_id)

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
