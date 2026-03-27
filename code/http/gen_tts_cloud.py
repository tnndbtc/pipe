#!/usr/bin/env python3
# =============================================================================
# gen_tts_cloud.py
# Azure Neural TTS backend for the AI Asset Generation Pipeline.
# CPU-only; no GPU or local model required.
#
# Reads vo_items from a VOPlan locale manifest and synthesises WAV
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
#   python gen_tts_cloud.py --manifest VOPlan.en.json
#   python gen_tts_cloud.py --manifest VOPlan.zh-Hans.json
#   python gen_tts_cloud.py --manifest VOPlan.en.json --asset-id vo-s01e02-sc01-001
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
import hashlib
import io
import json
import logging
import os
import re
import struct
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)

# SSML namespace constants — keep in sync with ssml_preprocess.py
_NS_SYNTHESIS = "http://www.w3.org/2001/10/synthesis"
_NS_MSTTS     = "http://www.w3.org/2001/mstts"
ET.register_namespace("",      _NS_SYNTHESIS)   # default ns
ET.register_namespace("mstts", _NS_MSTTS)
# Wrapper used when parsing ssml_inner fragments that use mstts: prefix
_SSML_VOICE_OPEN = (
    f'<voice xmlns="{_NS_SYNTHESIS}" xmlns:mstts="{_NS_MSTTS}">'
)

# =============================================================================
# Paths
# =============================================================================
PIPE_DIR      = Path(__file__).resolve().parent.parent.parent
PROJECTS_ROOT = PIPE_DIR / "projects"
SCRIPT_NAME   = "gen_tts_cloud"

# =============================================================================
# Batch synthesis config — loaded from prompts/azure_tts_batch.json
# Only voices listed in batch_voices support bookmark events required for batch.
# DragonHD / DragonHDFlash / DragonHDOmni do NOT support bookmarks.
# =============================================================================
_BATCH_CONFIG_PATH = PIPE_DIR / "prompts" / "azure_tts_batch.json"
_BATCH_CONFIG: dict = {}
if _BATCH_CONFIG_PATH.exists():
    try:
        _BATCH_CONFIG = json.loads(_BATCH_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as _exc:
        print(f"[WARN] Could not load {_BATCH_CONFIG_PATH}: {_exc}")

BATCH_VOICES: set[str] = set(_BATCH_CONFIG.get("batch_voices", []))
BATCH_MAX_VOICE_ELEMENTS: int = _BATCH_CONFIG.get("max_voice_elements", 50)

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


# ---------------------------------------------------------------------------
# Chinese phoneme corrections
# ---------------------------------------------------------------------------
# Azure TTS Chinese voices sometimes mispronounce certain characters.
# Add new entries here as they are discovered; the pinyin uses tone numbers
# (1–4, e.g. "hou4" = hòu).  Only applied when azure_lang starts with "zh".

_ZH_PHONEME_CORRECTIONS: dict[str, str] = {
    "后": "hou4",   # hòu (after/behind, tone 4) — mispronounced as hóu (monkey, tone 2)
}


def _xml_escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;").replace("'", "&apos;"))


def segment_zh_phonemes(text: str, azure_lang: str) -> list[dict]:
    """Split *text* into plain-text and phoneme-correction segments.

    Returns a list of dicts, each either:
        {'type': 'text',    'content': str}
        {'type': 'phoneme', 'content': str, 'pinyin': str}

    <phoneme> elements must be emitted as direct children of <voice>, OUTSIDE
    any <mstts:express-as> block — Azure rejects <phoneme> inside express-as
    with error 1007.

    For non-zh locales (or when there are no corrections) returns a single
    text segment so callers can use the same code path everywhere.
    """
    if not azure_lang.startswith("zh") or not _ZH_PHONEME_CORRECTIONS:
        return [{"type": "text", "content": text}]
    segments: list[dict] = []
    buf: list[str] = []
    for ch in text:
        pinyin = _ZH_PHONEME_CORRECTIONS.get(ch)
        if pinyin:
            if buf:
                segments.append({"type": "text", "content": "".join(buf)})
                buf = []
            segments.append({"type": "phoneme", "content": ch, "pinyin": pinyin})
        else:
            buf.append(ch)
    if buf:
        segments.append({"type": "text", "content": "".join(buf)})
    return segments or [{"type": "text", "content": text}]


def _wrap_spoken(escaped: str, prosody_parts: list[str],
                 style: str | None, style_degree: float) -> str:
    """Wrap an already-XML-escaped text fragment in prosody + express-as."""
    spoken = (
        f"<prosody {' '.join(prosody_parts)}>{escaped}</prosody>"
        if prosody_parts else escaped
    )
    if style:
        spoken = (
            f'<mstts:express-as style="{style}" styledegree="{style_degree}">'
            f"{spoken}</mstts:express-as>"
        )
    return spoken


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
    # Normalise newlines → single space so Azure TTS doesn't insert pauses
    text = re.sub(r'\s*\n+\s*', ' ', text).strip()

    # Prosody attrs
    prosody_parts: list[str] = []
    if duration_sec is not None:
        prosody_parts.append(f'duration="{duration_sec:.3f}s"')
    elif rate and rate != "0%":
        prosody_parts.append(f'rate="{rate}"')
    if pitch:
        prosody_parts.append(f'pitch="{pitch}"')

    # Split at phoneme-correction boundaries.  Each 'text' segment is wrapped
    # in the normal express-as/prosody stack; each 'phoneme' segment becomes a
    # bare <phoneme> direct child of <voice> (Azure rejects <phoneme> inside
    # <mstts:express-as>).
    voice_parts: list[str] = []
    for seg in segment_zh_phonemes(text, azure_lang):
        if seg["type"] == "text":
            escaped = _xml_escape(seg["content"])
            if break_ms > 0:
                escaped = re.sub(
                    r'([.!?](?:\s|$))',
                    lambda m: m.group(1).rstrip() + f'<break time="{break_ms}ms"/>'
                              + (m.group(1)[len(m.group(1).rstrip()):] or ' '),
                    escaped,
                )
            voice_parts.append(_wrap_spoken(escaped, prosody_parts, style, style_degree))
        else:  # phoneme
            voice_parts.append(
                f'<phoneme alphabet="pinyin" ph="{seg["pinyin"]}">'
                f'{_xml_escape(seg["content"])}</phoneme>'
            )

    return (
        f"<speak version='1.0' xml:lang='{azure_lang}' "
        f"xmlns='http://www.w3.org/2001/10/synthesis' "
        f"xmlns:mstts='http://www.w3.org/2001/mstts'>"
        f"<voice name='{voice}'>{''.join(voice_parts)}</voice>"
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
        text        = re.sub(r'\s*\n+\s*', ' ', item["text"]).strip()
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
# Chunk-alignment synthesis infrastructure (Phase 1 — az.txt plan)
# ---------------------------------------------------------------------------

_HD_OMNI_PATTERN = re.compile(r"DragonHDOmni",  re.IGNORECASE)
_HD_VOICE_PATTERN = re.compile(r"DragonHD",     re.IGNORECASE)

# Average spoken characters per second — conservative estimate for chunk sizing.
# Intentionally safe: underestimates → smaller chunks → easier debugging.
_CHARS_PER_SEC_BY_LOCALE: dict[str, float] = {
    "zh-hans": 4.5,
    "zh-cn":   4.5,
    "zh":      4.5,
    "ja":      5.0,
    "ko":      5.0,
    "en":      14.0,
    "en-us":   14.0,
}
_CHARS_PER_SEC_DEFAULT = 12.0

# Prosody rate string → multiplier (higher = slower speech = longer duration)
_RATE_TO_MULT: dict[str, float] = {
    "-25%": 1.33, "-20%": 1.25, "-15%": 1.18, "-10%": 1.11, "-5%": 1.05,
    "0%": 1.0,
    "+5%": 0.95, "+10%": 0.91, "+15%": 0.87, "+20%": 0.83, "+25%": 0.80,
    "slow": 1.33, "medium": 1.0, "fast": 0.75,
}

# TTS cache versioning — bump when synthesis config or normalization logic changes
_TTS_ENGINE_VERSION = "1"
_TTS_OUTPUT_FORMAT  = "Riff24Khz16BitMonoPcm"
_TTS_SAMPLE_RATE    = str(AZURE_SAMPLE_RATE)


def _rate_to_multiplier(rate_str: str) -> float:
    """Convert a prosody rate string to a duration multiplier (> 1 = slower)."""
    rate_str = (rate_str or "0%").strip()
    if rate_str in _RATE_TO_MULT:
        return _RATE_TO_MULT[rate_str]
    if rate_str.endswith("%"):
        try:
            pct = float(rate_str[:-1])
            return 1.0 / (1.0 + pct / 100.0) if pct != 0 else 1.0
        except ValueError:
            pass
    # Raw float (ssml_rate_raw e.g. "0.86")
    try:
        v = float(rate_str)
        return 1.0 / v if v > 0 else 1.0
    except ValueError:
        pass
    return 1.0


def _is_hd_voice(voice: str) -> bool:
    """True if voice is any DragonHD variant (HD, HDFlash, HDOmni)."""
    return bool(_HD_VOICE_PATTERN.search(voice))


def _is_hd_omni_voice(voice: str) -> bool:
    """True if voice is DragonHDOmni (the only HD variant with word-boundary events)."""
    return bool(_HD_OMNI_PATTERN.search(voice))


def _estimate_duration_sec(text: str, locale: str, rate: str) -> float:
    """Rough estimate of spoken duration for chunk-sizing heuristics.

    Uses locale-specific character-per-second rate adjusted by prosody rate.
    Conservative (may underestimate); intentionally safe for chunk planning.
    """
    locale_key = locale.lower()
    cps  = _CHARS_PER_SEC_BY_LOCALE.get(locale_key, _CHARS_PER_SEC_DEFAULT)
    mult = _rate_to_multiplier(rate)
    chars = len(text.strip())
    if chars == 0:
        return 0.0
    return (chars / cps) * mult


def _xml_escape(text: str) -> str:
    """Escape XML special characters for SSML text content."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;"))


def build_ssml_minimal(
    text: str,
    voice: str,
    azure_lang: str,
    *,
    rate: str,
    pitch: str | None,
    style: str | None,
    style_degree: float,
    break_ms: int = 0,
    inner_only: bool = False,
) -> str:
    """Build SSML for a single utterance — minimal overhead, no default breaks.

    Phase 1 change vs build_ssml():
    - break_ms injection is disabled by default (break_ms=0 means no injection).
      Callers must explicitly pass break_ms > 0 to enable it.
    - inner_only=True returns just the <voice> block without the <speak> wrapper,
      suitable for embedding in a multi-sentence chunk SSML.
    """
    # Normalise newlines → single space so Azure TTS doesn't insert pauses
    text = re.sub(r'\s*\n+\s*', ' ', text).strip()
    escaped = _xml_escape(text)

    # Explicit break injection — caller-controlled only, not default behaviour
    if break_ms > 0:
        escaped = re.sub(
            r'([.!?](?:\s|$))',
            lambda m: (m.group(1).rstrip()
                       + f'<break time="{break_ms}ms"/>'
                       + (m.group(1)[len(m.group(1).rstrip()):] or ' ')),
            escaped,
        )

    prosody_parts: list[str] = []
    if rate and rate != "0%":
        prosody_parts.append(f'rate="{rate}"')
    if pitch:
        prosody_parts.append(f'pitch="{pitch}"')

    spoken = (
        f"<prosody {' '.join(prosody_parts)}>{escaped}</prosody>"
        if prosody_parts else escaped
    )

    if style:
        spoken = (
            f'<mstts:express-as style="{style}" styledegree="{style_degree}">'
            f"{spoken}</mstts:express-as>"
        )

    voice_block = f"<voice name='{voice}'>{spoken}</voice>"

    if inner_only:
        return voice_block

    return (
        f"<speak version='1.0' xml:lang='{azure_lang}' "
        f"xmlns='http://www.w3.org/2001/10/synthesis' "
        f"xmlns:mstts='http://www.w3.org/2001/mstts'>"
        f"{voice_block}"
        f"</speak>"
    )


def group_sentences_into_chunks(
    sentence_frags: list[dict],
    items: list[dict],
    voice: str,
    style: str | None,
    style_degree: float,
    rate: str,
    pitch: str | None,
    azure_lang: str,
    locale: str,
    target_dur_sec: float = 35.0,
    max_dur_sec: float    = 45.0,
    max_chars: int        = 1000,
) -> list[dict]:
    """Group sentence fragments into synthesis chunks for chunk_alignment mode.

    Each chunk dict contains:
      {
        "sentences":          [{"text", "pause_ms", "item_id", "vo"}, ...],
        "total_chars":        int,
        "estimated_dur_sec":  float,
        "voice":              str,
        "style":              str|None,
        "style_degree":       float,
        "rate":               str,
        "pitch":              str|None,
        "azure_lang":         str,
      }

    Chunking strategy (duration-first, az.txt §Step 3):
    1. Estimate duration per sentence via _estimate_duration_sec().
    2. Hard flush at scene boundaries: if the previous sentence's pause_ms >= 2000ms,
       flush the current chunk before starting a new one.
    3. Hard flush when voice or style changes (handled by caller grouping).
    4. Soft flush when accumulated est_dur_sec reaches target_dur_sec.
    5. Hard flush when adding the next sentence would exceed max_dur_sec or max_chars.
    6. Minimum 1 sentence per chunk guaranteed.
    """
    total = min(len(items), len(sentence_frags))

    def _make_chunk(sentences: list[dict], dur: float, chars: int) -> dict:
        return {
            "sentences":         sentences,
            "total_chars":       chars,
            "estimated_dur_sec": dur,
            "voice":             voice,
            "style":             style,
            "style_degree":      style_degree,
            "rate":              rate,
            "pitch":             pitch,
            "azure_lang":        azure_lang,
            "locale":            locale,
        }

    chunks: list[dict] = []
    cur_sentences: list[dict] = []
    cur_dur   = 0.0
    cur_chars = 0

    for idx in range(total):
        frag  = sentence_frags[idx]
        vo    = items[idx]
        text  = frag["text"]
        est   = _estimate_duration_sec(text, locale, rate)
        chars = len(text)

        # Hard flush: scene boundary (pause_ms >= 2000ms on the last accumulated sentence)
        # per az.txt §Step 3 — "Hard break at scene boundaries (pauses ≥ 2000ms)"
        if cur_sentences and cur_sentences[-1].get("pause_ms", 0) >= 2000:
            chunks.append(_make_chunk(list(cur_sentences), cur_dur, cur_chars))
            cur_sentences = []
            cur_dur   = 0.0
            cur_chars = 0

        # Soft flush: sentence would exceed duration or char limits
        elif cur_sentences and (cur_dur + est > max_dur_sec or cur_chars + chars > max_chars):
            chunks.append(_make_chunk(list(cur_sentences), cur_dur, cur_chars))
            cur_sentences = []
            cur_dur   = 0.0
            cur_chars = 0

        cur_sentences.append({
            "text":     text,
            "pause_ms": frag.get("pause_ms", 0),
            "item_id":  vo["item_id"],
            "vo":       vo,
        })
        cur_dur   += est
        cur_chars += chars

        # Flush at soft target
        if cur_dur >= target_dur_sec:
            chunks.append(_make_chunk(list(cur_sentences), cur_dur, cur_chars))
            cur_sentences = []
            cur_dur   = 0.0
            cur_chars = 0

    if cur_sentences:
        chunks.append(_make_chunk(cur_sentences, cur_dur, cur_chars))

    return chunks


def _build_chunk_ssml(chunk: dict) -> str:
    """Build a single SSML document covering all sentences in a chunk.

    All sentences share one <speak> wrapper + one <voice> block + one
    <prosody>/<express-as> wrapper.  The ~175-char <speak> overhead is paid
    once per chunk instead of once per sentence.

    Inter-sentence pauses are injected as <break time="Nms"/> after each
    sentence (except the last) using the sentence's pause_ms value.
    """
    voice        = chunk["voice"]
    style        = chunk["style"]
    style_degree = chunk["style_degree"]
    rate         = chunk["rate"]
    pitch        = chunk["pitch"]
    azure_lang   = chunk["azure_lang"]
    sentences    = chunk["sentences"]

    parts: list[str] = []
    for i, sent in enumerate(sentences):
        escaped  = _xml_escape(re.sub(r'\s*\n+\s*', ' ', sent["text"]).strip())
        is_last  = (i == len(sentences) - 1)
        pause_ms = sent.get("pause_ms", 0) if not is_last else 0
        if pause_ms > 0:
            parts.append(f'{escaped}<break time="{pause_ms}ms"/>')
        else:
            parts.append(escaped)

    inner_text = " ".join(parts)

    prosody_parts: list[str] = []
    if rate and rate != "0%":
        prosody_parts.append(f'rate="{rate}"')
    if pitch:
        prosody_parts.append(f'pitch="{pitch}"')

    spoken = (
        f"<prosody {' '.join(prosody_parts)}>{inner_text}</prosody>"
        if prosody_parts else inner_text
    )

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


def _get_pcm_from_wav_bytes(wav_bytes: bytes) -> bytes:
    """Extract raw PCM payload from WAV bytes (data-chunk search, never hardcodes 44)."""
    data_tag_pos = wav_bytes.index(b"data")
    return wav_bytes[data_tag_pos + 8:]


def _pcm_duration_sec(pcm_bytes: bytes, sample_rate: int = AZURE_SAMPLE_RATE) -> float:
    """Duration of a 16-bit mono PCM byte string in seconds."""
    return len(pcm_bytes) / (sample_rate * 2)


def _align_proportional(chunk_wav_bytes: bytes, chunk: dict) -> list[dict]:
    """Align sentences within a chunk by proportional character ratio.

    Divides chunk duration proportionally to each sentence's character count.
    Simple, zero dependencies, always succeeds.

    Returns list of {"item_id": str, "start_sec": float, "end_sec": float}.
    """
    pcm        = _get_pcm_from_wav_bytes(chunk_wav_bytes)
    total_dur  = _pcm_duration_sec(pcm)
    sentences  = chunk["sentences"]
    total_chars = sum(len(s["text"]) for s in sentences)

    if total_chars == 0:
        per = total_dur / max(len(sentences), 1)
        return [
            {"item_id": s["item_id"],
             "start_sec": round(i * per, 4),
             "end_sec":   round((i + 1) * per, 4)}
            for i, s in enumerate(sentences)
        ]

    offsets: list[dict] = []
    cursor = 0.0
    for s in sentences:
        frac = len(s["text"]) / total_chars
        dur  = total_dur * frac
        offsets.append({
            "item_id":   s["item_id"],
            "start_sec": round(cursor, 4),
            "end_sec":   round(cursor + dur, 4),
        })
        cursor += dur

    return offsets


def _align_by_silence(
    chunk_wav_bytes: bytes,
    chunk: dict,
    silence_threshold: int = 500,
    min_silence_frames: int | None = None,
) -> list[dict] | None:
    """Align sentences by detecting silence gaps in 16-bit mono PCM.

    Uses the inter-sentence <break> pauses injected by _build_chunk_ssml as
    alignment anchors.  Only silence runs >= 80% of the chunk's minimum
    break_ms are considered, which filters out intra-sentence drama pauses
    (em-dash ~150-350ms, comma ~50-150ms) while reliably catching the
    explicit sentence-boundary breaks (typically >=200ms).

    Returns list of {item_id, start_sec, end_sec}, or None if the detected
    gap count doesn't match the expected (n_sentences - 1) boundaries.

    Args:
        silence_threshold  : PCM amplitude below which a frame is "silent".
                             500 works for Azure Neural/DragonHD whose <break>
                             pauses are digital silence; intra-sentence pauses
                             may have residual room tone at ~100–300 amplitude.
        min_silence_frames : minimum consecutive silent frames for a gap to count.
                             When None (default), auto-derived from the chunk's
                             pause_ms values: 80% of the shortest non-zero break,
                             minimum 100ms.  Overrides only needed in tests.
    """
    try:
        pcm       = _get_pcm_from_wav_bytes(chunk_wav_bytes)
        sentences = chunk["sentences"]
        n_sents   = len(sentences)

        # Derive min silence from the inter-sentence break settings in this chunk.
        # Use 80% of the minimum break so we detect only the explicit breaks, not
        # intra-sentence drama pauses (em-dashes / commas).
        if min_silence_frames is None:
            min_pauses = [s.get("pause_ms", 0) for s in sentences[:-1]
                          if s.get("pause_ms", 0) > 0]
            if min_pauses:
                min_silence_ms = max(100, int(min(min_pauses) * 0.80))
            else:
                min_silence_ms = 100  # default: 100ms when no breaks defined
            min_silence_frames = int(min_silence_ms / 1000 * AZURE_SAMPLE_RATE)

        if n_sents == 1:
            dur = _pcm_duration_sec(pcm)
            return [{"item_id": sentences[0]["item_id"],
                     "start_sec": 0.0, "end_sec": round(dur, 4)}]

        n_samples = len(pcm) // 2
        samples = struct.unpack(f"<{n_samples}h", pcm[:n_samples * 2])

        # Detect silence runs → midpoint timestamps
        silence_midpoints: list[float] = []
        in_silence   = False
        silence_start = 0
        for i, s in enumerate(samples):
            is_silent = abs(s) < silence_threshold
            if is_silent and not in_silence:
                in_silence    = True
                silence_start = i
            elif not is_silent and in_silence:
                in_silence = False
                run_len = i - silence_start
                if run_len >= min_silence_frames:
                    mid = (silence_start + i) / 2
                    silence_midpoints.append(mid / AZURE_SAMPLE_RATE)
        # Handle trailing silence
        if in_silence:
            run_len = n_samples - silence_start
            if run_len >= min_silence_frames:
                mid = (silence_start + n_samples) / 2
                silence_midpoints.append(mid / AZURE_SAMPLE_RATE)

        if len(silence_midpoints) < n_sents - 1:
            return None   # not enough gaps found at this threshold

        # Take the first n_sents-1 gaps as boundaries (they're already in time order)
        boundaries = sorted(silence_midpoints[: n_sents - 1])
        total_dur  = _pcm_duration_sec(pcm)

        offsets: list[dict] = []
        prev = 0.0
        for i, sent in enumerate(sentences):
            end = boundaries[i] if i < len(boundaries) else total_dur
            offsets.append({
                "item_id":   sent["item_id"],
                "start_sec": round(prev, 4),
                "end_sec":   round(end, 4),
            })
            prev = end
        # Ensure last sentence reaches chunk end
        offsets[-1]["end_sec"] = round(total_dur, 4)

        # Proportionality guard: reject alignment if any sentence's duration is
        # wildly shorter than its char-count share.  This catches intra-sentence
        # HD voice pauses being mis-identified as inter-sentence boundaries.
        #
        # Rules:
        #  - Skip the LAST sentence: always gets trailing chunk silence (inflated).
        #  - Skip sentences with < 20 chars: short dramatic lines are inherently
        #    variable (e.g. "Hush." gets a long break + Azure pad).
        #  - For others: reject if actual < 0.20 × proportional share.
        total_chars = sum(len(s.get("text", "")) for s in sentences)
        if total_chars > 0:
            for idx, (off, sent) in enumerate(zip(offsets, sentences)):
                if idx == len(offsets) - 1:
                    continue   # last sentence: skip (trailing silence is expected)
                sent_chars = len(sent.get("text", ""))
                if sent_chars < 20:
                    continue   # short lines: naturally variable ratio
                prop_dur   = (sent_chars / total_chars) * total_dur
                if prop_dur <= 0:
                    continue
                actual_dur = off["end_sec"] - off["start_sec"]
                ratio = actual_dur / prop_dur
                if ratio < 0.20:
                    log.debug(
                        f"[chunk_align] silence-gap proportionality fail: "
                        f"{sent['item_id']} actual={actual_dur:.2f}s "
                        f"prop={prop_dur:.2f}s ratio={ratio:.2f} < 0.20"
                    )
                    return None   # fall through to proportional

        return offsets

    except Exception as exc:
        log.debug(f"[chunk_align] silence detection failed: {exc}")
        return None


def _validate_alignment(
    offsets: list[dict],
    chunk_dur_sec: float,
    epsilon: float = 0.01,
    min_sent_sec: float = 0.08,
) -> bool:
    """Validate alignment output contract (az.txt §Step 4C).

    Rules checked:
    - sentence_i.start_sec <= sentence_i.end_sec
    - sentence_i.end_sec <= sentence_{i+1}.start_sec + epsilon  (no overlap)
    - all offsets within [0, chunk_dur_sec + epsilon]
    - no sentence shorter than min_sent_sec (80 ms) — logged as warning only

    Returns True only if all structural checks pass.
    """
    for i, off in enumerate(offsets):
        s = off["start_sec"]
        e = off["end_sec"]
        if s < -epsilon or e < -epsilon:
            log.warning(f"[validate_alignment] negative offset idx={i}: start={s} end={e}")
            return False
        if e > chunk_dur_sec + epsilon:
            log.warning(f"[validate_alignment] end exceeds chunk_dur at idx={i}: "
                        f"end={e} chunk_dur={chunk_dur_sec}")
            return False
        if s > e + epsilon:
            log.warning(f"[validate_alignment] start > end at idx={i}: start={s} end={e}")
            return False
        if (e - s) < min_sent_sec:
            log.debug(f"[validate_alignment] short sentence at idx={i}: {e-s:.3f}s")
        if i + 1 < len(offsets):
            next_start = offsets[i + 1]["start_sec"]
            if e > next_start + epsilon:
                log.warning(f"[validate_alignment] overlap at idx={i}: "
                            f"end={e} next_start={next_start}")
                return False
    return True


def _clamp_alignment(offsets: list[dict], chunk_dur_sec: float) -> list[dict]:
    """Clamp all alignment offset values to [0, chunk_dur_sec]."""
    return [
        {
            "item_id":   off["item_id"],
            "start_sec": max(0.0, min(off["start_sec"], chunk_dur_sec)),
            "end_sec":   max(0.0, min(off["end_sec"],   chunk_dur_sec)),
        }
        for off in offsets
    ]


def _write_sentence_wavs_from_chunk(
    chunk_wav_bytes: bytes,
    offsets: list[dict],
    out_dir: Path,
    voice: str,
    style: str | None,
) -> list[dict]:
    """Slice a chunk WAV into per-sentence WAV files.

    Args:
        chunk_wav_bytes : raw bytes from Azure TTS for the entire chunk
        offsets         : validated list of {item_id, start_sec, end_sec}
        out_dir         : directory to write {item_id}.wav files
        voice           : for license sidecar
        style           : for license sidecar

    Returns list of result dicts: {item_id, output, size_bytes, duration_sec, status}.
    """
    pcm            = _get_pcm_from_wav_bytes(chunk_wav_bytes)
    bytes_per_sec  = AZURE_SAMPLE_RATE * 2   # 16-bit mono = 2 bytes/sample

    results: list[dict] = []
    for off in offsets:
        item_id    = off["item_id"]
        start_byte = round(off["start_sec"] * bytes_per_sec)
        end_byte   = round(off["end_sec"]   * bytes_per_sec)

        # Align to 2-byte (16-bit sample) boundary
        start_byte = (start_byte // 2) * 2
        end_byte   = (end_byte   // 2) * 2

        # Clamp to PCM buffer
        start_byte = max(0, min(start_byte, len(pcm)))
        end_byte   = max(start_byte, min(end_byte, len(pcm)))

        pcm_slice = pcm[start_byte:end_byte]
        out_path  = out_dir / f"{item_id}.wav"

        _write_wav(str(out_path), pcm_slice, sample_rate=AZURE_SAMPLE_RATE, channels=1, bits=16)
        write_license_sidecar(out_path, voice, style)

        size       = out_path.stat().st_size
        duration_s = len(pcm_slice) / bytes_per_sec
        log.info(f"  [chunk] {out_path}  ({duration_s:.2f}s, {size:,} bytes)")
        print(f"  [chunk] {item_id}  ({duration_s:.2f}s, {size:,} bytes)")

        results.append({
            "item_id":               item_id,
            "output":                str(out_path),
            "size_bytes":            size,
            "duration_sec":          round(duration_s, 3),
            "chunk_offset_start_sec": round(off["start_sec"], 4),
            "chunk_offset_end_sec":   round(off["end_sec"],   4),
            "status":                "success",
        })

    return results


# ---------------------------------------------------------------------------
# TTS cache (SHA256-keyed WAV cache — az.txt §Step 5)
# ---------------------------------------------------------------------------

def _normalize_ssml_for_cache(ssml: str) -> str:
    """Normalize insignificant XML whitespace for cache key computation.

    Per az.txt §Step 5:
    - Normalize insignificant whitespace and attribute ordering only.
    - Do NOT lowercase spoken text — would change pronunciation/semantics.
    """
    # Collapse whitespace runs between tags to a single space
    normalized = re.sub(r'>\s+<', '> <', ssml)
    return normalized.strip()


def _minify_ssml(ssml: str) -> str:
    """Strip XML comments and collapse inter-tag whitespace before Azure API calls.

    Per az.txt §Step 11 (Phase 4):
    Produces the canonical SSML form used for both the Azure API call and the
    cache key.  Reduces billed characters by ~28 % on user-authored narration
    SSML with zero effect on synthesised audio output.

    Rules (safe by XML spec + Azure TTS behaviour):
      1. Remove <!-- ... --> comments entirely  (DOTALL: handles multi-line).
      2. Collapse runs of whitespace between tags (>\\s+<) to '><'.
      3. Strip leading/trailing whitespace from the full document.

    NOT done (would risk audio changes):
      - Whitespace inside text nodes (affects pronunciation / pausing)
      - Attribute value changes
      - Tag reordering or merging
    """
    # Rule 1: strip XML comments
    s = re.sub(r'<!--.*?-->', '', ssml, flags=re.DOTALL)
    # Rule 2: collapse inter-tag whitespace
    s = re.sub(r'>\s+<', '><', s)
    # Rule 3: trim
    return s.strip()


def _tts_cache_key(ssml: str, voice: str, locale: str) -> str:
    """Compute SHA256 cache key for a TTS request.

    Key inputs (per az.txt §Step 5 + §Step 11):
      normalized_ssml + voice + locale + output_format + sample_rate + engine_version

    Cache key is derived from the minified (canonical) form so that two SSML
    inputs differing only in comments or whitespace — which produce identical
    audio after _minify_ssml() — share a single cache entry.
    """
    canonical  = _minify_ssml(ssml)
    normalized = _normalize_ssml_for_cache(canonical)
    payload = "\n".join([
        normalized,
        voice,
        locale,
        _TTS_OUTPUT_FORMAT,
        _TTS_SAMPLE_RATE,
        _TTS_ENGINE_VERSION,
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tts_cache_get(cache_dir: Path, key: str) -> bytes | None:
    """Return cached WAV bytes for the given key, or None on cache miss."""
    wav_path = cache_dir / f"{key}.wav"
    if wav_path.exists() and wav_path.stat().st_size > 44:
        try:
            return wav_path.read_bytes()
        except OSError:
            return None
    return None


def _tts_cache_put(cache_dir: Path, key: str, wav_bytes: bytes, meta: dict) -> None:
    """Write WAV bytes and metadata JSON to the cache directory."""
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{key}.wav").write_bytes(wav_bytes)
        (cache_dir / f"{key}.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as exc:
        log.warning(f"[tts_cache] write failed: {exc}")


def _per_item_legacy_chunk(
    synthesizer,
    chunk: dict,
    voice: str,
    style: str | None,
    style_degree: float,
    rate: str,
    pitch: str | None,
    azure_lang: str,
    out_dir: Path,
) -> list[dict]:
    """Per-sentence fallback synthesis for a single chunk.

    Used when chunk synthesis or alignment fails.  Each sentence in the chunk
    is synthesised individually (original pre-Phase-1 behaviour).
    """
    results: list[dict] = []
    for sent in chunk["sentences"]:
        item_id  = sent["item_id"]
        text     = sent["text"]
        vo       = sent["vo"]
        out_path = out_dir / f"{item_id}.wav"

        ssml = build_ssml_minimal(
            text=text,
            voice=voice,
            azure_lang=azure_lang,
            rate=rate,
            pitch=pitch,
            style=style,
            style_degree=style_degree,
            break_ms=0,
        )

        try:
            wav_bytes = synthesise(synthesizer, ssml)
            if out_path.exists():
                out_path.unlink()
            out_path.write_bytes(wav_bytes)
            write_license_sidecar(out_path, voice, style)

            size       = out_path.stat().st_size
            pcm_len    = max(0, len(_get_pcm_from_wav_bytes(wav_bytes)))
            duration_s = pcm_len / (AZURE_SAMPLE_RATE * 2)
            print(f"  [fallback] {item_id}  ({duration_s:.2f}s, {size:,} bytes)")

            results.append({
                "item_id":      item_id,
                "speaker":      vo.get("speaker", "narrator"),
                "voice":        voice,
                "azure_lang":   azure_lang,
                "style":        style or "",
                "style_degree": style_degree,
                "rate":         rate,
                "output":       str(out_path),
                "size_bytes":   size,
                "duration_sec": round(duration_s, 3),
                "status":       "success",
            })
        except Exception as exc:
            print(f"  [fallback ERROR] {item_id}: {exc}")
            results.append({
                "item_id":    item_id,
                "speaker":    vo.get("speaker", "narrator"),
                "voice":      voice,
                "output":     str(out_path),
                "size_bytes": 0,
                "status":     "failed",
                "error":      str(exc),
            })

    return results


# ---------------------------------------------------------------------------
# Phase 2, Step 8 — HDOmni word-boundary alignment
# ---------------------------------------------------------------------------

def synthesise_with_word_boundaries(
    synthesizer,
    ssml: str,
) -> tuple[bytes, list[dict]]:
    """Synthesise SSML and collect per-word boundary events (HDOmni voices only).

    DragonHDOmni fires synthesis_word_boundary events with audio offset and
    duration per word, enabling accurate sentence alignment without local
    forced-alignment models.

    Returns:
        wav_bytes   : raw WAV bytes (same as synthesise())
        word_events : [{word, start_sec, end_sec, text_offset}, ...]
                      sorted by start_sec ascending
    """
    import azure.cognitiveservices.speech as speechsdk

    word_events_raw: list[dict] = []

    def _on_word_boundary(evt) -> None:
        # Skip Punctuation and SentenceBoundary pseudo-events — Word only
        if hasattr(evt, "boundary_type"):
            bt = str(evt.boundary_type)
            if "Punctuation" in bt or "Sentence" in bt:
                return
        word_events_raw.append({
            "word":           evt.text,
            "start_ticks":    evt.audio_offset,
            "duration_ticks": getattr(evt, "duration", 0),
            "text_offset":    getattr(evt, "text_offset", 0),
        })

    synthesizer.synthesis_word_boundary.connect(_on_word_boundary)
    try:
        result = synthesizer.speak_ssml_async(ssml).get()
    finally:
        synthesizer.synthesis_word_boundary.disconnect_all()

    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        details = result.cancellation_details
        err_lower = (details.error_details or "").lower()
        if any(t in err_lower for t in _REST_TRIGGER):
            # 429 / WebSocket throttle — REST fallback (no word boundaries)
            log.warning("[word_boundary] SDK throttled; falling back to REST (no word events)")
            return _synthesise_rest(ssml), []
        raise RuntimeError(
            f"HDOmni synthesis cancelled: {details.reason} — {details.error_details}"
        )

    # Convert 100-ns ticks to seconds
    word_events: list[dict] = sorted(
        [
            {
                "word":        ev["word"],
                "start_sec":   ev["start_ticks"] / 10_000_000,
                "end_sec":     (ev["start_ticks"] + ev["duration_ticks"]) / 10_000_000,
                "text_offset": ev["text_offset"],
            }
            for ev in word_events_raw
        ],
        key=lambda e: e["start_sec"],
    )
    return result.audio_data, word_events


def _align_by_word_boundaries(
    word_events: list[dict],
    chunk: dict,
) -> list[dict] | None:
    """Map per-word boundary events to sentence start/end offsets (HDOmni).

    Strategy (word-count based):
    1. Tokenise each sentence into words (whitespace split).
    2. Consume that many word events from the sorted event list.
    3. sentence_start = first consumed event's start_sec
       sentence_end   = last consumed event's end_sec

    Returns list of {item_id, start_sec, end_sec} or None if mapping fails.
    """
    sentences = chunk["sentences"]

    if not word_events:
        return None

    def _tokenise(text: str) -> list[str]:
        return [w for w in re.split(r'\s+', text.strip()) if w]

    ev_idx  = 0
    offsets: list[dict] = []

    for sent in sentences:
        words   = _tokenise(sent["text"])
        n_words = len(words)

        if n_words == 0:
            # Punctuation-only sentence — zero-width at current position
            cur_sec = word_events[ev_idx]["start_sec"] if ev_idx < len(word_events) else 0.0
            offsets.append({
                "item_id":   sent["item_id"],
                "start_sec": round(cur_sec, 4),
                "end_sec":   round(cur_sec, 4),
            })
            continue

        if ev_idx + n_words > len(word_events):
            log.warning(f"[word_boundary] insufficient events: need {n_words} more, "
                        f"only {len(word_events) - ev_idx} remain "
                        f"for '{sent['text'][:40]}'")
            return None

        start_sec = word_events[ev_idx]["start_sec"]
        end_sec   = word_events[ev_idx + n_words - 1]["end_sec"]
        offsets.append({
            "item_id":   sent["item_id"],
            "start_sec": round(start_sec, 4),
            "end_sec":   round(end_sec, 4),
        })
        ev_idx += n_words

    return offsets


# ---------------------------------------------------------------------------
# CTC forced alignment (DragonHD / HDFlash — no word-boundary events)
# ---------------------------------------------------------------------------

# Locale → BCP-47 language tag understood by ctc-forced-aligner
_CTC_LANG_MAP: dict[str, str] = {
    "zh-hans": "zh-Hans",
    "zh-cn":   "zh-Hans",
    "zh-tw":   "zh-Hant",
    "zh":      "zh-Hans",
    "en":      "eng",
    "en-us":   "eng",
    "en-gb":   "eng",
    "ja":      "jpn",
    "ko":      "kor",
    "fr":      "fra",
    "es":      "spa",
    "pt":      "por",
    "de":      "deu",
}


def _align_by_ctc(
    chunk_wav_bytes: bytes,
    chunk: dict,
    locale: str,
) -> "list[dict] | None":
    """Align sentences using CTC forced alignment (ctc-forced-aligner library).

    Used as primary alignment path for DragonHD / HDFlash voices (Step 4C
    Path 2 in az.txt) — these voices support neither bookmark events (Path A)
    nor word-boundary events (HDOmni only).

    Strategy:
    1. Concatenate all sentence texts (space-separated).
    2. Run word-level CTC alignment on the concatenated text.
    3. Map word offsets → sentence offsets using word-count bucketing
       (same approach as _align_by_word_boundaries).

    Requires:  pip install ctc-forced-aligner torch
    Falls back gracefully (returns None) if library is not installed or fails.

    Confidence check: text coverage < 0.85 → return None so caller falls
    back to silence-gap or proportional.
    """
    try:
        import ctc_forced_aligner as cfa  # type: ignore[import]
        import torch                       # type: ignore[import]
    except ImportError:
        log.debug("[ctc_align] ctc-forced-aligner / torch not installed — skipping")
        return None

    sentences  = chunk["sentences"]
    lang_key   = locale.lower()
    lang       = _CTC_LANG_MAP.get(lang_key, "eng")
    full_text  = " ".join(s["text"] for s in sentences)

    if not full_text.strip():
        return None

    tmp_wav: str | None = None
    try:
        # Write chunk WAV to a temp file (ctc-forced-aligner reads from path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tf.write(chunk_wav_bytes)
            tmp_wav = tf.name

        device = "cpu"
        model, tokenizer, dev = cfa.load_alignment_model(device, dtype=torch.float32)

        audio_waveform, audio_sample_rate = cfa.load_audio(
            tmp_wav, model.dtype, dev
        )

        emissions, stride = cfa.generate_emissions(
            model, audio_waveform,
            window_size=30, context_size=2, batch_size=8,
        )

        tokens = cfa.preprocess_text(full_text, lang, tokenizer)
        segments, scores = cfa.get_alignments(emissions, tokens, tokenizer)
        spans = cfa.get_segments(segments, audio_waveform, full_text)

        # Confidence check: text coverage
        aligned_chars = sum(len(sp.get("text", "")) for sp in spans)
        coverage = aligned_chars / max(len(full_text.replace(" ", "")), 1)
        if coverage < 0.85:
            log.warning(f"[ctc_align] low text coverage {coverage:.2f} < 0.85 — skipping")
            return None

        # Build word_events compatible with _align_by_word_boundaries
        word_events: list[dict] = [
            {
                "word":      sp.get("text", ""),
                "start_sec": float(sp.get("start", 0.0)),
                "end_sec":   float(sp.get("end",   0.0)),
            }
            for sp in spans
            if sp.get("text", "").strip()
        ]

        if not word_events:
            return None

        return _align_by_word_boundaries(word_events, chunk)

    except Exception as exc:
        log.warning(f"[ctc_align] alignment failed: {exc}")
        return None
    finally:
        if tmp_wav:
            try:
                os.unlink(tmp_wav)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Phase 2, Step 6 — Mode selection + chunk_alignment for manifest items
# ---------------------------------------------------------------------------

_MODE_BATCH_BOOKMARK  = "batch_bookmark"
_MODE_CHUNK_ALIGNMENT = "chunk_alignment"
_MODE_PER_ITEM_LEGACY = "per_item_legacy"


def _select_synthesis_mode(
    items: list[dict],
    ssml_narration: bool,
    asset_id: str | None,
) -> str:
    """Select the TTS synthesis mode for this run (az.txt §Step 6).

    Decision tree:
    ┌─ asset_id set                                    → per_item_legacy
    ├─ ssml_narration                                  → chunk_alignment
    │    (DragonHD — typical — can't use bookmarks)
    ├─ all voices in BATCH_VOICES + count ≤ max        → batch_bookmark
    ├─ any HD voice in unsupported set                 → chunk_alignment
    └─ other unsupported (custom endpoint, etc.)       → per_item_legacy
    """
    if asset_id is not None:
        return _MODE_PER_ITEM_LEGACY

    if ssml_narration:
        return _MODE_CHUNK_ALIGNMENT

    voices_used  = {it["voice"] for it in items}
    unsupported  = voices_used - BATCH_VOICES

    if not unsupported and len(items) <= BATCH_MAX_VOICE_ELEMENTS:
        return _MODE_BATCH_BOOKMARK

    if any(_is_hd_voice(v) for v in unsupported):
        return _MODE_CHUNK_ALIGNMENT

    return _MODE_PER_ITEM_LEGACY


def _chunk_meta(
    chunk_idx: int,
    chunk: dict,
    chunk_wav_path: "Path | None",
) -> dict:
    """Build chunk-level metadata dict for gen_tts_cloud_chunks.json."""
    meta: dict = {
        "chunk_id":    chunk_idx,
        "voice":       chunk["voice"],
        "style":       chunk["style"] or "",
        "rate":        chunk["rate"],
        "sentences":   [s["item_id"] for s in chunk["sentences"]],
        "total_chars": chunk["total_chars"],
        "est_dur_sec": round(chunk["estimated_dur_sec"], 2),
    }
    if chunk_wav_path is not None and chunk_wav_path.exists():
        meta["chunk_wav"] = str(chunk_wav_path)
    return meta


def run_chunk_alignment_from_items(
    synthesizer,
    items: list[dict],
    out_dir: Path,
    assets_dir: "Path | None" = None,
    keep_chunks: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Chunk-alignment synthesis for manifest-sourced vo_items.

    Phase 2 extension: used by run() when:
    - mode == chunk_alignment AND ssml_narration == False
    - Typical case: continuous_narration / illustrated_narration with HD voices
      that are not in BATCH_VOICES (no bookmark support).

    Multi-voice handling: chunk boundaries are forced at voice changes so every
    chunk is voice-homogeneous and shares one <voice>/<prosody>/<express-as>.

    Returns:
        (results, chunks_meta) where results = per-sentence result dicts and
        chunks_meta = list of dicts written to gen_tts_cloud_chunks.json.
    """
    t_start = time.time()
    if not items:
        return [], []

    locale     = items[0]["locale"]
    azure_lang = LOCALE_TO_AZURE_LANG.get(locale.lower(), "en-US")

    cache_dir = (assets_dir / "meta" / "tts_cache") if assets_dir else (
        out_dir.parent.parent.parent / "meta" / "tts_cache"
    )

    # Build sentence_frags from manifest items (text already present)
    sentence_frags = [
        {"text": it["text"], "pause_ms": it.get("break_ms", 0)} for it in items
    ]

    # Group into voice-homogeneous, duration-limited chunks.
    # Hard break at every voice change to keep chunks voice-homogeneous.
    all_chunks: list[dict] = []

    cur_items: list[dict] = []
    cur_frags: list[dict] = []

    def _flush_group() -> None:
        if not cur_items:
            return
        vo0   = cur_items[0]
        voice = vo0["voice"]
        style = vo0.get("style") or vo0.get("tts_style") or vo0.get("azure_style")
        s_deg = float(vo0.get("style_degree", 1.5))
        rate  = vo0.get("rate", "0%")
        pitch = vo0.get("pitch")
        lang  = vo0.get("azure_lang", azure_lang)

        group_chunks = group_sentences_into_chunks(
            sentence_frags = list(cur_frags),
            items          = list(cur_items),
            voice          = voice,
            style          = style,
            style_degree   = s_deg,
            rate           = rate,
            pitch          = pitch,
            azure_lang     = lang,
            locale         = locale,
        )
        all_chunks.extend(group_chunks)

    prev_voice: str | None = None
    for it, fr in zip(items, sentence_frags):
        cur_voice = it["voice"]
        if prev_voice is not None and cur_voice != prev_voice:
            _flush_group()
            cur_items = []
            cur_frags = []
        cur_items.append(it)
        cur_frags.append(fr)
        prev_voice = cur_voice
    _flush_group()

    n_chunks           = len(all_chunks)
    stat_raw_chars     = sum(len(it["text"]) for it in items)
    stat_ssml_chars    = 0
    stat_api_calls     = 0
    stat_cache_hits    = 0
    stat_align_success = 0
    stat_align_fallback = 0

    print(f"\n[CHUNK-ALIGN] {len(items)} items → {n_chunks} chunks  locale={locale}")
    print(f"[CHUNK-ALIGN] Raw spoken chars: {stat_raw_chars:,}")

    results:     list[dict] = []
    chunks_meta: list[dict] = []

    for chunk_idx, chunk in enumerate(all_chunks):
        sents       = chunk["sentences"]
        n           = len(sents)
        voice       = chunk["voice"]
        style       = chunk["style"]
        style_deg   = chunk["style_degree"]
        rate        = chunk["rate"]
        pitch       = chunk["pitch"]
        chunk_label = f"chunk {chunk_idx+1}/{n_chunks} ({n} sent, voice={voice.split(':')[-1]})"
        print(f"\n[chunk] {chunk_label}  est={chunk['estimated_dur_sec']:.1f}s")

        chunk_ssml = _build_chunk_ssml(chunk)
        ssml_chars = len(chunk_ssml)
        stat_ssml_chars += ssml_chars

        cache_key   = _tts_cache_key(chunk_ssml, voice, locale)
        wav_bytes   = _tts_cache_get(cache_dir, cache_key)
        word_events: list[dict] | None = None

        if wav_bytes is not None:
            stat_cache_hits += 1
            print(f"  [cache HIT]  {cache_key[:16]}…")
        else:
            try:
                if _is_hd_omni_voice(voice):
                    wav_bytes, word_events = synthesise_with_word_boundaries(
                        synthesizer, chunk_ssml
                    )
                else:
                    wav_bytes = synthesise(synthesizer, chunk_ssml)
                stat_api_calls += 1
                _tts_cache_put(cache_dir, cache_key, wav_bytes, {
                    "voice": voice, "locale": locale, "chunk_idx": chunk_idx,
                    "n_sentences": n, "ssml_chars": ssml_chars,
                })
            except Exception as exc:
                log.error(f"[chunk] synthesis error for {chunk_label}: {exc}")
                print(f"  [FALLBACK] per-sentence: {exc}")
                stat_align_fallback += n
                fb = _per_item_legacy_chunk(
                    synthesizer, chunk, voice, style, style_deg, rate, pitch,
                    chunk["azure_lang"], out_dir,
                )
                results.extend(fb)
                chunks_meta.append(_chunk_meta(chunk_idx, chunk, None))
                continue

        # --- Alignment ---
        pcm       = _get_pcm_from_wav_bytes(wav_bytes)
        chunk_dur = _pcm_duration_sec(pcm)
        aligned: list[dict] | None = None

        if n == 1:
            aligned = [{"item_id": sents[0]["item_id"],
                        "start_sec": 0.0, "end_sec": round(chunk_dur, 4)}]
            stat_align_success += 1
        else:
            # Path 0: word-boundary events (HDOmni only)
            if word_events is not None:
                aligned = _align_by_word_boundaries(word_events, chunk)
                if aligned is not None:
                    aligned = _clamp_alignment(aligned, chunk_dur)
                    if not _validate_alignment(aligned, chunk_dur):
                        aligned = None
                    else:
                        stat_align_success += n
                        print(f"  [align] word-boundary ({n} sentences)")

            # Path 1: CTC forced alignment (DragonHD / HDFlash — az.txt §Step 4C Path 2)
            if aligned is None and not _is_hd_omni_voice(voice):
                chunk_locale = chunk.get("locale", locale)
                aligned = _align_by_ctc(wav_bytes, chunk, chunk_locale)
                if aligned is not None:
                    aligned = _clamp_alignment(aligned, chunk_dur)
                    if not _validate_alignment(aligned, chunk_dur):
                        aligned = None
                    else:
                        stat_align_success += n
                        print(f"  [align] CTC forced alignment ({n} sentences)")

            # Path 2: silence-gap detection
            if aligned is None:
                aligned = _align_by_silence(wav_bytes, chunk)
                if aligned is not None:
                    aligned = _clamp_alignment(aligned, chunk_dur)
                    if not _validate_alignment(aligned, chunk_dur):
                        aligned = None
                    else:
                        stat_align_success += n
                        print(f"  [align] silence-gap ({n} sentences)")

            # Path 3: proportional fallback
            if aligned is None:
                aligned = _align_proportional(wav_bytes, chunk)
                aligned = _clamp_alignment(aligned, chunk_dur)
                if not _validate_alignment(aligned, chunk_dur):
                    log.error(f"[chunk] all alignment paths failed for {chunk_label} → per-sentence")
                    stat_align_fallback += n
                    fb = _per_item_legacy_chunk(
                        synthesizer, chunk, voice, style, style_deg, rate, pitch,
                        chunk["azure_lang"], out_dir,
                    )
                    results.extend(fb)
                    chunks_meta.append(_chunk_meta(chunk_idx, chunk, None))
                    continue
                else:
                    stat_align_fallback += 1
                    stat_align_success  += n - 1
                    print(f"  [align] proportional fallback (CTC+silence unavailable)")

        # --- Phase 3, Step 10: optionally keep full chunk WAV ---
        chunk_wav_path: Path | None = None
        if keep_chunks:
            chunk_wav_path = out_dir.parent / "vo_chunks" / f"chunk_{chunk_idx:04d}.wav"
            chunk_wav_path.parent.mkdir(parents=True, exist_ok=True)
            chunk_wav_path.write_bytes(wav_bytes)

        # --- Slice per-sentence WAVs ---
        slice_results = _write_sentence_wavs_from_chunk(
            chunk_wav_bytes = wav_bytes,
            offsets         = aligned,
            out_dir         = out_dir,
            voice           = voice,
            style           = style,
        )
        item_meta = {s["item_id"]: s["vo"] for s in sents}
        for r in slice_results:
            vo = item_meta.get(r["item_id"], {})
            r.update({
                "speaker":       vo.get("speaker", "narrator"),
                "voice":         voice,
                "azure_lang":    chunk["azure_lang"],
                "style":         style or "",
                "style_degree":  style_deg,
                "rate":          rate,
                "source_chunk":  chunk_idx,
            })
        results.extend(slice_results)
        chunks_meta.append(_chunk_meta(chunk_idx, chunk, chunk_wav_path))

    wall_time      = time.time() - t_start
    overhead_ratio = stat_ssml_chars / max(stat_raw_chars, 1)
    ok             = sum(1 for r in results if r.get("status") == "success")

    print(f"\n[CHUNK-ALIGN] ═══ Instrumentation ═══")
    print(f"  raw chars    : {stat_raw_chars:,}")
    print(f"  SSML chars   : {stat_ssml_chars:,}  (ratio {overhead_ratio:.2f}×)")
    print(f"  API calls    : {stat_api_calls}  cache hits: {stat_cache_hits}")
    print(f"  chunks       : {n_chunks}")
    print(f"  align ok     : {stat_align_success}  fallback: {stat_align_fallback}")
    print(f"  wall time    : {wall_time:.1f}s")
    print(f"  items OK     : {ok}/{len(items)}")

    stats = {
        "mode":              "chunk_alignment",
        "items_total":       len(items),
        "items_synthesized": ok,
        "items_skipped":     len(items) - ok,
        "items_cache_hit":   stat_cache_hits,
        "raw_chars":         stat_raw_chars,
        "ssml_chars":        stat_ssml_chars,
        "ssml_ratio":        round(overhead_ratio, 2),
        "api_calls":         stat_api_calls,
        "cache_hits":        stat_cache_hits,
        "chunks":            n_chunks,
        "align_ok":          stat_align_success,
        "align_fallback":    stat_align_fallback,
        "wall_time_sec":     round(wall_time, 1),
    }
    return results, chunks_meta, stats


# ---------------------------------------------------------------------------
# Phase 2, Step 7 — write_tts_results() centralization
# ---------------------------------------------------------------------------

def write_tts_results(
    results: list[dict],
    meta_dir: Path,
    chunks_meta: "list[dict] | None" = None,
) -> Path:
    """Write gen_tts_cloud_results.json (and gen_tts_cloud_chunks.json if provided).

    Centralises all TTS output writing — replaces scattered results_path logic
    in main().  Always called at the end of every synthesis run.

    Returns the path to gen_tts_cloud_results.json.
    """
    meta_dir.mkdir(parents=True, exist_ok=True)

    results_path = meta_dir / f"{SCRIPT_NAME}_results.json"
    if results_path.exists():
        results_path.unlink()
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if chunks_meta is not None:
        chunks_path = meta_dir / f"{SCRIPT_NAME}_chunks.json"
        chunks_path.write_text(
            json.dumps(chunks_meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return results_path


# ---------------------------------------------------------------------------
# TTS Audit Log
# ---------------------------------------------------------------------------

def _append_tts_audit_log(meta_dir: Path, locale: str, stats: dict) -> None:
    """Append a synthesis run entry to tts_audit_log.json.

    File location: assets/meta/tts_audit_log.json
    Accumulates across re-runs; each entry records per-run stats.
    Accumulated totals per locale track lifetime chars sent to Azure.
    """
    import datetime

    log_path = meta_dir / "tts_audit_log.json"

    if log_path.exists():
        try:
            audit = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            audit = {}
    else:
        audit = {}

    # Derive project_id / episode_id from path:
    # meta_dir = projects/{slug}/episodes/{ep_id}/assets/meta
    try:
        ep_id      = meta_dir.parent.parent.name
        project_id = meta_dir.parent.parent.parent.name
    except Exception:
        ep_id = project_id = ""

    audit.setdefault("project_id",   project_id)
    audit.setdefault("episode_id",   ep_id)
    audit.setdefault("runs",         [])
    audit.setdefault("accumulated",  {})

    entry = {
        "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "locale":    locale,
        **stats,
    }
    audit["runs"].append(entry)

    acc = audit["accumulated"].setdefault(locale, {
        "total_raw_chars":        0,
        "total_ssml_chars":       0,
        "total_api_calls":        0,
        "total_items_synthesized": 0,
        "total_items_skipped":    0,
        "runs":                   0,
    })
    acc["total_raw_chars"]         += stats.get("raw_chars",          0)
    acc["total_ssml_chars"]        += stats.get("ssml_chars",         0)
    acc["total_api_calls"]         += stats.get("api_calls",          0)
    acc["total_items_synthesized"] += stats.get("items_synthesized",  0)
    acc["total_items_skipped"]     += stats.get("items_skipped",      0)
    acc["runs"]                    += 1

    meta_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  ✓ tts_audit_log → {log_path}")


# ---------------------------------------------------------------------------
# Phase 3, Step 9 — Azure Batch Synthesis REST API (standard voices only)
# ---------------------------------------------------------------------------
#
# SCOPE: Standard Neural voices only.  HD/Dragon voices are real-time only
# (confirmed in az.txt §O4 — Batch Synthesis API does not support DragonHD,
# HDFlash, or HDOmni).  This path is NOT enabled by default; use
# --use-batch-rest to opt in.
#
# Why it matters: allows >50 sentences per request (SDK batch is capped at 50
# <voice> elements).  Returns sentence-boundary JSON → no bookmark split needed.
# Trade-off: async (submit → poll → download), adds ~30–120s latency overhead.
# ---------------------------------------------------------------------------

_BATCH_REST_API_VERSION = "2024-04-01"
_BATCH_REST_POLL_INTERVAL = 5.0   # seconds between status checks


def _batch_rest_headers(json_body: bool = True) -> dict:
    """Build HTTP headers for Azure Batch Synthesis REST requests."""
    key = os.environ.get("AZURE_SPEECH_KEY", "")
    if not key:
        raise RuntimeError("AZURE_SPEECH_KEY not set for Batch REST API")
    hdrs = {"Ocp-Apim-Subscription-Key": key}
    if json_body:
        hdrs["Content-Type"] = "application/json"
    return hdrs


def _batch_rest_base_url() -> str:
    """Construct the Batch Synthesis REST base URL from AZURE_SPEECH_REGION."""
    region = os.environ.get("AZURE_SPEECH_REGION", "")
    if not region:
        raise RuntimeError("AZURE_SPEECH_REGION not set for Batch REST API")
    return (
        f"https://{region}.customvoice.api.speech.microsoft.com"
        f"/api/texttospeech/3.1-preview1/batchsyntheses"
    )


def batch_synthesis_rest_submit(
    ssml: str,
    synthesis_id: str,
    output_format: str = "riff-24khz-16bit-mono-pcm",
) -> str:
    """Submit an Azure Batch Synthesis job via REST.

    Phase 3, Step 9 — standard Neural voices only.

    Args:
        ssml           : full SSML document string
        synthesis_id   : unique job identifier (caller-provided UUID)
        output_format  : Azure audio output format string

    Returns the job URL to use for polling.
    """
    import urllib.request
    import urllib.error

    base_url = _batch_rest_base_url()
    job_url  = f"{base_url}/{synthesis_id}?api-version={_BATCH_REST_API_VERSION}"
    headers  = _batch_rest_headers(json_body=True)
    body = json.dumps({
        "displayName": f"pipe-{synthesis_id[:8]}",
        "description": "gen_tts_cloud Phase 3 batch synthesis",
        "textType":    "SSML",
        "inputs":      [{"text": ssml}],
        "properties":  {
            "outputFormat":            output_format,
            "sentenceBoundaryEnabled": True,
            "wordBoundaryEnabled":     False,
        },
    }).encode("utf-8")

    req = urllib.request.Request(job_url, data=body, headers=headers, method="PUT")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()   # discard response body; status 201 = submitted
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Batch REST submit HTTP {exc.code}: {exc.read()[:300].decode()}"
        )

    log.info(f"[batch-rest] job submitted: {synthesis_id}")
    return job_url


def batch_synthesis_rest_poll(
    job_url: str,
    timeout_sec: float = 600.0,
) -> dict:
    """Poll a Batch Synthesis job until it succeeds, fails, or times out.

    Returns the final job response dict (status = "Succeeded").
    Raises RuntimeError on failure or TimeoutError on timeout.
    """
    import urllib.request
    import urllib.error

    headers  = _batch_rest_headers(json_body=False)
    deadline = time.time() + timeout_sec

    while time.time() < deadline:
        req = urllib.request.Request(job_url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                job = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Batch REST poll HTTP {exc.code}")

        status = job.get("status", "Unknown")
        print(f"  [batch-rest] status={status}")

        if status == "Succeeded":
            return job
        if status in ("Failed", "Canceled"):
            err = job.get("properties", {}).get("error", "")
            raise RuntimeError(f"Batch synthesis {status}: {err}")

        time.sleep(_BATCH_REST_POLL_INTERVAL)

    raise TimeoutError(f"Batch synthesis did not complete within {timeout_sec:.0f}s")


def _parse_timespan(ts: str) -> float:
    """Parse Azure TimeSpan string to seconds.

    Azure Batch Synthesis uses "hh:mm:ss.fffffff" (100-ns tick precision).
    Examples: "00:00:01.5000000" → 1.5,  "00:01:23.4567890" → 83.456789
    """
    ts = ts.strip()
    try:
        parts = ts.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        return float(ts)
    except (ValueError, IndexError):
        return 0.0


def batch_synthesis_rest_download(job: dict) -> tuple[bytes, list[dict]]:
    """Download WAV + timing JSON from a completed Batch Synthesis job.

    The result is a SAS-authenticated ZIP containing:
    - 0001.wav  : synthesised audio
    - 0001.json : boundary info (SentenceBoundary + WordBoundary entries)

    Returns:
        (wav_bytes, sentence_boundaries) where sentence_boundaries is a list of
        {"text_offset": int, "word_length": int,
         "start_sec": float, "end_sec": float, "text": str}
        for each SentenceBoundary entry, sorted by start_sec.
        Empty list if timing JSON is absent or unparseable.
    """
    import urllib.request
    import urllib.error

    outputs    = job.get("outputs", {})
    result_url = outputs.get("result")
    if not result_url:
        raise RuntimeError("Batch synthesis job has no 'result' URL in outputs")

    # SAS URL — no auth header needed
    req = urllib.request.Request(result_url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            zip_bytes = resp.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Batch REST download HTTP {exc.code}")

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()

        # Extract WAV
        wav_names = [n for n in names if n.lower().endswith(".wav")]
        if not wav_names:
            raise RuntimeError("Batch synthesis ZIP contains no WAV files")
        wav_bytes = zf.read(wav_names[0])

        # Extract + parse timing JSON
        json_names = [n for n in names if n.lower().endswith(".json")]
        sentence_boundaries: list[dict] = []

        for jname in json_names:
            try:
                timing = json.loads(zf.read(jname).decode("utf-8"))
                # timing may be a list of objects or a single object
                if isinstance(timing, dict):
                    timing = [timing]
                for doc in timing:
                    for entry in doc.get("Boundary", []):
                        if entry.get("BoundaryType") != "SentenceBoundary":
                            continue
                        start_sec = _parse_timespan(entry.get("AudioOffset", "0"))
                        dur_sec   = _parse_timespan(entry.get("Duration", "0"))
                        sentence_boundaries.append({
                            "text":        entry.get("Text", ""),
                            "text_offset": entry.get("TextOffset", 0),
                            "word_length": entry.get("WordLength", 0),
                            "start_sec":   start_sec,
                            "end_sec":     start_sec + dur_sec,
                        })
            except Exception as exc:
                log.warning(f"[batch-rest] could not parse timing JSON {jname}: {exc}")

        sentence_boundaries.sort(key=lambda e: e["start_sec"])
        if sentence_boundaries:
            print(f"  [batch-rest] parsed {len(sentence_boundaries)} sentence boundaries")

        return wav_bytes, sentence_boundaries


def synthesise_batch_rest(ssml: str) -> bytes:
    """Full Batch Synthesis REST cycle: submit → poll → download → WAV bytes.

    Phase 3, Step 9.  Standard Neural voices only.  Synchronous from caller.

    Use cases:
    - More than BATCH_MAX_VOICE_ELEMENTS sentences (SDK batch cap is 50)
    - --use-batch-rest flag explicitly requested
    - Avoids WebSocket entirely (different throttle budget from real-time API)
    """
    synthesis_id = str(uuid.uuid4())
    print(f"  [batch-rest] submitting {synthesis_id[:8]}…")
    job_url  = batch_synthesis_rest_submit(ssml, synthesis_id)
    job      = batch_synthesis_rest_poll(job_url)
    wav_bytes, _sent_bounds = batch_synthesis_rest_download(job)
    print(f"  [batch-rest] downloaded {len(wav_bytes):,} bytes")
    return wav_bytes


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def locale_from_manifest(manifest: dict, path: str) -> str:
    """Read locale from manifest field first; fall back to filename parsing.

    Manifest field (locale_scope='locale' manifests always have locale set):
      manifest["locale"] = "zh-Hans"  →  "zh-Hans"

    Filename fallback:
      VOPlan.zh-Hans.json  →  "zh-Hans"
      VOPlan.en.json       →  "en"
      VOPlan.json          →  "en"
    """
    if manifest.get("locale"):
        return manifest["locale"]
    stem = Path(path).stem          # e.g. "VOPlan.zh-Hans"
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
    """Load and resolve VO items from a locale VOPlan.

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


def _synthesise_rest(ssml: str, retries: int = 8) -> bytes:
    """Fallback: synthesise via HTTPS REST (avoids WebSocket 429 throttling).

    Uses exponential backoff on 429 / 5xx responses.
    Output format: riff-48khz-16bit-mono-pcm to match AZURE_SAMPLE_RATE.
    """
    import urllib.request
    import urllib.error

    ssml   = _minify_ssml(ssml)   # az.txt §Step 11: strip comments + whitespace
    key    = os.environ.get("AZURE_SPEECH_KEY", "")
    region = os.environ.get("AZURE_SPEECH_REGION", "")
    if not key or not region:
        raise RuntimeError("AZURE_SPEECH_KEY / AZURE_SPEECH_REGION not set for REST fallback")

    url     = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type":              "application/ssml+xml",
        "X-Microsoft-OutputFormat":  "riff-48khz-16bit-mono-pcm",
        "User-Agent":                "gen-tts-cloud-rest",
    }
    body = ssml.encode("utf-8")

    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504):
                wait = min(2 ** attempt, 60)
                print(f"  [REST] HTTP {exc.code} — backoff {wait}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
                continue
            raise RuntimeError(f"REST TTS HTTP {exc.code}: {exc.read()[:200]}")

    raise RuntimeError("REST TTS exhausted retries (still throttled or unreachable)")


_REST_TRIGGER = ("429", "too many requests", "websocket upgrade failed",
                 "websocket", "throttl")


def synthesise(synthesizer, ssml: str) -> bytes:
    """Submit SSML to Azure and return raw WAV bytes.

    Primary path: Azure Speech SDK (WebSocket).
    On 429 / throttle cancellation: automatic fallback to HTTPS REST with
    exponential backoff (up to 8 retries).
    """
    import azure.cognitiveservices.speech as speechsdk

    ssml   = _minify_ssml(ssml)   # az.txt §Step 11: strip comments + whitespace
    result = synthesizer.speak_ssml_async(ssml).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return result.audio_data

    details   = result.cancellation_details
    err_lower = (details.error_details or "").lower()

    # Detect 429 / WebSocket throttle → retry via REST
    if any(t in err_lower for t in _REST_TRIGGER):
        print(f"  [WARN] SDK throttled ({details.error_details[:120]}). "
              f"Switching to REST with backoff…")
        return _synthesise_rest(ssml)

    raise RuntimeError(
        f"Azure TTS cancelled: reason={details.reason}  detail={details.error_details}"
    )


# ---------------------------------------------------------------------------
# Backend runner
# ---------------------------------------------------------------------------

def _synthesise_per_item(synthesizer, items: list[dict], out_dir: Path, force: bool = False) -> list[dict]:
    """Per-item synthesis loop (original implementation, preserved as fallback).

    Called by run() when:
      - --asset-id is set (single-item path), OR
      - batch synthesis raises an exception (fallback path)
    """
    results = []
    total   = len(items)

    for idx, vo in enumerate(items, start=1):
        out_path = out_dir / f"{vo['item_id']}.wav"

        # Skip already-valid WAVs (size > 44 = more than just a WAV header)
        if not force and out_path.exists() and out_path.stat().st_size > 44:
            size      = out_path.stat().st_size
            pcm_bytes = max(0, size - 44)
            duration_s = pcm_bytes / (AZURE_SAMPLE_RATE * 2)
            print(f"\n[{idx}/{total}] {vo['item_id']}  [SKIP — already exists {duration_s:.2f}s]")
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
                "status":       "skipped",
            })
            continue

        _style_tag  = vo['style'] or 'no-style'
        _rate_tag   = f"rate={vo['rate']}" if vo.get('rate') else ''
        _pitch_tag  = f"pitch={vo['pitch']}" if vo.get('pitch') else ''
        _extra      = '  '.join(t for t in [_rate_tag, _pitch_tag] if t)
        _text_snip  = vo['text'][:80] + ('…' if len(vo['text']) > 80 else '')
        print(f"[{idx}/{total}] {vo['item_id']}  {vo['voice']} / {_style_tag}"
              + (f"  {_extra}" if _extra else "")
              + f"  \"{_text_snip}\"")

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
            # Delete empty/partial output so post_tts_analysis.py doesn't choke on it
            if out_path.exists() and out_path.stat().st_size <= 44:
                out_path.unlink()
                print(f"  [CLEANUP] Deleted empty/partial {out_path.name}")
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
# ssml_narration mode — wrapper-rebuild + inner passthrough
# ---------------------------------------------------------------------------

def _detect_sentence_boundaries(inner_xml: str, locale_hint: str = "en") -> list[tuple[int, int]]:
    """Find (start, end) char positions of each sentence in ssml_inner.xml content.

    Sentences are delimited by:
      - For zh: 。！？…
      - For en: . ! ? … followed by whitespace or end-of-string
    Returns a list of (start_pos, end_pos) tuples covering all text.
    """
    import re as _re
    # Strip XML tags to get pure text positions mapping
    # We need to identify sentence boundaries in the raw inner XML
    # Strategy: walk through the content, tracking text vs tags
    text_runs: list[tuple[int, int, str]] = []  # (start_in_xml, end_in_xml, text)
    i = 0
    while i < len(inner_xml):
        if inner_xml[i] == '<':
            # Skip over entire tag
            end = inner_xml.find('>', i)
            if end == -1:
                break
            i = end + 1
        else:
            # Text content
            start = i
            while i < len(inner_xml) and inner_xml[i] != '<':
                i += 1
            text_runs.append((start, i, inner_xml[start:i]))

    # Concatenate all text
    full_text = "".join(t for _, _, t in text_runs)

    # Find sentence boundaries in full text
    if locale_hint.startswith("zh"):
        pat = _re.compile(r'[。！？…]+')
    else:
        pat = _re.compile(r'[.!?…]+(?:\s|$)')

    boundaries: list[int] = [0]
    for m in pat.finditer(full_text):
        boundaries.append(m.end())
    if boundaries[-1] < len(full_text):
        boundaries.append(len(full_text))

    # Map text-position boundaries back to XML positions
    result: list[tuple[int, int]] = []
    text_pos = 0
    xml_boundary_positions: list[int] = [0]

    for xml_start, xml_end, text in text_runs:
        for b in boundaries:
            if text_pos < b <= text_pos + len(text):
                offset_in_run = b - text_pos
                xml_boundary_positions.append(xml_start + offset_in_run)
        text_pos += len(text)

    xml_boundary_positions.append(len(inner_xml))
    # Deduplicate and sort
    xml_boundary_positions = sorted(set(xml_boundary_positions))

    for i in range(len(xml_boundary_positions) - 1):
        s, e = xml_boundary_positions[i], xml_boundary_positions[i + 1]
        chunk = inner_xml[s:e].strip()
        # Keep chunks that contain actual text (not purely <break> tags).
        # Previous bug: startswith('<break') filtered out chunks like
        # "<break time='800ms'/>\n岁月拓宽了长河。" which DO have text.
        text_only = _re.sub(r'<[^>]+>', '', chunk).strip()
        if text_only:
            result.append((s, e))

    return result if result else [(0, len(inner_xml))]


def build_ssml_narration(
    ssml_inner: str,
    voice_cast: dict,
    locale: str,
    sentence_ids: list[str],
) -> tuple[str, list[str]]:
    """Build SSML for ssml_narration: wrapper-rebuild + inner passthrough.

    Reads VoiceCast.json for Layer 1 (voice/style/rate/pitch).
    Reads ssml_inner.xml for Layer 2 (text + breaks, immutable).
    Injects auto_sent_NNN bookmarks at sentence boundaries.

    Returns:
        ssml     : str       — the reconstructed SSML document
        bk_ids   : list[str] — bookmark IDs in order (auto_sent_001, ...)
    """
    azure_lang = LOCALE_TO_AZURE_LANG.get(locale.lower(), "en-US")

    # Find narrator in VoiceCast
    narrator = None
    for ch in voice_cast.get("characters", []):
        if ch.get("character_id") == "narrator":
            narrator = ch
            break
    if narrator is None:
        raise ValueError("VoiceCast.json has no narrator entry")

    # Get locale-specific voice params
    vc_locale = narrator.get(locale) or narrator.get("en") or {}
    voice     = vc_locale.get("azure_voice", "en-US-GuyNeural")
    style     = vc_locale.get("azure_style")
    style_deg = vc_locale.get("azure_style_degree", 1.5)
    rate      = vc_locale.get("azure_rate", "0%")
    pitch     = vc_locale.get("azure_pitch")

    # Detect sentence boundaries in inner XML
    locale_hint = "zh" if "zh" in locale.lower() else "en"
    boundaries = _detect_sentence_boundaries(ssml_inner, locale_hint)

    # Build inner content with bookmarks injected at sentence boundaries.
    # A tiny leading silence ensures Azure fires the first bookmark event —
    # without it, a bookmark at audio offset 0 may be silently dropped.
    bk_ids: list[str] = []
    parts: list[str] = ["<break time='100ms'/>"]
    prev_end = 0

    for idx, (s, e) in enumerate(boundaries):
        # Content between previous sentence end and this sentence start
        if s > prev_end:
            parts.append(ssml_inner[prev_end:s])

        bk_id = f"auto_sent_{idx+1:03d}"
        bk_ids.append(bk_id)
        parts.append(f"<bookmark mark='{bk_id}'/>")
        parts.append(ssml_inner[s:e])
        prev_end = e

    # Any trailing content
    if prev_end < len(ssml_inner):
        parts.append(ssml_inner[prev_end:])

    inner_with_bookmarks = "".join(parts)

    # Build prosody wrapper
    prosody_attrs: list[str] = []
    if rate and rate != "0%":
        prosody_attrs.append(f'rate="{rate}"')
    if pitch:
        prosody_attrs.append(f'pitch="{pitch}"')

    spoken = inner_with_bookmarks
    if prosody_attrs:
        spoken = f"<prosody {' '.join(prosody_attrs)}>{spoken}</prosody>"

    # Style wrapper
    if style:
        spoken = (
            f'<mstts:express-as style="{style}" styledegree="{style_deg}">'
            f"{spoken}</mstts:express-as>"
        )

    # Voice + speak wrappers
    ssml = (
        f"<speak version='1.0' xml:lang='{azure_lang}' "
        f"xmlns='http://www.w3.org/2001/10/synthesis' "
        f"xmlns:mstts='http://www.w3.org/2001/mstts'>\n"
        f"  <voice name='{voice}'>{spoken}</voice>\n"
        f"</speak>"
    )

    return ssml, bk_ids


def _parse_ssml_inner_fragments(ssml_inner: str) -> list[dict]:
    """Parse ssml_inner.xml into per-sentence fragments with pause_ms.

    Returns list of {"text": str, "pause_ms": int} dicts, one per sentence.
    Splitting logic mirrors ssml_preprocess.build_script() so the count
    matches the Script.json / manifest vo_items count.
    """
    import re as _re

    # Walk XML content extracting text runs and <break> tags
    fragments: list[tuple] = []     # ("text", str) | ("break", int)
    i = 0
    while i < len(ssml_inner):
        if ssml_inner[i] == '<':
            end = ssml_inner.find('>', i)
            if end == -1:
                break
            tag = ssml_inner[i:end + 1]
            m = _re.match(r'<break\s+time=["\'](\d+)\s*ms["\']\s*/>', tag, _re.IGNORECASE)
            if m:
                fragments.append(("break", int(m.group(1))))
            i = end + 1
        else:
            start = i
            while i < len(ssml_inner) and ssml_inner[i] != '<':
                i += 1
            text = ssml_inner[start:i].strip()
            if text:
                fragments.append(("text", text))

    # Group into sentences: accumulate text, breaks attach as trailing pause.
    # Mirrors ssml_preprocess.build_script() Phase 1 logic.
    raw_chunks: list[dict] = []
    current_text = ""
    for kind, value in fragments:
        if kind == "text":
            current_text += (" " if current_text else "") + value
        elif kind == "break":
            if current_text.strip():
                raw_chunks.append({"text": current_text.strip(), "pause_ms": value})
                current_text = ""
            elif raw_chunks:
                raw_chunks[-1]["pause_ms"] += value
    if current_text.strip():
        raw_chunks.append({"text": current_text.strip(), "pause_ms": 800})

    # Phase 2: split by sentence-ending punctuation within each chunk.
    # Mirrors ssml_preprocess._split_sentences() for zh locale.
    _ZH_RE = _re.compile(r"(?<=[。！？…])")
    result: list[dict] = []
    for chunk in raw_chunks:
        parts = _ZH_RE.split(chunk["text"])
        sents = [s.strip() for s in parts if s.strip()]
        for idx, sent in enumerate(sents):
            if idx < len(sents) - 1:
                result.append({"text": sent, "pause_ms": 800})
            else:
                result.append({"text": sent, "pause_ms": chunk["pause_ms"]})

    return result


def _parse_ssml_blocks(ssml_inner: str) -> list[tuple[str, int]]:
    """Parse ssml_inner into per-block (block_xml, break_ms_after) tuples.

    Each non-break direct child of <voice> is one block (typically one
    <mstts:express-as> element).  The <break> immediately following a content
    block is consumed and its duration attached as break_ms_after (0 if none).

    Returns list of (block_xml, break_ms_after).
    """
    root = ET.fromstring(f"{_SSML_VOICE_OPEN}{ssml_inner}</voice>")
    children = list(root)
    blocks: list[tuple[str, int]] = []
    i = 0
    while i < len(children):
        child = children[i]
        local = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if local != "break":
            block_xml = ET.tostring(child, encoding="unicode", short_empty_elements=True)
            break_ms = 0
            if i + 1 < len(children):
                nxt = children[i + 1]
                nxt_local = nxt.tag.split("}")[-1] if "}" in nxt.tag else nxt.tag
                if nxt_local == "break":
                    t = nxt.get("time", "0ms")
                    m = re.match(r"(\d+)\s*ms", t, re.IGNORECASE)
                    break_ms = int(m.group(1)) if m else 0
                    i += 1   # consume the break
            blocks.append((block_xml, break_ms))
        i += 1
    return blocks


def _run_ssml_narration_passthrough(
    ssml_inner: str,
    voice: str,
    azure_lang: str,
    sentence_frags: list[dict],
    items: list[dict],
    out_dir: Path,
    cache_dir: Path,
    synthesizer,
    locale: str,
    style: str | None,
) -> tuple[list[dict], dict]:
    """Per-block synthesis for multi-block SSML (Changes G + H).

    Each <mstts:express-as> block in ssml_inner is submitted as a separate Azure
    TTS call, producing a short block WAV (5-15 s, 1-4 sentences).  Silence-gap
    alignment runs within each short block WAV — the regime it was designed for.
    A missed boundary in one block corrupts at most that block's sentences; all
    other blocks are unaffected.

    §C3 RESOLVED: per-block synthesis (Option B) is the adopted architecture.
    Full-WAV passthrough is not used.
    """
    t_start = time.time()

    # ── GATE: count check MUST be first — before any I/O or API call ──────────
    if len(sentence_frags) != len(items):
        raise ValueError(
            f"[passthrough] ssml_inner sentence count ({len(sentence_frags)}) != "
            f"manifest items ({len(items)}).  Delete ssml_inner.xml and "
            f"re-run ssml_preprocess before retrying TTS."
        )

    total = len(items)
    print(f"\n[SSML-NARRATION] passthrough mode: {total} sentences  voice={voice}")
    print(f"[SSML-NARRATION] locale={locale}")

    # ── Parse ssml_inner into per-block (block_xml, break_ms_after) pairs ─────
    blocks = _parse_ssml_blocks(ssml_inner)
    print(f"[SSML-NARRATION] {len(blocks)} blocks parsed from ssml_inner")

    # ── Synthesise each block as a separate Azure call ─────────────────────────
    stat_api_calls   = 0
    stat_cache_hits  = 0
    total_ssml_chars = 0
    block_wavs: list[tuple[bytes, int]] = []   # (wav_bytes, break_ms_after)

    for block_idx, (block_xml, break_ms) in enumerate(blocks):
        block_ssml = _minify_ssml(
            f"<speak version='1.0' xml:lang='{azure_lang}'"
            f" xmlns='http://www.w3.org/2001/10/synthesis'"
            f" xmlns:mstts='http://www.w3.org/2001/mstts'>"
            f"<voice name='{voice}'>{block_xml}</voice>"
            f"</speak>"
        )
        ssml_chars = len(block_ssml)
        total_ssml_chars += ssml_chars

        cache_key = _tts_cache_key(block_ssml, voice, locale)
        wav_bytes = _tts_cache_get(cache_dir, cache_key)
        if wav_bytes is not None:
            stat_cache_hits += 1
            print(f"  [block {block_idx+1}/{len(blocks)}] cache HIT  "
                  f"{cache_key[:16]}…  ({len(wav_bytes):,} bytes)")
        else:
            wav_bytes = synthesise(synthesizer, block_ssml)
            print(f"  [block {block_idx+1}/{len(blocks)}] Azure TTS  "
                  f"{len(wav_bytes):,} bytes  chars={ssml_chars}")
            stat_api_calls += 1
            _tts_cache_put(cache_dir, cache_key, wav_bytes, {
                "voice":       voice,
                "locale":      locale,
                "char_count":  ssml_chars,
                "multi_block": True,
                "timestamp":   time.time(),
            })
        block_wavs.append((wav_bytes, break_ms))

    # ── Align and slice per block (Change H) ───────────────────────────────────
    # Sentence partitioning: for each block, count its sentences by running the
    # same fragment parser used for the full ssml_inner.  This guarantees the
    # partition cursor is stable and consistent with ssml_preprocess sentence counts.
    frag_cursor       = 0
    all_results: list[dict] = []
    stat_align_silence       = 0
    stat_align_proportional  = 0

    for block_idx, (wav_bytes, _break_ms) in enumerate(block_wavs):
        block_xml = blocks[block_idx][0]

        # Count sentences in this block using the same parser as the full inner.
        block_sents_parsed = _parse_ssml_inner_fragments(block_xml)
        n = len(block_sents_parsed)
        if n == 0:
            log.warning("[passthrough] block %d has 0 sentences — skipping", block_idx + 1)
            continue

        blk_sfrags = sentence_frags[frag_cursor : frag_cursor + n]
        blk_items  = items[frag_cursor : frag_cursor + n]
        frag_cursor += n

        chunk_dur = _pcm_duration_sec(_get_pcm_from_wav_bytes(wav_bytes))

        # Build chunk dict for existing alignment helpers.
        blk_chunk = {
            "sentences": [
                {
                    "item_id":  blk_items[i]["item_id"],
                    "text":     blk_sfrags[i]["text"],
                    "pause_ms": blk_sfrags[i].get("pause_ms", 800),
                }
                for i in range(len(blk_items))
            ]
        }

        # Path 1: silence-gap (short block WAV — operates in its designed regime).
        aligned = _align_by_silence(wav_bytes, blk_chunk)
        if aligned is not None:
            aligned = _clamp_alignment(aligned, chunk_dur)
            if not _validate_alignment(aligned, chunk_dur):
                log.warning("[passthrough] block %d silence-gap invalid — proportional",
                            block_idx + 1)
                aligned = None
            else:
                stat_align_silence += n

        # Path 2: proportional fallback (always succeeds).
        if aligned is None:
            aligned = _align_proportional(wav_bytes, blk_chunk)
            aligned = _clamp_alignment(aligned, chunk_dur)
            stat_align_proportional += n

        print(f"  [block {block_idx+1}] align={'silence' if aligned and stat_align_silence > 0 else 'proportional'}"
              f"  {n} sent  {chunk_dur:.1f}s")

        slice_results = _write_sentence_wavs_from_chunk(
            chunk_wav_bytes = wav_bytes,
            offsets         = aligned,
            out_dir         = out_dir,
            voice           = voice,
            style           = style,
        )

        # Augment with speaker/voice metadata.
        item_meta = {it["item_id"]: it.get("vo", {}) for it in blk_items}
        for r in slice_results:
            vo = item_meta.get(r["item_id"], {})
            r.update({
                "speaker":    vo.get("speaker", "narrator"),
                "voice":      voice,
                "azure_lang": azure_lang,
                "style":      style or "",
            })
        all_results.extend(slice_results)

    # ── Summary ────────────────────────────────────────────────────────────────
    wall_time = time.time() - t_start
    ok = sum(1 for r in all_results if r.get("status") == "success")
    print(f"\n[SSML-NARRATION] passthrough complete: {ok}/{total} OK  "
          f"blocks={len(blocks)}  api={stat_api_calls}  cache={stat_cache_hits}  "
          f"time={wall_time:.1f}s")
    print(f"  align: silence={stat_align_silence}  proportional={stat_align_proportional}")

    stats = {
        "mode":              "ssml_narration_passthrough",
        "items_total":       total,
        "items_synthesized": ok,
        "items_skipped":     total - ok,
        "items_cache_hit":   stat_cache_hits,
        "ssml_chars":        total_ssml_chars,
        "api_calls":         stat_api_calls,
        "cache_hits":        stat_cache_hits,
        "chunks":            len(blocks),
        "align_silence":     stat_align_silence,
        "align_fallback":    stat_align_proportional,
        "wall_time_sec":     round(wall_time, 1),
    }
    return all_results, stats


def run_ssml_narration(
    ssml_inner_path: str,
    voice_cast_path: str,
    manifest: dict,
    items: list[dict],
    out_dir: Path,
) -> list[dict]:
    """Synthesise ssml_narration content via chunk_alignment synthesis (Phase 1).

    Phase 1 optimization (az.txt plan):
    - Group sentences into ~35 s chunks to amortise <speak> SSML overhead.
    - Synthesise each chunk as a single Azure TTS call (one <speak> per chunk).
    - Align sentence boundaries within the chunk WAV using silence-gap detection
      with proportional character-ratio fallback.
    - Slice per-sentence WAVs for downstream compatibility (no changes to
      post_tts_analysis, gen_render_plan, or render_video required).
    - Cache chunk WAVs by SHA256(normalized_ssml+voice+locale+format+rate+ver).

    Fallback guarantee: if chunk synthesis or both alignment methods fail for
    a given chunk, that chunk falls back to per_item_legacy synthesis (one
    Azure call per sentence) so no items are silently dropped.

    DragonHD voices do not support bookmark events (batch path not viable).
    DragonHDOmni word-boundary alignment is reserved for Phase 2.

    Voice comes from VoiceCast (NOT from manifest — manifest may have wrong
    voice if it was copied from another locale).
    """
    t_start = time.time()

    # ── Approved-cache fast path ──────────────────────────────────────────────
    # Must run BEFORE load_azure_synthesizer() so that a re-run after "VO Approve"
    # never touches Azure — not even credential loading — when all WAVs are locked.
    _fp_locale = items[0]["locale"] if items else "en"
    _fp_cache_dir = out_dir.parent.parent.parent / "meta" / "vo_approved_cache" / _fp_locale
    if _fp_cache_dir.exists() and items:
        import shutil as _shutil_fp
        if all((_fp_cache_dir / f"{it['item_id']}.wav").exists() for it in items):
            print(f"\n[approved-cache] All {len(items)} item(s) have approved WAVs — "
                  f"skipping Azure TTS entirely")
            out_dir.mkdir(parents=True, exist_ok=True)
            for it in items:
                _shutil_fp.copy2(
                    _fp_cache_dir / f"{it['item_id']}.wav",
                    out_dir / f"{it['item_id']}.wav",
                )
            _fp_results = [{"item_id": it["item_id"], "status": "success"} for it in items]
            _fp_stats = {
                "mode": "ssml_narration", "items_total": len(items),
                "items_synthesized": 0, "items_skipped": len(items),
                "items_cache_hit": len(items), "raw_chars": 0, "ssml_chars": 0,
                "ssml_ratio": 0, "api_calls": 0, "cache_hits": len(items),
                "chunks": 0, "align_ok": 0, "align_fallback": 0,
                "wall_time_sec": round(time.time() - t_start, 1),
            }
            return _fp_results, _fp_stats
    # ─────────────────────────────────────────────────────────────────────────

    synthesizer = load_azure_synthesizer()
    ssml_inner  = Path(ssml_inner_path).read_text(encoding="utf-8")
    voice_cast  = json.loads(Path(voice_cast_path).read_text(encoding="utf-8"))

    locale     = items[0]["locale"] if items else "en"
    azure_lang = LOCALE_TO_AZURE_LANG.get(locale.lower(), "en-US")

    # --- Get narrator voice params from VoiceCast (Layer 1) ---
    narrator = None
    for ch in voice_cast.get("characters", []):
        if ch.get("character_id") == "narrator":
            narrator = ch
            break
    if narrator is None:
        raise ValueError("VoiceCast.json has no narrator entry")

    vc_locale = narrator.get(locale) or narrator.get("en") or {}
    voice     = vc_locale.get("azure_voice", "en-US-GuyNeural")
    style     = vc_locale.get("azure_style")
    style_deg = vc_locale.get("azure_style_degree", 1.5)
    rate      = vc_locale.get("azure_rate", "0%")
    pitch     = vc_locale.get("azure_pitch")

    # Change F: detect multi-block mode.
    # Primary signal: ssml_multi_block flag written by ssml_preprocess.py Change E.
    # Fallback: content-sniff ssml_inner in case VoiceCast predates the flag.
    multi_block = vc_locale.get("ssml_multi_block", False)
    if not multi_block:
        multi_block = "<mstts:express-as" in ssml_inner

    # --- Parse ssml_inner.xml into sentence fragments ---
    sentence_frags = _parse_ssml_inner_fragments(ssml_inner)

    total = min(len(items), len(sentence_frags))
    if len(sentence_frags) != len(items):
        log.warning("ssml_inner sentence count (%d) != manifest item count (%d); "
                    "using min()", len(sentence_frags), len(items))

    # --- Derive cache directory: assets/meta/tts_cache/ ---
    # out_dir is assets/{locale}/audio/vo — go up 3 levels to assets/
    cache_dir = out_dir.parent.parent.parent / "meta" / "tts_cache"

    # Change G: multi-block passthrough — single full-episode Azure call.
    # Branches before the chunk_alignment loop.  Count gate, synthesis, alignment,
    # and WAV slicing are all handled inside the helper.
    if multi_block:
        return _run_ssml_narration_passthrough(
            ssml_inner     = ssml_inner,
            voice          = voice,
            azure_lang     = azure_lang,
            sentence_frags = sentence_frags,
            items          = items,
            out_dir        = out_dir,
            cache_dir      = cache_dir,
            synthesizer    = synthesizer,
            locale         = locale,
            style          = style,
        )

    # --- Instrumentation counters (az.txt §Phase 1) ---
    stat_raw_chars      = sum(len(sentence_frags[i]["text"]) for i in range(total))
    stat_ssml_chars     = 0
    stat_api_calls      = 0
    stat_cache_hits     = 0
    stat_align_success  = 0
    stat_align_fallback = 0

    print(f"\n[SSML-NARRATION] chunk_alignment mode: {total} sentences  voice={voice}")
    print(f"[SSML-NARRATION] style={style}  rate={rate}  locale={locale}")
    print(f"[SSML-NARRATION] Raw spoken chars: {stat_raw_chars:,}")

    # --- Group sentences into synthesis chunks ---
    chunks = group_sentences_into_chunks(
        sentence_frags = sentence_frags[:total],
        items          = items[:total],
        voice          = voice,
        style          = style,
        style_degree   = style_deg,
        rate           = rate,
        pitch          = pitch,
        azure_lang     = azure_lang,
        locale         = locale,
    )
    n_chunks = len(chunks)
    print(f"[SSML-NARRATION] {n_chunks} chunks planned "
          f"(target_dur=35s, max_dur=45s, max_chars=1000)")

    results: list[dict] = []

    for chunk_idx, chunk in enumerate(chunks):
        sents       = chunk["sentences"]
        n           = len(sents)
        chunk_label = f"chunk {chunk_idx+1}/{n_chunks} ({n} sent)"
        print(f"\n[chunk] {chunk_label}  "
              f"est={chunk['estimated_dur_sec']:.1f}s  chars={chunk['total_chars']}")

        chunk_ssml = _build_chunk_ssml(chunk)
        ssml_chars = len(chunk_ssml)
        stat_ssml_chars += ssml_chars

        # --- TTS cache lookup ---
        cache_key = _tts_cache_key(chunk_ssml, voice, locale)
        wav_bytes = _tts_cache_get(cache_dir, cache_key)

        word_events: list[dict] | None = None

        if wav_bytes is not None:
            stat_cache_hits += 1
            print(f"  [cache HIT]  {cache_key[:16]}…  ({len(wav_bytes):,} bytes)")
        else:
            # --- Azure TTS synthesis ---
            try:
                if _is_hd_omni_voice(voice):
                    # HDOmni: capture word-boundary events for alignment (Phase 2, Step 8)
                    wav_bytes, word_events = synthesise_with_word_boundaries(
                        synthesizer, chunk_ssml
                    )
                    if word_events:
                        print(f"  [Azure TTS]  {len(wav_bytes):,} bytes  "
                              f"ssml_chars={ssml_chars}  word_events={len(word_events)}")
                    else:
                        print(f"  [Azure TTS]  {len(wav_bytes):,} bytes  ssml_chars={ssml_chars}"
                              f"  (no word events — REST fallback used)")
                else:
                    wav_bytes = synthesise(synthesizer, chunk_ssml)
                    print(f"  [Azure TTS]  {len(wav_bytes):,} bytes  ssml_chars={ssml_chars}")

                stat_api_calls += 1
                _tts_cache_put(cache_dir, cache_key, wav_bytes, {
                    "voice": voice, "locale": locale, "chunk_idx": chunk_idx,
                    "n_sentences": n, "ssml_chars": ssml_chars,
                })
            except Exception as exc:
                log.error(f"[chunk] synthesis failed for {chunk_label}: {exc}")
                print(f"  [FALLBACK] synthesis error → per-sentence: {exc}")
                stat_align_fallback += n
                results.extend(_per_item_legacy_chunk(
                    synthesizer, chunk, voice, style, style_deg,
                    rate, pitch, azure_lang, out_dir,
                ))
                continue

        # --- Sentence alignment ---
        pcm       = _get_pcm_from_wav_bytes(wav_bytes)
        chunk_dur = _pcm_duration_sec(pcm)
        aligned: list[dict] | None = None

        if n == 1:
            # Single sentence — entire chunk is the sentence; no alignment needed
            aligned = [{"item_id": sents[0]["item_id"],
                        "start_sec": 0.0, "end_sec": round(chunk_dur, 4)}]
            stat_align_success += 1
        else:
            # Path 0: word-boundary events (HDOmni only — Phase 2, Step 8)
            if word_events is not None:
                aligned = _align_by_word_boundaries(word_events, chunk)
                if aligned is not None:
                    aligned = _clamp_alignment(aligned, chunk_dur)
                    if not _validate_alignment(aligned, chunk_dur):
                        aligned = None
                    else:
                        stat_align_success += n
                        print(f"  [align] word-boundary ({n} sentences)")

            # Path 1: CTC forced alignment (DragonHD / HDFlash — az.txt §Step 4C Path 2)
            if aligned is None and not _is_hd_omni_voice(voice):
                aligned = _align_by_ctc(wav_bytes, chunk, locale)
                if aligned is not None:
                    aligned = _clamp_alignment(aligned, chunk_dur)
                    if not _validate_alignment(aligned, chunk_dur):
                        aligned = None
                    else:
                        stat_align_success += n
                        print(f"  [align] CTC forced alignment ({n} sentences)")

            # Path 2: silence-gap alignment
            if aligned is None:
                aligned = _align_by_silence(wav_bytes, chunk)
                if aligned is not None:
                    aligned = _clamp_alignment(aligned, chunk_dur)
                    if not _validate_alignment(aligned, chunk_dur):
                        log.warning(f"[chunk] silence alignment invalid — trying proportional")
                        aligned = None
                    else:
                        stat_align_success += n
                        print(f"  [align] silence-gap  ({n} sentences)")

            # Path 3: proportional fallback
            if aligned is None:
                aligned = _align_proportional(wav_bytes, chunk)
                aligned = _clamp_alignment(aligned, chunk_dur)
                if not _validate_alignment(aligned, chunk_dur):
                    # All paths failed — fall back to per_item_legacy for this chunk
                    log.error(f"[chunk] all alignment paths invalid for {chunk_label} "
                              f"— falling back to per-sentence")
                    print(f"  [FALLBACK] alignment failed → per-sentence for {chunk_label}")
                    stat_align_fallback += n
                    results.extend(_per_item_legacy_chunk(
                        synthesizer, chunk, voice, style, style_deg,
                        rate, pitch, azure_lang, out_dir,
                    ))
                    continue
                else:
                    stat_align_fallback += 1        # count proportional use as partial fallback
                    stat_align_success  += n - 1
                    print(f"  [align] proportional fallback (CTC+silence unavailable)")

        # --- Slice per-sentence WAVs ---
        slice_results = _write_sentence_wavs_from_chunk(
            chunk_wav_bytes = wav_bytes,
            offsets         = aligned,
            out_dir         = out_dir,
            voice           = voice,
            style           = style,
        )

        # Augment results with speaker/voice metadata from the vo item
        item_meta = {s["item_id"]: s["vo"] for s in sents}
        for r in slice_results:
            vo = item_meta.get(r["item_id"], {})
            r.update({
                "speaker":      vo.get("speaker", "narrator"),
                "voice":        voice,
                "azure_lang":   azure_lang,
                "style":        style or "",
                "style_degree": style_deg,
                "rate":         rate,
            })

        results.extend(slice_results)

    # --- Instrumentation summary (az.txt §Phase 1) ---
    wall_time      = time.time() - t_start
    overhead_ratio = stat_ssml_chars / max(stat_raw_chars, 1)
    ok             = sum(1 for r in results if r.get("status") == "success")

    print(f"\n[SSML-NARRATION] ═══ Phase 1 Instrumentation ═══")
    print(f"  raw spoken chars     : {stat_raw_chars:,}")
    print(f"  SSML chars sent      : {stat_ssml_chars:,}  (ratio {overhead_ratio:.2f}×)")
    print(f"  Azure API calls      : {stat_api_calls}")
    print(f"  cache hits           : {stat_cache_hits} / {n_chunks} chunks")
    print(f"  chunk count          : {n_chunks}")
    print(f"  align success        : {stat_align_success}  fallback: {stat_align_fallback}")
    print(f"  wall-clock TTS time  : {wall_time:.1f}s")
    print(f"  items OK             : {ok}/{total}")

    stats = {
        "mode":              "ssml_narration",
        "items_total":       total,
        "items_synthesized": ok,
        "items_skipped":     total - ok,
        "items_cache_hit":   stat_cache_hits,
        "raw_chars":         stat_raw_chars,
        "ssml_chars":        stat_ssml_chars,
        "ssml_ratio":        round(overhead_ratio, 2),
        "api_calls":         stat_api_calls,
        "cache_hits":        stat_cache_hits,
        "chunks":            n_chunks,
        "align_ok":          stat_align_success,
        "align_fallback":    stat_align_fallback,
        "wall_time_sec":     round(wall_time, 1),
    }
    return results, stats


def run(
    items: list[dict],
    out_dir: Path,
    asset_id: str | None = None,
    ssml_narration: bool = False,
    ssml_inner_path: str | None = None,
    voice_cast_path: str | None = None,
    manifest: dict | None = None,
    assets_dir: "Path | None" = None,
    keep_chunks: bool = False,
    use_batch_rest: bool = False,
    force: bool = False,
) -> tuple[list[dict], list[dict], dict]:
    """Synthesise all items with Azure TTS.

    Returns (results, chunks_meta, stats):
      results     : per-sentence result dicts (same schema as before)
      chunks_meta : chunk-level metadata (non-empty only for chunk_alignment mode)
      stats       : instrumentation dict written to tts_audit_log.json by main()

    Mode selection (Phase 2, Step 6 — az.txt §Step 6):
    ┌─ asset_id set                      → per_item_legacy
    ├─ ssml_narration                    → chunk_alignment
    ├─ all voices in whitelist + ≤ max   → batch_bookmark
    ├─ any HD voice not in whitelist     → chunk_alignment
    └─ other unsupported voices          → per_item_legacy
    """
    mode = _select_synthesis_mode(items, ssml_narration, asset_id)
    print(f"\n[MODE] {mode}")

    # ── per_item_legacy: single-item re-synthesis or last resort ──────────────
    if mode == _MODE_PER_ITEM_LEGACY:
        synthesizer = load_azure_synthesizer()
        if ssml_narration:
            if not ssml_inner_path or not voice_cast_path:
                raise ValueError("ssml_narration requires --ssml-inner and --voice-cast")
            results, stats = run_ssml_narration(
                ssml_inner_path, voice_cast_path, manifest or {}, items, out_dir
            )
            return results, [], stats
        results = _synthesise_per_item(synthesizer, items, out_dir, force=force)
        n_syn  = sum(1 for r in results if r.get("status") == "success")
        n_skip = sum(1 for r in results if r.get("status") == "skipped")
        stats  = {
            "mode":              "per_item_legacy",
            "items_total":       len(items),
            "items_synthesized": n_syn,
            "items_skipped":     n_skip,
            "items_cache_hit":   0,
            "raw_chars":         sum(len(it.get("text", "")) for it in items
                                     if it["item_id"] not in
                                     {r["item_id"] for r in results
                                      if r.get("status") == "skipped"}),
            "ssml_chars":        0,
            "api_calls":         n_syn,
            "cache_hits":        0,
            "chunks":            0,
        }
        return results, [], stats

    # ── chunk_alignment: ssml_narration OR HD voices in non-batch episodes ────
    if mode == _MODE_CHUNK_ALIGNMENT:
        if ssml_narration:
            if not ssml_inner_path or not voice_cast_path:
                raise ValueError("ssml_narration requires --ssml-inner and --voice-cast")
            synthesizer = load_azure_synthesizer()
            results, stats = run_ssml_narration(
                ssml_inner_path, voice_cast_path, manifest or {}, items, out_dir
            )
            return results, [], stats
        else:
            # Non-ssml_narration with HD voices (continuous / illustrated narration)
            synthesizer = load_azure_synthesizer()
            return run_chunk_alignment_from_items(
                synthesizer, items, out_dir,
                assets_dir  = assets_dir,
                keep_chunks = keep_chunks,
            )

    # ── batch_bookmark: standard Neural voices, full episode, SDK path ────────
    locale      = items[0]["locale"] if items else "en"
    synthesizer = load_azure_synthesizer()

    # Phase 3, Step 9: REST batch path (standard Neural voices, opt-in via --use-batch-rest)
    # Submits full episode SSML, polls until done, downloads WAV + sentence-boundary JSON,
    # then splits WAV by sentence offsets (no bookmark events needed).
    if use_batch_rest:
        try:
            print(f"\n[BATCH-REST] Attempting REST batch for {len(items)} items…")
            rest_ssml = build_episode_ssml(items, locale)
            synthesis_id = str(uuid.uuid4())
            job_url = batch_synthesis_rest_submit(rest_ssml, synthesis_id)
            job     = batch_synthesis_rest_poll(job_url)
            wav_bytes_rest, sent_bounds = batch_synthesis_rest_download(job)

            if sent_bounds and len(sent_bounds) == len(items):
                # Map sentence boundaries → per-sentence WAVs using _write_sentence_wavs_from_chunk
                # Build a synthetic chunk covering all items
                rest_chunk = {
                    "sentences": [
                        {"item_id": it["item_id"], "text": it["text"],
                         "pause_ms": it.get("break_ms", 0), "vo": it}
                        for it in items
                    ],
                    "voice":        items[0]["voice"],
                    "style":        items[0].get("style", ""),
                    "style_degree": items[0].get("style_degree", 1.5),
                    "rate":         items[0].get("rate", "0%"),
                    "pitch":        items[0].get("pitch"),
                    "azure_lang":   items[0].get("azure_lang", "en-US"),
                }
                offsets = [
                    {"item_id": items[i]["item_id"],
                     "start_sec": b["start_sec"], "end_sec": b["end_sec"]}
                    for i, b in enumerate(sent_bounds)
                ]
                slice_results = _write_sentence_wavs_from_chunk(
                    wav_bytes_rest, offsets, out_dir,
                    items[0]["voice"], items[0].get("style"),
                )
                item_meta = {it["item_id"]: it for it in items}
                for r in slice_results:
                    it = item_meta.get(r["item_id"], {})
                    r.update({
                        "speaker": it.get("speaker", ""),
                        "voice": it.get("voice", ""),
                        "azure_lang": it.get("azure_lang", ""),
                        "style": it.get("style", ""),
                        "style_degree": it.get("style_degree", 1.5),
                        "rate": it.get("rate", "0%"),
                    })
                print(f"[BATCH-REST] Complete: {len(slice_results)} items")
                rest_ssml_chars = len(rest_ssml)
                rest_raw_chars  = sum(len(it.get("text", "")) for it in items)
                rest_stats = {
                    "mode":              "batch_rest",
                    "items_total":       len(items),
                    "items_synthesized": len(slice_results),
                    "items_skipped":     0,
                    "items_cache_hit":   0,
                    "raw_chars":         rest_raw_chars,
                    "ssml_chars":        rest_ssml_chars,
                    "ssml_ratio":        round(rest_ssml_chars / max(rest_raw_chars, 1), 3),
                    "api_calls":         1,
                    "cache_hits":        0,
                    "chunks":            1,
                }
                return slice_results, [], rest_stats
            else:
                log.warning(f"[BATCH-REST] Sentence boundary count mismatch "
                            f"({len(sent_bounds)} vs {len(items)} items) — falling back to SDK batch")
        except Exception as exc:
            log.warning(f"[BATCH-REST] Failed ({exc}), falling back to SDK batch")

    # SDK batch: one call for entire episode using bookmark events
    try:
        log.info("Batch synthesis: building episode SSML…")
        print(f"\n[BATCH] Building episode SSML for {len(items)} items…")
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

        print(f"[BATCH] Complete: {len(items)} items → {out_dir}")

        results: list[dict] = []
        for vo in items:
            out_path = out_dir / f"{vo['item_id']}.wav"
            if out_path.exists():
                size = out_path.stat().st_size
                with open(str(out_path), "rb") as fh:
                    raw = fh.read()
                data_tag_pos = raw.index(b"data")
                pcm_len      = max(0, len(raw) - (data_tag_pos + 8))
                duration_s   = pcm_len / (AZURE_SAMPLE_RATE * 2)
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
                    "error":      "WAV not found after batch split",
                })
        batch_ssml    = build_episode_ssml(items, locale)
        batch_raw     = sum(len(it.get("text", "")) for it in items)
        batch_ssml_ch = len(batch_ssml)
        n_ok   = sum(1 for r in results if r.get("status") == "success")
        sdk_stats = {
            "mode":              "batch_bookmark",
            "items_total":       len(items),
            "items_synthesized": n_ok,
            "items_skipped":     0,
            "items_cache_hit":   0,
            "raw_chars":         batch_raw,
            "ssml_chars":        batch_ssml_ch,
            "ssml_ratio":        round(batch_ssml_ch / max(batch_raw, 1), 3),
            "api_calls":         1,
            "cache_hits":        0,
            "chunks":            1,
        }
        return results, [], sdk_stats

    except Exception as exc:
        log.warning(f"Batch synthesis failed ({exc}); falling back to per-item")
        print(f"\n[BATCH] Failed ({exc}); falling back to per-item…")
        fallback_results = _synthesise_per_item(synthesizer, items, out_dir, force=force)
        fb_syn  = sum(1 for r in fallback_results if r.get("status") == "success")
        fb_skip = sum(1 for r in fallback_results if r.get("status") == "skipped")
        fb_raw  = sum(len(it.get("text", "")) for it in items
                      if it["item_id"] not in
                      {r["item_id"] for r in fallback_results if r.get("status") == "skipped"})
        fallback_stats = {
            "mode":              "batch_bookmark_fallback",
            "items_total":       len(items),
            "items_synthesized": fb_syn,
            "items_skipped":     fb_skip,
            "items_cache_hit":   0,
            "raw_chars":         fb_raw,
            "ssml_chars":        0,
            "ssml_ratio":        0,
            "api_calls":         fb_syn,
            "cache_hits":        0,
            "chunks":            0,
        }
        return fallback_results, [], fallback_stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_manifest_from_script(
    script_path: Path,
    voicecast_path: Path,
    locale: str,
) -> tuple[dict, Path]:
    """Build a minimal locale VOPlan from Script.json + VoiceCast.json.

    Walks Script.json scenes[].actions for type="dialogue" entries and assigns
    item_ids in the form "vo-{scene_id}-{NNN:03d}" (per-scene counter, 001-based).
    TTS parameters are populated from VoiceCast.json character[locale] entries.

    Returns:
        (manifest_dict, manifest_path) — manifest_path is
        {ep_dir}/VOPlan.{locale}.json
    """
    script = json.loads(script_path.read_text(encoding="utf-8"))

    # Load VoiceCast.json
    voice_cast_map: dict[str, dict] = {}
    if voicecast_path.exists():
        try:
            vc_data = json.loads(voicecast_path.read_text(encoding="utf-8"))
            for ch in vc_data.get("characters", []):
                cid = ch.get("character_id", "")
                if cid:
                    voice_cast_map[cid] = ch
            print(f"  [SCRIPT-MODE] Loaded VoiceCast.json ({len(voice_cast_map)} characters)")
        except Exception as exc:
            print(f"  [WARN] Could not load VoiceCast.json: {exc}")
    else:
        print(f"  [WARN] VoiceCast.json not found: {voicecast_path}")

    # Build vo_items from Script.json scenes[].actions where type="dialogue"
    vo_items = []
    for scene in script.get("scenes", []):
        scene_id = scene.get("scene_id", "sc01")
        counter = 1
        for action in scene.get("actions", []):
            if action.get("type") != "dialogue":
                continue
            item_id = f"vo-{scene_id}-{counter:03d}"
            counter += 1

            speaker_id = (
                action.get("speaker_id")
                or action.get("character_id")
                or "narrator"
            )
            text = (
                action.get("text")
                or action.get("line")
                or action.get("content")
                or ""
            )

            # Build tts_prompt from VoiceCast.json character[locale]
            vc_char        = voice_cast_map.get(speaker_id, {})
            vc_locale_entry = vc_char.get(locale, {})
            tts_prompt: dict = {"locale": locale}
            for field in ("azure_voice", "azure_style", "azure_style_degree",
                          "azure_rate", "azure_pitch", "azure_break_ms"):
                val = vc_locale_entry.get(field)
                if val is not None:
                    tts_prompt[field] = val

            vo_items.append({
                "item_id":    item_id,
                "speaker_id": speaker_id,
                "text":       text,
                "license_type": "commercial_reusable",
                "tts_prompt": tts_prompt,
            })

    print(f"  [SCRIPT-MODE] Built {len(vo_items)} vo_items from Script.json")

    # Infer project_id / episode_id from script_path
    # Expected path: …/projects/{project_id}/episodes/{episode_id}/Script.json
    parts = script_path.resolve().parts
    project_id = ""
    episode_id = ""
    try:
        ep_idx = next(i for i, p in enumerate(parts) if p == "episodes")
        project_id = parts[ep_idx - 1]
        episode_id = parts[ep_idx + 1]
    except (StopIteration, IndexError):
        script_id  = script.get("script_id", "unknown-ep01")
        parts_sid  = script_id.split("-")
        project_id = parts_sid[0] if parts_sid else "unknown"
        episode_id = parts_sid[-1] if len(parts_sid) > 1 else "ep01"

    manifest_id = f"{project_id}-{episode_id}-{locale}-manifest"
    manifest = {
        "schema_id":            "VOPlan",
        "schema_version":       "1.0.0",
        "manifest_id":          manifest_id,
        "project_id":           project_id,
        "episode_id":           episode_id,
        "locale":               locale,
        "locale_scope":         "merged",
        "shared_ref":           "AssetManifest.shared.json",
        "shotlist_ref":         "ShotList.json",
        "character_packs":      [],
        "backgrounds":          [],
        "background_overrides": [],
        "vo_items":             vo_items,
    }

    # Write the draft manifest to ep_dir
    ep_dir = script_path.resolve().parent
    manifest_path = ep_dir / f"VOPlan.{locale}.json"
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(manifest_path)
    print(f"  [SCRIPT-MODE] Wrote {manifest_path.name}  ({len(vo_items)} vo_items)")

    return manifest, manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Azure Neural TTS voice-over generator (CPU-only, no GPU required).\n\n"
            "Reads vo_items from a locale VOPlan and synthesises WAV files.\n"
            "Maps tts_prompt fields to Azure SSML: voice, express-as style, prosody rate.\n\n"
            "Credentials via environment variables:\n"
            "  export AZURE_SPEECH_KEY='your-key'\n"
            "  export AZURE_SPEECH_REGION='eastus'\n\n"
            "Examples:\n"
            "  python gen_tts_cloud.py --manifest VOPlan.en.json\n"
            "  python gen_tts_cloud.py --manifest VOPlan.zh-Hans.json\n"
            "  python gen_tts_cloud.py --manifest VOPlan.en.json "
            "--asset-id vo-s01e02-sc01-001\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest", type=str, required=False, default=None,
        help="Path to a locale VOPlan JSON (locale_scope='locale', 'monolithic', or 'merged'). "
             "Mutually exclusive with --script.",
    )
    parser.add_argument(
        "--script", type=str, default=None, dest="script",
        help="Path to Script.json. When set, builds a manifest from Script.json instead of "
             "reading --manifest. Requires --voicecast and --locale.",
    )
    parser.add_argument(
        "--voicecast", type=str, default=None, dest="voicecast",
        help="Path to VoiceCast.json (required when --script is set).",
    )
    parser.add_argument(
        "--locale", type=str, default=None, dest="locale_override",
        help="Target locale (e.g. 'en', 'zh-Hans'). Required when --script is set.",
    )
    parser.add_argument(
        "--stage", type=str, default=None, dest="stage",
        help="Pipeline stage identifier (e.g. '3.5'). Used for logging only.",
    )
    parser.add_argument(
        "--asset-id", type=str, default=None, dest="asset_id",
        help="Process only the item with this item_id (default: all vo_items).",
    )
    parser.add_argument(
        "--ssml-narration", action="store_true", default=False,
        dest="ssml_narration",
        help="ssml_narration mode: wrapper-rebuild + inner passthrough.",
    )
    parser.add_argument(
        "--ssml-inner", type=str, default=None, dest="ssml_inner",
        help="Path to ssml_inner.xml (required when --ssml-narration).",
    )
    parser.add_argument(
        "--voice-cast", type=str, default=None, dest="voice_cast",
        help="Path to VoiceCast.json (required when --ssml-narration).",
    )
    parser.add_argument(
        "--keep-chunks", action="store_true", default=False, dest="keep_chunks",
        help=(
            "Phase 3, Step 10: retain full chunk WAVs in assets/{locale}/audio/vo_chunks/ "
            "after slicing into sentence WAVs.  Enables future deferred-slicing render path."
        ),
    )
    parser.add_argument(
        "--use-batch-rest", action="store_true", default=False, dest="use_batch_rest",
        help=(
            "Phase 3, Step 9: submit standard-voice episodes via Azure Batch Synthesis "
            "REST API instead of SDK batch.  Experimental — sentence-boundary parsing "
            "from REST response not yet fully implemented."
        ),
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="Overwrite existing WAV files instead of skipping them. "
             "Used by polish_locale_vo re-synthesis loop.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Input mode: --script or --manifest ────────────────────────────────────
    if args.script:
        # Stage 3.5 / --script mode: build manifest from Script.json
        if not args.locale_override:
            raise SystemExit("--script requires --locale (e.g. --locale en)")
        if not args.voicecast:
            raise SystemExit("--script requires --voicecast (e.g. --voicecast projects/slug/VoiceCast.json)")
        script_path   = Path(args.script).resolve()
        voicecast_path = Path(args.voicecast).resolve()
        if not script_path.exists():
            raise SystemExit(f"[ERROR] Script.json not found: {script_path}")
        stage_label = f"Stage {args.stage}" if args.stage else "Script mode"
        print(f"\n[{stage_label}] Building manifest from Script.json…")
        manifest, manifest_path_obj = build_manifest_from_script(
            script_path, voicecast_path, args.locale_override
        )
        manifest_path_str = str(manifest_path_obj)
    elif args.manifest:
        manifest_path_str = args.manifest
        with open(manifest_path_str, encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        raise SystemExit("[ERROR] Either --manifest or --script is required.")
    # ── end input mode ────────────────────────────────────────────────────────

    # Validate ssml_narration args
    if args.ssml_narration:
        if not args.ssml_inner or not args.voice_cast:
            raise SystemExit("--ssml-narration requires --ssml-inner and --voice-cast")

    # Guard: reject shared manifests early
    if manifest.get("locale_scope") == "shared":
        raise SystemExit(
            "[ERROR] Passed a shared manifest to gen_tts_cloud.py — "
            "use a locale manifest (locale_scope='locale') instead."
        )

    locale     = locale_from_manifest(manifest, manifest_path_str)
    assets_dir = assets_dir_from_manifest(manifest)
    out_dir    = assets_dir / locale / "audio" / "vo"
    meta_dir   = assets_dir / "meta"

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ASSETS] {assets_dir}")
    print(f"[OUTPUT] {out_dir}")

    items = load_items_from_manifest(manifest, manifest_path_str, args.asset_id)

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
        # Extract item_id: "vo-sc01-001.wav" → "vo-sc01-001"
        #                   "vo-sc01-001.source.wav" → "vo-sc01-001"
        stem = wav.stem
        if stem.endswith(".source"):
            stem = stem[:-len(".source")]
        if stem not in all_manifest_ids:
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

    t_start = time.time()
    results, chunks_meta, stats = run(
        items,
        out_dir,
        asset_id        = args.asset_id,
        ssml_narration  = args.ssml_narration,
        ssml_inner_path = args.ssml_inner,
        voice_cast_path = args.voice_cast,
        manifest        = manifest,
        assets_dir      = assets_dir,
        keep_chunks     = args.keep_chunks,
        use_batch_rest  = args.use_batch_rest,
        force           = args.force,
    )
    stats["wall_time_sec"] = round(time.time() - t_start, 1)
    _append_tts_audit_log(meta_dir, locale, stats)

    # ── Two-file WAV model: create source.wav + apply trim overrides (INVARIANT A/B) ──
    # After synthesis, for each successfully written {item_id}.wav:
    #   1. Copy/rename to {item_id}.source.wav  (raw TTS output, never trimmed)
    #   2. Call apply_vo_trims_for_item()        (writes {item_id}.wav)
    # On first run, no trim overrides exist → .wav = copy of source.wav.
    try:
        import sys as _sys
        import os as _os
        _sys.path.insert(0, _os.path.dirname(__file__))
        from vo_utils import apply_vo_trims_for_item as _apply_trim
        from vo_utils import invalidate_vo_state as _invalidate
        from vo_utils import get_primary_locale as _get_primary_locale

        ep_dir = assets_dir.parent  # assets_dir = ep_dir/assets
        _n_converted = 0
        for r in results:
            if r.get("status") != "success":
                continue
            item_id  = r.get("item_id", "")
            wav_path = out_dir / f"{item_id}.wav"
            src_path = out_dir / f"{item_id}.source.wav"
            if not wav_path.exists():
                continue
            # Step 1: copy .wav → .source.wav (raw output)
            import shutil as _shutil
            _shutil.copy2(str(wav_path), str(src_path))
            # Step 2: apply_vo_trims_for_item writes .wav (first run: .wav = source copy)
            try:
                _apply_trim(item_id, ep_dir, locale)
            except FileNotFoundError:
                pass  # source.wav just created — should not happen
            _n_converted += 1

        if _n_converted > 0:
            print(f"\n[SOURCE.WAV] Created {_n_converted} .source.wav files.")
            # Invalidate sentinel since TTS just ran
            _primary = _get_primary_locale(ep_dir)
            _invalidate(ep_dir, _primary)
            print(f"[SOURCE.WAV] VO state invalidated (sentinel deleted).")

    except ImportError:
        print("[WARN] vo_utils not available — skipping source.wav creation.",
              file=__import__("sys").stderr)
    except Exception as _exc:
        print(f"[WARN] source.wav post-processing error: {_exc}",
              file=__import__("sys").stderr)
    # ── end two-file WAV model ────────────────────────────────────────────────

    # ── ssml_narration: patch manifest vo_items text with source-locale text ──
    # The zh-Hans manifest may have been copied from en with English text.
    # After TTS, patch vo_items.text with the Chinese text from ssml_inner.xml
    # so downstream consumers (gen_render_plan → write_srt) use the correct text.
    if args.ssml_narration and args.ssml_inner:
        ssml_inner_text = Path(args.ssml_inner).read_text(encoding="utf-8")
        frags = _parse_ssml_inner_fragments(ssml_inner_text)
        vo_items = manifest.get("vo_items", [])
        patched = 0
        for idx, frag in enumerate(frags):
            if idx < len(vo_items):
                vo_items[idx]["text"] = frag["text"]
                vo_items[idx]["pause_after_ms"] = frag["pause_ms"]
                patched += 1
        manifest_path = Path(manifest_path_str)
        # Re-read vo_approval from disk just before writing so a concurrent or
        # sequential VO approval is never wiped by this SSML-narration re-run.
        if manifest_path.is_file():
            _on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
            if "vo_approval" in _on_disk:
                manifest["vo_approval"] = _on_disk["vo_approval"]
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n[SSML-NARRATION] Patched {patched} manifest vo_items with "
              f"source-locale text → {manifest_path.name}")

    # Phase 2, Step 7: centralised result writer — also writes chunks.json when present
    results_path = write_tts_results(
        results,
        meta_dir,
        chunks_meta = chunks_meta if chunks_meta else None,
    )

    ok_count    = sum(1 for r in results if r.get("status") == "success")
    total_bytes = sum(r.get("size_bytes", 0) for r in results)
    failed      = [r for r in results if r.get("status") != "success"]

    print(f"\n{'='*60}")
    print(f"{ok_count}/{len(results)} completed | {total_bytes:,} bytes total")
    if chunks_meta:
        print(f"Chunks   : {len(chunks_meta)} chunks written to {meta_dir}/{SCRIPT_NAME}_chunks.json")
    if failed:
        print(f"FAILED ({len(failed)}):")
        for r in failed:
            print(f"  {r['item_id']}: {r.get('error', '?')}")
    print(f"Results  : {results_path}")


if __name__ == "__main__":
    main()
