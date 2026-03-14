#!/usr/bin/env python3
"""
vo_utils.py — Shared VO utility functions for the pipeline.

Used by both test_server.py (server endpoints) and gen_tts_cloud.py (CLI).

Core functions:
  wav_duration(path)                                      → float (seconds)
  load_vo_trim_overrides(ep_dir, locale)                  → dict
  save_vo_trim_overrides(ep_dir, locale, data)
  apply_vo_trims_for_item(item_id, ep_dir, locale, override=None) → float
  get_primary_locale(ep_dir)                              → str
  invalidate_vo_state(ep_dir, primary_locale)             → None
  compute_sentinel_hashes(ep_dir, locale)                 → dict
  write_sentinel(ep_dir, locale, hashes)                  → None   [writes vo_preview_approved.{locale}.json]
  verify_sentinel(ep_dir, locale)                         → bool   [checks vo_preview_approved.{locale}.json]
  write_vo_preview_approved(ep_dir, locale, stage, items, hashes) → None
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import time
from pathlib import Path
from typing import Optional


# ── Audio helpers ─────────────────────────────────────────────────────────────

def wav_duration(path) -> float:
    """Return duration in seconds of a WAV file (stdlib only, no soundfile).

    Properly walks RIFF chunks so it works with Azure TTS WAVs that include
    extra chunks (fact, smpl, LIST, etc.) before the data chunk.
    """
    path = Path(path)
    with open(path, "rb") as f:
        raw = f.read()

    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError(f"Not a valid WAV file: {path}")

    # Walk RIFF chunks to find fmt and data
    fmt_data: bytes | None = None
    data_size: int | None  = None

    pos = 12
    while pos + 8 <= len(raw):
        chunk_id   = raw[pos:pos+4]
        chunk_size, = struct.unpack_from("<I", raw, pos+4)
        if chunk_id == b"fmt ":
            fmt_data = raw[pos+8 : pos+8+chunk_size]
        elif chunk_id == b"data":
            data_size = chunk_size
            break  # data chunk found; no need to read further
        pos += 8 + chunk_size + (chunk_size % 2)  # WAV chunks are word-aligned

    if fmt_data is None:
        raise ValueError(f"Missing fmt chunk in {path}")
    if data_size is None:
        raise ValueError(f"Missing data chunk in {path}")

    num_channels,    = struct.unpack_from("<H", fmt_data, 2)
    sample_rate,     = struct.unpack_from("<I", fmt_data, 4)
    bits_per_sample, = struct.unpack_from("<H", fmt_data, 14)

    if sample_rate == 0 or num_channels == 0 or bits_per_sample == 0:
        raise ValueError(f"Invalid WAV header in {path}")

    bytes_per_sample = bits_per_sample // 8
    block_align      = num_channels * bytes_per_sample
    if block_align == 0:
        raise ValueError(f"block_align is 0 in {path}")

    n_samples = data_size // block_align
    return n_samples / sample_rate


def _wav_duration_robust(path) -> float:
    """Try wav_duration first; fall back to soundfile if available."""
    try:
        return wav_duration(path)
    except Exception:
        try:
            import soundfile as sf
            return sf.info(str(path)).duration
        except Exception:
            raise


def _copy_wav(src: Path, dst: Path) -> None:
    """Atomic copy: write to .tmp then rename."""
    tmp = dst.with_suffix(".tmp")
    shutil.copy2(str(src), str(tmp))
    tmp.rename(dst)


def _slice_wav(src: Path, dst: Path, t_start: float, t_end: float) -> None:
    """Slice a WAV from t_start to t_end seconds and write to dst.

    Only handles standard PCM (16-bit / 24-bit / 8-bit mono or stereo).
    If slice fails for any reason, falls back to full copy.
    """
    try:
        with open(src, "rb") as f:
            raw = f.read()

        if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
            raise ValueError("Not a RIFF WAV")

        # Find fmt and data chunks
        pos = 12
        fmt_data = None
        audio_data_offset = None
        audio_data_size   = None

        while pos + 8 <= len(raw):
            chunk_id   = raw[pos:pos+4]
            chunk_size, = struct.unpack_from("<I", raw, pos+4)
            if chunk_id == b"fmt ":
                fmt_data = raw[pos+8 : pos+8+chunk_size]
            elif chunk_id == b"data":
                audio_data_offset = pos + 8
                audio_data_size   = chunk_size
                break
            pos += 8 + chunk_size + (chunk_size % 2)  # WAV chunks are word-aligned

        if fmt_data is None or audio_data_offset is None:
            raise ValueError("Missing fmt or data chunk")

        num_channels,   = struct.unpack_from("<H", fmt_data, 2)
        sample_rate,    = struct.unpack_from("<I", fmt_data, 4)
        bits_per_sample, = struct.unpack_from("<H", fmt_data, 14)

        block_align = num_channels * (bits_per_sample // 8)
        if block_align == 0 or sample_rate == 0:
            raise ValueError("Invalid fmt parameters")

        start_byte = int(t_start * sample_rate) * block_align
        end_byte   = int(t_end   * sample_rate) * block_align
        start_byte = max(0, min(start_byte, audio_data_size))
        end_byte   = max(start_byte, min(end_byte, audio_data_size))

        sliced_pcm = raw[audio_data_offset + start_byte : audio_data_offset + end_byte]

        # Re-build minimal RIFF header
        data_size    = len(sliced_pcm)
        byte_rate    = sample_rate * block_align
        riff_size    = 36 + data_size
        wav_header   = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", riff_size, b"WAVE",
            b"fmt ", 16,
            1, num_channels, sample_rate, byte_rate,
            block_align, bits_per_sample,
            b"data", data_size,
        )
        tmp = dst.with_suffix(".tmp")
        tmp.write_bytes(wav_header + sliced_pcm)
        tmp.rename(dst)

    except Exception as exc:
        # Fallback: full copy (better than failing)
        import warnings
        warnings.warn(f"vo_utils._slice_wav fallback ({exc}) — copying full WAV")
        _copy_wav(src, dst)


# ── Trim overrides sidecar ────────────────────────────────────────────────────

def _trim_overrides_path(ep_dir, locale: str) -> Path:
    return Path(ep_dir) / f"vo_trim_overrides.{locale}.json"


def load_vo_trim_overrides(ep_dir, locale: str) -> dict:
    """Load vo_trim_overrides sidecar; returns {} if not found."""
    p = _trim_overrides_path(ep_dir, locale)
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_vo_trim_overrides(ep_dir, locale: str, overrides: dict) -> None:
    """Atomically save vo_trim_overrides sidecar."""
    p = _trim_overrides_path(ep_dir, locale)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=2)
    tmp.rename(p)


# ── Primary-locale helper ─────────────────────────────────────────────────────

def get_primary_locale(ep_dir) -> str:
    """Read primary locale from meta.json (first locale in locales_str).

    Falls back to 'en' if meta.json is not found or locales field is missing.
    """
    meta_path = Path(ep_dir) / "meta.json"
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            locales_str = meta.get("locales", "en")
            return locales_str.split(",")[0].strip() or "en"
        except Exception:
            pass
    return "en"


# ── INVARIANT B — apply_vo_trims_for_item ────────────────────────────────────

def apply_vo_trims_for_item(
    item_id: str,
    ep_dir,
    locale: str,
    override: Optional[dict] = None,
) -> float:
    """Apply trim override for item_id and write {item_id}.wav.

    This is the ONLY function that writes {item_id}.wav (INVARIANT B).

    Args:
        item_id:  VO item ID (e.g. "vo-sc01-001")
        ep_dir:   Absolute path to the episode directory
        locale:   Locale string (e.g. "en")
        override: If provided, use this override dict instead of reading sidecar.
                  Format: {"trim_start_sec": float, "trim_end_sec": float, ...}
                  Pass None to read from vo_trim_overrides.json sidecar.

    Returns:
        Duration of the written {item_id}.wav in seconds.

    Clamp logic (INVARIANT B):
        source_dur = wav_duration(source.wav)
        t_start = min(override.trim_start_sec, source_dur)
        t_end   = min(override.trim_end_sec,   source_dur)
        if t_start >= t_end or no override:
            copy source.wav → .wav
            if invalid override: delete override from sidecar, log warning
        else:
            slice source.wav[t_start:t_end] → .wav
            persist clamped values back to sidecar
    """
    ep_dir = Path(ep_dir)
    vo_dir = ep_dir / "assets" / locale / "audio" / "vo"
    source_path = vo_dir / f"{item_id}.source.wav"
    wav_path    = vo_dir / f"{item_id}.wav"

    if not source_path.exists():
        # No source WAV — nothing to do (item not yet synthesized)
        raise FileNotFoundError(f"source.wav not found: {source_path}")

    source_dur = _wav_duration_robust(source_path)

    # Determine override to apply
    if override is None:
        overrides = load_vo_trim_overrides(ep_dir, locale)
        override  = overrides.get(item_id)

    if override is None:
        # No override → copy source to .wav
        _copy_wav(source_path, wav_path)
        return source_dur

    # Apply clamp
    t_start = min(float(override.get("trim_start_sec", 0.0)), source_dur)
    t_end   = min(float(override.get("trim_end_sec",   source_dur)), source_dur)

    if t_start >= t_end:
        # Invalid override (e.g. source is now shorter than trim bounds)
        import warnings
        warnings.warn(
            f"[vo_utils] Invalid trim for {item_id}: "
            f"t_start={t_start:.3f} >= t_end={t_end:.3f} "
            f"(source_dur={source_dur:.3f}) — resetting to full copy"
        )
        # Remove invalid override from sidecar
        overrides = load_vo_trim_overrides(ep_dir, locale)
        if item_id in overrides:
            del overrides[item_id]
            save_vo_trim_overrides(ep_dir, locale, overrides)
        _copy_wav(source_path, wav_path)
        return source_dur

    # Persist clamped values back to sidecar (if values changed)
    overrides = load_vo_trim_overrides(ep_dir, locale)
    entry = overrides.get(item_id, {})
    clamped_changed = (
        abs(entry.get("trim_start_sec", -999) - t_start) > 1e-6
        or abs(entry.get("trim_end_sec",   -999) - t_end)   > 1e-6
    )
    if clamped_changed:
        overrides[item_id] = {
            **entry,
            "trim_start_sec": round(t_start, 6),
            "trim_end_sec":   round(t_end,   6),
            "source_duration_sec": round(source_dur, 6),
            "clamped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        save_vo_trim_overrides(ep_dir, locale, overrides)

    # Slice
    _slice_wav(source_path, wav_path, t_start, t_end)
    return t_end - t_start


# ── INVARIANT H — invalidate_vo_state ────────────────────────────────────────

def invalidate_vo_state(ep_dir, primary_locale: str) -> None:
    """Delete sentinel + durations export; mark non-primary draft manifests stale.

    Called by every VO write endpoint after any modification.
    (INVARIANT H)

    Does NOT delete RenderPlan or media manifests (those are handled by
    vo_merge specifically, since item_id changes require deeper invalidation).

    ── Asset Preservation rule (MUST NOT VIOLATE) ──────────────────────────────
    This function and every caller MUST NOT delete or overwrite any files under:
        assets/music/   — music loop WAVs, MusicPlan.json, MusicApprovalSnapshot.json
        assets/sfx/     — SFX files and SfxPlan.json
        assets/media/   — media selections and selections.json
    These directories are owned exclusively by the Music, SFX, and Media tabs.
    Music/SFX/Media curation is expensive user effort that survives VO re-runs.
    Any timing misalignment after VO re-approval is the user's responsibility to
    review and fix — the pipeline never auto-deletes or invalidates these assets.
    ────────────────────────────────────────────────────────────────────────────
    """
    ep_dir = Path(ep_dir)

    # Delete legacy sentinel (tts_review_complete.json)
    sentinel = ep_dir / "tts_review_complete.json"
    if sentinel.exists():
        sentinel.unlink()

    # Delete new vo_preview_approved.{locale}.json for ALL locales
    for approved in ep_dir.glob("vo_preview_approved.*.json"):
        approved.unlink()

    # Delete durations export
    durations = ep_dir / f"{primary_locale}_vo_durations.json"
    if durations.exists():
        durations.unlink()

    # Mark non-primary draft manifests stale
    for draft in ep_dir.glob("AssetManifest_draft.*.json"):
        # Skip shared and primary locale
        stem = draft.stem  # "AssetManifest_draft.zh-Hans"
        # Extract locale from filename: AssetManifest_draft.{locale}.json
        parts = draft.name[len("AssetManifest_draft."):][:-len(".json")]
        locale_part = parts  # e.g. "zh-Hans" or "shared" or "en"
        if locale_part == "shared" or locale_part == primary_locale:
            continue
        stale_path = draft.with_name(draft.name + ".stale")
        if stale_path.exists():
            stale_path.unlink()
        draft.rename(stale_path)


# ── INVARIANT I — sentinel hash computation and verification ──────────────────

def _file_sha256(path: Path) -> str:
    """SHA256 of file bytes. Returns SHA256(b'') if file does not exist."""
    import hashlib
    if not path.exists():
        return hashlib.sha256(b"").hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_sentinel_hashes(ep_dir, locale: str) -> dict:
    """Compute the four sentinel hashes for the current state of ep_dir/locale.

    Returns a dict with keys:
        manifest_hash, wav_set_hash, trim_overrides_hash, merge_log_hash
    """
    import hashlib

    ep_dir = Path(ep_dir)

    # manifest_hash
    manifest_path = ep_dir / f"AssetManifest_merged.{locale}.json"
    manifest_hash = _file_sha256(manifest_path)

    # wav_set_hash: SHA256 of concatenated "fname:SHA256(content)" for each .wav
    vo_dir = ep_dir / "assets" / locale / "audio" / "vo"
    wav_entries: list[str] = []
    if vo_dir.exists():
        for wav in sorted(vo_dir.glob("*.wav")):
            content_hash = _file_sha256(wav)
            wav_entries.append(f"{wav.name}:{content_hash}")
    wav_set_hash = hashlib.sha256(
        "\n".join(wav_entries).encode()
    ).hexdigest()

    # trim_overrides_hash
    overrides_path = _trim_overrides_path(ep_dir, locale)
    trim_overrides_hash = _file_sha256(overrides_path)

    # merge_log_hash
    merge_log_path = ep_dir / "vo_merge_log.json"
    merge_log_hash = _file_sha256(merge_log_path)

    return {
        "manifest_hash":       manifest_hash,
        "wav_set_hash":        wav_set_hash,
        "trim_overrides_hash": trim_overrides_hash,
        "merge_log_hash":      merge_log_hash,
    }


def write_sentinel(ep_dir, locale: str, hashes: dict) -> None:
    """Write vo_preview_approved.{locale}.json sentinel with the given hashes.

    Also writes the legacy tts_review_complete.json for backward compatibility
    with any existing scripts that check for it.
    """
    ep_dir   = Path(ep_dir)

    # New sentinel: vo_preview_approved.{locale}.json
    sentinel = ep_dir / f"vo_preview_approved.{locale}.json"
    doc = {
        "approved_at":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "locale":              locale,
        "stage":               "legacy",
        "items":               [],   # populated by write_vo_preview_approved; empty from write_sentinel
        "wav_set_hash":        hashes.get("wav_set_hash", ""),
        "trim_overrides_hash": hashes.get("trim_overrides_hash", ""),
        "manifest_hash":       hashes.get("manifest_hash", ""),
        "merge_log_hash":      hashes.get("merge_log_hash", ""),
    }
    tmp = sentinel.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(sentinel)

    # Legacy sentinel for backward compat
    legacy = ep_dir / "tts_review_complete.json"
    legacy_doc = {
        "locale":              locale,
        "completed_at":        doc["approved_at"],
        "manifest_hash":       hashes.get("manifest_hash", ""),
        "wav_set_hash":        hashes.get("wav_set_hash", ""),
        "trim_overrides_hash": hashes.get("trim_overrides_hash", ""),
        "merge_log_hash":      hashes.get("merge_log_hash", ""),
    }
    ltmp = legacy.with_suffix(".tmp")
    ltmp.write_text(json.dumps(legacy_doc, indent=2), encoding="utf-8")
    ltmp.rename(legacy)


def verify_sentinel(ep_dir, locale: str) -> bool:
    """Return True iff a valid VO approval sentinel exists for the given locale.

    Checks vo_preview_approved.{locale}.json first (new format).
    Falls back to tts_review_complete.json for backward compatibility.

    For the new-format sentinel (stage != 'legacy'), existence alone is
    sufficient (items and timing are already stored).
    For hash-based sentinels, all four hashes must match (INVARIANT I).
    """
    ep_dir = Path(ep_dir)

    # 1. Check new sentinel: vo_preview_approved.{locale}.json
    new_sentinel = ep_dir / f"vo_preview_approved.{locale}.json"
    if new_sentinel.exists():
        try:
            stored = json.loads(new_sentinel.read_text(encoding="utf-8"))
        except Exception:
            return False
        if stored.get("locale") != locale:
            return False
        stage = stored.get("stage", "legacy")
        if stage not in ("legacy", ""):
            # Non-legacy stages (3.5, 8.5): existence with correct locale is sufficient
            return True
        # Legacy stage: verify hashes
        current = compute_sentinel_hashes(ep_dir, locale)
        return (
            stored.get("manifest_hash")       == current["manifest_hash"]
            and stored.get("wav_set_hash")        == current["wav_set_hash"]
            and stored.get("trim_overrides_hash") == current["trim_overrides_hash"]
            and stored.get("merge_log_hash")      == current["merge_log_hash"]
        )

    # 2. Backward compat: check legacy tts_review_complete.json
    legacy = ep_dir / "tts_review_complete.json"
    if not legacy.exists():
        return False
    try:
        stored = json.loads(legacy.read_text(encoding="utf-8"))
    except Exception:
        return False
    if stored.get("locale") != locale:
        return False
    current = compute_sentinel_hashes(ep_dir, locale)
    return (
        stored.get("manifest_hash")       == current["manifest_hash"]
        and stored.get("wav_set_hash")        == current["wav_set_hash"]
        and stored.get("trim_overrides_hash") == current["trim_overrides_hash"]
        and stored.get("merge_log_hash")      == current["merge_log_hash"]
    )


def write_vo_preview_approved(
    ep_dir,
    locale: str,
    stage: str,
    items: list,
    hashes: dict,
) -> None:
    """Write vo_preview_approved.{locale}.json with full timing data.

    Called by /api/vo_approve to record the user's VO approval.

    Schema:
      {
        "approved_at":         ISO timestamp,
        "locale":              "en",
        "stage":               "3.5" | "8.5" | "legacy",
        "items": [
          {
            "item_id":      "vo-sc01-001",
            "speaker_id":   "narrator",
            "text":         "...",
            "duration_sec": 2.34,
            "start_sec":    0.0,
            "end_sec":      2.34
          }, ...
        ],
        "wav_set_hash":        SHA256 of WAV set,
        "trim_overrides_hash": SHA256 of trim overrides,
      }

    Args:
        ep_dir:  Episode directory (absolute path).
        locale:  Locale string (e.g. "en", "zh-Hans").
        stage:   Pipeline stage ("3.5", "8.5", or "legacy").
        items:   List of vo_item dicts with timing fields.
        hashes:  Dict with wav_set_hash, trim_overrides_hash (from compute_sentinel_hashes).
    """
    ep_dir = Path(ep_dir)
    path   = ep_dir / f"vo_preview_approved.{locale}.json"
    doc = {
        "approved_at":         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "locale":              locale,
        "stage":               stage,
        "items":               items,
        "wav_set_hash":        hashes.get("wav_set_hash", ""),
        "trim_overrides_hash": hashes.get("trim_overrides_hash", ""),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(path)
