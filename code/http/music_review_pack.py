#!/usr/bin/env python3
# =============================================================================
# music_review_pack.py — Generate music review artifacts for human inspection
# =============================================================================
#
# Runs AFTER post_tts_analysis.py (Stage 9[4/8]) when both music WAVs and VO
# timing exist.  Produces two review artifacts:
#
#   1. timeline.txt      — human-readable shot timeline with music, VO, ducking
#   2. preview_audio.wav — VO + music mix (no SFX, no video) with ducking
#
# Timeline data is returned in-memory by build_timeline() and is no longer
# persisted to timeline.json — test_server.py calls build_timeline() directly.
#
# Usage:
#   python music_review_pack.py \
#       --manifest projects/slug/ep/VOPlan.en.json
#
#   python music_review_pack.py \
#       --manifest projects/slug/ep/VOPlan.en.json \
#       --output   projects/slug/ep/assets/music/MusicReviewPack/
#
# Requirements:
#   pip install soundfile numpy
# =============================================================================

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PIPE_DIR = Path(__file__).resolve().parent.parent.parent

SAMPLE_RATE = 48000
CHANNELS = 2
BASE_MUSIC_DB = -6.0
DEFAULT_DUCK_DB = -6.0
DEFAULT_FADE_SEC = 0.8
DEFAULT_PAUSE_SEC = 0.3    # inter-line pause (same as post_tts_analysis.py)


# ── I/O helpers ──────────────────────────────────────────────────────────────

def load_manifest(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_shotlist(manifest, manifest_path):
    """Load ShotList referenced by shotlist_ref. Returns [] if not found."""
    shotlist_ref = manifest.get("shotlist_ref", "")
    if not shotlist_ref:
        return []
    candidates = [
        manifest_path.parent / shotlist_ref,
        manifest_path.parent.parent / shotlist_ref,
        manifest_path.parent / "ShotList.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                with open(candidate, encoding="utf-8") as f:
                    return json.load(f).get("shots", [])
            except Exception as exc:
                print(f"[WARN] Could not parse ShotList {candidate}: {exc}")
                return []
    print(f"[WARN] ShotList not found: {shotlist_ref}")
    return []


def load_loop_candidates(episode_dir):
    """Load music_loop_candidates.json if present. Returns {} on failure."""
    path = episode_dir / "assets" / "music" / "music_loop_candidates.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ── Duck interval helpers ────────────────────────────────────────────────────

def merge_overlapping(intervals):
    """Merge a list of (t0, t1) tuples into non-overlapping sorted ranges."""
    if not intervals:
        return []
    sorted_ivs = sorted(intervals, key=lambda x: x[0])
    merged = [list(sorted_ivs[0])]
    for t0, t1 in sorted_ivs[1:]:
        if t0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], t1)
        else:
            merged.append([t0, t1])
    return [[round(a, 3), round(b, 3)] for a, b in merged]


def compute_duck_intervals(vo_items_for_shot, fade_sec, shot_start_offset_sec=0.0):
    """Compute music duck intervals from VO line positions within a shot.

    vo_items_for_shot: VO items whose start_sec/end_sec are episode-relative.
    shot_start_offset_sec: cumulative episode offset at the start of this shot.
      Subtracting it converts episode-relative times to shot-relative times.
    """
    raw = []
    for item in vo_items_for_shot:
        start = item.get("start_sec")
        end = item.get("end_sec")
        if start is None or end is None:
            continue
        # Convert episode-relative → shot-relative
        rel_start = start - shot_start_offset_sec
        rel_end   = end   - shot_start_offset_sec
        t0 = max(0.0, rel_start - fade_sec)
        t1 = rel_end + fade_sec
        raw.append((t0, t1))
    return merge_overlapping(raw)


# ── Audio helpers ────────────────────────────────────────────────────────────

def read_wav_mono_or_stereo(path, target_sr=SAMPLE_RATE):
    """Read a WAV file and return a (samples, 2) stereo numpy array at target_sr."""
    data, sr = sf.read(str(path), dtype="float64")
    if data.ndim == 1:
        data = np.column_stack([data, data])
    elif data.shape[1] > 2:
        data = data[:, :2]
    if sr != target_sr:
        # Simple linear resampling (good enough for preview)
        n_out = int(len(data) * target_sr / sr)
        indices = np.linspace(0, len(data) - 1, n_out)
        idx_floor = np.floor(indices).astype(int)
        idx_ceil = np.minimum(idx_floor + 1, len(data) - 1)
        frac = (indices - idx_floor)[:, np.newaxis]
        data = data[idx_floor] * (1 - frac) + data[idx_ceil] * frac
    return data


def build_shot_envelope(n_samples, base_db, fade_sec, fade_in=True, fade_out=True):
    """
    Build a shot-boundary amplitude envelope for music.

    Holds at base_db level throughout the shot.
    Fades in at the start and/or out at the end only at shot boundaries
    (skipped when the same track continues from the previous/into the next shot).

    Ducking under VO is handled separately by the duck_intervals envelope
    in render_preview_audio — not here.

    fade_in/fade_out: False when same track continues across shot boundary.
    """
    base_amp = 10 ** (base_db / 20.0)

    envelope = np.full(n_samples, base_amp, dtype=np.float64)
    fade_samples = min(int(fade_sec * SAMPLE_RATE), n_samples // 2)

    if fade_in and fade_samples > 0:
        envelope[:fade_samples] = np.linspace(0.0, base_amp, fade_samples)

    if fade_out and fade_samples > 0:
        envelope[n_samples - fade_samples:] = np.linspace(base_amp, 0.0, fade_samples)

    return envelope


# ── Timeline builders ────────────────────────────────────────────────────────

def build_timeline(shots, manifest, vo_shot_map, music_index, loop_info):
    """
    Build the timeline data structure used for both text and JSON output.

    Returns a list of shot dicts with enriched timeline info, plus total_duration_sec.
    """
    project_id = manifest.get("project_id", "")
    episode_id = manifest.get("episode_id", "")

    timeline_shots = []
    cumulative_sec = 0.0

    for shot in shots:
        shot_id = shot["shot_id"]
        duration = float(shot.get("duration_sec", 0))

        # Find music item for this shot
        music_item = music_index.get(shot_id)
        duck_db = 0.0          # Option A: full volume by default, no per-VO pumping
        fade_sec = DEFAULT_FADE_SEC
        start_sec = 0.0
        music_mood = ""
        music_item_id = ""
        duck_intervals = []

        if music_item:
            music_item_id = music_item.get("item_id", "")
            music_mood = music_item.get("music_mood", "")
            # Option A: ignore manifest duck_db (legacy -12/-15 values from gen_music_clip.py)
            # User overrides duck_db via MusicPlan.json shot_overrides; default is 0 (full vol)
            duck_db = 0.0
            fade_sec = float(music_item.get("fade_sec", DEFAULT_FADE_SEC))
            duck_intervals = music_item.get("duck_intervals", [])
            start_sec = float(music_item.get("start_sec", 0.0))

        # Compute duck intervals from VO timing.
        vo_items_for_shot = vo_shot_map.get(shot_id, [])
        if not duck_intervals and vo_items_for_shot:
            # Use manifest vo_items timing (episode-relative) minus cumulative offset
            duck_intervals = compute_duck_intervals(
                vo_items_for_shot, fade_sec, shot_start_offset_sec=cumulative_sec)

        # VO lines
        vo_lines = []
        for vo in vo_items_for_shot:
            vo_lines.append({
                "item_id": vo.get("item_id", ""),
                "start_sec": vo.get("start_sec"),
                "end_sec": vo.get("end_sec"),
                "speaker_id": vo.get("speaker_id", ""),
                "text": vo.get("text", ""),
                "pause_after_ms": int(vo.get("pause_after_ms") or 0),
            })

        # Loop candidate info
        loop_candidate = None
        if music_item_id and loop_info:
            # loop_info keyed by stem or item_id
            loop_candidate = loop_info.get(music_item_id)

        entry = {
            "shot_id": shot_id,
            "offset_sec": round(cumulative_sec, 3),
            "duration_sec": duration,
            "music_item_id": music_item_id,
            "music_mood": music_mood,
            "start_sec": start_sec,
            "duck_db": duck_db,
            "fade_sec": fade_sec,
            "duck_intervals": duck_intervals,
            "vo_lines": vo_lines,
        }
        if loop_candidate:
            entry["loop_candidate"] = loop_candidate

        timeline_shots.append(entry)
        cumulative_sec += duration

    # When scene_heads bake an offset into VO start_sec/end_sec, the last VO
    # item can end beyond the ShotList shot total (e.g. 70.3s > 55.6s).
    # total_duration_sec must cover the full VO extent so callers (JS ruler,
    # preview audio trim) know the true content length.
    # Use ALL manifest vo_items for last_vo_end — not vo_shot_map — because some
    # vo_items may fall outside real shot windows (e.g. after the last shot ends)
    # and would be absent from vo_shot_map after the time-overlap mapping fix.
    last_vo_end = max(
        ((vo.get("end_sec") or 0.0) + (vo.get("pause_after_ms") or 0) / 1000
         for vo in manifest.get("vo_items", [])),
        default=0.0,
    )
    total_dur = max(round(cumulative_sec, 3), last_vo_end)
    return timeline_shots, total_dur


def write_timeline_txt(timeline_shots, total_dur, episode_id, out_path):
    """Write human-readable timeline.txt."""
    lines = []
    lines.append(f"Music Review Timeline — {episode_id}")
    lines.append(f"Total duration: {total_dur:.2f}s")
    lines.append("")

    for entry in timeline_shots:
        shot_id = entry["shot_id"]
        dur = entry["duration_sec"]
        offset = entry["offset_sec"]
        lines.append(f"{'═' * 3} Shot {shot_id} ({dur:.2f}s @ {offset:.2f}s) {'═' * 3}")

        if entry["music_item_id"]:
            lines.append(
                f"  Music: {entry['music_item_id']} "
                f"— \"{entry['music_mood']}\" "
                f"(duck_db={entry['duck_db']}, fade={entry['fade_sec']}s)"
            )
        else:
            lines.append("  Music: (none)")

        if entry.get("loop_candidate"):
            lc = entry["loop_candidate"]
            lines.append(f"  Loop candidate: {json.dumps(lc)}")

        if entry["vo_lines"]:
            lines.append("  VO lines:")
            for vo in entry["vo_lines"]:
                s = vo.get("start_sec")
                e = vo.get("end_sec")
                s_str = f"{s:.3f}" if s is not None else "?.???"
                e_str = f"{e:.3f}" if e is not None else "?.???"
                lines.append(
                    f"    {s_str}s – {e_str}s  "
                    f"[{vo['speaker_id']}] \"{vo['text']}\""
                )
        else:
            lines.append("  VO lines: (none)")

        if entry["duck_intervals"]:
            ivs = " ".join(f"[{a:.3f}, {b:.3f}]" for a, b in entry["duck_intervals"])
            lines.append(f"  Duck intervals: {ivs}")

        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] {out_path}")


# ── Preview audio renderer ───────────────────────────────────────────────────

def render_preview_audio(timeline_shots, total_dur, manifest, manifest_path, out_path,
                         music_plan=None):
    """
    Render a VO + music mix WAV with ducking applied.
    No SFX, no video — purely for music review listening.
    """
    project_id = manifest.get("project_id", "")
    episode_id = manifest.get("episode_id", "")
    locale = manifest.get("locale", "en")

    VO_OUTRO_SEC = 5.0   # fade-out window after last VO line
    n_samples = int(total_dur * SAMPLE_RATE) + SAMPLE_RATE  # +1s safety

    # ── Grow buffer to accommodate episode-absolute VO timing (scene tails
    # and scene heads are already baked into manifest.vo_items[].start_sec by
    # the approve flow, so the last VO item may end beyond total_dur which is
    # shots-only). ────────────────────────────────────────────────────────────
    _scene_tails_cfg = manifest.get("scene_tails", {})
    _scene_heads_cfg = manifest.get("scene_heads", {})  # values in seconds
    _DEFAULT_SCENE_TAIL_MS = 2000
    _prev_scene_id = None
    _shift_acc     = 0.0
    for _vo_item in manifest.get("vo_items", []):
        _iid   = _vo_item.get("item_id", "")
        _parts = _iid.split("-")
        _scn   = _parts[1] if len(_parts) >= 3 else ""
        if _prev_scene_id is not None and _scn != _prev_scene_id:
            _tail_ms = int(_scene_tails_cfg.get(
                _scn, _scene_tails_cfg.get(_prev_scene_id, _DEFAULT_SCENE_TAIL_MS)))
            _shift_acc += _tail_ms / 1000.0
        if _scn != _prev_scene_id:
            # scene_heads are in seconds (not ms)
            _shift_acc += float(_scene_heads_cfg.get(_scn, 0.0))
        _prev_scene_id = _scn
    n_samples += int(_shift_acc * SAMPLE_RATE)
    max_pause_sec = max(
        ((vo.get("pause_after_ms") or 0) / 1000
         for vo in manifest.get("vo_items", [])),
        default=0.0,
    )
    n_samples += int(max_pause_sec * SAMPLE_RATE)
    n_samples += int(VO_OUTRO_SEC * SAMPLE_RATE)   # headroom for post-VO fade-out

    buf = np.zeros((n_samples, CHANNELS), dtype=np.float64)

    episode_dir = PIPE_DIR / "projects" / project_id / "episodes" / episode_id
    vo_dir = episode_dir / "assets" / locale / "audio" / "vo"
    music_dir = episode_dir / "assets" / "music"

    missing_vo = 0
    missing_music = 0
    any_timing_missing = False

    # ── Place VO — direct from manifest.vo_items (no shot decomposition) ──────
    # vo_items carry episode-absolute start_sec/end_sec baked by the approve flow.
    # No shot context needed — place each item directly at its start_sec position.
    for _vo in manifest.get("vo_items", []):
        _start = _vo.get("start_sec")
        if _start is None:
            any_timing_missing = True
            continue
        _iid = _vo.get("item_id", "")
        _wp = vo_dir / f"{_iid}.wav"
        if not _wp.exists():
            missing_vo += 1
            continue
        _vd = read_wav_mono_or_stereo(_wp)
        _vdb = float(_vo.get("volume_db", 0.0) or 0.0)
        if _vdb != 0.0:
            _gain = 10 ** (_vdb / 20.0)
            _vd = _vd * _gain
        _s = int(_start * SAMPLE_RATE)
        _e = min(_s + len(_vd), n_samples)
        _l = _e - _s
        if _l > 0:
            buf[_s:_e] += _vd[:_l]

    # ── Place music — segment-direct from MusicPlan ─────────────────────────
    # Iterate shot_overrides at episode-absolute coords — no shot decomposition.
    # seg_start_sec IS the buffer write position; no shot offset cancellation needed.
    # music_plan is passed by the caller (already merged: disk + payload overrides).
    # Fall back to disk only when called from the CLI (music_plan=None).
    if music_plan is None:
        _ep_dir = manifest_path.parent
        _music_plan: dict = {}
        _mp_path = _ep_dir / "MusicPlan.json"
        if _mp_path.exists():
            try:
                _music_plan = json.loads(_mp_path.read_text(encoding="utf-8"))
            except Exception as _mp_err:
                print(f"  [WARN] MusicPlan load failed: {_mp_err}")
    else:
        _music_plan = music_plan

    _track_volumes = _music_plan.get("track_volumes", {})
    _clip_volumes  = _music_plan.get("clip_volumes",  {})

    for _seg in _music_plan.get("shot_overrides", []):
        _seg_start = float(_seg.get("start_sec", 0.0))
        _seg_end   = float(_seg.get("end_sec",   0.0))
        _asset_id  = _seg.get("music_asset_id", "")
        _clip_id   = _seg.get("music_clip_id", "")
        _duck_db   = float(_seg.get("duck_db",  DEFAULT_DUCK_DB))
        _fade_sec  = float(_seg.get("fade_sec", DEFAULT_FADE_SEC))

        if not _asset_id or _seg_end <= _seg_start:
            continue

        # Resolve music WAV path (same search order as before)
        _music_path = music_dir / f"{_asset_id}.loop.wav"
        if not _music_path.exists():
            _music_path = music_dir / f"{_asset_id}.wav"
        if not _music_path.exists():
            _resources_music = PIPE_DIR / "projects" / project_id / "resources" / "music"
            _seg_found = False
            if _resources_music.is_dir():
                for _ext in (".mp3", ".wav", ".flac", ".ogg"):
                    _cand = _resources_music / f"{_asset_id}{_ext}"
                    if _cand.exists():
                        _music_path = _cand
                        _seg_found = True
                        break
            if not _seg_found:
                missing_music += 1
                continue

        _seg_samples = max(0, int((_seg_end - _seg_start) * SAMPLE_RATE))
        if _seg_samples == 0:
            continue

        _music_data = read_wav_mono_or_stereo(_music_path)
        if len(_music_data) >= _seg_samples:
            _music_data = _music_data[:_seg_samples]
        else:
            _reps = (_seg_samples // len(_music_data)) + 1
            _music_data = np.tile(_music_data, (_reps, 1))[:_seg_samples]

        # base_db from track/clip volumes
        _stem    = _clip_id.split(":")[0] if _clip_id else ""
        _base_db = BASE_MUSIC_DB
        _base_db += _track_volumes.get(_stem, 0)
        _base_db += _clip_volumes.get(_asset_id, _clip_volumes.get(_clip_id, 0))

        # Segment-boundary envelope (always fade in/out at edges)
        _envelope = build_shot_envelope(
            _seg_samples, _base_db, _fade_sec, fade_in=True, fade_out=True
        )

        # Duck intervals from manifest VO items overlapping this segment
        _seg_vo = [
            vo for vo in manifest.get("vo_items", [])
            if vo.get("start_sec") is not None and vo.get("end_sec") is not None
            and float(vo["end_sec"]) > _seg_start
            and float(vo["start_sec"]) < _seg_end
        ]
        if _seg_vo and _duck_db != 0:
            _base_amp = 10 ** (_base_db / 20.0)
            _duck_amp = _base_amp * (10 ** (_duck_db / 20.0))
            _t_arr    = np.arange(_seg_samples) / SAMPLE_RATE
            # compute_duck_intervals subtracts shot_start_offset_sec → segment-relative t0/t1
            _duck_ivs = compute_duck_intervals(
                _seg_vo, _fade_sec, shot_start_offset_sec=_seg_start
            )
            _duck_env = np.full(_seg_samples, _base_amp, dtype=np.float64)
            for _t0, _t1 in _duck_ivs:
                if _t1 <= 0:
                    continue
                _t0 = max(_t0, 0.0)
                _fade = min(_fade_sec, max(0.0, (_t1 - _t0) / 2.0))
                _t_fi_end   = _t0 + _fade
                _t_fo_start = _t1 - _fade
                _in_iv = (_t_arr >= _t0) & (_t_arr <= _t1)
                _fi_mask = _in_iv & (_t_arr <= _t_fi_end) & (_fade > 0)
                _duck_env[_fi_mask] = _base_amp + (_duck_amp - _base_amp) * \
                    (_t_arr[_fi_mask] - _t0) / _fade
                _duck_env[_in_iv & (_t_arr > _t_fi_end) & (_t_arr < _t_fo_start)] = _duck_amp
                _fo_mask = _in_iv & (_t_arr >= _t_fo_start) & (_fade > 0)
                _duck_env[_fo_mask] = _duck_amp + (_base_amp - _duck_amp) * \
                    (_t_arr[_fo_mask] - _t_fo_start) / _fade
            _envelope = _envelope * (_duck_env / _base_amp)

        _music_data = _music_data * _envelope[:, np.newaxis]
        _s = int(_seg_start * SAMPLE_RATE)
        _e = min(_s + len(_music_data), n_samples)
        _length = _e - _s
        if _length > 0:
            buf[_s:_e] += _music_data[:_length]

    # Trim to last VO end + 5s fade-out, so music review doesn't play 45s of
    # music-only after narration finishes (accumulated music-loop padding).
    # BUG FIX: use ALL manifest.vo_items (same source as build_timeline's total_dur)
    # instead of timeline_shots[].vo_lines. vo_lines only contains VO items that were
    # mapped to a shot window; VO items whose start_sec falls outside all shot windows
    # (e.g. the last item when scene_heads push VO beyond ShotList cumulative) are
    # absent from vo_lines, causing last_vo_end_sec to be 3s too short (35s → 32s).
    last_vo_end_sec = max(
        ((vo.get("end_sec") or 0.0) + (vo.get("pause_after_ms") or 0) / 1000
         for vo in manifest.get("vo_items", [])),
        default=0.0,
    )
    # Trim to last_vo_end_sec which already includes pause_after_ms.
    # pause_after_ms IS the musical tail — adding VO_OUTRO_SEC on top double-counts
    # it and produces 85.348s instead of the correct 80.348s.
    # Apply the fade within the pause_after_ms window instead.
    trim_sec = last_vo_end_sec if last_vo_end_sec > 0 else total_dur
    trim_samples = int(trim_sec * SAMPLE_RATE)
    buf = buf[:trim_samples]
    # Apply a short fade-out so music doesn't cut abruptly (within pause_after_ms window)
    fade_len = min(int(VO_OUTRO_SEC * SAMPLE_RATE), trim_samples)
    fade_env = np.linspace(1.0, 0.0, fade_len)
    buf[-fade_len:] *= fade_env[:, None] if CHANNELS > 1 else fade_env
    total_dur = trim_sec
    print(f"  [TRIM] last VO end={last_vo_end_sec:.3f}s (incl. pause_after_ms) → preview={trim_sec:.3f}s")

    # Clip to prevent distortion
    peak = np.max(np.abs(buf))
    if peak > 1.0:
        buf = buf / peak
        print(f"  [WARN] Peak clipped from {peak:.2f} to 1.0")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), buf, SAMPLE_RATE, subtype="PCM_16")

    if any_timing_missing:
        print(f"  [INFO] VO timing not in manifest — computed on the fly from WAV durations")
    if missing_vo:
        print(f"  [WARN] {missing_vo} VO WAV(s) not found — silence in their place")
    if missing_music:
        print(f"  [WARN] {missing_music} music WAV(s) not found — silence in their place")
    print(f"  [OK] {out_path}  ({total_dur:.2f}s, {SAMPLE_RATE}Hz stereo)")


# ── Override application ─────────────────────────────────────────────────────

def apply_music_plan_overrides(timeline_shots, segment_list, source_name, vo_shot_map):
    """Apply a list of MusicPlan music_segment dicts to timeline_shots in-place.

    Each segment is matched to a timeline entry by episode-absolute start_sec/end_sec
    overlap (v2 schema).  No shot_id or item_id lookups are performed.

    Parameters
    ----------
    timeline_shots : list[dict]  — in-memory timeline built by build_timeline()
    segment_list   : list[dict]  — shot_overrides array from MusicPlan.json or payload
                                   Each entry has episode-absolute start_sec / end_sec.
    source_name    : str         — label used in log messages ("MusicPlan.json", etc.)
    vo_shot_map    : dict        — {shot_id: [vo_item, …]} for duck-interval recompute
    """
    applied = 0
    for seg in segment_list:
        seg_start = float(seg.get("start_sec", 0.0))
        seg_end   = float(seg.get("end_sec", 0.0))

        # Find the timeline entry whose episode window overlaps this segment.
        for entry in timeline_shots:
            entry_start = float(entry.get("offset_sec", 0.0))
            entry_end   = entry_start + float(entry.get("duration_sec", 0.0))
            if seg_start >= entry_end or seg_end <= entry_start:
                continue

            shot_id = entry.get("shot_id", "")
            _shot_offset = float(entry.get("offset_sec", 0.0))

            if "duck_db" in seg:
                entry["duck_db"] = float(seg["duck_db"])
            if "fade_sec" in seg:
                entry["fade_sec"] = float(seg["fade_sec"])
            # start_sec and end_sec in the segment are episode-absolute;
            # convert to within-shot coordinates for the renderer.
            if "start_sec" in seg:
                entry["start_sec"] = max(0.0, seg_start - _shot_offset)
            if "end_sec" in seg:
                # Store as music_end_sec (separate from shot duration_sec to avoid
                # corruption).  Clamp to the shot's actual end so music placed for
                # shot N never bleeds past the shot boundary into shot N+1 territory
                # without the correct duck intervals for N+1's VO items.
                entry["music_end_sec"] = max(0.0, min(seg_end, entry_end) - _shot_offset)
            if "music_asset_id" in seg:
                # Store override separately — keep original music_item_id intact.
                # Renderer uses _override for audio selection.
                entry["music_item_id_override"] = seg["music_asset_id"]
            # Recompute duck intervals only if fade_sec changed (which affects
            # the margin around each VO line). Use manifest vo_items timing.
            if "fade_sec" in seg:
                _new_fade = float(entry.get("fade_sec", DEFAULT_FADE_SEC))
                _vo_items = vo_shot_map.get(shot_id, [])
                if _vo_items:
                    entry["duck_intervals"] = compute_duck_intervals(
                        _vo_items, _new_fade,
                        shot_start_offset_sec=_shot_offset)
            applied += 1
            # NOTE: no break — a segment may span multiple shots (e.g. [0,15]
            # covers both sh01 [0,12) and sh02 [12,20)). Continue iterating so
            # every overlapping shot receives the music assignment and duck intervals.

    if applied:
        print(f"  [INFO] Applied {applied} user segment(s) from {source_name}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate music review artifacts (timeline + preview audio) "
                    "from a merged VOPlan.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", required=True, metavar="PATH",
                   help="Path to VOPlan.{locale}.json")
    p.add_argument("--output", default=None, metavar="DIR",
                   help="Output directory for review pack. "
                        "Default: assets/music/MusicReviewPack/ under the episode.")
    p.add_argument("--overrides", default=None, metavar="PATH",
                   help="JSON file with user overrides (shot_overrides array). "
                        "Each entry: {start_sec, end_sec, duck_db, fade_sec, music_asset_id}. "
                        "Applied on top of manifest values before rendering.")
    p.add_argument("--preview-only", action="store_true",
                   help="Only regenerate preview_audio.wav (skip timeline.txt).")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    manifest = load_manifest(manifest_path)

    locale_scope = manifest.get("locale_scope")
    if locale_scope not in ("merged", "monolithic", "locale", None):
        raise SystemExit(
            f"[ERROR] Expected a VOPlan manifest (locale_scope='locale', 'merged', or 'monolithic'), "
            f"got '{locale_scope}'."
        )

    project_id = manifest.get("project_id", "")
    episode_id = manifest.get("episode_id", "")
    locale = manifest.get("locale", "en")

    # Resolve output directory
    if args.output:
        out_dir = Path(args.output).resolve()
    else:
        episode_dir = PIPE_DIR / "projects" / project_id / "episodes" / episode_id
        out_dir = episode_dir / "assets" / "music" / "MusicReviewPack"

    # Load ShotList for shot ordering
    shots = load_shotlist(manifest, manifest_path)
    if not shots:
        raise SystemExit("[ERROR] ShotList not found — cannot determine shot ordering.")

    # Build vo_item_id → shot_id mapping from ShotList
    vo_id_to_shot = {}
    for shot in shots:
        shot_id = shot.get("shot_id", "")
        for vid in shot.get("audio_intent", {}).get("vo_item_ids", []):
            vo_id_to_shot[vid] = shot_id

    # Group vo_items by shot_id
    vo_shot_map = {}
    for vo in manifest.get("vo_items", []):
        sid = vo_id_to_shot.get(vo.get("item_id")) or vo.get("shot_id", "")
        if sid:
            vo_shot_map.setdefault(sid, []).append(vo)

    # music_items no longer stored in VOPlan manifest; MusicPlan is the source of truth.
    # music_index starts empty — music bars only appear after a MusicPlan is created.
    music_index = {}

    # Load loop candidates (optional)
    episode_dir = PIPE_DIR / "projects" / project_id / "episodes" / episode_id
    loop_info = load_loop_candidates(episode_dir)

    # ── Load RenderPlan VO lines for accurate shot-relative duck intervals. ──
    # Duration override removed — ShotList.json already has correct values.
    _rp_path = episode_dir / f"RenderPlan.{locale}.json"
    _rp_shot_dur: dict[str, float] = {}
    if _rp_path.exists():
        try:
            import json as _json
            _rp = _json.loads(_rp_path.read_text(encoding="utf-8"))
            for _rs in _rp.get("shots", []):
                _sid = _rs.get("shot_id") or _rs.get("id", "")
                _dur_ms = _rs.get("duration_ms")
                if _sid and _dur_ms is not None:
                    _rp_shot_dur[_sid] = _dur_ms / 1000.0
            print(f"  [RenderPlan] Loaded {len(_rp_shot_dur)} shot durations from RenderPlan.{locale}.json (ShotList used — no override applied)")
        except Exception as _e:
            print(f"  [WARN] Could not load RenderPlan.{locale}.json: {_e} — using ShotList durations")
    else:
        print(f"  [WARN] RenderPlan.{locale}.json not found — using ShotList durations (run step 10 first for accurate music preview)")

    print("=" * 60)
    print("  music_review_pack")
    print(f"  Manifest : {manifest_path.name}")
    print(f"  Locale   : {locale}")
    print(f"  Shots    : {len(shots)}")
    print(f"  Music    : {len(music_index)} items")
    print(f"  VO items : {len(manifest.get('vo_items', []))}")
    print(f"  Output   : {out_dir}")
    print("=" * 60)

    # Build timeline
    timeline_shots, total_dur = build_timeline(
        shots, manifest, vo_shot_map, music_index, loop_info,
    )

    # Apply user overrides on top of timeline (duck_db, fade_sec, music_asset_id, etc.)

    # Step 1: auto-load MusicPlan.json from the episode dir (always honoured)
    music_plan_path = episode_dir / "MusicPlan.json"
    _loaded_music_plan = None
    if music_plan_path.exists():
        try:
            with open(music_plan_path, encoding="utf-8") as f:
                _loaded_music_plan = json.load(f)
            plan_segments = _loaded_music_plan.get("shot_overrides", [])
            if plan_segments:
                apply_music_plan_overrides(timeline_shots, plan_segments, "MusicPlan.json", vo_shot_map)
        except Exception as exc:
            print(f"  [WARN] Could not read MusicPlan.json: {exc}", file=sys.stderr)

    # Step 2: explicit --overrides file takes precedence over MusicPlan.json
    if args.overrides:
        overrides_path = Path(args.overrides).resolve()
        if overrides_path.exists():
            with open(overrides_path, encoding="utf-8") as f:
                user_overrides = json.load(f)
            apply_music_plan_overrides(timeline_shots, user_overrides, overrides_path.name, vo_shot_map)
        else:
            print(f"  [WARN] Overrides file not found: {overrides_path}")

    # ── Apply track_volumes and clip_volumes to each shot's base_db ──────────
    # These are per-stem or per-clip dB offsets set in the Source Music / Generated
    # Clips panels.  They must be reflected in the preview audio (base_db).
    import re as _re
    _track_vols = {}
    _clip_vols  = {}
    if _loaded_music_plan:
        _track_vols = _loaded_music_plan.get("track_volumes", {})
        _clip_vols  = _loaded_music_plan.get("clip_volumes",  {})
    if _track_vols or _clip_vols:
        _applied_vol = 0
        for _entry in timeline_shots:
            _render_id = _entry.get("music_item_id_override") or _entry.get("music_item_id") or ""
            if not _render_id:
                continue
            _db_off = 0.0
            # Track volume: source stem is prefix before the "_XXs-XXXs" timestamp suffix.
            # e.g. "cher1_11_1s-23_0s" → stem "cher1"
            _stem = _re.sub(r'_\d[\d_]*s-[\d_\.]+s$', '', _render_id)
            _db_off += float(_track_vols.get(_stem, 0))
            # Clip volume: keyed by item_id ("music-sc01-sh01") or clip_id ("cher1:11.1s-23.0s")
            _db_off += float(_clip_vols.get(_render_id, 0))
            # Also try clip_id format derived from WAV stem: "cher1_11_1s-23_0s" → "cher1:11.1s-23.0s"
            if _db_off == 0.0:
                _m = _re.match(r'^(.+?)_(\d+)_(\d+)s-(\d+)_(\d+)s$', _render_id)
                if _m:
                    _cid = f"{_m.group(1)}:{_m.group(2)}.{_m.group(3)}s-{_m.group(4)}.{_m.group(5)}s"
                    _db_off += float(_clip_vols.get(_cid, 0))
            if _db_off != 0.0:
                _entry["base_db"] = _entry.get("base_db", BASE_MUSIC_DB) + _db_off
                _applied_vol += 1
        if _applied_vol:
            print(f"  [INFO] Applied track/clip volume offsets to {_applied_vol} shot(s)")

    # Write artifacts
    if not args.preview_only:
        write_timeline_txt(timeline_shots, total_dur, episode_id, out_dir / "timeline.txt")
    render_preview_audio(timeline_shots, total_dur, manifest, manifest_path,
                         out_dir / "preview_audio.wav")

    print(f"\n{'=' * 60}")
    print("  SUMMARY — music_review_pack")
    print(f"{'=' * 60}")
    print(f"  Total duration : {total_dur:.2f}s")
    print(f"  Shots          : {len(timeline_shots)}")
    print(f"  Output dir     : {out_dir}")
    print(f"  Files written  : timeline.txt, preview_audio.wav")


if __name__ == "__main__":
    main()
