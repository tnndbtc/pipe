#!/usr/bin/env python3
# sfx_preview_pack.py — Generate SFX preview WAV (VO + SFX + optional Music)

import argparse, json, math, os, sys
from pathlib import Path
import numpy as np
import soundfile as sf

PIPE_DIR = Path(__file__).resolve().parent.parent.parent
SAMPLE_RATE = 48000
CHANNELS = 2
SFX_DB = -3.0
BASE_MUSIC_DB = -6.0
DEFAULT_DUCK_DB = -12.0   # matches render_video.py default
DEFAULT_FADE_SEC = 0.8
DEFAULT_VO_PAUSE_SEC = 0.3   # gap between sequential VO lines when no timing


# ── I/O helpers ──────────────────────────────────────────────────────────────

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


def compute_duck_intervals(vo_items_for_shot, fade_sec):
    """Compute music duck intervals from VO line positions within a shot."""
    raw = []
    for item in vo_items_for_shot:
        start = item.get("start_sec")
        end = item.get("end_sec")
        if start is None or end is None:
            continue
        t0 = max(0.0, start - fade_sec)
        t1 = end + fade_sec
        raw.append((t0, t1))
    return merge_overlapping(raw)


# ── Audio helpers ────────────────────────────────────────────────────────────

def read_wav_mono_or_stereo(path, target_sr=SAMPLE_RATE):
    """Read any audio file (WAV/MP3/FLAC/OGG) and return a (samples, 2) stereo numpy array at target_sr.
    Uses librosa for format-agnostic loading with proper resampling."""
    try:
        import librosa
        # librosa.load returns mono float32, resampled to target_sr
        data, _ = librosa.load(str(path), sr=target_sr, mono=False)
        # librosa returns (channels, samples) for multi-channel or (samples,) for mono
        if data.ndim == 1:
            data = np.column_stack([data, data])
        else:
            data = data.T  # (channels, samples) → (samples, channels)
            if data.shape[1] == 1:
                data = np.column_stack([data[:, 0], data[:, 0]])
            elif data.shape[1] > 2:
                data = data[:, :2]
        return data.astype(np.float64)
    except ImportError:
        pass
    # Fallback to soundfile (WAV/FLAC/OGG only, no MP3)
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


# ── Timeline builder ─────────────────────────────────────────────────────────

def build_shot_timeline(shots, manifest, vo_shot_map, music_index, loop_info):
    """
    Build the timeline data structure used for both text and JSON output.

    Adapted from music_review_pack.py build_timeline() with scene_id added.
    Returns a list of shot dicts with enriched timeline info, plus total_duration_sec.
    """
    timeline_shots = []
    cumulative_sec = 0.0

    for shot in shots:
        shot_id = shot["shot_id"]
        duration = float(shot.get("duration_sec", 0))

        # Find music item for this shot
        music_item = music_index.get(shot_id)
        duck_db = 0.0
        fade_sec = DEFAULT_FADE_SEC
        start_sec = 0.0
        music_mood = ""
        music_item_id = ""
        duck_intervals = []

        if music_item:
            music_item_id = music_item.get("item_id", "")
            music_mood = music_item.get("music_mood", "")
            duck_db = 0.0
            fade_sec = float(music_item.get("fade_sec", DEFAULT_FADE_SEC))
            duck_intervals = music_item.get("duck_intervals", [])
            start_sec = float(music_item.get("start_sec", 0.0))

        # If duck_intervals not in manifest, compute from VO
        vo_items_for_shot = vo_shot_map.get(shot_id, [])
        if not duck_intervals and vo_items_for_shot:
            duck_intervals = compute_duck_intervals(vo_items_for_shot, fade_sec)

        # VO lines
        vo_lines = []
        for vo in vo_items_for_shot:
            vo_lines.append({
                "item_id": vo.get("item_id", ""),
                "start_sec": vo.get("start_sec"),
                "end_sec": vo.get("end_sec"),
                "speaker_id": vo.get("speaker_id", ""),
                "text": vo.get("text", ""),
            })

        entry = {
            "shot_id": shot_id,
            "scene_id": shot.get("scene_id", ""),   # Issue 18 fix
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

        timeline_shots.append(entry)
        cumulative_sec += duration

    # Derive total_dur from VOPlan vo_items — authoritative source for episode end.
    # Never use ShotList cumulative: ShotList is downstream and can be stale after
    # scene_heads are applied (e.g. scene_heads["sc01"]=15 shifts all VO timings
    # forward but does NOT update ShotList.json).
    # Derive total_dur from VOPlan vo_items — authoritative source for episode end.
    # Never use ShotList cumulative: ShotList is downstream and can be stale after
    # scene_heads are applied (e.g. scene_heads["sc01"]=15 shifts all VO timings
    # forward but does NOT update ShotList.json).
    total_dur = max(
        ((vo.get("end_sec") or 0.0) + (vo.get("pause_after_ms") or 0) / 1000.0
         for sh_vos in vo_shot_map.values()
         for vo in sh_vos),
        default=0.0,
    )
    return timeline_shots, total_dur


# ── Per-VO-line duck envelope ────────────────────────────────────────────────

def build_per_vo_envelope(n_samples, duck_intervals, duck_db, fade_sec, sr):
    """Per-VO-line ducking envelope matching render_video.py build_duck_expr()."""
    base_amp = 10 ** (BASE_MUSIC_DB / 20.0)
    duck_amp = base_amp * (10 ** (duck_db / 20.0))
    fade_n = int(fade_sec * sr)
    env = np.full(n_samples, base_amp, dtype=np.float64)
    for t0, t1 in duck_intervals:
        s0 = int(t0 * sr)
        s1 = int(t1 * sr)
        # Clamp to valid buffer range
        s0 = max(0, min(s0, n_samples))
        s1 = max(0, min(s1, n_samples))
        # Fade-in ramp: base → duck, ending at s0
        ramp_start = max(0, s0 - fade_n)
        if ramp_start < s0:
            length = s0 - ramp_start
            env[ramp_start:s0] = np.linspace(base_amp, duck_amp, length)
        # Duck hold: s0 → s1
        if s0 < s1:
            env[s0:s1] = duck_amp
        # Fade-out ramp: duck → base, starting at s1
        ramp_end = min(n_samples, s1 + fade_n)
        if s1 < ramp_end:
            length = ramp_end - s1
            env[s1:ramp_end] = np.linspace(duck_amp, base_amp, length)
    return env


def render_sfx_preview(timeline_shots, total_dur, manifest, manifest_path,
                       ep_dir, locale, sfx_selections, output_path,
                       include_music=False):
    """Render SFX preview WAV and return the tl_doc dict (no timeline.json written).

    Parameters
    ----------
    timeline_shots  : list[dict]  — from build_shot_timeline()
    total_dur       : float       — total episode duration in seconds
    manifest        : dict        — parsed VOPlan
    manifest_path   : Path        — path to the manifest (for relative-path helpers)
    ep_dir          : Path        — episode root directory
    locale          : str         — locale code (e.g. "en")
    sfx_selections  : dict        — {item_id: {source_file, start, end, preview_url}}
    output_path     : Path | None — if None, skip WAV write (compute-only mode)
    include_music   : bool        — mix music stems into preview

    Returns
    -------
    dict  tl_doc with keys: total_dur_sec, timing_source, shots, vo_items,
          sfx_items, music_items
    """
    pipe_root = str(PIPE_DIR.resolve())

    # music_items no longer stored in VOPlan manifest; MusicPlan is the source of truth.
    # music_index starts empty — music mixing is driven solely by MusicPlan data below.
    music_index = {}

    # Allocate buffer — total_dur is from ShotList which already bakes in
    # scene tails (last shot of each scene carries the tail duration).
    # No extra scene-tail shift needed here.
    n_samples = int(total_dur * SAMPLE_RATE) + SAMPLE_RATE
    buf = np.zeros((n_samples, CHANNELS), dtype=np.float64)

    vo_dir = ep_dir / "assets" / locale / "audio" / "vo"
    music_dir = ep_dir / "assets" / "music"

    # Load MusicPlan if include_music
    music_plan = {}
    if include_music:
        mp_path = ep_dir / "MusicPlan.json"
        if mp_path.exists():
            music_plan = json.loads(mp_path.read_text(encoding="utf-8"))

    loop_selections = music_plan.get("loop_selections", {})
    shot_overrides_list = music_plan.get("shot_overrides", [])
    # Index shot_overrides by start_sec for O(1) lookup; keep full list for range queries.
    shot_overrides_by_start = {seg["start_sec"]: seg for seg in shot_overrides_list if "start_sec" in seg}
    track_volumes = music_plan.get("track_volumes", {})
    clip_volumes = music_plan.get("clip_volumes", {})

    # Build sfx_ready_list from SfxPlan v2 sfx_entries[].
    # Each segment carries start_sec/end_sec/source_file directly (no shot_id/item_id).
    sfx_entries_input = sfx_selections if isinstance(sfx_selections, list) else []
    print(f"  [SFX] Building sfx_ready_list from {len(sfx_entries_input)} segment(s)")
    sfx_ready_list: list[dict] = []
    for idx_seg, seg in enumerate(sfx_entries_input):
        source_file = seg.get("source_file")
        if not source_file:
            print(f"  [SKIP] segment[{idx_seg}]: no source_file")
            continue
        # Path security
        p = Path(source_file).resolve()
        if not str(p).startswith(pipe_root + os.sep) and str(p) != pipe_root:
            print(f"  [SKIP] segment[{idx_seg}]: path outside project root: {p}")
            continue
        if not p.is_file():
            print(f"  [SKIP] segment[{idx_seg}]: file not found: {p}")
            continue
        sfx_ready_list.append({
            "item_id":       seg.get("clip_id") or f"seg{idx_seg}",
            "source_file":   str(p),
            "start":         float(seg.get("start_sec", 0)),
            "end":           seg.get("end_sec"),  # None = auto
            "volume_db":     float(seg.get("volume_db",     0) or 0),
            "duck_db":       float(seg.get("duck_db",       0) or 0),
            "fade_sec":      float(seg.get("fade_sec",      0) or 0),
            "clip_volume_db":0.0,
            "is_cut_clip":   bool(seg.get("clip_path")),
        })

    # Timeline data collections
    tl_shots_out = []
    tl_vo_out = []
    tl_sfx_out = []
    tl_music_out = []

    for idx, entry in enumerate(timeline_shots):
        shot_id = entry["shot_id"]
        # entry["offset_sec"] is ShotList-based cumulative — already episode-absolute
        # including scene tails. Do NOT add any extra scene-tail shift.
        shot_offset = entry["offset_sec"]
        shot_dur = entry["duration_sec"]
        offset_samples = int(shot_offset * SAMPLE_RATE)

        tl_shots_out.append({
            "shot_id": shot_id,
            "scene_id": entry.get("scene_id", ""),
            "offset_sec": round(shot_offset, 3),
            "duration_sec": round(shot_dur, 3),
        })

        # ── Place VO ──
        # vo_cursor: shot-relative cursor used when no timing is available
        vo_cursor = 0.0
        has_timing = any(
            v.get("start_sec") is not None for v in entry.get("vo_lines", [])
        )
        for vo in entry.get("vo_lines", []):
            iid = vo.get("item_id", "")

            if has_timing:
                raw_start = vo.get("start_sec")
                if raw_start is None:
                    continue
                vo_start = raw_start
            else:
                # No timing in manifest or RenderPlan — place sequentially
                # within the shot using WAV durations (same fallback as
                # music_review_pack.py).
                vo_start = shot_offset + vo_cursor

            vo_path = vo_dir / f"{iid}.wav"
            if not vo_path.exists():
                if not has_timing:
                    vo_cursor += DEFAULT_VO_PAUSE_SEC
                continue
            try:
                vo_data = read_wav_mono_or_stereo(str(vo_path), SAMPLE_RATE)
            except Exception as e:
                print(f"  [WARN] VO read failed {vo_path}: {e}")
                if not has_timing:
                    vo_cursor += DEFAULT_VO_PAUSE_SEC
                continue

            vo_dur = len(vo_data) / SAMPLE_RATE
            if not has_timing:
                vo_cursor += vo_dur + DEFAULT_VO_PAUSE_SEC

            raw_end = vo.get("end_sec")
            if raw_end is not None and has_timing:
                vo_end = raw_end
            else:
                vo_end = vo_start + vo_dur

            s = int(vo_start * SAMPLE_RATE)
            e2 = min(s + len(vo_data), n_samples)
            if s < n_samples and e2 > s:
                buf[s:e2] += vo_data[:e2 - s]

            tl_vo_out.append({
                "item_id": iid,
                "start_sec": round(vo_start, 3),
                "end_sec": round(vo_start + vo_dur, 3),
                "speaker_id": vo.get("speaker_id", ""),
                "shot_id": shot_id,
            })

        # ── Place Music ──
        # Use range-overlap (not point-in-range) to match segments: a shot at
        # offset=0 dur=28s must match a segment [5, 20] even though 0 < 5.
        # Mirror music_review_pack.apply_music_plan_overrides overlap logic.
        if include_music:
            _shot_end = shot_offset + shot_dur
            ovr = next(
                (seg for seg in shot_overrides_list
                 if not (float(seg.get("start_sec", 0)) >= _shot_end
                         or float(seg.get("end_sec", 0)) <= shot_offset)),
                {},
            )
            mid = entry.get("music_item_id") or ovr.get("music_asset_id", "")
        if include_music and mid:
            clip_id = ovr.get("music_clip_id", "")

            # Manifest music item carries duck_db / fade_sec authored by the LLM.
            # MusicPlan shot_overrides (ovr) take priority; fall back to manifest values
            # before the hardcoded defaults so the preview matches the actual render.
            manifest_music_item = music_index.get(shot_id, {})
            manifest_duck_db  = float(manifest_music_item.get("duck_db",  DEFAULT_DUCK_DB))
            manifest_fade_sec = float(manifest_music_item.get("fade_sec", DEFAULT_FADE_SEC))
            duck_db  = float(ovr.get("duck_db",  manifest_duck_db))
            fade_sec = float(ovr.get("fade_sec", manifest_fade_sec))
            # shot_overrides carry episode-absolute coords; convert to within-shot
            music_start_sec = max(0.0, float(ovr["start_sec"]) - shot_offset) if "start_sec" in ovr else 0.0
            music_end_sec   = max(0.0, float(ovr["end_sec"]) - shot_offset)   if "end_sec"   in ovr else shot_dur

            # Resolve base_db from track/clip volumes
            stem = clip_id.split(":")[0] if clip_id else ""
            base_db = BASE_MUSIC_DB
            base_db += track_volumes.get(stem, 0)
            base_db += clip_volumes.get(mid, clip_volumes.get(clip_id, 0))

            # Resolve actual WAV asset id — MusicPlan music_segment carries
            # music_asset_id (e.g. "cher1_43_8s-51_9s") which is the filename stem.
            # Fall back to mid (music_item_id) if no segment matches.
            music_asset_id = ovr.get("music_asset_id", mid)
            # Find music WAV: prefer loop variant, then clip WAV by asset_id, then by item_id
            music_path = music_dir / f"{music_asset_id}.loop.wav"
            if not music_path.exists():
                music_path = music_dir / f"{music_asset_id}.wav"
            if not music_path.exists():
                # legacy fallback: try by item_id (mid)
                music_path = music_dir / f"{mid}.loop.wav"
                if not music_path.exists():
                    music_path = music_dir / f"{mid}.wav"
            if not music_path.exists():
                print(f"  [SKIP] Music WAV not found for {mid} (asset_id={music_asset_id})")
                # skip — jump past the else block
            else:
                try:
                    music_data = read_wav_mono_or_stereo(str(music_path), SAMPLE_RATE)
                except Exception as e:
                    print(f"  [WARN] Music read failed {music_path}: {e}")
                    music_data = None

                if music_data is not None:
                    # How many samples of music content we need (start→end within shot)
                    shot_samples = max(0, int((music_end_sec - music_start_sec) * SAMPLE_RATE))
                    # Loop the WAV if it is shorter than the required content duration
                    if shot_samples > 0 and len(music_data) > 0 and len(music_data) < shot_samples:
                        reps = (shot_samples // len(music_data)) + 1
                        music_data = np.tile(music_data, (reps, 1) if music_data.ndim == 2 else reps)
                    music_data = music_data[:shot_samples]

                    # Per-VO-line duck envelope — VO start/end in vo_lines are absolute
                    # episode positions; duck envelope is shot-relative, so subtract shot_offset.
                    shot_relative_vo = [
                        {"start_sec": (v.get("start_sec") or 0) - shot_offset,
                         "end_sec":   (v.get("end_sec")   or 0) - shot_offset}
                        for v in entry.get("vo_lines", [])
                        if v.get("start_sec") is not None
                    ]
                    duck_intervals = compute_duck_intervals(shot_relative_vo, fade_sec)
                    env = build_per_vo_envelope(len(music_data), duck_intervals, duck_db, fade_sec, SAMPLE_RATE)

                    # Apply base_db offset
                    base_scale = 10 ** ((base_db - BASE_MUSIC_DB) / 20.0)
                    # Handle both stereo (2D) and mono (1D) — read_wav_mono_or_stereo always returns 2D
                    if music_data.ndim == 2:
                        music_data = music_data * env[:, np.newaxis]
                    else:
                        music_data = music_data * env
                    music_data = music_data * base_scale

                    # Place music at shot_offset + music_start_sec (within-shot delay)
                    s = int((shot_offset + music_start_sec) * SAMPLE_RATE)
                    e2 = min(s + len(music_data), n_samples)
                    if s < n_samples and e2 > s:
                        buf[s:e2] += music_data[:e2 - s]

                    music_actual_start = shot_offset + music_start_sec
                    music_actual_end   = music_actual_start + shot_samples / SAMPLE_RATE
                    tl_music_out.append({
                        "item_id": mid,
                        "start_sec": round(music_actual_start, 3),
                        "end_sec": round(music_actual_end, 3),
                        "shot_id": shot_id,
                        "music_mood": entry.get("music_mood", ""),
                    })

    # ── Place SFX (v2: iterate sfx_entries directly, episode-absolute timing) ──
    # Each segment from SfxPlan v2 carries start_sec/end_sec/source_file directly.
    # No shot_id lookup — place each segment at its episode-absolute start_sec.
    for sel in sfx_ready_list:
        abs_start    = float(sel.get("start", 0))
        sfx_end_ep   = sel.get("end")
        sfx_duration = (float(sfx_end_ep) - abs_start) if sfx_end_ep is not None else None

        try:
            print(f"  [SFX] Reading {sel['item_id']} from {sel['source_file']}")
            sfx_data = read_wav_mono_or_stereo(sel["source_file"], SAMPLE_RATE)
            print(f"  [SFX] Read OK: {len(sfx_data)} samples ({len(sfx_data)/SAMPLE_RATE:.2f}s)")
        except Exception as e:
            print(f"  [WARN] SFX read failed {sel['source_file']}: {e}")
            continue

        if sfx_duration is not None:
            max_samples = max(0, int(float(sfx_duration) * SAMPLE_RATE))
        else:
            max_samples = len(sfx_data)  # no cap — use full clip

        sfx_data = sfx_data[:max_samples]
        _sfx_vol_db  = float(sel.get("volume_db",     0) or 0)
        _sfx_duck_db = float(sel.get("duck_db",       0) or 0)
        _sfx_clip_db = float(sel.get("clip_volume_db",0) or 0)
        sfx_data = sfx_data * (10 ** ((SFX_DB + _sfx_vol_db + _sfx_duck_db + _sfx_clip_db) / 20.0))

        _sfx_fade_sec = float(sel.get("fade_sec", 0) or 0)
        if _sfx_fade_sec > 0 and len(sfx_data) > 0:
            fade_n = int(_sfx_fade_sec * SAMPLE_RATE)
            if fade_n > 0:
                in_n  = min(fade_n, len(sfx_data))
                out_n = min(fade_n, len(sfx_data))
                sfx_data[:in_n]  *= np.linspace(0.0, 1.0, in_n).reshape(-1, 1) if sfx_data.ndim == 2 else np.linspace(0.0, 1.0, in_n)
                sfx_data[-out_n:] *= np.linspace(1.0, 0.0, out_n).reshape(-1, 1) if sfx_data.ndim == 2 else np.linspace(1.0, 0.0, out_n)

        s = int(abs_start * SAMPLE_RATE)
        e2 = min(s + len(sfx_data), n_samples)
        if s < n_samples and e2 > s:
            buf[s:e2] += sfx_data[:e2 - s]

        tl_sfx_out.append({
            "item_id": sel["item_id"],
            "start_sec": round(abs_start, 3),
            # end_sec represents the user's intended window (start_sec→end_sec from SfxPlan),
            # not the physical clip length.  If the clip is shorter the audio simply ends early
            # inside that window; if longer it is capped.  Use sfx_end_ep when available.
            "end_sec": round(float(sfx_end_ep), 3) if sfx_end_ep is not None
                       else round(abs_start + len(sfx_data) / SAMPLE_RATE, 3),
        })

    # Trim buffer to total_dur (which already includes pause_after_ms of the last VO item).
    # Do NOT use last_vo_sec + 5: that ignores pause_after_ms (e.g. 10s tail on last item)
    # and would cut the preview short.  total_dur = max(end_sec + pause_after_ms/1000) from
    # VOPlan, so it is the authoritative episode end.
    trim_samples = min(int(total_dur * SAMPLE_RATE), n_samples)
    buf = buf[:trim_samples]

    # Loudnorm pass targeting −16 LUFS
    try:
        import pyloudnorm as pyln
        _meter = pyln.Meter(SAMPLE_RATE)
        _loud = _meter.integrated_loudness(buf)
        if _loud is not None and not (_loud != _loud) and _loud > -70.0:  # finite and not silence
            buf = pyln.normalize.loudness(buf, _loud, -16.0)
    except Exception:
        # Fallback: peak limiter at -1 dBFS
        _peak = float(abs(buf).max()) if len(buf) else 0.0
        if _peak > 0.891:
            buf = buf * (0.891 / _peak)

    tl_doc = {
        "total_dur_sec": round(total_dur, 3),
        "timing_source": "manifest",
        "shots": tl_shots_out,
        "vo_items": tl_vo_out,
        "sfx_items": tl_sfx_out,
        "music_items": tl_music_out,
    }

    # Write WAV (atomic)
    if output_path is not None:
        buf_out = buf.astype(np.float32)
        tmp_path = output_path.with_suffix(".tmp.wav")
        sf.write(str(tmp_path), buf_out, SAMPLE_RATE, subtype="PCM_16")
        tmp_path.rename(output_path)
        print(f"  [OK] {output_path}  ({len(buf)/SAMPLE_RATE:.2f}s)")

    return tl_doc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sfx-selections", required=True)
    parser.add_argument("--include-music", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    project_id = manifest.get("project_id", "")
    episode_id = manifest.get("episode_id", "")
    locale = manifest.get("locale", "") or manifest_path.stem.split(".")[-1]

    ep_dir = PIPE_DIR / "projects" / project_id / "episodes" / episode_id

    # Load sfx_selections — v2: the file is a SfxPlan JSON; extract sfx_entries[].
    _sfx_raw = json.loads(Path(args.sfx_selections).read_text(encoding="utf-8"))
    if isinstance(_sfx_raw, dict):
        sfx_selections = _sfx_raw.get("shot_overrides", [])
    else:
        # Already a bare list (legacy callers)
        sfx_selections = _sfx_raw

    # Path security: validate all source_file paths are under PIPE_DIR
    pipe_root = str(PIPE_DIR.resolve())

    # Output dir
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = ep_dir / "assets" / "sfx" / "SfxPreviewPack"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "preview_audio.wav"

    # Load ShotList
    shots = load_shotlist(manifest, manifest_path)

    # Build vo_item_id → shot_id mapping via ShotList audio_intent (vo_items have no shot_id field)
    vo_id_to_shot = {}
    for shot in shots:
        sid = shot.get("shot_id", "")
        for vid in shot.get("audio_intent", {}).get("vo_item_ids", []):
            vo_id_to_shot[vid] = sid

    # Group vo_items by shot_id
    vo_shot_map = {}
    for vo in manifest.get("vo_items", []):
        sid = vo_id_to_shot.get(vo.get("item_id", "")) or vo.get("shot_id", "")
        if sid:
            vo_shot_map.setdefault(sid, []).append(vo)

    # Build music_index: { shot_id: music_item }
    music_index = {}
    for mi in manifest.get("music_items", []):
        sid = mi.get("shot_id", "")
        if sid:
            music_index[sid] = mi

    # Load RenderPlan for shot duration patching (VO ceiling)
    render_plan = None
    render_plan_path = ep_dir / f"RenderPlan.{locale}.json"
    if render_plan_path.exists():
        try:
            render_plan = json.loads(render_plan_path.read_text(encoding="utf-8"))
            print(f"  [INFO] Loaded RenderPlan for shot durations: {render_plan_path.name}")
        except Exception as e:
            print(f"  [WARN] Failed to load RenderPlan: {e}")
    else:
        print(f"  [INFO] No RenderPlan found, using ShotList durations")

    # Patch ShotList durations with VO-ceiling values from RenderPlan.
    # gen_render_plan applies a VO ceiling (last_vo_out_ms + tail) that makes
    # many shots shorter than the ShotList estimate.  Without this patch, the
    # SFX tab accumulates wrong offsets from shot 1 onward — exactly the same
    # bug that music_review_pack.py fixes with its _rp_shot_dur block.
    if render_plan:
        _rp_shot_dur: dict[str, float] = {}
        for _rs in render_plan.get("shots", []):
            _sid = _rs.get("shot_id", "")
            _dur_ms = _rs.get("duration_ms")
            if _sid and _dur_ms is not None:
                _rp_shot_dur[_sid] = _dur_ms / 1000.0
        if _rp_shot_dur:
            _patched = 0
            for _shot in shots:
                _sid = _shot.get("shot_id", "")
                if _sid in _rp_shot_dur:
                    _shot["duration_sec"] = _rp_shot_dur[_sid]
                    _patched += 1
            print(f"  [INFO] Patched {_patched} shot duration(s) from RenderPlan (VO ceiling applied)")

    # Build shot timeline
    timeline_shots, total_dur = build_shot_timeline(shots, manifest, vo_shot_map, music_index, {})

    # Render preview (returns tl_doc; writes WAV to output_path)
    render_sfx_preview(
        timeline_shots, total_dur, manifest, manifest_path,
        ep_dir, locale, sfx_selections, output_path,
        include_music=args.include_music,
    )


if __name__ == "__main__":
    main()
