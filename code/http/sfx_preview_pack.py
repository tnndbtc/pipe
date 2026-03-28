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


def render_sfx_preview(manifest, manifest_path,
                       ep_dir, locale, sfx_selections, output_path,
                       include_music=False):
    """Render SFX preview WAV and return the tl_doc dict (no timeline.json written).

    Parameters
    ----------
    manifest        : dict        — parsed VOPlan
    manifest_path   : Path        — path to the manifest (for relative-path helpers)
    ep_dir          : Path        — episode root directory
    locale          : str         — locale code (e.g. "en")
    sfx_selections  : list        — SfxPlan sfx_entries segments
    output_path     : Path | None — if None, skip WAV write (compute-only mode)
    include_music   : bool        — mix music stems into preview

    Returns
    -------
    dict  tl_doc with keys: total_dur_sec, timing_source, vo_items,
          sfx_items, music_items
    """
    pipe_root = str(PIPE_DIR.resolve())

    # total_dur from manifest.vo_items — authoritative source, no ShotList needed.
    total_dur = max(
        ((vo.get("end_sec") or 0.0) + (vo.get("pause_after_ms") or 0) / 1000.0
         for vo in manifest.get("vo_items", [])),
        default=0.0,
    )

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
    tl_vo_out = []
    tl_sfx_out = []
    tl_music_out = []

    # ── Place VO — direct from manifest.vo_items (no shot decomposition) ──────
    # vo_items carry episode-absolute start_sec/end_sec baked by the approve flow.
    # No shot context needed — place each item directly at its start_sec position.
    for _vo in manifest.get("vo_items", []):
        _start = _vo.get("start_sec")
        if _start is None:
            continue
        _iid = _vo.get("item_id", "")
        _vp = vo_dir / f"{_iid}.wav"
        if not _vp.exists():
            continue
        try:
            _vd = read_wav_mono_or_stereo(str(_vp), SAMPLE_RATE)
        except Exception as _ve:
            print(f"  [WARN] VO read failed {_vp}: {_ve}")
            continue
        _vdb = float(_vo.get("volume_db", 0.0) or 0.0)
        if _vdb != 0.0:
            _gain = 10 ** (_vdb / 20.0)
            _vd = _vd * _gain
        _s = int(_start * SAMPLE_RATE)
        _e2 = min(_s + len(_vd), n_samples)
        if _s < n_samples and _e2 > _s:
            buf[_s:_e2] += _vd[:_e2 - _s]
        tl_vo_out.append({
            "item_id": _iid,
            "start_sec": round(_start, 3),
            "end_sec": round(_start + len(_vd) / SAMPLE_RATE, 3),
            "speaker_id": _vo.get("speaker_id", ""),
            "shot_id": "",
        })

    # ── Place Music — segment-direct from MusicPlan ─────────────────────────
    # Iterate shot_overrides at episode-absolute coords — no shot decomposition.
    # seg_start_sec IS the buffer write position; no shot offset involved.
    if include_music:
        for _seg in shot_overrides_list:
            _seg_start = float(_seg.get("start_sec", 0.0))
            _seg_end   = float(_seg.get("end_sec",   0.0))
            _asset_id  = _seg.get("music_asset_id", "")
            _clip_id   = _seg.get("music_clip_id", "")
            _duck_db   = float(_seg.get("duck_db",  DEFAULT_DUCK_DB))
            _fade_sec  = float(_seg.get("fade_sec", DEFAULT_FADE_SEC))

            if not _asset_id or _seg_end <= _seg_start:
                continue

            # Resolve base_db from track/clip volumes
            _stem    = _clip_id.split(":")[0] if _clip_id else ""
            _base_db = BASE_MUSIC_DB
            _base_db += track_volumes.get(_stem, 0)
            _base_db += clip_volumes.get(_asset_id, clip_volumes.get(_clip_id, 0))

            # Resolve music WAV path
            _music_path = music_dir / f"{_asset_id}.loop.wav"
            if not _music_path.exists():
                _music_path = music_dir / f"{_asset_id}.wav"
            if not _music_path.exists():
                print(f"  [SKIP] Music WAV not found for asset_id={_asset_id}")
                continue

            try:
                _music_data = read_wav_mono_or_stereo(str(_music_path), SAMPLE_RATE)
            except Exception as _me:
                print(f"  [WARN] Music read failed {_music_path}: {_me}")
                continue

            _seg_samples = max(0, int((_seg_end - _seg_start) * SAMPLE_RATE))
            if _seg_samples == 0 or len(_music_data) == 0:
                continue

            if len(_music_data) < _seg_samples:
                _reps = (_seg_samples // len(_music_data)) + 1
                _music_data = np.tile(_music_data, (_reps, 1) if _music_data.ndim == 2 else _reps)
            _music_data = _music_data[:_seg_samples]

            # Duck intervals from manifest VO items overlapping this segment.
            # compute_duck_intervals expects segment-relative times — subtract _seg_start.
            _seg_vo_raw = [
                {"start_sec": (v.get("start_sec") or 0) - _seg_start,
                 "end_sec":   (v.get("end_sec")   or 0) - _seg_start}
                for v in manifest.get("vo_items", [])
                if v.get("start_sec") is not None
                and float(v.get("end_sec", 0)) > _seg_start
                and float(v.get("start_sec", 0)) < _seg_end
            ]
            _duck_intervals = compute_duck_intervals(_seg_vo_raw, _fade_sec)
            _env = build_per_vo_envelope(_seg_samples, _duck_intervals, _duck_db, _fade_sec, SAMPLE_RATE)

            # Apply base_db offset and envelope
            _base_scale = 10 ** ((_base_db - BASE_MUSIC_DB) / 20.0)
            if _music_data.ndim == 2:
                _music_data = _music_data * _env[:, np.newaxis]
            else:
                _music_data = _music_data * _env
            _music_data = _music_data * _base_scale

            # Place at episode-absolute segment start
            _s = int(_seg_start * SAMPLE_RATE)
            _e2 = min(_s + len(_music_data), n_samples)
            if _s < n_samples and _e2 > _s:
                buf[_s:_e2] += _music_data[:_e2 - _s]

            tl_music_out.append({
                "item_id": _asset_id,
                "start_sec": round(_seg_start, 3),
                "end_sec": round(_seg_end, 3),
                "shot_id": "",
                "music_mood": "",
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

    # Output dir
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = ep_dir / "assets" / "sfx" / "SfxPreviewPack"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "preview_audio.wav"

    # Render preview (returns tl_doc; writes WAV to output_path)
    render_sfx_preview(
        manifest, manifest_path,
        ep_dir, locale, sfx_selections, output_path,
        include_music=args.include_music,
    )


if __name__ == "__main__":
    main()
