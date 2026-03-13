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

    return timeline_shots, round(cumulative_sec, 3)


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

    # Load sfx_selections
    sfx_selections = json.loads(Path(args.sfx_selections).read_text(encoding="utf-8"))

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

    # Build shot timeline
    timeline_shots, total_dur = build_shot_timeline(shots, manifest, vo_shot_map, music_index, {})

    # Scene-tail shift — keyed by scene_id from ShotList (Issue 18 fix)
    _scene_tails_cfg = manifest.get("scene_tails", {})
    _DEFAULT_SCENE_TAIL_MS = 2000
    _scene_shift_map = {}
    _prev_scene = None
    _shift_acc = 0.0
    for entry in timeline_shots:
        sc = entry["scene_id"]
        if _prev_scene is not None and sc != _prev_scene:
            _tail_ms = int(_scene_tails_cfg.get(sc, _scene_tails_cfg.get(_prev_scene, _DEFAULT_SCENE_TAIL_MS)))
            _shift_acc += _tail_ms / 1000.0
        if sc not in _scene_shift_map:
            _scene_shift_map[sc] = _shift_acc
        _prev_scene = sc

    _shot_scene_shift = {e["shot_id"]: _scene_shift_map.get(e["scene_id"], 0.0) for e in timeline_shots}
    _vo_scene_shift = {}
    for entry in timeline_shots:
        shift = _shot_scene_shift[entry["shot_id"]]
        for vo in entry.get("vo_lines", []):
            _vo_scene_shift[vo["item_id"]] = shift

    # Allocate buffer
    n_samples = int((total_dur + _shift_acc) * SAMPLE_RATE) + SAMPLE_RATE
    buf = np.zeros((n_samples, CHANNELS), dtype=np.float64)

    vo_dir = ep_dir / "assets" / locale / "audio" / "vo"
    music_dir = ep_dir / "assets" / "music"

    # Load MusicPlan if --include-music
    music_plan = {}
    if args.include_music:
        mp_path = ep_dir / "assets" / "music" / "MusicPlan.json"
        if mp_path.exists():
            music_plan = json.loads(mp_path.read_text(encoding="utf-8"))

    loop_selections = music_plan.get("loop_selections", {})
    shot_overrides_list = music_plan.get("shot_overrides", [])
    shot_overrides = {o["item_id"]: o for o in shot_overrides_list if "item_id" in o}
    track_volumes = music_plan.get("track_volumes", {})
    clip_volumes = music_plan.get("clip_volumes", {})

    # Feature C: auto-detect RenderPlan for VO timing parity
    render_plan = None
    render_plan_path = ep_dir / f"RenderPlan.{locale}.json"
    if render_plan_path.exists():
        try:
            render_plan = json.loads(render_plan_path.read_text(encoding="utf-8"))
            print(f"  [INFO] Using RenderPlan for VO timing: {render_plan_path.name}")
        except Exception as e:
            print(f"  [WARN] Failed to load RenderPlan: {e}")
    else:
        print(f"  [INFO] No RenderPlan found, using manifest start_sec")

    # Build render_plan vo timing lookup if available
    rp_vo_timing = {}  # { item_id: timeline_in_sec }
    if render_plan:
        for rp_shot in render_plan.get("shots", []):
            for vo_line in rp_shot.get("vo_lines", []):
                iid = vo_line.get("item_id", "")
                t_ms = vo_line.get("timeline_in_ms")
                if iid and t_ms is not None:
                    rp_vo_timing[iid] = t_ms / 1000.0

    # Build sfx_index: { shot_id: [sel_entry] }
    print(f"  [SFX] Building sfx_index from {len(sfx_selections)} selection(s): {list(sfx_selections.keys())}")
    sfx_index = {}
    for item_id, sel in sfx_selections.items():
        source_file = sel.get("source_file")
        if not source_file:
            print(f"  [SKIP] {item_id}: no local audio file (source_file is null)")
            continue
        # Path security
        p = Path(source_file).resolve()
        if not str(p).startswith(pipe_root + os.sep) and str(p) != pipe_root:
            print(f"  [SKIP] {item_id}: path outside project root: {p}")
            continue
        if not p.is_file():
            print(f"  [SKIP] {item_id}: file not found: {p}")
            continue
        # Find shot_id for this sfx item
        sfx_meta = next((s for s in manifest.get("sfx_items", []) if s.get("item_id") == item_id), None)
        if not sfx_meta:
            print(f"  [SKIP] {item_id}: not found in manifest sfx_items")
            continue
        shot_id = sfx_meta.get("shot_id", "")
        sfx_index.setdefault(shot_id, []).append({
            "item_id": item_id,
            "source_file": str(p),
            "start": float(sel.get("start", 0)),
            "end": sel.get("end"),  # None = auto
            "duration_sec": float(sfx_meta.get("duration_sec", 0)),
        })

    # Timeline data for timeline.json
    tl_shots_out = []
    tl_vo_out = []
    tl_sfx_out = []
    tl_music_out = []

    for idx, entry in enumerate(timeline_shots):
        shot_id = entry["shot_id"]
        scene_shift = _shot_scene_shift.get(shot_id, 0.0)
        shot_offset = entry["offset_sec"] + scene_shift
        shot_dur = entry["duration_sec"]
        offset_samples = int(shot_offset * SAMPLE_RATE)

        tl_shots_out.append({
            "shot_id": shot_id,
            "scene_id": entry.get("scene_id", ""),
            "offset_sec": round(shot_offset, 3),
            "duration_sec": round(shot_dur, 3),
            "scene_shift_sec": round(scene_shift, 3),
        })

        # ── Place VO ──
        for vo in entry.get("vo_lines", []):
            iid = vo.get("item_id", "")
            vo_shift = _vo_scene_shift.get(iid, scene_shift)

            if render_plan and iid in rp_vo_timing:
                # Feature C: use RenderPlan timing
                vo_start = shot_offset + rp_vo_timing[iid]
            else:
                raw_start = vo.get("start_sec")
                if raw_start is None:
                    continue
                vo_start = raw_start + vo_shift

            raw_end = vo.get("end_sec")
            vo_end = (raw_end + vo_shift) if raw_end is not None else vo_start

            vo_path = vo_dir / f"{iid}.wav"
            if not vo_path.exists():
                continue
            try:
                vo_data = read_wav_mono_or_stereo(str(vo_path), SAMPLE_RATE)
            except Exception as e:
                print(f"  [WARN] VO read failed {vo_path}: {e}")
                continue
            s = int(vo_start * SAMPLE_RATE)
            e2 = min(s + len(vo_data), n_samples)
            if s < n_samples and e2 > s:
                buf[s:e2] += vo_data[:e2 - s]

            tl_vo_out.append({
                "item_id": iid,
                "start_sec": round(vo_start, 3),
                "end_sec": round(vo_start + len(vo_data) / SAMPLE_RATE, 3),
                "speaker_id": vo.get("speaker_id", ""),
                "shot_id": shot_id,
            })

        # ── Place SFX ──
        for sel in sfx_index.get(shot_id, []):
            sfx_start_in_shot = max(0.0, float(sel.get("start", 0)))
            sfx_end_in_shot = sel.get("end")
            abs_start = shot_offset + sfx_start_in_shot

            try:
                print(f"  [SFX] Reading {sel['item_id']} from {sel['source_file']}")
                sfx_data = read_wav_mono_or_stereo(sel["source_file"], SAMPLE_RATE)
                print(f"  [SFX] Read OK: {len(sfx_data)} samples ({len(sfx_data)/SAMPLE_RATE:.2f}s)")
            except Exception as e:
                print(f"  [WARN] SFX read failed {sel['source_file']}: {e}")
                continue

            if sfx_end_in_shot is not None:
                max_samples = max(0, int((float(sfx_end_in_shot) - sfx_start_in_shot) * SAMPLE_RATE))
            else:
                remaining = shot_dur - sfx_start_in_shot
                max_samples = max(0, int(remaining * SAMPLE_RATE))

            sfx_data = sfx_data[:max_samples]
            sfx_data = sfx_data * (10 ** (SFX_DB / 20.0))

            s = int(abs_start * SAMPLE_RATE)
            e2 = min(s + len(sfx_data), n_samples)
            if s < n_samples and e2 > s:
                buf[s:e2] += sfx_data[:e2 - s]

            tl_sfx_out.append({
                "item_id": sel["item_id"],
                "start_sec": round(abs_start, 3),
                "end_sec": round(abs_start + len(sfx_data) / SAMPLE_RATE, 3),
                "shot_id": shot_id,
                "tag": next((x.get("tag","") for x in manifest.get("sfx_items",[]) if x.get("item_id")==sel["item_id"]), ""),
            })

        # ── Place Music ──
        if args.include_music and entry.get("music_item_id"):
            mid = entry["music_item_id"]
            ovr = shot_overrides.get(mid, {})
            clip_id = ovr.get("music_clip_id", "")

            duck_db = float(ovr.get("duck_db", DEFAULT_DUCK_DB))
            fade_sec = float(ovr.get("fade_sec", DEFAULT_FADE_SEC))
            music_start_sec = float(ovr.get("start_sec", 0.0))

            # Resolve base_db from track/clip volumes
            stem = clip_id.split(":")[0] if clip_id else ""
            base_db = BASE_MUSIC_DB
            base_db += track_volumes.get(stem, 0)
            base_db += clip_volumes.get(mid, clip_volumes.get(clip_id, 0))

            # Resolve actual WAV asset id — MusicPlan shot_override carries
            # music_asset_id (e.g. "cher1_43_8s-51_9s") which is the filename stem.
            # Fall back to mid (music_item_id) if no override exists.
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
                    music_offset_samples = int(music_start_sec * SAMPLE_RATE)
                    music_data = music_data[music_offset_samples:]
                    shot_samples = max(0, int((shot_dur - music_start_sec) * SAMPLE_RATE))
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

                    s = int(shot_offset * SAMPLE_RATE)
                    e2 = min(s + len(music_data), n_samples)
                    if s < n_samples and e2 > s:
                        buf[s:e2] += music_data[:e2 - s]

                    tl_music_out.append({
                        "item_id": mid,
                        "start_sec": round(shot_offset, 3),
                        "end_sec": round(shot_offset + shot_dur, 3),
                        "shot_id": shot_id,
                        "music_mood": entry.get("music_mood", ""),
                    })

    # Trim buffer to last VO end + 5s (avoids writing excess silence)
    if tl_vo_out:
        last_vo_sec = max(v["end_sec"] for v in tl_vo_out)
    else:
        last_vo_sec = total_dur + _shift_acc
    trim_samples = min(int((last_vo_sec + 5.0) * SAMPLE_RATE), n_samples)
    buf = buf[:trim_samples]

    # Peak normalize
    peak = np.abs(buf).max()
    if peak > 1.0:
        buf /= peak

    # Write timeline.json FIRST (survives crash)
    tl_path = output_dir / "timeline.json"
    tl_doc = {
        "total_dur_sec": round(total_dur + _shift_acc, 3),
        "timing_source": "render_plan" if (render_plan and rp_vo_timing) else "manifest",
        "shots": tl_shots_out,
        "vo_items": tl_vo_out,
        "sfx_items": tl_sfx_out,
        "music_items": tl_music_out,
    }
    with open(tl_path, "w", encoding="utf-8") as f:
        json.dump(tl_doc, f, indent=2, ensure_ascii=False)
    print(f"  [OK] {tl_path}")

    # Atomic write
    buf_out = buf.astype(np.float32)
    tmp_path = output_path.with_suffix(".tmp.wav")
    sf.write(str(tmp_path), buf_out, SAMPLE_RATE, subtype="PCM_16")
    tmp_path.rename(output_path)
    print(f"  [OK] {output_path}  ({peak:.3f} peak, {len(buf)/SAMPLE_RATE:.2f}s)")


if __name__ == "__main__":
    main()
