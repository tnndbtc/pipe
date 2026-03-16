#!/usr/bin/env python3
# =============================================================================
# music_review_pack.py — Generate music review artifacts for human inspection
# =============================================================================
#
# Runs AFTER post_tts_analysis.py (Stage 9[4/8]) when both music WAVs and VO
# timing exist.  Produces three review artifacts:
#
#   1. timeline.txt   — human-readable shot timeline with music, VO, ducking
#   2. timeline.json  — machine-readable version for Web UI
#   3. preview_audio.wav — VO + music mix (no SFX, no video) with ducking
#
# Usage:
#   python music_review_pack.py \
#       --manifest projects/slug/ep/AssetManifest.en.json
#
#   python music_review_pack.py \
#       --manifest projects/slug/ep/AssetManifest.en.json \
#       --output   projects/slug/ep/assets/music/MusicReviewPack/
#
# Requirements:
#   pip install soundfile numpy
# =============================================================================

import argparse
import datetime
import json
import sys
import tempfile
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

    return timeline_shots, round(cumulative_sec, 3)


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


def write_timeline_json(timeline_shots, total_dur, episode_id, out_path):
    """Write machine-readable timeline.json."""
    doc = {
        "episode_id": episode_id,
        "total_duration_sec": total_dur,
        "shots": timeline_shots,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"  [OK] {out_path}")


# ── Preview audio renderer ───────────────────────────────────────────────────

def render_preview_audio(timeline_shots, total_dur, manifest, manifest_path, out_path):
    """
    Render a VO + music mix WAV with ducking applied.
    No SFX, no video — purely for music review listening.

    After writing preview_audio.wav, also writes MusicApprovalSnapshot.json
    to {ep_dir}/assets/music/ capturing the exact values used in the render.
    """
    project_id = manifest.get("project_id", "")
    episode_id = manifest.get("episode_id", "")
    locale = manifest.get("locale", "en")

    n_samples = int(total_dur * SAMPLE_RATE) + SAMPLE_RATE  # +1s safety

    # ── Build scene-shift lookup so VO gets the same inter-scene pauses
    # that the VO-tab "Generate Preview" adds (scene_tails from manifest). ──────
    _scene_tails_cfg = manifest.get("scene_tails", {})
    _DEFAULT_SCENE_TAIL_MS = 2000
    _vo_scene_shift: dict = {}   # item_id → extra seconds from accumulated scene pauses
    _prev_scene_id  = None
    _shift_acc      = 0.0
    for _vo_item in manifest.get("vo_items", []):
        _iid   = _vo_item.get("item_id", "")
        _parts = _iid.split("-")
        _scn   = _parts[1] if len(_parts) >= 3 else ""   # "sc01" from "vo-sc01-001"
        if _prev_scene_id is not None and _scn != _prev_scene_id:
            _tail_ms = int(_scene_tails_cfg.get(
                _scn, _scene_tails_cfg.get(_prev_scene_id, _DEFAULT_SCENE_TAIL_MS)))
            _shift_acc += _tail_ms / 1000.0
        _vo_scene_shift[_iid] = _shift_acc
        _prev_scene_id = _scn
    # Grow buffer to fit the extra scene-tail time
    n_samples += int(_shift_acc * SAMPLE_RATE)

    buf = np.zeros((n_samples, CHANNELS), dtype=np.float64)

    episode_dir = PIPE_DIR / "projects" / project_id / "episodes" / episode_id
    vo_dir = episode_dir / "assets" / locale / "audio" / "vo"
    music_dir = episode_dir / "assets" / "music"

    missing_vo = 0
    missing_music = 0
    any_timing_missing = False

    # Pre-compute render_id for each shot (for shot-continuity fade decisions)
    def _resolve_render_id(entry):
        mid = entry.get("music_item_id")
        if not mid:
            return None
        return entry.get("music_item_id_override", mid)

    render_ids = [_resolve_render_id(e) for e in timeline_shots]

    # Per-shot snapshot records (FIX A) — one entry per shot, filled during render
    snapshot_shots = []

    for idx, entry in enumerate(timeline_shots):
        offset_samples = int(entry["offset_sec"] * SAMPLE_RATE)

        # Initialise snapshot record for this shot with defaults (no music)
        snap = {
            "shot_id": entry["shot_id"],
            "music_asset_id": None,
            "duration_ms": int(entry["duration_sec"] * 1000),
            "duck_intervals": [],
            "duck_db": float(entry.get("duck_db", 0.0)),
            "base_db": BASE_MUSIC_DB,
            "fade_sec": float(entry.get("fade_sec", DEFAULT_FADE_SEC)),
            "music_delay_sec": float(entry.get("start_sec", 0.0)),
            "loop_wav_path": None,
        }
        snapshot_shots.append(snap)

        # ── Place VO lines ──────────────────────────────────────────────────
        vo_cursor = 0.0
        has_timing = any(v.get("start_sec") is not None for v in entry["vo_lines"])
        if not has_timing and entry["vo_lines"]:
            any_timing_missing = True

        for vo in entry["vo_lines"]:
            item_id = vo["item_id"]
            wav_path = vo_dir / f"{item_id}.wav"
            if not wav_path.exists():
                missing_vo += 1
                if not has_timing:
                    vo_cursor += DEFAULT_PAUSE_SEC
                continue

            vo_data = read_wav_mono_or_stereo(wav_path)
            vo_dur = len(vo_data) / SAMPLE_RATE

            if has_timing:
                start = vo.get("start_sec", 0.0)
                # start_sec is absolute global time; offset already baked in.
                # Add per-item scene-tail shift so scene boundaries match the
                # VO-tab preview (which inserts scene_tails silence).
                start += _vo_scene_shift.get(item_id, 0.0)
                s = int(start * SAMPLE_RATE)
            else:
                start = vo_cursor
                vo_cursor += vo_dur + DEFAULT_PAUSE_SEC
                # vo_cursor is shot-relative; add shot offset to get global pos
                s = offset_samples + int(start * SAMPLE_RATE)
            e = min(s + len(vo_data), n_samples)
            length = e - s
            if length > 0:
                buf[s:e] += vo_data[:length]  # VO at 0 dB

        # ── Place music with shot-boundary envelope ─────────────────────────
        render_id = render_ids[idx]
        if not render_id:
            continue

        # Resolve music file path
        music_path = music_dir / f"{render_id}.loop.wav"
        if not music_path.exists():
            music_path = music_dir / f"{render_id}.wav"
        if not music_path.exists():
            resources_music = PIPE_DIR / "projects" / project_id / "resources" / "music"
            found_source = False
            if resources_music.is_dir():
                for ext in (".mp3", ".wav", ".flac", ".ogg"):
                    candidate = resources_music / f"{render_id}{ext}"
                    if candidate.exists():
                        music_path = candidate
                        found_source = True
                        break
            if not found_source:
                missing_music += 1
                continue

        music_data = read_wav_mono_or_stereo(music_path)

        # start_sec: how many seconds into the shot before music begins
        start_sec = float(entry.get("start_sec", 0.0))
        music_offset_samples = int(start_sec * SAMPLE_RATE)
        # Music fills from its start point to the end of the shot (or overridden music end)
        shot_samples = max(0, int((entry.get("music_end_sec", entry["duration_sec"]) - start_sec) * SAMPLE_RATE))

        # Truncate or tile music to fill the remaining shot duration
        if shot_samples == 0:
            continue
        if len(music_data) >= shot_samples:
            music_data = music_data[:shot_samples]
        else:
            repeats = (shot_samples // len(music_data)) + 1
            music_data = np.tile(music_data, (repeats, 1))[:shot_samples]

        # Shot-boundary fades: skip fade when same track continues across boundary
        prev_id = render_ids[idx - 1] if idx > 0 else None
        next_id = render_ids[idx + 1] if idx < len(render_ids) - 1 else None
        do_fade_in  = (prev_id != render_id)
        do_fade_out = (next_id != render_id)

        duck_db      = float(entry.get("duck_db", 0))
        base_db      = float(entry.get("base_db", BASE_MUSIC_DB))
        fade_sec     = float(entry.get("fade_sec", DEFAULT_FADE_SEC))
        duck_intervals_shot = entry.get("duck_intervals", [])  # shot-relative seconds

        # Build shot-boundary envelope (fade in/out at track boundaries).
        # Holds at base_db level — duck_db is applied separately below,
        # only during VO intervals.
        envelope = build_shot_envelope(
            len(music_data), base_db, fade_sec,
            fade_in=do_fade_in, fade_out=do_fade_out,
        )

        # Apply per-VO ducking envelope — mirrors render_video.py's build_duck_expr.
        # duck_intervals are shot-relative; music_data starts at start_sec within the shot,
        # so subtract start_sec to convert to music-buffer-relative time.
        if duck_intervals_shot and duck_db != 0:
            base_amp  = 10 ** (base_db / 20.0)
            duck_amp  = base_amp * (10 ** (duck_db / 20.0))
            n         = len(music_data)
            t_arr     = (np.arange(n) / SAMPLE_RATE)  # time within music_data (0-based)
            # music_data t=0 corresponds to shot time start_sec
            duck_env  = np.full(n, base_amp, dtype=np.float64)
            for t0_shot, t1_shot in duck_intervals_shot:
                t0 = t0_shot - start_sec   # convert to music-buffer-relative
                t1 = t1_shot - start_sec
                if t1 <= 0:
                    continue   # duck interval ends before music starts — skip
                t0 = max(t0, 0.0)
                fade = min(fade_sec, max(0.0, (t1 - t0) / 2.0))
                t_fi_end   = t0 + fade
                t_fo_start = t1 - fade
                in_interval = (t_arr >= t0) & (t_arr <= t1)
                # Fade in: t0 → t0+fade  (base_amp → duck_amp)
                fade_in_mask = in_interval & (t_arr <= t_fi_end) & (fade > 0)
                duck_env[fade_in_mask] = base_amp + (duck_amp - base_amp) * \
                    (t_arr[fade_in_mask] - t0) / fade
                # Hold: t0+fade → t1-fade
                duck_env[in_interval & (t_arr > t_fi_end) & (t_arr < t_fo_start)] = duck_amp
                # Fade out: t1-fade → t1  (duck_amp → base_amp)
                fade_out_mask = in_interval & (t_arr >= t_fo_start) & (fade > 0)
                duck_env[fade_out_mask] = duck_amp + (base_amp - duck_amp) * \
                    (t_arr[fade_out_mask] - t_fo_start) / fade
            # Combine: shot-boundary envelope × duck envelope / base_amp
            # (shot envelope already encodes base_amp level; divide to avoid double-applying)
            envelope = envelope * (duck_env / base_amp)

        # Apply envelope (stereo)
        music_data = music_data * envelope[:, np.newaxis]

        # Place in buffer at shot start + start_sec offset
        s = offset_samples + music_offset_samples
        e = min(s + len(music_data), n_samples)
        length = e - s
        if length > 0:
            buf[s:e] += music_data[:length]

        # ── Update snapshot record with exact values used (FIX A) ──────────
        # duration_ms: shot duration from ShotList (already set above from entry["duration_sec"]).
        # Do NOT overwrite with music buffer length — shot duration is what render_video needs.
        snap["music_asset_id"] = render_id
        snap["duck_intervals"] = entry.get("duck_intervals", [])  # shot-relative (FIX C)
        snap["duck_db"] = duck_db
        snap["base_db"] = base_db
        snap["fade_sec"] = fade_sec
        snap["music_delay_sec"] = start_sec
        # music_end_sec: shot-relative position where music stops.
        # Uses overridden music_end_sec if set; falls back to full shot duration.
        snap["music_end_sec"] = float(entry.get("music_end_sec", entry["duration_sec"]))
        snap["loop_wav_path"] = str(music_path)

    # Trim to last VO end + 5s fade-out, so music review doesn't play 45s of
    # music-only after narration finishes (accumulated music-loop padding).
    last_vo_end_sec = 0.0
    for entry in timeline_shots:
        for vo in entry.get("vo_lines", []):
            end = vo.get("end_sec")
            if end is not None:
                # Account for scene-tail shift applied during VO placement
                shifted_end = end + _vo_scene_shift.get(vo.get("item_id", ""), 0.0)
                last_vo_end_sec = max(last_vo_end_sec, shifted_end)
    VO_OUTRO_SEC = 5.0
    trim_sec = min(last_vo_end_sec + VO_OUTRO_SEC, total_dur) if last_vo_end_sec > 0 else total_dur
    trim_samples = int(trim_sec * SAMPLE_RATE)
    buf = buf[:trim_samples]
    # Apply a short fade-out on the last VO_OUTRO_SEC so music doesn't cut abruptly
    fade_len = min(int(VO_OUTRO_SEC * SAMPLE_RATE), trim_samples)
    fade_env = np.linspace(1.0, 0.0, fade_len)
    buf[-fade_len:] *= fade_env[:, None] if CHANNELS > 1 else fade_env
    total_dur = trim_sec
    print(f"  [TRIM] last VO at {last_vo_end_sec:.1f}s → preview trimmed to {trim_sec:.1f}s (+{VO_OUTRO_SEC:.0f}s fade)")

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

    # ── Write MusicApprovalSnapshot.json (FIX A) ─────────────────────────────
    snapshot_path = episode_dir / "assets" / "music" / "MusicApprovalSnapshot.json"
    try:
        saved_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        snapshot_doc = {
            "schema": "MusicApprovalSnapshot",
            "version": "1.0",
            "saved_at": saved_at,
            "locale": locale,
            "shots": snapshot_shots,
        }
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        # Write atomically via temp file + rename
        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=snapshot_path.parent, prefix=".MusicApprovalSnapshot.", suffix=".tmp"
        )
        try:
            with open(tmp_fd, "w", encoding="utf-8") as tf:
                json.dump(snapshot_doc, tf, indent=2, ensure_ascii=False)
                tf.write("\n")
            Path(tmp_name).replace(snapshot_path)
        except Exception:
            # Clean up temp file if rename/write failed
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except Exception:
                pass
            raise
        print(f"  [OK] {snapshot_path}")
    except Exception as exc:
        print(f"  [WARN] Could not write MusicApprovalSnapshot.json: {exc}", file=sys.stderr)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate music review artifacts (timeline + preview audio) "
                    "from a merged AssetManifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", required=True, metavar="PATH",
                   help="Path to AssetManifest.{locale}.json")
    p.add_argument("--output", default=None, metavar="DIR",
                   help="Output directory for review pack. "
                        "Default: assets/music/MusicReviewPack/ under the episode.")
    p.add_argument("--overrides", default=None, metavar="PATH",
                   help="JSON file with user overrides (shot_overrides array). "
                        "Each entry: {item_id, duck_db, fade_sec, start_sec, music_asset_id}. "
                        "Applied on top of manifest values before rendering.")
    p.add_argument("--preview-only", action="store_true",
                   help="Only regenerate preview_audio.wav (skip timeline.txt/json).")
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
    if locale_scope not in ("merged", "monolithic", None):
        raise SystemExit(
            f"[ERROR] Expected merged manifest (locale_scope='merged'), "
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

    # Index music_items by shot_id
    music_index = {}
    for mi in manifest.get("music_items", []):
        sid = mi.get("shot_id", "")
        if sid:
            music_index[sid] = mi

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
            print(f"  [RenderPlan] Loaded {len(_rp_shot_dur)} shot durations from RenderPlan.{locale}.json (VO ceiling applied)")
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
    # Helper: apply a list of override dicts to timeline_shots in-place
    def _apply_overrides(override_list, source_name):
        ovr_by_item = {o["item_id"]: o for o in override_list if "item_id" in o}
        applied = 0
        for entry in timeline_shots:
            mid = entry.get("music_item_id", "")
            if mid in ovr_by_item:
                ovr = ovr_by_item[mid]
                if "duck_db" in ovr:
                    entry["duck_db"] = float(ovr["duck_db"])
                if "fade_sec" in ovr:
                    entry["fade_sec"] = float(ovr["fade_sec"])
                if "start_sec" in ovr:
                    # MusicPlan stores episode-absolute coords; convert to within-shot
                    _shot_offset = float(entry.get("offset_sec", 0.0))
                    entry["start_sec"] = max(0.0, float(ovr["start_sec"]) - _shot_offset)
                if "end_sec" in ovr:
                    # MusicPlan stores episode-absolute coords; convert to within-shot.
                    # Store as music_end_sec (separate from shot duration_sec to avoid corruption).
                    _shot_offset = float(entry.get("offset_sec", 0.0))
                    entry["music_end_sec"] = max(0.0, float(ovr["end_sec"]) - _shot_offset)
                if "music_asset_id" in ovr:
                    # Store override separately — keep original music_item_id
                    # for UI (timeline.json). Renderer uses _override for audio.
                    entry["music_item_id_override"] = ovr["music_asset_id"]
                # Recompute duck intervals only if fade_sec changed (which affects
                # the margin around each VO line). Use manifest vo_items timing.
                if "fade_sec" in ovr:
                    _sid = entry["shot_id"]
                    _new_fade = float(entry.get("fade_sec", DEFAULT_FADE_SEC))
                    _vo_items = vo_shot_map.get(_sid, [])
                    if _vo_items:
                        entry["duck_intervals"] = compute_duck_intervals(
                            _vo_items, _new_fade,
                            shot_start_offset_sec=float(entry.get("offset_sec", 0.0)))
                applied += 1
        if applied:
            print(f"  [INFO] Applied {applied} user override(s) from {source_name}")

    # Step 1: auto-load MusicPlan.json from the episode dir (always honoured)
    music_plan_path = episode_dir / "MusicPlan.json"
    _loaded_music_plan = None
    if music_plan_path.exists():
        try:
            with open(music_plan_path, encoding="utf-8") as f:
                _loaded_music_plan = json.load(f)
            plan_overrides = _loaded_music_plan.get("shot_overrides", [])
            if plan_overrides:
                _apply_overrides(plan_overrides, "MusicPlan.json")
        except Exception as exc:
            print(f"  [WARN] Could not read MusicPlan.json: {exc}", file=sys.stderr)

    # Step 2: explicit --overrides file takes precedence over MusicPlan.json
    if args.overrides:
        overrides_path = Path(args.overrides).resolve()
        if overrides_path.exists():
            with open(overrides_path, encoding="utf-8") as f:
                user_overrides = json.load(f)
            _apply_overrides(user_overrides, overrides_path.name)
        else:
            print(f"  [WARN] Overrides file not found: {overrides_path}")

    # ── Apply track_volumes and clip_volumes to each shot's base_db ──────────
    # These are per-stem or per-clip dB offsets set in the Source Music / Generated
    # Clips panels.  They must be reflected in both the preview audio AND the
    # MusicApprovalSnapshot (so the render reads the correct base_db).
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
        write_timeline_json(timeline_shots, total_dur, episode_id, out_dir / "timeline.json")
    render_preview_audio(timeline_shots, total_dur, manifest, manifest_path,
                         out_dir / "preview_audio.wav")

    print(f"\n{'=' * 60}")
    print("  SUMMARY — music_review_pack")
    print(f"{'=' * 60}")
    print(f"  Total duration : {total_dur:.2f}s")
    print(f"  Shots          : {len(timeline_shots)}")
    print(f"  Output dir     : {out_dir}")
    print(f"  Files written  : timeline.txt, timeline.json, preview_audio.wav")


if __name__ == "__main__":
    main()
