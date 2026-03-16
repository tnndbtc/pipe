#!/usr/bin/env python3
"""
media_preview_pack.py — Generate a preview video for the Media tab.

Reads:
  --input <json>  temp JSON with keys: ep_dir, locale, selections, include_music, include_sfx

selections format: { shot_id: [ { media_type, url, start_sec, end_sec }, ... ] }
  - shot_id keys are real shot_ids (already inverted by JS)
  - start_sec and end_sec are guaranteed non-null (JS null-coalescing applied)

Outputs:
  {ep_dir}/assets/media/MediaPreviewPack/preview_video.mp4
"""

import argparse
import json
import os
import sys
import tempfile
import subprocess
from pathlib import Path
from urllib.parse import urlparse, unquote

PIPE_DIR = Path(__file__).resolve().parent.parent.parent
W, H, FPS = 1280, 720, 25
SAMPLE_RATE = 44100
CHANNELS = 2


def url_to_path(url: str) -> str:
    """Convert file:// URI to filesystem path for ffmpeg."""
    if url.startswith("file://"):
        return unquote(urlparse(url).path)
    return url


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def detect_shot_durations(ep_dir: Path, locale: str) -> tuple:
    """
    Returns (shot_id -> duration_ms dict, source_description).
    Priority: ShotList.json > RenderPlan > MusicApprovalSnapshot.

    ShotList.json is the authoritative timing source for all display tabs
    (Media, SFX, Music).  The Media tab UI derives its shot duration labels
    (e.g. "0:26.1 – 0:51.9") from ShotList; the preview must use the same
    source so clip length matches the displayed range.  RenderPlan may be
    stale (pre-Stage-10) and should only be used as a fallback.
    """
    # Priority 1: ShotList.json — single authoritative timing source
    shot_path = ep_dir / "ShotList.json"
    if shot_path.exists():
        sl = load_json(shot_path)
        durations = {}
        for shot in sl.get("shots", []):
            sid = shot.get("shot_id", "")
            dur_sec = shot.get("duration_sec", 0) or 0
            if sid and dur_sec:
                durations[sid] = round(dur_sec * 1000)
        if durations:
            print(f"  [dur] Using ShotList.json ({len(durations)} shots)")
            return durations, "ShotList.json"

    # Priority 2: RenderPlan (fallback — may be stale before Step 10)
    rp_path = ep_dir / f"RenderPlan.{locale}.json"
    if rp_path.exists():
        rp = load_json(rp_path)
        durations = {s["shot_id"]: s["duration_ms"]
                     for s in rp.get("shots", [])
                     if "shot_id" in s and "duration_ms" in s}
        if durations:
            print(f"  [dur] Using RenderPlan.{locale}.json ({len(durations)} shots) — ShotList unavailable")
            return durations, f"RenderPlan.{locale}.json"

    # Priority 3: MusicApprovalSnapshot
    snap_path = ep_dir / "assets" / "music" / "MusicApprovalSnapshot.json"
    if snap_path.exists():
        snap = load_json(snap_path)
        durations = {s["shot_id"]: s["duration_ms"]
                     for s in snap.get("shots", [])
                     if "shot_id" in s and "duration_ms" in s}
        if durations:
            print(f"  [dur] Using MusicApprovalSnapshot.json ({len(durations)} shots)")
            return durations, "MusicApprovalSnapshot.json"

    print(f"ERROR: No shot duration source found for locale '{locale}'.")
    print(f"  Checked: {shot_path}")
    print(f"  Checked: {rp_path}")
    print(f"  Checked: {snap_path}")
    sys.exit(1)


def build_silent_audio(duration_sec: float, out_path: Path) -> None:
    """Generate silent audio file of given duration."""
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
        "-t", str(duration_sec),
        str(out_path)
    ], capture_output=True, check=True)


def _anim_vf(anim_type: str, clip_dur: float) -> str:
    """
    Return a zoompan ffmpeg filter string for the given animation type,
    to be appended after scale/pad (input is already W×H).
    Returns empty string for static / no animation.
    """
    d = max(1, round(clip_dur * FPS))
    if anim_type == "zoom_in":
        return (f"zoompan=z='1+0.3*on/{d}':x='(iw-iw/zoom)/2':y='(ih-ih/zoom)/2'"
                f":d={d}:s={W}x{H}:fps={FPS}")
    if anim_type == "zoom_out":
        return (f"zoompan=z='1.3-0.3*on/{d}':x='(iw-iw/zoom)/2':y='(ih-ih/zoom)/2'"
                f":d={d}:s={W}x{H}:fps={FPS}")
    if anim_type == "pan_lr":
        return (f"zoompan=z='1.1':x='(iw-iw/zoom)*on/{d}':y='(ih-ih/zoom)/2'"
                f":d={d}:s={W}x{H}:fps={FPS}")
    if anim_type == "pan_rl":
        return (f"zoompan=z='1.1':x='(iw-iw/zoom)*(1-on/{d})':y='(ih-ih/zoom)/2'"
                f":d={d}:s={W}x{H}:fps={FPS}")
    if anim_type == "pan_up":
        return (f"zoompan=z='1.1':x='(iw-iw/zoom)/2':y='(ih-ih/zoom)*(1-on/{d})'"
                f":d={d}:s={W}x{H}:fps={FPS}")
    if anim_type == "ken_burns":
        return (f"zoompan=z='1+0.25*on/{d}'"
                f":x='(iw-iw/zoom)*0.3*on/{d}':y='(ih-ih/zoom)*0.3*on/{d}'"
                f":d={d}:s={W}x{H}:fps={FPS}")
    return ""  # static / none


def _seg_display_dur(seg: dict) -> float:
    """
    Return how long this segment should play, exactly as the user set it.
    Returns 0.0 if no duration is specified — caller must skip or warn.

    - images: start_sec/end_sec are shot-timeline markers, NOT trim bounds.
              Use hold_sec (user-set display duration) first.
    - videos: start_sec/end_sec ARE trim bounds; use end-start first.
    """
    media_type = seg.get("media_type", "image")

    if media_type != "image":
        # Video: in/out trim range takes priority
        start = float(seg.get("start_sec") or 0)
        end   = float(seg.get("end_sec")   or 0)
        if end > start:
            return end - start

    # Image (or video without valid trim): use explicit hold/duration
    hold = seg.get("hold_sec")
    if hold:
        return float(hold)
    dur = seg.get("duration_override_sec") or seg.get("duration_sec")
    if dur:
        return float(dur)
    return 0.0  # unknown — caller should warn and skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to temp JSON config file")
    args = ap.parse_args()

    cfg = load_json(Path(args.input))
    ep_dir        = Path(cfg["ep_dir"])
    locale        = cfg.get("locale", "en")
    selections    = cfg.get("selections", {})   # { shot_id: [segments] }
    include_music = cfg.get("include_music", False)
    include_sfx   = cfg.get("include_sfx", False)
    shot_ids_filter = cfg.get("shot_ids", None)   # None = all shots; list = restrict to these
    out_name      = cfg.get("out_name", None) or "preview_video.mp4"

    print(f"  [media_preview] ep_dir={ep_dir} locale={locale}")
    print(f"  [media_preview] shots with selections: {len(selections)}")
    print(f"  [media_preview] include_music={include_music} include_sfx={include_sfx}")
    if shot_ids_filter is not None:
        print(f"  [media_preview] shot_ids filter: {shot_ids_filter}")

    # Output directory
    out_dir = ep_dir / "assets" / "media" / "MediaPreviewPack"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name

    # Shot durations
    shot_dur_ms, dur_source = detect_shot_durations(ep_dir, locale)
    print(f"  [dur] source: {dur_source}")

    # Load RenderPlan for VO line timing (if available)
    rp_path = ep_dir / f"RenderPlan.{locale}.json"
    rp_shots = {}  # { shot_id: shot_dict }
    rp_data = None
    if rp_path.exists():
        rp_data = load_json(rp_path)
        for s in rp_data.get("shots", []):
            rp_shots[s["shot_id"]] = s

    # Build ordered shot list — prefer ShotList order (matches UI tab display).
    # detect_shot_durations already loaded ShotList and preserves its insertion
    # order in shot_dur_ms (Python 3.7+ dicts maintain insertion order), so
    # list(shot_dur_ms.keys()) gives ShotList order when ShotList was the source.
    # Fall back to RenderPlan order if ShotList was unavailable.
    if "ShotList" in dur_source:
        ordered_shots = list(shot_dur_ms.keys())
    elif rp_data:
        ordered_shots = [s["shot_id"] for s in rp_data.get("shots", [])]
    else:
        ordered_shots = list(shot_dur_ms.keys())

    # Apply shot_ids filter (per-scene preview)
    if shot_ids_filter is not None:
        _filter_set = set(shot_ids_filter)
        ordered_shots = [s for s in ordered_shots if s in _filter_set]
        print(f"  [filter] Rendering {len(ordered_shots)} shot(s): {ordered_shots}")

    # VO items by shot — derive from ShotList vo_item_ids and place lines
    # sequentially using WAV durations. This matches the music_review_pack.py
    # fallback so VO is always audible even before Stage 10 (RenderPlan) has run.
    vo_items_by_shot = {}  # { shot_id: [ {item_id, timeline_in_ms} ] }

    vo_dir = ep_dir / "assets" / locale / "audio" / "vo"

    _DEFAULT_VO_PAUSE_MS = 300  # 0.3 s between lines when no timing
    shot_path = ep_dir / "ShotList.json"
    if shot_path.exists():
        sl = load_json(shot_path)
        for shot in sl.get("shots", []):
            sid = shot.get("shot_id", "")
            cursor_ms = 0
            for iid in shot.get("audio_intent", {}).get("vo_item_ids", []):
                wav = vo_dir / f"{iid}.wav"
                dur_ms = 0
                if wav.exists():
                    try:
                        import wave
                        with wave.open(str(wav)) as wf:
                            dur_ms = int(wf.getnframes() / wf.getframerate() * 1000)
                    except Exception:
                        dur_ms = 2000  # safe fallback if unreadable
                vo_items_by_shot.setdefault(sid, []).append({
                    "item_id": iid,
                    "timeline_in_ms": cursor_ms,
                })
                cursor_ms += dur_ms + _DEFAULT_VO_PAUSE_MS
        print(f"  [vo] Placed {sum(len(v) for v in vo_items_by_shot.values())} VO lines sequentially from ShotList")

    # Load MusicApprovalSnapshot or MusicPlan for music
    music_data = None
    snap_path = ep_dir / "assets" / "music" / "MusicApprovalSnapshot.json"
    music_plan_path = ep_dir / "MusicPlan.json"
    if include_music:
        if snap_path.exists():
            music_data = {"type": "snapshot", "data": load_json(snap_path)}
            print("  [music] Using MusicApprovalSnapshot.json")
        elif music_plan_path.exists():
            music_data = {"type": "plan", "data": load_json(music_plan_path)}
            print("  [music] Using MusicPlan.json (fallback)")
        else:
            print("  [music] WARNING: No music source found — music track skipped")

    # Load SfxPlan for SFX
    sfx_index = {}  # { shot_id: [entry] }
    if include_sfx:
        sfx_path = ep_dir / "SfxPlan.json"
        if sfx_path.exists():
            sfx_plan = load_json(sfx_path)
            for entry in sfx_plan.get("sfx_entries", sfx_plan.get("entries", [])):
                sfx_index.setdefault(entry.get("shot_id", ""), []).append(entry)
            n_sfx = sum(len(v) for v in sfx_index.values())
            print(f"  [sfx] Loaded SfxPlan with {n_sfx} entries")
        else:
            print("  [sfx] WARNING: SfxPlan.json not found — SFX track skipped")

    # Generate per-shot video clips in a temp dir, then concatenate
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        clip_files = []

        for i, shot_id in enumerate(ordered_shots):
            dur_ms = shot_dur_ms.get(shot_id, 0)
            if dur_ms <= 0:
                print(f"  [skip] {shot_id}: duration 0ms")
                continue
            dur_sec = dur_ms / 1000.0
            clip_path = tmp / f"clip_{i:04d}_{shot_id}.mp4"

            segs = selections.get(shot_id, [])
            scale_pad = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
                         f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1")

            if not segs:
                # Black background for full shot duration
                cmd = [
                    "ffmpeg", "-y", "-f", "lavfi",
                    "-i", f"color=c=black:size={W}x{H}:rate={FPS}:duration={dur_sec:.3f}",
                    "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
                    "-t", f"{dur_sec:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(clip_path)
                ]
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    print(f"  [warn] black clip failed for {shot_id}: {r.stderr[-200:]}")
                    continue
                clip_files.append(clip_path)
                print(f"  [clip] {shot_id}: {dur_sec:.3f}s ok (black)")
            else:
                # Build one sub-clip per segment, then concat into the shot clip.
                # Each segment plays EXACTLY its designated duration.
                # After all segments, remaining shot time is filled with black.
                # The pipeline never adjusts or re-scales user-set durations.

                sub_clip_paths = []
                remaining = dur_sec  # tracks how much shot time is left
                segs_rendered = 0   # counts segments that actually produced a clip

                for seg_idx, seg in enumerate(segs):
                    media_path = url_to_path(seg.get("url", ""))
                    media_type = seg.get("media_type", "image")
                    seg_start  = float(seg.get("start_sec") or 0)
                    anim_type  = (seg.get("animation_type") or "none").lower()
                    seg_dur    = _seg_display_dur(seg)
                    print(f"  [seg] {shot_id} seg{seg_idx} type={media_type} "
                          f"hold_sec={seg.get('hold_sec')} start_sec={seg.get('start_sec')} "
                          f"end_sec={seg.get('end_sec')} duration_sec={seg.get('duration_sec')} "
                          f"→ seg_dur={seg_dur:.3f}s")

                    if seg_dur <= 0:
                        print(f"  [warn] {shot_id} seg {seg_idx} ({media_type}): duration is 0 — skipped")
                        continue

                    if remaining <= 0:
                        print(f"  [warn] {shot_id} seg {seg_idx} ({media_type}): no remaining shot time — skipped")
                        break

                    # Cap to remaining shot time (don't exceed total shot duration)
                    if seg_dur > remaining + 0.001:
                        print(f"  [warn] {shot_id} seg {seg_idx}: seg_dur {seg_dur:.3f}s exceeds remaining {remaining:.3f}s — capped")
                        seg_dur = remaining

                    sub_clip = tmp / f"clip_{i:04d}_{shot_id}_s{seg_idx}.mp4"

                    if media_type == "image":
                        zoompan = _anim_vf(anim_type, seg_dur)
                        vf = (scale_pad + "," + zoompan) if zoompan else scale_pad
                        cmd = [
                            "ffmpeg", "-y",
                            "-loop", "1", "-framerate", str(FPS),
                            "-t", f"{seg_dur:.3f}", "-i", media_path,
                            "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
                            "-vf", vf, "-r", str(FPS), "-pix_fmt", "yuv420p",
                            "-t", f"{seg_dur:.3f}",
                            "-c:v", "libx264", "-c:a", "aac",
                            "-shortest", str(sub_clip)
                        ]
                    else:
                        # Video: trim start_sec → start_sec+seg_dur; normalise to FPS/yuv420p
                        # so all sub-clips have identical codec params for concat
                        cmd = [
                            "ffmpeg", "-y",
                            "-ss", f"{seg_start:.3f}", "-t", f"{seg_dur:.3f}",
                            "-i", media_path,
                            "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
                            "-vf", scale_pad,
                            "-r", str(FPS), "-pix_fmt", "yuv420p",
                            "-t", f"{seg_dur:.3f}",
                            "-c:v", "libx264", "-c:a", "aac",
                            "-shortest", str(sub_clip)
                        ]

                    r = subprocess.run(cmd, capture_output=True, text=True)
                    if r.returncode != 0:
                        print(f"  [warn] seg {seg_idx} ({media_type}) failed for {shot_id}: {r.stderr[-300:]}")
                        # Fallback: black sub-clip for this segment's duration
                        cmd2 = [
                            "ffmpeg", "-y", "-f", "lavfi",
                            "-i", f"color=c=black:size={W}x{H}:rate={FPS}:duration={seg_dur:.3f}",
                            "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
                            "-t", f"{seg_dur:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-shortest", str(sub_clip)
                        ]
                        r2 = subprocess.run(cmd2, capture_output=True, text=True)
                        if r2.returncode != 0:
                            print(f"  [warn] {shot_id} seg {seg_idx}: black fallback also failed — segment skipped")
                            continue
                        print(f"  [warn] {shot_id} seg {seg_idx}: used black fallback (source failed)")

                    sub_clip_paths.append(sub_clip)
                    remaining -= seg_dur
                    segs_rendered += 1

                # Fill remaining shot time with black
                if remaining > 0.05:
                    fill_clip = tmp / f"clip_{i:04d}_{shot_id}_fill.mp4"
                    cmd_fill = [
                        "ffmpeg", "-y", "-f", "lavfi",
                        "-i", f"color=c=black:size={W}x{H}:rate={FPS}:duration={remaining:.3f}",
                        "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
                        "-t", f"{remaining:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", str(fill_clip)
                    ]
                    r_fill = subprocess.run(cmd_fill, capture_output=True, text=True)
                    if r_fill.returncode == 0:
                        sub_clip_paths.append(fill_clip)
                        print(f"  [fill] {shot_id}: black fill {remaining:.3f}s (segments used {dur_sec - remaining:.3f}s of {dur_sec:.3f}s)")
                    else:
                        print(f"  [warn] {shot_id}: black fill failed: {r_fill.stderr[-200:]}")

                if not sub_clip_paths:
                    print(f"  [skip] {shot_id}: all segments failed")
                    continue

                if len(sub_clip_paths) == 1:
                    clip_path = sub_clip_paths[0]
                else:
                    # Concat sub-clips into one shot clip — re-encode to normalise fps/pix_fmt
                    # (sub-clips may mix 25fps images with source-fps videos; -c copy would fail)
                    sub_list = tmp / f"sublist_{i:04d}.txt"
                    with open(sub_list, "w") as f:
                        for sc in sub_clip_paths:
                            f.write(f"file '{sc}'\n")
                    r = subprocess.run([
                        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                        "-i", str(sub_list),
                        "-c:v", "libx264", "-r", str(FPS), "-pix_fmt", "yuv420p",
                        "-c:a", "aac",
                        str(clip_path)
                    ], capture_output=True, text=True)
                    if r.returncode != 0:
                        print(f"  [warn] sub-concat failed for {shot_id}: {r.stderr[-400:]}")
                        # Fallback: black clip for full shot duration (do NOT silently drop segments)
                        cmd_blk = [
                            "ffmpeg", "-y", "-f", "lavfi",
                            "-i", f"color=c=black:size={W}x{H}:rate={FPS}:duration={dur_sec:.3f}",
                            "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
                            "-t", f"{dur_sec:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-shortest", str(clip_path)
                        ]
                        r_blk = subprocess.run(cmd_blk, capture_output=True, text=True)
                        if r_blk.returncode != 0:
                            print(f"  [warn] {shot_id}: sub-concat black fallback also failed — shot skipped: {r_blk.stderr[-200:]}")
                            continue
                        print(f"  [warn] {shot_id}: sub-concat fallback — showing black for full shot")

                clip_files.append(clip_path)
                segs_used = dur_sec - remaining
                skipped = len(segs) - segs_rendered
                skip_note = f" ({skipped} skipped)" if skipped else ""
                print(f"  [clip] {shot_id}: {segs_rendered} seg(s){skip_note} {segs_used:.3f}s, fill {remaining:.3f}s, total {dur_sec:.3f}s ok")

        if not clip_files:
            print("ERROR: No clips generated.")
            sys.exit(1)

        # Concatenate clips
        concat_list = tmp / "concat.txt"
        with open(concat_list, "w") as f:
            for cp in clip_files:
                f.write(f"file '{cp}'\n")

        concat_video = tmp / "concat.mp4"
        result = subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy", str(concat_video)
        ], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR: concat failed: {result.stderr[-500:]}")
            sys.exit(1)

        # Build VO audio track
        total_dur = sum(shot_dur_ms.get(s, 0) for s in ordered_shots) / 1000.0
        vo_audio = tmp / "vo_mix.wav"

        vo_inputs = []
        vo_delays = []
        cumulative_ms = 0
        for shot_id in ordered_shots:
            dur_ms = shot_dur_ms.get(shot_id, 0)
            vo_lines = vo_items_by_shot.get(shot_id, [])
            for line in vo_lines:
                wav_path = vo_dir / f"{line['item_id']}.wav"
                if wav_path.exists():
                    vo_inputs.append(str(wav_path))
                    delay_ms = cumulative_ms + line["timeline_in_ms"]
                    vo_delays.append(delay_ms)
            cumulative_ms += dur_ms

        if vo_inputs:
            # Build adelay filter for VO mixing
            filter_parts = []
            for idx, delay in enumerate(vo_delays):
                filter_parts.append(f"[{idx}]adelay={delay}|{delay}[d{idx}]")
            mix_inputs = "".join(f"[d{i}]" for i in range(len(vo_inputs)))
            filter_str = (";".join(filter_parts)
                          + f";{mix_inputs}amix=inputs={len(vo_inputs)}:normalize=0,apad=pad_dur={total_dur:.3f}[out]")

            cmd = ["ffmpeg", "-y"]
            for inp in vo_inputs:
                cmd += ["-i", inp]
            cmd += [
                "-filter_complex", filter_str,
                "-map", "[out]",
                "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS),
                "-t", str(total_dur),
                str(vo_audio)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  [warn] VO mix failed: {result.stderr[-300:]}")
                build_silent_audio(total_dur, vo_audio)
        else:
            print("  [vo] No VO WAV files found — silent VO track")
            build_silent_audio(total_dur, vo_audio)

        # ── Build music audio track ───────────────────────────────────────────
        music_audio = None
        if music_data:
            snap_shots = music_data["data"].get("shots", [])
            music_by_shot = {s["shot_id"]: s for s in snap_shots if "shot_id" in s}

            m_inputs, m_delays, m_volumes = [], [], []
            cum_ms = 0
            for shot_id in ordered_shots:
                dur_ms = shot_dur_ms.get(shot_id, 0)
                entry = music_by_shot.get(shot_id)
                if entry:
                    wav = entry.get("loop_wav_path", "")
                    if wav and Path(wav).exists():
                        delay_ms = cum_ms + float(entry.get("music_delay_sec", 0.0)) * 1000
                        base_db  = float(entry.get("base_db", -6.0))
                        m_inputs.append(wav)
                        m_delays.append(delay_ms)
                        m_volumes.append(10 ** (base_db / 20.0))
                cum_ms += dur_ms

            if m_inputs:
                music_audio = tmp / "music_mix.wav"
                f_parts = [
                    f"[{i}]adelay={d:.0f}|{d:.0f},volume={v:.4f}[m{i}]"
                    for i, (d, v) in enumerate(zip(m_delays, m_volumes))
                ]
                mix_ins = "".join(f"[m{i}]" for i in range(len(m_inputs)))
                f_str = ";".join(f_parts) + f";{mix_ins}amix=inputs={len(m_inputs)}:normalize=0,apad=pad_dur={total_dur:.3f}[out]"
                cmd = ["ffmpeg", "-y"]
                for inp in m_inputs:
                    cmd += ["-i", inp]
                cmd += ["-filter_complex", f_str, "-map", "[out]",
                        "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS),
                        "-t", str(total_dur), str(music_audio)]
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode != 0:
                    print(f"  [warn] Music mix failed: {res.stderr[-300:]}")
                    music_audio = None
                else:
                    print(f"  [music] Mixed {len(m_inputs)} music clip(s)")
            else:
                print("  [music] No music WAV files found for these shots — music skipped")

        # ── Build SFX audio track ─────────────────────────────────────────────
        sfx_audio = None
        if sfx_index:
            s_inputs, s_delays = [], []
            cum_ms = 0
            for shot_id in ordered_shots:
                dur_ms = shot_dur_ms.get(shot_id, 0)
                for entry in sfx_index.get(shot_id, []):
                    src = entry.get("source_file") or entry.get("local_path") or ""
                    if src and Path(src).exists():
                        delay_ms = cum_ms + float(entry.get("start_sec", 0)) * 1000
                        s_inputs.append(src)
                        s_delays.append(delay_ms)
                cum_ms += dur_ms

            if s_inputs:
                sfx_audio = tmp / "sfx_mix.wav"
                f_parts = [f"[{i}]adelay={d:.0f}|{d:.0f}[s{i}]" for i, d in enumerate(s_delays)]
                mix_ins = "".join(f"[s{i}]" for i in range(len(s_inputs)))
                f_str = ";".join(f_parts) + f";{mix_ins}amix=inputs={len(s_inputs)}:normalize=0,apad=pad_dur={total_dur:.3f}[out]"
                cmd = ["ffmpeg", "-y"]
                for inp in s_inputs:
                    cmd += ["-i", inp]
                cmd += ["-filter_complex", f_str, "-map", "[out]",
                        "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS),
                        "-t", str(total_dur), str(sfx_audio)]
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode != 0:
                    print(f"  [warn] SFX mix failed: {res.stderr[-300:]}")
                    sfx_audio = None
                else:
                    print(f"  [sfx] Mixed {len(s_inputs)} SFX clip(s)")
            else:
                print("  [sfx] No SFX WAV files found for these shots — SFX skipped")

        # ── Combine VO + music + SFX into final audio ─────────────────────────
        extra_tracks = [a for a in [music_audio, sfx_audio] if a and a.exists()]
        if extra_tracks:
            final_audio = tmp / "final_audio.wav"
            all_tracks = [str(vo_audio)] + [str(a) for a in extra_tracks]
            n = len(all_tracks)
            mix_ins = "".join(f"[{i}]" for i in range(n))
            f_str = f"{mix_ins}amix=inputs={n}:normalize=0,apad=pad_dur={total_dur:.3f}[out]"
            cmd = ["ffmpeg", "-y"]
            for t in all_tracks:
                cmd += ["-i", t]
            cmd += ["-filter_complex", f_str, "-map", "[out]",
                    "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS),
                    "-t", str(total_dur), str(final_audio)]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"  [warn] Final audio combine failed: {res.stderr[-300:]}")
                final_audio = vo_audio  # fall back to VO-only
            else:
                print(f"  [audio] Combined {n} track(s): VO"
                      + (" + Music" if music_audio else "")
                      + (" + SFX"   if sfx_audio   else ""))
        else:
            final_audio = vo_audio

        # ── Mux video + final audio ───────────────────────────────────────────
        result = subprocess.run([
            "ffmpeg", "-y",
            "-i", str(concat_video),
            "-i", str(final_audio),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(out_path)
        ], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR: final mux failed: {result.stderr[-500:]}")
            sys.exit(1)

    print(f"  [done] Preview written: {out_path}")


if __name__ == "__main__":
    main()
