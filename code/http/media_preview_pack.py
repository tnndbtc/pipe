#!/usr/bin/env python3
"""
media_preview_pack.py — Generate a preview video for the Media tab.

Single source of truth: VOPlan.{locale}.json
  - total_dur   = max(vo_item.end_sec + pause_after_ms/1000)
  - scene slots = derived from vo_item start_sec + scene_heads offsets
  - VO delays   = vo_item.start_sec (episode-absolute, already correct for aware VOPlans)
  - Music/SFX   = MusicPlan/SfxPlan start_sec (episode-absolute)

ShotList.json is NOT read. Shot slot durations come from VOPlan scene windows.
"""

import argparse
import json
import os
import re
import sys
import tempfile
import subprocess
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse, unquote

PIPE_DIR    = Path(__file__).resolve().parent.parent.parent
W, H, FPS   = 1280, 720, 25
SAMPLE_RATE = 44100
CHANNELS    = 2


# ── helpers ───────────────────────────────────────────────────────────────────

def url_to_path(url: str) -> str:
    if url.startswith("file://"):
        return unquote(urlparse(url).path)
    return url


def resolve_media_path(seg: dict, ep_dir: Path) -> str:
    p = seg.get("path", "")
    if p:
        if os.path.isabs(p):
            if os.path.exists(p):
                return p
        else:
            for base in (ep_dir, PIPE_DIR):
                c = base / p
                if c.exists():
                    return str(c)
    return url_to_path(seg.get("url", ""))


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_silent_audio(duration_sec: float, out_path: Path) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
        "-t", str(duration_sec), str(out_path)
    ], capture_output=True, check=True)


def _anim_vf(anim_type: str, clip_dur: float) -> str:
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
    return ""


def _seg_display_dur(seg: dict) -> float:
    media_type = seg.get("media_type", "image")
    if media_type != "image":
        start = float(seg.get("clip_in")  if seg.get("clip_in")  is not None else (seg.get("start_sec") or 0))
        end   = float(seg.get("clip_out") if seg.get("clip_out") is not None else (seg.get("end_sec")   or 0))
        if end > start:
            return end - start
    hold = seg.get("hold_sec")
    if hold:
        return float(hold)
    dur = seg.get("duration_override_sec") or seg.get("duration_sec")
    if dur:
        return float(dur)
    return 0.0


def _scene_id_from_shot_id(shot_id: str) -> str:
    """'sc01-sh01' → 'sc01'"""
    m = re.match(r'^(sc\d+)', shot_id)
    return m.group(1) if m else ""


def _scene_id_from_item_id(item_id: str) -> str:
    """'vo-sc01-001' → 'sc01'"""
    m = re.search(r'(sc\d+)', item_id)
    return m.group(1) if m else ""


def _shot_sort_key(shot_id: str):
    return tuple(int(n) for n in re.findall(r'\d+', shot_id))


# ── scene timeline builder ────────────────────────────────────────────────────

def build_scene_timeline(vo_items: list, scene_heads: dict, voplan_total_dur: float):
    """
    Returns:
      scene_slots      : { scene_id: (start_sec, dur_sec) }  episode-absolute
      video_total_dur  : float  (may exceed voplan_total_dur when heads are unaware)
      vo_head_offsets  : { scene_id: float }  ms to ADD to vi.start_sec for VO delay
                         0.0 for all scenes when VOPlan is aware (heads already embedded)

    Aware VOPlan: scene_heads are already baked into vo_item.start_sec.
      Detected when every scene-with-head has first_vo.start_sec >= head - 0.1s.

    Unaware VOPlan: vo_item.start_sec values are head-naive.
      Scene slots are extended by head_sec; VO delays get a cumulative head offset added.
    """
    # Group vo_items by scene, find the first (lowest start_sec) per scene.
    scene_first_vo: dict = {}
    for vi in sorted(vo_items, key=lambda v: v.get("start_sec", 0)):
        scid = _scene_id_from_item_id(vi.get("item_id", ""))
        if scid and scid not in scene_first_vo:
            scene_first_vo[scid] = vi

    scenes_in_order = sorted(
        scene_first_vo.keys(),
        key=lambda s: scene_first_vo[s]["start_sec"]
    )

    # Detect aware vs unaware.
    _aware = not any(
        scene_first_vo[scid]["start_sec"] < (head - 0.1)
        for scid, head in scene_heads.items()
        if scid in scene_first_vo
    )
    print(f"  [scene_heads] mode: {'aware' if _aware else 'unaware'}")

    scene_slots: dict    = {}
    vo_head_offsets: dict = {}

    if _aware:
        # VO start_sec already embeds the head offset.
        # scene_start = first_vo.start_sec - head
        for i, scid in enumerate(scenes_in_order):
            head     = scene_heads.get(scid, 0.0)
            sc_start = max(0.0, scene_first_vo[scid]["start_sec"] - head)
            if i + 1 < len(scenes_in_order):
                next_scid = scenes_in_order[i + 1]
                next_head = scene_heads.get(next_scid, 0.0)
                sc_end    = max(sc_start, scene_first_vo[next_scid]["start_sec"] - next_head)
            else:
                sc_end = voplan_total_dur
            scene_slots[scid]     = (sc_start, max(0.0, sc_end - sc_start))
            vo_head_offsets[scid] = 0.0  # already embedded

        video_total_dur = voplan_total_dur

    else:
        # VO start_sec values are head-naive; reconstruct episode-absolute positions.
        ep_cursor      = 0.0
        cum_head_added = 0.0
        for i, scid in enumerate(scenes_in_order):
            head     = scene_heads.get(scid, 0.0)
            sc_start = ep_cursor
            cum_head_added += head
            vo_head_offsets[scid] = cum_head_added  # offset to apply to vi.start_sec

            if i + 1 < len(scenes_in_order):
                next_scid    = scenes_in_order[i + 1]
                natural_dur  = (scene_first_vo[next_scid]["start_sec"]
                                - scene_first_vo[scid]["start_sec"])
            else:
                natural_dur = voplan_total_dur - scene_first_vo[scid]["start_sec"]

            sc_dur = natural_dur + head
            scene_slots[scid] = (sc_start, max(0.0, sc_dur))
            ep_cursor = sc_start + sc_dur

        video_total_dur = ep_cursor  # voplan_total_dur + sum(heads)

    for scid in scenes_in_order:
        s, d = scene_slots[scid]
        print(f"  [scene] {scid}: {s:.3f}s – {s+d:.3f}s  dur={d:.3f}s  "
              f"head={scene_heads.get(scid, 0.0):.1f}s  vo_offset={vo_head_offsets[scid]:.1f}s")

    return scene_slots, video_total_dur, vo_head_offsets


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    cfg             = load_json(Path(args.input))
    ep_dir          = Path(cfg["ep_dir"])
    locale          = cfg.get("locale", "en")
    selections      = cfg.get("selections", {})
    include_music   = cfg.get("include_music", False)
    include_sfx     = cfg.get("include_sfx", False)
    shot_ids_filter = cfg.get("shot_ids", None)
    out_name        = cfg.get("out_name", None) or "preview_video.mp4"

    print(f"  [media_preview] ep_dir={ep_dir} locale={locale}")
    print(f"  [media_preview] shots with selections: {len(selections)}")
    print(f"  [media_preview] include_music={include_music} include_sfx={include_sfx}")

    out_dir = ep_dir / "assets" / "media" / "MediaPreviewPack"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name

    # ── 1. VOPlan — single source of truth ────────────────────────────────────
    vp_path = ep_dir / f"VOPlan.{locale}.json"
    if not vp_path.exists():
        print(f"ERROR: VOPlan not found: {vp_path}")
        sys.exit(1)

    vp          = load_json(vp_path)
    vo_items    = [v for v in vp.get("vo_items", []) if v.get("end_sec") is not None]
    scene_heads = {k: float(v) for k, v in vp.get("scene_heads", {}).items() if float(v) > 0}

    if not vo_items:
        print("ERROR: VOPlan has no vo_items with end_sec.")
        sys.exit(1)

    voplan_total_dur = max(
        v["end_sec"] + (v.get("pause_after_ms") or 0) / 1000.0
        for v in vo_items
    )
    print(f"  [dur] voplan_total_dur={voplan_total_dur:.3f}s  scene_heads={scene_heads}")

    # ── 2. Build scene timeline from VOPlan ───────────────────────────────────
    scene_slots, video_total_dur, vo_head_offsets = build_scene_timeline(
        vo_items, scene_heads, voplan_total_dur
    )

    # ── 3. Shot list and per-shot durations ───────────────────────────────────
    # Shots come from selections keys; ordering inferred from shot_id naming.
    all_shot_ids = sorted(selections.keys(), key=_shot_sort_key)

    if shot_ids_filter is not None:
        _fs          = set(shot_ids_filter)
        all_shot_ids = [s for s in all_shot_ids if s in _fs]
        print(f"  [filter] Rendering shots: {all_shot_ids}")

    # Group shots by scene; split scene slot equally among shots in that scene.
    shots_by_scene: dict = defaultdict(list)
    for sid in all_shot_ids:
        scid = _scene_id_from_shot_id(sid)
        shots_by_scene[scid].append(sid)

    shot_dur_sec: dict = {}
    for scid, shot_list in shots_by_scene.items():
        if scid not in scene_slots:
            print(f"  [warn] No scene slot for scene '{scid}' — shots {shot_list} skipped")
            continue
        _, sc_dur = scene_slots[scid]
        per_shot  = sc_dur / len(shot_list)
        for sid in shot_list:
            shot_dur_sec[sid] = per_shot

    ordered_shots = [s for s in all_shot_ids if s in shot_dur_sec]

    # Fallback: selections was empty — create one black placeholder shot per scene
    # so audio tracks (VO, music, SFX) still render correctly.
    if not ordered_shots:
        for scid in sorted(scene_slots.keys(), key=lambda s: scene_slots[s][0]):
            pid = f"{scid}-sh01"
            shot_dur_sec[pid] = scene_slots[scid][1]
            ordered_shots.append(pid)
        print(f"  [shots] No selections — rendering {len(ordered_shots)} black placeholder shot(s)")

    if not ordered_shots:
        print("ERROR: No renderable shots.")
        sys.exit(1)

    for sid in ordered_shots:
        print(f"  [shot] {sid}: {shot_dur_sec[sid]:.3f}s")

    # ── 4. Total duration for filtered (per-scene) preview ────────────────────
    if shot_ids_filter is not None and ordered_shots:
        _starts = [scene_slots[_scene_id_from_shot_id(s)][0] for s in ordered_shots]
        _ends   = [scene_slots[_scene_id_from_shot_id(s)][0]
                   + scene_slots[_scene_id_from_shot_id(s)][1] for s in ordered_shots]
        window_start    = min(_starts)
        video_total_dur = max(_ends) - window_start
        print(f"  [filter] window_start={window_start:.3f}s  total_dur={video_total_dur:.3f}s")
    else:
        window_start = 0.0

    total_dur = video_total_dur
    print(f"  [dur] total_dur (video)={total_dur:.3f}s")

    # ── 5. Build per-shot video clips ─────────────────────────────────────────
    vo_dir = ep_dir / "assets" / locale / "audio" / "vo"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp        = Path(tmp_dir)
        clip_files = []

        for i, shot_id in enumerate(ordered_shots):
            dur_sec   = shot_dur_sec[shot_id]
            segs      = selections.get(shot_id, [])
            scale_pad = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
                         f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1")
            clip_path = tmp / f"clip_{i:04d}_{shot_id}.mp4"

            if not segs:
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
                sub_clip_paths = []
                remaining      = dur_sec
                segs_rendered  = 0

                for seg_idx, seg in enumerate(segs):
                    media_path = resolve_media_path(seg, ep_dir)
                    media_type = seg.get("media_type", "image")
                    seg_start  = float(seg.get("clip_in") if seg.get("clip_in") is not None
                                       else (seg.get("start_sec") or 0))
                    anim_type  = (seg.get("animation_type") or "none").lower()
                    seg_dur    = _seg_display_dur(seg)
                    print(f"  [seg] {shot_id} seg{seg_idx} type={media_type} "
                          f"hold_sec={seg.get('hold_sec')} clip_in={seg.get('clip_in')} "
                          f"clip_out={seg.get('clip_out')} duration_sec={seg.get('duration_sec')} "
                          f"→ seg_dur={seg_dur:.3f}s")

                    if seg_dur <= 0:
                        print(f"  [warn] {shot_id} seg {seg_idx}: duration 0 — skipped")
                        continue
                    if remaining <= 0:
                        print(f"  [warn] {shot_id} seg {seg_idx}: no remaining shot time — skipped")
                        break
                    if seg_dur > remaining + 0.001:
                        print(f"  [warn] {shot_id} seg {seg_idx}: {seg_dur:.3f}s > remaining {remaining:.3f}s — capped")
                        seg_dur = remaining

                    sub_clip = tmp / f"clip_{i:04d}_{shot_id}_s{seg_idx}.mp4"

                    if media_type == "image":
                        zoompan = _anim_vf(anim_type, seg_dur)
                        vf      = (scale_pad + "," + zoompan) if zoompan else scale_pad
                        cmd     = [
                            "ffmpeg", "-y",
                            "-loop", "1", "-framerate", str(FPS),
                            "-t", f"{seg_dur:.3f}", "-i", media_path,
                            "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
                            "-vf", vf, "-r", str(FPS), "-pix_fmt", "yuv420p",
                            "-t", f"{seg_dur:.3f}", "-c:v", "libx264", "-c:a", "aac",
                            "-shortest", str(sub_clip)
                        ]
                    else:
                        cmd = [
                            "ffmpeg", "-y",
                            "-ss", f"{seg_start:.3f}", "-t", f"{seg_dur:.3f}",
                            "-i", media_path,
                            "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
                            "-vf", scale_pad, "-r", str(FPS), "-pix_fmt", "yuv420p",
                            "-t", f"{seg_dur:.3f}", "-c:v", "libx264", "-c:a", "aac",
                            "-shortest", str(sub_clip)
                        ]

                    r = subprocess.run(cmd, capture_output=True, text=True)
                    if r.returncode != 0:
                        print(f"  [warn] seg {seg_idx} ({media_type}) failed: {r.stderr[-300:]}")
                        cmd2 = [
                            "ffmpeg", "-y", "-f", "lavfi",
                            "-i", f"color=c=black:size={W}x{H}:rate={FPS}:duration={seg_dur:.3f}",
                            "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
                            "-t", f"{seg_dur:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-shortest", str(sub_clip)
                        ]
                        r2 = subprocess.run(cmd2, capture_output=True, text=True)
                        if r2.returncode != 0:
                            print(f"  [warn] {shot_id} seg {seg_idx}: black fallback failed — skipped")
                            continue
                        print(f"  [warn] {shot_id} seg {seg_idx}: black fallback used")

                    sub_clip_paths.append(sub_clip)
                    remaining     -= seg_dur
                    segs_rendered += 1

                # Fill remaining shot time with black
                if remaining > 0.05:
                    fill_clip = tmp / f"clip_{i:04d}_{shot_id}_fill.mp4"
                    cmd_fill  = [
                        "ffmpeg", "-y", "-f", "lavfi",
                        "-i", f"color=c=black:size={W}x{H}:rate={FPS}:duration={remaining:.3f}",
                        "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
                        "-t", f"{remaining:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", str(fill_clip)
                    ]
                    r_fill = subprocess.run(cmd_fill, capture_output=True, text=True)
                    if r_fill.returncode == 0:
                        sub_clip_paths.append(fill_clip)
                        print(f"  [fill] {shot_id}: black fill {remaining:.3f}s")
                    else:
                        print(f"  [warn] {shot_id}: black fill failed: {r_fill.stderr[-200:]}")

                if not sub_clip_paths:
                    print(f"  [skip] {shot_id}: all segments failed")
                    continue

                if len(sub_clip_paths) == 1:
                    clip_path = sub_clip_paths[0]
                else:
                    sub_list = tmp / f"sublist_{i:04d}.txt"
                    with open(sub_list, "w") as f:
                        for sc in sub_clip_paths:
                            f.write(f"file '{sc}'\n")
                    r = subprocess.run([
                        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                        "-i", str(sub_list),
                        "-c:v", "libx264", "-r", str(FPS), "-pix_fmt", "yuv420p",
                        "-c:a", "aac", str(clip_path)
                    ], capture_output=True, text=True)
                    if r.returncode != 0:
                        print(f"  [warn] sub-concat failed for {shot_id}: {r.stderr[-400:]}")
                        cmd_blk = [
                            "ffmpeg", "-y", "-f", "lavfi",
                            "-i", f"color=c=black:size={W}x{H}:rate={FPS}:duration={dur_sec:.3f}",
                            "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
                            "-t", f"{dur_sec:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-shortest", str(clip_path)
                        ]
                        r_blk = subprocess.run(cmd_blk, capture_output=True, text=True)
                        if r_blk.returncode != 0:
                            print(f"  [warn] {shot_id}: fallback also failed — shot skipped")
                            continue
                        print(f"  [warn] {shot_id}: sub-concat fallback — showing black")

                clip_files.append(clip_path)
                segs_used  = dur_sec - remaining
                skipped    = len(segs) - segs_rendered
                skip_note  = f" ({skipped} skipped)" if skipped else ""
                print(f"  [clip] {shot_id}: {segs_rendered} seg(s){skip_note} "
                      f"{segs_used:.3f}s media + {remaining:.3f}s fill = {dur_sec:.3f}s ok")

        if not clip_files:
            print("ERROR: No clips generated.")
            sys.exit(1)

        # ── 6. Concatenate shot clips ──────────────────────────────────────────
        concat_list = tmp / "concat.txt"
        with open(concat_list, "w") as f:
            for cp in clip_files:
                f.write(f"file '{cp}'\n")

        concat_video = tmp / "concat.mp4"
        result = subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list), "-c", "copy", str(concat_video)
        ], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR: concat failed: {result.stderr[-500:]}")
            sys.exit(1)

        # ── 7. Build VO audio ──────────────────────────────────────────────────
        # VO delays come directly from VOPlan vo_item.start_sec (episode-absolute)
        # plus any head offset needed for unaware VOPlans.
        # No cursor accumulation — no ShotList reads.
        vo_audio  = tmp / "vo_mix.wav"
        vo_inputs = []
        vo_delays = []
        window_start_ms = int(window_start * 1000)

        for vi in sorted(vo_items, key=lambda v: v.get("start_sec", 0)):
            wav_path = vo_dir / f"{vi['item_id']}.wav"
            if not wav_path.exists():
                continue
            scid       = _scene_id_from_item_id(vi.get("item_id", ""))
            head_off   = vo_head_offsets.get(scid, 0.0)
            delay_ms   = int((vi["start_sec"] + head_off) * 1000) - window_start_ms
            if delay_ms < 0:
                continue  # before the filtered window
            vo_inputs.append(str(wav_path))
            vo_delays.append(delay_ms)

        if vo_inputs:
            f_parts    = [f"[{idx}]adelay={d}|{d}[d{idx}]" for idx, d in enumerate(vo_delays)]
            mix_inputs = "".join(f"[d{i}]" for i in range(len(vo_inputs)))
            f_str      = (";".join(f_parts)
                          + f";{mix_inputs}amix=inputs={len(vo_inputs)}:normalize=0,"
                          f"apad=pad_dur={total_dur:.3f}[out]")
            cmd = ["ffmpeg", "-y"]
            for inp in vo_inputs:
                cmd += ["-i", inp]
            cmd += ["-filter_complex", f_str, "-map", "[out]",
                    "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS),
                    "-t", str(total_dur), str(vo_audio)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  [warn] VO mix failed: {result.stderr[-300:]}")
                build_silent_audio(total_dur, vo_audio)
            else:
                print(f"  [vo] Mixed {len(vo_inputs)} VO line(s)")
        else:
            print("  [vo] No VO WAV files — silent VO track")
            build_silent_audio(total_dur, vo_audio)

        # ── 8. Build music audio ───────────────────────────────────────────────
        # MusicPlan shot_overrides carry episode-absolute start_sec/end_sec.
        # No shot_id resolution needed — iterate overrides directly.
        music_audio = None
        if include_music:
            music_path = ep_dir / "MusicPlan.json"
            if music_path.exists():
                import re as _re
                music_data = load_json(music_path)
                _clip_vol  = music_data.get("clip_volumes", {})
                _track_vol = music_data.get("track_volumes", {})
                BASE_DB    = -6.0
                m_inputs, m_delays, m_volumes, m_clip_durs = [], [], [], []

                for ovr in music_data.get("shot_overrides", []):
                    _asset = ovr.get("music_asset_id", "")
                    _start = ovr.get("start_sec")
                    _end   = ovr.get("end_sec")
                    if _start is None:
                        continue
                    delay_ms = max(0.0, float(_start) - window_start) * 1000
                    clip_dur = (max(0.0, float(_end) - float(_start))
                                if _end is not None else None)

                    _wav_loop = ep_dir / "assets" / "music" / f"{_asset}.loop.wav"
                    _wav_base = ep_dir / "assets" / "music" / f"{_asset}.wav"
                    wav = (str(_wav_loop) if _wav_loop.exists()
                           else str(_wav_base) if _wav_base.exists() else "")
                    if not wav:
                        print(f"  [WARN] music wav not found: {_asset}")
                        continue

                    _stem   = _re.sub(r'_\d[\d_]*s-[\d_\.]+s$', '', _asset)
                    _db_off = _track_vol.get(_stem, 0.0) + _clip_vol.get(_asset, 0.0)
                    volume  = 10 ** ((BASE_DB + _db_off) / 20.0)
                    m_inputs.append(wav)
                    m_delays.append(delay_ms)
                    m_volumes.append(volume)
                    m_clip_durs.append(clip_dur)
                    print(f"  [music] {_asset}: delay={delay_ms:.0f}ms clip_dur={clip_dur}")

                if m_inputs:
                    music_audio = tmp / "music_mix.wav"
                    f_parts = []
                    for idx, (d, v, cd) in enumerate(zip(m_delays, m_volumes, m_clip_durs)):
                        if cd is not None:
                            f_parts.append(
                                f"[{idx}]atrim=duration={cd:.3f},volume={v:.4f},"
                                f"adelay={d:.0f}|{d:.0f}[m{idx}]")
                        else:
                            f_parts.append(
                                f"[{idx}]adelay={d:.0f}|{d:.0f},volume={v:.4f}[m{idx}]")
                    mix_ins = "".join(f"[m{i}]" for i in range(len(m_inputs)))
                    f_str   = (";".join(f_parts)
                               + f";{mix_ins}amix=inputs={len(m_inputs)}:normalize=0,"
                               f"apad=pad_dur={total_dur:.3f}[out]")
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
                    print("  [music] No music WAVs found — music skipped")
            else:
                print("  [music] MusicPlan.json not found — music skipped")

        # ── 9. Build SFX audio ─────────────────────────────────────────────────
        # SfxPlan entries carry episode-absolute start_sec/end_sec — use directly.
        sfx_audio = None
        if include_sfx:
            sfx_path = ep_dir / "SfxPlan.json"
            if sfx_path.exists():
                sfx_plan = load_json(sfx_path)
                s_inputs, s_delays, s_durs = [], [], []
                for entry in sfx_plan.get("sfx_entries", sfx_plan.get("entries", [])):
                    src = entry.get("source_file") or entry.get("local_path") or ""
                    if src and Path(src).exists():
                        start_sfx = float(entry.get("start_sec", 0))
                        end_sfx   = entry.get("end_sec")
                        delay_ms  = max(0.0, start_sfx - window_start) * 1000
                        clip_dur  = (max(0.0, float(end_sfx) - start_sfx)
                                     if end_sfx is not None else None)
                        s_inputs.append(src)
                        s_delays.append(delay_ms)
                        s_durs.append(clip_dur)

                if s_inputs:
                    sfx_audio = tmp / "sfx_mix.wav"
                    f_parts   = [
                        (f"[{i}]atrim=duration={dur:.3f},adelay={d:.0f}|{d:.0f}[s{i}]" if dur
                         else f"[{i}]adelay={d:.0f}|{d:.0f}[s{i}]")
                        for i, (d, dur) in enumerate(zip(s_delays, s_durs))
                    ]
                    mix_ins = "".join(f"[s{i}]" for i in range(len(s_inputs)))
                    f_str   = (";".join(f_parts)
                               + f";{mix_ins}amix=inputs={len(s_inputs)}:normalize=0,"
                               f"apad=pad_dur={total_dur:.3f}[out]")
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
                    print("  [sfx] No SFX WAVs found — SFX skipped")
            else:
                print("  [sfx] SfxPlan.json not found — SFX skipped")

        # ── 10. Combine audio tracks ───────────────────────────────────────────
        extra_tracks = [a for a in [music_audio, sfx_audio] if a and Path(a).exists()]
        if extra_tracks:
            final_audio = tmp / "final_audio.wav"
            all_tracks  = [str(vo_audio)] + [str(a) for a in extra_tracks]
            n           = len(all_tracks)
            mix_ins     = "".join(f"[{i}]" for i in range(n))
            f_str       = (f"{mix_ins}amix=inputs={n}:normalize=0,"
                           f"apad=pad_dur={total_dur:.3f}[out]")
            cmd = ["ffmpeg", "-y"]
            for t in all_tracks:
                cmd += ["-i", t]
            cmd += ["-filter_complex", f_str, "-map", "[out]",
                    "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS),
                    "-t", str(total_dur), str(final_audio)]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"  [warn] Audio combine failed: {res.stderr[-300:]}")
                final_audio = vo_audio
            else:
                print(f"  [audio] Combined {n} track(s): VO"
                      + (" + Music" if music_audio else "")
                      + (" + SFX"   if sfx_audio   else ""))
        else:
            final_audio = vo_audio

        # ── 11. Mux video + audio ──────────────────────────────────────────────
        result = subprocess.run([
            "ffmpeg", "-y",
            "-i", str(concat_video),
            "-i", str(final_audio),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-t", str(total_dur),
            str(out_path)
        ], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR: final mux failed: {result.stderr[-500:]}")
            sys.exit(1)

    print(f"  [done] Preview written: {out_path}")


if __name__ == "__main__":
    main()
