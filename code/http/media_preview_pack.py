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
    """Return the best local filesystem path for a MediaPlan segment.

    Priority: 'path' field (absolute or ep_dir-relative) → file:// url → url.
    When 'path' is set but the file is missing and 'url' is an HTTP URL,
    downloads the file to 'path' so the renderer can use it directly.
    """
    p = seg.get("path", "")
    if p:
        if os.path.isabs(p):
            if os.path.exists(p):
                return p
            # File missing — try to download from url
            url = seg.get("url", "")
            if url and not url.startswith("/serve_media") and (url.startswith("http://") or url.startswith("https://")):
                try:
                    import urllib.request as _urlreq
                    Path(p).parent.mkdir(parents=True, exist_ok=True)
                    print(f"  [resolve] downloading missing asset → {Path(p).name}")
                    _urlreq.urlretrieve(url, p)
                    if os.path.exists(p):
                        print(f"  [resolve] download ok: {p}")
                        return p
                except Exception as _dl_exc:
                    print(f"  [resolve] download failed ({url}): {_dl_exc}")
            return p  # missing — caller's existence check will log it
        else:
            for base in (ep_dir, PIPE_DIR):
                c = base / p
                if c.exists():
                    return str(c)
            return str(ep_dir / p)
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


def _scene_id_from_item_id(item_id: str) -> str:
    """'vo-sc01-001' → 'sc01'"""
    m = re.search(r'(sc\d+)', item_id)
    return m.group(1) if m else ""



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


# ── music ducking helper ──────────────────────────────────────────────────────

def _build_duck_vol_filter(vo_items: list, seg_start_ep: float, seg_end_ep: float,
                           duck_db: float, fade_sec: float,
                           base_db: float, vol_offset_db: float) -> str:
    """Return an FFmpeg volume filter string that ducks during VO lines.

    The returned string is used BEFORE adelay, so ``t`` in the expression is
    segment-relative (t=0 == seg_start_ep in the episode).

    When no VO lines overlap the segment a plain ``volume=<scalar>`` is returned.
    """
    base_amp = 10 ** (base_db / 20.0)
    duck_amp = base_amp * (10 ** (duck_db / 20.0))
    scale    = 10 ** (vol_offset_db / 20.0)

    intervals = []
    for vi in vo_items:
        vi_start = float(vi.get("start_sec") or 0)
        vi_end   = float(vi.get("end_sec")   or 0)
        if vi_end <= seg_start_ep or vi_start >= seg_end_ep:
            continue
        # Convert to segment-relative time; expand by fade_sec on each side
        t0 = max(0.0, vi_start - fade_sec - seg_start_ep)
        t1 = vi_end + fade_sec - seg_start_ep
        if t1 > t0:
            intervals.append((round(t0, 3), round(t1, 3)))

    if not intervals:
        return f"volume={base_amp * scale:.6f}"

    cond = "+".join(f"between(t,{t0},{t1})" for t0, t1 in intervals)
    expr = f"if(gt({cond},0),{duck_amp * scale:.6f},{base_amp * scale:.6f})"
    return f"volume=volume='{expr}':eval=frame"


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    cfg           = load_json(Path(args.input))
    ep_dir        = Path(cfg["ep_dir"])
    locale        = cfg.get("locale", "en")
    include_music = cfg.get("include_music", False)
    include_sfx   = cfg.get("include_sfx",  False)
    out_name      = cfg.get("out_name", None) or "preview_video.mp4"

    print(f"  [media_preview] ep_dir={ep_dir} locale={locale}")
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

    # ── 3. Load media segments from POST body (DOM state) ────────────────────
    # Segments come from _mediaSegments (live DOM state) sent by the frontend.
    # MediaPlan.json on disk is NOT read here — the caller sends current UI state.
    media_segments = cfg.get("media_segments", [])
    print(f"  [media] {len(media_segments)} segment(s) from POST body")

    # ── 4. Compute total video duration ───────────────────────────────────────
    window_start = 0.0

    def _mp_seg_dur(seg: dict) -> float:
        """Duration of one MediaPlan segment in seconds."""
        seg_type = seg.get("type", "image")
        if seg_type == "video":
            ci = float(seg.get("clip_in") or 0)
            co = seg.get("clip_out")
            if co is not None and float(co) > ci:
                return float(co) - ci
            dur = seg.get("duration_sec")
            if dur:
                return float(dur)
            return 0.0
        # image
        hold = seg.get("hold_sec")
        if hold:
            return float(hold)
        dur = seg.get("duration_sec")
        if dur:
            return float(dur)
        return 3.0  # default hold

    if media_segments:
        seg_total = sum(_mp_seg_dur(s) for s in media_segments)
        total_dur = max(seg_total, voplan_total_dur)
        print(f"  [dur] seg_total={seg_total:.3f}s  voplan={voplan_total_dur:.3f}s  total_dur={total_dur:.3f}s")
    else:
        total_dur = video_total_dur  # from VOPlan scene timeline
    print(f"  [dur] total_dur (video)={total_dur:.3f}s")

    # ── 5. Build video clips from flat MediaPlan segments ─────────────────────
    vo_dir    = ep_dir / "assets" / locale / "audio" / "vo"
    scale_pad = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
                 f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp        = Path(tmp_dir)
        clip_files = []

        if not media_segments:
            # No MediaPlan — single black clip for the full duration
            black_path = tmp / "clip_0000_black.mp4"
            cmd = [
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", f"color=c=black:size={W}x{H}:rate={FPS}:duration={total_dur:.3f}",
                "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
                "-t", f"{total_dur:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest", str(black_path)
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                clip_files.append(black_path)
                print(f"  [clip] black placeholder {total_dur:.3f}s ok")
            else:
                print(f"ERROR: black placeholder failed: {r.stderr[-300:]}")
                sys.exit(1)
        else:
            for seg_idx, seg in enumerate(media_segments):
                seg_type   = seg.get("type", "image")
                seg_dur    = _mp_seg_dur(seg)
                anim_type  = (seg.get("animation_type") or seg.get("animation") or "none").lower()
                media_path = resolve_media_path(seg, ep_dir)
                clip_path  = tmp / f"clip_{seg_idx:04d}.mp4"

                print(f"  [seg] {seg_idx} type={seg_type} dur={seg_dur:.3f}s "
                      f"path={str(media_path)[:80] if media_path else 'MISSING'}")

                if seg_dur <= 0:
                    print(f"  [warn] seg {seg_idx}: duration 0 — skipped")
                    continue

                if not media_path or not Path(media_path).exists():
                    print(f"  [warn] seg {seg_idx}: media not found '{media_path}' — black")
                    cmd = [
                        "ffmpeg", "-y", "-f", "lavfi",
                        "-i", f"color=c=black:size={W}x{H}:rate={FPS}:duration={seg_dur:.3f}",
                        "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
                        "-t", f"{seg_dur:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", str(clip_path)
                    ]
                    r = subprocess.run(cmd, capture_output=True, text=True)
                    if r.returncode == 0:
                        clip_files.append(clip_path)
                    continue

                if seg_type == "image":
                    zoompan = _anim_vf(anim_type, seg_dur)
                    vf      = (scale_pad + "," + zoompan) if zoompan else scale_pad
                    cmd     = [
                        "ffmpeg", "-y",
                        "-loop", "1", "-framerate", str(FPS),
                        "-t", f"{seg_dur:.3f}", "-i", media_path,
                        "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
                        "-vf", vf, "-r", str(FPS), "-pix_fmt", "yuv420p",
                        "-t", f"{seg_dur:.3f}", "-c:v", "libx264", "-c:a", "aac",
                        "-shortest", str(clip_path)
                    ]
                else:
                    seg_start = float(seg.get("clip_in") or 0)
                    cmd = [
                        "ffmpeg", "-y",
                        "-ss", f"{seg_start:.3f}", "-t", f"{seg_dur:.3f}",
                        "-i", media_path,
                        "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
                        "-vf", scale_pad, "-r", str(FPS), "-pix_fmt", "yuv420p",
                        "-t", f"{seg_dur:.3f}", "-c:v", "libx264", "-c:a", "aac",
                        "-shortest", str(clip_path)
                    ]

                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    print(f"  [warn] seg {seg_idx} ({seg_type}) failed: {r.stderr[-300:]}")
                    cmd2 = [
                        "ffmpeg", "-y", "-f", "lavfi",
                        "-i", f"color=c=black:size={W}x{H}:rate={FPS}:duration={seg_dur:.3f}",
                        "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
                        "-t", f"{seg_dur:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", str(clip_path)
                    ]
                    r2 = subprocess.run(cmd2, capture_output=True, text=True)
                    if r2.returncode == 0:
                        clip_files.append(clip_path)
                        print(f"  [warn] seg {seg_idx}: black fallback")
                    else:
                        print(f"  [warn] seg {seg_idx}: black fallback also failed — skipped")
                    continue

                clip_files.append(clip_path)
                print(f"  [clip] seg {seg_idx}: {seg_type} {seg_dur:.3f}s ok")

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
        vo_volumes = []
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
            vo_volumes.append(float(vi.get("volume_db", 0.0) or 0.0))

        if vo_inputs:
            f_parts = []
            for idx, (d, vdb) in enumerate(zip(vo_delays, vo_volumes)):
                if vdb != 0.0:
                    f_parts.append(f"[{idx}]volume={vdb}dB,adelay={d}|{d}[d{idx}]")
                else:
                    f_parts.append(f"[{idx}]adelay={d}|{d}[d{idx}]")
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

        # ── 8. Build music audio with VO ducking ───────────────────────────────
        # MusicPlan shot_overrides[] carry episode-absolute start_sec/end_sec and
        # duck_db.  Each segment is volume-automated so music dips during every VO
        # line (± fade_sec), matching the ducking in music_review_pack.py.
        music_audio = None
        if include_music:
            music_path = ep_dir / "MusicPlan.json"
            if music_path.exists():
                import re as _re
                music_data = load_json(music_path)
                _clip_vol  = music_data.get("clip_volumes", {})
                _track_vol = music_data.get("track_volumes", {})
                BASE_DB    = -6.0
                DEFAULT_DUCK_DB   = -12.0
                DEFAULT_FADE_SEC  =   0.5
                m_inputs, m_delays, m_clip_durs, m_vol_filters = [], [], [], []

                for ovr in music_data.get("shot_overrides", []):
                    _asset    = ovr.get("music_asset_id", "")
                    _start    = ovr.get("start_sec")
                    _end      = ovr.get("end_sec")
                    _duck_db  = float(ovr.get("duck_db",  DEFAULT_DUCK_DB))
                    _fade_sec = float(ovr.get("fade_sec", DEFAULT_FADE_SEC))
                    if _start is None:
                        continue
                    _start_f = float(_start)
                    _end_f   = float(_end) if _end is not None else total_dur
                    delay_ms = max(0.0, _start_f - window_start) * 1000
                    clip_dur = max(0.0, _end_f - _start_f) if _end is not None else None

                    _wav_loop = ep_dir / "assets" / "music" / f"{_asset}.loop.wav"
                    _wav_base = ep_dir / "assets" / "music" / f"{_asset}.wav"
                    wav = (str(_wav_loop) if _wav_loop.exists()
                           else str(_wav_base) if _wav_base.exists() else "")
                    if not wav:
                        print(f"  [WARN] music wav not found: {_asset}")
                        continue

                    _stem   = _re.sub(r'_\d[\d_]*s-[\d_\.]+s$', '', _asset)
                    _db_off = _track_vol.get(_stem, 0.0) + _clip_vol.get(_asset, 0.0)

                    vol_filter = _build_duck_vol_filter(
                        vo_items, _start_f, _end_f,
                        _duck_db, _fade_sec, BASE_DB, _db_off)

                    m_inputs.append(wav)
                    m_delays.append(delay_ms)
                    m_clip_durs.append(clip_dur)
                    m_vol_filters.append(vol_filter)
                    print(f"  [music] {_asset}: delay={delay_ms:.0f}ms "
                          f"clip_dur={clip_dur} duck_db={_duck_db}")

                if m_inputs:
                    music_audio = tmp / "music_mix.wav"
                    f_parts = []
                    for idx, (d, vf, cd) in enumerate(
                            zip(m_delays, m_vol_filters, m_clip_durs)):
                        if cd is not None:
                            f_parts.append(
                                f"[{idx}]atrim=duration={cd:.3f},{vf},"
                                f"adelay={d:.0f}|{d:.0f}[m{idx}]")
                        else:
                            f_parts.append(
                                f"[{idx}]{vf},adelay={d:.0f}|{d:.0f}[m{idx}]")
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
                        print(f"  [music] Mixed {len(m_inputs)} music clip(s) with ducking")
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
                # Build clip_id → absolute path lookup from cut_clips.
                # source_file in shot_overrides may be a bare clip_id string (e.g.
                # "ambulance_1.2s-3.5s") rather than a filesystem path — the SFX tab
                # stores the dropdown value (clip_id) there, not the WAV path.
                _cut_clip_abs = {}
                for _cc in sfx_plan.get("cut_clips", []):
                    _cid  = _cc.get("clip_id", "")
                    _crel = _cc.get("path", "")
                    if _cid and _crel:
                        _cabs = Path(_crel) if Path(_crel).is_absolute() else ep_dir / _crel
                        _cut_clip_abs[_cid] = str(_cabs)

                s_inputs, s_delays, s_durs = [], [], []
                for entry in sfx_plan.get("shot_overrides", []):
                    src = entry.get("source_file") or entry.get("local_path") or ""
                    # Resolve clip_id → absolute path when src is not a valid file path
                    if src and not Path(src).exists():
                        src = _cut_clip_abs.get(src, src)
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
