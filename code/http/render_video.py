#!/usr/bin/env python3
# =============================================================================
# render_video.py — Produce output.mp4 from VOPlan + MusicPlan + SfxPlan + MediaPlan
# =============================================================================
#
# ShotList-free renderer.  All timing is episode-absolute from plan files:
#   VOPlan.{locale}.json  — VO items with start_sec/end_sec, scene_heads
#   MediaPlan.json        — ordered visual segments (images/video clips)
#   MusicPlan.json        — music segments with episode-absolute start/end + duck params
#   SfxPlan.json          — SFX segments with episode-absolute start/end
#
# Pipeline:
#   1. Build video clip per MediaPlan segment → temp MP4 files
#   2. Concat clips with optional dissolve/flash transitions → concat.mp4
#   3. Mix VO audio from VOPlan (adelay episode-absolute) → vo_mix.wav
#   4. Mix music from MusicPlan (duck under VO via volume filter) → music_mix.wav
#   5. Mix SFX from SfxPlan → sfx_mix.wav
#   6. Combine video + all audio tracks → output.mp4
#   7. Write output.{locale}.srt + render_output.json
#
# Usage:
#   python render_video.py \
#       --plan projects/slug/ep/VOPlan.en.json
#
#   python render_video.py \
#       --plan   projects/slug/ep/VOPlan.en.json \
#       --locale en \
#       --out    projects/slug/ep/renders/en \
#       --profile preview_local \
#       --keep-intermediates \
#       --verbose
#
# Output (default next to the plan in renders/{locale}/):
#   output.mp4
#   output.srt
#   render_output.json
#
# Requirements: stdlib only + FFmpeg in PATH
# =============================================================================

import argparse
import datetime
import hashlib
import math
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

# resolve_assets lives alongside this file; insert its directory so it imports cleanly
# whether render_video is invoked directly or via subprocess from test_server.py.
sys.path.insert(0, str(Path(__file__).parent))
from resolve_assets import resolve_all as _ra_resolve_all  # noqa: E402

PRODUCER = "render_video.py"

# ── Frame geometry ─────────────────────────────────────────────────────────────
W, H   = 1280, 720      # output resolution
FPS    = 24             # frames per second

# ── Music transition durations ─────────────────────────────────────────────────
MUSIC_FADEOUT_SEC   = 0.5   # music → no-music fade (configurable via CLI)

# ── Encode profiles ────────────────────────────────────────────────────────────
PROFILES: dict[str, dict] = {
    "preview":       {"crf": 28, "preset": "medium"},
    "preview_local": {"crf": 28, "preset": "medium"},
    "high":          {"crf": 18, "preset": "slow"},
    "draft_720p":    {"crf": 28, "preset": "medium"},
}
DEFAULT_PROFILE = "preview_local"

# ── Audio constants ────────────────────────────────────────────────────────────
SAMPLE_RATE = 44100
CHANNELS    = 2


# ── I/O helpers ────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(doc: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")


def uri_to_path(uri: str) -> Path | None:
    """Convert a file:// URI to a local Path. Return None for placeholder:// URIs."""
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return None
    return Path(unquote(parsed.path))


# ── Shot-stitch transitions ────────────────────────────────────────────────────

# Durations (seconds) for each dissolve variant
_DISSOLVE_DURATIONS: dict[str, float] = {
    "dissolve_short":  0.3,
    "dissolve_medium": 0.5,
    "dissolve_long":   1.0,
}
# Duration of the inserted colour clip for hard-stop transitions
_FLASH_DURATIONS: dict[str, float] = {
    "fade_black":  0.5,
    "flash_white": 0.1,
}


def _make_color_clip(path: Path, color: str, dur: float, fps: int,
                     w: int, h: int, sample_rate: int) -> None:
    """Generate a solid-colour silent clip (used for fade-through-black / white flash)."""
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c={color}:size={w}x{h}:rate={fps}:duration={dur:.3f}",
        "-f", "lavfi", "-i", f"anullsrc=r={sample_rate}:cl=stereo",
        "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", str(sample_rate),
        "-shortest", str(path),
    ], capture_output=True, text=True, check=False)


def _concat_with_transitions(
    clip_files:  list,
    clip_durs:   list,
    clip_trans:  list,
    output:      Path,
    tmp:         Path,
    fps:         int,
    w:           int,
    h:           int,
    sample_rate: int,
) -> subprocess.CompletedProcess:
    """
    Join *clip_files* with per-boundary transitions.

    clip_trans[i]  = transition keyword at the start of clip i
                     (clip_trans[0] is always ignored — no "before first clip").
    Supported values:
        "none"           → hard cut  (stream-copy concat, fastest)
        "dissolve_short" → 0.3 s xfade/acrossfade
        "dissolve_medium"→ 0.5 s xfade/acrossfade
        "dissolve_long"  → 1.0 s xfade/acrossfade
        "fade_black"     → 0.5 s black clip inserted between the two clips (hard cuts)
        "flash_white"    → 0.1 s white clip inserted between the two clips (hard cuts)
    """
    n = len(clip_files)

    # ── Fast path: no transitions ──────────────────────────────────────────────
    if all((clip_trans[i] or "none") == "none" for i in range(1, n)):
        lst = tmp / "concat.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in clip_files))
        return subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(lst), "-c", "copy", str(output)],
            capture_output=True, text=True,
        )

    # ── Expand: insert colour clips for fade_black / flash_white ──────────────
    exp_files: list = [clip_files[0]]
    exp_durs:  list = [clip_durs[0]]
    exp_trans: list = ["none"]

    for i in range(1, n):
        t = clip_trans[i] or "none"
        if t in _FLASH_DURATIONS:
            color    = "black" if t == "fade_black" else "white"
            dur_fd   = _FLASH_DURATIONS[t]
            fd_path  = tmp / f"fade_{i:04d}_{color}.mp4"
            _make_color_clip(fd_path, color, dur_fd, fps, w, h, sample_rate)
            exp_files.append(fd_path);       exp_durs.append(dur_fd);      exp_trans.append("none")
            exp_files.append(clip_files[i]); exp_durs.append(clip_durs[i]); exp_trans.append("none")
        else:
            exp_files.append(clip_files[i]); exp_durs.append(clip_durs[i]); exp_trans.append(t)

    # ── If still no dissolves, concat-copy is enough ──────────────────────────
    if all((exp_trans[i] or "none") == "none" for i in range(1, len(exp_files))):
        lst = tmp / "concat.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in exp_files))
        return subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(lst), "-c", "copy", str(output)],
            capture_output=True, text=True,
        )

    # ── Build filter_complex with chained xfade / concat ──────────────────────
    m = len(exp_files)
    ffmpeg_inputs: list = []
    for p in exp_files:
        ffmpeg_inputs += ["-i", str(p)]

    parts:   list  = []
    cur_v          = "[0:v]"
    cur_a          = "[0:a]"
    cum_dur: float = exp_durs[0]

    for i in range(1, m):
        t  = exp_trans[i] or "none"
        d  = exp_durs[i]
        lv = f"v{i}"
        la = f"a{i}"

        if t == "none":
            # settb normalises concat's 1/1000000 output back to the libx264
            # clip timebase (1/(fps*512)) so a downstream xfade sees matching
            # timebases on both of its inputs.
            parts.append(f"{cur_v}[{i}:v]concat=n=2:v=1:a=0,settb=1/{fps*512}[{lv}]")
            parts.append(f"{cur_a}[{i}:a]concat=n=2:v=0:a=1[{la}]")
            cum_dur += d
        else:
            td     = _DISSOLVE_DURATIONS[t]
            offset = max(0.0, cum_dur - td)
            parts.append(
                f"{cur_v}[{i}:v]xfade=transition=fade:"
                f"duration={td:.3f}:offset={offset:.3f},format=yuv420p[{lv}]"
            )
            parts.append(f"{cur_a}[{i}:a]acrossfade=d={td:.3f}[{la}]")
            cum_dur = cum_dur + d - td   # xfade overlaps the two clips by td

        cur_v = f"[{lv}]"
        cur_a = f"[{la}]"

    filter_complex = ";".join(parts)
    print(f"  [transitions] filter_complex built ({len(parts)} filter parts for {m} clips)")

    return subprocess.run(
        ["ffmpeg", "-y"]
        + ffmpeg_inputs
        + [
            "-filter_complex", filter_complex,
            "-map", cur_v, "-map", cur_a,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", str(sample_rate),
            str(output),
        ],
        capture_output=True, text=True,
    )


# ── MediaPlan-mode helpers ─────────────────────────────────────────────────────
# Used by the ShotList-free render path (main below).

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
            c = ep_dir / p
            if c.exists():
                return str(c)
            return str(c)
    url = seg.get("url", "")
    if url.startswith("file://"):
        return unquote(urlparse(url).path)
    return url


def build_silent_audio(duration_sec: float, out_path: Path) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
        "-t", str(duration_sec), str(out_path)
    ], capture_output=True, check=True)


def _anim_vf(anim_type: str, clip_dur: float) -> str:
    """Return ffmpeg zoompan filter string for the given animation type."""
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


def _mp_seg_dur(seg: dict) -> float:
    """Duration in seconds of one MediaPlan segment."""
    seg_type = seg.get("type", "image")
    if seg_type != "image":
        ci = float(seg.get("clip_in") or 0)
        co = seg.get("clip_out")
        if co is not None and float(co) > ci:
            return float(co) - ci
        dur = seg.get("duration_sec")
        if dur:
            return float(dur)
        return 0.0
    hold = seg.get("hold_sec")
    if hold:
        return float(hold)
    dur = seg.get("duration_sec")
    if dur:
        return float(dur)
    return 3.0   # default image hold


def _scene_id_from_item_id(item_id: str) -> str:
    """'vo-sc01-001' → 'sc01'"""
    m = re.search(r'(sc\d+)', item_id)
    return m.group(1) if m else ""


def build_scene_timeline(
    vo_items: list,
    scene_heads: dict,
    voplan_total_dur: float,
) -> tuple:
    """Derive scene slots from VOPlan.

    Returns:
      scene_slots      : { scene_id: (start_sec, dur_sec) }  episode-absolute
      video_total_dur  : float
      vo_head_offsets  : { scene_id: float }  ms offset for VO delay adjustment
    """
    scene_first_vo: dict = {}
    for vi in sorted(vo_items, key=lambda v: v.get("start_sec", 0)):
        scid = _scene_id_from_item_id(vi.get("item_id", ""))
        if scid and scid not in scene_first_vo:
            scene_first_vo[scid] = vi

    scenes_in_order = sorted(
        scene_first_vo.keys(),
        key=lambda s: scene_first_vo[s]["start_sec"]
    )

    _aware = not any(
        scene_first_vo[scid]["start_sec"] < (head - 0.1)
        for scid, head in scene_heads.items()
        if scid in scene_first_vo
    )
    print(f"  [scene_heads] mode: {'aware' if _aware else 'unaware'}")

    scene_slots: dict     = {}
    vo_head_offsets: dict = {}

    if _aware:
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
            vo_head_offsets[scid] = 0.0
        video_total_dur = voplan_total_dur
    else:
        ep_cursor      = 0.0
        cum_head_added = 0.0
        for i, scid in enumerate(scenes_in_order):
            head     = scene_heads.get(scid, 0.0)
            sc_start = ep_cursor
            cum_head_added += head
            vo_head_offsets[scid] = cum_head_added
            if i + 1 < len(scenes_in_order):
                next_scid   = scenes_in_order[i + 1]
                natural_dur = (scene_first_vo[next_scid]["start_sec"]
                               - scene_first_vo[scid]["start_sec"])
            else:
                natural_dur = voplan_total_dur - scene_first_vo[scid]["start_sec"]
            sc_dur = natural_dur + head
            scene_slots[scid] = (sc_start, max(0.0, sc_dur))
            ep_cursor = sc_start + sc_dur
        video_total_dur = ep_cursor

    for scid in scenes_in_order:
        s, d = scene_slots[scid]
        print(f"  [scene] {scid}: {s:.3f}s – {s+d:.3f}s  dur={d:.3f}s  "
              f"head={scene_heads.get(scid, 0.0):.1f}s  vo_offset={vo_head_offsets[scid]:.1f}s")

    return scene_slots, video_total_dur, vo_head_offsets


def write_srt_from_voplan(vo_items: list, srt_path: Path, subs_path: Path) -> None:
    """Write SRT + subs.json directly from VOPlan items (episode-absolute timing)."""
    lines: list[str] = []
    subs:  list[dict] = []
    seq = 1
    for vi in sorted(vo_items, key=lambda v: v.get("start_sec", 0)):
        text = vi.get("text", "").strip()
        if not text:
            continue
        abs_in  = round(vi["start_sec"] * 1000)
        abs_out = round(vi["end_sec"]   * 1000)
        timecode = f"{ms_to_srt_ts(abs_in)} --> {ms_to_srt_ts(abs_out)}"
        lines += [str(seq), timecode, text, ""]
        subs.append({
            "line_id":  vi.get("item_id", ""),
            "timecode": timecode,
            "text":     text,
        })
        seq += 1
    srt_path.write_text("\n".join(lines), encoding="utf-8")
    subs_path.write_text(json.dumps(subs, ensure_ascii=False, indent=2), encoding="utf-8")


def write_license_manifest_flat(
    segments: list,
    output_dir: Path,
    locale: str,
) -> "Path | None":
    """Write licenses.json from flat MediaPlan.shot_overrides[] segments.

    Each entry carries timing in the final video (computed from segment ORDER,
    not from start_sec), plus full license/attribution metadata from the
    segment's 'source' block.
    """
    entries: list[dict] = []
    cursor = 0.0
    for seg in segments:
        seg_type = seg.get("type", "image")
        seg_dur  = _mp_seg_dur(seg)
        source   = seg.get("source") or {}

        video_start = round(cursor, 6)
        video_end   = round(cursor + seg_dur, 6)

        entry: dict = {
            "locale":               locale,
            "video_start_sec":      video_start,
            "video_end_sec":        video_end,
            "video_duration_sec":   round(seg_dur, 6),
            "media_type":           seg_type,
            "url":                  seg.get("url", ""),
            "clip_start_sec":       float(seg.get("clip_in")  or 0.0),
            "clip_end_sec":         float(seg.get("clip_out") or 0.0),
            "hold_sec":             float(seg.get("hold_sec") or 0.0) if seg_type == "image" else None,
            "animation_type":       seg.get("animation_type"),
            # attribution / license from source block
            "title":                source.get("title"),
            "photographer":         source.get("photographer"),
            "attribution_text":     source.get("attribution_text"),
            "attribution_required": source.get("attribution_required"),
            "license_summary":      source.get("license_summary"),
            "license_url":          source.get("license_url"),
            "asset_page_url":       source.get("asset_page_url"),
            "file_url":             source.get("file_url"),
            "source_site":          source.get("source_site"),
            "width":                source.get("width"),
            "height":               source.get("height"),
            "tags":                 source.get("tags"),
        }
        entry = {k: v for k, v in entry.items() if v is not None}
        entries.append(entry)
        cursor += seg_dur

    if not entries:
        return None

    manifest = {
        "schema_id":      "license_manifest",
        "schema_version": "1.0.0",
        "locale":         locale,
        "generated_at":   datetime.datetime.utcnow().isoformat() + "Z",
        "total_segments": len(entries),
        "segments":       entries,
    }
    out_path = output_dir / "licenses.json"
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [license] Wrote {len(entries)} segment(s) → {out_path}")
    return out_path


# ── SRT writer ────────────────────────────────────────────────────────────────

def ms_to_srt_ts(ms: int) -> str:
    """Convert milliseconds to SRT timestamp HH:MM:SS,mmm."""
    h, ms  = divmod(ms, 3_600_000)
    m, ms  = divmod(ms,    60_000)
    s, ms  = divmod(ms,     1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(shots: list[dict], srt_path: Path, subs_path: Path, fps: int,
              vo_timing_by_shot: dict | None = None) -> None:
    """
    Write output.{locale}.srt with cumulative absolute timestamps and a parallel
    output.subs.json sidecar keyed by line_id for cross-language SRT generation.
    Accounts for 1-frame black frames inserted at scene boundaries.

    vo_timing_by_shot: {shot_id: [{item_id, text, shot_rel_in_ms, shot_rel_out_ms}]}
    When None, SRT is written empty.
    """
    lines: list[str] = []
    subs:  list[dict] = []
    seq      = 1
    frame_ms = round(1000 / fps)

    for i, shot in enumerate(shots):
        shot_id = shot.get("shot_id", "")
        for vl in (vo_timing_by_shot or {}).get(shot_id, []):
            text = vl.get("text", "").strip()
            if not text:
                continue
            # ep_start_sec/ep_end_sec are episode-absolute; use them directly.
            abs_in  = round(vl["ep_start_sec"] * 1000)
            abs_out = round(vl["ep_end_sec"]   * 1000)
            timecode = f"{ms_to_srt_ts(abs_in)} --> {ms_to_srt_ts(abs_out)}"
            lines += [str(seq), timecode, text, ""]
            subs.append({
                "line_id":  vl.get("item_id", vl.get("line_id", "")),
                "timecode": timecode,
                "text":     text,
            })
            seq += 1

        # Add black frame duration at scene boundaries (used for scene-boundary detection)
        if i < len(shots) - 1:
            if shots[i + 1].get("scene_id") != shot.get("scene_id"):
                pass  # scene boundary noted; offset_ms no longer tracked here

    srt_path.write_text("\n".join(lines), encoding="utf-8")
    subs_path.write_text(json.dumps(subs, ensure_ascii=False, indent=2), encoding="utf-8")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render VOPlan.{locale}.json → output.mp4 using FFmpeg.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--plan", required=True, metavar="PATH",
        help="Path to VOPlan.{locale}.json (or legacy RenderPlan.{locale}.json).",
    )
    p.add_argument(
        "--locale", default=None, metavar="LOCALE",
        help="Locale string (e.g. en, zh-Hans). Auto-detected from filename if omitted.",
    )
    p.add_argument(
        "--reference-approval", default=None, metavar="PATH",
        help="Path to VOPlan.en.json for non-EN locale EN-floor duration computation. "
             "render_video.py exits 1 if this flag is passed but the file is missing.",
    )
    p.add_argument(
        "--out", default=None, metavar="DIR",
        help="Output directory. Default: {episode_dir}/renders/{locale}/",
    )
    p.add_argument(
        "--profile", default=None, metavar="PROFILE",
        help=f"Encode profile ({', '.join(PROFILES)}). Default: {DEFAULT_PROFILE}",
    )
    p.add_argument(
        "--music-fadeout-sec", type=float, default=MUSIC_FADEOUT_SEC,
        metavar="SEC",
        help=f"Music fade-out duration when next shot has no music (default: {MUSIC_FADEOUT_SEC}s).",
    )
    p.add_argument(
        "--keep-intermediates", action="store_true",
        help="Keep per-shot MKV intermediates after rendering (default: delete).",
    )
    p.add_argument(
        "--no-music", action="store_true",
        help="Skip music entirely — renders VO + SFX only.",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print FFmpeg commands as they run.",
    )
    p.add_argument(
        "--format", default=None, metavar="FORMAT",
        help="Project format (e.g. 'mtv'). MTV skips VO mixing and music ducking.",
    )
    return p.parse_args()


# ── Duck volume filter ────────────────────────────────────────────────────────

def _build_duck_vol_filter(vo_items: list, seg_start_ep: float, seg_end_ep: float,
                           duck_db: float, fade_sec: float,
                           base_db: float, vol_offset_db: float) -> str:
    """Return an FFmpeg volume filter string that ducks during VO lines.

    Applied BEFORE adelay so ``t`` in the expression is segment-relative
    (t=0 == seg_start_ep in the episode). When no VO lines overlap the
    segment a plain ``volume=<scalar>`` is returned.
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
        t0 = max(0.0, vi_start - fade_sec - seg_start_ep)
        t1 = vi_end + fade_sec - seg_start_ep
        if t1 > t0:
            intervals.append((round(t0, 3), round(t1, 3)))

    if not intervals:
        return f"volume={base_amp * scale:.6f}"

    cond = "+".join(f"between(t,{t0},{t1})" for t0, t1 in intervals)
    expr = f"if(gt({cond},0),{duck_amp * scale:.6f},{base_amp * scale:.6f})"
    return f"volume=volume='{expr}':eval=frame"


# ── Main ──────────────────────────────────────────────────────────────────────
# MediaPlan-mode renderer: VOPlan is the single source of truth for timing.
# Segments come from MediaPlan.json shot_overrides[] in ORDER (no start_sec needed).
# ShotList.json is NOT read.

def main() -> None:
    args = parse_args()

    plan_path = Path(args.plan).resolve()
    if not plan_path.exists():
        print(f"[ERROR] VOPlan not found: {plan_path}", file=sys.stderr)
        sys.exit(1)

    voplan = load_json(plan_path)

    # ── Locale detection ────────────────────────────────────────────────────
    # Prefer explicit --locale arg; otherwise derive from the filename:
    # "VOPlan.zh-Hans.json" → "zh-Hans"  (handles compound locales with hyphens)
    locale = args.locale
    if not locale:
        stem = plan_path.stem   # e.g. "VOPlan.zh-Hans"
        if stem.startswith("VOPlan."):
            locale = stem[len("VOPlan."):]
        elif stem.startswith("RenderPlan."):
            locale = stem[len("RenderPlan."):]
        else:
            locale = voplan.get("locale") or voplan.get("plan_id", "en").rsplit("-", 1)[-1]

    # ── Paths ────────────────────────────────────────────────────────────────
    episode_dir = plan_path.parent
    output_dir  = Path(args.out).resolve() if args.out else \
                  (episode_dir / "renders" / locale)
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_name = args.profile or DEFAULT_PROFILE
    profile      = PROFILES.get(profile_name, PROFILES[DEFAULT_PROFILE])
    crf          = profile["crf"]
    preset_str   = profile["preset"]

    # ── 1. VOPlan — single source of truth for timing ────────────────────────
    vo_items = [v for v in voplan.get("vo_items", []) if v.get("end_sec") is not None]
    scene_heads = {
        k: float(v) for k, v in voplan.get("scene_heads", {}).items() if float(v) > 0
    }
    if not vo_items:
        print("[ERROR] VOPlan has no vo_items with end_sec.", file=sys.stderr)
        sys.exit(1)
    voplan_total_dur = max(
        v["end_sec"] + (v.get("pause_after_ms") or 0) / 1000.0
        for v in vo_items
    )

    print("=" * 60)
    print("  render_video  [MediaPlan mode — ShotList-free]")
    print(f"  Plan    : {plan_path.name}")
    print(f"  Locale  : {locale}")
    print(f"  Profile : {profile_name}  (CRF {crf}, {preset_str})")
    print(f"  Output  : {output_dir}")
    print(f"  [dur] voplan_total_dur={voplan_total_dur:.3f}s  scene_heads={scene_heads}")
    print("=" * 60)

    # ── 2. Build scene timeline from VOPlan ───────────────────────────────────
    scene_slots, video_total_dur, vo_head_offsets = build_scene_timeline(
        vo_items, scene_heads, voplan_total_dur
    )

    # ── 3. Read MediaPlan.json segments ───────────────────────────────────────
    media_plan_path = episode_dir / "MediaPlan.json"
    media_segments: list = []
    if media_plan_path.exists():
        _mp_data       = load_json(media_plan_path)
        media_segments = _mp_data.get("shot_overrides", [])
        print(f"  [media] {len(media_segments)} segment(s) from MediaPlan.json")
    else:
        print("  [media] MediaPlan.json not found — rendering black background")

    # ── 4. Compute total duration ─────────────────────────────────────────────
    if media_segments:
        _seg_total = sum(_mp_seg_dur(s) for s in media_segments)
        print(f"  [dur] seg_total={_seg_total:.3f}s  voplan={voplan_total_dur:.3f}s")
    # Snap to frame boundary so ffmpeg never rounds the last partial frame up.
    # e.g. 34.997s × 24fps = 839.928 → 840 frames → 35.000s without this snap.
    total_dur = math.floor(voplan_total_dur * FPS) / FPS
    print(f"  [dur] total_dur={total_dur:.3f}s")

    _scale_pad = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
                  f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1")

    with tempfile.TemporaryDirectory() as _tmp_dir:
        _tmp = Path(_tmp_dir)
        _clip_files: list = []
        _clip_durs:  list = []   # parallel: rendered duration of each clip in _clip_files
        _clip_trans: list = []   # parallel: transition keyword at the *start* of each clip

        # ── 5. Build video clips from segments ──────────────────────────────
        if not media_segments:
            _bp = _tmp / "clip_0000_black.mp4"
            _r = subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", f"color=c=black:size={W}x{H}:rate={FPS}:duration={total_dur:.3f}",
                "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
                "-t", f"{total_dur:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest", str(_bp)
            ], capture_output=True, text=True)
            if _r.returncode == 0:
                _clip_files.append(_bp)
                _clip_durs.append(total_dur)
                _clip_trans.append("none")
                print(f"  [clip] black placeholder {total_dur:.3f}s ok")
            else:
                print(f"[ERROR] black placeholder failed: {_r.stderr[-300:]}", file=sys.stderr)
                sys.exit(1)
        else:
            for _si, _seg in enumerate(media_segments):
                _stype    = _seg.get("type", "image")
                _sdur     = _mp_seg_dur(_seg)
                _anim     = (_seg.get("animation_type") or _seg.get("animation") or "none").lower()
                _mpath    = resolve_media_path(_seg, episode_dir)
                _cpath    = _tmp / f"clip_{_si:04d}.mp4"

                print(f"  [seg] {_si} type={_stype} dur={_sdur:.3f}s "
                      f"path={str(_mpath)[:80] if _mpath else 'MISSING'}")

                if _sdur <= 0:
                    print(f"  [warn] seg {_si}: duration 0 — skipped")
                    continue

                if not _mpath or not Path(_mpath).exists():
                    print(f"  [warn] seg {_si}: media not found '{_mpath}' — black")
                    _r = subprocess.run([
                        "ffmpeg", "-y", "-f", "lavfi",
                        "-i", f"color=c=black:size={W}x{H}:rate={FPS}:duration={_sdur:.3f}",
                        "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
                        "-t", f"{_sdur:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", str(_cpath)
                    ], capture_output=True, text=True)
                    if _r.returncode == 0:
                        _clip_files.append(_cpath)
                        _clip_durs.append(_sdur)
                        _clip_trans.append(_seg.get("transition") or "none")
                    continue

                if _stype == "image":
                    _zoompan = _anim_vf(_anim, _sdur)
                    _vf      = (_scale_pad + "," + _zoompan) if _zoompan else _scale_pad
                    _cmd     = [
                        "ffmpeg", "-y",
                        "-loop", "1", "-framerate", str(FPS),
                        "-t", f"{_sdur:.3f}", "-i", _mpath,
                        "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
                        "-vf", _vf, "-r", str(FPS), "-pix_fmt", "yuv420p",
                        "-t", f"{_sdur:.3f}", "-c:v", "libx264", "-c:a", "aac",
                        "-shortest", str(_cpath)
                    ]
                else:
                    _seg_start_t = float(_seg.get("clip_in") or 0)
                    _cmd = [
                        "ffmpeg", "-y",
                        "-ss", f"{_seg_start_t:.3f}", "-t", f"{_sdur:.3f}",
                        "-i", _mpath,
                        "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
                        "-vf", _scale_pad, "-r", str(FPS), "-pix_fmt", "yuv420p",
                        "-t", f"{_sdur:.3f}", "-c:v", "libx264", "-c:a", "aac",
                        "-shortest", str(_cpath)
                    ]

                _r = subprocess.run(_cmd, capture_output=True, text=True)
                if _r.returncode != 0:
                    print(f"  [warn] seg {_si} ({_stype}) failed: {_r.stderr[-300:]}")
                    _r2 = subprocess.run([
                        "ffmpeg", "-y", "-f", "lavfi",
                        "-i", f"color=c=black:size={W}x{H}:rate={FPS}:duration={_sdur:.3f}",
                        "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
                        "-t", f"{_sdur:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", str(_cpath)
                    ], capture_output=True, text=True)
                    if _r2.returncode == 0:
                        _clip_files.append(_cpath)
                        _clip_durs.append(_sdur)
                        _clip_trans.append(_seg.get("transition") or "none")
                        print(f"  [warn] seg {_si}: black fallback")
                    continue

                _clip_files.append(_cpath)
                _clip_durs.append(_sdur)
                _clip_trans.append(_seg.get("transition") or "none")
                print(f"  [clip] seg {_si}: {_stype} {_sdur:.3f}s ok")

        if not _clip_files:
            print("[ERROR] No clips generated.", file=sys.stderr)
            sys.exit(1)

        # ── 6. Concatenate clips (with optional transitions) ─────────────────
        _concat_video = _tmp / "concat.mp4"
        _result = _concat_with_transitions(
            _clip_files, _clip_durs, _clip_trans,
            _concat_video, _tmp, FPS, W, H, SAMPLE_RATE,
        )
        if _result.returncode != 0:
            print(f"[ERROR] concat failed: {_result.stderr[-500:]}", file=sys.stderr)
            sys.exit(1)

        # ── 7. Build VO audio (episode-absolute from VOPlan) ─────────────────
        _is_mtv = (getattr(args, "format", None) or "").lower() == "mtv"
        _vo_dir    = episode_dir / "assets" / locale / "audio" / "vo"
        _vo_audio  = _tmp / "vo_mix.wav"
        _vo_inputs: list = []
        _vo_delays: list = []
        _vo_volumes: list = []

        # MTV: skip VO audio entirely — no WAV files exist, music is the sole audio
        if not _is_mtv:
            for _vi in sorted(vo_items, key=lambda v: v.get("start_sec", 0)):
                _wav = _vo_dir / f"{_vi['item_id']}.wav"
                if not _wav.exists():
                    continue
                _scid    = _scene_id_from_item_id(_vi.get("item_id", ""))
                _head_off = vo_head_offsets.get(_scid, 0.0)
                _delay_ms = int((_vi["start_sec"] + _head_off) * 1000)
                if _delay_ms < 0:
                    continue
                _vo_inputs.append(str(_wav))
                _vo_delays.append(_delay_ms)
                _vo_volumes.append(float(_vi.get("volume_db", 0.0) or 0.0))

        if _vo_inputs:
            _vf_parts = []
            for i, (d, vdb) in enumerate(zip(_vo_delays, _vo_volumes)):
                if vdb != 0.0:
                    _vf_parts.append(f"[{i}]volume={vdb}dB,adelay={d}|{d}[d{i}]")
                else:
                    _vf_parts.append(f"[{i}]adelay={d}|{d}[d{i}]")
            _vmix_ins = "".join(f"[d{i}]" for i in range(len(_vo_inputs)))
            _vf_str   = (";".join(_vf_parts)
                         + f";{_vmix_ins}amix=inputs={len(_vo_inputs)}:normalize=0,"
                         f"apad=pad_dur={total_dur:.3f}[out]")
            _cmd = ["ffmpeg", "-y"]
            for _inp in _vo_inputs:
                _cmd += ["-i", _inp]
            _cmd += ["-filter_complex", _vf_str, "-map", "[out]",
                     "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS),
                     "-t", str(total_dur), str(_vo_audio)]
            _result = subprocess.run(_cmd, capture_output=True, text=True)
            if _result.returncode != 0:
                print(f"  [warn] VO mix failed: {_result.stderr[-300:]}")
                build_silent_audio(total_dur, _vo_audio)
            else:
                print(f"  [vo] Mixed {len(_vo_inputs)} VO line(s)")
        else:
            if _is_mtv:
                print("  [vo] MTV mode — skipping VO audio (music is primary audio)")
            else:
                print("  [vo] No VO WAV files — silent VO track")
            build_silent_audio(total_dur, _vo_audio)

        # ── 8. Build music audio (from MusicPlan.json) ───────────────────────
        _music_audio = None
        if not args.no_music:
            _mus_path = episode_dir / "MusicPlan.json"
            if _mus_path.exists():
                _mus_doc    = load_json(_mus_path)
                _clip_vol   = _mus_doc.get("clip_volumes",  {})
                _track_vol  = _mus_doc.get("track_volumes", {})
                _BASE_MDB        = -6.0
                _DEFAULT_DUCK_DB = -12.0
                _DEFAULT_FADE    =   0.5
                _mi, _md, _mvf, _mcd = [], [], [], []
                for _ovr in _mus_doc.get("shot_overrides", []):
                    _asset    = _ovr.get("music_asset_id", "")
                    _mst      = _ovr.get("start_sec")
                    _men      = _ovr.get("end_sec")
                    _duck_db  = float(_ovr.get("duck_db",  _DEFAULT_DUCK_DB))
                    _fade_sec = float(_ovr.get("fade_sec", _DEFAULT_FADE))
                    if _mst is None:
                        continue
                    _mst_f = float(_mst)
                    _men_f = float(_men) if _men is not None else total_dur
                    _mdms  = max(0.0, _mst_f) * 1000
                    _mcdur = max(0.0, _men_f - _mst_f) if _men is not None else None
                    _wl    = episode_dir / "assets" / "music" / f"{_asset}.loop.wav"
                    _wb    = episode_dir / "assets" / "music" / f"{_asset}.wav"
                    _wav   = (str(_wl) if _wl.exists()
                              else str(_wb) if _wb.exists() else "")
                    if not _wav:
                        print(f"  [WARN] music wav not found: {_asset}")
                        continue
                    _stem   = re.sub(r'_\d[\d_]*s-[\d_\.]+s$', '', _asset)
                    _db_off = _track_vol.get(_stem, 0.0) + _clip_vol.get(_asset, 0.0)
                    # MTV: no ducking — pass empty list so music plays at full volume
                    _duck_items = [] if _is_mtv else vo_items
                    _vf     = _build_duck_vol_filter(
                        _duck_items, _mst_f, _men_f,
                        _duck_db, _fade_sec, _BASE_MDB, _db_off)
                    _mi.append(_wav)
                    _md.append(_mdms)
                    _mvf.append(_vf)
                    _mcd.append(_mcdur)
                    print(f"  [music] {_asset}: delay={_mdms:.0f}ms "
                          f"clip_dur={_mcdur} duck_db={_duck_db}")
                if _mi:
                    _music_audio = _tmp / "music_mix.wav"
                    _mf_parts = []
                    for _idx, (_d, _vf, _cd) in enumerate(zip(_md, _mvf, _mcd)):
                        if _cd is not None:
                            _mf_parts.append(
                                f"[{_idx}]atrim=duration={_cd:.3f},{_vf},"
                                f"adelay={_d:.0f}|{_d:.0f}[m{_idx}]")
                        else:
                            _mf_parts.append(
                                f"[{_idx}]{_vf},adelay={_d:.0f}|{_d:.0f}[m{_idx}]")
                    _mmix_ins = "".join(f"[m{i}]" for i in range(len(_mi)))
                    _mf_str   = (";".join(_mf_parts)
                                 + f";{_mmix_ins}amix=inputs={len(_mi)}:normalize=0,"
                                 f"apad=pad_dur={total_dur:.3f}[out]")
                    _cmd = ["ffmpeg", "-y"]
                    for _inp in _mi:
                        _cmd += ["-i", _inp]
                    _cmd += ["-filter_complex", _mf_str, "-map", "[out]",
                             "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS),
                             "-t", str(total_dur), str(_music_audio)]
                    _res = subprocess.run(_cmd, capture_output=True, text=True)
                    if _res.returncode != 0:
                        print(f"  [warn] Music mix failed: {_res.stderr[-300:]}")
                        _music_audio = None
                    else:
                        print(f"  [music] Mixed {len(_mi)} music clip(s)")
                else:
                    print("  [music] No music WAVs found — music skipped")
            else:
                print("  [music] MusicPlan.json not found — music skipped")

        # ── 9. Build SFX audio (from SfxPlan.json) ───────────────────────────
        _sfx_audio = None
        _sfx_path  = episode_dir / "SfxPlan.json"
        if _sfx_path.exists():
            _sfx_plan = load_json(_sfx_path)
            _cut_abs: dict = {}
            _cc_src2 = list(_sfx_plan.get("cut_clips", []))
            _sfx_cc_path = episode_dir / "assets" / "sfx" / "sfx_cut_clips.json"
            if _sfx_cc_path.exists():
                try:
                    _extra2 = load_json(_sfx_cc_path)
                    if isinstance(_extra2, list):
                        _existing2 = {c.get("clip_id") for c in _cc_src2}
                        _cc_src2 += [c for c in _extra2 if c.get("clip_id") not in _existing2]
                except Exception:
                    pass
            for _cc in _cc_src2:
                _cid  = _cc.get("clip_id", "")
                _crel = _cc.get("path", "")
                if _cid and _crel:
                    _cabs = Path(_crel) if Path(_crel).is_absolute() else episode_dir / _crel
                    _cut_abs[_cid] = str(_cabs)
            _si2, _sd2, _sdur2 = [], [], []
            for _se in _sfx_plan.get("shot_overrides", []):
                _src = _se.get("source_file") or _se.get("local_path") or ""
                if _src and not Path(_src).exists():
                    _src = _cut_abs.get(_src, _src)
                if _src and Path(_src).exists():
                    _ss2   = float(_se.get("start_sec", 0))
                    _se_e  = _se.get("end_sec")
                    _si2.append(_src)
                    _sd2.append(max(0.0, _ss2) * 1000)
                    _sdur2.append(max(0.0, float(_se_e) - _ss2)
                                  if _se_e is not None else None)
            if _si2:
                _sfx_audio = _tmp / "sfx_mix.wav"
                _sf_parts  = [
                    (f"[{i}]atrim=duration={dur:.3f},adelay={d:.0f}|{d:.0f}[s{i}]" if dur
                     else f"[{i}]adelay={d:.0f}|{d:.0f}[s{i}]")
                    for i, (d, dur) in enumerate(zip(_sd2, _sdur2))
                ]
                _smix_ins = "".join(f"[s{i}]" for i in range(len(_si2)))
                _sf_str   = (";".join(_sf_parts)
                             + f";{_smix_ins}amix=inputs={len(_si2)}:normalize=0,"
                             f"apad=pad_dur={total_dur:.3f}[out]")
                _cmd = ["ffmpeg", "-y"]
                for _inp in _si2:
                    _cmd += ["-i", _inp]
                _cmd += ["-filter_complex", _sf_str, "-map", "[out]",
                         "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS),
                         "-t", str(total_dur), str(_sfx_audio)]
                _res = subprocess.run(_cmd, capture_output=True, text=True)
                if _res.returncode != 0:
                    print(f"  [warn] SFX mix failed: {_res.stderr[-300:]}")
                    _sfx_audio = None
                else:
                    print(f"  [sfx] Mixed {len(_si2)} SFX clip(s)")
            else:
                print("  [sfx] No SFX WAVs found — SFX skipped")
        else:
            print("  [sfx] SfxPlan.json not found — SFX skipped")

        # ── 10. Combine audio tracks ─────────────────────────────────────────
        _extra = [a for a in [_music_audio, _sfx_audio] if a and Path(a).exists()]
        if _extra:
            _final_audio = _tmp / "final_audio.wav"
            _all_t = [str(_vo_audio)] + [str(a) for a in _extra]
            _nt    = len(_all_t)
            _amix  = "".join(f"[{i}]" for i in range(_nt))
            _af    = (f"{_amix}amix=inputs={_nt}:normalize=0,"
                      f"apad=pad_dur={total_dur:.3f}[out]")
            _cmd = ["ffmpeg", "-y"]
            for _t in _all_t:
                _cmd += ["-i", _t]
            _cmd += ["-filter_complex", _af, "-map", "[out]",
                     "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS),
                     "-t", str(total_dur), str(_final_audio)]
            _res = subprocess.run(_cmd, capture_output=True, text=True)
            if _res.returncode != 0:
                print(f"  [warn] Audio combine failed: {_res.stderr[-300:]}")
                _final_audio = _vo_audio
            else:
                print(f"  [audio] Combined {_nt} track(s): VO"
                      + (" + Music" if _music_audio else "")
                      + (" + SFX"   if _sfx_audio   else ""))
        else:
            _final_audio = _vo_audio

        # ── 11. Mux video + audio with quality profile ───────────────────────
        final_mp4 = output_dir / "output.mp4"
        _result = subprocess.run([
            "ffmpeg", "-y",
            "-i", str(_concat_video),
            "-i", str(_final_audio),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-crf", str(crf), "-preset", preset_str,
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-t", str(total_dur),
            str(final_mp4)
        ], capture_output=True, text=True)
        if _result.returncode != 0:
            print(f"[ERROR] final mux failed: {_result.stderr[-500:]}", file=sys.stderr)
            sys.exit(1)
        print(f"  [mux] {final_mp4}")

    # ── 12. SRT + subtitle sidecar ───────────────────────────────────────────
    srt_path  = output_dir / f"output.{locale}.srt"
    subs_path = output_dir / "output.subs.json"
    write_srt_from_voplan(vo_items, srt_path, subs_path)
    print(f"  SRT:  {srt_path}")
    print(f"  Subs: {subs_path}")

    # ── 13. render_output.json  (contract: RenderOutput.v1.json) ────────────
    total_ms  = round(total_dur * 1000)
    _plan_id  = voplan.get("plan_id", voplan.get("manifest_id", ""))
    _ro = {
        # ── required by RenderOutput.v1.json ──────────────────────────────
        "schema_id":      "render_output",
        "schema_version": "1.0.0",
        "output_id":      f"{_plan_id}-{locale}" if _plan_id else f"render-{locale}",
        "video_uri":      final_mp4.as_uri(),
        "captions_uri":   srt_path.as_uri(),
        "hashes": {
            "video_sha256":    hashlib.sha256(final_mp4.read_bytes()).hexdigest()
                               if final_mp4.exists() else None,
            "captions_sha256": hashlib.sha256(srt_path.read_bytes()).hexdigest()
                               if srt_path.exists() else None,
        },
        # ── extra fields ──────────────────────────────────────────────────
        "producer":          PRODUCER,
        "plan_id":           _plan_id,
        "locale":            locale,
        "output_video":      str(final_mp4),
        "output_srt":        str(srt_path),
        "output_subs":       str(subs_path),
        "total_segments":    len(media_segments),
        "total_duration_ms": total_ms,
        "profile":           profile_name,
    }
    _ro_path = output_dir / "render_output.json"
    save_json(_ro, _ro_path)

    # ── Contract verification ─────────────────────────────────────────────────
    _verify_py = Path(__file__).parent.parent.parent / "contracts" / "tools" / "verify_contracts.py"
    _schemas_dir = Path(__file__).parent.parent.parent / "contracts" / "schemas"
    if _verify_py.exists():
        _vr = subprocess.run(
            [sys.executable, str(_verify_py), str(_ro_path), "RenderOutput",
             "--schemas-dir", str(_schemas_dir)],
            capture_output=True, text=True
        )
        _vout = (_vr.stdout + _vr.stderr).strip()
        if _vr.returncode == 0:
            print(f"  [contract] render_output.json  ✓ PASS")
        else:
            print(f"  [contract] render_output.json  ✗ FAIL — {_vout}")
    else:
        print(f"  [contract] verify_contracts.py not found — skipped")

    # ── 14. License manifest ─────────────────────────────────────────────────
    _lic_path = write_license_manifest_flat(
        segments=media_segments,
        output_dir=output_dir,
        locale=locale,
    )
    if _lic_path:
        print(f"  Licenses : {_lic_path}")

    print(f"\n  [OK] {final_mp4}")
    print(f"  Duration : {total_dur:.3f}s")
    print(f"  Segments : {len(media_segments)}")


if __name__ == "__main__":
    main()
