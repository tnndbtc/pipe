#!/usr/bin/env python3
# =============================================================================
# render_video.py — Produce output.mp4 from VOPlan + MusicPlan + SfxPlan + MediaPlan
# =============================================================================
#
# Reads VOPlan.{locale}.json + ShotList.json and resolves each shot into an MKV
# intermediate, then concatenates into a final output.mp4.
#
# Architecture (per /tmp/v1 spec):
#   1. Per-shot render → MKV intermediates (libx264 + pcm_s16le) in .shots/
#   2. Concat intermediates with scene-boundary black frames
#   3. Apply loudnorm (-16 LUFS, linear=true) + encode AAC → output.mp4
#   4. Write output.{locale}.srt (absolute timestamps) + output.subs.json sidecar
#   5. Write render_output.json (stats + placeholder_count)
#
# Character compositing:
#   • Layout: 1=centre, 2=left-third/right-third, 3=even thirds, 4+=quarters
#   • Active speaker (vo_line.speaker_id matches char): scale×1.05, opacity 1.0
#   • Inactive: scale×0.95, opacity 0.80
#   • Transition: instant cut on VO boundary (enable= expression)
#
# Audio mixing:
#   • VO at 0 dB (unity), offset with adelay
#   • Music at −6 dB un-ducked; ducked via volume=expr:eval=frame
#   • SFX at −3 dB
#   • Final loudnorm on full episode in concat pass
#
# Music continuity:
#   • Same music_asset_id across shots → seamless (seek to offset)
#   • Different music → hard cut (Phase 1; crossfade deferred)
#   • Music → no music → 0.5 s fade-out at end of shot
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
import shlex
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

# Character compositing constants
SLOT_H   = int(H * 0.55)   # 396 px — character slot height
BOTTOM_Y = int(H * 0.97)   # 698 px — feet anchor (bottom of character)

CHAR_SCALE_INACTIVE = 0.95
CHAR_SCALE_ACTIVE   = 1.05
CHAR_OPACITY_INACTIVE = 0.80

# ── Audio levels ───────────────────────────────────────────────────────────────
BASE_MUSIC_DB = -6.0   # music un-ducked level
SFX_DB        = -3.0   # SFX level

# ── Music transition durations ─────────────────────────────────────────────────
MUSIC_FADEOUT_SEC   = 0.5   # music → no-music fade (configurable via CLI)
MUSIC_CROSSFADE_SEC = 0.3   # different-music boundary (Phase 1: not implemented)

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

# Flags applied to all encoding passes for determinism
BITEXACT_FLAGS = ["-fflags", "+bitexact", "-flags:v", "+bitexact", "-map_metadata", "-1"]


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


_VO_TAIL_MS = 2000  # fallback tail when scene tail cannot be derived from ShotList


def build_asset_map(source: dict) -> dict[str, dict]:
    """Build {asset_id → resolved_asset} from resolved_assets[]."""
    return {item["asset_id"]: item for item in source.get("resolved_assets", [])}


def _find_music_segment(segments: list, shot_start_sec: float, shot_end_sec: float) -> dict:
    """Return the first shot_overrides[] entry whose window overlaps the shot's time range.

    A segment overlaps a shot when seg.start_sec < shot_end_sec and
    (seg.end_sec is None or seg.end_sec > shot_start_sec).
    Returns {} when no segment overlaps.
    """
    for seg in segments:
        seg_start = seg.get("start_sec")
        seg_end   = seg.get("end_sec")
        if seg_start is None:
            continue
        seg_start = float(seg_start)
        if seg_start >= shot_end_sec:
            continue
        if seg_end is not None and float(seg_end) <= shot_start_sec:
            continue
        return seg
    return {}


def build_media_map(media: dict) -> dict:
    """Build media lookup from resolved_assets items (same logic as gen_render_plan.py).

    Per-shot entries (with shot_id)    → key = "bg_id:shot_id"
    Background-level entries           → key = "bg_id"
    Multi-segment entries              → grouped in "_segments" sub-dict,
                                         keyed as "bg_id:shot_id" → [sorted by segment_index]
    """
    out: dict = {}
    segments: dict = {}
    for item in media.get("items", []):
        aid = item["asset_id"]
        if "segment_index" in item:
            key = f"{aid}:{item['shot_id']}"
            segments.setdefault(key, []).append(item)
        elif "shot_id" in item:
            out[f"{aid}:{item['shot_id']}"] = item
        else:
            out[aid] = item
    for seg_list in segments.values():
        seg_list.sort(key=lambda x: x.get("segment_index", 0))
    if segments:
        out["_segments"] = segments
    return out


def compute_duck_intervals_from_vo(
    vo_items: list,
    fade_ms: int,
    shot_start_sec: float = 0.0,
) -> list:
    """Compute shot-relative music duck intervals from VO items (episode-absolute timing)."""
    raw: list = []
    for vi in vo_items:
        s = vi.get("start_sec")
        e = vi.get("end_sec")
        if s is None or e is None:
            continue
        t0 = max(0.0, ((s - shot_start_sec) * 1000 - fade_ms) / 1000.0)
        t1 =          ((e - shot_start_sec) * 1000 + fade_ms) / 1000.0
        raw.append((t0, t1))
    if not raw:
        return []
    ivs = sorted(raw, key=lambda x: x[0])
    merged = [list(ivs[0])]
    for t0, t1 in ivs[1:]:
        if t0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], t1)
        else:
            merged.append([t0, t1])
    return [[round(a, 3), round(b, 3)] for a, b in merged]


def build_episode_timeline(
    shotlist: dict,
    media_map: dict,
    vo_items: list,
    music_plan_overrides: list,
    scene_tails: dict | None = None,
    ref_dur_map: dict | None = None,
) -> list:
    """Build shot dicts for render_shot() from ShotList + VOPlan + MusicPlan.

    Builds shot dicts equivalent to what RenderPlan used to provide. Each returned dict has all fields
    render_shot() reads: shot_id, scene_id, duration_ms, background_asset_id,
    background_segments, character_asset_ids, music_asset_id, duck_intervals,
    duck_db, music_fade_sec, start_sec, render_start_sec.
    """
    vo_lookup = {v["item_id"]: v for v in vo_items}
    shots_out = []
    # Track the cumulative render position (frame-snapped) so that last_ms for
    # each shot is computed relative to where that shot actually starts in the
    # rendered video — not the stale ShotList.start_sec.  When an earlier shot
    # is VO-floor-extended beyond its ShotList duration, every subsequent
    # ShotList.start_sec becomes too small, inflating last_ms and wrongly
    # triggering the floor formula on those shots.  Using the running cumulative
    # position (same formula as render loop line 1710) keeps the two in sync.
    _cumulative_render_sec = 0.0

    for shot in shotlist.get("shots", []):
        shot_id      = shot["shot_id"]
        scene_id     = shot.get("scene_id", "")
        shot_start   = shot.get("start_sec", 0.0)  # ShotList origin — used for SFX/music/duck (ShotList frame)
        duration_sec = shot.get("duration_sec", 0.0)
        render_start_sec = _cumulative_render_sec  # episode-absolute start of this shot in the rendered video

        # ── background_asset_id + background_segments ────────────────────────
        bg_id      = shot.get("background_id")
        bg_media   = None
        bg_segments: list | None = None
        if bg_id:
            seg_key  = f"{bg_id}:{shot_id}"
            seg_list = media_map.get("_segments", {}).get(seg_key)
            if seg_list:
                bg_segments = []
                for si in seg_list:
                    uri  = si.get("uri", "")
                    ext  = Path(uri.split("://", 1)[-1] if "://" in uri else uri).suffix.lower()
                    mtyp = "video" if ext in {".mp4", ".mov", ".webm", ".mkv"} else "image"
                    se   = {
                        "asset_id":       si["asset_id"],
                        "uri":            uri,
                        "media_type":     mtyp,
                        "duration_sec":   si.get("duration_sec"),
                        "hold_sec":       si.get("hold_sec"),
                        "animation_type": si.get("animation_type"),
                    }
                    _si_in  = si.get("clip_in")  if si.get("clip_in")  is not None else si.get("start_sec")
                    _si_out = si.get("clip_out") if si.get("clip_out") is not None else si.get("end_sec")
                    if _si_in is not None:
                        se["start_sec"] = _si_in
                    if _si_out is not None:
                        se["end_sec"]       = _si_out
                        se["duration_sec"]  = _si_out - se.get("start_sec", 0.0)
                    bg_segments.append(se)
                bg_media = seg_list[0]
            else:
                bg_media = media_map.get(f"{bg_id}:{shot_id}") or media_map.get(bg_id)
        background_asset_id = bg_media["asset_id"] if bg_media else None

        # ── character_asset_ids ───────────────────────────────────────────────
        char_ids: list[str] = []
        for char in shot.get("characters", []):
            cid = char.get("character_id")
            if cid and cid in media_map:
                char_ids.append(media_map[cid]["asset_id"])

        # ── music_asset_id from MusicPlan shot_overrides[] ───────────────────
        # v2 schema: segments carry episode-absolute start_sec/end_sec; find
        # the first segment whose window overlaps this shot's time range.
        _shot_end_sec = shot_start + duration_sec
        _ovr = _find_music_segment(music_plan_overrides, shot_start, _shot_end_sec)
        music_asset_id = _ovr.get("music_asset_id")

        # ── duration_ms — three-condition formula ─────────────────────────────
        _vo_ids  = shot.get("audio_intent", {}).get("vo_item_ids", [])
        _shot_vo = [vo_lookup[i] for i in _vo_ids
                    if i in vo_lookup and vo_lookup[i].get("end_sec") is not None]
        base_ms  = round(duration_sec * 1000)
        if _shot_vo:
            last_ms  = round((max(v["end_sec"] for v in _shot_vo) - _cumulative_render_sec) * 1000)
            # scene_tail_ms: three-level resolution
            _tail_ms = _VO_TAIL_MS
            if duration_sec:
                _derived = round(duration_sec * 1000) - last_ms
                if _derived > 0:
                    _tail_ms = _derived
            if scene_tails:
                _tail_ms = int(scene_tails.get(scene_id, scene_tails.get(shot_id, _tail_ms)))
            duration_ms = max(last_ms + _tail_ms, base_ms)
        else:
            duration_ms = base_ms
        if ref_dur_map:
            duration_ms = max(duration_ms, ref_dur_map.get(shot_id, 0))

        # ── duck_intervals from VO timing ─────────────────────────────────────
        _fade_sec = float(_ovr.get("fade_sec", 0.15))
        _fade_ms  = round(_fade_sec * 1000)
        # Duck intervals are in the ShotList coordinate frame: music and SFX
        # are placed using ShotList.start_sec as origin, so duck intervals must
        # also use that origin to remain aligned with the music stream.
        duck_intervals = compute_duck_intervals_from_vo(_shot_vo, _fade_ms, shot_start)

        shots_out.append({
            "shot_id":             shot_id,
            "scene_id":            scene_id,
            "start_sec":           shot_start,
            "render_start_sec":    render_start_sec,
            "duration_ms":         duration_ms,
            "background_asset_id": background_asset_id,
            "background_segments": bg_segments,
            "character_asset_ids": char_ids,
            "music_asset_id":      music_asset_id,
            "duck_intervals":      duck_intervals,
            "duck_db":             float(_ovr.get("duck_db", -12.0)),
            "music_fade_sec":      _fade_sec,
        })

        # Advance the cumulative render clock (frame-snapped, same formula as
        # render loop line 1710) so the next shot's last_ms origin is correct.
        _cumulative_render_sec += round(duration_ms * FPS / 1000) / FPS

    return shots_out


def load_vo_timing(manifest_path: Path, shotlist_path: Path) -> dict:
    """Load VO timing from VOPlan.{locale}.json + ShotList.json.

    Returns {shot_id: [{item_id, speaker_id, text,
                         shot_rel_in_ms, shot_rel_out_ms,
                         audio_chunk_uri, audio_start_sec, audio_end_sec}]}

    shot_rel_in_ms / shot_rel_out_ms are milliseconds relative to shot start.
    audio_chunk_uri / audio_start_sec / audio_end_sec are optional chunk-WAV fields.
    """
    result: dict = {}

    if not manifest_path.exists():
        print(f"  [render] WARNING: VOPlan not found: {manifest_path}")
        return result
    if not shotlist_path.exists():
        print(f"  [render] WARNING: ShotList not found: {shotlist_path}")
        return result

    manifest  = load_json(manifest_path)
    shotlist  = load_json(shotlist_path)
    vo_lookup = {v["item_id"]: v for v in manifest.get("vo_items", [])}

    for shot in shotlist.get("shots", []):
        sid        = shot.get("shot_id", "")
        shot_start = shot.get("start_sec", 0.0)
        lines = []
        seen: set = set()
        for iid in shot.get("audio_intent", {}).get("vo_item_ids", []):
            v = vo_lookup.get(iid)
            if not v:
                continue
            start_sec = v.get("start_sec")
            end_sec   = v.get("end_sec")
            if start_sec is None or end_sec is None:
                continue
            # Deduplicate by (speaker_id, text) within shot
            dedup = (v.get("speaker_id", ""), v.get("text", "").strip())
            if dedup in seen:
                continue
            seen.add(dedup)
            entry = {
                "item_id":          iid,
                "speaker_id":       v.get("speaker_id", ""),
                "text":             v.get("text", ""),
                "ep_start_sec":     start_sec,                              # episode-absolute
                "ep_end_sec":       end_sec,                                # episode-absolute
                "shot_rel_in_ms":   0,  # legacy field — not used; ep_start_sec/ep_end_sec are authoritative
                "shot_rel_out_ms":  0,  # legacy field — not used; ep_start_sec/ep_end_sec are authoritative
            }
            # Optional chunk-WAV fields (Phase 3 deferred slicing)
            if v.get("audio_chunk_uri"):
                entry["audio_chunk_uri"]  = v["audio_chunk_uri"]
                entry["audio_start_sec"]  = v.get("audio_start_sec", 0.0)
                entry["audio_end_sec"]    = v.get("audio_end_sec",   0.0)
            lines.append(entry)
        result[sid] = lines

    total = sum(len(v) for v in result.values())
    print(f"  [render] VO timing loaded: {total} lines across {len(result)} shots")
    return result


def run_ffmpeg(cmd: list[str], verbose: bool = False) -> None:
    """Run an FFmpeg command; print stderr and exit on failure."""
    if verbose:
        print("  $ " + shlex.join(cmd))
    result = subprocess.run(cmd, capture_output=not verbose, text=True)
    if result.returncode != 0:
        print(f"\n[ERROR] FFmpeg exited with code {result.returncode}", file=sys.stderr)
        if not verbose:
            # Print last 4 000 chars of stderr for diagnostics
            stderr_tail = result.stderr[-4000:] if result.stderr else "(no stderr)"
            print(stderr_tail, file=sys.stderr)
        sys.exit(1)


# ── Geometry helpers ───────────────────────────────────────────────────────────

def get_slot_geometry(n_chars: int, idx: int) -> tuple[int, int]:
    """
    Return (slot_center_x, slot_width) for character at index ``idx``
    in a layout of ``n_chars`` characters.

    Layout rules (W = 1280):
      1 char  → centred (cx=640, slot_w=1280)
      2 chars → left-third / right-third; middle third empty
      3 chars → even thirds
      4+ chars→ even quarters (first row only)
    """
    if n_chars == 1:
        return W // 2, W
    elif n_chars == 2:
        slot_w = W // 3        # 426
        # Left char: centre of left-third (slot_w//2 = 213)
        # Right char: centre of right-third (W - slot_w//2 = 1067)
        centers = [slot_w // 2, W - slot_w // 2]
        return centers[min(idx, 1)], slot_w
    elif n_chars == 3:
        slot_w = W // 3
        centers = [slot_w // 2 + i * slot_w for i in range(3)]
        return centers[min(idx, 2)], slot_w
    else:
        n_cols  = min(n_chars, 4)
        slot_w  = W // n_cols
        col_idx = idx % n_cols
        return slot_w // 2 + col_idx * slot_w, slot_w


# ── FFmpeg expression builders ─────────────────────────────────────────────────

def build_enable_expr(vo_lines: list[dict], speaker_id: str,
                       shot_offset_sec: float = 0.0) -> str:
    """
    Build an FFmpeg ``enable=`` expression that evaluates to non-zero
    whenever ``speaker_id`` is the active VO speaker.

    vo_lines entries carry ep_start_sec / ep_end_sec (episode-absolute seconds).
    shot_offset_sec is the episode-absolute start of the current shot
    (shot["render_start_sec"]) so that the enable window is expressed as
    seconds-into-the-shot, which is what FFmpeg's ``t`` variable measures
    inside a shot filter graph.

    Returns '0' if the speaker never appears.
    """
    windows = [vl for vl in vo_lines if vl.get("speaker_id") == speaker_id]
    if not windows:
        return "0"
    parts = [
        f"between(t,{max(0.0, vl['ep_start_sec'] - shot_offset_sec):.3f},"
        f"{max(0.0, vl['ep_end_sec'] - shot_offset_sec):.3f})"
        for vl in windows
    ]
    return "+".join(parts)


def build_duck_expr(
    duck_intervals: list[list[float]],
    duck_db: float,
    fade_sec: float,
    base_db: float = BASE_MUSIC_DB,
) -> str:
    """
    Build an FFmpeg ``volume=`` expression for music ducking.

    Returns a linear amplitude multiplier expression.

    Outside all duck intervals: base_amp = 10^(base_db / 20)
    Inside a duck interval [t0, t1] (already fade-padded by gen_render_plan):
      - t0 … t0+fade_sec : linear ramp base_amp → duck_amp
      - t0+fade_sec … t1−fade_sec : hold at duck_amp
      - t1−fade_sec … t1 : linear ramp duck_amp → base_amp
    where duck_amp = 10^(duck_db / 20).
    """
    base_amp = 10 ** (base_db / 20.0)
    # duck_db is attenuation RELATIVE to the base (spec: "ramp from 1.0 → duck_linear").
    # The un-ducked volume in the expression is base_amp (= -6 dB).
    # The ducked volume is base_amp × 10^(duck_db/20), i.e. -6 dB + duck_db.
    # e.g. duck_db=-12 → ducked at -18 dB total, giving a 12 dB attenuation.
    duck_amp = base_amp * (10 ** (duck_db / 20.0))

    if not duck_intervals:
        return f"{base_amp:.6f}"

    def _interval_expr(t0: float, t1: float, fade: float) -> str:
        fade = min(fade, (t1 - t0) / 2.0)  # clamp if interval is very short
        t_fi_end  = t0 + fade               # fade-in end
        t_fo_start = t1 - fade              # fade-out start
        ba = f"{base_amp:.6f}"
        da = f"{duck_amp:.6f}"
        fade_in  = f"{ba}+({da}-{ba})*(t-{t0:.3f})/{fade:.3f}"
        hold     = da
        fade_out = f"{da}+({ba}-{da})*(t-{t_fo_start:.3f})/{fade:.3f}"
        return (
            f"if(lte(t,{t_fi_end:.3f}),{fade_in},"
            f"if(lte(t,{t_fo_start:.3f}),{hold},{fade_out}))"
        )

    # Build nested if(between(t, t0, t1), inner_expr, fallback)
    expr = f"{base_amp:.6f}"  # default: un-ducked
    for t0, t1 in reversed(duck_intervals):
        inner = _interval_expr(t0, t1, fade_sec)
        expr  = f"if(between(t,{t0:.3f},{t1:.3f}),{inner},{expr})"
    return expr


# ── Animation filter builder ──────────────────────────────────────────────────

def _build_anim_filter(
    anim_type: str | None,
    in_label:  str,
    out_label: str,
    dur_sec:   float,
    W: int,
    H: int,
    fps: int,
) -> str | None:
    """
    Return an ffmpeg filter-graph fragment that animates a looped still image
    for `dur_sec` seconds, scaling from W×H input to W×H output.

    Returns None when anim_type is None / "none" (caller uses plain scale+crop).

    Supported values
    ----------------
    zoom_in   – slow Ken-Burns-style zoom 1.0 → 1.3
    zoom_out  – slow zoom 1.3 → 1.0
    pan_lr    – pan left→right (translate -10 % → +10 % of width)
    pan_rl    – pan right→left (+10 % → -10 %)
    pan_up    – pan bottom→top (+10 % → -10 % of height)
    ken_burns – combined zoom-in + diagonal pan (bottom-left → centre)
    """
    if not anim_type or anim_type == "none":
        return None

    n_frames = int(dur_sec * fps)
    if n_frames < 1:
        n_frames = 1

    # zoompan filter operates on each frame independently:
    #   z  = zoom expression (relative to input dimensions, must be ≥ 1)
    #   x  = crop origin-x expression (pixels in zoomed space)
    #   y  = crop origin-y expression
    #   d  = total duration in frames
    #   s  = output size
    # 'on' is the current output frame counter (0-based).
    # 'iw'/'ih' are input width/height.
    d = n_frames
    s = f"{W}x{H}"

    if anim_type == "zoom_in":
        z = f"1.0+0.3*on/{d}"
        x = f"iw/2*(1-1/zoom)"
        y = f"ih/2*(1-1/zoom)"
    elif anim_type == "zoom_out":
        z = f"1.3-0.3*on/{d}"
        x = f"iw/2*(1-1/zoom)"
        y = f"ih/2*(1-1/zoom)"
    elif anim_type == "pan_lr":
        # Fixed zoom 1.2, pan x from −10 % to +10 % of (iw*(zoom-1))
        z = "1.2"
        x = f"(iw*(zoom-1)/2) * (-1 + 2.0*on/{d})"
        y = "ih/2*(1-1/zoom)"
    elif anim_type == "pan_rl":
        z = "1.2"
        x = f"(iw*(zoom-1)/2) * (1 - 2.0*on/{d})"
        y = "ih/2*(1-1/zoom)"
    elif anim_type == "pan_up":
        z = "1.2"
        x = "iw/2*(1-1/zoom)"
        y = f"(ih*(zoom-1)/2) * (1 - 2.0*on/{d})"
    elif anim_type == "ken_burns":
        z = f"1.0+0.28*on/{d}"
        # Pan diagonally: start at bottom-left corner, drift toward centre
        x = f"iw*(zoom-1) * (1-on/({d}.0))"
        y = f"ih*(zoom-1) * (1-on/({d}.0))"
    else:
        return None  # unknown type → static

    # trim+setpts after zoompan: zoompan does not signal EOF cleanly when used
    # inside a concat filter on looped still-image inputs.  Without the explicit
    # trim the last frame is held indefinitely and the concat never switches to
    # the next segment.  trim=duration caps the output; setpts=PTS-STARTPTS
    # resets timestamps to 0 so the concat filter stitches correctly.
    dur_sec_frag = dur_sec  # captured by the outer call; d/fps gives the same value
    frag = (
        f"{in_label}scale={W*2}:{H*2}:force_original_aspect_ratio=increase,"
        f"crop={W*2}:{H*2},"
        f"zoompan=z='{z}':x='{x}':y='{y}':d={d}:s={s}:fps={fps},"
        f"trim=duration={dur_sec_frag:.3f},setpts=PTS-STARTPTS,"
        f"setsar=1{out_label}"
    )
    return frag


# ── Per-shot renderer ─────────────────────────────────────────────────────────

def render_shot(
    shot:               dict,
    vo_lines:           list,
    asset_map:          dict[str, dict],
    shot_index:         int,
    shots_dir:          Path,
    fps:                int,
    profile:            dict,
    music_start_sec:    float = 0.0,
    music_apply_fadeout: bool = False,
    music_fadeout_sec:  float = MUSIC_FADEOUT_SEC,
    no_music:           bool  = False,
    verbose:            bool  = False,
    music_plan_data:    dict  = None,
    sfx_plan_override:  dict  = None,
) -> Path:
    """
    Render one shot to an MKV intermediate.
    Returns the path to the output MKV (created or already-existing).
    """
    shot_id = shot["shot_id"]
    dur_ms  = shot["duration_ms"]   # authoritative: set by gen_render_plan (VO ceiling)
    # Snap to the nearest video frame so -t is always a whole-frame multiple of 1/fps.
    # Without snapping, dur_ms/1000 is not generally a multiple of 1/fps; ffmpeg can
    # only encode whole frames, so each shot's video ends up to 41 ms short of its
    # audio.  Across many shots the shortfall accumulates to seconds of video/audio
    # desync in the final MP4.
    dur_sec = round(dur_ms * fps / 1000) / fps

    out_path = shots_dir / f"{shot_index:04d}_{shot_id}.mkv"
    if out_path.exists():
        print(f"  [skip] {shot_id} (already rendered)")
        return out_path

    # inputs[i] = (extra_input_args: list[str], source: str)
    inputs: list[tuple[list[str], str]] = []
    filter_parts: list[str] = []

    def add_input(extra_args: list[str], source: str) -> int:
        idx = len(inputs)
        inputs.append((extra_args, source))
        return idx

    # ── 1. Background ──────────────────────────────────────────────────────
    bg_segments = shot.get("background_segments")
    bg_id   = shot.get("background_asset_id")
    bg_info = asset_map.get(bg_id, {}) if bg_id else {}
    bg_uri  = bg_info.get("uri", "")
    bg_path = uri_to_path(bg_uri)
    bg_is_video = bg_path and bg_path.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov", ".avi")

    if bg_segments and len(bg_segments) >= 1:
        # ── Confirmed background segments (single clip v2, or multi-segment v3) ──
        seg_labels: list[str] = []
        for si, seg in enumerate(bg_segments):
            seg_uri  = seg.get("uri", "")
            seg_path = uri_to_path(seg_uri)
            seg_type = seg.get("media_type", "image")

            # ── Segment field guards ──────────────────────────────────────────
            if seg_type == "image":
                _hold_guard = seg.get("hold_sec")
                if _hold_guard is None or float(_hold_guard) <= 0:
                    raise ValueError(
                        f"render_shot: {shot_id} bg_segments[{si}] is an image segment "
                        f"with hold_sec={_hold_guard!r} — must be > 0"
                    )
            else:
                _ci_guard = seg.get("clip_in") if seg.get("clip_in") is not None else seg.get("start_sec")
                _co_guard = seg.get("clip_out") if seg.get("clip_out") is not None else seg.get("end_sec")
                if _ci_guard is None or _co_guard is None or float(_co_guard) <= float(_ci_guard):
                    raise ValueError(
                        f"render_shot: {shot_id} bg_segments[{si}] is a video segment "
                        f"with clip_in={_ci_guard!r} clip_out={_co_guard!r} — "
                        "clip_out must be > clip_in and neither may be None"
                    )

            if seg_type == "video" and seg_path and seg_path.exists():
                # duration_override_sec: user-specified trim (play first N seconds).
                # Falls back to natural duration_sec, then full shot dur_sec.
                seg_dur = seg.get("duration_override_sec") or seg.get("duration_sec") or dur_sec
                # Clip trim range: clip_in / clip_out allow sub-clip selection.
                # Falls back to start_sec / end_sec for backward compat.
                # Uses ffmpeg trim filter (frame-accurate) rather than input-level -ss.
                seg_start = float(seg.get("clip_in") if seg.get("clip_in") is not None else (seg.get("start_sec") or 0))
                # Don't loop — use natural duration, trim if needed
                seg_idx = add_input([], str(seg_path))
                if seg_start > 0:
                    # Frame-accurate seek + trim via filter
                    trim_frag = f"trim=start={seg_start:.3f}:duration={seg_dur:.3f}"
                else:
                    trim_frag = f"trim=duration={seg_dur:.3f}"
                filter_parts.append(
                    f"[{seg_idx}:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
                    f"crop={W}:{H},setsar=1,{trim_frag},"
                    f"setpts=PTS-STARTPTS[seg{si}]"
                )
            elif seg_path and seg_path.exists():
                seg_dur = seg.get("hold_sec") or dur_sec
                seg_idx = add_input(["-loop", "1", "-t", f"{seg_dur:.3f}"], str(seg_path))
                anim_frag = _build_anim_filter(
                    seg.get("animation_type"), f"[{seg_idx}:v]", f"[seg{si}]",
                    seg_dur, W, H, fps,
                )
                if anim_frag:
                    filter_parts.append(anim_frag)
                else:
                    filter_parts.append(
                        f"[{seg_idx}:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
                        f"crop={W}:{H},setsar=1[seg{si}]"
                    )
            else:
                # Placeholder segment (seg_path is None or file missing)
                # Use explicit None checks so hold_sec=0.0 is not treated as absent.
                _h = seg.get("hold_sec")
                _d = seg.get("duration_sec")
                seg_dur = (_h if _h is not None else (_d if _d is not None else 2.0))
                if seg_dur <= 0:
                    seg_dur = seg.get("duration_override_sec") or 2.0
                print(
                    f"  [render] WARNING: segment {si} path not found "
                    f"({seg.get('uri', '?')!r}); using {seg_dur:.3f}s grey placeholder."
                )
                seg_idx = add_input(
                    ["-f", "lavfi"],
                    f"color=c=606060:size={W}x{H}:rate={fps}:duration={seg_dur:.3f}",
                )
                filter_parts.append(f"[{seg_idx}:v]copy[seg{si}]")
            seg_labels.append(f"[seg{si}]")

        # Compute cumulative durations to determine total media duration vs episode.
        # duration[i] = hold_sec (image) or clip_out - clip_in (video)
        # start[i]    = sum(duration[0..i-1])
        # end[i]      = start[i] + duration[i]
        _seg_durations: list[float] = []
        for _sd_seg in bg_segments:
            _sd_type = _sd_seg.get("media_type", "image")
            if _sd_type == "image":
                _sd_dur = float(_sd_seg.get("hold_sec", 0))
            else:
                _sd_ci = _sd_seg.get("clip_in") if _sd_seg.get("clip_in") is not None else _sd_seg.get("start_sec", 0)
                _sd_co = _sd_seg.get("clip_out") if _sd_seg.get("clip_out") is not None else _sd_seg.get("end_sec", 0)
                _sd_dur = max(0.0, float(_sd_co) - float(_sd_ci))
            _seg_durations.append(_sd_dur)
        _total_media_dur = sum(_seg_durations)

        if _total_media_dur < dur_sec - 0.001:
            # Total media shorter than episode: append black fill to reach dur_sec
            _fill_dur = dur_sec - _total_media_dur
            _fill_idx = add_input(
                ["-f", "lavfi"],
                f"color=c=black:size={W}x{H}:rate={fps}:duration={_fill_dur:.3f}",
            )
            filter_parts.append(f"[{_fill_idx}:v]copy[seg_fill]")
            seg_labels.append("[seg_fill]")
            print(f"  [render] {shot_id}: media {_total_media_dur:.3f}s < shot {dur_sec:.3f}s — "
                  f"black fill {_fill_dur:.3f}s appended")

        # Concat all segments (+ optional fill) into [bg_raw]
        n_segs = len(seg_labels)
        concat_in = "".join(seg_labels)
        concat_lbl = "bg_raw" if _total_media_dur > dur_sec + 0.001 else "bg"
        filter_parts.append(f"{concat_in}concat=n={n_segs}:v=1:a=0[{concat_lbl}]")

        if _total_media_dur > dur_sec + 0.001:
            # Total media longer than episode: trim at episode end
            filter_parts.append(
                f"[bg_raw]trim=duration={dur_sec:.3f},setpts=PTS-STARTPTS[bg]"
            )
            print(f"  [render] {shot_id}: media {_total_media_dur:.3f}s > shot {dur_sec:.3f}s — "
                  f"trimmed at {dur_sec:.3f}s")

    elif bg_path and bg_path.exists() and not bg_info.get("is_placeholder", True):
        # ── Single background (existing code, unchanged) ──
        if bg_is_video:
            # Video background: loop if short, trim to dur_sec
            bg_idx = add_input(["-stream_loop", "-1"], str(bg_path))
            filter_parts.append(
                f"[{bg_idx}:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
                f"crop={W}:{H},setsar=1,trim=duration={dur_sec:.3f},"
                f"setpts=PTS-STARTPTS[bg]"
            )
        else:
            # Static image background (single segment — bg_segments has 0 or 1 entry)
            bg_idx = add_input(["-loop", "1", "-t", f"{dur_sec:.3f}"], str(bg_path))
            single_seg = (bg_segments[0] if bg_segments else None)
            anim_frag = _build_anim_filter(
                single_seg.get("animation_type") if single_seg else None,
                f"[{bg_idx}:v]", "[bg]", dur_sec, W, H, fps,
            )
            if anim_frag:
                filter_parts.append(anim_frag)
            else:
                filter_parts.append(
                    f"[{bg_idx}:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
                    f"crop={W}:{H},setsar=1[bg]"
                )
    else:
        # Placeholder background: grey fill with label
        label = f"BG PENDING {shot_id}"
        bg_idx = add_input(
            ["-f", "lavfi"],
            f"color=c=606060:size={W}x{H}:rate={fps}:duration={dur_sec:.3f}",
        )
        filter_parts.append(
            f"[{bg_idx}:v]"
            f"drawtext=text='{label}':fontcolor=white:fontsize=32:"
            f"x=(w-text_w)/2:y=(h-text_h)/2[bg]"
        )

    # ── 2. Characters ──────────────────────────────────────────────────────
    char_ids = shot.get("character_asset_ids", [])
    n_chars  = len(char_ids)
    current_video = "[bg]"

    for ci, char_id in enumerate(char_ids):
        char_info = asset_map.get(char_id, {})
        char_uri  = char_info.get("uri", "")
        char_path = uri_to_path(char_uri)
        is_ph     = char_info.get("is_placeholder", True)

        cx, slot_w = get_slot_geometry(n_chars, ci)

        inact_w = int(slot_w * CHAR_SCALE_INACTIVE)
        inact_h = int(SLOT_H  * CHAR_SCALE_INACTIVE)
        act_w   = int(slot_w * CHAR_SCALE_ACTIVE)
        act_h   = int(SLOT_H  * CHAR_SCALE_ACTIVE)

        inact_lbl = f"c{ci}i"
        act_lbl   = f"c{ci}a"

        # Compute enable_expr BEFORE adding any active-stream filters so we
        # can skip [c{ci}a] entirely when the character never speaks.
        # An unconnected filter output causes FFmpeg to abort.
        enable_expr  = build_enable_expr(vo_lines, char_id, shot["render_start_sec"])
        is_last_char = (ci == n_chars - 1)
        v_act_out    = "vout" if is_last_char else f"v{ci}a"
        has_speaking = (enable_expr != "0")

        if char_path and char_path.exists() and not is_ph:
            c_idx = add_input(["-loop", "1", "-t", f"{dur_sec:.3f}"], str(char_path))
            # Inactive: always added
            filter_parts.append(
                f"[{c_idx}:v]scale=w={inact_w}:h={inact_h}:"
                f"force_original_aspect_ratio=decrease,"
                f"format=rgba,colorchannelmixer=aa={CHAR_OPACITY_INACTIVE:.2f}[{inact_lbl}]"
            )
            # Active: only when there are speaking windows
            if has_speaking:
                filter_parts.append(
                    f"[{c_idx}:v]scale=w={act_w}:h={act_h}:"
                    f"force_original_aspect_ratio=decrease,"
                    f"format=rgba[{act_lbl}]"
                )
        else:
            # Placeholder: grey box with char_id label
            label = char_id.replace("'", "").replace("\\", "")
            p_idx = add_input(
                ["-f", "lavfi"],
                f"color=c=808080:size={inact_w}x{inact_h}:rate={fps}:duration={dur_sec:.3f}",
            )
            filter_parts.append(
                f"[{p_idx}:v]drawtext=text='{label}':fontcolor=white:fontsize=20:"
                f"x=(w-text_w)/2:y=(h-text_h)/2,"
                f"format=rgba,colorchannelmixer=aa={CHAR_OPACITY_INACTIVE:.2f}[{inact_lbl}]"
            )
            if has_speaking:
                p_idx2 = add_input(
                    ["-f", "lavfi"],
                    f"color=c=808080:size={act_w}x{act_h}:rate={fps}:duration={dur_sec:.3f}",
                )
                filter_parts.append(
                    f"[{p_idx2}:v]drawtext=text='{label}':fontcolor=white:fontsize=20:"
                    f"x=(w-text_w)/2:y=(h-text_h)/2,"
                    f"format=rgba[{act_lbl}]"
                )

        # Overlay: inactive always on
        v_inact_out = f"v{ci}i"
        filter_parts.append(
            f"{current_video}[{inact_lbl}]overlay="
            f"x={cx}-overlay_w/2:y={BOTTOM_Y}-overlay_h:"
            f"format=auto[{v_inact_out}]"
        )

        # Overlay: active during speaking windows (or passthrough if silent)
        if not has_speaking:
            # No speaking windows: rename inactive output directly
            filter_parts.append(f"[{v_inact_out}]copy[{v_act_out}]")
        else:
            filter_parts.append(
                f"[{v_inact_out}][{act_lbl}]overlay="
                f"x={cx}-overlay_w/2:y={BOTTOM_Y}-overlay_h:"
                f"enable='{enable_expr}':format=auto[{v_act_out}]"
            )

        current_video = f"[{v_act_out}]"

    if n_chars == 0:
        filter_parts.append("[bg]copy[vout]")

    # ── 3. Silence pad (ensures audio covers full shot duration) ───────────
    filter_parts.append(
        f"aevalsrc=0:c=stereo:s=48000:d={dur_sec:.3f}[silence_pad]"
    )
    all_audio: list[str] = ["[silence_pad]"]

    # ── 4. VO audio streams ────────────────────────────────────────────────
    for vo_i, vl in enumerate(vo_lines):
        line_id  = vl.get("item_id", vl.get("line_id", ""))
        # Use episode-absolute timing so VO lands at the correct position
        # even when scene_heads push start_sec beyond the ShotList boundary.
        # render_start_sec is the episode-absolute start of THIS shot in
        # the rendered video, so (ep_start_sec - render_start_sec) gives
        # the correct shot-relative delay.
        delay_ms = max(0, round((vl["ep_start_sec"] - shot["render_start_sec"]) * 1000))
        lbl      = f"vo{vo_i}"

        # Phase 3, Step 10: chunk-WAV deferred slicing path
        chunk_uri = vl.get("audio_chunk_uri", "")
        if chunk_uri:
            chunk_path = uri_to_path(chunk_uri)
            if chunk_path and chunk_path.exists():
                start_sec  = float(vl.get("audio_start_sec", 0.0))
                end_sec    = float(vl.get("audio_end_sec",   0.0))
                vo_dur_sec = max(0.001, end_sec - start_sec)
                v_idx = add_input(
                    ["-ss", f"{start_sec:.4f}", "-t", f"{vo_dur_sec:.4f}"],
                    str(chunk_path),
                )
                filter_parts.append(
                    f"[{v_idx}:a]aformat=sample_rates=48000:channel_layouts=stereo,"
                    f"adelay={delay_ms}|{delay_ms}[{lbl}]"
                )
                all_audio.append(f"[{lbl}]")
                continue  # handled — skip per-sentence WAV path below

        # Original per-sentence WAV path (backward compatible)
        vo_info = asset_map.get(line_id, {})
        vo_uri  = vo_info.get("uri", "")
        vo_path = uri_to_path(vo_uri)
        if not vo_path or not vo_path.exists() or vo_info.get("is_placeholder", True):
            continue  # missing VO → silence for that line
        v_idx = add_input([], str(vo_path))
        filter_parts.append(
            f"[{v_idx}:a]aformat=sample_rates=48000:channel_layouts=stereo,"
            f"adelay={delay_ms}|{delay_ms}[{lbl}]"
        )
        all_audio.append(f"[{lbl}]")

    # ── 5. SFX audio streams ───────────────────────────────────────────────
    sfx_amp = 10 ** (SFX_DB / 20.0)
    for sfx_i, sfx_id in enumerate(shot.get("sfx_asset_ids", [])):
        sfx_info = asset_map.get(sfx_id, {})
        sfx_uri  = sfx_info.get("uri", "")
        sfx_path = uri_to_path(sfx_uri)
        if not sfx_path or not sfx_path.exists() or sfx_info.get("is_placeholder", True):
            continue  # silence for missing SFX
        s_idx = add_input([], str(sfx_path))
        lbl   = f"sfx{sfx_i}"
        filter_parts.append(
            f"[{s_idx}:a]aformat=sample_rates=48000:channel_layouts=stereo,"
            f"volume={sfx_amp:.6f}[{lbl}]"
        )
        all_audio.append(f"[{lbl}]")

    # ── 5b. SFX plan entries (user-selected SFX with timing from SFX tab) ────
    # sfx_plan_entries each carry source_file (local path), start_sec (delay
    # from shot start), and optionally end_sec (trim length).
    #
    # SfxPlan.json is the authoritative source for SFX entries.
    # If sfx_plan_override is provided (SfxPlan.json was loaded), use it.
    # A missing key means the user placed no SFX on this shot (empty list).
    if sfx_plan_override is not None:
        _sfx_entries = sfx_plan_override.get(shot_id, [])
    else:
        # No SfxPlan available — fall back to sfx_plan_entries on the shot dict (if any).
        _sfx_entries = shot.get("sfx_plan_entries", [])
    for sp_i, sp_entry in enumerate(_sfx_entries):
        sp_vol_db      = float(sp_entry.get("volume_db",     0) or 0)
        sp_duck_db     = float(sp_entry.get("duck_db",       0) or 0)
        sp_clip_vol_db = float(sp_entry.get("clip_volume_db",0) or 0)
        sp_fade_sec    = float(sp_entry.get("fade_sec",      0) or 0)
        sfx_plan_amp   = 10 ** ((SFX_DB + sp_vol_db + sp_duck_db + sp_clip_vol_db) / 20.0)

        # Resolve source: cut clip path takes priority over source_file + start/end.
        # start_sec is always the episode-absolute placement time regardless of source.
        sp_clip_path = sp_entry.get("clip_path")
        if sp_clip_path:
            if not os.path.isabs(sp_clip_path):
                _episode_dir_rv = str(shots_dir.parent.parent.parent)
                sp_clip_path = os.path.join(_episode_dir_rv, sp_clip_path)
            sfx_source = sp_clip_path
            sp_start   = float(sp_entry.get("start_sec") or 0.0)
            sp_end     = sp_entry.get("end_sec")
        else:
            sfx_source = sp_entry.get("source_file", "")
            sp_start   = float(sp_entry.get("start_sec") or 0.0)
            sp_end     = sp_entry.get("end_sec")

        if not sfx_source:
            continue
        sp_path = Path(sfx_source)
        if not sp_path.exists():
            print(f"  [WARN] SFX plan entry missing: {sfx_source}")
            continue
        sp_start_ep  = sp_start   # episode-absolute (ShotList frame)
        sp_end_ep    = sp_end     # episode-absolute (or None)
        sp_idx       = add_input([], str(sp_path))
        lbl          = f"sfxp{sp_i}"
        # SfxPlan.start_sec and ShotList.shot.start_sec are both episode-absolute
        # in the same ShotList coordinate frame — subtract directly to get
        # shot-relative delay.  render_start_sec must NOT be used here: it
        # is in the rendered-video frame, which diverges from the ShotList
        # frame whenever an earlier shot is VO-floor-extended.
        sp_start_rel = max(0.0, sp_start_ep - shot.get("start_sec", 0.0))
        delay_ms_sp  = round(sp_start_rel * 1000)
        filt = (
            f"[{sp_idx}:a]aformat=sample_rates=48000:channel_layouts=stereo,"
            f"volume={sfx_plan_amp:.6f}"
        )
        if sp_end_ep is not None and float(sp_end_ep) > sp_start_ep:
            trim_dur = float(sp_end_ep) - sp_start_ep
            filt += f",atrim=duration={trim_dur:.3f}"
            if sp_fade_sec > 0:
                fade_out_st = max(0.0, trim_dur - sp_fade_sec)
                filt += (
                    f",afade=t=in:st=0:d={sp_fade_sec:.3f}"
                    f",afade=t=out:st={fade_out_st:.3f}:d={sp_fade_sec:.3f}"
                )
        elif sp_fade_sec > 0:
            filt += f",afade=t=in:st=0:d={sp_fade_sec:.3f}"   # fade-in only (end unknown)
        filt += f",adelay={delay_ms_sp}|{delay_ms_sp}[{lbl}]"
        filter_parts.append(filt)
        all_audio.append(f"[{lbl}]")

    # ── 6. Music audio stream ──────────────────────────────────────────────
    music_id   = shot.get("music_asset_id")
    music_info = asset_map.get(music_id, {}) if music_id else {}
    music_uri  = music_info.get("uri", "")
    music_path = uri_to_path(music_uri)

    # Prefer pre-looped WAV if it exists (generated by apply_music_plan.py)
    if music_path and music_path.exists():
        _loop_path = music_path.with_name(music_path.stem + ".loop.wav")
        if _loop_path.exists():
            music_path = _loop_path

    # --- MusicPlan override lookup (v2: shot_overrides[], episode-absolute) ---
    _pd = music_plan_data or {}
    _shot_start_sec = shot.get("start_sec", 0.0)
    _shot_dur_sec   = dur_ms / 1000.0
    _shot_end_sec   = _shot_start_sec + _shot_dur_sec
    _plan_ovr = _find_music_segment(
        _pd.get("shot_overrides", []), _shot_start_sec, _shot_end_sec
    )

    _plan_music_id = _plan_ovr.get("music_asset_id") or music_id
    _plan_provided_music = False

    if _plan_music_id:
        # Derive loop_wav_path (.loop.wav preferred, .wav fallback)
        # shots_dir is output_dir/.shots  (ep_dir/renders/{locale}/.shots)
        # → go up 3 levels to reach ep_dir
        _episode_dir_str = str(shots_dir.parent.parent.parent)
        _wav_loop = os.path.join(_episode_dir_str, "assets", "music", f"{_plan_music_id}.loop.wav")
        _wav_base = os.path.join(_episode_dir_str, "assets", "music", f"{_plan_music_id}.wav")
        if os.path.isfile(_wav_loop):
            shot["loop_wav_path"] = _wav_loop
            music_path = Path(_wav_loop)
            _plan_provided_music = True
        elif os.path.isfile(_wav_base):
            shot["loop_wav_path"] = _wav_base
            music_path = Path(_wav_base)
            _plan_provided_music = True

        # Derive base_db (additive: BASE_MUSIC_DB + track offset + clip offset)
        import re as _re
        _stem = _re.sub(r'_\d[\d_]*s-[\d_\.]+s$', '', _plan_music_id)
        _db_off = (_pd.get("track_volumes", {}).get(_stem, 0.0)
                   + _pd.get("clip_volumes",  {}).get(_plan_music_id, 0.0))
        shot["base_db"] = BASE_MUSIC_DB + _db_off

    if _plan_ovr:
        # duck_db and fade_sec from override
        _plan_duck_db = _plan_ovr.get("duck_db")
        if _plan_duck_db is not None:
            shot["duck_db"] = _plan_duck_db

        _plan_fade_sec = _plan_ovr.get("fade_sec")
        if _plan_fade_sec is not None:
            shot["fade_sec"] = _plan_fade_sec

        # music_delay_sec / music_end_sec: MusicPlan.start_sec/end_sec are
        # ShotList-absolute (same frame as ShotList.shot.start_sec).
        # Subtract shot["start_sec"] (ShotList origin) — NOT render_start_sec —
        # to get shot-relative offsets that align with duck_intervals (also
        # computed in the ShotList frame).
        _plan_start_sec = _plan_ovr.get("start_sec")
        if _plan_start_sec is not None:
            shot["music_delay_sec"] = max(0.0, float(_plan_start_sec) - shot.get("start_sec", 0.0))

        _plan_end_sec = _plan_ovr.get("end_sec")
        if _plan_end_sec is not None:
            shot["music_end_sec"] = max(0.0, float(_plan_end_sec) - shot.get("start_sec", 0.0))

        print(f"  [{shot_id}] Using MusicPlan overrides — approved timing.")

    # duck_intervals: use value computed in build_episode_timeline() (unchanged)

    if not no_music and music_path and music_path.exists() and (_plan_provided_music or not music_info.get("is_placeholder", True)):
        duck_intervals  = shot.get("duck_intervals", [])
        duck_db         = shot.get("duck_db", -12.0)
        fade_sec        = shot.get("music_fade_sec", 0.15)
        music_base_db   = shot.get("base_db", BASE_MUSIC_DB)
        music_delay_sec = shot.get("music_delay_sec", 0.0)
        music_end_sec   = shot.get("music_end_sec", None)  # None = play to shot end

        # duck_intervals from snapshot are shot-relative (t=0 = shot start).
        # The FFmpeg volume filter sees music-file-relative time (t=0 = first sample
        # of the music WAV = shot t=music_delay_sec).  Subtract the delay to align.
        _adj_duck = []
        for _t0, _t1 in duck_intervals:
            _t0a = _t0 - music_delay_sec
            _t1a = _t1 - music_delay_sec
            if _t1a <= 0:
                continue    # interval ends before music starts — skip
            _adj_duck.append([max(_t0a, 0.0), _t1a])
        duck_expr = build_duck_expr(_adj_duck, duck_db, fade_sec,
                                    base_db=music_base_db)

        # Seek to music_start_sec for seamless same-music continuation.
        # Do NOT use -stream_loop -1 here: an infinite loop combined with
        # amix duration=longest causes FFmpeg to never stop.  If the music
        # WAV is shorter than the shot, amix's silence_pad will pad the rest.
        m_extra = []
        if music_start_sec > 0.001:
            m_extra = ["-ss", f"{music_start_sec:.3f}"]
        m_idx = add_input(m_extra, str(music_path))

        music_filter = f"[{m_idx}:a]aformat=sample_rates=48000:channel_layouts=stereo,"

        # Enforce music end time: if user set music to stop before shot end,
        # trim the music clip to exactly that window duration.
        if music_end_sec is not None:
            _music_window = music_end_sec - music_delay_sec
            if 0 < _music_window < dur_sec - 0.01:
                music_filter += f"atrim=duration={_music_window:.3f},"

        music_filter += f"volume=volume='{duck_expr}':eval=frame"

        # music_delay_sec: delay before music starts within the shot
        if music_delay_sec > 0.001:
            delay_ms = round(music_delay_sec * 1000)
            music_filter += f",adelay={delay_ms}|{delay_ms}"
        if music_apply_fadeout:
            fade_start = max(0.0, dur_sec - music_fadeout_sec)
            music_filter += (
                f",afade=t=out:st={fade_start:.3f}:d={music_fadeout_sec:.3f}"
            )
        music_filter += "[music_v]"
        filter_parts.append(music_filter)
        all_audio.append("[music_v]")

    # ── 7. Audio mix ───────────────────────────────────────────────────────
    n_audio = len(all_audio)
    if n_audio == 1:
        filter_parts.append(f"{all_audio[0]}anull[aout]")
    else:
        joined = "".join(all_audio)
        filter_parts.append(
            f"{joined}amix=inputs={n_audio}:normalize=0:dropout_transition=0[aout]"
        )

    # ── 8. Build FFmpeg command ────────────────────────────────────────────
    cmd = ["ffmpeg", "-y"]
    for extra_args, source in inputs:
        cmd.extend(extra_args)
        cmd.extend(["-i", source])

    filter_complex = ";".join(filter_parts)

    prof = profile
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264",
        "-crf", str(prof["crf"]),
        "-preset", prof["preset"],
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-c:a", "pcm_s16le",
        "-ar", "48000",
        # Explicit duration cap: guarantees output is exactly dur_sec even
        # if the amix output extends past the video track end.
        # Use 6 decimal places so the frame-snapped value (e.g. 1376/24 = 57.333333…)
        # is not re-quantised to 3 decimal places, which would re-introduce a
        # sub-frame error detectable by the KW-17 regression test.
        "-t", f"{dur_sec:.6f}",
    ] + BITEXACT_FLAGS + [str(out_path)]

    print(f"  [{shot_index + 1:02d}] {shot_id}  ({dur_ms} ms, {n_chars} chars, "
          f"{len(vo_lines)} VO, music={'yes' if music_id else 'no'})")
    run_ffmpeg(cmd, verbose=verbose)
    return out_path


# ── Black frame generator ──────────────────────────────────────────────────────

def generate_black_frame(shots_dir: Path, fps: int) -> Path:
    """
    Generate a 1-frame black MKV for scene-boundary transitions.
    Written once; returned path is reused in concat list.
    """
    path    = shots_dir / "black_frame.mkv"
    dur_sec = 1.0 / fps
    if path.exists():
        return path
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=black:size={W}x{H}:rate={fps}:duration={dur_sec:.6f}",
        "-f", "lavfi",
        "-i", f"aevalsrc=0:c=stereo:s=48000:d={dur_sec:.6f}",
        "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p", "-r", str(fps),
        "-c:a", "pcm_s16le", "-ar", "48000",
        str(path),
    ]
    run_ffmpeg(cmd)
    return path


# ── Concat + loudnorm ──────────────────────────────────────────────────────────

def concat_to_mp4(
    concat_list: Path,
    output:      Path,
    profile:     dict,
) -> None:
    """
    Concat MKV intermediates (concat demuxer), apply single-pass loudnorm
    at −16 LUFS, encode AAC at 192 kbps → output.mp4.

    Video is stream-copied (no re-encode) from the per-shot MKVs.
    """
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-filter_complex",
        "[0:a]loudnorm=I=-16:LRA=11:TP=-1:linear=true[aout]",
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        "-map_metadata", "-1",
        str(output),
    ]
    print(f"\n  Concat + loudnorm → {output.name}")
    run_ffmpeg(cmd)


# ── MediaPlan-mode helpers ─────────────────────────────────────────────────────
# Used by the ShotList-free render path (main below).

def resolve_media_path(seg: dict, ep_dir: Path) -> str:
    """Return the best local filesystem path for a MediaPlan segment."""
    p = seg.get("path", "")
    if p:
        if os.path.isabs(p):
            if os.path.exists(p):
                return p
        else:
            c = ep_dir / p
            if c.exists():
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
    as returned by load_vo_timing(). When None, SRT is written empty.
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
    return p.parse_args()


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
                        print(f"  [warn] seg {_si}: black fallback")
                    continue

                _clip_files.append(_cpath)
                print(f"  [clip] seg {_si}: {_stype} {_sdur:.3f}s ok")

        if not _clip_files:
            print("[ERROR] No clips generated.", file=sys.stderr)
            sys.exit(1)

        # ── 6. Concatenate clips ─────────────────────────────────────────────
        _concat_list  = _tmp / "concat.txt"
        _concat_video = _tmp / "concat.mp4"
        with open(_concat_list, "w") as _cf:
            for _cp in _clip_files:
                _cf.write(f"file '{_cp}'\n")
        _result = subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(_concat_list), "-c", "copy", str(_concat_video)
        ], capture_output=True, text=True)
        if _result.returncode != 0:
            print(f"[ERROR] concat failed: {_result.stderr[-500:]}", file=sys.stderr)
            sys.exit(1)

        # ── 7. Build VO audio (episode-absolute from VOPlan) ─────────────────
        _vo_dir    = episode_dir / "assets" / locale / "audio" / "vo"
        _vo_audio  = _tmp / "vo_mix.wav"
        _vo_inputs: list = []
        _vo_delays: list = []

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

        if _vo_inputs:
            _vf_parts = [f"[{i}]adelay={d}|{d}[d{i}]" for i, d in enumerate(_vo_delays)]
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
                _BASE_MDB   = -6.0
                _mi, _md, _mv, _mcd = [], [], [], []
                for _ovr in _mus_doc.get("shot_overrides", []):
                    _asset = _ovr.get("music_asset_id", "")
                    _mst   = _ovr.get("start_sec")
                    _men   = _ovr.get("end_sec")
                    if _mst is None:
                        continue
                    _mdms  = max(0.0, float(_mst)) * 1000
                    _mcdur = (max(0.0, float(_men) - float(_mst))
                              if _men is not None else None)
                    _wl    = episode_dir / "assets" / "music" / f"{_asset}.loop.wav"
                    _wb    = episode_dir / "assets" / "music" / f"{_asset}.wav"
                    _wav   = (str(_wl) if _wl.exists()
                              else str(_wb) if _wb.exists() else "")
                    if not _wav:
                        print(f"  [WARN] music wav not found: {_asset}")
                        continue
                    _stem   = re.sub(r'_\d[\d_]*s-[\d_\.]+s$', '', _asset)
                    _db_off = _track_vol.get(_stem, 0.0) + _clip_vol.get(_asset, 0.0)
                    _vol    = 10 ** ((_BASE_MDB + _db_off) / 20.0)
                    _mi.append(_wav)
                    _md.append(_mdms)
                    _mv.append(_vol)
                    _mcd.append(_mcdur)
                    print(f"  [music] {_asset}: delay={_mdms:.0f}ms clip_dur={_mcdur}")
                if _mi:
                    _music_audio = _tmp / "music_mix.wav"
                    _mf_parts = []
                    for _idx, (_d, _v, _cd) in enumerate(zip(_md, _mv, _mcd)):
                        if _cd is not None:
                            _mf_parts.append(
                                f"[{_idx}]atrim=duration={_cd:.3f},volume={_v:.4f},"
                                f"adelay={_d:.0f}|{_d:.0f}[m{_idx}]")
                        else:
                            _mf_parts.append(
                                f"[{_idx}]adelay={_d:.0f}|{_d:.0f},volume={_v:.4f}[m{_idx}]")
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
            for _cc in _sfx_plan.get("cut_clips", []):
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
