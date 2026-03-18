#!/usr/bin/env python3
# =============================================================================
# render_video.py — Produce output.mp4 from VOPlan.{locale}.json + ShotList.json
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
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

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


def build_shots_for_render(
    shotlist: dict,
    media_map: dict,
    vo_items: list,
    music_plan_overrides: dict,
    scene_tails: dict | None = None,
    ref_dur_map: dict | None = None,
) -> list:
    """Build shot dicts for render_shot() from ShotList + VOPlan + MusicPlan.

    Builds shot dicts equivalent to what RenderPlan used to provide. Each returned dict has all fields
    render_shot() reads: shot_id, scene_id, duration_ms, background_asset_id,
    background_segments, character_asset_ids, music_asset_id, duck_intervals,
    duck_db, music_fade_sec, start_sec.
    """
    vo_lookup = {v["item_id"]: v for v in vo_items}
    shots_out = []

    for shot in shotlist.get("shots", []):
        shot_id      = shot["shot_id"]
        scene_id     = shot.get("scene_id", "")
        shot_start   = shot.get("start_sec", 0.0)
        duration_sec = shot.get("duration_sec", 0.0)

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
                    if si.get("start_sec") is not None:
                        se["start_sec"] = si["start_sec"]
                    if si.get("end_sec") is not None:
                        se["end_sec"]       = si["end_sec"]
                        se["duration_sec"]  = si["end_sec"] - se.get("start_sec", 0.0)
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

        # ── music_asset_id from MusicPlan override ───────────────────────────
        _ovr           = music_plan_overrides.get(shot_id, {})
        music_asset_id = _ovr.get("music_asset_id")

        # ── duration_ms — three-condition formula ─────────────────────────────
        _vo_ids  = shot.get("audio_intent", {}).get("vo_item_ids", [])
        _shot_vo = [vo_lookup[i] for i in _vo_ids
                    if i in vo_lookup and vo_lookup[i].get("end_sec") is not None]
        base_ms  = round(duration_sec * 1000)
        if _shot_vo:
            last_ms  = round((max(v["end_sec"] for v in _shot_vo) - shot_start) * 1000)
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
        duck_intervals = compute_duck_intervals_from_vo(_shot_vo, _fade_ms, shot_start)

        shots_out.append({
            "shot_id":             shot_id,
            "scene_id":            scene_id,
            "start_sec":           shot_start,
            "duration_ms":         duration_ms,
            "background_asset_id": background_asset_id,
            "background_segments": bg_segments,
            "character_asset_ids": char_ids,
            "music_asset_id":      music_asset_id,
            "duck_intervals":      duck_intervals,
            "duck_db":             float(_ovr.get("duck_db", -12.0)),
            "music_fade_sec":      _fade_sec,
        })

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
                "shot_rel_in_ms":   round((start_sec - shot_start) * 1000),
                "shot_rel_out_ms":  round((end_sec   - shot_start) * 1000),
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

def build_enable_expr(vo_lines: list[dict], speaker_id: str) -> str:
    """
    Build an FFmpeg ``enable=`` expression that evaluates to non-zero
    whenever ``speaker_id`` is the active VO speaker.

    vo_lines entries use shot_rel_in_ms / shot_rel_out_ms (milliseconds,
    shot-relative) as written by load_vo_timing().

    Returns '0' if the speaker never appears.
    """
    windows = [vl for vl in vo_lines if vl.get("speaker_id") == speaker_id]
    if not windows:
        return "0"
    parts = [
        f"between(t,{vl['shot_rel_in_ms'] / 1000:.3f},{vl['shot_rel_out_ms'] / 1000:.3f})"
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
    _cumulative_shot_sec: float = 0.0,
) -> Path:
    """
    Render one shot to an MKV intermediate.
    Returns the path to the output MKV (created or already-existing).
    """
    shot_id = shot["shot_id"]
    dur_ms  = shot["duration_ms"]   # authoritative: set by gen_render_plan (VO ceiling)
    dur_sec = dur_ms / 1000.0

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

    if bg_segments and len(bg_segments) > 1:
        # ── Multi-segment background (v3): concat filter, no looping ──
        seg_labels: list[str] = []
        for si, seg in enumerate(bg_segments):
            seg_uri  = seg.get("uri", "")
            seg_path = uri_to_path(seg_uri)
            seg_type = seg.get("media_type", "image")

            if seg_type == "video" and seg_path and seg_path.exists():
                # duration_override_sec: user-specified trim (play first N seconds).
                # Falls back to natural duration_sec, then full shot dur_sec.
                seg_dur = seg.get("duration_override_sec") or seg.get("duration_sec") or dur_sec
                # Clip trim range: start_sec / end_sec allow sub-clip selection.
                # Uses ffmpeg trim filter (frame-accurate) rather than input-level -ss.
                seg_start = float(seg.get("start_sec") or 0)
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

        # Concat all segments into [bg]
        n_segs = len(seg_labels)
        concat_in = "".join(seg_labels)
        filter_parts.append(f"{concat_in}concat=n={n_segs}:v=1:a=0[bg]")

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
        enable_expr  = build_enable_expr(vo_lines, char_id)
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
        delay_ms = vl["shot_rel_in_ms"]
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
    sfx_plan_amp = 10 ** (SFX_DB / 20.0)
    for sp_i, sp_entry in enumerate(_sfx_entries):
        sp_path_str = sp_entry.get("source_file", "")
        if not sp_path_str:
            continue
        sp_path = Path(sp_path_str)
        if not sp_path.exists():
            print(f"  [WARN] SFX plan entry missing: {sp_path_str}")
            continue
        sp_start_sec = float(sp_entry.get("start_sec") or 0.0)
        sp_end_sec   = sp_entry.get("end_sec")
        sp_idx       = add_input([], str(sp_path))
        lbl          = f"sfxp{sp_i}"
        delay_ms_sp  = round(sp_start_sec * 1000)
        filt = (
            f"[{sp_idx}:a]aformat=sample_rates=48000:channel_layouts=stereo,"
            f"volume={sfx_plan_amp:.6f}"
        )
        if sp_end_sec is not None and float(sp_end_sec) > sp_start_sec:
            trim_dur = float(sp_end_sec) - sp_start_sec
            filt += f",atrim=duration={trim_dur:.3f}"
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

    # --- MusicPlan override lookup ---
    _pd = music_plan_data or {}
    _shot_id_key = shot.get("shot_id") or shot.get("id", "")
    _plan_ovr = _pd.get("plan_overrides", {}).get(_shot_id_key, {})

    _plan_music_id = _plan_ovr.get("music_asset_id") or music_id
    _plan_provided_music = False

    if _plan_music_id:
        # Derive loop_wav_path (.loop.wav preferred, .wav fallback)
        # shots_dir is output_dir/.shots; episode_dir is plan_path.parent (2 levels up from shots_dir)
        _episode_dir_str = str(shots_dir.parent.parent)
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

        # music_delay_sec: episode-absolute start_sec minus cumulative shot offset
        _plan_start_sec = _plan_ovr.get("start_sec")
        if _plan_start_sec is not None:
            shot["music_delay_sec"] = max(0.0, float(_plan_start_sec) - _cumulative_shot_sec)

        # music_end_sec: episode-absolute end_sec minus cumulative shot offset
        _plan_end_sec = _plan_ovr.get("end_sec")
        if _plan_end_sec is not None:
            shot["music_end_sec"] = max(0.0, float(_plan_end_sec) - _cumulative_shot_sec)

        print(f"  [{_shot_id_key}] Using MusicPlan overrides — approved timing.")

    # duck_intervals: use value computed in build_shots_for_render() (unchanged)

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
        "-t", f"{dur_sec:.3f}",
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
    seq        = 1
    offset_ms  = 0
    frame_ms   = round(1000 / fps)

    for i, shot in enumerate(shots):
        shot_id = shot.get("shot_id", "")
        for vl in (vo_timing_by_shot or {}).get(shot_id, []):
            text = vl.get("text", "").strip()
            if not text:
                continue
            abs_in  = offset_ms + vl["shot_rel_in_ms"]
            abs_out = offset_ms + vl["shot_rel_out_ms"]
            timecode = f"{ms_to_srt_ts(abs_in)} --> {ms_to_srt_ts(abs_out)}"
            lines += [str(seq), timecode, text, ""]
            subs.append({
                "line_id":  vl.get("item_id", vl.get("line_id", "")),
                "timecode": timecode,
                "text":     text,
            })
            seq += 1

        offset_ms += shot["duration_ms"]

        # Add black frame duration at scene boundaries
        if i < len(shots) - 1:
            if shots[i + 1].get("scene_id") != shot.get("scene_id"):
                offset_ms += frame_ms

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

    shots_dir = output_dir / ".shots"
    shots_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ShotList.json ───────────────────────────────────────────────────
    _shotlist_path = episode_dir / "ShotList.json"
    if not _shotlist_path.exists():
        print(f"[ERROR] ShotList.json not found: {_shotlist_path}", file=sys.stderr)
        sys.exit(1)
    shotlist = load_json(_shotlist_path)

    # ── Plan-hash cache invalidation (keyed on ShotList timing_lock_hash) ───
    _plan_hash = (shotlist.get("timing_lock_hash")
                  or hashlib.md5(json.dumps(shotlist, sort_keys=True).encode()).hexdigest())
    _hash_file = shots_dir / ".plan_hash"
    if _hash_file.exists() and _hash_file.read_text().strip() != _plan_hash:
        print(f"  [render] ShotList changed — clearing cached shots")
        shutil.rmtree(shots_dir)
        shots_dir.mkdir(parents=True, exist_ok=True)
    _hash_file.write_text(_plan_hash)

    fps          = FPS
    profile_name = args.profile or DEFAULT_PROFILE
    profile      = PROFILES.get(profile_name, PROFILES[DEFAULT_PROFILE])

    # ── Build media_map + asset_map from VOPlan.resolved_assets[] ───────────
    _media_shim = {"items": voplan.get("resolved_assets", [])}
    media_map   = build_media_map(_media_shim)
    asset_map   = build_asset_map(voplan)

    # ── EN locale reference floor (--reference-approval) ────────────────────
    _ref_dur_map: dict = {}
    if args.reference_approval:
        _ref_path = Path(args.reference_approval).resolve()
        if not _ref_path.exists():
            print(f"[ERROR] --reference-approval file not found: {_ref_path}", file=sys.stderr)
            sys.exit(1)
        try:
            _ref_voplan  = load_json(_ref_path)
            _ref_vo      = _ref_voplan.get("vo_items", [])
            _ref_shotlist_path = _ref_path.parent / "ShotList.json"
            _ref_shotlist = load_json(_ref_shotlist_path) if _ref_shotlist_path.exists() else shotlist
            # Build ref duration_ms for each shot using same three-condition formula
            _ref_vo_lookup = {v["item_id"]: v for v in _ref_vo}
            _ref_scene_tails = _ref_voplan.get("scene_tails")
            for _rs in _ref_shotlist.get("shots", []):
                _rs_id   = _rs["shot_id"]
                _rs_st   = _rs.get("start_sec", 0.0)
                _rs_dur  = _rs.get("duration_sec", 0.0)
                _rs_vids = _rs.get("audio_intent", {}).get("vo_item_ids", [])
                _rs_vo   = [_ref_vo_lookup[i] for i in _rs_vids
                            if i in _ref_vo_lookup and _ref_vo_lookup[i].get("end_sec") is not None]
                _rs_base = round(_rs_dur * 1000)
                if _rs_vo:
                    _rs_last = round((max(v["end_sec"] for v in _rs_vo) - _rs_st) * 1000)
                    _rs_tail = _VO_TAIL_MS
                    if _rs_dur:
                        _d = round(_rs_dur * 1000) - _rs_last
                        if _d > 0:
                            _rs_tail = _d
                    if _ref_scene_tails:
                        _sc = _rs.get("scene_id", "")
                        _rs_tail = int(_ref_scene_tails.get(_sc, _ref_scene_tails.get(_rs_id, _rs_tail)))
                    _ref_dur_map[_rs_id] = max(_rs_last + _rs_tail, _rs_base)
                else:
                    _ref_dur_map[_rs_id] = _rs_base
            print(f"  [render] EN floor loaded from {_ref_path.name} — {len(_ref_dur_map)} shots")
        except Exception as _ref_err:
            print(f"  [render] WARNING: Could not load --reference-approval: {_ref_err}")

    # ── Build shots list from ShotList + VOPlan + MusicPlan ─────────────────
    # (MusicPlan is loaded first so music_asset_id is available per shot)

    # --- Load MusicPlan.json (replaces MusicApprovalSnapshot.json) ---
    _music_plan_overrides = {}
    _music_clip_volumes   = {}
    _music_track_volumes  = {}
    _mp_path = os.path.join(episode_dir, "MusicPlan.json")
    if os.path.isfile(_mp_path):
        try:
            _mp_doc = json.loads(open(_mp_path, encoding="utf-8").read())
            _music_plan_overrides = {
                o["shot_id"]: o
                for o in _mp_doc.get("shot_overrides", [])
                if "shot_id" in o
            }
            _music_clip_volumes  = _mp_doc.get("clip_volumes",  {})
            _music_track_volumes = _mp_doc.get("track_volumes", {})
            if not _music_plan_overrides and _mp_doc.get("shot_overrides"):
                print("[render] WARNING: MusicPlan overrides lack shot_id — "
                      "run migration script before rendering.")
            print(f"[render] MusicPlan loaded — {len(_music_plan_overrides)} shot overrides")
        except Exception as _mpe:
            print(f"[render] WARNING: MusicPlan.json load failed: {_mpe}")
    else:
        print("[render] WARNING: MusicPlan.json not found — music timing is approximate.")

    # ── Load SfxPlan.json (once) — live user selections from SFX tab ────────
    # Read directly at render time so edits made in the SFX tab are honoured
    # without requiring a gen_render_plan re-run (mirrors MusicPlan override logic).
    _sfx_plan_path = os.path.join(episode_dir, "SfxPlan.json")
    _sfx_plan_by_shot: dict = {}   # shot_id -> list[sfx_entry]
    if os.path.isfile(_sfx_plan_path):
        try:
            _sfx_plan_data = json.loads(open(_sfx_plan_path, encoding="utf-8").read())
            for _se in _sfx_plan_data.get("sfx_entries", []):
                _sid = _se.get("shot_id", "")
                if _sid:
                    _sfx_plan_by_shot.setdefault(_sid, []).append(_se)
            _n_sfx = sum(len(v) for v in _sfx_plan_by_shot.values())
            print(f"  [render] SfxPlan.json loaded — {_n_sfx} entr{'y' if _n_sfx == 1 else 'ies'} "
                  f"across {len(_sfx_plan_by_shot)} shot(s)")
        except Exception as _sfx_err:
            print(f"  [render] WARNING: Could not load SfxPlan.json: {_sfx_err}")
    # If SfxPlan.json is absent, _sfx_plan_by_shot stays {} and no SFX is mixed.

    # ── Load VO timing from VOPlan + ShotList (for render_shot subtitle track) ─
    _manifest_path = episode_dir / f"VOPlan.{locale}.json"
    _vo_timing_by_shot = load_vo_timing(_manifest_path, _shotlist_path)

    # ── Build shot dicts from ShotList + VOPlan + MusicPlan ──────────────────
    shots = build_shots_for_render(
        shotlist           = shotlist,
        media_map          = media_map,
        vo_items           = voplan.get("vo_items", []),
        music_plan_overrides = _music_plan_overrides,
        scene_tails        = voplan.get("scene_tails"),
        ref_dur_map        = _ref_dur_map if _ref_dur_map else None,
    )

    # ── Load confirmed media selections (CHANGE 5) ───────────────────────────
    def _url_to_path(url: str) -> str:
        """Return a URI that uri_to_path() can resolve.

        MediaPlan.json stores file:// URIs.  The previous implementation stripped
        the 'file://' prefix, producing a bare absolute path which uri_to_path()
        cannot handle (it requires the file:// scheme).  Now we preserve the scheme.
        Plain absolute paths are wrapped so they also work downstream.
        """
        if url.startswith("file://"):
            return url          # keep intact — uri_to_path() requires the file:// prefix
        if url and Path(url).is_absolute():
            return Path(url).as_uri()   # /abs/path → file:///abs/path
        # Relative path (should not appear in practice but handle gracefully)
        if url:
            return (_sel_path.parent / url).resolve().as_uri()
        return url

    _sel_path = episode_dir / "MediaPlan.json"
    _shot_to_segments = None   # None = no MediaPlan; {} = file present but empty

    if _sel_path.exists():
        try:
            _sel_data = json.loads(_sel_path.read_text(encoding="utf-8"))
            _shot_to_segments = {}
            for _bg_id, _bg_data in _sel_data.get("selections", {}).items():
                if not isinstance(_bg_data, dict):
                    continue
                for _shot_id, _shot_data in _bg_data.get("per_shot", {}).items():
                    _shot_to_segments[_shot_id] = _shot_data.get("segments", [])
            # Locale mismatch warning
            _confirmed_locale = _sel_data.get("confirmed_locale")
            if _confirmed_locale and _confirmed_locale != locale:
                print(
                    f"INFO: Media selections confirmed in locale '{_confirmed_locale}' "
                    f"but rendering locale '{locale}'. "
                    "Re-confirm selections in this locale if timing is incorrect."
                )
            print(f"  [media] Loaded confirmed selections: {len(_shot_to_segments)} shots")
        except Exception as _exc:
            print(f"  [media] WARNING: Failed to load MediaPlan.json: {_exc}")
            _shot_to_segments = None
    else:
        print(
            "WARNING: No confirmed media selections found (MediaPlan.json missing).\n"
            "         Video trim offsets (start_sec/end_sec from Confirm Selections) are NOT applied.\n"
            "         To apply confirmed trims, run 'Confirm Selections' in the Media tab first."
        )

    print("=" * 60)
    print("  render_video")
    print(f"  Plan    : {plan_path.name}")
    print(f"  Locale  : {locale}")
    print(f"  Shots   : {len(shots)}")
    print(f"  Profile : {profile_name}  (CRF {profile['crf']}, {profile['preset']})")
    print(f"  Output  : {output_dir}")
    print("=" * 60)

    # ── Pre-compute music continuity params ─────────────────────────────────
    # music_start_sec: where in the WAV to seek (for same-music continuation)
    # music_apply_fadeout: True when next shot has different/no music
    music_offset: dict[str, float] = {}  # {music_asset_id → accumulated_sec}
    shot_music_params: list[tuple[float, bool]] = []

    for i, shot in enumerate(shots):
        mid = shot.get("music_asset_id")
        if mid is None:
            shot_music_params.append((0.0, False))
            continue

        start_sec = music_offset.get(mid, 0.0)

        next_shot = shots[i + 1] if i + 1 < len(shots) else None
        next_mid  = next_shot.get("music_asset_id") if next_shot else None
        apply_fo  = (next_mid != mid)   # different or absent → fade out

        shot_music_params.append((start_sec, apply_fo))

        # Advance offset for next shot with same music_asset_id
        music_offset[mid] = start_sec + shot["duration_ms"] / 1000.0

    # ── Per-shot render ─────────────────────────────────────────────────────
    print()
    shot_mkv_pairs: list[tuple[dict, Path]] = []
    placeholder_count = 0
    _cumulative_shot_sec = 0.0

    for i, shot in enumerate(shots):
        # CHANGE 5: override bg_segments from confirmed selections if present
        _shot_id_cur = shot.get("shot_id", "")
        if _shot_to_segments is not None:
            if _shot_id_cur in _shot_to_segments:
                _confirmed_segs = _shot_to_segments[_shot_id_cur]
                shot = dict(shot)  # shallow copy — do not mutate shots list
                shot["background_segments"] = [
                    {
                        "uri":                   _url_to_path(seg.get("url", "")),
                        "media_type":            seg.get("media_type", "image"),
                        "animation_type":        seg.get("animation_type"),
                        "start_sec":             float(seg.get("start_sec") or 0.0),
                        "duration_override_sec": max(0.0,
                                                     float(seg["end_sec"] if seg.get("end_sec") is not None else (seg.get("hold_sec") or 0))
                                                     - float(seg["start_sec"] if seg.get("start_sec") is not None else 0.0)),
                        "hold_sec":              float(seg.get("hold_sec") or 0.0),
                    }
                    for seg in _confirmed_segs
                ]
                print(f"  [{_shot_id_cur}] Using confirmed media selections — overriding computed bg.")
            else:
                # MediaPlan.json present but shot not confirmed → black background
                shot = dict(shot)
                shot["background_segments"] = []
                print(f"  [{_shot_id_cur}] No confirmed selection — rendering black background.")
        # else: _shot_to_segments is None → use bg_segments from build_shots_for_render (no change)

        m_start, m_fadeout = shot_music_params[i]
        mkv = render_shot(
            shot=shot,
            vo_lines=_vo_timing_by_shot.get(shot.get("shot_id", ""), []),
            asset_map=asset_map,
            shot_index=i,
            shots_dir=shots_dir,
            fps=fps,
            profile=profile,
            music_start_sec=m_start,
            music_apply_fadeout=m_fadeout,
            music_fadeout_sec=args.music_fadeout_sec,
            no_music=args.no_music,
            verbose=args.verbose,
            music_plan_data={
                "plan_overrides": _music_plan_overrides,
                "clip_volumes":   _music_clip_volumes,
                "track_volumes":  _music_track_volumes,
            },
            sfx_plan_override=_sfx_plan_by_shot or None,
            _cumulative_shot_sec=_cumulative_shot_sec,
        )
        shot_mkv_pairs.append((shot, mkv))
        _cumulative_shot_sec += shot.get("duration_ms", 0) / 1000.0

        # Count placeholders in this shot
        for asset_id in [shot.get("background_asset_id")] \
                        + shot.get("character_asset_ids", []) \
                        + shot.get("sfx_asset_ids", []):
            if asset_id and asset_map.get(asset_id, {}).get("is_placeholder", True):
                placeholder_count += 1
        if shot.get("music_asset_id") and \
           asset_map.get(shot["music_asset_id"], {}).get("is_placeholder", True):
            placeholder_count += 1

    # ── Scene-boundary black frame ──────────────────────────────────────────
    black_frame = generate_black_frame(shots_dir, fps)
    frame_ms    = round(1000 / fps)

    # ── Concat list ─────────────────────────────────────────────────────────
    concat_list = shots_dir / "concat.txt"
    total_ms    = 0
    n_scenes    = 0

    with open(concat_list, "w", encoding="utf-8") as f:
        for idx, (shot, mkv) in enumerate(shot_mkv_pairs):
            f.write(f"file '{mkv}'\n")
            total_ms += shot["duration_ms"]   # VO-ceiling authoritative duration

            if idx < len(shot_mkv_pairs) - 1:
                next_shot = shot_mkv_pairs[idx + 1][0]
                if next_shot.get("scene_id") != shot.get("scene_id"):
                    f.write(f"file '{black_frame}'\n")
                    total_ms += frame_ms
                    n_scenes += 1

    print(f"\n  Scene boundaries (black frames inserted): {n_scenes}")

    # ── Concat + loudnorm → output.mp4 ─────────────────────────────────────
    final_mp4 = output_dir / "output.mp4"
    concat_to_mp4(concat_list, final_mp4, profile)

    # ── SRT + subtitle sidecar ───────────────────────────────────────────────
    srt_path  = output_dir / f"output.{locale}.srt"
    subs_path = output_dir / "output.subs.json"
    write_srt(shots, srt_path, subs_path, fps, vo_timing_by_shot=_vo_timing_by_shot)
    print(f"  SRT:  {srt_path}")
    print(f"  Subs: {subs_path}")

    # ── render_output.json ──────────────────────────────────────────────────
    render_output = {
        "schema_id":        "render_output",
        "schema_version":   "1.0.0",
        "producer":         PRODUCER,
        "plan_id":          voplan.get("plan_id", voplan.get("manifest_id", "")),
        "locale":           locale,
        "output_video":     str(final_mp4),
        "output_srt":       str(srt_path),
        "output_subs":      str(subs_path),
        "total_shots":      len(shots),
        "total_duration_ms": total_ms,
        "placeholder_count": placeholder_count,
        "profile":          profile_name,
    }
    save_json(render_output, output_dir / "render_output.json")

    # ── License manifest ─────────────────────────────────────────────────────
    if _shot_to_segments is not None:
        _license_path = write_license_manifest(
            shot_mkv_pairs=shot_mkv_pairs,
            shot_to_segments=_shot_to_segments,
            fps=fps,
            output_dir=output_dir,
            locale=locale,
        )
        if _license_path:
            print(f"  Licenses : {_license_path}")

    print(f"\n  [OK] {final_mp4}")
    print(f"  Placeholders : {placeholder_count}")
    print(f"  Duration     : {total_ms / 1000:.1f} s")

    # ── Cleanup intermediates ───────────────────────────────────────────────
    if not args.keep_intermediates:
        shutil.rmtree(shots_dir, ignore_errors=True)
        print(f"  Cleaned .shots/ scratch directory")


# ─────────────────────────────────────────────────────────────────────────────
# License manifest
# ─────────────────────────────────────────────────────────────────────────────

def write_license_manifest(
    shot_mkv_pairs: list,
    shot_to_segments: dict,
    fps: float,
    output_dir: "Path",
    locale: str,
) -> "Path | None":
    """Write licenses.json to output_dir.

    One entry per confirmed media segment, each with:
      - Timing in the final video (video_start_sec / video_end_sec)
      - Clip offsets within the source file (start_sec / end_sec)
      - Full license / attribution metadata from MediaPlan.json
    """
    entries: list[dict] = []
    total_sec = 0.0
    frame_sec = 1.0 / fps

    for idx, (shot, _mkv) in enumerate(shot_mkv_pairs):
        shot_id   = shot.get("shot_id", "")
        scene_id  = shot.get("scene_id", "")
        shot_dur  = shot.get("duration_ms", 0) / 1000.0
        shot_start = total_sec

        raw_segs = shot_to_segments.get(shot_id, [])
        if raw_segs:
            seg_cursor = 0.0           # cursor within this shot (seconds)
            for seg in raw_segs:
                media_type = seg.get("media_type", "image")

                # Duration of this segment as it appears in the video
                if media_type == "image":
                    seg_dur = float(seg.get("hold_sec") or 0.0)
                else:
                    # video clip: end_sec - start_sec
                    s_start = float(seg.get("start_sec") or 0.0)
                    s_end   = float(seg.get("end_sec") or 0.0)
                    seg_dur = max(0.0, s_end - s_start)

                video_start = round(shot_start + seg_cursor, 6)
                video_end   = round(video_start + seg_dur,   6)

                # Pull the full source/license block from the raw segment
                source: dict = seg.get("source") or {}

                entry: dict = {
                    # ── Identity ─────────────────────────────────────────
                    "shot_id":            shot_id,
                    "scene_id":           scene_id,
                    "locale":             locale,
                    # ── Position in final video ───────────────────────────
                    "video_start_sec":    video_start,
                    "video_end_sec":      video_end,
                    "video_duration_sec": round(seg_dur, 6),
                    # ── Media file info ───────────────────────────────────
                    "media_type":         media_type,
                    "url":                seg.get("url", ""),
                    "clip_start_sec":     float(seg.get("start_sec") or 0.0),
                    "clip_end_sec":       float(seg.get("end_sec") or 0.0),
                    "hold_sec":           float(seg.get("hold_sec") or 0.0)
                                          if media_type == "image" else None,
                    "animation_type":     seg.get("animation_type"),
                    # ── License / attribution ─────────────────────────────
                    "title":              source.get("title"),
                    "photographer":       source.get("photographer"),
                    "attribution_text":   source.get("attribution_text"),
                    "attribution_required": source.get("attribution_required"),
                    "license_summary":    source.get("license_summary"),
                    "license_url":        source.get("license_url"),
                    "asset_page_url":     source.get("asset_page_url"),
                    "file_url":           source.get("file_url"),
                    "source_site":        source.get("source_site"),
                    # ── Media dimensions / tags ───────────────────────────
                    "width":              source.get("width"),
                    "height":             source.get("height"),
                    "tags":               source.get("tags"),
                }
                # Strip keys whose value is None to keep the file concise
                entry = {k: v for k, v in entry.items() if v is not None}
                entries.append(entry)

                seg_cursor += seg_dur

        # Advance global clock
        total_sec += shot_dur

        # Black frame between scene boundaries
        if idx < len(shot_mkv_pairs) - 1:
            next_shot = shot_mkv_pairs[idx + 1][0]
            if next_shot.get("scene_id") != scene_id:
                total_sec += frame_sec

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
    out_path = Path(output_dir) / "licenses.json"
    out_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  [license] Wrote {len(entries)} segment(s) → {out_path}")
    return out_path


if __name__ == "__main__":
    main()
