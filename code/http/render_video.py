#!/usr/bin/env python3
# =============================================================================
# render_video.py — Produce output.mp4 from RenderPlan + AssetManifest.media
# =============================================================================
#
# Reads RenderPlan.{locale}.json and resolves each shot into an MKV
# intermediate, then concatenates into a final output.mp4.
#
# Architecture (per /tmp/v1 spec):
#   1. Per-shot render → MKV intermediates (libx264 + pcm_s16le) in .shots/
#   2. Concat intermediates with scene-boundary black frames
#   3. Apply loudnorm (-16 LUFS, linear=true) + encode AAC → output.mp4
#   4. Write sidecar output.srt (absolute timestamps)
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
#       --plan projects/slug/ep/RenderPlan.en.json
#
#   python render_video.py \
#       --plan   projects/slug/ep/RenderPlan.en.json \
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
import json
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


def build_asset_map(render_plan: dict) -> dict[str, dict]:
    """Build {asset_id → resolved_asset} from RenderPlan.resolved_assets."""
    return {item["asset_id"]: item for item in render_plan.get("resolved_assets", [])}


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

    Returns '0' if the speaker never appears.
    """
    windows = [vl for vl in vo_lines if vl.get("speaker_id") == speaker_id]
    if not windows:
        return "0"
    parts = [
        f"between(t,{vl['timeline_in_ms'] / 1000:.3f},{vl['timeline_out_ms'] / 1000:.3f})"
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


# ── Per-shot renderer ─────────────────────────────────────────────────────────

def render_shot(
    shot:               dict,
    asset_map:          dict[str, dict],
    shot_index:         int,
    shots_dir:          Path,
    fps:                int,
    profile:            dict,
    music_start_sec:    float = 0.0,
    music_apply_fadeout: bool = False,
    music_fadeout_sec:  float = MUSIC_FADEOUT_SEC,
    verbose:            bool  = False,
) -> Path:
    """
    Render one shot to an MKV intermediate.
    Returns the path to the output MKV (created or already-existing).
    """
    shot_id = shot["shot_id"]
    dur_ms  = shot["duration_ms"]
    dur_sec = dur_ms / 1000.0
    vo_lines = shot.get("vo_lines", [])

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
    bg_id   = shot.get("background_asset_id")
    bg_info = asset_map.get(bg_id, {}) if bg_id else {}
    bg_uri  = bg_info.get("uri", "")
    bg_path = uri_to_path(bg_uri)
    bg_is_video = bg_path and bg_path.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov", ".avi")

    if bg_path and bg_path.exists() and not bg_info.get("is_placeholder", True):
        if bg_is_video:
            # Video background: loop if short, trim to dur_sec
            bg_idx = add_input(["-stream_loop", "-1"], str(bg_path))
            filter_parts.append(
                f"[{bg_idx}:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
                f"crop={W}:{H},setsar=1,trim=duration={dur_sec:.3f},"
                f"setpts=PTS-STARTPTS[bg]"
            )
        else:
            # Static image background
            bg_idx = add_input(["-loop", "1", "-t", f"{dur_sec:.3f}"], str(bg_path))
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
        line_id = vl.get("line_id", "")
        vo_info = asset_map.get(line_id, {})
        vo_uri  = vo_info.get("uri", "")
        vo_path = uri_to_path(vo_uri)
        if not vo_path or not vo_path.exists() or vo_info.get("is_placeholder", True):
            continue  # missing VO → silence for that line
        delay_ms = vl["timeline_in_ms"]
        v_idx    = add_input([], str(vo_path))
        lbl      = f"vo{vo_i}"
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

    # ── 6. Music audio stream ──────────────────────────────────────────────
    music_id   = shot.get("music_asset_id")
    music_info = asset_map.get(music_id, {}) if music_id else {}
    music_uri  = music_info.get("uri", "")
    music_path = uri_to_path(music_uri)

    if music_path and music_path.exists() and not music_info.get("is_placeholder", True):
        duck_intervals = shot.get("duck_intervals", [])
        duck_db        = shot.get("duck_db", -12.0)
        fade_sec       = shot.get("music_fade_sec", 0.15)
        duck_expr      = build_duck_expr(duck_intervals, duck_db, fade_sec)

        # Seek to music_start_sec for seamless same-music continuation.
        # Do NOT use -stream_loop -1 here: an infinite loop combined with
        # amix duration=longest causes FFmpeg to never stop.  If the music
        # WAV is shorter than the shot, amix's silence_pad will pad the rest.
        m_extra = []
        if music_start_sec > 0.001:
            m_extra = ["-ss", f"{music_start_sec:.3f}"]
        m_idx = add_input(m_extra, str(music_path))

        music_filter = (
            f"[{m_idx}:a]aformat=sample_rates=48000:channel_layouts=stereo,"
            f"volume=volume='{duck_expr}':eval=frame"
        )
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


def write_srt(shots: list[dict], srt_path: Path, fps: int) -> None:
    """
    Write output.srt with cumulative absolute timestamps.
    Accounts for 1-frame black frames inserted at scene boundaries.
    """
    lines: list[str] = []
    seq        = 1
    offset_ms  = 0
    frame_ms   = round(1000 / fps)

    for i, shot in enumerate(shots):
        for vl in shot.get("vo_lines", []):
            text = vl.get("text", "").strip()
            if not text:
                continue
            abs_in  = offset_ms + vl["timeline_in_ms"]
            abs_out = offset_ms + vl["timeline_out_ms"]
            lines += [
                str(seq),
                f"{ms_to_srt_ts(abs_in)} --> {ms_to_srt_ts(abs_out)}",
                text,
                "",
            ]
            seq += 1

        offset_ms += shot["duration_ms"]

        # Add black frame duration at scene boundaries
        if i < len(shots) - 1:
            if shots[i + 1].get("scene_id") != shot.get("scene_id"):
                offset_ms += frame_ms

    srt_path.write_text("\n".join(lines), encoding="utf-8")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render RenderPlan.{locale}.json → output.mp4 using FFmpeg.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--plan", required=True, metavar="PATH",
        help="Path to RenderPlan.{locale}.json.",
    )
    p.add_argument(
        "--locale", default=None, metavar="LOCALE",
        help="Locale string (e.g. en, zh-Hans). Auto-detected from plan_id if omitted.",
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
        "--verbose", action="store_true",
        help="Print FFmpeg commands as they run.",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    plan_path = Path(args.plan).resolve()
    if not plan_path.exists():
        print(f"[ERROR] RenderPlan not found: {plan_path}", file=sys.stderr)
        sys.exit(1)

    rp = load_json(plan_path)

    # ── Locale detection ────────────────────────────────────────────────────
    # Prefer explicit --locale arg; otherwise derive from the filename:
    # "RenderPlan.zh-Hans.json" → "zh-Hans"  (handles compound locales with hyphens)
    # Fall back to the plan_id field only if the filename doesn't match.
    locale = args.locale
    if not locale:
        stem = plan_path.stem   # e.g. "RenderPlan.zh-Hans"
        if stem.startswith("RenderPlan."):
            locale = stem[len("RenderPlan."):]
        else:
            plan_id = rp.get("plan_id", "")
            locale  = plan_id.rsplit("-", 1)[-1] if plan_id else "en"

    # ── Paths ────────────────────────────────────────────────────────────────
    episode_dir = plan_path.parent
    output_dir  = Path(args.out).resolve() if args.out else \
                  (episode_dir / "renders" / locale)
    output_dir.mkdir(parents=True, exist_ok=True)

    shots_dir = output_dir / ".shots"
    shots_dir.mkdir(parents=True, exist_ok=True)

    fps          = rp.get("fps", FPS)
    profile_name = args.profile or rp.get("profile", DEFAULT_PROFILE)
    profile      = PROFILES.get(profile_name, PROFILES[DEFAULT_PROFILE])

    shots     = rp["shots"]
    asset_map = build_asset_map(rp)

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

    for i, shot in enumerate(shots):
        m_start, m_fadeout = shot_music_params[i]
        mkv = render_shot(
            shot=shot,
            asset_map=asset_map,
            shot_index=i,
            shots_dir=shots_dir,
            fps=fps,
            profile=profile,
            music_start_sec=m_start,
            music_apply_fadeout=m_fadeout,
            music_fadeout_sec=args.music_fadeout_sec,
            verbose=args.verbose,
        )
        shot_mkv_pairs.append((shot, mkv))

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
            total_ms += shot["duration_ms"]

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

    # ── SRT ─────────────────────────────────────────────────────────────────
    srt_path = output_dir / "output.srt"
    write_srt(shots, srt_path, fps)
    print(f"  SRT: {srt_path}")

    # ── render_output.json ──────────────────────────────────────────────────
    render_output = {
        "schema_id":        "render_output",
        "schema_version":   "1.0.0",
        "producer":         PRODUCER,
        "plan_id":          rp.get("plan_id", ""),
        "locale":           locale,
        "output_video":     str(final_mp4),
        "output_srt":       str(srt_path),
        "total_shots":      len(shots),
        "total_duration_ms": total_ms,
        "placeholder_count": placeholder_count,
        "profile":          profile_name,
    }
    save_json(render_output, output_dir / "render_output.json")

    print(f"\n  [OK] {final_mp4}")
    print(f"  Placeholders : {placeholder_count}")
    print(f"  Duration     : {total_ms / 1000:.1f} s")

    # ── Cleanup intermediates ───────────────────────────────────────────────
    if not args.keep_intermediates:
        shutil.rmtree(shots_dir)
        print(f"  Cleaned .shots/ scratch directory")


if __name__ == "__main__":
    main()
