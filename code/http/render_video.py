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

# ── Overlay / subtitle constants ───────────────────────────────────────────────
# NotoSansCJK covers both Latin and Chinese/Japanese/Korean characters.
NOTO_CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

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

# ── faster-whisper model cache (loaded lazily, reused across all VO items) ─────
_faster_whisper_model: "object | None" = None
_faster_whisper_model_size: str = ""


def _load_faster_whisper(model_size: str = "base") -> "object | None":
    """Return cached faster-whisper model, loading it on first call."""
    global _faster_whisper_model, _faster_whisper_model_size
    if _faster_whisper_model is not None and _faster_whisper_model_size == model_size:
        return _faster_whisper_model
    try:
        from faster_whisper import WhisperModel  # type: ignore
        print(f"  [whisper] Loading faster-whisper model '{model_size}'...", flush=True)
        _faster_whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
        _faster_whisper_model_size = model_size
        return _faster_whisper_model
    except Exception as exc:
        print(f"  [whisper] faster-whisper not available: {exc}", flush=True)
        return None


def _whisper_align(wav_path: "Path", text: str, locale: str,
                   model_size: str = "base") -> "tuple[list, list]":
    """Transcribe the WAV with word-level timestamps via faster-whisper.

    Returns (words, seg_ends) where:
      words    — word-level dicts [{"word": str, "start": float, "end": float}, ...]
                 in WAV-relative seconds.
      seg_ends — Whisper segment end times [float, ...] in WAV-relative seconds.
                 For a multi-sentence VO item, seg_ends[i] is the end of the i-th
                 Whisper segment and approximates the i-th sentence boundary.
                 Prefer seg_ends over _char_time for sentence boundary computation
                 because Whisper's segment breaks track actual speech pauses, while
                 _char_time only does a proportional char→word-index mapping that
                 is inaccurate when speech rate is non-uniform.

    Returns ([], []) on any failure so callers fall back to proportional timing.
    """
    if not wav_path.exists() or not text.strip():
        return [], []
    model = _load_faster_whisper(model_size)
    if model is None:
        return [], []
    try:
        lang = locale.split("-")[0]
        segments, _ = model.transcribe(
            str(wav_path),
            language=lang,
            word_timestamps=True,
            condition_on_previous_text=False,
        )
        words: list = []
        seg_ends: list = []
        for seg in segments:
            seg_ends.append(round(seg.end, 3))
            if seg.words:
                for w in seg.words:
                    wt = w.word.strip()
                    if wt:
                        words.append({"word": wt,
                                      "start": round(w.start, 3),
                                      "end":   round(w.end, 3)})
        return words, seg_ends
    except Exception as exc:
        print(f"  [whisper] transcribe failed for {wav_path.name}: {exc}", flush=True)
        return [], []


def _char_time(words: list, char_pos: int, total_chars: int,
               fallback_sec: float = 0.0) -> float:
    """Map a character position in the full text to a WAV-relative timestamp.

    Uses the word-level alignment list produced by _whisper_align.
    Proportionally maps char_pos → word index → word end time.
    """
    if not words or total_chars <= 0:
        return fallback_sec
    frac = min(char_pos / total_chars, 1.0)
    idx  = min(int(frac * len(words)), len(words) - 1)
    return words[idx]["end"]


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


_SUB_MAX_CHARS = 24   # max CJK characters per subtitle card (≈1 line on phone)
_SUB_BREAK     = "。！？…；，,!?;,"  # preferred break-after characters


def _split_subtitle_text(text: str) -> list[str]:
    """Split a long narration paragraph into short subtitle cards.

    Strategy:
    1. Try to break at sentence-ending punctuation within the first MAX+1 chars.
    2. Hard-split at _SUB_MAX_CHARS but never in the middle of an ASCII word.
    Each returned card is ≤ ~_SUB_MAX_CHARS characters.
    """
    import unicodedata as _ud

    def _visual_width(s: str) -> int:
        """CJK/fullwidth chars count as 2, ASCII as 1."""
        return sum(2 if _ud.east_asian_width(c) in ("W", "F") else 1 for c in s)

    _VW_MAX = _SUB_MAX_CHARS * 2  # visual-width budget (≈24 CJK = 48 units)

    cards: list[str] = []
    remaining = text.strip()
    while _visual_width(remaining) > _VW_MAX:
        # Walk character by character until we exceed visual width budget
        vw = 0
        limit_idx = len(remaining)
        for i, ch in enumerate(remaining):
            vw += 2 if _ud.east_asian_width(ch) in ("W", "F") else 1
            if vw > _VW_MAX:
                limit_idx = i
                break

        # Find best punctuation break within [0, limit_idx]
        best = -1
        for i, ch in enumerate(remaining[:limit_idx + 1]):
            if ch in _SUB_BREAK:
                best = i

        if best >= 0:
            cut = best + 1
        else:
            # Hard cut — but don't split in the middle of an ASCII word
            cut = limit_idx
            # Walk back until we're not mid-ASCII-word
            while cut > 0 and remaining[cut - 1].isascii() and remaining[cut - 1].isalpha() \
                    and cut < len(remaining) and remaining[cut].isascii() and remaining[cut].isalpha():
                cut -= 1
            if cut == 0:          # couldn't find a word boundary → hard cut at limit
                cut = limit_idx

        # If what's left after this cut is a tiny tail (≤ 4 visual units),
        # absorb it into the current card rather than creating a stray entry.
        tail = remaining[cut:].strip()
        tail_vw = _visual_width(tail)
        if 0 < tail_vw <= 8:
            card = remaining.strip()
            if card:
                cards.append(card)
            remaining = ""
        else:
            card = remaining[:cut].strip()
            if card:
                cards.append(card)
            remaining = tail
    if remaining:
        cards.append(remaining)
    return cards or [text.strip()]


# ── Sentence-aware subtitle helpers ──────────────────────────────────────────

# Sentence terminators: fullwidth AND halfwidth so we catch both styles.
_SENT_TERM = frozenset("。！？!?")


def _split_into_sentences(text: str) -> list[str]:
    """Split text at sentence terminators (。！？!?), keeping terminator with
    its sentence.  Any trailing fragment without a terminator is merged back
    into the preceding sentence."""
    parts = re.split(r'(?<=[。！？!?])', text.strip())
    sents = [p for p in parts if p.strip()]
    if not sents:
        return [text] if text.strip() else []
    # Merge orphan trailing fragment (no closing terminator) with previous
    if sents[-1] and sents[-1][-1] not in _SENT_TERM:
        if len(sents) >= 2:
            sents[-2] += sents[-1]
            sents.pop()
    return sents if sents else [text]


def _wav_sentence_boundaries(
    wav_path: "Path",
    n_sentences: int,
    char_counts: list,
    wav_dur: float,
    azure_break_ms: int = 500,
) -> list:
    """Return sorted list of N-1 boundary midpoints (seconds within the WAV)
    between N sentences, derived from ffmpeg silencedetect.

    Algorithm:
      1. Run silencedetect (noise=-35dB, min_dur=0.12 s) on the WAV.
      2. Collect interior silence midpoints (exclude leading <0.1 s and
         trailing that end within 0.1 s of WAV end).
      3. Take the first N-1 midpoints whose silence duration ≥ threshold
         (= azure_break_ms × 0.6, then lowered 25 % each retry until 100 ms).
      4. Post-check: if a boundary > 2 × its proportional expected position,
         replace it with the proportional estimate.  This handles the case
         where an em-dash or other prosody pause is larger than the actual
         sentence-break pause.

    Falls back to proportional char-count distribution if WAV is missing,
    ffmpeg fails, or not enough silences can be found.
    """
    if n_sentences <= 1:
        return []

    total_chars = sum(char_counts)
    cum = 0
    proportional: list = []
    for c in char_counts[:-1]:
        cum += c
        proportional.append(cum / total_chars * wav_dur)

    if not wav_path.exists():
        return proportional

    try:
        res = subprocess.run(
            ["ffmpeg", "-i", str(wav_path), "-af",
             "silencedetect=noise=-35dB:duration=0.12", "-f", "null", "-"],
            capture_output=True, text=True, timeout=15,
        )
        silences: list = []
        pending = None
        for line in res.stderr.splitlines():
            ms = re.search(r'silence_start: ([\d.]+)', line)
            if ms:
                pending = float(ms.group(1))
            me = re.search(r'silence_end: ([\d.]+)', line)
            if me and pending is not None:
                silences.append((pending, float(me.group(1))))
                pending = None

        # Interior only: exclude leading (<0.1 s start) and trailing (end near WAV end)
        interior = [
            (s, e) for s, e in silences
            if s >= 0.1 and e <= wav_dur - 0.05
        ]

        n_bounds = n_sentences - 1
        if len(interior) < n_bounds:
            return proportional

        # Try decreasing thresholds until we find at least n_bounds qualifying silences
        threshold_ms = max(120.0, azure_break_ms * 0.6)
        boundaries: "list | None" = None
        while threshold_ms >= 100:
            qualifying = [
                (s + e) / 2.0
                for s, e in interior
                if (e - s) * 1000 >= threshold_ms
            ]
            if len(qualifying) >= n_bounds:
                boundaries = sorted(qualifying)[:n_bounds]
                break
            threshold_ms *= 0.75

        if boundaries is None:
            return proportional

        # Post-check: cap any boundary that is grossly out of range.
        # Too-late guard (b > 2×exp): em-dash / drama pauses larger than sentence break.
        # Too-early guard (b < exp/2): comma/clause pauses picked instead of sentence break.
        for i, (b, exp) in enumerate(zip(boundaries, proportional)):
            if b > 2 * exp or b < exp / 2:
                boundaries[i] = exp

        return boundaries

    except Exception:
        return proportional


def write_srt_from_voplan(
    vo_items: list,
    srt_path: Path,
    subs_path: Path,
    title_offset: float = 0.0,
    vo_dir: "Path | None" = None,
) -> None:
    """Write SRT + subs.json directly from VOPlan items (episode-absolute timing).

    Each vo_item is split into SENTENCES (at 。！？!?) and each sentence is
    shown as a separate subtitle card.  Sentence timing is derived from silence
    detection on the corresponding WAV file so that each card appears in sync
    with the speech — one line on screen at a time.

    Long sentences that exceed _SUB_MAX_CHARS are visually wrapped with \\n
    inside a single SRT entry (so they still appear/disappear as one card).

    Args:
        vo_items:     List of dicts with keys text, start_sec, end_sec, item_id,
                      tts_prompt.
        srt_path:     Output .srt file path.
        subs_path:    Output .subs.json file path.
        title_offset: Seconds to add to all timestamps (default 0.0).
        vo_dir:       Directory containing {item_id}.wav files.  When supplied,
                      silence detection is used for per-sentence timing.
                      When None, proportional char-count timing is used.
    """
    lines: list[str] = []
    subs:  list[dict] = []
    seq = 1

    for vi in sorted(vo_items, key=lambda v: v.get("start_sec", 0)):
        text = vi.get("text", "").strip()
        if not text:
            continue
        abs_in  = round((vi["start_sec"] + title_offset) * 1000)
        abs_out = round((vi["end_sec"]   + title_offset) * 1000)
        if abs_out - abs_in <= 0:
            continue

        sentences = _split_into_sentences(text)
        n_sents   = len(sentences)
        wav_dur   = vi["end_sec"] - vi["start_sec"]

        # ── Whisper forced alignment (used for both sentence and card timing) ──
        # Run once per VO item; results are used by all branches below.
        # Falls back gracefully: _align_words = [] means proportional is used.
        _wav_item_start = 0.0   # alignment always returns WAV-relative times
        _align_total_chars = len(text)
        _align_words: list = []
        _align_seg_ends: list = []   # Whisper segment end times (WAV-relative)
        if vo_dir:
            _locale = vi.get("tts_prompt", {}).get("locale", "zh")
            _wav_path = vo_dir / f"{vi['item_id']}.wav"
            _align_words, _align_seg_ends = _whisper_align(_wav_path, text, _locale)

        if n_sents == 1:
            # Single sentence: one or more cards spanning the full item duration.
            cards = _split_subtitle_text(text)
            if len(cards) == 1:
                # Fits on one line — single card spanning the full item duration.
                timecode = f"{ms_to_srt_ts(abs_in)} --> {ms_to_srt_ts(abs_out)}"
                lines += [str(seq), timecode, cards[0], ""]
                subs.append({"line_id": vi.get("item_id", ""), "timecode": timecode,
                             "text": cards[0]})
                seq += 1
            else:
                # Too long for one line — split cards.
                # Use Whisper word-level timestamps when available, else proportional.
                total_c = sum(len(c) for c in cards)
                t = abs_in
                char_off = 0
                for card in cards:
                    if _align_words:
                        card_end = abs_in + round(
                            (_char_time(_align_words, char_off + len(card),
                                        _align_total_chars, wav_dur) - _wav_item_start)
                            * 1000
                        )
                    else:
                        card_end = t + round((abs_out - abs_in) * len(card) / total_c)
                    card_end = max(card_end, t + 100)
                    timecode = f"{ms_to_srt_ts(t)} --> {ms_to_srt_ts(card_end)}"
                    lines += [str(seq), timecode, card, ""]
                    subs.append({"line_id": vi.get("item_id", ""), "timecode": timecode,
                                 "text": card})
                    seq += 1
                    t = card_end
                    char_off += len(card)
        else:
            # Multiple sentences.
            # Use Whisper word-level timestamps (from forced alignment) when available.
            # Falls back to proportional char-count when Whisper is unavailable.
            char_counts = [len(s) for s in sentences]

            # Sentence boundary timestamps (WAV-relative seconds).
            # Priority:
            #   1. Whisper segment ends match n_sents exactly → use directly.
            #   2. Whisper has more segments than sentences → pick the segment
            #      boundary (internal seg_ends) closest to each proportional
            #      char-split point.  This handles CJK text where Whisper often
            #      splits one Chinese sentence into several smaller segments.
            #   3. Proportional char-count → last resort when Whisper unavailable.
            if _align_seg_ends and len(_align_seg_ends) == n_sents:
                # seg_ends[-1] is the WAV end; boundaries are seg_ends[:-1].
                bounds = [round(e - _wav_item_start, 4)
                          for e in _align_seg_ends[:-1]]
            elif _align_seg_ends and len(_align_seg_ends) > n_sents:
                # More Whisper segments than sentences: pick the seg_end closest
                # to each proportional char split.  seg_ends[-1] is WAV end —
                # only consider internal boundaries (all but the last).
                internal = _align_seg_ends[:-1]
                total_c  = sum(char_counts)
                cum = 0
                bounds = []
                for c in char_counts[:-1]:
                    cum += c
                    expected = (cum / total_c) * wav_dur
                    best_seg = min(internal, key=lambda t: abs(t - expected))
                    bounds.append(round(best_seg - _wav_item_start, 4))
            else:
                total_c = sum(char_counts)
                cum = 0
                bounds = []
                for c in char_counts[:-1]:
                    cum += c
                    bounds.append(cum / total_c * wav_dur)

            # Convert WAV-relative bounds to absolute episode ms timestamps.
            # bound_ms[i] = start of sentence i; bound_ms[i+1] = end of sentence i.
            bound_ms = [abs_in] + [
                round((vi["start_sec"] + title_offset + b) * 1000)
                for b in bounds
            ] + [abs_out]

            sent_char_off = 0
            for i, sent in enumerate(sentences):
                t_in  = max(bound_ms[i],     abs_in)
                t_out = min(bound_ms[i + 1], abs_out)
                t_out = max(t_out, t_in + 100)   # guarantee ≥100 ms visibility
                cards = _split_subtitle_text(sent)
                if len(cards) == 1:
                    timecode = f"{ms_to_srt_ts(t_in)} --> {ms_to_srt_ts(t_out)}"
                    lines += [str(seq), timecode, cards[0], ""]
                    subs.append({"line_id": vi.get("item_id", ""), "timecode": timecode,
                                 "text": cards[0]})
                    seq += 1
                else:
                    # Cards within this sentence — use proportional within
                    # the sentence window [t_in, t_out].
                    # _char_time is NOT used here: it maps char positions across
                    # the full WAV (all sentences), so for sentence N it returns
                    # a time near the sentence N start — collapsing the card.
                    # Since t_in/t_out are already precisely bounded (by Whisper
                    # segment ends or proportional), proportional within that
                    # window is the correct approach for card splits.
                    total_c = sum(len(c) for c in cards)
                    t = t_in
                    for ci, card in enumerate(cards):
                        card_end = t + round((t_out - t_in) * len(card) / total_c)
                        card_end = min(max(card_end, t + 100), t_out)
                        timecode = f"{ms_to_srt_ts(t)} --> {ms_to_srt_ts(card_end)}"
                        lines += [str(seq), timecode, card, ""]
                        subs.append({"line_id": vi.get("item_id", ""), "timecode": timecode,
                                     "text": card})
                        seq += 1
                        t = card_end
                sent_char_off += len(sent)

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


# ── Light streak effect ───────────────────────────────────────────────────────

def _build_light_streak_vf(bg_path: str, mode: str = "auto") -> str:
    """Build a 4-strip geq streak filter chain.

    mode values:
      "auto"         — measure bg image avg luma, pick automatically:
                         luma < 110  → light_streak (bright strips)
                         luma > 145  → dark_streak  (dark strips)
                         otherwise   → alternating  (bright/dark/bright/dark)
      "light_streak" — always use bright strips (+60/+65 luma delta)
      "dark_streak"  — always use dark strips   (−35/−30 luma delta)

    4 strips, one per quarter-width lane (W/8, W/8*3, W/8*5, W/8*7).
    All strips share the same sine wave (period 14 s), amplitude W/8 per lane.
    Returns a comma-separated vf filter string ready to append after scale/pad.
    """
    # ── Determine strip layout ────────────────────────────────────────────────
    if mode == "light_streak":
        resolved = "light_streak"
    elif mode == "dark_streak":
        resolved = "dark_streak"
    else:  # "auto" or anything else
        avg_luma = 128.0  # safe mid-tone fallback
        try:
            from PIL import Image as _PILImage, ImageStat as _PILStat  # type: ignore
            _img = _PILImage.open(bg_path).convert("L")
            avg_luma = _PILStat.Stat(_img).mean[0]
        except Exception:
            pass
        if avg_luma < 110:
            resolved = "light_streak"
        elif avg_luma > 145:
            resolved = "dark_streak"
        else:
            resolved = "alternating"
        print(f"  [light_streak] auto: avg_luma={avg_luma:.1f} → {resolved}")

    sin_expr = "sin(2*3.14159*T/14)"

    # Each tuple: (lane_multiplier, is_bright, delta, half_width_px)
    if resolved == "light_streak":
        strips = [(1, True, 60, 38), (3, True, 65, 32), (5, True, 60, 38), (7, True, 65, 32)]
    elif resolved == "dark_streak":
        strips = [(1, False, 35, 22), (3, False, 30, 26), (5, False, 35, 22), (7, False, 30, 26)]
    else:  # alternating
        strips = [(1, True, 60, 38), (3, False, 35, 22), (5, True, 65, 32), (7, False, 30, 26)]

    filters = []
    for lane, is_bright, delta, hw in strips:
        center  = f"W/8*{lane}+W/8*{sin_expr}"
        falloff = f"max(0,1-abs(X-({center}))/{hw})"
        lum     = (f"min(255,lum(X,Y)+{delta}*{falloff})" if is_bright
                   else f"max(0,lum(X,Y)-{delta}*{falloff})")
        filters.append(f"geq=lum='{lum}':cb='cb(X,Y)':cr='cr(X,Y)'")

    return ",".join(filters)


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
    p.add_argument(
        "--title-card", action="store_true",
        help="Burn story title overlay for first 1.0s, fade out 0.7→1.0s. "
             "Delays VO audio by 1.0s (title-only pre-roll). "
             "Title text read from episode meta.json → story_title.",
    )
    p.add_argument(
        "--subtitles", action="store_true", default=True,
        help="Burn subtitles at the bottom of the video, synced to TTS timing. "
             "When --title-card is also active, subtitles start at t=1.0s. "
             "Default: on. Pass --no-subtitles to disable.",
    )
    p.add_argument(
        "--no-subtitles", dest="subtitles", action="store_false",
        help="Disable burned-in subtitles.",
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
    _bg_resolved_path: str | None = None   # fallback bg from VOPlan.resolved_assets
    if media_plan_path.exists():
        _mp_data       = load_json(media_plan_path)
        media_segments = _mp_data.get("shot_overrides", [])
        print(f"  [media] {len(media_segments)} segment(s) from MediaPlan.json")
    else:
        # simple_narration: no MediaPlan — check resolved_assets for bg-provided
        for _ra in voplan.get("resolved_assets", []):
            if (_ra.get("asset_id") == "bg-provided"
                    and _ra.get("asset_type") == "background"):
                _uri = _ra.get("uri", "")
                _p = _uri[len("file://"):] if _uri.startswith("file://") else _uri
                if _p and Path(_p).exists():
                    _bg_resolved_path = _p
                    print(f"  [media] no MediaPlan.json — using bg-provided: {Path(_p).name}")
                    break
        if not _bg_resolved_path:
            print("  [media] MediaPlan.json not found — rendering black background")

    # ── 4. Compute total duration ─────────────────────────────────────────────
    if media_segments:
        _seg_total = sum(_mp_seg_dur(s) for s in media_segments)
        print(f"  [dur] seg_total={_seg_total:.3f}s  voplan={voplan_total_dur:.3f}s")
    # Snap to frame boundary so ffmpeg never rounds the last partial frame up.
    # e.g. 34.997s × 24fps = 839.928 → 840 frames → 35.000s without this snap.
    total_dur = math.floor(voplan_total_dur * FPS) / FPS
    # MTV: VOPlan ends at the last lyric, but the music track may have an
    # instrumental outro. Extend total_dur to cover the full music clip so
    # the rendered video is not cut short.
    if (getattr(args, "format", None) or "").lower() == "mtv":
        _mus_plan_path = episode_dir / "MusicPlan.json"
        if _mus_plan_path.exists():
            try:
                _mp_doc = load_json(_mus_plan_path)
                _music_end = max(
                    (float(o.get("end_sec", 0)) for o in _mp_doc.get("shot_overrides", [])),
                    default=0.0,
                )
                if _music_end > total_dur:
                    total_dur = math.floor(_music_end * FPS) / FPS
                    print(f"  [dur] MTV: extended total_dur to music end {_music_end:.3f}s")
            except Exception as _e:
                print(f"  [warn] MTV: could not read MusicPlan for duration: {_e}")
    # ── Subtitle env-var supplement (simple_narration pipeline) ─────────────
    if not args.subtitles and os.environ.get("SIMPLE_NARRATION_SUBTITLES"):
        args.subtitles = True

    # ── Video-effect env-var supplement (simple_narration pipeline) ──────────
    _video_effect = os.environ.get("SIMPLE_NARRATION_VIDEO_EFFECT", "").strip()

    # ── Story segments (multi-story overlay, no silent pre-roll) ─────────────
    # story_segments is written by simple_narration_setup.py and preserved by
    # post_tts_analysis.py.  Each entry carries the first_item_id so we can
    # look up the exact start_sec after TTS timing is known.
    TITLE_CARD_OFFSET = 0.0   # no silent pre-roll in multi-story format
    story_segments = voplan.get("story_segments", [])
    # Build item_id → start_sec lookup from resolved vo_items
    _item_start: dict[str, float] = {
        v["item_id"]: float(v.get("start_sec", 0.0)) for v in vo_items
    }

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
            _bp = _tmp / "clip_0000_bg.mp4"
            _used_bg = False
            if _bg_resolved_path:
                _bg_ext = Path(_bg_resolved_path).suffix.lower()
                _is_vid = _bg_ext in {".mp4", ".mov", ".mkv", ".webm", ".avi"}
                if _is_vid:
                    _bg_cmd = [
                        "ffmpeg", "-y",
                        "-stream_loop", "-1", "-i", _bg_resolved_path,
                        "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
                        "-vf", _scale_pad, "-r", str(FPS), "-pix_fmt", "yuv420p",
                        "-t", f"{total_dur:.3f}", "-c:v", "libx264", "-c:a", "aac",
                        "-shortest", str(_bp),
                    ]
                else:
                    # Static still image — plain hold, with optional light streak overlay
                    _bg_vf = _scale_pad
                    if _video_effect in ("light_streak", "dark_streak", "auto"):
                        _streak = _build_light_streak_vf(_bg_resolved_path, mode=_video_effect)
                        _bg_vf  = _scale_pad + "," + _streak
                        print(f"  [clip] streak effect mode={_video_effect!r}")
                    _bg_cmd = [
                        "ffmpeg", "-y",
                        "-loop", "1", "-framerate", str(FPS), "-i", _bg_resolved_path,
                        "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo",
                        "-vf", _bg_vf, "-r", str(FPS), "-pix_fmt", "yuv420p",
                        "-t", f"{total_dur:.3f}", "-c:v", "libx264", "-c:a", "aac",
                        "-shortest", str(_bp),
                    ]
                    _r = subprocess.run(_bg_cmd, capture_output=True, text=True)
                if _r.returncode == 0:
                    _clip_files.append(_bp)
                    _clip_durs.append(total_dur)
                    _clip_trans.append("none")
                    _kind = "video" if _is_vid else "image"
                    print(f"  [clip] bg-provided ({_kind}) {total_dur:.3f}s ok")
                    _used_bg = True
                else:
                    print(f"  [warn] bg-provided render failed — falling back to black:\n"
                          f"         {_r.stderr[-300:]}")
            if not _used_bg:
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
                _delay_ms = int((_vi["start_sec"] + _head_off + TITLE_CARD_OFFSET) * 1000)
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

        # ── 11a. Generate SRT (must exist before mux when burning subtitles) ───
        # Always generate here so the file is ready; step 12 will skip if already done.
        srt_path  = output_dir / f"output.{locale}.srt"
        subs_path = output_dir / "output.subs.json"
        write_srt_from_voplan(vo_items, srt_path, subs_path,
                              title_offset=TITLE_CARD_OFFSET,
                              vo_dir=_vo_dir)
        if args.subtitles:
            print(f"  [subtitles] SRT: {srt_path}  (offset={TITLE_CARD_OFFSET:.1f}s)")

        # ── 11b. Build video filter chain ────────────────────────────────────
        # Base: freeze last frame to fill any gap between concat and total_dur.
        _vf_chain = [f"tpad=stop_mode=clone:stop_duration={total_dur:.3f}"]

        # Per-story badge overlays: [01/05] Title, shown for 1.2s at story start.
        # drawbox gives a semi-transparent black pill behind the text.
        def _esc(t: str) -> str:
            return (t.replace("\\", "\\\\")
                     .replace("'",  "\u2019")
                     .replace(":",  "\\:")
                     .replace("[",  "\\[")
                     .replace("]",  "\\]"))

        _font_arg = (f":fontfile={NOTO_CJK_FONT}"
                     if Path(NOTO_CJK_FONT).exists() else "")
        _OVERLAY_DUR = 5.0   # seconds each story title is visible

        if story_segments:
            for _seg in story_segments:
                _idx   = _seg["story_index"]
                _total = _seg["story_count"]
                _stitle = _seg.get("title", "")
                _fid   = _seg.get("first_item_id", "")
                _t0    = _item_start.get(_fid, 0.0)
                _t1    = _t0 + _OVERLAY_DUR
                _badge = f"\\[{_idx:02d}/{_total:02d}\\] {_esc(_stitle)}"
                _enable = f"between(t,{_t0:.3f},{_t1:.3f})"
                # Full-width semi-transparent band at the top
                _vf_chain.append(
                    f"drawbox=x=0:y=0:w=iw:h=90"
                    f":color=black@0.65:t=fill:enable='{_enable}'"
                )
                # Title centered horizontally, sitting inside the top band
                _vf_chain.append(
                    f"drawtext=text='{_badge}'{_font_arg}"
                    f":fontsize=52:fontcolor=white"
                    f":borderw=2:bordercolor=black@0.6"
                    f":x=(w-tw)/2:y=20:enable='{_enable}'"
                )
            print(f"  [story-overlay] {len(story_segments)} badge(s) queued")

        # Burned-in subtitles: white text + black outline at bottom.
        if args.subtitles and srt_path.exists():
            _srt_esc = str(srt_path).replace("\\", "/").replace(":", "\\:")
            _vf_chain.append(
                f"subtitles='{_srt_esc}'"
                f":force_style='FontName=Noto Sans CJK SC"
                f",FontSize=24,Alignment=2,MarginV=40"
                f",PrimaryColour=&H00FFFFFF"
                f",OutlineColour=&H00000000"
                f",Outline=2,Shadow=1'"
            )

        _vf_str = ",".join(_vf_chain)

        # ── 11. Mux video + audio with quality profile ───────────────────────
        # tpad freezes the last frame when the concat video is shorter than
        # total_dur (e.g. because xfade dissolve transitions absorb overlap
        # time, reducing the concat duration relative to the sum of clip durs).
        # -t total_dur then trims the padded stream to the exact target length.
        final_mp4 = output_dir / "output.mp4"
        _result = subprocess.run([
            "ffmpeg", "-y",
            "-i", str(_concat_video),
            "-i", str(_final_audio),
            "-map", "0:v:0", "-map", "1:a:0",
            "-vf", _vf_str,
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
    print(f"  SRT:  {srt_path}")
    print(f"  Subs: {subs_path}")

    # ── 12a. YouTube chapters ─────────────────────────────────────────────────
    if story_segments:
        chapters_path = output_dir / "chapters.txt"
        with open(chapters_path, "w", encoding="utf-8") as _cf:
            for _seg in story_segments:
                _t0 = _item_start.get(_seg.get("first_item_id", ""), 0.0)
                _mm = int(_t0 // 60)
                _ss = int(_t0 % 60)
                _cf.write(f"{_mm}:{_ss:02d} {_seg['title']}\n")
        print(f"  Chapters: {chapters_path}")

    # ── 12b. Thumbnail — clean background frame + large centered title ───────
    # Use the background source directly (no burned-in subtitles/badges).
    # Then overlay the first story title as large, prominent center text.
    # YouTube recommended: 1280×720, <2 MB.
    thumb_path  = output_dir / "thumbnail.png"
    thumb_raw   = output_dir / "thumbnail_raw.png"
    _BOLD_FONT  = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    _thumb_font = _BOLD_FONT if Path(_BOLD_FONT).exists() else NOTO_CJK_FONT

    # Step 1: grab a raw frame from the background source (clean, no text).
    # Use bg source when available (avoids burned-in subtitles/badges).
    # Seek to min(5s, 10% of total_dur) for videos; skip -ss for still images
    # because seeking past frame 0 on a PNG/JPG makes ffmpeg exit 0 but write
    # no output file, which causes a FileNotFoundError in the Pillow step.
    _thumb_src      = _bg_resolved_path or str(final_mp4)
    _thumb_is_still = Path(_thumb_src).suffix.lower() not in {
        ".mp4", ".mov", ".mkv", ".webm", ".avi"
    }
    _thumb_ss = f"{min(5.0, total_dur * 0.1):.2f}"
    _thumb_cmd = ["ffmpeg", "-y"]
    if not _thumb_is_still:
        _thumb_cmd += ["-ss", _thumb_ss]
    _thumb_cmd += [
        "-i", _thumb_src,
        "-frames:v", "1",
        "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",
        "-q:v", "2", str(thumb_raw),
    ]
    _tr1 = subprocess.run(_thumb_cmd, capture_output=True, text=True)

    if _tr1.returncode == 0 and story_segments:
        # Step 2: overlay title using Pillow (avoids ffmpeg filter reinit bugs)
        try:
            from PIL import Image, ImageDraw, ImageFont  # type: ignore

            _thumb_title = story_segments[0]["title"]
            _W, _H = 1280, 720
            _FONT_SIZE = 72
            _LINE_PAD  = 16   # px between lines
            _BOX_PAD   = 24   # px above/below text block

            # Load bold font
            try:
                _pil_font = ImageFont.truetype(_thumb_font, _FONT_SIZE)
            except Exception:
                _pil_font = ImageFont.load_default()

            # Split title into 1-2 balanced lines for thumbnail.
            # Finds punctuation nearest to the midpoint; hard-splits if none.
            def _split(title: str) -> list[str]:
                title = title.strip()
                if len(title) <= 14:          # short enough for one line
                    return [title]
                mid = len(title) // 2
                best = -1
                for d in range(len(title)):   # search outward from midpoint
                    for i in [mid - d, mid + d]:
                        if 0 < i < len(title) and title[i] in "，,。！？ ：:":
                            best = i
                            break
                    if best != -1:
                        break
                cut = (best + 1) if best != -1 else mid
                return [title[:cut].strip(), title[cut:].strip()]

            _lines   = _split(_thumb_title)
            _img     = Image.open(str(thumb_raw)).convert("RGB")
            _img     = _img.resize((_W, _H), Image.LANCZOS)
            _draw    = ImageDraw.Draw(_img, "RGBA")

            # Measure each line
            _line_dims = []
            for _ln in _lines:
                _bb = _draw.textbbox((0, 0), _ln, font=_pil_font)
                _line_dims.append((_bb[2] - _bb[0], _bb[3] - _bb[1]))

            _max_tw   = max(d[0] for d in _line_dims)
            _line_h   = max(d[1] for d in _line_dims) + _LINE_PAD
            _block_h  = _line_h * len(_lines) + _BOX_PAD * 2
            _box_y0   = (_H - _block_h) // 2
            _box_y1   = _box_y0 + _block_h

            # Dark semi-transparent band
            _draw.rectangle([(0, _box_y0), (_W, _box_y1)], fill=(0, 0, 0, 178))

            # Draw each line centered
            for _li, (_ln, (_tw, _th)) in enumerate(zip(_lines, _line_dims)):
                _tx = (_W - _tw) // 2
                _ty = _box_y0 + _BOX_PAD + _li * _line_h
                # Shadow/border
                for _dx, _dy in [(-2,0),(2,0),(0,-2),(0,2),(-2,-2),(2,2)]:
                    _draw.text((_tx + _dx, _ty + _dy), _ln, font=_pil_font, fill=(0, 0, 0, 200))
                _draw.text((_tx, _ty), _ln, font=_pil_font, fill=(255, 255, 255, 255))

            _img.save(str(thumb_path), "PNG")
            thumb_raw.unlink(missing_ok=True)
            print(f"  Thumbnail: {thumb_path}")

        except Exception as _te:
            # Pillow failed — use raw frame as fallback
            thumb_raw.rename(thumb_path)
            print(f"  [warn] thumbnail title overlay failed: {_te}")

    elif _tr1.returncode == 0:
        # No story_segments — use raw frame as-is
        thumb_raw.rename(thumb_path)
        print(f"  Thumbnail: {thumb_path}")
    else:
        print(f"  [warn] thumbnail extraction failed: {_tr1.stderr[-200:]}")

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
