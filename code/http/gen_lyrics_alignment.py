#!/usr/bin/env python3
"""gen_lyrics_alignment.py — Lyrics alignment/transcription for MTV projects.

Two modes:
  1. Forced alignment (--lyrics provided): aligns known lyrics to music timestamps.
  2. Transcription (no --lyrics): transcribes vocals from music, producing both
     lyrics text and timestamps. User can fix lyrics in the VO tab afterwards.

Usage:
    # With lyrics (forced alignment):
    python3 gen_lyrics_alignment.py \
        --music  /path/to/music.mp3 \
        --lyrics /path/to/story.txt \
        --out    /path/to/episode/dir \
        --locale en

    # Without lyrics (transcription):
    python3 gen_lyrics_alignment.py \
        --music  /path/to/music.mp3 \
        --out    /path/to/episode/dir \
        --locale en
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Align lyrics text to music audio → VOPlan.{locale}.json",
    )
    p.add_argument("--music", required=True, metavar="PATH",
                   help="Path to the music audio file (mp3/wav/flac/ogg).")
    p.add_argument("--lyrics", required=False, default=None, metavar="PATH",
                   help="Path to lyrics text file (one line per lyric line). "
                        "If omitted, transcribes vocals from the music audio.")
    p.add_argument("--out", required=True, metavar="DIR",
                   help="Episode directory — writes VOPlan.{locale}.json here.")
    p.add_argument("--locale", default="en", metavar="LOCALE",
                   help="Locale code (default: en).")
    p.add_argument("--model", default="base", metavar="MODEL",
                   help="Whisper model size (default: base). Options: tiny, base, small, medium, large.")
    return p.parse_args()


def _split_lyrics(text: str) -> list[str]:
    """Split lyrics into non-empty lines, stripping whitespace.

    Removes section headers like [Verse], [Chorus], [Bridge – Epic Rise], etc.
    These are metadata, not sung text — including them would confuse alignment.
    """
    lines = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        # Skip section headers: [Verse], [Chorus], [Bridge – Epic Rise], etc.
        if re.match(r"^\[.*\]$", ln):
            continue
        lines.append(ln)
    return lines


def _align_with_stable_ts(music_path: str, lyrics_lines: list[str],
                          model_size: str = "base") -> list[dict]:
    """Run stable-ts forced alignment and return per-line timestamps.

    Returns list of {text, start_sec, end_sec} dicts.
    """
    try:
        import stable_whisper
    except ImportError:
        print("[ERROR] stable-ts is not installed. Install with: pip install stable-ts",
              file=sys.stderr)
        sys.exit(1)

    print(f"  [align] Loading whisper model '{model_size}'...")
    model = stable_whisper.load_model(model_size)

    # Join lyrics into single text for forced alignment
    lyrics_text = "\n".join(lyrics_lines)

    print(f"  [align] Running forced alignment on: {music_path}")
    print(f"  [align] Lyrics: {len(lyrics_lines)} lines")

    # stable-ts forced alignment: given known text, find timestamps
    result = model.align(music_path, lyrics_text, language="en")

    # Collect all words with timestamps across all segments
    all_words: list[dict] = []
    for seg in result.segments:
        for w in seg.words:
            all_words.append({
                "word":      w.word.strip(),
                "start_sec": round(w.start, 3),
                "end_sec":   round(w.end, 3),
            })

    if not all_words:
        print("  [WARN] Alignment produced no words — falling back to even split",
              file=sys.stderr)
        return _even_split_fallback(music_path, lyrics_lines)

    # Map word-level timestamps back to original lyric lines.
    # Walk through words, matching them to each lyric line in order.
    aligned: list[dict] = []
    word_idx = 0
    for line in lyrics_lines:
        line_words = line.split()
        if not line_words:
            continue
        n = len(line_words)
        # Find the span of words for this line
        start_wi = word_idx
        end_wi = min(word_idx + n, len(all_words))
        if start_wi >= len(all_words):
            # No more aligned words — skip remaining lines
            print(f"  [WARN] Ran out of aligned words at line: {line!r}",
                  file=sys.stderr)
            break
        aligned.append({
            "text":      line,
            "start_sec": all_words[start_wi]["start_sec"],
            "end_sec":   all_words[end_wi - 1]["end_sec"],
        })
        word_idx = end_wi

    if not aligned:
        print("  [WARN] No lines aligned — falling back to even split",
              file=sys.stderr)
        return _even_split_fallback(music_path, lyrics_lines)

    print(f"  [align] Aligned {len(aligned)} line(s) from {len(all_words)} words")
    return aligned


def _even_split_fallback(music_path: str, lyrics_lines: list[str]) -> list[dict]:
    """Fallback: evenly distribute lyrics across the audio duration."""
    import subprocess
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", music_path],
        capture_output=True, text=True,
    )
    duration = float(result.stdout.strip()) if result.stdout.strip() else 180.0
    n = len(lyrics_lines)
    seg_dur = duration / n if n > 0 else duration
    aligned = []
    for i, line in enumerate(lyrics_lines):
        aligned.append({
            "text":      line,
            "start_sec": round(i * seg_dur, 3),
            "end_sec":   round((i + 1) * seg_dur, 3),
        })
    return aligned


def _transcribe_with_stable_ts(music_path: str,
                               model_size: str = "base") -> list[dict]:
    """Transcribe vocals from music audio — no lyrics input needed.

    Returns list of {text, start_sec, end_sec} dicts.
    """
    try:
        import stable_whisper
    except ImportError:
        print("[ERROR] stable-ts is not installed. Install with: pip install stable-ts",
              file=sys.stderr)
        sys.exit(1)

    print(f"  [transcribe] Loading whisper model '{model_size}'...")
    model = stable_whisper.load_model(model_size)

    print(f"  [transcribe] Transcribing vocals from: {music_path}")
    result = model.transcribe(music_path)

    segments: list[dict] = []
    for seg in result.segments:
        text = seg.text.strip()
        if not text:
            continue
        segments.append({
            "text":      text,
            "start_sec": round(seg.start, 3),
            "end_sec":   round(seg.end, 3),
        })

    if not segments:
        print("  [WARN] Transcription produced no segments", file=sys.stderr)
        return []

    print(f"  [transcribe] Detected {len(segments)} segment(s)")
    return segments


def build_voplan(aligned: list[dict], locale: str, episode_dir: str) -> dict:
    """Build a VOPlan dict from aligned segments."""
    vo_items = []
    for idx, seg in enumerate(aligned):
        item_id = f"vo-sc01-{idx + 1:03d}"
        start   = seg["start_sec"]
        end     = seg["end_sec"]
        vo_items.append({
            "item_id":       item_id,
            "speaker_id":    "lyrics",
            "text":          seg["text"],
            "license_type":  "original",
            "tts_prompt":    {},            # empty — no TTS for MTV
            "start_sec":     start,
            "end_sec":       end,
            "duration_sec":  round(end - start, 3),
            "pause_after_ms": 0,
            "phoneme_overrides": {},
        })

    # Derive project metadata from episode_dir path
    ep_path = Path(episode_dir).resolve()
    ep_id   = ep_path.name
    slug    = ep_path.parent.parent.name if ep_path.parent.parent else ""

    voplan = {
        "schema_id":      "VOPlan",
        "schema_version": "1.0.0",
        "manifest_id":    f"mtv-align-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "project_id":     slug,
        "episode_id":     ep_id,
        "locale":         locale,
        "scene_heads":    {"sc01": vo_items[0]["start_sec"] if vo_items else 0.0},
        "vo_items":       vo_items,
    }
    return voplan


def main() -> None:
    args = parse_args()

    music_path = Path(args.music).resolve()
    if not music_path.exists():
        print(f"[ERROR] Music file not found: {music_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out).resolve()
    if not out_dir.exists():
        print(f"[ERROR] Output directory not found: {out_dir}", file=sys.stderr)
        sys.exit(1)

    # Determine mode: forced alignment (lyrics provided) vs transcription
    if args.lyrics:
        lyrics_path = Path(args.lyrics).resolve()
        if not lyrics_path.exists():
            print(f"[ERROR] Lyrics file not found: {lyrics_path}", file=sys.stderr)
            sys.exit(1)
        lyrics_text = lyrics_path.read_text(encoding="utf-8")
        lyrics_lines = _split_lyrics(lyrics_text)
        if not lyrics_lines:
            print("[WARN] Lyrics file is empty — switching to transcription mode",
                  file=sys.stderr)
            print(f"  [transcribe] Music: {music_path}")
            aligned = _transcribe_with_stable_ts(str(music_path), args.model)
        else:
            print(f"  [align] Music: {music_path}")
            print(f"  [align] Lyrics: {len(lyrics_lines)} lines from {lyrics_path}")
            aligned = _align_with_stable_ts(str(music_path), lyrics_lines, args.model)
    else:
        # No lyrics — transcribe vocals from the music audio
        print(f"  [transcribe] Music: {music_path}")
        print(f"  [transcribe] No lyrics provided — transcribing from audio")
        aligned = _transcribe_with_stable_ts(str(music_path), args.model)

    if not aligned:
        print("[ERROR] No segments produced from alignment or transcription.",
              file=sys.stderr)
        sys.exit(1)

    # Build and write VOPlan
    voplan = build_voplan(aligned, args.locale, str(out_dir))
    out_file = out_dir / f"VOPlan.{args.locale}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(voplan, f, indent=2, ensure_ascii=False)

    print(f"  [align] Wrote {out_file} with {len(voplan['vo_items'])} vo_items")


if __name__ == "__main__":
    main()
