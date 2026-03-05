#!/usr/bin/env python3
# =============================================================================
# music_review_pack.py — Generate music review artifacts for human inspection
# =============================================================================
#
# Runs AFTER post_tts_analysis.py (Stage 10[4]) when both music WAVs and VO
# timing exist.  Produces three review artifacts:
#
#   1. timeline.txt   — human-readable shot timeline with music, VO, ducking
#   2. timeline.json  — machine-readable version for Web UI
#   3. preview_audio.wav — VO + music mix (no SFX, no video) with ducking
#
# Usage:
#   python music_review_pack.py \
#       --manifest projects/slug/ep/AssetManifest_merged.en.json
#
#   python music_review_pack.py \
#       --manifest projects/slug/ep/AssetManifest_merged.en.json \
#       --output   projects/slug/ep/assets/music/MusicReviewPack/
#
# Requirements:
#   pip install soundfile numpy
# =============================================================================

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PIPE_DIR = Path(__file__).resolve().parent.parent.parent

SAMPLE_RATE = 48000
CHANNELS = 2
BASE_MUSIC_DB = -6.0
DEFAULT_DUCK_DB = -12.0
DEFAULT_FADE_SEC = 0.15
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


def compute_duck_intervals(vo_items_for_shot, fade_sec):
    """Compute music duck intervals from VO line positions within a shot."""
    raw = []
    for item in vo_items_for_shot:
        start = item.get("start_sec")
        end = item.get("end_sec")
        if start is None or end is None:
            continue
        t0 = max(0.0, start - fade_sec)
        t1 = end + fade_sec
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


def build_shot_envelope(n_samples, duck_db, fade_sec, fade_in=True, fade_out=True):
    """
    Build a shot-boundary amplitude envelope for music.

    Music holds at a constant level throughout the shot.
    Fades in at the start and/or out at the end only at shot boundaries
    (skipped when the same track continues from the previous/into the next shot).

    duck_db = 0   → hold at base_amp (full music volume, no attenuation)
    duck_db = -8  → hold 8 dB below base (quieter under dialogue)
    fade_in/fade_out: False when same track continues across shot boundary.
    """
    base_amp = 10 ** (BASE_MUSIC_DB / 20.0)
    hold_amp = base_amp * (10 ** (duck_db / 20.0)) if duck_db != 0 else base_amp

    envelope = np.full(n_samples, hold_amp, dtype=np.float64)
    fade_samples = min(int(fade_sec * SAMPLE_RATE), n_samples // 2)

    if fade_in and fade_samples > 0:
        envelope[:fade_samples] = np.linspace(0.0, hold_amp, fade_samples)

    if fade_out and fade_samples > 0:
        envelope[n_samples - fade_samples:] = np.linspace(hold_amp, 0.0, fade_samples)

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

        # If duck_intervals not in manifest, compute from VO
        vo_items_for_shot = vo_shot_map.get(shot_id, [])
        if not duck_intervals and vo_items_for_shot:
            duck_intervals = compute_duck_intervals(vo_items_for_shot, fade_sec)

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
    """
    project_id = manifest.get("project_id", "")
    episode_id = manifest.get("episode_id", "")
    locale = manifest.get("locale", "en")

    n_samples = int(total_dur * SAMPLE_RATE) + SAMPLE_RATE  # +1s safety
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

    for idx, entry in enumerate(timeline_shots):
        offset_samples = int(entry["offset_sec"] * SAMPLE_RATE)

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
            else:
                start = vo_cursor
                vo_cursor += vo_dur + DEFAULT_PAUSE_SEC

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
        # Music fills from its start point to the end of the shot
        shot_samples = max(0, int((entry["duration_sec"] - start_sec) * SAMPLE_RATE))

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

        duck_db  = float(entry.get("duck_db", 0))
        fade_sec = float(entry.get("fade_sec", DEFAULT_FADE_SEC))
        envelope = build_shot_envelope(
            len(music_data), duck_db, fade_sec,
            fade_in=do_fade_in, fade_out=do_fade_out,
        )

        # Apply envelope (stereo)
        music_data = music_data * envelope[:, np.newaxis]

        # Place in buffer at shot start + start_sec offset
        s = offset_samples + music_offset_samples
        e = min(s + len(music_data), n_samples)
        length = e - s
        if length > 0:
            buf[s:e] += music_data[:length]

    # Trim trailing silence
    buf = buf[:int(total_dur * SAMPLE_RATE)]

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


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate music review artifacts (timeline + preview audio) "
                    "from a merged AssetManifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", required=True, metavar="PATH",
                   help="Path to AssetManifest_merged.{locale}.json")
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
        shots, manifest, vo_shot_map, music_index, loop_info
    )

    # Apply user overrides on top of timeline (duck_db, fade_sec, music_asset_id, etc.)
    if args.overrides:
        overrides_path = Path(args.overrides).resolve()
        if overrides_path.exists():
            with open(overrides_path, encoding="utf-8") as f:
                user_overrides = json.load(f)
            ovr_by_item = {o["item_id"]: o for o in user_overrides
                           if "item_id" in o}
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
                        entry["start_sec"] = float(ovr["start_sec"])
                    if "duration_sec" in ovr:
                        entry["duration_sec"] = float(ovr["duration_sec"])
                    if "music_asset_id" in ovr:
                        # Store override separately — keep original music_item_id
                        # for UI (timeline.json). Renderer uses _override for audio.
                        entry["music_item_id_override"] = ovr["music_asset_id"]
                    # Recompute duck intervals with new fade_sec
                    vo_items_for_shot = vo_shot_map.get(entry["shot_id"], [])
                    if vo_items_for_shot:
                        entry["duck_intervals"] = compute_duck_intervals(
                            vo_items_for_shot, float(entry.get("fade_sec", DEFAULT_FADE_SEC)))
                    applied += 1
            print(f"  [INFO] Applied {applied} user override(s) from {overrides_path.name}")
        else:
            print(f"  [WARN] Overrides file not found: {overrides_path}")

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
