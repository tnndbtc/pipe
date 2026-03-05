#!/usr/bin/env python3
# =============================================================================
# music_prepare_loops.py — Find best loopable segments in source music tracks
# =============================================================================
#
# Runs AFTER gen_music_clip.py (Stage 10[1]). For each unique source music
# track used by an episode, analyzes the track with librosa to find the best
# loopable segments (20-35s, low spectral flux, consistent RMS at boundaries).
#
# Usage:
#   python music_prepare_loops.py \
#       --manifest projects/slug/episodes/ep/AssetManifest_draft.shared.json
#
#   python music_prepare_loops.py --manifest ... --resources projects/resources/music/
#   python music_prepare_loops.py --manifest ... --top-n 3
#
# Requirements:
#   pip install librosa soundfile numpy
# =============================================================================

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SUPPORTED_EXTS = {".mp3", ".wav", ".flac", ".ogg"}

SR = 48_000  # analysis sample rate

# Repo root — two levels up from code/http/
PIPE_DIR = Path(__file__).resolve().parent.parent.parent


# ── Audio helpers ─────────────────────────────────────────────────────────────

def load_audio_48k(path: Path) -> np.ndarray:
    """Load any audio file as mono float32 at 48 kHz."""
    import librosa
    audio, _ = librosa.load(str(path), sr=SR, mono=True)
    return audio.astype(np.float32)


# ── Loop analysis ────────────────────────────────────────────────────────────

def analyze_track(audio: np.ndarray, top_n: int) -> dict:
    """
    Analyze a track for loopable segments.

    Returns dict with bpm, duration, and ranked candidate list.
    """
    import librosa

    duration_total = len(audio) / SR

    # BPM detection
    tempo, _beats = librosa.beat.beat_track(y=audio, sr=SR)
    if hasattr(tempo, '__len__'):
        bpm = float(tempo[0])
    else:
        bpm = float(tempo)

    # Onset strength envelope (spectral flux proxy)
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=audio, sr=SR, hop_length=hop_length)

    # RMS energy profile
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]

    # Sliding window scan: 20-35s segments, 2s hop
    min_dur = 20.0
    max_dur = 35.0
    scan_hop_sec = 2.0
    scan_hop_samples = int(scan_hop_sec * SR)

    # Convert frame-level arrays to per-second resolution info
    frames_per_sec = SR / hop_length

    candidates = []

    for dur_sec in np.arange(min_dur, max_dur + 1, 2.0):
        window_samples = int(dur_sec * SR)
        if window_samples > len(audio):
            continue

        for start_sample in range(0, len(audio) - window_samples, scan_hop_samples):
            end_sample = start_sample + window_samples

            # Map to frame indices in onset_env / rms arrays
            frame_start = int(start_sample / hop_length)
            frame_end = int(end_sample / hop_length)
            if frame_end > len(onset_env):
                frame_end = len(onset_env)
            if frame_start >= frame_end:
                continue

            seg_onset = onset_env[frame_start:frame_end]
            seg_rms = rms[frame_start:frame_end]

            # Score component 1: spectral stability (low variance = good)
            onset_var = np.var(seg_onset)
            # Normalize: invert so lower variance = higher score
            stability_score = 1.0 / (1.0 + onset_var)

            # Score component 2: RMS consistency at boundaries
            # Compare first and last ~0.5s of the segment
            boundary_frames = max(1, int(0.5 * frames_per_sec))
            rms_start = np.mean(seg_rms[:boundary_frames])
            rms_end = np.mean(seg_rms[-boundary_frames:])
            if max(rms_start, rms_end) > 0:
                rms_ratio = min(rms_start, rms_end) / max(rms_start, rms_end)
            else:
                rms_ratio = 1.0  # silence matches silence

            # Combined score (weighted)
            score = 0.5 * stability_score + 0.5 * rms_ratio

            candidates.append({
                "start_sec": round(start_sample / SR, 1),
                "duration_sec": round(dur_sec, 1),
                "score": round(float(score), 4),
            })

    # Sort by score descending, take top N
    candidates.sort(key=lambda c: c["score"], reverse=True)

    # Deduplicate overlapping candidates (keep higher-scoring one)
    filtered = []
    for c in candidates:
        overlap = False
        for kept in filtered:
            # Check if segments overlap by more than 50%
            c_end = c["start_sec"] + c["duration_sec"]
            k_end = kept["start_sec"] + kept["duration_sec"]
            overlap_start = max(c["start_sec"], kept["start_sec"])
            overlap_end = min(c_end, k_end)
            if overlap_end > overlap_start:
                overlap_dur = overlap_end - overlap_start
                min_seg = min(c["duration_sec"], kept["duration_sec"])
                if overlap_dur / min_seg > 0.5:
                    overlap = True
                    break
        if not overlap:
            filtered.append(c)
        if len(filtered) >= top_n:
            break

    # Assign ranks
    for i, c in enumerate(filtered):
        c["rank"] = i + 1

    return {
        "duration_total_sec": round(duration_total, 1),
        "bpm": round(bpm, 1),
        "candidates": filtered,
    }


# ── Preview extraction ───────────────────────────────────────────────────────

def extract_preview(
    audio: np.ndarray,
    start_sec: float,
    duration_sec: float,
    out_path: Path,
) -> None:
    """Write a candidate preview as 48 kHz 16-bit mono WAV."""
    import soundfile as sf

    start_sample = int(start_sec * SR)
    n_frames = int(duration_sec * SR)
    clip = audio[start_sample: start_sample + n_frames]

    if len(clip) < n_frames:
        clip = np.pad(clip, (0, n_frames - len(clip)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), clip, SR, subtype="PCM_16")


# ── Manifest / results helpers ───────────────────────────────────────────────

def load_manifest(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_used_tracks(meta_dir: Path, resources_dir: Path) -> list[Path]:
    """
    Determine which source tracks to analyze.

    If gen_music_clip_results.json exists, extract unique source files from it.
    Otherwise, scan all tracks in the resources directory.
    """
    results_path = meta_dir / "gen_music_clip_results.json"
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            results = json.load(f)
        source_names = set()
        for r in results:
            name = r.get("source_file")
            if name:
                source_names.add(name)
        if source_names:
            tracks = []
            for name in sorted(source_names):
                p = resources_dir / name
                if p.exists():
                    tracks.append(p)
                else:
                    print(f"  [WARN] Source file from results not found: {p}")
            return tracks

    # Fallback: scan all audio files in resources dir
    return sorted(
        p for p in resources_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Analyze source music tracks to find best loopable segments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "workflow:\n"
            "  1. Run gen_music_clip.py first\n"
            "  2. python music_prepare_loops.py --manifest AssetManifest_draft.shared.json\n"
        ),
    )
    p.add_argument("--manifest", required=True, metavar="PATH",
                   help="Path to the SHARED AssetManifest draft.")
    p.add_argument("--resources", default=None, metavar="DIR",
                   help="Directory containing source music files. "
                        "Auto-detected from manifest if omitted.")
    p.add_argument("--top-n", type=int, default=5, metavar="N",
                   help="Number of loop candidates per track (default: 5).")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    manifest_path = Path(args.manifest).resolve()

    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    manifest = load_manifest(manifest_path)
    project_id = manifest.get("project_id")
    episode_id = manifest.get("episode_id")
    if not project_id or not episode_id:
        raise SystemExit(
            f"[ERROR] Manifest {manifest_path} is missing 'project_id' or 'episode_id'."
        )

    out_dir = PIPE_DIR / "projects" / project_id / "episodes" / episode_id / "assets"
    music_dir = out_dir / "music"
    candidates_dir = music_dir / "candidates"
    meta_dir = out_dir / "meta"
    music_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve resources dir ────────────────────────────────────────────────
    if args.resources:
        resources_dir = Path(args.resources).resolve()
        if not resources_dir.exists():
            print(f"[ERROR] Resources directory not found: {resources_dir}",
                  file=sys.stderr)
            sys.exit(1)
    else:
        candidates_dirs = [
            PIPE_DIR / "projects" / project_id / "resources" / "music",
            PIPE_DIR / "projects" / "resources" / "music",
        ]
        resources_dir = next(
            (d for d in candidates_dirs if d.is_dir() and
             any(f.suffix.lower() in SUPPORTED_EXTS for f in d.iterdir() if f.is_file())),
            None,
        )
        if resources_dir is None:
            print(
                f"[SKIP] No music resources found — skipping loop analysis.\n"
                f"  Searched: {', '.join(str(d) for d in candidates_dirs)}\n"
                f"  To enable: add audio files to one of those dirs."
            )
            return

    # ── Discover tracks to analyze ───────────────────────────────────────────
    tracks = get_used_tracks(meta_dir, resources_dir)
    if not tracks:
        print("[SKIP] No source tracks found — nothing to analyze.")
        return

    print("=" * 60)
    print("  music_prepare_loops")
    print(f"  Manifest  : {manifest_path.name}")
    print(f"  Resources : {resources_dir}")
    print(f"  Output    : {music_dir}")
    print(f"  Tracks    : {len(tracks)}")
    print(f"  Top-N     : {args.top_n}")
    print("=" * 60)

    # ── Analyze each track ───────────────────────────────────────────────────
    output = {
        "episode_id": episode_id,
        "tracks": {},
    }

    for i, track_path in enumerate(tracks, 1):
        track_name = track_path.stem
        print(f"\n[{i}/{len(tracks)}] {track_path.name}")

        try:
            audio = load_audio_48k(track_path)
            print(f"  Loaded: {len(audio)/SR:.1f}s @ {SR}Hz")

            info = analyze_track(audio, args.top_n)
            print(f"  BPM: {info['bpm']}  |  Candidates: {len(info['candidates'])}")

            # Extract preview WAVs for each candidate
            for cand in info["candidates"]:
                preview_name = (
                    f"{track_name}__cand{cand['rank']:02d}"
                    f"__ss{cand['start_sec']}__t{cand['duration_sec']}.wav"
                )
                preview_path = candidates_dir / preview_name
                extract_preview(audio, cand["start_sec"], cand["duration_sec"], preview_path)
                cand["preview_path"] = str(preview_path.relative_to(PIPE_DIR))
                print(f"    #{cand['rank']}  {cand['start_sec']}s  "
                      f"dur={cand['duration_sec']}s  score={cand['score']}")

            output["tracks"][track_name] = {
                "source_path": str(track_path.relative_to(PIPE_DIR)),
                "duration_total_sec": info["duration_total_sec"],
                "bpm": info["bpm"],
                "candidates": info["candidates"],
            }

        except Exception as exc:
            print(f"  [ERROR] {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    # ── Write output JSON ────────────────────────────────────────────────────
    output_path = music_dir / "music_loop_candidates.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    print(f"\n{'='*60}")
    print("  SUMMARY — music_prepare_loops")
    print(f"{'='*60}")
    for name, track_info in output["tracks"].items():
        n_cands = len(track_info["candidates"])
        print(f"  {name}: {n_cands} candidates  "
              f"(BPM={track_info['bpm']}, {track_info['duration_total_sec']}s)")
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
