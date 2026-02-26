#!/usr/bin/env python3
# =============================================================================
# gen_music_clip.py — Extract best-matching music segments from source tracks
# =============================================================================
#
# For each music_item in an AssetManifest JSON:
#   1. Reads TXXX:MOOD tags from all files in projects/resources/music/
#   2. Picks the best-matching source file using CLAP text-text similarity
#      between the file's MOOD tag and music_item.music_mood
#   3. Scans that file with a CLAP sliding window to find the best segment
#   4. Extracts the clip and writes it to the episode's assets directory
#
# Run tag_music.py FIRST to populate the MOOD tags in your source files.
# Run this script BEFORE gen_music.py — gen_music.py will skip any
# music_*.wav that already exists in the output directory.
#
# Usage:
#   python gen_music_clip.py \
#       --manifest projects/slug/episodes/ep/AssetManifest_draft.json
#
#   python gen_music_clip.py --manifest ...  --resources projects/resources/music/
#   python gen_music_clip.py --manifest ...  --output-dir /path/to/assets/en/
#   python gen_music_clip.py --manifest ...  --item-id music-s01-sh02
#   python gen_music_clip.py --manifest ...  --hop 1.0   # finer scan
#   python gen_music_clip.py --manifest ...  --force     # redo existing clips
#
# Requirements (CPU-only — no GPU needed):
#   pip install laion-clap mutagen librosa soundfile numpy scipy
#
# CLAP checkpoint (~400 MB) is downloaded automatically to
#   ~/.cache/laion_clap/  on first run.
# =============================================================================

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SUPPORTED_EXTS = {".mp3", ".wav", ".flac", ".ogg"}

CKPT_URL      = ("https://huggingface.co/lukewys/laion_clap/resolve/main/"
                 "music_audioset_epoch_15_esc_90.14.pt")
CKPT_FILENAME = "music_audioset_epoch_15_esc_90.14.pt"

CLAP_SR   = 48_000
CHUNK_SEC = 10     # CLAP maximum receptive field (seconds)

# Repo root — two levels up from code/ai/
PIPE_DIR = Path(__file__).resolve().parent.parent.parent


# ── Model loading ─────────────────────────────────────────────────────────────

def load_clap(ckpt_dir: Path):
    """Load CLAP model, auto-downloading checkpoint if needed."""
    import laion_clap

    ckpt_path = ckpt_dir / CKPT_FILENAME
    if not ckpt_path.exists():
        print(f"[CLAP] Checkpoint not found — downloading to {ckpt_path} …")
        import urllib.request
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CKPT_URL, ckpt_path)
        print("[CLAP] Download complete.")

    print(f"[CLAP] Loading {CKPT_FILENAME} …")
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
    model.load_ckpt(str(ckpt_path))
    model.eval()
    print("[CLAP] Model ready.\n")
    return model


# ── Audio helpers ─────────────────────────────────────────────────────────────

def load_audio_48k(path: Path) -> np.ndarray:
    """Load any audio file as mono float32 at 48 kHz."""
    import librosa
    audio, _ = librosa.load(str(path), sr=CLAP_SR, mono=True)
    return audio.astype(np.float32)


# ── Tag reading ───────────────────────────────────────────────────────────────

def read_mood_tag(path: Path) -> str | None:
    """Return the TXXX:MOOD / MOOD tag value, or None if absent."""
    try:
        suf = path.suffix.lower()
        if suf == ".mp3":
            from mutagen.id3 import ID3, ID3NoHeaderError
            try:
                txxx = ID3(str(path)).get("TXXX:MOOD")
                return str(txxx.text[0]) if txxx else None
            except ID3NoHeaderError:
                return None
        elif suf == ".flac":
            from mutagen.flac import FLAC
            vals = FLAC(str(path)).get("MOOD")
            return vals[0] if vals else None
        elif suf == ".ogg":
            from mutagen.oggvorbis import OggVorbis
            vals = OggVorbis(str(path)).get("MOOD")
            return vals[0] if vals else None
    except Exception:
        pass
    return None


# ── Resource scanning ──────────────────────────────────────────────────────────

def scan_resources(resources_dir: Path) -> list[dict]:
    """
    Scan resources_dir for audio files.
    Returns list of {path, mood_tag} dicts.
    Files without a MOOD tag are included (tag=None) and rank last.
    """
    entries = []
    for p in sorted(resources_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            tag = read_mood_tag(p)
            entries.append({"path": p, "mood_tag": tag})
            if tag:
                preview = tag[:70] + "…" if len(tag) > 70 else tag
                print(f"  {p.name:<40}  tag: {preview}")
            else:
                print(f"  {p.name:<40}  [no MOOD tag — run tag_music.py first]")
    return entries


# ── File matching ─────────────────────────────────────────────────────────────

def match_best_file(
    model,
    music_mood: str,
    candidates: list[dict],
) -> tuple[dict, float]:
    """
    Pick the source file whose MOOD tag is most similar to music_mood.
    Uses CLAP text-text cosine similarity (both strings encoded in one batch).

    Returns (best_candidate_dict, score).
    Falls back to the first untagged file if nothing is tagged, with a warning.
    """
    tagged   = [c for c in candidates if c["mood_tag"]]
    untagged = [c for c in candidates if not c["mood_tag"]]

    if not tagged:
        print("  [WARN] No MOOD-tagged files — using first file. "
              "Run tag_music.py for better matching.")
        return untagged[0], 0.0

    # Embed music_mood + all mood tags in a single batch (efficient)
    texts    = [music_mood] + [c["mood_tag"] for c in tagged]
    embs     = model.get_text_embedding(texts, use_tensor=False)  # (N+1, 512)

    query    = embs[0]       # (512,)
    tag_embs = embs[1:]      # (N, 512)
    scores   = tag_embs @ query                                   # (N,)

    best_idx = int(np.argmax(scores))
    return tagged[best_idx], float(scores[best_idx])


# ── Segment extraction ────────────────────────────────────────────────────────

def find_best_segment(
    model,
    audio: np.ndarray,
    music_mood: str,
    duration_sec: float,
    hop_sec: float,
) -> tuple[int, float]:
    """
    Slide a window of duration_sec over audio at hop_sec intervals.
    Score each window against music_mood with CLAP.
    Returns (best_start_sample, best_score).

    Window is capped at CLAP's 10-second maximum; clips longer than 10s
    are scored on their first 10s (best proxy for the whole clip's match).
    """
    window_frames = min(int(duration_sec * CLAP_SR), CHUNK_SEC * CLAP_SR)
    hop_frames    = int(hop_sec * CLAP_SR)
    total_frames  = len(audio)

    if total_frames <= window_frames:
        # Track is shorter than the window — use the whole thing
        return 0, 0.0

    # Pre-compute query text embedding once
    query_emb = model.get_text_embedding([music_mood], use_tensor=False)[0]  # (512,)

    starts    = list(range(0, total_frames - window_frames, hop_frames))
    n_windows = len(starts)
    print(f"  Scanning {n_windows} windows "
          f"({duration_sec:.1f}s window, {hop_sec:.1f}s hop) …")

    best_score = -1.0
    best_start = 0

    for i, start in enumerate(starts):
        chunk = audio[start: start + window_frames]
        # Pad to full CLAP chunk length if the window falls short
        if len(chunk) < CHUNK_SEC * CLAP_SR:
            chunk = np.pad(chunk, (0, CHUNK_SEC * CLAP_SR - len(chunk)))

        audio_emb = model.get_audio_embedding_from_data([chunk], use_tensor=False)[0]
        score     = float(np.dot(query_emb, audio_emb))

        if score > best_score:
            best_score = score
            best_start = start

        if (i + 1) % 30 == 0:
            pct = (i + 1) / n_windows * 100
            print(f"    … {i+1}/{n_windows} ({pct:.0f}%)  "
                  f"best so far: {best_score:+.3f} @ {best_start/CLAP_SR:.1f}s")

    return best_start, best_score


def extract_clip(
    audio: np.ndarray,
    start_sample: int,
    duration_sec: float,
    out_path: Path,
) -> int:
    """Write the selected segment to a 48 kHz mono WAV. Returns file size."""
    import soundfile as sf

    n_frames = int(duration_sec * CLAP_SR)
    clip     = audio[start_sample: start_sample + n_frames]

    # Safety pad in case we ended up near the tail
    if len(clip) < n_frames:
        clip = np.pad(clip, (0, n_frames - len(clip)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), clip, CLAP_SR, subtype="PCM_16")
    return out_path.stat().st_size


# ── License sidecar ───────────────────────────────────────────────────────────

def write_license_sidecar(path: Path) -> None:
    """
    Write a CC0 license sidecar JSON alongside every generated audio file.

    The resolver refuses to serve any audio asset that is missing its sidecar.
    Format matches the pipeline contract:
      { "spdx_id": "CC0", "attribution_required": false, "text": "" }
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"spdx_id": "CC0", "attribution_required": False, "text": ""},
            f,
            indent=2,
        )
        f.write("\n")


# ── Manifest helpers ──────────────────────────────────────────────────────────

def load_manifest(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def locale_from_path(path: Path) -> str:
    """AssetManifest_draft.zh-Hans.json → zh-Hans; AssetManifest_draft.json → en"""
    parts = path.stem.split(".")
    return parts[-1] if len(parts) > 1 else "en"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract best-matching music clips from tagged source tracks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "workflow:\n"
            "  1. Drop MP3/WAV files into projects/resources/music/\n"
            "  2. python tag_music.py --dir projects/resources/music/\n"
            "  3. python gen_music_clip.py --manifest <path>\n"
            "  4. python gen_music.py --manifest <path>    "
            "# generates only items not yet covered\n"
        ),
    )
    p.add_argument("--manifest", required=True, metavar="PATH",
                   help="Path to AssetManifest_draft.json (or locale variant).")
    p.add_argument("--resources", default=None, metavar="DIR",
                   help="Directory containing tagged source music files. "
                        "Default: <repo_root>/projects/resources/music/")
    p.add_argument("--output-dir", default=None, metavar="DIR",
                   help="Base output directory. Audio is written to <DIR>/audio/ "
                        "and license sidecars to <DIR>/licenses/. "
                        "Default: <episode_dir>/assets/<locale>/")
    p.add_argument("--item-id", default=None, metavar="ID",
                   help="Process only this music item_id (e.g. music-s01-sh02).")
    p.add_argument("--hop", type=float, default=2.0, metavar="SEC",
                   help="Sliding-window hop in seconds (default: 2.0). "
                        "Smaller = more precise but slower.")
    p.add_argument("--force", action="store_true",
                   help="Re-process items whose output clip already exists.")
    p.add_argument("--ckpt-dir", default=None, metavar="DIR",
                   help="Directory for CLAP checkpoint. "
                        "Default: ~/.cache/laion_clap/")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    manifest_path = Path(args.manifest).resolve()
    resources_dir = (Path(args.resources).resolve() if args.resources
                     else PIPE_DIR / "projects" / "resources" / "music")
    ckpt_dir      = (Path(args.ckpt_dir) if args.ckpt_dir
                     else Path.home() / ".cache" / "laion_clap")

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)
    if not resources_dir.exists():
        print(f"[ERROR] Resources directory not found: {resources_dir}",
              file=sys.stderr)
        sys.exit(1)

    # ── Load manifest ─────────────────────────────────────────────────────────
    manifest    = load_manifest(manifest_path)
    locale      = locale_from_path(manifest_path)

    # Derive canonical output path from manifest's project_id + episode_id so
    # assets land in the right place even when --manifest points outside the
    # repo tree (e.g. /tmp/).  Falls back to manifest file location for old
    # manifests that pre-date the episode_id field.
    project_id  = manifest.get("project_id")
    episode_id  = manifest.get("episode_id")
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    elif project_id and episode_id:
        out_dir = (PIPE_DIR / "projects" / project_id
                   / "episodes" / episode_id / "assets" / locale)
    else:
        # Fallback: put assets next to the manifest (old behaviour)
        out_dir = manifest_path.parent / "assets" / locale
    audio_dir    = out_dir / "audio"
    licenses_dir = out_dir / "licenses"
    audio_dir.mkdir(parents=True, exist_ok=True)
    licenses_dir.mkdir(parents=True, exist_ok=True)

    music_items = manifest.get("music_items", [])
    if not music_items:
        print("[INFO] No music_items in manifest — nothing to do.")
        return

    if args.item_id:
        music_items = [m for m in music_items if m["item_id"] == args.item_id]
        if not music_items:
            print(f"[ERROR] item_id '{args.item_id}' not found in manifest.",
                  file=sys.stderr)
            sys.exit(1)

    # ── Print plan ────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  gen_music_clip")
    print(f"  Manifest  : {manifest_path.name}")
    print(f"  Resources : {resources_dir}")
    print(f"  Audio out : {audio_dir}")
    print(f"  Licenses  : {licenses_dir}")
    print(f"  Items     : {len(music_items)}")
    print(f"  Hop       : {args.hop}s")
    print("=" * 60)

    # ── Scan resource files ───────────────────────────────────────────────────
    print("\n── Resource files ──────────────────────────────────────")
    candidates = scan_resources(resources_dir)
    if not candidates:
        print(f"[ERROR] No audio files found in {resources_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Load CLAP once ────────────────────────────────────────────────────────
    model = load_clap(ckpt_dir)

    # Audio cache: avoid reloading the same source track multiple times
    # (multiple shots often map to the same file)
    _audio_cache: dict[str, np.ndarray] = {}

    results = []
    total   = len(music_items)

    for idx, item in enumerate(music_items, 1):
        item_id      = item["item_id"]
        shot_id      = item["shot_id"]
        music_mood   = item["music_mood"]
        duration     = float(item["duration_sec"])
        out_path     = audio_dir    / f"{item_id}.wav"
        license_path = licenses_dir / f"{item_id}.license.json"

        print(f"\n[{idx}/{total}] {item_id}  ({duration}s)")
        print(f"  Mood : \"{music_mood}\"")

        # Skip if already done (gen_music.py also respects this)
        if out_path.exists() and not args.force:
            print(f"  [SKIP] {out_path.name} already exists")
            # Ensure sidecar exists even for skipped clips (idempotent)
            if not license_path.exists():
                write_license_sidecar(license_path)
                print(f"  [SIDECAR] {license_path.name}  ← written (was missing)")
            results.append({
                "item_id":      item_id,
                "shot_id":      shot_id,
                "output":       str(out_path),
                "license":      str(license_path),
                "size_bytes":   out_path.stat().st_size,
                "status":       "skipped",
            })
            continue

        try:
            # ── Step 1: pick best matching source file ────────────────────────
            best_file, match_score = match_best_file(model, music_mood, candidates)
            print(f"  File : {best_file['path'].name}  "
                  f"(match score {match_score:+.3f})")

            # ── Step 2: load audio (cached) ───────────────────────────────────
            cache_key = str(best_file["path"])
            if cache_key not in _audio_cache:
                print(f"  Loading {best_file['path'].name} …")
                _audio_cache[cache_key] = load_audio_48k(best_file["path"])
            audio        = _audio_cache[cache_key]
            track_dur    = len(audio) / CLAP_SR
            print(f"  Track duration: {track_dur:.1f}s")

            # ── Step 3: find best segment ─────────────────────────────────────
            start_sample, seg_score = find_best_segment(
                model, audio, music_mood, duration, args.hop
            )
            start_sec = start_sample / CLAP_SR
            print(f"  Segment: {start_sec:.1f}s – {start_sec + duration:.1f}s  "
                  f"(score {seg_score:+.3f})")

            # ── Step 4: extract, save, and write license sidecar ─────────────
            size = extract_clip(audio, start_sample, duration, out_path)
            write_license_sidecar(license_path)
            print(f"  [OK]      {out_path.name}  ({size:,} bytes)")
            print(f"  [LICENSE] {license_path.name}")

            results.append({
                "item_id":      item_id,
                "shot_id":      shot_id,
                "source_file":  best_file["path"].name,
                "source_tag":   best_file["mood_tag"],
                "match_score":  round(match_score, 4),
                "start_sec":    round(start_sec, 2),
                "duration_sec": duration,
                "seg_score":    round(seg_score, 4),
                "output":       str(out_path),
                "license":      str(license_path),
                "size_bytes":   size,
                "status":       "success",
            })

        except Exception as exc:
            print(f"  [ERROR] {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            results.append({
                "item_id":    item_id,
                "shot_id":    shot_id,
                "output":     str(out_path),
                "license":    str(license_path),
                "size_bytes": 0,
                "status":     "failed",
                "error":      str(exc),
            })

    # ── Write results manifest ────────────────────────────────────────────────
    results_path = out_dir / "gen_music_clip_results.json"
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY — gen_music_clip")
    print(f"{'='*60}")
    for r in results:
        stem = Path(r["output"]).name
        if r["status"] == "success":
            print(f"  [OK]      audio/{stem}"
                  f"  ← {r['source_file']} @ {r['start_sec']}s")
            print(f"            licenses/{Path(r['license']).name}")
        elif r["status"] == "skipped":
            print(f"  [SKIPPED] audio/{stem}")
        else:
            print(f"  [FAILED]  {r.get('item_id')}  — {r.get('error')}")

    ok          = sum(1 for r in results if r["status"] in ("success", "skipped"))
    total_bytes = sum(r["size_bytes"] for r in results)
    print(f"\n{ok}/{total} completed | {total_bytes:,} bytes | "
          f"results: {results_path}")


if __name__ == "__main__":
    main()
