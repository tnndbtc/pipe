# =============================================================================
# gen_lipsync.py
# Drive lip-sync on animated character videos using VO audio files.
# Input: character MP4 (from gen_character_animation.py) + VO WAV
#        (from gen_tts.py).
# Output: new MP4 with mouth movement matching the audio.
#
# 8 GB VRAM FALLBACK: Wav2Lip (~500 MB, runs on any GPU or CPU)
# Original target model (LatentSync 1.5) requires ~16 GB VRAM and OOMs
# on an RTX 4060 8 GB during the denoising forward pass. See
# placeholder_for_lipsync.py for the full GPU requirements.
# =============================================================================
#
# requirements.txt (pip install before running):
#   torch>=2.4.1
#   opencv-python>=4.9.0
#   numpy>=1.24.0,<2.0.0
#   huggingface_hub>=0.21.0
#   imageio[ffmpeg]>=2.34.0
#   soundfile>=0.12.0
#   tqdm>=4.0.0
#
# ---------------------------------------------------------------------------
# Hardware Target: NVIDIA RTX 4060 8 GB VRAM (or CPU)
# ---------------------------------------------------------------------------
# Memory-saving techniques:
#   Wav2Lip is a small GAN (~500 MB total) that processes face crops
#   frame-by-frame.  No diffusion loop -- VRAM usage is trivially low.
#
#   Pipeline:
#     1. Face detection on each video frame using a lightweight S3FD or
#        similar face detector bundled with the Wav2Lip repo.
#     2. Audio mel-spectrogram extraction via librosa.
#     3. Per-frame: feed (face crop + mel window) through Wav2Lip GAN to
#        produce a lip-synced face crop.
#     4. Paste the generated crop back into the full frame.
#     5. Reassemble frames + merge original audio track.
#
# SETUP REQUIREMENT -- Wav2Lip repo:
#   git clone https://github.com/Rudrabha/Wav2Lip
#   cd Wav2Lip
#   # Download checkpoint (choose one):
#   #   Wav2Lip:     https://iiitaphyd-my.sharepoint.com/... (see repo README)
#   #   Wav2Lip-GAN: higher quality, same link
#   # Place checkpoint at: Wav2Lip/checkpoints/wav2lip_gan.pth
#   pip install -r Wav2Lip/requirements.txt
#
#   Then run:
#   python gen_lipsync.py --wav2lip_dir Wav2Lip
#
# FALLBACK: Without the repo, the script merges VO audio into the video
#   track (no face manipulation) so the pipeline I/O is still validated.
#
# UPGRADE PATH: For LatentSync 1.5 quality, see placeholder_for_lipsync.py
#   (requires RTX 4080 16 GB or better).
# ---------------------------------------------------------------------------

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# DEFAULTS -- fully populated; script runs with no CLI flags.
# ---------------------------------------------------------------------------
PROJECTS_ROOT = Path(__file__).resolve().parent.parent.parent / "projects"
OUTPUT_DIR    = PROJECTS_ROOT / "the-pharaoh-who-defied-death" / "episodes" / "s01e01" / "assets"
SCRIPT_NAME = "gen_lipsync"

LIPSYNC_JOBS = [
    {
        "vo_item_id": "vo-s01-001",
        "character_video": "char-amunhotep-v1-anim.mp4",
        "vo_audio": "vo-s01-001.wav",
        "output": "char-amunhotep-lipsync-s01-001.mp4",
        "text": "This was not carved by our people. It predates the Old Kingdom. Perhaps everything.",
    },
    {
        "vo_item_id": "vo-s01-006",
        "character_video": "char-ramesses_ka-v1-anim.mp4",
        "vo_audio": "vo-s01-006.wav",
        "output": "char-ramesses_ka-lipsync-s01-006.mp4",
        "text": "Then why are you afraid?",
    },
    {
        "vo_item_id": "vo-s02-003",
        "character_video": "char-neferet-v1-anim.mp4",
        "vo_audio": "vo-s02-003.wav",
        "output": "char-neferet-lipsync-s02-003.mp4",
        "text": "The First Opener brought judgment upon the Two Lands.",
    },
    {
        "vo_item_id": "vo-s03-002",
        "character_video": "char-khamun-v1-anim.mp4",
        "vo_audio": "vo-s03-002.wav",
        "output": "char-khamun-lipsync-s03-002.mp4",
        "text": "What kind of tomb requires silence enforced by execution?",
    },
    # Note: voice_of_gate (vo-s04-002) has visual=false -- no lipsync needed.
]

WAV2LIP_CHECKPOINT = "checkpoints/wav2lip_gan.pth"   # relative to wav2lip_dir


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply lip-sync to character videos using Wav2Lip.",
        epilog=(
            "Model used:\n\n"
            "  Wav2Lip-GAN   Rudrabha/Wav2Lip  (local repo clone required)\n"
            "                GAN-based lip-sync, ~500 MB checkpoint.\n"
            "                Processes face crops frame-by-frame -- minimal VRAM.\n"
            "                Falls back to ffmpeg audio-merge if repo is not present.\n\n"
            "  SETUP:\n"
            "    git clone https://github.com/Rudrabha/Wav2Lip\n"
            "    # Place checkpoint at: Wav2Lip/checkpoints/wav2lip_gan.pth\n"
            "    # (download link in the Wav2Lip repo README)\n"
            "    pip install -r Wav2Lip/requirements.txt\n"
            "    python gen_lipsync.py --manifest ..\\AssetManifest.json --wav2lip_dir Wav2Lip\n\n"
            "  Quick test (ffmpeg passthrough, no Wav2Lip needed):\n"
            "    python gen_lipsync.py --manifest ..\\AssetManifest.json\n\n"
            "  UPGRADE: LatentSync 1.5 requires RTX 4080 16 GB+.\n"
            "  This script has no --model flag; Wav2Lip is the only supported model."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory containing input videos/audio and output destination.")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to AssetManifest JSON. When given, overrides hardcoded LIPSYNC_JOBS.")
    parser.add_argument("--asset-id", type=str, default=None, dest="asset_id",
                        help="Process only this character asset_id (requires --manifest).")
    parser.add_argument("--anim-variant", type=str, default="slow-subtle", dest="anim_variant",
                        help="Animation variant suffix to look for (default: slow-subtle). "
                             "Matches files named {asset_id}-anim-{variant}.mp4.")
    parser.add_argument("--tts-backend", type=str, default="kokoro", dest="tts_backend",
                        help="TTS backend subfolder under assets/locale/ where WAVs live "
                             "(default: kokoro). Matches gen_tts.py --tts_model output.")
    parser.add_argument("--wav2lip_dir", type=str, default="Wav2Lip",
                        help="Path to cloned Wav2Lip repo.")
    parser.add_argument("--checkpoint", type=str, default=WAV2LIP_CHECKPOINT,
                        help="Path to Wav2Lip checkpoint, relative to wav2lip_dir.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if output already exists.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Manifest loader
# ---------------------------------------------------------------------------
def load_from_manifest(manifest_path: str, asset_id_filter, anim_variant: str, tts_backend: str) -> list[dict]:
    """
    Build lipsync jobs from AssetManifest JSON.
    Pairs each vo_item (visual=true) with its speaker's character animation MP4.
    """
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    char_map = {p["asset_id"]: p for p in manifest.get("character_packs", [])}

    jobs = []
    for vo in manifest.get("vo_items", []):
        if not vo.get("visual", True):
            continue
        speaker = vo.get("speaker_id", "")
        if asset_id_filter and speaker != asset_id_filter:
            continue
        if speaker not in char_map:
            continue
        item_id = vo["item_id"]
        jobs.append({
            "vo_item_id":       item_id,
            "character_video":  f"{speaker}-anim-{anim_variant}.mp4",
            "vo_audio":         f"audio/vo/{item_id}.wav",
            "output":           f"{speaker}-lipsync-{item_id}.mp4",
            "text":             vo.get("text", ""),
        })
    return jobs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def locale_from_manifest_path(path: str) -> str:
    """'AssetManifest.zh-Hans.json' -> 'zh-Hans', 'AssetManifest.json' -> 'en'"""
    stem = Path(path).stem
    parts = stem.split('.')
    return parts[-1] if len(parts) > 1 else 'en'


def output_dir_from_manifest(manifest_path: str, locale: str) -> Path:
    """Derive assets output dir from manifest's project_id + episode_id."""
    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)
    return PROJECTS_ROOT / m["project_id"] / "episodes" / m["episode_id"] / "assets" / locale


def ensure_tts(manifest_path: str) -> None:
    """Auto-run gen_tts.py when VO WAV files are missing."""
    tts_script = Path(__file__).resolve().parent / "gen_tts.py"
    if not tts_script.exists():
        print(f"[AUTO-TTS] gen_tts.py not found -- skipping.")
        return
    print(f"\n[AUTO-TTS] VO WAVs missing -- running gen_tts.py...")
    cmd = [sys.executable, str(tts_script)]
    if manifest_path:
        cmd += ["--manifest", manifest_path]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[AUTO-TTS] gen_tts.py exited with errors -- some WAVs may still be missing.")
    print()


# ---------------------------------------------------------------------------
# Strategy 1 -- subprocess calling Wav2Lip's official inference.py
# ---------------------------------------------------------------------------
def _run_wav2lip(
    wav2lip_dir: Path,
    checkpoint_path: Path,
    video_path: Path,
    audio_path: Path,
    out_path: Path,
) -> bool:
    """
    Call Wav2Lip inference.py via subprocess.
    Returns True on success, False if repo/checkpoint not found.
    """
    inference_script = wav2lip_dir / "inference.py"
    if not inference_script.exists():
        print(f"  [WARN] Wav2Lip inference.py not found at {inference_script}")
        return False

    if not checkpoint_path.exists():
        print(f"  [WARN] Wav2Lip checkpoint not found at {checkpoint_path}")
        print(f"  Download from the Wav2Lip repo README and place at: {checkpoint_path}")
        return False

    cmd = [
        sys.executable, str(inference_script),
        "--checkpoint_path", str(checkpoint_path),
        "--face", str(video_path),
        "--audio", str(audio_path),
        "--outfile", str(out_path),
        "--resize_factor", "1",        # no resizing -- keep input resolution
        "--nosmooth",                   # skip temporal smoothing for speed
    ]
    print(f"  [CMD] Running Wav2Lip inference...")
    result = subprocess.run(cmd, cwd=str(wav2lip_dir), capture_output=False)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Strategy 2 -- audio-merge passthrough (no face manipulation)
# ---------------------------------------------------------------------------
def _find_ffmpeg() -> str:
    """Return ffmpeg executable path -- system PATH first, imageio_ffmpeg bundle fallback."""
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    raise RuntimeError(
        "ffmpeg not found. Install it (winget install Gyan.FFmpeg) or "
        "pip install imageio[ffmpeg]."
    )


def _merge_audio_into_video(video_path: Path, audio_path: Path, out_path: Path):
    """
    Fallback: replace the video's audio track with the VO using ffmpeg.
    Frame content is unchanged -- this validates pipeline I/O without
    requiring the Wav2Lip repo to be present.
    Uses system ffmpeg if available, otherwise imageio_ffmpeg bundle.
    """
    ffmpeg_exe = _find_ffmpeg()

    cmd = [
        ffmpeg_exe, "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    locale = locale_from_manifest_path(args.manifest) if args.manifest else 'en'
    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif args.manifest:
        out_dir = output_dir_from_manifest(args.manifest, locale)
    else:
        out_dir = OUTPUT_DIR / locale
    out_dir.mkdir(parents=True, exist_ok=True)
    wav2lip_dir = Path(args.wav2lip_dir).resolve()
    checkpoint_path = wav2lip_dir / args.checkpoint

    jobs = LIPSYNC_JOBS
    if args.manifest:
        jobs = load_from_manifest(args.manifest, args.asset_id, args.anim_variant, args.tts_backend)
        if not jobs:
            print("[WARN] No visual vo_items found in manifest. Nothing to do.")
            return

    # Auto-TTS: run gen_tts.py if any VO WAVs are missing
    if any(not (out_dir / job["vo_audio"]).exists() for job in jobs):
        ensure_tts(args.manifest)

    results = []
    total = len(jobs)

    for idx, job in enumerate(jobs, start=1):
        out_path = out_dir / job["output"]
        video_path = out_dir / job["character_video"]
        audio_path = out_dir / job["vo_audio"]

        print(f"\n[{idx}/{total}] {job['vo_item_id']} -> {job['output']}")
        print(f"  Video: {video_path}")
        print(f"  Audio: {audio_path}")
        print(f"  Text:  \"{job['text'][:60]}{'...' if len(job['text'])>60 else ''}\"")

        if out_path.exists() and not args.force:
            print(f"  [SKIP] {job['output']} already exists")
            results.append({
                "vo_item_id": job["vo_item_id"],
                "output": str(out_path),
                "size_bytes": out_path.stat().st_size,
                "status": "skipped",
            })
            continue

        if not video_path.exists():
            print(f"  [SKIP] Input video missing: {video_path}")
            print(f"  [HINT] Run gen_character_animation.py first.")
            results.append({
                "vo_item_id": job["vo_item_id"],
                "output": str(out_path),
                "size_bytes": 0,
                "status": "skipped",
                "error": f"Input video missing: {video_path}",
            })
            continue

        if not audio_path.exists():
            print(f"  [SKIP] VO audio missing: {audio_path}")
            print(f"  [HINT] Run gen_tts.py first.")
            results.append({
                "vo_item_id": job["vo_item_id"],
                "output": str(out_path),
                "size_bytes": 0,
                "status": "skipped",
                "error": f"VO audio missing: {audio_path}",
            })
            continue

        try:
            success = False
            method = "passthrough"

            # Strategy 1: Wav2Lip repo (real lip-sync)
            if wav2lip_dir.exists():
                print(f"  [STRATEGY 1] Wav2Lip repo found at {wav2lip_dir}")
                success = _run_wav2lip(
                    wav2lip_dir, checkpoint_path, video_path, audio_path, out_path
                )
                if success:
                    method = "wav2lip"
                    print("  [OK] Wav2Lip inference succeeded.")
                else:
                    print("  [WARN] Wav2Lip failed -- falling back to audio merge.")

            # Strategy 2: audio-merge passthrough
            if not success:
                print("  [STRATEGY 2] Audio-merge passthrough (no face manipulation).")
                _merge_audio_into_video(video_path, audio_path, out_path)
                success = out_path.exists()

            if not success or not out_path.exists():
                raise RuntimeError("Both strategies failed.")

            size = out_path.stat().st_size
            print(f"  [OK] {out_path}  ({size:,} bytes)  [{method}]")
            results.append({
                "vo_item_id": job["vo_item_id"],
                "output": str(out_path),
                "size_bytes": size,
                "method": method,
                "status": "success",
            })

        except Exception as exc:
            print(f"  [ERROR] {job['vo_item_id']}: {exc}")
            results.append({
                "vo_item_id": job["vo_item_id"],
                "output": str(out_path),
                "size_bytes": 0,
                "status": "failed",
                "error": str(exc),
            })
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    # Write manifest
    manifest_path = out_dir / f"{SCRIPT_NAME}_results.json"
    with open(manifest_path, "w") as fh:
        json.dump(results, fh, indent=2)

    # Summary
    wav2lip_count   = sum(1 for r in results if r.get("method") == "wav2lip")
    passthrough_count = sum(1 for r in results if r.get("method") == "passthrough")
    mode_label = "Wav2Lip" if wav2lip_count else "passthrough"
    print("\n" + "=" * 60)
    print(f"SUMMARY -- gen_lipsync  [{mode_label}]")
    print("=" * 60)
    for r in results:
        tag = "OK" if r["status"] == "success" else r["status"].upper()
        method = f"  [{r.get('method', '')}]" if r["status"] == "success" else ""
        print(f"  [{tag}]{method}  {r['output']}  ({r['size_bytes']:,} bytes)")
    ok_count = sum(1 for r in results if r["status"] in ("success", "skipped"))
    total_bytes = sum(r["size_bytes"] for r in results)
    print(f"\n{ok_count}/{len(jobs)} completed | {total_bytes:,} bytes total")
    if wav2lip_count:
        print(f"Wav2Lip: {wav2lip_count} clips | Passthrough: {passthrough_count} clips")
    print(f"Manifest: {manifest_path}")

    if not wav2lip_count:
        print()
        print("NOTE: All clips used audio-merge passthrough (lips do not move).")
        print("  To enable real lip-sync, set up Wav2Lip:")
        print("    git clone https://github.com/Rudrabha/Wav2Lip")
        print("    # Download wav2lip_gan.pth -- see Wav2Lip repo README")
        print("    # Place at: Wav2Lip/checkpoints/wav2lip_gan.pth")
        print("    python gen_lipsync.py --wav2lip_dir Wav2Lip --force")
        print()
        print("  UPGRADE -- LatentSync 1.5 quality (needs RTX 4080 16 GB+):")
        print("    python placeholder_for_lipsync.py")


if __name__ == "__main__":
    main()
