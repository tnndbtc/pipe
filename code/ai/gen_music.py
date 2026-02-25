# =============================================================================
# gen_music.py
# Generate background music tracks for each shot in s01e01.
# =============================================================================
#
# requirements.txt (pip install before running):
#   audiocraft>=1.3.0        # Meta AudioCraft — includes MusicGen
#   torch>=2.1.0
#   torchaudio>=2.1.0
#   soundfile>=0.12.0
#   huggingface_hub>=0.21.0
#
# ---------------------------------------------------------------------------
# Hardware Target: NVIDIA RTX 4060 8 GB VRAM
# ---------------------------------------------------------------------------
# Memory-saving techniques:
#   MusicGen small (300M params) uses ~4-6 GB VRAM and is the primary model.
#   MusicGen medium (1.5B params) uses ~8-10 GB and is attempted if the user
#   passes --model medium; it may fit if no other processes hold VRAM.
#
#   Key optimisations:
#     - The model is loaded once and reused across all jobs.
#     - Each track is generated with batch_size=1.
#     - torch.cuda.empty_cache() + gc.collect() after each track.
#     - CPU fallback via --device cpu for low-VRAM environments (slower).
#     - Output is stereo 32 kHz WAV (MusicGen native format).
#
# NOTE: MusicGen weights are downloaded automatically from HuggingFace
#   (facebook/musicgen-small, MIT licence). Medium is also MIT.
# ---------------------------------------------------------------------------

import argparse
import gc
import json
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# DEFAULTS — fully populated; script runs with no CLI flags.
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "projects" / "the-pharaoh-who-defied-death" / "episodes" / "s01e01" / "assets"
SCRIPT_NAME = "gen_music"

MUSIC_JOBS = [
    {
        "shot_id": "s01-sh01", "duration": 6.0,
        "prompt": "low ceremonial dread, ancient Egyptian ritual, slow deep drums, ominous strings, dark atmosphere",
    },
    {
        "shot_id": "s01-sh02", "duration": 10.0,
        "prompt": "sparse tense underscore, minimal strings, uneasy silence punctuated by low tones, ancient mystery",
    },
    {
        "shot_id": "s01-sh03", "duration": 8.0,
        "prompt": "single sustained low tone, deep cello drone, growing unease, ancient Egyptian horror",
    },
    {
        "shot_id": "s02-sh01", "duration": 5.0,
        "prompt": "quiet neutral ambient, soft warm tones, peaceful temple interior, gentle and undramatic",
    },
    {
        "shot_id": "s02-sh02", "duration": 8.0,
        "prompt": "silence then low subsonic pulse, creeping dread, ancient horror building slowly",
    },
    {
        "shot_id": "s03-sh01", "duration": 6.0,
        "prompt": "low military tension, slow steady drums, brass undertone, oppressive authority, ancient Egyptian",
    },
    {
        "shot_id": "s03-sh02", "duration": 7.0,
        "prompt": "near silence, single low cello note, moral weight, dark and still",
    },
    {
        "shot_id": "s04-sh01", "duration": 5.0,
        "prompt": "vast hollow ceremonial dread, enormous reverb, ancient choral whispers, supernatural awe",
    },
    {
        "shot_id": "s04-sh02", "duration": 8.0,
        "prompt": "building ancient ritual music, accelerating percussion, chanting choir, supernatural energy rising",
    },
    {
        "shot_id": "s04-sh04", "duration": 6.0,
        "prompt": "silence then deep subsonic pulse, infrasound dread, otherworldly resonance",
    },
    {
        "shot_id": "s04-sh06", "duration": 6.0,
        "prompt": "single massive orchestral impact then total silence, shocking revelation sting",
    },
    {
        "shot_id": "s05-sh01", "duration": 6.0,
        "prompt": "low ominous strings, slow cello ostinato, creeping inevitable doom, ancient Egyptian",
    },
    {
        "shot_id": "s05-sh02", "duration": 10.0,
        "prompt": "cold inevitable underscore, sparse piano and strings, isolation and dread, haunting",
    },
]

MUSICGEN_SMALL = "facebook/musicgen-small"
MUSICGEN_MEDIUM = "facebook/musicgen-medium"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate music WAV tracks for s01e01 using MusicGen."
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cuda / cpu).",
    )
    parser.add_argument(
        "--model",
        choices=["small", "medium"],
        default="small",
        help="MusicGen model size. small=~4-6 GB VRAM, medium=~8-10 GB.",
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to AssetManifest JSON. Uses music_items section if present; "
             "otherwise falls back to hardcoded MUSIC_JOBS.",
    )
    parser.add_argument(
        "--asset-id", type=str, default=None, dest="asset_id",
        help="Process only the music job matching this shot_id (requires --manifest).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Manifest loader
# ---------------------------------------------------------------------------
def load_from_manifest(manifest_path: str, asset_id_filter):
    """
    Load music job list from AssetManifest JSON (section: music_items).
    Returns None if the music_items section is absent (caller should use hardcoded MUSIC_JOBS).
    """
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    music_items = manifest.get("music_items")
    if music_items is None:
        return None  # section absent — caller falls back to hardcoded
    if asset_id_filter:
        music_items = [j for j in music_items if j.get("shot_id") == asset_id_filter]
    return music_items


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
def load_musicgen(size: str, device: str):
    """Load MusicGen model."""
    from audiocraft.models import MusicGen

    model_id = MUSICGEN_MEDIUM if size == "medium" else MUSICGEN_SMALL
    print(f"[MODEL] Loading MusicGen {size} ({model_id})...")
    # AudioCraft models don't support .to(device) — pass device to get_pretrained()
    model = MusicGen.get_pretrained(model_id, device=device)
    print(f"[MODEL] MusicGen {size} ready.")
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def locale_from_manifest_path(path: str) -> str:
    """Extract locale from manifest filename.
    'AssetManifest_draft.zh-Hans.json' -> 'zh-Hans'
    'AssetManifest_draft.json'          -> 'en'
    """
    stem = Path(path).stem
    parts = stem.split('.')
    return parts[-1] if len(parts) > 1 else 'en'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    locale = locale_from_manifest_path(args.manifest) if args.manifest else 'en'
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR / locale
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine job list: manifest music_items if present, else hardcoded
    music_jobs = MUSIC_JOBS
    if args.manifest:
        manifest_jobs = load_from_manifest(args.manifest, args.asset_id)
        if manifest_jobs is None:
            print("[INFO] No music_items section in manifest — using hardcoded MUSIC_JOBS.")
        elif not manifest_jobs:
            print("[WARN] music_items section is empty (or no match for --asset-id). Nothing to do.")
            return
        else:
            music_jobs = manifest_jobs

    torch.manual_seed(args.seed)

    model = load_musicgen(args.model, args.device)

    results = []
    total = len(music_jobs)

    for idx, job in enumerate(music_jobs, start=1):
        filename = f"music_{job['shot_id']}.wav"
        out_path = out_dir / filename
        print(f"\n[{idx}/{total}] {job['shot_id']}  ({job['duration']}s)")
        print(f"  Prompt: \"{job['prompt'][:70]}{'...' if len(job['prompt'])>70 else ''}\"")

        if out_path.exists():
            print(f"  [SKIP] {filename} already exists")
            results.append({
                "shot_id": job["shot_id"],
                "output": str(out_path),
                "size_bytes": out_path.stat().st_size,
                "status": "skipped",
            })
            continue

        try:
            # Set duration for this specific track
            model.set_generation_params(
                duration=job["duration"],
                # top_k=250, temperature=1.0 are MusicGen defaults
            )
            # Generate stereo music (batch size = 1)
            wav = model.generate([job["prompt"]])  # shape: [1, channels, samples]

            # Write stereo WAV (MusicGen outputs 2 channels at 32 kHz)
            import soundfile as sf
            audio_np = wav[0].cpu().numpy()  # shape: [channels, samples]
            audio_np = audio_np.T            # soundfile expects [samples, channels]
            sf.write(str(out_path), audio_np, model.sample_rate, subtype="PCM_16")

            size = out_path.stat().st_size
            print(f"  [OK] {out_path}  ({job['duration']}s, {size:,} bytes)")
            results.append({
                "shot_id": job["shot_id"],
                "output": str(out_path),
                "size_bytes": size,
                "status": "success",
            })
        except Exception as exc:
            print(f"  [ERROR] {job['shot_id']}: {exc}")
            results.append({
                "shot_id": job["shot_id"],
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
    print("\n" + "=" * 60)
    print("SUMMARY — gen_music")
    print("=" * 60)
    for r in results:
        tag = "OK" if r["status"] == "success" else r["status"].upper()
        print(f"  [{tag}]  {r['output']}  ({r['size_bytes']:,} bytes)")
    ok_count = sum(1 for r in results if r["status"] in ("success", "skipped"))
    total_bytes = sum(r["size_bytes"] for r in results)
    print(f"\n{ok_count}/{total} completed | {total_bytes:,} bytes total")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
