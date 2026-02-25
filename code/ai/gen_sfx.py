# =============================================================================
# gen_sfx.py
# Generate short sound-effect audio clips from text descriptions for s01e01.
# =============================================================================
#
# requirements.txt (pip install before running):
#   audiocraft>=1.3.0        # Meta AudioCraft — includes AudioGen
#   torch>=2.1.0
#   torchaudio>=2.1.0
#   soundfile>=0.12.0
#   huggingface_hub>=0.21.0
#
# ---------------------------------------------------------------------------
# Hardware Target: NVIDIA RTX 4060 8 GB VRAM
# ---------------------------------------------------------------------------
# Memory-saving techniques:
#   AudioGen medium (~8 GB VRAM) is used when available; if it OOMs, the
#   script catches the exception and reloads AudioGen small (~4 GB).
#
#   Key optimisations:
#     - Model is kept loaded across all SFX jobs and only freed at the end.
#     - Each job is generated as a single batch item to avoid batch-size
#       memory overhead.
#     - torch.cuda.empty_cache() + gc.collect() after each generation.
#     - If the GPU is insufficient, set DEVICE = "cpu" via --device flag;
#       AudioGen runs on CPU at roughly 0.3× real-time.
#
# NOTE: AudioCraft models are downloaded automatically from HuggingFace
#   (facebook/audiogen-medium or facebook/audiogen-small, MIT licence).
# ---------------------------------------------------------------------------

import argparse
import gc
import json
import re
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# DEFAULTS — fully populated; script runs with no CLI flags.
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("projects/the-pharaoh-who-defied-death/episodes/s01e01/assets")
SCRIPT_NAME = "gen_sfx"

SFX_JOBS = [
    {
        "shot_id": "s01-sh01", "duration": 6.0,
        "tags": [
            "stone scraping in an ancient stone chamber",
            "distant desert wind howling",
            "oil lamp flame flickering",
        ],
    },
    {
        "shot_id": "s01-sh02", "duration": 10.0,
        "tags": ["oil lamp flame flickering softly"],
    },
    {
        "shot_id": "s01-sh03", "duration": 8.0,
        "tags": [
            "multiple oil lamps flickering violently",
            "deep stone hum resonance",
        ],
    },
    {
        "shot_id": "s02-sh01", "duration": 5.0,
        "tags": [
            "papyrus scroll rustling quietly",
            "distant temple ritual chanting",
        ],
    },
    {
        "shot_id": "s02-sh02", "duration": 8.0,
        "tags": ["writing stylus stopping mid-stroke on papyrus"],
    },
    {
        "shot_id": "s03-sh01", "duration": 6.0,
        "tags": [
            "heavy stone blocks being dragged across sand",
            "dry desert wind",
            "distant shouted orders echoing",
        ],
    },
    {
        "shot_id": "s03-sh02", "duration": 7.0,
        "tags": ["footsteps fading away on stone and sand"],
    },
    {
        "shot_id": "s04-sh01", "duration": 5.0,
        "tags": [
            "deep stone resonance booming in a large underground chamber",
            "distant low chanting beginning",
        ],
    },
    {
        "shot_id": "s04-sh02", "duration": 8.0,
        "tags": [
            "bronze rods humming with supernatural energy",
            "ritual chanting building",
            "crackling blue energy ignition",
        ],
    },
    {
        "shot_id": "s04-sh03", "duration": 6.0,
        "tags": [
            "man screaming in terror",
            "sudden total silence",
        ],
    },
    {
        "shot_id": "s04-sh04", "duration": 6.0,
        "tags": [
            "liquid stone surface undulating like black water",
            "whispers building from inside stone",
        ],
    },
    {
        "shot_id": "s04-sh06", "duration": 6.0,
        "tags": ["single massive low-frequency impact cut to silence"],
    },
    {
        "shot_id": "s05-sh01", "duration": 6.0,
        "tags": ["oil lamp crackling in a quiet stone room"],
    },
    {
        "shot_id": "s05-sh02", "duration": 10.0,
        "tags": [
            "oil lamp flickering",
            "deep silence",
        ],
    },
]

AUDIOGEN_MEDIUM = "facebook/audiogen-medium"
AUDIOGEN_SMALL = "facebook/audiogen-small"


# ---------------------------------------------------------------------------
# Manifest loader
# ---------------------------------------------------------------------------
def load_from_manifest(manifest_path: str, asset_id_filter):
    """
    Load SFX job list from AssetManifest JSON (section: sfx_items).
    Returns None if the sfx_items section is absent (caller should use hardcoded SFX_JOBS).
    """
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    sfx_items = manifest.get("sfx_items")
    if sfx_items is None:
        return None  # section absent — caller falls back to hardcoded
    if asset_id_filter:
        sfx_items = [j for j in sfx_items if j.get("shot_id") == asset_id_filter]
    return sfx_items


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def tag_to_slug(tag: str) -> str:
    """Convert a descriptive tag into a safe filename slug (max 50 chars)."""
    slug = re.sub(r"[^a-z0-9]+", "-", tag.lower()).strip("-")
    return slug[:50]


def build_output_filename(shot_id: str, tag: str) -> str:
    """sfx_{shot_id}_{tag_slug}.wav  e.g. sfx_s01-sh01_stone-scraping.wav"""
    return f"sfx_{shot_id}_{tag_to_slug(tag)}.wav"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SFX WAV files for s01e01 using AudioGen."
    )
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cuda / cpu).",
    )
    parser.add_argument(
        "--model",
        choices=["medium", "small", "auto"],
        default="auto",
        help="AudioGen model size. 'auto' tries medium, falls back to small.",
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to AssetManifest JSON. Uses sfx_items section if present; "
             "otherwise falls back to hardcoded SFX_JOBS.",
    )
    parser.add_argument(
        "--asset-id", type=str, default=None, dest="asset_id",
        help="Process only the sfx job matching this shot_id (requires --manifest).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
def load_audiogen(preference: str, device: str):
    """Load AudioGen, falling back to small if medium OOMs."""
    from audiocraft.models import AudioGen

    if preference == "small":
        print(f"[MODEL] Loading AudioGen small...")
        model = AudioGen.get_pretrained(AUDIOGEN_SMALL, device=device)
        return model

    try:
        print(f"[MODEL] Loading AudioGen medium...")
        model = AudioGen.get_pretrained(AUDIOGEN_MEDIUM, device=device)
        # Probe with a tiny generation to confirm it fits in VRAM
        model.set_generation_params(duration=1.0)
        model.generate(["test"])
        torch.cuda.empty_cache()
        print("[MODEL] AudioGen medium fits — using medium.")
        return model
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        print(f"[WARN] AudioGen medium OOM ({exc}). Falling back to small.")
        torch.cuda.empty_cache()
        gc.collect()
        model = AudioGen.get_pretrained(AUDIOGEN_SMALL, device=device)
        return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine job list: manifest sfx_items if present, else hardcoded
    sfx_jobs = SFX_JOBS
    if args.manifest:
        manifest_jobs = load_from_manifest(args.manifest, args.asset_id)
        if manifest_jobs is None:
            print("[INFO] No sfx_items section in manifest — using hardcoded SFX_JOBS.")
        elif not manifest_jobs:
            print("[WARN] sfx_items section is empty (or no match for --asset-id). Nothing to do.")
            return
        else:
            sfx_jobs = manifest_jobs

    # Set global seed for reproducibility
    torch.manual_seed(args.seed)

    model = load_audiogen(args.model, args.device)

    results = []

    # Count total SFX clips to generate (one per tag per shot)
    all_jobs = [
        (job, tag)
        for job in sfx_jobs
        for tag in job["tags"]
    ]
    total = len(all_jobs)
    counter = 0

    for job in sfx_jobs:
        for tag in job["tags"]:
            counter += 1
            filename = build_output_filename(job["shot_id"], tag)
            out_path = out_dir / filename
            print(f"\n[{counter}/{total}] {job['shot_id']} — \"{tag}\"")

            if out_path.exists():
                print(f"  [SKIP] {filename} already exists")
                results.append({
                    "shot_id": job["shot_id"],
                    "tag": tag,
                    "output": str(out_path),
                    "size_bytes": out_path.stat().st_size,
                    "status": "skipped",
                })
                continue

            try:
                # Configure duration for this specific clip
                model.set_generation_params(
                    duration=job["duration"],
                    # top_k=250 and temperature=1.0 are AudioGen defaults
                )
                # Generate one clip (batch size = 1)
                wav = model.generate([tag])  # shape: [1, channels, samples]

                # Save using soundfile for reliable WAV writing
                import soundfile as sf
                audio_np = wav[0, 0].cpu().numpy()  # mono: take channel 0
                sf.write(str(out_path), audio_np, model.sample_rate, subtype="PCM_16")

                size = out_path.stat().st_size
                print(f"  [OK] {out_path}  ({job['duration']}s, {size:,} bytes)")
                results.append({
                    "shot_id": job["shot_id"],
                    "tag": tag,
                    "output": str(out_path),
                    "size_bytes": size,
                    "status": "success",
                })
            except Exception as exc:
                print(f"  [ERROR] {filename}: {exc}")
                results.append({
                    "shot_id": job["shot_id"],
                    "tag": tag,
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
    print("SUMMARY — gen_sfx")
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
