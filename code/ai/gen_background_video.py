# =============================================================================
# gen_background_video.py
# Generate short cinematic background video clips for shots that require
# camera motion (pan, crane, zoom) in s01e01.
#
# 8 GB VRAM FALLBACK: CogVideoX-2b (~6 GB)
# Original target model (LTX-Video 0.9.8-13B) requires ~24 GB VRAM and
# cannot run on an RTX 4060 8 GB. See placeholder_for_background_video.py
# for the full GPU requirements of the original model.
# =============================================================================
#
# requirements.txt (pip install before running):
#   torch>=2.4.1
#   diffusers>=0.30.0        # CogVideoXPipeline added in 0.30
#   transformers>=4.40.0
#   accelerate>=0.30.0
#   imageio[ffmpeg]>=2.34.0
#   huggingface_hub>=0.21.0
#   safetensors>=0.4.0
#
# ---------------------------------------------------------------------------
# Hardware Target: NVIDIA RTX 4060 8 GB VRAM
# ---------------------------------------------------------------------------
# Memory-saving techniques:
#   CogVideoX-2b (2B params) in FP16 fits within 8 GB with:
#
#   1. torch.float16: halves weight memory vs FP32.
#   2. enable_model_cpu_offload(): transformer and VAE stream through GPU
#      only during their individual forward passes — persistent VRAM is just
#      the active module (~4-5 GB peak).
#   3. enable_vae_slicing(): VAE decode processes the video in frame-sliced
#      chunks, keeping decode peak VRAM flat regardless of frame count.
#   4. enable_vae_tiling(): spatial tiling further reduces activation memory
#      for each frame during the VAE decode pass.
#   5. num_inference_steps=25 (vs default 50) for faster prototype runs;
#      increase to 50 for better quality if time allows.
#   6. torch.cuda.empty_cache() + gc.collect() between videos.
#
# OUTPUT NOTE: CogVideoX-2b default resolution is 720×480 (3:2 ratio).
#   True 16:9 (e.g. 768×432) is not guaranteed to be supported by this
#   model version without fine-tuning. Use 720×480 for prototyping.
#
# UPGRADE PATH: For full LTX-Video 0.9.8-13B quality, see:
#   placeholder_for_background_video.py  (requires RTX 3090/4090 24 GB)
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
SCRIPT_NAME = "gen_background_video"

VIDEO_BACKGROUNDS = [
    {
        "asset_id": "bg-desert-excavation-site-v1",
        "prompt": (
            "Ancient Egyptian desert plateau excavation site at dusk, deep rectangular "
            "shaft descending into the sand, workers silhouetted against amber sunset sky, "
            "torchlight from below, dramatic long shadows, photorealistic, cinematic wide, "
            "slow camera pan left to right"
        ),
        "negative_prompt": "blurry, low quality, modern, fast motion, shaky",
        "motion_description": (
            "slow wide pan across the excavation site, revealing the scale of the shaft "
            "and the workers descending"
        ),
        "duration_sec": 6.0,
        "num_frames": 49,   # 6s × 8fps + 1
        "output": "bg-desert-excavation-site-v1.mp4",
    },
    {
        "asset_id": "bg-underground-chamber-v1",
        "prompt": (
            "Enormous circular underground chamber of polished black stone, ancient Egyptian "
            "star map engraved into domed ceiling, oil lamps nearly extinguished, eerie blue "
            "supernatural glow emanating from a vertical black slab at center, ring of "
            "shadowed priests, photorealistic, extreme wide cinematic, slow camera descending "
            "from ceiling"
        ),
        "negative_prompt": "blurry, low quality, modern, fast motion, shaky",
        "motion_description": (
            "slow descending crane shot from ceiling height down to chamber floor, revealing "
            "the full scale of the chamber and the glowing slab at its center"
        ),
        "duration_sec": 5.0,
        "num_frames": 41,   # 5s × 8fps + 1
        "output": "bg-underground-chamber-v1.mp4",
    },
]

COGVIDEOX_MODEL_ID = "THUDM/CogVideoX-2b"

# CogVideoX-2b native resolution (3:2 landscape); best to use as-is
WIDTH = 720
HEIGHT = 480
FPS = 8
NUM_STEPS = 25      # 50 for quality; 25 for faster prototype
GUIDANCE_SCALE = 6.0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate background video clips using CogVideoX-2b (RTX 4060 8 GB)."
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--steps", type=int, default=NUM_STEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to AssetManifest JSON. When given, overrides the hardcoded VIDEO_BACKGROUNDS list.",
    )
    parser.add_argument(
        "--asset-id", type=str, default=None, dest="asset_id",
        help="Process only this asset_id (requires --manifest).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Manifest loader
# ---------------------------------------------------------------------------
def load_from_manifest(manifest_path: str, asset_id_filter):
    """Load animated background job list from AssetManifest JSON (motion.type == "camera")."""
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    videos = []
    for bg in manifest.get("backgrounds", []):
        motion = bg.get("motion")
        if not motion or motion.get("type") != "camera":
            continue
        if asset_id_filter and bg["asset_id"] != asset_id_filter:
            continue
        aid = bg["asset_id"]
        duration = float(motion.get("duration_sec", 5.0))
        num_frames = int(duration * 8) + 1
        motion_desc = motion.get("description", "slow camera motion")
        videos.append({
            "asset_id":         aid,
            "prompt":           bg["ai_prompt"] + ", slow camera motion",
            "negative_prompt":  "blurry, low quality, modern, fast motion, shaky",
            "motion_description": motion_desc,
            "duration_sec":     duration,
            "num_frames":       num_frames,
            "output":           f"{aid}.mp4",
        })
    return videos


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
def load_cogvideox_pipeline():
    """
    Load CogVideoX-2b in FP16 with full CPU offload + VAE slicing/tiling.
    Requires diffusers >= 0.30.0.
    """
    from diffusers import CogVideoXPipeline

    print(f"[MODEL] Loading {COGVIDEOX_MODEL_ID} (FP16 + CPU offload)...")
    pipe = CogVideoXPipeline.from_pretrained(
        COGVIDEOX_MODEL_ID,
        torch_dtype=torch.float16,
    )
    # Each sub-module moves to GPU only for its forward pass
    pipe.enable_model_cpu_offload()
    # Process VAE in frame slices to cap decode VRAM
    pipe.enable_vae_slicing()
    # Process each frame in spatial tiles to reduce activation memory
    pipe.enable_vae_tiling()
    print("[MODEL] CogVideoX-2b pipeline ready.")
    return pipe


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def generate_video(pipe, job: dict, args) -> list:
    """Run CogVideoX inference and return a list of PIL Image frames."""
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    generator.manual_seed(args.seed)

    result = pipe(
        prompt=job["prompt"],
        negative_prompt=job["negative_prompt"],
        height=args.height,
        width=args.width,
        num_frames=job["num_frames"],
        num_inference_steps=args.steps,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    )
    # CogVideoX returns result.frames as list-of-lists:
    # result.frames[0] = PIL Image list for the first batch item
    return result.frames[0]


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

    video_backgrounds = VIDEO_BACKGROUNDS
    if args.manifest:
        video_backgrounds = load_from_manifest(args.manifest, args.asset_id)
        if not video_backgrounds:
            print("[WARN] No matching animated backgrounds (motion.type=camera) in manifest. Nothing to do.")
            return

    pipe = load_cogvideox_pipeline()

    results = []
    total = len(video_backgrounds)

    for idx, job in enumerate(video_backgrounds, start=1):
        out_path = out_dir / job["output"]
        print(f"\n[{idx}/{total}] Generating {job['asset_id']}...")
        print(f"  Motion:   {job['motion_description']}")
        print(f"  Duration: {job['duration_sec']}s  Frames: {job['num_frames']}  @ {args.fps} fps")

        if out_path.exists():
            print(f"  [SKIP] {job['output']} already exists")
            results.append({
                "asset_id": job["asset_id"],
                "output": str(out_path),
                "size_bytes": out_path.stat().st_size,
                "status": "skipped",
            })
            continue

        try:
            from diffusers.utils import export_to_video

            frames = generate_video(pipe, job, args)
            export_to_video(frames, str(out_path), fps=args.fps)
            size = out_path.stat().st_size
            print(f"  [OK] {out_path}  ({len(frames)} frames, {size:,} bytes)")
            results.append({
                "asset_id": job["asset_id"],
                "output": str(out_path),
                "size_bytes": size,
                "num_frames": len(frames),
                "model": COGVIDEOX_MODEL_ID,
                "status": "success",
            })
        except Exception as exc:
            print(f"  [ERROR] {job['asset_id']}: {exc}")
            results.append({
                "asset_id": job["asset_id"],
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
    print("SUMMARY — gen_background_video  [CogVideoX-2b fallback]")
    print("=" * 60)
    for r in results:
        tag = "OK" if r["status"] == "success" else r["status"].upper()
        print(f"  [{tag}]  {r['output']}  ({r['size_bytes']:,} bytes)")
    ok_count = sum(1 for r in results if r["status"] in ("success", "skipped"))
    total_bytes = sum(r["size_bytes"] for r in results)
    print(f"\n{ok_count}/{total} completed | {total_bytes:,} bytes total")
    print(f"Manifest: {manifest_path}")
    print(f"\nNote: Using CogVideoX-2b (8 GB fallback). For LTX-Video 13B quality,")
    print(f"run placeholder_for_background_video.py to see GPU requirements.")


if __name__ == "__main__":
    main()
