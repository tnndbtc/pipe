# =============================================================================
# gen_background_images.py
# Generate wide cinematic background plate images for s01e01.
# =============================================================================
#
# requirements.txt (pip install before running):
#   torch>=2.1.0
#   diffusers>=0.28.0
#   transformers>=4.38.0
#   accelerate>=0.27.0
#   bitsandbytes>=0.43.0
#   Pillow>=10.0.0
#   huggingface_hub>=0.21.0
#   safetensors>=0.4.0
#
# ---------------------------------------------------------------------------
# Hardware Target: NVIDIA RTX 4060 8 GB VRAM
# ---------------------------------------------------------------------------
# Memory-saving techniques:
#   PRIMARY — FLUX.1-schnell 4-bit bitsandbytes quantisation:
#     The FLUX transformer is quantised from ~24 GB (FP32) to ~6 GB via
#     load_in_4bit.  enable_model_cpu_offload() streams text encoders and
#     VAE through the GPU only during their forward pass.  enable_vae_tiling()
#     is used here (instead of slicing) because the landscape 1280×720
#     latent is large — tiling processes sub-regions of the latent grid
#     independently, keeping decode VRAM flat regardless of resolution.
#
#   FALLBACK — SDXL 1.0 FP16 + CPU offload:
#     Identical strategy to gen_character_images.py.  SDXL native resolution
#     is 1024 px; we use 1280×720 which is slightly larger on one axis but
#     FP16 + CPU offload keeps it within budget.
#
#   torch.cuda.empty_cache() + gc.collect() after each image.
#
# NOTE: FLUX.1-schnell requires accepting the licence at:
#   https://huggingface.co/black-forest-labs/FLUX.1-schnell
#   Run `huggingface-cli login` before the first run.
# ---------------------------------------------------------------------------

import argparse
import gc
import json
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# DEFAULTS — fully populated; script runs with no CLI flags.
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("projects/the-pharaoh-who-defied-death/episodes/s01e01/assets")
SCRIPT_NAME = "gen_background_images"

BACKGROUNDS = [
    {
        "asset_id": "bg-karnak-inner-sanctuary-v1",
        "prompt": (
            "Ancient Egyptian inner temple sanctuary at night, massive hieroglyph-covered "
            "stone columns, oil lamps in wall sconces casting deep amber shadows, sacred "
            "stone altar at center, dramatic chiaroscuro, photorealistic, "
            "ultra-wide cinematic 2.39:1"
        ),
        "negative_prompt": "people, characters, modern, blurry, low quality",
        "color_mood": "dark",
        "output": "bg-karnak-inner-sanctuary-v1.png",
    },
    {
        "asset_id": "bg-temple-forbidden-archives-v1",
        "prompt": (
            "Ancient Egyptian temple archive room, towering stone shelves packed with "
            "papyrus scroll tubes, warm golden daylight through high clerestory windows, "
            "dust motes in air, stone floor with faded painted borders, "
            "photorealistic, cinematic wide"
        ),
        "negative_prompt": "people, characters, modern, blurry, low quality",
        "color_mood": "warm",
        "output": "bg-temple-forbidden-archives-v1.png",
    },
]

# Landscape 16:9 — background plates behind characters
WIDTH = 1280
HEIGHT = 720
NUM_STEPS_FLUX = 4
NUM_STEPS_SDXL = 25
GUIDANCE_FLUX = 0.0   # Schnell is guidance-distilled
GUIDANCE_SDXL = 7.5

FLUX_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate cinematic background plate images for s01e01."
    )
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        choices=["flux", "sdxl", "auto"],
        default="auto",
        help="Model to use. 'auto' tries FLUX first, falls back to SDXL.",
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to AssetManifest JSON. When given, overrides the hardcoded BACKGROUNDS list.",
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
    """Load static background job list from AssetManifest JSON (section: backgrounds, motion==null)."""
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    bgs = []
    for bg in manifest.get("backgrounds", []):
        if bg.get("motion") is not None:
            continue  # skip animated backgrounds
        if asset_id_filter and bg["asset_id"] != asset_id_filter:
            continue
        aid = bg["asset_id"]
        bgs.append({
            "asset_id": aid,
            "prompt": bg["ai_prompt"],
            "negative_prompt": "people, characters, modern, blurry, low quality",
            "color_mood": bg.get("search_filters", {}).get("color_mood", "neutral"),
            "output": f"{aid}.png",
        })
    return bgs


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------
def load_flux_pipeline():
    """Load FLUX.1-schnell with 4-bit quantisation + VAE tiling for large images."""
    from diffusers import FluxPipeline
    from transformers import BitsAndBytesConfig

    print("[MODEL] Loading FLUX.1-schnell (4-bit quantised, VAE tiling)...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    pipe = FluxPipeline.from_pretrained(
        FLUX_MODEL_ID,
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    # Tiling processes the latent in spatial patches — better for 1280×720
    pipe.enable_vae_tiling()
    return pipe, "flux"


def load_sdxl_pipeline():
    """Load SDXL 1.0 FP16 with CPU offload and VAE tiling."""
    from diffusers import StableDiffusionXLPipeline

    print("[MODEL] Loading SDXL 1.0 (FP16 + CPU offload + VAE tiling)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()
    return pipe, "sdxl"


def load_pipeline(preference: str):
    if preference == "sdxl":
        return load_sdxl_pipeline()
    try:
        return load_flux_pipeline()
    except Exception as exc:
        print(f"[WARN] FLUX load failed ({exc}). Falling back to SDXL.")
        return load_sdxl_pipeline()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def generate_image(pipe, model_type: str, bg: dict, args, generator):
    """Run inference and return a PIL Image."""
    if model_type == "flux":
        result = pipe(
            prompt=bg["prompt"],
            width=args.width,
            height=args.height,
            num_inference_steps=NUM_STEPS_FLUX,
            guidance_scale=GUIDANCE_FLUX,
            generator=generator,
        )
    else:  # sdxl
        result = pipe(
            prompt=bg["prompt"],
            negative_prompt=bg["negative_prompt"],
            width=args.width,
            height=args.height,
            num_inference_steps=NUM_STEPS_SDXL,
            guidance_scale=GUIDANCE_SDXL,
            generator=generator,
        )
    return result.images[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backgrounds = BACKGROUNDS
    if args.manifest:
        backgrounds = load_from_manifest(args.manifest, args.asset_id)
        if not backgrounds:
            print("[WARN] No matching static backgrounds in manifest. Nothing to do.")
            return

    pipe, model_type = load_pipeline(args.model)
    print(f"[MODEL] Active pipeline: {model_type.upper()}")

    results = []
    total = len(backgrounds)

    for idx, bg in enumerate(backgrounds, start=1):
        out_path = out_dir / bg["output"]
        print(f"\n[{idx}/{total}] Generating {bg['asset_id']} (mood: {bg['color_mood']})...")

        if out_path.exists():
            print(f"  [SKIP] {bg['output']} already exists")
            results.append({
                "asset_id": bg["asset_id"],
                "output": str(out_path),
                "size_bytes": out_path.stat().st_size,
                "status": "skipped",
            })
            continue

        try:
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            image = generate_image(pipe, model_type, bg, args, generator)
            image.save(str(out_path), format="PNG")
            size = out_path.stat().st_size
            print(f"  [OK] {out_path}  ({size:,} bytes)")
            results.append({
                "asset_id": bg["asset_id"],
                "output": str(out_path),
                "size_bytes": size,
                "status": "success",
            })
        except Exception as exc:
            print(f"  [ERROR] {bg['asset_id']}: {exc}")
            results.append({
                "asset_id": bg["asset_id"],
                "output": str(out_path),
                "size_bytes": 0,
                "status": "failed",
                "error": str(exc),
            })
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    # Write JSON manifest
    manifest_path = out_dir / f"{SCRIPT_NAME}_results.json"
    with open(manifest_path, "w") as fh:
        json.dump(results, fh, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — gen_background_images")
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
