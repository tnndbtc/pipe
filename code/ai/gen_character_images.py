# =============================================================================
# gen_character_images.py
# Generate photorealistic character portrait images for s01e01.
# =============================================================================
#
# requirements.txt (pip install before running):
#   torch>=2.1.0
#   diffusers>=0.28.0
#   transformers>=4.38.0
#   accelerate>=0.27.0
#   bitsandbytes>=0.43.0      # 4-bit quant for FLUX transformer
#   Pillow>=10.0.0
#   huggingface_hub>=0.21.0
#   safetensors>=0.4.0
#
# ---------------------------------------------------------------------------
# Hardware Target: NVIDIA RTX 4060 8 GB VRAM
# ---------------------------------------------------------------------------
# Memory-saving techniques:
#   PRIMARY — FLUX.1-schnell with 4-bit bitsandbytes quantisation:
#     The FLUX transformer is ~24 GB in FP32; load_in_4bit collapses it to
#     ~6 GB on-GPU.  Text encoders + VAE are CPU-offloaded via
#     enable_model_cpu_offload(), so they only occupy VRAM during their
#     individual forward passes.  vae_slicing() further splits the VAE
#     decode into row-by-row chunks to keep peak activation VRAM low.
#
#   FALLBACK — SDXL 1.0 in FP16 + enable_model_cpu_offload():
#     The full SDXL UNet is ~5 GB in FP16.  CPU offload streams each
#     sub-module to GPU only when needed, keeping peak VRAM ~4-5 GB.
#
#   Between images: torch.cuda.empty_cache() + gc.collect() are called
#   after every generation to release fragmented allocations.
#
#   Resolution 512×768 (below SDXL native 1024 px) further reduces
#   activation memory during the UNet forward pass.
#
# NOTE: FLUX.1-schnell requires accepting the licence at:
#   https://huggingface.co/black-forest-labs/FLUX.1-schnell
#   Run `huggingface-cli login` before the first run.
# ---------------------------------------------------------------------------

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# DEFAULTS — fully populated; script runs with no CLI flags.
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("projects/the-pharaoh-who-defied-death/episodes/s01e01/assets")
SCRIPT_NAME = "gen_character_images"

CHARACTERS = [
    {
        "asset_id": "char-ramesses_ka-v1",
        "prompt": (
            "Pharaoh of ancient Egypt, middle-aged male, white and gold royal robes, "
            "double crown of Egypt (pschent), kohl-lined eyes, commanding and obsessive "
            "expression, dramatic side lighting, portrait, photorealistic, cinematic"
        ),
        "negative_prompt": "blurry, low quality, modern clothing, fantasy, cartoon",
        "output": "char-ramesses_ka-v1.png",
    },
    {
        "asset_id": "char-amunhotep-v1",
        "prompt": (
            "Ancient Egyptian High Priest, aged male 60s, shaved head, white ceremonial "
            "robes with gold and lapis lazuli pectoral, dark kohl-lined eyes, expression "
            "of deep fear and reluctant duty, warm oil-lamp portrait lighting, "
            "photorealistic, cinematic"
        ),
        "negative_prompt": "blurry, low quality, modern clothing, fantasy, cartoon",
        "output": "char-amunhotep-v1.png",
    },
    {
        "asset_id": "char-neferet-v1",
        "prompt": (
            "Young Egyptian woman, mid-20s, white linen scribe robes, dark kohl-lined "
            "eyes, intelligent and quietly troubled expression, holding a reed stylus, "
            "warm lamp-lit portrait, ancient Egypt, photorealistic, cinematic"
        ),
        "negative_prompt": "blurry, low quality, modern clothing, fantasy, cartoon",
        "output": "char-neferet-v1.png",
    },
    {
        "asset_id": "char-khamun-v1",
        "prompt": (
            "Ancient Egyptian military general, powerfully built male 40s, bronze chest "
            "armor with cartouche engravings, ceremonial blue war crown, stoic and "
            "morally conflicted expression, warm amber dusk lighting, portrait, "
            "photorealistic"
        ),
        "negative_prompt": "blurry, low quality, modern clothing, fantasy, cartoon",
        "output": "char-khamun-v1.png",
    },
    {
        "asset_id": "char-prisoner-v1",
        "prompt": (
            "Ancient Egyptian condemned prisoner, gaunt male, rough linen garment, "
            "wrists bound with rope, eyes wide with pure terror, harsh dramatic "
            "underlit portrait, photorealistic"
        ),
        "negative_prompt": "blurry, low quality, modern clothing, fantasy, cartoon",
        "output": "char-prisoner-v1.png",
    },
]

# Image dimensions (portrait orientation)
WIDTH = 512
HEIGHT = 768
NUM_STEPS_FLUX = 4    # Schnell is distilled — 4 steps is optimal
NUM_STEPS_SDXL = 25   # SDXL needs more steps
GUIDANCE_FLUX = 0.0   # Schnell is guidance-distilled; CFG scale unused
GUIDANCE_SDXL = 7.5

FLUX_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate character portrait images for s01e01."
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
        help="Path to AssetManifest JSON. When given, overrides the hardcoded CHARACTERS list.",
    )
    parser.add_argument(
        "--asset-id", type=str, default=None, dest="asset_id",
        help="Process only this asset_id (requires --manifest).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------
def load_flux_pipeline():
    """Load FLUX.1-schnell with 4-bit bitsandbytes quantisation."""
    from diffusers import FluxPipeline
    from transformers import BitsAndBytesConfig

    print("[MODEL] Loading FLUX.1-schnell (4-bit quantised)...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    pipe = FluxPipeline.from_pretrained(
        FLUX_MODEL_ID,
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
    )
    # CPU offload: each sub-model moves to GPU only during its forward pass
    pipe.enable_model_cpu_offload()
    # VAE slicing decodes the latent in row chunks — lowers peak VRAM
    pipe.enable_vae_slicing()
    return pipe, "flux"


def load_sdxl_pipeline():
    """Load SDXL 1.0 in FP16 with CPU offload — peak ~4-5 GB VRAM."""
    from diffusers import StableDiffusionXLPipeline

    print("[MODEL] Loading SDXL 1.0 (FP16 + CPU offload)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    return pipe, "sdxl"


def load_pipeline(preference: str):
    """Load the requested model; fall back to SDXL if FLUX fails."""
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
def generate_image(pipe, model_type: str, char: dict, args, generator):
    """Run one inference call and return a PIL Image."""
    if model_type == "flux":
        result = pipe(
            prompt=char["prompt"],
            width=args.width,
            height=args.height,
            num_inference_steps=NUM_STEPS_FLUX,
            guidance_scale=GUIDANCE_FLUX,
            generator=generator,
        )
    else:  # sdxl
        result = pipe(
            prompt=char["prompt"],
            negative_prompt=char["negative_prompt"],
            width=args.width,
            height=args.height,
            num_inference_steps=NUM_STEPS_SDXL,
            guidance_scale=GUIDANCE_SDXL,
            generator=generator,
        )
    return result.images[0]


# ---------------------------------------------------------------------------
# Manifest loader
# ---------------------------------------------------------------------------
def load_from_manifest(manifest_path: str, asset_id_filter):
    """Load character job list from AssetManifest JSON (section: character_packs)."""
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    chars = []
    for pack in manifest.get("character_packs", []):
        if asset_id_filter and pack["asset_id"] != asset_id_filter:
            continue
        aid = pack["asset_id"]
        chars.append({
            "asset_id": aid,
            "prompt": pack["ai_prompt"],
            "negative_prompt": "blurry, low quality, modern clothing, fantasy, cartoon",
            "output": f"{aid}.png",
        })
    return chars


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    characters = CHARACTERS
    if args.manifest:
        characters = load_from_manifest(args.manifest, args.asset_id)
        if not characters:
            print("[WARN] No matching character_packs in manifest. Nothing to do.")
            return

    pipe, model_type = load_pipeline(args.model)
    print(f"[MODEL] Active pipeline: {model_type.upper()}")

    results = []
    total = len(characters)

    for idx, char in enumerate(characters, start=1):
        out_path = out_dir / char["output"]
        print(f"\n[{idx}/{total}] Generating {char['asset_id']}...")

        # Skip if the output already exists
        if out_path.exists():
            print(f"  [SKIP] {char['output']} already exists")
            results.append({
                "asset_id": char["asset_id"],
                "output": str(out_path),
                "size_bytes": out_path.stat().st_size,
                "status": "skipped",
            })
            continue

        try:
            # Seed per-image so each character is reproducible independently
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            image = generate_image(pipe, model_type, char, args, generator)
            image.save(str(out_path), format="PNG")
            size = out_path.stat().st_size
            print(f"  [OK] {out_path}  ({size:,} bytes)")
            results.append({
                "asset_id": char["asset_id"],
                "output": str(out_path),
                "size_bytes": size,
                "status": "success",
            })
        except Exception as exc:
            print(f"  [ERROR] {char['asset_id']}: {exc}")
            results.append({
                "asset_id": char["asset_id"],
                "output": str(out_path),
                "size_bytes": 0,
                "status": "failed",
                "error": str(exc),
            })
        finally:
            # Release GPU allocations before the next image
            torch.cuda.empty_cache()
            gc.collect()

    # Write JSON manifest
    manifest_path = out_dir / f"{SCRIPT_NAME}_results.json"
    with open(manifest_path, "w") as fh:
        json.dump(results, fh, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY — gen_character_images")
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
