# =============================================================================
# gen_background_images.py
# Generate wide cinematic background plate images for s01e01.
# STATUS: VALIDATED
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
# Supported models (--model flag):
#
#   flux-schnell  black-forest-labs/FLUX.1-schnell
#                 4-bit BnB quant, ~6 GB VRAM. 4 steps. Fast.
#                 Needs HF login: huggingface-cli login
#
#   flux-dev      black-forest-labs/FLUX.1-dev
#                 4-bit BnB quant, ~6 GB VRAM. 28 steps. Best FLUX quality.
#                 Needs HF login (gated model).
#
#   sdxl          stabilityai/stable-diffusion-xl-base-1.0
#                 FP16 + CPU offload, ~5 GB VRAM. 25 steps.
#
#   sdxl-turbo    stabilityai/sdxl-turbo
#                 FP16 + CPU offload, ~5 GB VRAM. 4 steps. Fast.
#
#   auto          Tries flux-schnell first, falls back to sdxl on failure.
#
#   all           Runs every model above in sequence. Outputs are saved with
#                 model suffix: bg-karnak-inner-sanctuary-v1_flux-schnell.png
#                 Use this to compare quality across models.
#
# Memory-saving techniques:
#   FLUX: 4-bit bitsandbytes quantisation collapses transformer from ~24 GB
#   to ~6 GB. enable_model_cpu_offload() streams text encoder + VAE to CPU.
#   enable_vae_tiling() (not slicing) is used here because the landscape
#   1280×720 latent is large — tiling processes spatial sub-regions of the
#   latent grid independently, keeping decode VRAM flat at any resolution.
#   SDXL: FP16 UNet ~5 GB. CPU offload + VAE tiling keeps peak VRAM ~4-5 GB.
#   Between images: torch.cuda.empty_cache() + gc.collect() after every gen.
#
# NOTE: FLUX.1-schnell requires accepting the licence at:
#   https://huggingface.co/black-forest-labs/FLUX.1-schnell
#   Run `huggingface-cli login` before the first run.
# ---------------------------------------------------------------------------

import argparse
import gc
import json
import re
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# DEFAULTS — fully populated; script runs with no CLI flags.
# ---------------------------------------------------------------------------
OUTPUT_DIR  = Path(__file__).resolve().parent.parent.parent / "projects" / "the-pharaoh-who-defied-death" / "episodes" / "s01e01" / "assets"
SCRIPT_NAME = "gen_background_images"
PROMPT_OUT_DIR = Path(__file__).resolve().parent / "gen_image_output"

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
WIDTH  = 1280
HEIGHT = 720

# ---------------------------------------------------------------------------
# Model registry
# Each entry: model_id, steps, guidance, loader_family ("flux" | "sdxl")
# ---------------------------------------------------------------------------
MODELS = {
    "flux-schnell": {
        "model_id": "black-forest-labs/FLUX.1-schnell",
        "steps":    4,
        "guidance": 0.0,   # guidance-distilled — CFG unused
        "family":   "flux",
        "notes":    "Fast distilled FLUX, 4-bit quant, ~6 GB VRAM",
    },
    "flux-dev": {
        "model_id": "black-forest-labs/FLUX.1-dev",
        "steps":    28,
        "guidance": 3.5,
        "family":   "flux",
        "notes":    "Higher quality FLUX, 4-bit quant, ~6 GB VRAM. Needs HF auth (gated).",
    },
    "sdxl": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "steps":    25,
        "guidance": 7.5,
        "family":   "sdxl",
        "notes":    "SDXL 1.0 FP16 + CPU offload, ~5 GB VRAM",
    },
    "sdxl-turbo": {
        "model_id": "stabilityai/sdxl-turbo",
        "steps":    4,
        "guidance": 0.0,   # guidance-distilled
        "family":   "sdxl",
        "notes":    "Distilled SDXL, 4 steps, FP16 + CPU offload, ~5 GB VRAM",
    },
}

ALL_MODEL_KEYS = list(MODELS.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def prompt_slug(prompt: str, max_len: int = 48) -> str:
    """Derive a filesystem-safe slug from a prompt string."""
    slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")
    return slug[:max_len]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    model_help = "\n".join(
        f"  {key:<14} {cfg['model_id']}  [{cfg['steps']} steps, guidance={cfg['guidance']}]\n"
        f"               {cfg['notes']}"
        for key, cfg in MODELS.items()
    )
    parser = argparse.ArgumentParser(
        description="Generate cinematic background plate images.",
        epilog=(
            "Available --model values:\n\n"
            + model_help +
            "\n\n  auto           Tries flux-schnell, falls back to sdxl on failure."
            "\n  all            Runs every model above in sequence for comparison."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help=(
            "Generate a single image from this prompt, bypassing the manifest "
            "and hardcoded background list. Output goes to gen_image_output/ "
            "next to this script (override with --output or --output_dir)."
        ),
    )
    parser.add_argument(
        "--negative-prompt", dest="negative_prompt",
        default="blurry, low quality, distorted",
        help="Negative prompt for SDXL (ignored by FLUX). Default: 'blurry, low quality, distorted'",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Full output file path for --prompt mode (PNG). Overrides --output_dir.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--width",  type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument(
        "--model",
        choices=ALL_MODEL_KEYS + ["auto", "all"],
        default="auto",
        help=(
            "Model to use. "
            "'auto' tries flux-schnell, falls back to sdxl. "
            "'all' runs every model in sequence for side-by-side comparison."
        ),
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to AssetManifest JSON. When given, overrides the hardcoded BACKGROUNDS list.",
    )
    parser.add_argument(
        "--asset-id", type=str, default=None, dest="asset_id",
        help="Process only this asset_id (requires --manifest).",
    )
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output files instead of skipping them.")
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
def load_flux_pipeline(model_key: str):
    """Load a FLUX model with 4-bit quantisation + VAE tiling for large images.

    diffusers >=0.32 no longer accepts quantization_config on the pipeline
    directly.  The transformer must be quantized separately, then injected.
    """
    from diffusers import FluxPipeline, FluxTransformer2DModel
    from transformers import BitsAndBytesConfig

    cfg = MODELS[model_key]
    print(f"[MODEL] Loading {model_key} ({cfg['model_id']}) — 4-bit quant (transformer) + VAE tiling...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        cfg["model_id"],
        subfolder="transformer",
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
    )
    print(f"[MODEL] Loading full pipeline with quantized transformer...")
    pipe = FluxPipeline.from_pretrained(
        cfg["model_id"],
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()   # tiling for 1280×720 latent, not slicing
    return pipe


def load_sdxl_pipeline(model_key: str):
    """Load an SDXL-family model in FP16 with CPU offload + VAE tiling."""
    from diffusers import StableDiffusionXLPipeline

    cfg = MODELS[model_key]
    print(f"[MODEL] Loading {model_key} ({cfg['model_id']}) — FP16 + CPU offload + VAE tiling...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        cfg["model_id"],
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()
    return pipe


def load_pipeline(model_key: str):
    """Load the pipeline for model_key. Returns (pipe, model_key)."""
    family = MODELS[model_key]["family"]
    if family == "flux":
        return load_flux_pipeline(model_key), model_key
    else:
        return load_sdxl_pipeline(model_key), model_key


def unload_pipeline(pipe):
    """Aggressively free VRAM after a model run."""
    del pipe
    torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def generate_image(pipe, model_key: str, bg: dict, args, generator):
    """Run inference and return a PIL Image."""
    cfg = MODELS[model_key]
    print(f"    prompt: {bg['prompt'][:100]}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    if cfg["family"] == "flux":
        result = pipe(
            prompt=bg["prompt"],
            width=args.width,
            height=args.height,
            num_inference_steps=cfg["steps"],
            guidance_scale=cfg["guidance"],
            generator=generator,
        )
    else:  # sdxl family
        result = pipe(
            prompt=bg["prompt"],
            negative_prompt=bg.get("negative_prompt", ""),
            width=args.width,
            height=args.height,
            num_inference_steps=cfg["steps"],
            guidance_scale=cfg["guidance"],
            generator=generator,
        )

    elapsed = time.perf_counter() - t0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"    [PERF]  time={elapsed:.1f}s  peak_vram={peak_vram_gb:.2f} GB")
    else:
        print(f"    [PERF]  time={elapsed:.1f}s  (no CUDA device)")

    return result.images[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def locale_from_manifest_path(path: str) -> str:
    stem  = Path(path).stem
    parts = stem.split(".")
    return parts[-1] if len(parts) > 1 else "en"


def output_filename(bg: dict, model_key: str, multi_model: bool) -> str:
    """
    Single model  → original filename unchanged  (bg-karnak-inner-sanctuary-v1.png)
    All models    → model suffix inserted         (bg-karnak-inner-sanctuary-v1_flux-schnell.png)
    """
    if not multi_model:
        return bg["output"]
    stem = Path(bg["output"]).stem
    return f"{stem}_{model_key}.png"


# ---------------------------------------------------------------------------
# Per-model generation run
# ---------------------------------------------------------------------------
def run_model(model_key: str, backgrounds: list, out_dir: Path, args,
              multi_model: bool) -> list[dict]:
    """Load, generate all backgrounds, unload. Returns result dicts."""
    cfg = MODELS[model_key]
    print(f"\n{'='*60}")
    print(f"MODEL : {model_key}")
    print(f"ID    : {cfg['model_id']}")
    print(f"Notes : {cfg['notes']}")
    print(f"{'='*60}")

    try:
        pipe, _ = load_pipeline(model_key)
    except Exception as exc:
        print(f"[ERROR] Failed to load {model_key}: {exc}")
        return [
            {
                "asset_id": bg["asset_id"],
                "model": model_key,
                "output": str(out_dir / output_filename(bg, model_key, multi_model)),
                "size_bytes": 0,
                "status": "failed",
                "error": str(exc),
            }
            for bg in backgrounds
        ]

    results = []
    total   = len(backgrounds)

    for idx, bg in enumerate(backgrounds, start=1):
        fname    = output_filename(bg, model_key, multi_model)
        out_path = out_dir / fname
        print(f"\n  [{idx}/{total}] {bg['asset_id']} (mood: {bg['color_mood']}) → {fname}")

        if out_path.exists() and not args.force:
            print(f"    [SKIP] already exists")
            results.append({
                "asset_id": bg["asset_id"],
                "model": model_key,
                "output": str(out_path),
                "size_bytes": out_path.stat().st_size,
                "status": "skipped",
            })
            continue

        try:
            t_img     = time.perf_counter()
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            image     = generate_image(pipe, model_key, bg, args, generator)
            image.save(str(out_path), format="PNG")
            size      = out_path.stat().st_size
            time_s    = time.perf_counter() - t_img
            peak_vram_gb = (
                torch.cuda.max_memory_allocated() / 1024**3
                if torch.cuda.is_available() else 0.0
            )
            print(f"    [OK] {size:,} bytes")
            results.append({
                "asset_id": bg["asset_id"],
                "model": model_key,
                "output": str(out_path),
                "size_bytes": size,
                "status": "success",
                "time_s": round(time_s, 1),
                "peak_vram_gb": round(peak_vram_gb, 2),
            })
        except Exception as exc:
            print(f"    [ERROR] {exc}")
            results.append({
                "asset_id": bg["asset_id"],
                "model": model_key,
                "output": str(out_path),
                "size_bytes": 0,
                "status": "failed",
                "error": str(exc),
            })
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    unload_pipeline(pipe)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # --prompt mode: one-off generation, bypasses manifest/hardcoded list
    if args.prompt:
        if args.output:
            out_path_override = Path(args.output)
            out_dir    = out_path_override.parent
            bg_output  = out_path_override.name
        else:
            out_dir   = Path(args.output_dir) if args.output_dir else PROMPT_OUT_DIR
            bg_output = f"{prompt_slug(args.prompt)}.png"
        out_dir.mkdir(parents=True, exist_ok=True)
        backgrounds = [{
            "asset_id": "oneoff",
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "color_mood": "neutral",
            "output": bg_output,
        }]
    else:
        locale  = locale_from_manifest_path(args.manifest) if args.manifest else "en"
        out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR / locale
        out_dir.mkdir(parents=True, exist_ok=True)
        backgrounds = BACKGROUNDS
        if args.manifest:
            backgrounds = load_from_manifest(args.manifest, args.asset_id)
            if not backgrounds:
                print("[WARN] No matching static backgrounds in manifest. Nothing to do.")
                return

    # Resolve which model keys to run
    if args.model == "all":
        model_keys  = ALL_MODEL_KEYS
        multi_model = True
    elif args.model == "auto":
        model_keys  = ["flux-schnell"]
        multi_model = False
    else:
        model_keys  = [args.model]
        multi_model = False

    if args.model == "all":
        print(f"[MODE] Comparing all {len(model_keys)} models: {', '.join(model_keys)}")
        print(f"[INFO] Outputs will be saved as <name>_<model>.png for side-by-side comparison")

    all_results = []
    for model_key in model_keys:
        results = run_model(model_key, backgrounds, out_dir, args, multi_model)
        all_results.extend(results)

    # Auto fallback: if flux-schnell failed entirely, try sdxl
    if args.model == "auto" and all(r["status"] == "failed" for r in all_results):
        print("\n[WARN] flux-schnell failed for all backgrounds. Falling back to sdxl.")
        all_results = run_model("sdxl", backgrounds, out_dir, args, multi_model=False)

    # Write JSON results
    results_path = out_dir / f"{SCRIPT_NAME}_results.json"
    with open(results_path, "w") as fh:
        json.dump(all_results, fh, indent=2)

    # Summary
    total       = len(all_results)
    ok_count    = sum(1 for r in all_results if r["status"] in ("success", "skipped"))
    total_bytes = sum(r["size_bytes"] for r in all_results)
    print("\n" + "=" * 60)
    print(f"SUMMARY — gen_background_images ({args.model})")
    print("=" * 60)
    for r in all_results:
        label    = "OK" if r["status"] == "success" else r["status"].upper()
        model    = r.get("model", "")
        time_s   = f"  time={r['time_s']:.1f}s" if "time_s" in r else ""
        vram     = f"  peak_vram={r['peak_vram_gb']:.2f} GB" if "peak_vram_gb" in r else ""
        print(f"  [{label}]  [{model}]  {r['output']}  ({r['size_bytes']:,} bytes){time_s}{vram}")
    print(f"\n{ok_count}/{total} completed | {total_bytes:,} bytes total")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
