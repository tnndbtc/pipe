# =============================================================================
# gen_character_images.py
# Generate photorealistic character portrait images for s01e01.
# STATUS: VALIDATED
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
#                 model suffix: char-amunhotep-v1_flux-schnell.png etc.
#                 Use this to compare quality across models.
#
# Memory-saving techniques:
#   FLUX: 4-bit bitsandbytes quantisation collapses transformer from ~24 GB
#   to ~6 GB. enable_model_cpu_offload() streams text encoder + VAE to CPU.
#   vae_slicing() splits the VAE decode into row-by-row chunks.
#   SDXL: FP16 UNet ~5 GB. CPU offload keeps peak VRAM ~4-5 GB.
#   Between images: torch.cuda.empty_cache() + gc.collect() after every gen.
#   Resolution 512x768 (below SDXL native 1024 px) reduces activation VRAM.
# ---------------------------------------------------------------------------

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# DEFAULTS -- fully populated; script runs with no CLI flags.
# ---------------------------------------------------------------------------
OUTPUT_DIR  = Path(__file__).resolve().parent.parent.parent / "projects" / "the-pharaoh-who-defied-death" / "episodes" / "s01e01" / "assets"
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
WIDTH  = 512
HEIGHT = 768

# ---------------------------------------------------------------------------
# Model registry
# Each entry: model_id, steps, guidance, loader_family ("flux" | "sdxl")
# ---------------------------------------------------------------------------
MODELS = {
    "flux-schnell": {
        "model_id": "black-forest-labs/FLUX.1-schnell",
        "steps":    4,
        "guidance": 0.0,   # guidance-distilled -- CFG unused
        "family":   "flux",
        "notes":    "Fast distilled FLUX, 4-bit quant, ~6 GB VRAM",
    },
    "flux-dev": {
        "model_id":  "black-forest-labs/FLUX.1-dev",
        "steps":     28,
        "guidance":  3.5,
        "family":    "flux",
        "outdated":  True,
        "successor": "flux-schnell",
        "notes":     "[OUTDATED] Non-commercial licence only. Use flux-schnell (Apache 2.0) for commercial projects.",
    },
    "sdxl": {
        "model_id":  "stabilityai/stable-diffusion-xl-base-1.0",
        "steps":     25,
        "guidance":  7.5,
        "family":    "sdxl",
        "outdated":  True,
        "successor": "flux-schnell or flux-dev",
        "notes":     "[OUTDATED] SDXL 1.0 FP16 + CPU offload, ~5 GB VRAM. Superseded by FLUX for character portraits.",
    },
    "sdxl-turbo": {
        "model_id":  "stabilityai/sdxl-turbo",
        "steps":     4,
        "guidance":  0.0,   # guidance-distilled
        "family":    "sdxl",
        "outdated":  True,
        "successor": "flux-schnell",
        "notes":     "[OUTDATED] Distilled SDXL, 4 steps, FP16 + CPU offload, ~5 GB VRAM. Superseded by flux-schnell.",
    },
}

ALL_MODEL_KEYS = list(MODELS.keys())   # ["flux-schnell", "flux-dev", "sdxl", "sdxl-turbo"]


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
        description="Generate character portrait images.",
        epilog=(
            "Available --model values:\n\n"
            + model_help +
            "\n\n  auto           Tries flux-schnell, falls back to sdxl on failure."
            "\n  all            Runs every model above in sequence for comparison."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--width",  type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output files instead of skipping them.")
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
        help="Path to AssetManifest JSON. When given, overrides the hardcoded CHARACTERS list.",
    )
    parser.add_argument(
        "--asset-id", type=str, default=None, dest="asset_id",
        help="Process only this asset_id (requires --manifest).",
    )
    parser.add_argument(
        "--no-matte", action="store_true", dest="no_matte",
        help="Skip automatic background removal after generation. "
             "By default, rembg runs after generation to produce {asset_id}-rgba.png.",
    )
    parser.add_argument(
        "--bg-hint", type=str, default=None, dest="bg_hint",
        help=(
            "Optional prompt suffix appended to every character prompt to guide "
            "background style -- makes rembg removal more reliable. "
            "The original pipeline prompt is preserved; this is appended. "
            "Example: --bg-hint \"plain white studio background, solid backdrop, no scenery\""
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------
def load_flux_pipeline(model_key: str):
    """Load a FLUX model with 4-bit bitsandbytes quantisation.

    diffusers >=0.32 no longer accepts quantization_config on the pipeline
    directly.  The transformer must be quantized separately, then injected.
    """
    from diffusers import FluxPipeline, FluxTransformer2DModel
    from transformers import BitsAndBytesConfig

    cfg     = MODELS[model_key]
    print(f"[MODEL] Loading {model_key} ({cfg['model_id']}) -- 4-bit quant (transformer)...")
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
    pipe.enable_vae_slicing()
    return pipe


def load_sdxl_pipeline(model_key: str):
    """Load an SDXL-family model in FP16 with CPU offload."""
    from diffusers import StableDiffusionXLPipeline

    cfg = MODELS[model_key]
    print(f"[MODEL] Loading {model_key} ({cfg['model_id']}) -- FP16 + CPU offload...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        cfg["model_id"],
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
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
def generate_image(pipe, model_key: str, char: dict, args, generator):
    """Run one inference call and return a PIL Image."""
    cfg = MODELS[model_key]
    if cfg["family"] == "flux":
        result = pipe(
            prompt=char["prompt"],
            width=args.width,
            height=args.height,
            num_inference_steps=cfg["steps"],
            guidance_scale=cfg["guidance"],
            generator=generator,
        )
    else:  # sdxl family
        result = pipe(
            prompt=char["prompt"],
            negative_prompt=char.get("negative_prompt", ""),
            width=args.width,
            height=args.height,
            num_inference_steps=cfg["steps"],
            guidance_scale=cfg["guidance"],
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
# Helpers
# ---------------------------------------------------------------------------
def locale_from_manifest_path(path: str) -> str:
    stem  = Path(path).stem
    parts = stem.split(".")
    return parts[-1] if len(parts) > 1 else "en"


def output_filename(char: dict, model_key: str, multi_model: bool) -> str:
    """
    Single model  -> original filename unchanged  (char-amunhotep-v1.png)
    All models    -> model suffix inserted         (char-amunhotep-v1_flux-schnell.png)
    """
    if not multi_model:
        return char["output"]
    stem = Path(char["output"]).stem
    return f"{stem}_{model_key}.png"


# ---------------------------------------------------------------------------
# Per-model generation run
# ---------------------------------------------------------------------------
def run_model(model_key: str, characters: list, out_dir: Path, args,
              multi_model: bool) -> list[dict]:
    """Load, generate all characters, unload. Returns result dicts."""
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
                "asset_id": c["asset_id"],
                "model": model_key,
                "output": str(out_dir / output_filename(c, model_key, multi_model)),
                "size_bytes": 0,
                "status": "failed",
                "error": str(exc),
            }
            for c in characters
        ]

    results = []
    total   = len(characters)

    for idx, char in enumerate(characters, start=1):
        fname    = output_filename(char, model_key, multi_model)
        out_path = out_dir / fname
        print(f"\n  [{idx}/{total}] {char['asset_id']} -> {fname}")

        if out_path.exists() and not args.force:
            print(f"    [SKIP] already exists")
            results.append({
                "asset_id": char["asset_id"],
                "model": model_key,
                "output": str(out_path),
                "size_bytes": out_path.stat().st_size,
                "status": "skipped",
            })
            continue

        try:
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            image     = generate_image(pipe, model_key, char, args, generator)
            image.save(str(out_path), format="PNG")
            size      = out_path.stat().st_size
            print(f"    [OK] {size:,} bytes")
            results.append({
                "asset_id": char["asset_id"],
                "model": model_key,
                "output": str(out_path),
                "size_bytes": size,
                "status": "success",
            })
        except Exception as exc:
            print(f"    [ERROR] {exc}")
            results.append({
                "asset_id": char["asset_id"],
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

    # Reject outdated models early with a clear message.
    if args.model not in ("auto", "all"):
        cfg = MODELS[args.model]
        if cfg.get("outdated"):
            print(f"[ERROR] '{args.model}' is outdated and no longer supported for character generation.")
            print(f"        Reason : {cfg['notes']}")
            print(f"        Use    : --model {cfg['successor']}")
            sys.exit(1)

    locale  = locale_from_manifest_path(args.manifest) if args.manifest else "en"
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR / locale
    out_dir.mkdir(parents=True, exist_ok=True)

    characters = CHARACTERS
    if args.manifest:
        characters = load_from_manifest(args.manifest, args.asset_id)
        if not characters:
            print("[WARN] No matching character_packs in manifest. Nothing to do.")
            return

    # Append background hint to each prompt if provided
    if args.bg_hint:
        for char in characters:
            char["prompt"] = char["prompt"] + ", " + args.bg_hint
        print(f"[BG-HINT] Appended to all prompts: \"{args.bg_hint}\"")

    # Resolve which model keys to run
    if args.model == "all":
        model_keys  = ALL_MODEL_KEYS
        multi_model = True
    elif args.model == "auto":
        model_keys  = ["flux-schnell"]   # run_model handles fallback on load failure
        multi_model = False
    else:
        model_keys  = [args.model]
        multi_model = False

    if args.model == "all":
        print(f"[MODE] Comparing all {len(model_keys)} models: {', '.join(model_keys)}")
        print(f"[INFO] Outputs will be saved as <name>_<model>.png for side-by-side comparison")

    all_results = []
    for model_key in model_keys:
        results = run_model(model_key, characters, out_dir, args, multi_model)
        all_results.extend(results)

    # Auto fallback for 'auto' mode: if flux-schnell failed entirely, try sdxl
    if args.model == "auto" and all(r["status"] == "failed" for r in all_results):
        print("\n[WARN] flux-schnell failed for all characters. Falling back to sdxl.")
        all_results = run_model("sdxl", characters, out_dir, args, multi_model=False)

    # Write JSON results
    results_path = out_dir / f"{SCRIPT_NAME}_results.json"
    with open(results_path, "w") as fh:
        json.dump(all_results, fh, indent=2)

    # Summary
    total       = len(all_results)
    ok_count    = sum(1 for r in all_results if r["status"] in ("success", "skipped"))
    total_bytes = sum(r["size_bytes"] for r in all_results)
    print("\n" + "=" * 60)
    print(f"SUMMARY -- gen_character_images ({args.model})")
    print("=" * 60)
    for r in all_results:
        label = "OK" if r["status"] == "success" else r["status"].upper()
        model = r.get("model", "")
        print(f"  [{label}]  [{model}]  {r['output']}  ({r['size_bytes']:,} bytes)")
    print(f"\n{ok_count}/{total} completed | {total_bytes:,} bytes total")
    print(f"Results: {results_path}")

    # ---------------------------------------------------------------------------
    # Auto-matte post-processing  (RMBG-1.4 background removal)
    # Produces {asset_id}-rgba.png alongside each generated image.
    # Skip with --no-matte.
    # ---------------------------------------------------------------------------
    if not args.no_matte:
        to_matte = [
            r for r in all_results
            if r["status"] in ("success", "skipped") and Path(r["output"]).is_file()
        ]
        if not to_matte:
            print("\n[MATTE] No images available to process.")
        else:
            from gen_character_mattes import load_rmbg, remove_background  # noqa: PLC0415

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"\n[MATTE] Running RMBG-1.4 background removal on "
                  f"{len(to_matte)} image(s)...")
            matte_model = load_rmbg(device)

            for idx, r in enumerate(to_matte, start=1):
                input_path  = Path(r["output"])
                output_path = input_path.with_name(input_path.stem + "-rgba.png")
                print(f"\n  [{idx}/{len(to_matte)}] {input_path.name} -> {output_path.name}")

                if output_path.exists() and not args.force:
                    print(f"    [SKIP] already exists")
                    continue

                try:
                    size = remove_background(
                        matte_model, input_path, output_path, device, threshold=0.5
                    )
                    print(f"    [OK] {size:,} bytes  (RGBA, transparent background)")
                except Exception as exc:
                    print(f"    [ERROR] {exc}")
                finally:
                    torch.cuda.empty_cache()
                    gc.collect()

            del matte_model
            torch.cuda.empty_cache()
            gc.collect()
            print(f"\n[MATTE] Done. RGBA images saved alongside originals in {out_dir}")


if __name__ == "__main__":
    main()
