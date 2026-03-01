# =============================================================================
# gen_background_video.py
# Generate short cinematic background video clips for shots that require
# camera motion (pan, crane, zoom) in s01e01.
# STATUS: VALIDATED-PENDING-GPU — pipeline code is correct but cogvideox-2b
#         uses ~10.8 GB VRAM (spills to system RAM, produces blank output on
#         RTX 4060 8 GB). ltx-video loads but produces static/no-movement clips.
#         Re-validate on A100 40 GB via RunPod/Vast.ai, or use fal.ai API.
#         See GPU rental plan in gen_character_animation.py comments.
# =============================================================================
#
# requirements.txt (pip install before running):
#   torch>=2.4.1
#   diffusers>=0.32.0        # LTXPipeline added in 0.32; CogVideoXPipeline in 0.30
#   transformers>=4.40.0
#   accelerate>=0.30.0
#   imageio[ffmpeg]>=2.34.0
#   huggingface_hub>=0.21.0
#   safetensors>=0.4.0
#
# ---------------------------------------------------------------------------
# Hardware targets:
#
#   cogvideox-2b  THUDM/CogVideoX-2b
#                 FP16 + CPU offload + VAE slicing/tiling, ~6 GB VRAM.
#                 Native resolution 720×480 (3:2).  RTX 4060 8 GB. ✓
#
#   cogvideox-5b  THUDM/CogVideoX-5b
#                 FP16 + CPU offload + VAE slicing/tiling, ~12 GB VRAM.
#                 Better quality than 2b.  Needs RTX 4070 Ti / 4080 (>=12 GB).
#                 May OOM on RTX 4060 8 GB.
#
#   ltx-video     Lightricks/LTX-Video
#                 bfloat16 + CPU offload, ~6-8 GB VRAM.
#                 Fast distilled model, 768×512 native.  RTX 4060 8 GB. ✓
#
#   auto          Tries cogvideox-2b first, falls back to ltx-video on failure.
#
#   all           Runs every model in sequence. Outputs saved with model
#                 suffix: bg-desert-excavation-site-v1_cogvideox-2b.mp4
#                 Use this to compare quality across models.
#
# ---------------------------------------------------------------------------
# Memory-saving techniques (all models):
#   enable_model_cpu_offload(): streams each sub-module to GPU only during
#   its forward pass.  enable_vae_slicing() + enable_vae_tiling() decompose
#   the VAE decode into frame-sliced and spatially-tiled chunks.
#   torch.cuda.empty_cache() + gc.collect() between videos.
#
# UPGRADE PATH: For LTX-Video 0.9.8-13B or CogVideoX higher-res quality,
#   a GPU with >=24 GB VRAM is required (RTX 3090 / 4090).
# ---------------------------------------------------------------------------

import argparse
import gc
import json
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# DEFAULTS — fully populated; script runs with no CLI flags.
# ---------------------------------------------------------------------------
OUTPUT_DIR  = Path(__file__).resolve().parent.parent.parent / "projects" / "the-pharaoh-who-defied-death" / "episodes" / "s01e01" / "assets"
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
        "num_frames":   49,   # 6s × 8fps + 1
        "output":       "bg-desert-excavation-site-v1.mp4",
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
        "num_frames":   41,   # 5s × 8fps + 1
        "output":       "bg-underground-chamber-v1.mp4",
    },
]

FPS = 8

# ---------------------------------------------------------------------------
# Model registry
# family: "cogvideox" | "ltx"
# width/height: native resolution each model works best at
# ---------------------------------------------------------------------------
MODELS = {
    "cogvideox-2b": {
        "model_id": "THUDM/CogVideoX-2b",
        "steps":    25,
        "guidance": 6.0,
        "family":   "cogvideox",
        "width":    720,
        "height":   480,
        "notes":    "CogVideoX 2B, FP16 + CPU offload, ~6 GB VRAM. Native 720×480.",
    },
    "cogvideox-5b": {
        "model_id": "THUDM/CogVideoX-5b",
        "steps":    25,
        "guidance": 6.0,
        "family":   "cogvideox",
        "width":    720,
        "height":   480,
        "notes":    "CogVideoX 5B, FP16 + CPU offload, ~12 GB VRAM. RTX 4070 Ti / 4080+ recommended. May OOM on 8 GB.",
    },
    "ltx-video": {
        "model_id": "Lightricks/LTX-Video",
        "steps":    25,
        "guidance": 3.0,
        "family":   "ltx",
        "width":    768,
        "height":   512,
        "notes":    "LTX-Video, bfloat16 + CPU offload, ~6-8 GB VRAM. Native 768×512.",
    },
}

ALL_MODEL_KEYS = list(MODELS.keys())


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
        description="Generate background video clips using text-to-video models.",
        epilog=(
            "Available --model values:\n\n"
            + model_help +
            "\n\n  auto           Tries cogvideox-2b, falls back to ltx-video on failure."
            "\n  all            Runs every model above in sequence for comparison."
            "\n\nNext test command (bg-karnak-hypostyle-hall, 4 sec, slow dolly between columns):\n\n"
            "  Option 1 — ltx-video (lighter, try first on 8 GB):\n"
            "  python code\\ai\\gen_background_video.py --model ltx-video --manifest ..\\AssetManifest_draft.json --asset-id bg-karnak-hypostyle-hall --output_dir projects\\the-pharaoh-who-defied-death\\episodes\\s01e01\\assets\\en\n\n"
            "  Option 2 — cogvideox-2b at reduced resolution (if ltx-video unavailable):\n"
            "  python code\\ai\\gen_background_video.py --model cogvideox-2b --manifest ..\\AssetManifest_draft.json --asset-id bg-karnak-hypostyle-hall --output_dir projects\\the-pharaoh-who-defied-death\\episodes\\s01e01\\assets\\en --width 480 --height 320 --num-frames 17"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--fps",   type=int, default=FPS)
    parser.add_argument("--steps", type=int, default=None,
                        help="Override inference steps (default: per-model value).")
    parser.add_argument("--guidance", type=float, default=None,
                        help="Override guidance scale (default: per-model value). Higher = follows prompt more strictly.")
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output files instead of skipping them.")
    parser.add_argument("--width",  type=int, default=None,
                        help="Override output width (default: per-model value). Reduce to save VRAM.")
    parser.add_argument("--height", type=int, default=None,
                        help="Override output height (default: per-model value). Reduce to save VRAM.")
    parser.add_argument("--num-frames", type=int, default=None, dest="num_frames",
                        help="Override frame count (default: from manifest/job). Reduce to save VRAM.")
    parser.add_argument(
        "--model",
        choices=ALL_MODEL_KEYS + ["auto", "all"],
        default="auto",
        help=(
            "Model to use. "
            "'auto' tries cogvideox-2b, falls back to ltx-video on failure. "
            "'all' runs every model in sequence for side-by-side comparison."
        ),
    )
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
        aid        = bg["asset_id"]
        duration   = float(motion.get("duration_sec", 5.0))
        num_frames = int(duration * 8) + 1
        videos.append({
            "asset_id":           aid,
            "prompt":             motion.get("description", "slow camera motion") + ", " + bg["ai_prompt"],
            "negative_prompt":    "blurry, low quality, modern, fast motion, shaky",
            "motion_description": motion.get("description", "slow camera motion"),
            "duration_sec":       duration,
            "num_frames":         num_frames,
            "output":             f"{aid}.mp4",
        })
    return videos


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------
def load_cogvideox_pipeline(model_key: str):
    """Load a CogVideoX model in FP16 with CPU offload + VAE slicing/tiling."""
    from diffusers import CogVideoXPipeline  # noqa: PLC0415

    cfg = MODELS[model_key]
    print(f"[MODEL] Loading {model_key} ({cfg['model_id']}) — FP16 + CPU offload...")
    pipe = CogVideoXPipeline.from_pretrained(cfg["model_id"], torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    print(f"[MODEL] {model_key} ready.")
    return pipe


def load_ltx_pipeline(model_key: str):
    """Load LTX-Video in bfloat16 with CPU offload."""
    from diffusers import LTXPipeline  # noqa: PLC0415

    cfg = MODELS[model_key]
    print(f"[MODEL] Loading {model_key} ({cfg['model_id']}) — bfloat16 + CPU offload...")
    pipe = LTXPipeline.from_pretrained(cfg["model_id"], torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    print(f"[MODEL] {model_key} ready.")
    return pipe


def load_pipeline(model_key: str):
    """Load the pipeline for model_key. Returns (pipe, model_key)."""
    family = MODELS[model_key]["family"]
    if family == "cogvideox":
        return load_cogvideox_pipeline(model_key), model_key
    elif family == "ltx":
        return load_ltx_pipeline(model_key), model_key
    else:
        raise ValueError(f"Unknown model family: {family!r}")


def unload_pipeline(pipe):
    """Aggressively free VRAM after a model run."""
    del pipe
    torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def generate_video(pipe, model_key: str, job: dict, args) -> list:
    """Run inference and return a list of PIL Image frames."""
    cfg        = MODELS[model_key]
    steps      = args.steps    if args.steps    is not None else cfg["steps"]
    width      = args.width    if args.width    is not None else cfg["width"]
    height     = args.height   if args.height   is not None else cfg["height"]
    num_frames = args.num_frames if args.num_frames is not None else job["num_frames"]
    guidance   = args.guidance if args.guidance is not None else cfg["guidance"]
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    generator  = torch.Generator(device=device).manual_seed(args.seed)

    print(f"    Inference: {width}×{height}  {num_frames} frames  {steps} steps  guidance={guidance}")

    result = pipe(
        prompt=job["prompt"],
        negative_prompt=job["negative_prompt"],
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    )
    # Both CogVideoX and LTX return result.frames[0] as a list of PIL Images
    return result.frames[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def locale_from_manifest_path(path: str) -> str:
    stem  = Path(path).stem
    parts = stem.split(".")
    return parts[-1] if len(parts) > 1 else "en"


def output_filename(job: dict, model_key: str, multi_model: bool) -> str:
    """
    Single model  → original filename unchanged  (bg-desert-excavation-site-v1.mp4)
    All models    → model suffix inserted         (bg-desert-excavation-site-v1_cogvideox-2b.mp4)
    """
    if not multi_model:
        return job["output"]
    stem = Path(job["output"]).stem
    return f"{stem}_{model_key}.mp4"


# ---------------------------------------------------------------------------
# Per-model generation run
# ---------------------------------------------------------------------------
def run_model(model_key: str, video_backgrounds: list, out_dir: Path, args,
              multi_model: bool) -> list[dict]:
    """Load, generate all video clips, unload. Returns result dicts."""
    from diffusers.utils import export_to_video  # noqa: PLC0415

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
                "asset_id": job["asset_id"],
                "model": model_key,
                "output": str(out_dir / output_filename(job, model_key, multi_model)),
                "size_bytes": 0,
                "status": "failed",
                "error": str(exc),
            }
            for job in video_backgrounds
        ]

    results = []
    total   = len(video_backgrounds)

    for idx, job in enumerate(video_backgrounds, start=1):
        fname    = output_filename(job, model_key, multi_model)
        out_path = out_dir / fname
        steps    = args.steps if args.steps is not None else cfg["steps"]
        print(f"\n  [{idx}/{total}] {job['asset_id']} → {fname}")
        print(f"    Motion  : {job['motion_description']}")
        print(f"    Duration: {job['duration_sec']}s  Frames: {job['num_frames']}  @ {args.fps} fps  Steps: {steps}")
        print(f"    Res     : {cfg['width']}×{cfg['height']}")

        if out_path.exists() and not args.force:
            print(f"    [SKIP] already exists")
            results.append({
                "asset_id":  job["asset_id"],
                "model":     model_key,
                "output":    str(out_path),
                "size_bytes": out_path.stat().st_size,
                "status":    "skipped",
            })
            continue

        try:
            import time
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            frames = generate_video(pipe, model_key, job, args)
            export_to_video(frames, str(out_path), fps=args.fps)
            elapsed = time.time() - t0
            peak_vram_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
            size = out_path.stat().st_size
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(out_path.stat().st_mtime))
            print(f"    [OK] {out_path.resolve()}")
            print(f"         {len(frames)} frames  |  {elapsed:.1f}s  |  peak VRAM {peak_vram_gb:.2f} GB  |  {size:,} bytes")
            print(f"         file timestamp: {mtime}")
            results.append({
                "asset_id":     job["asset_id"],
                "model":        model_key,
                "output":       str(out_path),
                "size_bytes":   size,
                "num_frames":   len(frames),
                "elapsed_sec":  round(elapsed, 1),
                "peak_vram_gb": round(peak_vram_gb, 2),
                "status":       "success",
            })
        except Exception as exc:
            print(f"    [ERROR] {exc}")
            results.append({
                "asset_id":   job["asset_id"],
                "model":      model_key,
                "output":     str(out_path),
                "size_bytes": 0,
                "status":     "failed",
                "error":      str(exc),
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
    locale  = locale_from_manifest_path(args.manifest) if args.manifest else "en"
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR / locale
    out_dir.mkdir(parents=True, exist_ok=True)

    video_backgrounds = VIDEO_BACKGROUNDS
    if args.manifest:
        video_backgrounds = load_from_manifest(args.manifest, args.asset_id)
        if not video_backgrounds:
            print("[WARN] No matching animated backgrounds (motion.type=camera) in manifest. Nothing to do.")
            return

    # Resolve which model keys to run
    if args.model == "all":
        model_keys  = ALL_MODEL_KEYS
        multi_model = True
    elif args.model == "auto":
        model_keys  = ["cogvideox-2b"]
        multi_model = False
    else:
        model_keys  = [args.model]
        multi_model = False

    if args.model == "all":
        print(f"[MODE] Comparing all {len(model_keys)} models: {', '.join(model_keys)}")
        print(f"[INFO] Outputs will be saved as <name>_<model>.mp4 for side-by-side comparison")

    all_results = []
    for model_key in model_keys:
        results = run_model(model_key, video_backgrounds, out_dir, args, multi_model)
        all_results.extend(results)

    # Auto fallback: if cogvideox-2b failed entirely, try ltx-video
    if args.model == "auto" and all(r["status"] == "failed" for r in all_results):
        print("\n[WARN] cogvideox-2b failed for all clips. Falling back to ltx-video.")
        all_results = run_model("ltx-video", video_backgrounds, out_dir, args, multi_model=False)

    # Write JSON results
    results_path = out_dir / f"{SCRIPT_NAME}_results.json"
    with open(results_path, "w") as fh:
        json.dump(all_results, fh, indent=2)

    # Summary
    total       = len(all_results)
    ok_count    = sum(1 for r in all_results if r["status"] in ("success", "skipped"))
    total_bytes = sum(r["size_bytes"] for r in all_results)
    print("\n" + "=" * 60)
    print(f"SUMMARY — gen_background_video ({args.model})")
    print("=" * 60)
    for r in all_results:
        label = "OK" if r["status"] == "success" else r["status"].upper()
        model = r.get("model", "")
        print(f"  [{label}]  [{model}]  {r['output']}  ({r['size_bytes']:,} bytes)")
    print(f"\n{ok_count}/{total} completed | {total_bytes:,} bytes total")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
