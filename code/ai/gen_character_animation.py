# =============================================================================
# gen_character_animation.py
# Animate static character portrait PNGs into short video clips using
# AnimateDiff v3 + Stable Diffusion 1.5 for s01e01.
# =============================================================================
#
# requirements.txt (pip install before running):
#   torch>=2.1.0
#   diffusers>=0.28.0
#   transformers>=4.38.0
#   accelerate>=0.27.0
#   imageio[ffmpeg]>=2.34.0
#   Pillow>=10.0.0
#   huggingface_hub>=0.21.0
#   safetensors>=0.4.0
#
# ---------------------------------------------------------------------------
# Hardware Target: NVIDIA RTX 4060 8 GB VRAM
# ---------------------------------------------------------------------------
# Memory-saving techniques:
#   1. enable_model_cpu_offload(): SD1.5 UNet (~3.4 GB FP32) is kept on CPU
#      and moved to GPU only during each UNet forward pass.  The text encoder
#      and VAE are similarly CPU-resident.  Peak active VRAM stays ~3-4 GB.
#
#   2. FP16 precision: halves weight memory vs FP32.
#
#   3. 16 frames maximum: AnimateDiff temporal attention scales with T²,
#      so keeping frames ≤ 16 is essential for 8 GB budget.
#
#   4. 512×512 resolution: SD1.5 native resolution, no upscaling.
#
#   5. torch.cuda.empty_cache() + gc.collect() after each video.
#
# APPROACH — AnimateDiffVideoToVideoPipeline (img2vid):
#   Standard AnimateDiff is text-to-video.  To honour the input_image
#   parameter, this script uses AnimateDiffVideoToVideoPipeline which
#   accepts a "video" (we supply the portrait image repeated as N frames)
#   and a strength parameter (how much the motion model deviates from the
#   input).  strength=0.75 preserves character appearance while adding motion.
#
#   If AnimateDiffVideoToVideoPipeline is unavailable in the installed
#   diffusers version, the script falls back to AnimateDiffPipeline (t2v)
#   using only the text prompt.
#
# NOTE: Both guoyww/animatediff-motion-adapter-v1-5-3 and
#   runwayml/stable-diffusion-v1-5 are free-weight models on HuggingFace.
#   No licence agreement required.
# ---------------------------------------------------------------------------

import argparse
import gc
import json
from pathlib import Path

import torch
from PIL import Image

# ---------------------------------------------------------------------------
# DEFAULTS — fully populated; script runs with no CLI flags.
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "projects" / "the-pharaoh-who-defied-death" / "episodes" / "s01e01" / "assets"
SCRIPT_NAME = "gen_character_animation"

ANIMATIONS = [
    {
        "asset_id": "char-ramesses_ka-v1",
        "input_image": "char-ramesses_ka-v1.png",
        "prompt": (
            "Pharaoh of ancient Egypt in white and gold royal robes, walking slowly and "
            "deliberately forward, jaw set, utterly fearless, photorealistic, cinematic "
            "lighting, smooth motion"
        ),
        "negative_prompt": "blurry, jitter, fast motion, low quality, deformed",
        "motion_type": "walk",
        "duration_sec": 4.0,
        "num_frames": 16,
        "output": "char-ramesses_ka-v1-anim.mp4",
    },
    {
        "asset_id": "char-amunhotep-v1",
        "input_image": "char-amunhotep-v1.png",
        "prompt": (
            "Ancient Egyptian High Priest raising trembling arms in ritual invocation, "
            "lips moving in chant, then reaching one arm out desperately, expression of "
            "terrified dread, photorealistic, smooth motion"
        ),
        "negative_prompt": "blurry, jitter, fast motion, low quality, deformed",
        "motion_type": "gesture",
        "duration_sec": 4.0,
        "num_frames": 16,
        "output": "char-amunhotep-v1-anim.mp4",
    },
    {
        "asset_id": "char-neferet-v1",
        "input_image": "char-neferet-v1.png",
        "prompt": (
            "Young Egyptian woman sitting frozen over an open papyrus scroll, staring in "
            "dawning dread, lamp light flickering across her face, subtle breathing motion, "
            "photorealistic"
        ),
        "negative_prompt": "blurry, jitter, fast motion, low quality, deformed",
        "motion_type": "idle",
        "duration_sec": 3.0,
        "num_frames": 16,
        "output": "char-neferet-v1-anim.mp4",
    },
    {
        "asset_id": "char-khamun-v1",
        "input_image": "char-khamun-v1.png",
        "prompt": (
            "Ancient Egyptian military general standing rigid at the edge of a deep shaft, "
            "arms clasped behind back, jaw tight, staring downward into darkness, subtle "
            "breathing, photorealistic"
        ),
        "negative_prompt": "blurry, jitter, fast motion, low quality, deformed",
        "motion_type": "idle",
        "duration_sec": 3.0,
        "num_frames": 16,
        "output": "char-khamun-v1-anim.mp4",
    },
]

MOTION_ADAPTER_ID = "guoyww/animatediff-motion-adapter-v1-5-3"
SD15_MODEL_ID = "runwayml/stable-diffusion-v1-5"

# img2vid strength: 1.0 = pure text, 0.0 = copy input unchanged
# 0.75 preserves ~25% of input appearance while adding motion
IMG2VID_STRENGTH = 0.75
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 25
FPS = 8
FRAME_SIZE = 512  # SD1.5 native


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Animate character portraits using AnimateDiff for s01e01."
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory containing input portrait PNGs.")
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--strength", type=float, default=IMG2VID_STRENGTH,
                        help="img2vid strength (0.0-1.0); higher = more motion.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to AssetManifest JSON. When given, overrides the hardcoded ANIMATIONS list.",
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
    """Load animation job list from AssetManifest JSON (character_packs where motion != null)."""
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    anims = []
    for pack in manifest.get("character_packs", []):
        if not pack.get("motion"):
            continue  # skip characters with no motion (e.g. prisoner)
        if asset_id_filter and pack["asset_id"] != asset_id_filter:
            continue
        aid = pack["asset_id"]
        motion = pack["motion"]
        duration = float(motion.get("duration_sec", 3.0))
        num_frames = min(16, max(1, int(duration * 8) + 1))
        motion_desc = motion.get("description", "")
        prompt = f"{pack['ai_prompt']}, {motion_desc}, photorealistic, smooth motion"
        anims.append({
            "asset_id":       aid,
            "input_image":    f"{aid}.png",
            "prompt":         prompt,
            "negative_prompt": "blurry, jitter, fast motion, low quality, deformed",
            "motion_type":    motion.get("type", "idle"),
            "duration_sec":   duration,
            "num_frames":     num_frames,
            "output":         f"{aid}-anim.mp4",
        })
    return anims


# ---------------------------------------------------------------------------
# Model loader — tries V2V first, falls back to T2V
# ---------------------------------------------------------------------------
def load_pipeline():
    """
    Try to load AnimateDiffVideoToVideoPipeline for img2vid.
    Fall back to AnimateDiffPipeline (text-to-video) if unavailable.
    """
    from diffusers import MotionAdapter, DDIMScheduler
    import diffusers

    print("[MODEL] Loading AnimateDiff motion adapter v1-5-3...")
    adapter = MotionAdapter.from_pretrained(
        MOTION_ADAPTER_ID,
        torch_dtype=torch.float16,
    )

    # Attempt img2vid pipeline first
    try:
        from diffusers import AnimateDiffVideoToVideoPipeline
        print("[MODEL] Loading AnimateDiffVideoToVideoPipeline + SD1.5...")
        pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(
            SD15_MODEL_ID,
            motion_adapter=adapter,
            torch_dtype=torch.float16,
        )
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            beta_schedule="linear",
            clip_sample=False,
            timestep_spacing="linspace",
            steps_offset=1,
        )
        pipe.enable_model_cpu_offload()
        print("[MODEL] Using AnimateDiffVideoToVideoPipeline (img2vid).")
        return pipe, "v2v"
    except (ImportError, AttributeError) as exc:
        print(f"[WARN] VideoToVideo pipeline unavailable ({exc}). Falling back to T2V.")

    # Fallback: text-to-video
    from diffusers import AnimateDiffPipeline
    print("[MODEL] Loading AnimateDiffPipeline + SD1.5 (text-to-video)...")
    pipe = AnimateDiffPipeline.from_pretrained(
        SD15_MODEL_ID,
        motion_adapter=adapter,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        beta_schedule="linear",
        clip_sample=False,
        timestep_spacing="linspace",
        steps_offset=1,
    )
    pipe.enable_model_cpu_offload()
    print("[MODEL] Using AnimateDiffPipeline (text-to-video fallback).")
    return pipe, "t2v"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_and_resize_image(image_path: Path, size: int) -> Image.Image:
    """Load a portrait PNG and resize/crop to square for SD1.5."""
    img = Image.open(str(image_path)).convert("RGB")
    # Centre-crop to square then resize to model native size
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size), Image.LANCZOS)
    return img


def make_init_video(frame: Image.Image, num_frames: int) -> list:
    """Repeat a single image N times to create a static init video."""
    return [frame] * num_frames


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def generate_animation(pipe, mode: str, anim: dict, input_dir: Path, args) -> list:
    """Run inference and return list of PIL Image frames."""
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    if mode == "v2v":
        # Load input portrait as init video
        img_path = input_dir / anim["input_image"]
        if img_path.exists():
            frame = load_and_resize_image(img_path, FRAME_SIZE)
            init_video = make_init_video(frame, anim["num_frames"])
        else:
            print(f"  [WARN] Input image not found: {img_path}. Using noise init.")
            init_video = [Image.new("RGB", (FRAME_SIZE, FRAME_SIZE), (128, 128, 128))] * anim["num_frames"]

        result = pipe(
            video=init_video,
            prompt=anim["prompt"],
            negative_prompt=anim["negative_prompt"],
            num_frames=anim["num_frames"],
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=args.steps,
            strength=args.strength,
            generator=generator,
        )
    else:  # t2v fallback
        result = pipe(
            prompt=anim["prompt"],
            negative_prompt=anim["negative_prompt"],
            num_frames=anim["num_frames"],
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=args.steps,
            generator=generator,
        )

    # diffusers AnimateDiff returns frames as result.frames[0]
    return result.frames[0]


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
    input_dir = Path(args.input_dir) if args.input_dir else OUTPUT_DIR / locale
    out_dir.mkdir(parents=True, exist_ok=True)

    animations = ANIMATIONS
    if args.manifest:
        animations = load_from_manifest(args.manifest, args.asset_id)
        if not animations:
            print("[WARN] No animated character_packs in manifest (all have motion:null?). Nothing to do.")
            return

    pipe, mode = load_pipeline()
    print(f"[MODE] {mode.upper()} pipeline active")

    results = []
    total = len(animations)

    for idx, anim in enumerate(animations, start=1):
        out_path = out_dir / anim["output"]
        print(f"\n[{idx}/{total}] Animating {anim['asset_id']} ({anim['motion_type']})...")

        if out_path.exists():
            print(f"  [SKIP] {anim['output']} already exists")
            results.append({
                "asset_id": anim["asset_id"],
                "output": str(out_path),
                "size_bytes": out_path.stat().st_size,
                "status": "skipped",
            })
            continue

        try:
            from diffusers.utils import export_to_video

            frames = generate_animation(pipe, mode, anim, input_dir, args)
            export_to_video(frames, str(out_path), fps=args.fps)
            size = out_path.stat().st_size
            print(f"  [OK] {out_path}  ({len(frames)} frames, {size:,} bytes)")
            results.append({
                "asset_id": anim["asset_id"],
                "output": str(out_path),
                "size_bytes": size,
                "num_frames": len(frames),
                "status": "success",
            })
        except Exception as exc:
            print(f"  [ERROR] {anim['asset_id']}: {exc}")
            results.append({
                "asset_id": anim["asset_id"],
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
    print("SUMMARY — gen_character_animation")
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
