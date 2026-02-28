# =============================================================================
# gen_character_animation.py
# Animate static character portrait PNGs into short video clips.
# Supports two models selectable via --model:
#
#   svd           Stable Video Diffusion XT (default, recommended)
#                 stabilityai/stable-video-diffusion-img2vid-xt
#                 True img2vid — smooth, no flickering. ~7-8 GB VRAM.
#
#   animatediff   AnimateDiff v1-5-3 + SD1.5
#                 guoyww/animatediff-motion-adapter-v1-5-3
#                 img2vid via AnimateDiffVideoToVideoPipeline. ~3-4 GB VRAM.
#                 Prone to flickering at high --strength values.
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
# Memory-saving techniques (both models):
#   1. enable_model_cpu_offload(): model weights live on CPU, moved to GPU
#      only during each forward pass.
#   2. FP16 precision: halves weight memory vs FP32.
#   3. torch.cuda.empty_cache() + gc.collect() after each video.
#
# SVD-specific:
#   4. decode_chunk_size=4: VAE decodes 4 frames at a time instead of all 25,
#      keeping peak decode VRAM within 8 GB.
#   5. 512×512 resolution: SVD trained at 1024×576 but runs fine at 512×512
#      and stays safely within 8 GB.
#
# AnimateDiff-specific:
#   6. 16 frames maximum: temporal attention scales with T², so ≤16 is
#      essential for 8 GB budget.
# ---------------------------------------------------------------------------
#
# =============================================================================
# MODEL REFERENCE — Image-to-Video with Text Prompt Support
# =============================================================================
# The manifest file contains detailed action descriptions (e.g. "high priest
# scrubbing forearm in horror, then binding arm with linen cloth"). Neither
# SVD nor AnimateDiff can faithfully execute these directives. The models below
# CAN follow image + text-prompt action descriptions. Listed for future upgrade.
#
# KEY:  ✓ = supports image + text prompt → video
#       VRAM figures are for fp16 inference without quantization unless noted.
# -----------------------------------------------------------------------------
#
# --- LOCAL MODELS (self-hosted) ----------------------------------------------
#
# CogVideoX-5B-I2V
#   Repo    : THUDM/CogVideoX-5b-I2V  (HuggingFace)
#   Input   : image + text prompt → video
#   Frames  : up to 49 frames (~6 sec at 8 fps)
#   VRAM    : ~24 GB fp16  |  ~16 GB with int8 quantization
#   GPU     : RTX 4090 (24 GB) — minimum for fp16
#             RTX 3090 (24 GB) — works with int8
#             A100 40 GB       — comfortable
#   Notes   : Best local option for following specific action descriptions.
#             Diffusers pipeline: CogVideoXImageToVideoPipeline
#             pip install diffusers>=0.30.0 transformers accelerate
#
# Wan2.1-I2V-14B
#   Repo    : Wan-AI/Wan2.1-I2V-14B  (HuggingFace)
#   Input   : image + text prompt → video
#   Frames  : up to 81 frames (~5 sec at 16 fps)
#   VRAM    : ~48 GB fp16  |  ~24 GB with int8 quantization
#   GPU     : RTX 6000 Ada (48 GB) — minimum for fp16
#             A100 80 GB            — recommended
#             2× A100 40 GB         — works with model sharding
#   Notes   : Highest quality local model for directed character action.
#             Diffusers pipeline: WanImageToVideoPipeline
#             pip install diffusers>=0.32.0
#
# DynamiCrafter (image + text → video)
#   Repo    : Doubiiu/DynamiCrafter_1024  (HuggingFace)
#   Input   : image + text prompt → video
#   Frames  : 16 frames
#   VRAM    : ~16 GB fp16  |  ~10 GB with fp8
#   GPU     : RTX 4080 (16 GB) — minimum
#             RTX 3090 / 4090 (24 GB) — comfortable
#   Notes   : Older model, reasonable prompt following, lower quality than
#             CogVideoX. No native diffusers pipeline; uses custom inference.
#             GitHub: Doubiiu/DynamiCrafter
#
# I2VGen-XL
#   Repo    : ali-vilab/i2vgen-xl  (HuggingFace)
#   Input   : image + text prompt → video
#   Frames  : 16 frames
#   VRAM    : ~18 GB fp16
#   GPU     : RTX 4090 (24 GB)  |  A100 40 GB
#   Notes   : Diffusers pipeline: I2VGenXLPipeline
#             pip install diffusers>=0.28.0
#             Weaker prompt adherence than CogVideoX.
#
# --- CLOUD APIs (no local GPU required) --------------------------------------
#
# Kling 1.6 / 2.0  (Kuaishou)
#   API     : https://klingai.com  /  https://kling.kuaishou.com
#   Input   : image + text prompt → video
#   Frames  : up to 5 sec or 10 sec
#   VRAM    : cloud — no local GPU needed
#   Notes   : Best-in-class prompt following for specific character actions.
#             Paid API. Python SDK available.
#
# RunwayML Gen-3 Alpha / Turbo
#   API     : https://runwayml.com
#   Input   : image + text prompt → video
#   VRAM    : cloud — no local GPU needed
#   Notes   : Strong prompt adherence, good motion quality. Paid API.
#             Python SDK: pip install runwayml
#
# Pika 2.2
#   API     : https://pika.art
#   Input   : image + text prompt → video
#   VRAM    : cloud — no local GPU needed
#   Notes   : Good for cinematic portrait animation. Paid API.
#
# Luma Dream Machine
#   API     : https://lumalabs.ai/dream-machine
#   Input   : image + text prompt → video
#   VRAM    : cloud — no local GPU needed
#   Notes   : Paid API. Smooth output, decent prompt following.
#
# --- CURRENT SCRIPT LIMITATIONS ----------------------------------------------
#
# RTX 4060 8 GB: only SVD and AnimateDiff fit in VRAM.
#   SVD         — smooth motion, ignores text prompt entirely.
#   AnimateDiff — reads text prompt but produces flickering workaround output.
#
# Minimum GPU to run CogVideoX-5B-I2V (smallest viable local model):
#   RTX 4090 24 GB  or  RTX 3090 24 GB (with int8 quantization)
#
# --- UPGRADE PLAN (revisit later) --------------------------------------------
#
# OPTION A — Rent a GPU, run Wan2.1-I2V-14B locally for ~$1-2 per test session
#
#   Model    : Wan-AI/Wan2.1-I2V-14B  (free download from HuggingFace)
#   License  : Accept on HuggingFace model page before downloading.
#              Free for research/personal use. Check terms for commercial use.
#   Download : huggingface-cli download Wan-AI/Wan2.1-I2V-14B
#              (~28 GB, counts toward rental time — download takes ~10-15 min)
#
#   GPU rental platforms:
#     Vast.ai      https://vast.ai       — cheapest, spot marketplace
#     RunPod       https://runpod.io     — easy UI, good for first-timers
#     Lambda Labs  https://lambdalabs.com — clean, datacenter grade
#     Paperspace   https://paperspace.com — Jupyter-friendly
#
#   Minimum GPU:
#     A100 40 GB  — int8 quantized (~24 GB active VRAM), ~$0.50-1.50/hr spot
#     A100 80 GB  — fp16 full precision,                  ~$2-3/hr
#     RTX 6000 Ada 48 GB — fp16 comfortable,              ~$0.80-1.50/hr
#
#   Estimated cost for 5-experiment test session (amunhotep):
#     ~15 min setup + download + ~25 min inference = ~45-60 min total
#     A100 40 GB spot at $1/hr → ~$0.75-$1.00 for the whole session
#
#   Diffusers pipeline : WanImageToVideoPipeline
#   Install            : pip install diffusers>=0.32.0
#
# OPTION B — Use fal.ai API (no GPU rental, pay per video)
#
#   Platform : https://fal.ai
#   Model    : Search "wan" on https://fal.ai/models to find Wan2.1/2.2 I2V
#              endpoint. Verify model ID before coding — it changes.
#   Pricing  : Check model page for rate (per output-second or GPU-second).
#              Wan 2.2 @ 480p was quoted at ~$0.04/sec — verify current rate.
#              Confirm whether "/sec" means output video seconds or GPU seconds.
#   Install  : pip install fal-client
#   Auth     : Set FAL_KEY environment variable with your API key
#   Upload   : Use fal_client.upload_file() to get a public URL for your PNG
#              (fal.ai cannot read local files directly)
#   Free credits on signup — enough to run a few test clips before paying.
#
#   General call pattern (verify exact parameter names on model page):
#     import fal_client
#     url = fal_client.upload_file("amunhotep.png")
#     result = fal_client.subscribe(
#         "fal-ai/wan-i2v",          # verify exact model ID on fal.ai/models
#         arguments={
#             "image_url": url,
#             "prompt": "<motion description from manifest>",
#         }
#     )
#
# RECOMMENDATION
#   First test : fal.ai (zero setup, pay only for what you run, ~$1 total)
#   At scale   : rent GPU via RunPod + Wan2.1-I2V-14B (cheaper per clip)
# =============================================================================

import argparse
import gc
import json
from pathlib import Path


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

# ---------------------------------------------------------------------------
# AnimateDiff constants
# ---------------------------------------------------------------------------
MOTION_ADAPTER_ID = "guoyww/animatediff-motion-adapter-v1-5-3"
SD15_MODEL_ID = "runwayml/stable-diffusion-v1-5"
IMG2VID_STRENGTH = 0.4   # lower = less flicker; 0.3-0.5 recommended
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 25
FPS = 8
FRAME_SIZE = 512  # SD1.5 native

# ---------------------------------------------------------------------------
# SVD constants
# ---------------------------------------------------------------------------
SVD_MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"
SVD_NUM_FRAMES = 25          # SVD-XT fixed output length
SVD_FPS_ID = 6               # motion speed hint to the model (4-30); lower = faster motion
SVD_NOISE_AUG = 0.02         # conditioning noise; 0.0 = sharp input, 0.1 = more creative
SVD_DECODE_CHUNK = 4         # decode N frames at a time to stay within 8 GB
SVD_WIDTH = 512
SVD_HEIGHT = 512

# motion_type → motion_bucket_id (0=still, 255=maximum motion)
MOTION_TYPE_BUCKET: dict[str, int] = {
    "idle":    60,    # subtle breathing / micro-movement
    "gesture": 110,   # arm/hand movement
    "walk":    127,   # standard locomotion (SVD default)
    "run":     160,   # fast movement
    "turn":    100,   # rotation
}

# ---------------------------------------------------------------------------
# Motion LoRAs — camera-movement LoRAs from guoyww (Apache 2.0)
# These add cinematic motion on top of static portraits.
# ---------------------------------------------------------------------------
# All LoRA repos live under guoyww/animatediff-motion-lora-<name>
MOTION_LORA_IDS: dict[str, str] = {
    "zoom-in":              "guoyww/animatediff-motion-lora-zoom-in",
    "zoom-out":             "guoyww/animatediff-motion-lora-zoom-out",
    "pan-left":             "guoyww/animatediff-motion-lora-pan-left",
    "pan-right":            "guoyww/animatediff-motion-lora-pan-right",
    "tilt-up":              "guoyww/animatediff-motion-lora-tilt-up",
    "tilt-down":            "guoyww/animatediff-motion-lora-tilt-down",
    "roll-cw":              "guoyww/animatediff-motion-lora-rolling-clockwise",
    "roll-ccw":             "guoyww/animatediff-motion-lora-rolling-anticlockwise",
}

# motion_type → LoRA name (used when --motion-lora auto)
MOTION_TYPE_LORA: dict[str, str] = {
    "walk":    "zoom-in",    # dolly-in simulates forward movement
    "run":     "zoom-in",
    "gesture": "tilt-up",    # upward pan matches raised/extending arms
    "idle":    "pan-right",  # gentle drift keeps static portrait alive
    "turn":    "pan-left",
}
DEFAULT_LORA_STRENGTH = 0.7


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Animate character portrait PNGs into MP4 clips.",
        epilog=(
            "Models (--model):\n\n"
            "  svd           stabilityai/stable-video-diffusion-img2vid-xt\n"
            "                True img2vid. Smooth output, no flickering.\n"
            "                FP16 + CPU offload, ~7-8 GB VRAM. 512×512.\n"
            "                Does not use --strength, --motion-lora, or --guidance-scale.\n\n"
            "  animatediff   guoyww/animatediff-motion-adapter-v1-5-3\n"
            "                + runwayml/stable-diffusion-v1-5\n"
            "                AnimateDiffVideoToVideoPipeline (img2vid).\n"
            "                FP16 + CPU offload, ~3-4 GB VRAM. 512×512.\n"
            "                Use --strength 0.3-0.5 to reduce flickering.\n"
            "\n"
            "Next test command:\n\n"
            "  python code\\ai\\gen_character_animation.py --model svd --params ..\\all_chars_slow_subtle.json --input_dir projects\\the-pharaoh-who-defied-death\\episodes\\s01e01\\assets\\en --output_dir projects\\the-pharaoh-who-defied-death\\episodes\\s01e01\\assets\\en\n"
            "\n"
            "  Runs 5 SVD experiments for amunhotep and saves:\n"
            "    amunhotep-anim-slow-subtle.mp4\n"
            "    amunhotep-anim-slow-moderate.mp4\n"
            "    amunhotep-anim-gesture-standard.mp4\n"
            "    amunhotep-anim-gesture-expressive.mp4\n"
            "    amunhotep-anim-slow-seed2.mp4\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", choices=["svd", "animatediff"], default="svd",
        help="Model to use for animation (default: svd).",
    )
    parser.add_argument(
        "--params", type=str, default=None,
        help="Path to a per-asset SVD experiment params JSON file. "
             "Runs one inference per experiment entry and skips --manifest inference for that asset.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory containing input portrait PNGs.")
    parser.add_argument("--fps", type=int, default=FPS,
                        help="Output video FPS (default: 8).")
    parser.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS,
                        help="Denoising steps (default: 25).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output files instead of skipping them.")
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to AssetManifest JSON. When given, overrides the hardcoded ANIMATIONS list.",
    )
    parser.add_argument(
        "--asset-id", type=str, default=None, dest="asset_id",
        help="Process only this asset_id (requires --manifest).",
    )
    # AnimateDiff-only
    parser.add_argument(
        "--strength", type=float, default=IMG2VID_STRENGTH,
        help="[animatediff] img2vid strength (0.0-1.0). Lower = less flicker. Default: 0.4.",
    )
    parser.add_argument(
        "--motion-lora", type=str, default="auto", dest="motion_lora",
        help=(
            "[animatediff] Motion LoRA: 'auto' picks by motion_type, "
            "'none' disables, or one of: " + ", ".join(MOTION_LORA_IDS) + "."
        ),
    )
    parser.add_argument(
        "--lora-strength", type=float, default=DEFAULT_LORA_STRENGTH, dest="lora_strength",
        help="[animatediff] LoRA adapter weight (0.0-1.0). Default: 0.7.",
    )
    # SVD-only
    parser.add_argument(
        "--motion-bucket-id", type=int, default=None, dest="motion_bucket_id",
        help=(
            "[svd] Motion intensity 0-255. Default: auto from motion_type "
            "(idle=60, gesture=110, walk=127, run=160). Higher = more motion."
        ),
    )
    parser.add_argument(
        "--noise-aug", type=float, default=SVD_NOISE_AUG, dest="noise_aug",
        help="[svd] Conditioning noise strength (0.0-0.1). Default: 0.02.",
    )
    parser.add_argument(
        "--fps-id", type=int, default=SVD_FPS_ID, dest="fps_id",
        help="[svd] Motion speed hint to the model (4-30). Lower = faster apparent motion. Default: 6.",
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
        motion_desc = motion.get("description", "").strip()
        motion_style = motion.get("style", "")
        prompt = f"{pack['ai_prompt']}, {motion_desc}, photorealistic, smooth motion"
        anims.append({
            "asset_id":       aid,
            "input_image":    f"{aid}.png",
            "prompt":         prompt,
            "negative_prompt": "blurry, jitter, fast motion, low quality, deformed",
            "motion_type":    motion.get("type", "idle"),
            "motion_desc":    motion_desc,
            "motion_style":   motion_style,
            "duration_sec":   duration,
            "num_frames":     num_frames,
            "output":         f"{aid}-anim.mp4",
        })
    return anims


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------
def load_animatediff_pipeline():
    """Load AnimateDiffVideoToVideoPipeline. Exits if unavailable."""
    import torch
    from diffusers import MotionAdapter, DDIMScheduler

    print("[MODEL] Loading AnimateDiff motion adapter v1-5-3...")
    adapter = MotionAdapter.from_pretrained(
        MOTION_ADAPTER_ID,
        torch_dtype=torch.float16,
    )

    try:
        from diffusers import AnimateDiffVideoToVideoPipeline
    except (ImportError, AttributeError) as exc:
        raise SystemExit(
            f"[ERROR] AnimateDiffVideoToVideoPipeline not available in installed diffusers: {exc}\n"
            "        Upgrade: pip install 'diffusers>=0.28.0'"
        )

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
    return pipe


def load_svd_pipeline():
    """Load StableVideoDiffusionPipeline (SVD-XT). Exits if unavailable."""
    import torch

    try:
        from diffusers import StableVideoDiffusionPipeline
    except (ImportError, AttributeError) as exc:
        raise SystemExit(
            f"[ERROR] StableVideoDiffusionPipeline not available in installed diffusers: {exc}\n"
            "        Upgrade: pip install 'diffusers>=0.24.0'"
        )

    print(f"[MODEL] Loading StableVideoDiffusionPipeline (SVD-XT) from {SVD_MODEL_ID}...")
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        SVD_MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.enable_model_cpu_offload()
    print("[MODEL] Using StableVideoDiffusionPipeline (SVD-XT, img2vid).")
    return pipe


# ---------------------------------------------------------------------------
# Motion LoRA loader
# ---------------------------------------------------------------------------
def load_motion_loras(pipe, lora_names: list[str]) -> None:
    """
    Download and register motion LoRA adapters into the pipeline.
    Each LoRA is stored under its short name so set_adapters() can pick
    the right one per-clip without reloading weights.
    """
    for name in lora_names:
        repo_id = MOTION_LORA_IDS[name]
        print(f"[LORA] Loading motion LoRA '{name}' from {repo_id}...")
        pipe.load_lora_weights(repo_id, adapter_name=name)
    if lora_names:
        print(f"[LORA] {len(lora_names)} LoRA(s) registered: {', '.join(lora_names)}")


def resolve_lora_name(motion_lora_arg: str, motion_type: str) -> str | None:
    """
    Return the LoRA short-name to use for this clip, or None if disabled.
    - 'none'    → None (skip LoRA)
    - 'auto'    → look up MOTION_TYPE_LORA for motion_type; None if not mapped
    - anything else → use as-is (validated by parse_args choices)
    """
    if motion_lora_arg == "none":
        return None
    if motion_lora_arg == "auto":
        return MOTION_TYPE_LORA.get(motion_type)
    return motion_lora_arg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_and_resize_image(image_path: Path, width: int, height: int = None):
    """Load a portrait PNG, centre-crop and resize to (width, height)."""
    from PIL import Image
    if height is None:
        height = width
    img = Image.open(str(image_path)).convert("RGB")
    w, h = img.size
    target_ratio = width / height
    src_ratio = w / h
    if src_ratio > target_ratio:
        # image is wider than target — crop sides
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        # image is taller than target — crop top/bottom
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
    img = img.resize((width, height), Image.LANCZOS)
    return img


def make_init_video(frame, num_frames: int) -> list:
    """Repeat a single image N times to create a static init video."""
    return [frame] * num_frames


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def generate_animation(
    pipe,
    anim: dict,
    input_dir: Path,
    args,
    lora_name: str | None = None,
) -> list:
    """Run inference and return list of PIL Image frames."""
    import torch
    from PIL import Image
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    # Activate or deactivate motion LoRA for this clip
    if lora_name:
        pipe.set_adapters([lora_name], adapter_weights=[args.lora_strength])
        print(f"  [LORA] Using '{lora_name}' (strength={args.lora_strength})")
    else:
        try:
            pipe.disable_lora()
        except Exception:
            pass  # no-op if no LoRA was ever loaded

    img_path = input_dir / anim["input_image"]
    if img_path.exists():
        frame = load_and_resize_image(img_path, FRAME_SIZE)
        init_video = make_init_video(frame, anim["num_frames"])
    else:
        print(f"  [WARN] Input image not found: {img_path}. Using grey init.")
        init_video = [Image.new("RGB", (FRAME_SIZE, FRAME_SIZE), (128, 128, 128))] * anim["num_frames"]


    result = pipe(
        video=init_video,
        prompt=anim["prompt"],
        negative_prompt=anim["negative_prompt"],
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=args.steps,
        strength=args.strength,
        generator=generator,
    )

    # diffusers AnimateDiff returns frames as result.frames[0]
    return result.frames[0]


def compute_svd_size(img_w: int, img_h: int, max_side: int = 512) -> tuple[int, int]:
    """
    Compute SVD output (width, height) that preserves the input aspect ratio.
    Longest side is clamped to max_side and both dimensions snapped to multiples of 8.
    """
    scale = max_side / max(img_w, img_h)
    w = max(8, round(img_w * scale / 8) * 8)
    h = max(8, round(img_h * scale / 8) * 8)
    return w, h


def generate_svd_animation(
    pipe,
    anim: dict,
    input_dir: Path,
    args,
) -> list:
    """Run SVD-XT inference and return list of PIL Image frames."""
    import torch
    from PIL import Image
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    img_path = input_dir / anim["input_image"]
    if img_path.exists():
        with Image.open(str(img_path)) as tmp:
            orig_w, orig_h = tmp.size
        svd_w, svd_h = compute_svd_size(orig_w, orig_h, max_side=max(SVD_WIDTH, SVD_HEIGHT))
        image = load_and_resize_image(img_path, svd_w, svd_h)
        print(f"  [SVD] input {orig_w}×{orig_h} → output {svd_w}×{svd_h}")
    else:
        print(f"  [WARN] Input image not found: {img_path}. Using grey init.")
        svd_w, svd_h = SVD_WIDTH, SVD_HEIGHT
        image = Image.new("RGB", (svd_w, svd_h), (128, 128, 128))

    # Use explicit --motion-bucket-id or fall back to motion_type mapping
    if args.motion_bucket_id is not None:
        motion_bucket_id = args.motion_bucket_id
    else:
        motion_bucket_id = MOTION_TYPE_BUCKET.get(anim["motion_type"], 127)
    print(f"  [SVD] motion_bucket_id={motion_bucket_id}  noise_aug={args.noise_aug}  fps={args.fps_id}")

    result = pipe(
        image=image,
        width=svd_w,
        height=svd_h,
        num_frames=SVD_NUM_FRAMES,
        fps=args.fps_id,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=args.noise_aug,
        num_inference_steps=args.steps,
        decode_chunk_size=SVD_DECODE_CHUNK,
        generator=generator,
    )
    return result.frames[0]


# ---------------------------------------------------------------------------
# Animation plan printer
# ---------------------------------------------------------------------------
_PROMPT_MIN_WORDS = 6  # fewer words → flag as unclear

def _prompt_clarity(motion_desc: str) -> tuple[bool, str]:
    """
    Return (is_clear, reason).
    is_clear=False means the description is too vague to drive good results.
    """
    if not motion_desc:
        return False, "no motion description in manifest"
    words = motion_desc.split()
    if len(words) < _PROMPT_MIN_WORDS:
        return False, f"description too short ({len(words)} words, need ≥{_PROMPT_MIN_WORDS})"
    return True, ""


def print_animation_plan(animations: list, args) -> None:
    """
    Print, for every clip, what motion description was read from the manifest
    and what inference parameters will be used — before any model is loaded.
    """
    W = 70
    print("\n" + "=" * W)
    print(f"  ANIMATION PLAN   model={args.model}   seed={args.seed}   steps={args.steps}")
    print("=" * W)

    for i, anim in enumerate(animations, 1):
        motion_desc = anim.get("motion_desc", "")
        motion_style = anim.get("motion_style", "")
        is_clear, reason = _prompt_clarity(motion_desc)

        print(f"\n  [{i}/{len(animations)}] {anim['asset_id']}")
        print(f"    motion_type  : {anim['motion_type']}"
              + (f"  (style: {motion_style})" if motion_style else ""))
        print(f"    duration_sec : {anim['duration_sec']}")

        # Motion description / prompt
        if is_clear:
            # Word-wrap at 60 chars for readability
            words = motion_desc.split()
            lines, line = [], []
            for w in words:
                if sum(len(x) + 1 for x in line) + len(w) > 60:
                    lines.append(" ".join(line))
                    line = []
                line.append(w)
            if line:
                lines.append(" ".join(line))
            prefix = "    motion_desc  : "
            indent = " " * len(prefix)
            print(prefix + lines[0])
            for l in lines[1:]:
                print(indent + l)
        else:
            print(f"    motion_desc  : [NOT CLEAR — {reason}]")
            print(f"                   Add a detailed 'description' to this character's")
            print(f"                   motion block in the manifest to improve results.")

        # Per-model parameters
        if args.model == "svd":
            bucket = (
                args.motion_bucket_id
                if args.motion_bucket_id is not None
                else MOTION_TYPE_BUCKET.get(anim["motion_type"], 127)
            )
            print(f"    --- SVD parameters ---")
            print(f"    frames         : {SVD_NUM_FRAMES}  (SVD-XT fixed)")
            print(f"    motion_bucket  : {bucket}  (0=still → 255=max motion)")
            print(f"    fps_id         : {args.fps_id}  (lower=faster apparent motion)")
            print(f"    noise_aug      : {args.noise_aug}")
            print(f"    NOTE: SVD does not use the text prompt.")
            if not is_clear:
                print(f"          Adjust motion_bucket_id to control motion intensity instead.")
        else:
            lora = resolve_lora_name(args.motion_lora, anim["motion_type"])
            print(f"    --- AnimateDiff parameters ---")
            print(f"    frames         : {anim['num_frames']}")
            print(f"    strength       : {args.strength}  (lower=less flicker)")
            print(f"    motion_lora    : {lora or 'none'}")
            print(f"    lora_strength  : {args.lora_strength}")
            if is_clear:
                print(f"    NOTE: prompt IS used by AnimateDiff (text guides the motion).")
            else:
                print(f"    NOTE: prompt will be generic — improve manifest description")
                print(f"          for better directed motion.")

        print(f"    output         : {anim['output']}")

    print("\n" + "=" * W + "\n")


# ---------------------------------------------------------------------------
# Params-file runner (SVD experiments)
# ---------------------------------------------------------------------------
def run_params_experiments(params_path: str, pipe, input_dir: Path, out_dir: Path, args) -> list:
    """
    Load a params JSON file and run one SVD inference per experiment entry.
    Returns a results list in the same format as main()'s results.
    """
    import copy
    with open(params_path, encoding="utf-8") as f:
        params = json.load(f)

    model      = params.get("model", "svd")
    note       = params.get("note", "")
    intent     = params.get("manifest_intent", "")
    limit_note = params.get("svd_limitation", "")
    experiments = params.get("experiments", [])

    # Support both single asset_id and list of asset_ids
    if "asset_ids" in params:
        asset_ids = params["asset_ids"]
    elif "asset_id" in params:
        asset_ids = [params["asset_id"]]
    else:
        raise SystemExit("[ERROR] params file must have 'asset_id' or 'asset_ids'.")

    if model != "svd":
        raise SystemExit(f"[ERROR] params file model='{model}' — only 'svd' is supported in --params mode.")

    total_jobs = len(asset_ids) * len(experiments)
    print("\n" + "=" * 70)
    print(f"  PARAMS EXPERIMENT   file={params_path}")
    print("=" * 70)
    if note:
        print(f"  Note            : {note}")
    if intent:
        print(f"  Manifest intent : {intent}")
    if limit_note:
        print(f"  SVD limitation  : {limit_note}")
    print(f"  Characters      : {', '.join(asset_ids)}")
    print(f"  Experiments     : {len(experiments)}  ({total_jobs} total clips)")
    print("=" * 70 + "\n")

    results = []
    job = 0

    for asset_id in asset_ids:
        for idx, exp in enumerate(experiments, 1):
            job += 1
            exp_id   = exp.get("id", f"exp{idx}")
            exp_note = exp.get("note", "")
            out_name = f"{asset_id}-anim-{exp_id}.mp4"
            out_path = out_dir / out_name

            print(f"[{job}/{total_jobs}] {asset_id}  /  {exp_id}")
            print(f"  note           : {exp_note}")
            print(f"  motion_bucket  : {exp.get('motion_bucket_id', 127)}")
            print(f"  fps_id         : {exp.get('fps_id', SVD_FPS_ID)}")
            print(f"  noise_aug      : {exp.get('noise_aug', SVD_NOISE_AUG)}")
            print(f"  steps          : {exp.get('steps', args.steps)}")
            print(f"  seed           : {exp.get('seed', args.seed)}")
            print(f"  output         : {out_name}")

            base_anim = {
                "asset_id":    asset_id,
                "input_image": f"{asset_id}.png",
                "motion_type": "gesture",
                "motion_desc": intent,
                "motion_style": "",
                "duration_sec": 4.0,
                "num_frames":  SVD_NUM_FRAMES,
            }

            if out_path.exists() and not args.force:
                print(f"  [SKIP] already exists\n")
                results.append({"asset_id": asset_id, "experiment": exp_id,
                                "output": str(out_path), "size_bytes": out_path.stat().st_size,
                                "status": "skipped"})
                continue

            # Build a temporary args-like namespace overriding with experiment values
            exp_args = copy.copy(args)
            exp_args.motion_bucket_id = exp.get("motion_bucket_id", args.motion_bucket_id)
            exp_args.fps_id           = exp.get("fps_id",           args.fps_id)
            exp_args.noise_aug        = exp.get("noise_aug",        args.noise_aug)
            exp_args.steps            = exp.get("steps",            args.steps)
            exp_args.seed             = exp.get("seed",             args.seed)

            try:
                import time
                import torch
                from diffusers.utils import export_to_video
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                frames = generate_svd_animation(pipe, base_anim, input_dir, exp_args)
                export_to_video(frames, str(out_path), fps=args.fps)
                elapsed = time.time() - t0
                peak_vram_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
                size = out_path.stat().st_size
                print(f"  [OK] {out_path.resolve()}")
                print(f"       {len(frames)} frames  |  {elapsed:.1f}s  |  peak VRAM {peak_vram_gb:.2f} GB  |  {size:,} bytes\n")
                results.append({"asset_id": asset_id, "experiment": exp_id,
                                "output": str(out_path), "size_bytes": size,
                                "num_frames": len(frames), "elapsed_sec": round(elapsed, 1),
                                "peak_vram_gb": round(peak_vram_gb, 2), "status": "success"})
            except Exception as exc:
                print(f"  [ERROR] {exc}\n")
                results.append({"asset_id": asset_id, "experiment": exp_id,
                                "output": str(out_path), "size_bytes": 0,
                                "status": "failed", "error": str(exc)})
            finally:
                import torch
                torch.cuda.empty_cache()
                gc.collect()

    return results


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

    if args.model == "svd":
        pipe = load_svd_pipeline()
    else:
        pipe = load_animatediff_pipeline()

    # --params: run SVD experiments from file, skip normal manifest loop
    if args.params:
        if args.model != "svd":
            raise SystemExit("[ERROR] --params is only supported with --model svd")
        results = run_params_experiments(args.params, pipe, input_dir, out_dir, args)
        manifest_path = out_dir / f"{SCRIPT_NAME}_results.json"
        with open(manifest_path, "w") as fh:
            json.dump(results, fh, indent=2)
        ok = sum(1 for r in results if r["status"] in ("success", "skipped"))
        print(f"\n{ok}/{len(results)} experiments completed  |  manifest: {manifest_path}")
        return

    print_animation_plan(animations, args)

    # Pre-load AnimateDiff LoRAs (SVD has no LoRA support here)
    needed_loras: list[str] = []
    if args.model == "animatediff" and args.motion_lora != "none":
        for anim in animations:
            name = resolve_lora_name(args.motion_lora, anim["motion_type"])
            if name and name not in needed_loras:
                needed_loras.append(name)
    if needed_loras:
        load_motion_loras(pipe, needed_loras)

    results = []
    total = len(animations)

    for idx, anim in enumerate(animations, start=1):
        out_path = out_dir / anim["output"]
        lora_name = resolve_lora_name(args.motion_lora, anim["motion_type"]) if args.model == "animatediff" else None
        print(f"\n[{idx}/{total}] [{args.model}] Animating {anim['asset_id']} ({anim['motion_type']})...")

        if out_path.exists() and not args.force:
            print(f"  [SKIP] {anim['output']} already exists")
            results.append({
                "asset_id": anim["asset_id"],
                "output": str(out_path),
                "size_bytes": out_path.stat().st_size,
                "status": "skipped",
            })
            continue

        try:
            import time
            import torch
            from diffusers.utils import export_to_video
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            if args.model == "svd":
                frames = generate_svd_animation(pipe, anim, input_dir, args)
            else:
                frames = generate_animation(pipe, anim, input_dir, args, lora_name=lora_name)
            export_to_video(frames, str(out_path), fps=args.fps)
            elapsed = time.time() - t0
            peak_vram_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
            size = out_path.stat().st_size
            print(f"  [OK] {out_path.resolve()}")
            print(f"       {len(frames)} frames  |  {elapsed:.1f}s  |  peak VRAM {peak_vram_gb:.2f} GB  |  {size:,} bytes")
            results.append({
                "asset_id": anim["asset_id"],
                "output": str(out_path),
                "size_bytes": size,
                "num_frames": len(frames),
                "elapsed_sec": round(elapsed, 1),
                "peak_vram_gb": round(peak_vram_gb, 2),
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
            import torch
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
