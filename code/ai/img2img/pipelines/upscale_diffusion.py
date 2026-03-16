# code/ai/img2img/pipelines/upscale_diffusion.py
"""
pipe  = SD x4 Upscaler pipeline (load_sd_upscaler)
args  = Namespace: input, scale (default 4), prompt (optional)

SD x4 Upscaler always runs at 4x internally.
If args.scale == 2: output is downsampled 2x after 4x upscale for antialiasing.
VRAM is managed by enable_attention_slicing() applied in load_sd_upscaler().

NOTE: SD x4 Upscaler requires the input to be ≤512px on the longest side.
For most pipeline outputs (backgrounds, composites, character mattes), prefer
gen_upscale.py (Real-ESRGAN), which handles arbitrary input sizes, is faster,
preserves alpha channels, and is deterministic. Use this mode only when you
specifically want the model to hallucinate new fine-grain texture detail.
"""

import logging
from PIL import Image

log = logging.getLogger(__name__)


def run(pipe, config, args) -> Image.Image:
    raw = Image.open(args.input)
    if raw.mode == "RGBA":
        log.warning(
            "Input image is RGBA — alpha channel will be discarded by SD x4 Upscaler. "
            "For RGBA character mattes use gen_upscale.py (Real-ESRGAN) instead."
        )
    image = raw.convert("RGB")
    orig_w, orig_h = image.size

    prompt = getattr(args, "prompt", None) or "high resolution, sharp details, photorealistic"
    target_scale = getattr(args, "scale", None) or config.DEFAULTS["upscale"]["scale"]

    # SD x4 Upscaler input must be ≤512 on the longest side.
    max_input = 512
    if max(orig_w, orig_h) > max_input:
        log.warning(
            f"Input is {orig_w}x{orig_h} — larger than SD x4 Upscaler's 512px limit. "
            "Image will be downscaled before upscaling, losing detail in the process. "
            "Consider using gen_upscale.py (Real-ESRGAN) instead, which handles "
            "arbitrary input sizes without pre-downscaling."
        )
        ratio = max_input / max(orig_w, orig_h)
        image = image.resize((int(orig_w * ratio), int(orig_h * ratio)), Image.LANCZOS)
        log.info(f"Resized input to {image.size} for SD x4 Upscaler.")

    upscaled = pipe(
        prompt=prompt,
        image=image,
        num_inference_steps=20,
        guidance_scale=7.5,
    ).images[0]

    # Downscale to target_scale if not 4x
    if target_scale != 4:
        final_w = int(orig_w * target_scale)
        final_h = int(orig_h * target_scale)
        upscaled = upscaled.resize((final_w, final_h), Image.LANCZOS)

    log.info(f"Diffusion upscale done: {orig_w}x{orig_h} → {upscaled.size} ({target_scale}x)")
    return upscaled
