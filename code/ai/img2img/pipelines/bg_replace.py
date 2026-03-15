# code/ai/img2img/pipelines/bg_replace.py
"""
pipe  = {"sdxl": sdxl_img2img_pipe | None}
args  = Namespace: input, bg, blend_strength

Uses rembg (RMBG-1.4, already in requirements.txt).
Step 1: Remove background → RGBA foreground.
Step 2: Alpha-composite onto new background.
Step 3 (optional): SDXL img2img blend pass if blend_strength > 0.
"""

import logging
from PIL import Image
from rembg import remove

log = logging.getLogger(__name__)


def run(pipe: dict, config, args) -> Image.Image:
    image = Image.open(args.input)
    bg    = Image.open(args.bg).convert("RGBA")

    # Step 1: Remove background
    log.info("Running RMBG-1.4 background removal...")
    fg_rgba = remove(image)  # returns RGBA PIL image
    log.info("Background removed.")

    # Step 2: Resize fg to match bg, composite
    fg_resized = fg_rgba.resize(bg.size, Image.LANCZOS)
    canvas = bg.copy()
    canvas.alpha_composite(fg_resized)

    # Step 3: Optional SDXL blend
    blend_strength = getattr(args, "blend_strength",
                              config.DEFAULTS["bg_replace"]["blend_strength"])
    if blend_strength and blend_strength > 0.0 and pipe.get("sdxl"):
        from img2img.io_utils import resize_to_sdxl
        canvas_rgb = canvas.convert("RGB")
        canvas_resized = resize_to_sdxl(canvas_rgb)
        result = pipe["sdxl"](
            prompt=getattr(args, "prompt", "photorealistic composite, natural lighting"),
            negative_prompt=config.NEGATIVE_PROMPT,
            image=canvas_resized,
            strength=blend_strength,
            num_inference_steps=20,
            guidance_scale=5.0,
        ).images[0]
        result = result.resize(canvas.size, Image.LANCZOS)
        canvas = result.convert("RGBA")
        log.info(f"SDXL blend pass done (strength={blend_strength}).")

    log.info(f"Background replace done → {canvas.size}")
    return canvas
