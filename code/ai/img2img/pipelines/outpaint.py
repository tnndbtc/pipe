# code/ai/img2img/pipelines/outpaint.py
"""
pipe  = SDXL Inpainting pipeline
args  = Namespace: input, direction (left|right|up|down|all), pixels, prompt
"""

import logging
from PIL import Image, ImageDraw

log = logging.getLogger(__name__)


def _build_outpaint_inputs(image: Image.Image, direction: str,
                            pixels: int) -> tuple[Image.Image, Image.Image]:
    """Return (expanded_canvas, mask) where white mask = region to generate."""
    w, h = image.size
    dirs = ["left", "right", "up", "down"] if direction == "all" else [direction]

    lpad = pixels if "left" in dirs else 0
    tpad = pixels if "up"   in dirs else 0

    nw = w + (pixels if "left" in dirs else 0) + (pixels if "right" in dirs else 0)
    nh = h + (pixels if "up"   in dirs else 0) + (pixels if "down"  in dirs else 0)

    canvas = Image.new("RGB", (nw, nh), (128, 128, 128))
    canvas.paste(image.convert("RGB"), (lpad, tpad))

    # Mask: white everywhere the original image is NOT
    mask = Image.new("L", (nw, nh), 255)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([lpad, tpad, lpad + w - 1, tpad + h - 1], fill=0)

    return canvas, mask


def run(pipe, config, args) -> Image.Image:
    image = Image.open(args.input).convert("RGB")
    direction = getattr(args, "direction", "right")
    pixels    = getattr(args, "pixels", config.DEFAULTS["outpaint"]["pixels"])

    canvas, mask = _build_outpaint_inputs(image, direction, pixels)

    from img2img.io_utils import resize_to_sdxl
    canvas_r = resize_to_sdxl(canvas)
    mask_r   = mask.resize(canvas_r.size, Image.NEAREST)
    orig_size = canvas.size

    result = pipe(
        prompt=args.prompt,
        negative_prompt=config.NEGATIVE_PROMPT,
        image=canvas_r,
        mask_image=mask_r,
        strength=0.99,
        num_inference_steps=getattr(args, "steps", config.DEFAULTS["outpaint"]["steps"]),
        guidance_scale=getattr(args, "guidance", config.DEFAULTS["outpaint"]["guidance"]),
    ).images[0]

    result = result.resize(orig_size, Image.LANCZOS)
    log.info(f"Outpaint done → {orig_size}, direction={direction}, pixels={pixels}")
    return result
