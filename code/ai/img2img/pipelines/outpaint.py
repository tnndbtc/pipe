# code/ai/img2img/pipelines/outpaint.py
"""
pipe  = FLUX.1-Fill-dev pipeline (load_flux_fill)
args  = Namespace: input, direction (left|right|up|down|all), pixels, prompt

Strategy: expand the canvas in the requested direction(s), then fill the new
region with FLUX Fill. FLUX conditions on the full scene so the generated
extension naturally matches the existing image in lighting, style, and content.
Dimensions are snapped to multiples of 16 before inference, then restored.
"""

import logging
import numpy as np
import cv2
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
    d = config.DEFAULTS["outpaint"]
    direction = getattr(args, "direction", None) or d["direction"]
    pixels    = getattr(args, "pixels", None) or d["pixels"]

    canvas, mask = _build_outpaint_inputs(image, direction, pixels)
    orig_size = canvas.size
    W, H = orig_size

    # Light feather on mask edges for seamless compositing
    mask_np = np.array(mask)
    feather  = 10
    mask_feathered_np = cv2.GaussianBlur(
        mask_np.astype(np.float32),
        (feather * 2 + 1, feather * 2 + 1),
        feather / 3,
    )
    mask_feathered_np = np.clip(mask_feathered_np, 0, 255).astype(np.uint8)
    mask_feathered = Image.fromarray(mask_feathered_np)

    # Snap to multiple of 16 (FLUX requirement)
    nw = (W // 16) * 16
    nh = (H // 16) * 16
    canvas_in = canvas.resize((nw, nh), Image.LANCZOS) if (nw, nh) != (W, H) else canvas
    mask_in   = mask_feathered.resize((nw, nh), Image.LANCZOS) if (nw, nh) != (W, H) else mask_feathered

    log.info(f"[outpaint] {canvas_in.size}  direction={direction}  pixels={pixels}")

    result = pipe(
        prompt=args.prompt,
        image=canvas_in,
        mask_image=mask_in,
        height=nh,
        width=nw,
        num_inference_steps=getattr(args, "steps", None) or d["steps"],
        guidance_scale=getattr(args, "guidance", None) or d["guidance"],
    ).images[0]

    if result.size != orig_size:
        result = result.resize(orig_size, Image.LANCZOS)

    # Composite: FLUX result inside mask, original canvas outside
    out = canvas.copy()
    out.paste(result, (0, 0), mask_feathered)

    log.info(f"Outpaint (FLUX Fill) done → {orig_size}, direction={direction}, pixels={pixels}")
    return out
