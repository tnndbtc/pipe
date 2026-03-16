# code/ai/img2img/pipelines/inpaint_flux.py
"""
pipe  = FLUX.1-Fill-dev pipeline (load_flux_fill)
args  = Namespace: input, mask, prompt, steps, guidance, output

Strategy: full-image FLUX Fill.
  FLUX Fill operates on the complete image at native resolution — no patch
  cropping needed. It uses the full scene as context for the filled region,
  which produces far better continuity than patch-based SDXL inpainting.
  No prefill step: FLUX Fill handles object removal natively.

  Mask is lightly feathered for seamless edge blending. The raw FLUX output
  is composited back over the original using the mask so pixels outside the
  masked region are pixel-perfect originals.
"""

import logging

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def run(pipe, config, args) -> Image.Image:
    image = Image.open(args.input).convert("RGB")
    mask  = Image.open(args.mask).convert("L")   # white = fill

    orig_size = image.size
    W, H = orig_size

    # --- Light feather on mask edges for seamless compositing ---
    mask_np = np.array(mask)
    feather = 10
    mask_feathered_np = cv2.GaussianBlur(
        mask_np.astype(np.float32),
        (feather * 2 + 1, feather * 2 + 1),
        feather / 3,
    )
    mask_feathered_np = np.clip(mask_feathered_np, 0, 255).astype(np.uint8)
    mask_feathered    = Image.fromarray(mask_feathered_np)

    # --- Snap to multiple of 16 (FLUX requirement) ---
    nw = (W // 16) * 16
    nh = (H // 16) * 16
    image_in = image.resize((nw, nh), Image.LANCZOS) if (nw, nh) != (W, H) else image
    mask_in  = mask_feathered.resize((nw, nh), Image.LANCZOS) if (nw, nh) != (W, H) else mask_feathered

    log.info(f"[inpaint-flux] {image_in.size}  mask={mask_np.sum() / (W * H * 255):.1%} of image")

    d = config.DEFAULTS["inpaint"]
    result = pipe(
        prompt=args.prompt,
        image=image_in,
        mask_image=mask_in,
        height=nh,
        width=nw,
        num_inference_steps=getattr(args, "steps", None) or d["steps"],
        guidance_scale=getattr(args, "guidance", None) or 30.0,
    ).images[0]

    # --- Resize back to original if snapping changed dimensions ---
    if result.size != orig_size:
        result = result.resize(orig_size, Image.LANCZOS)

    # --- Composite: FLUX result inside mask, original outside ---
    out = image.copy()
    out.paste(result, (0, 0), mask_feathered)

    log.info(f"Inpaint (FLUX Fill) done → {orig_size}")
    return out
