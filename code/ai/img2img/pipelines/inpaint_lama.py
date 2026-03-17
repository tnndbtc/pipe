# code/ai/img2img/pipelines/inpaint_lama.py
"""
pipe  = SimpleLama instance  (simple-lama-inpainting)
args  = Namespace: input, mask, output
        prompt is intentionally ignored — LaMa is image-driven only.

Strategy: LaMa (Large Mask inpainting) — pure texture continuation.
  No diffusion, no text generation. LaMa extends the surrounding texture
  pattern into the masked region using Fourier convolutions trained
  specifically for large-mask object removal tasks.

  Why use this over FLUX / SDXL:
    - Never "imagines" new content — it only continues what is already there.
    - Ideal for repetitive architectural backgrounds: stone floors, brick walls,
      tiled surfaces, wooden floors.  Exactly what Grok-style object removal does.
    - Fast: ~1-2s on CPU, <0.5s on GPU. No diffusion steps.
    - Deterministic: same input always produces the same output.

  Compositing: LaMa result is pasted back with a feathered mask so every pixel
  outside the masked region is pixel-perfect from the original.

Requirements:
  pip install simple-lama-inpainting
"""

import logging

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def run(pipe, config, args) -> Image.Image:
    image = Image.open(args.input).convert("RGB")
    mask  = Image.open(args.mask).convert("L")   # white = fill region

    orig_size = image.size
    W, H = orig_size

    mask_np = np.array(mask)
    coverage = mask_np.sum() / (W * H * 255)
    log.info(f"[inpaint-lama] {W}x{H}  mask coverage={coverage:.1%}")

    # LaMa expects (PIL Image, PIL Image) — white mask = inpaint region
    result = pipe(image, mask)

    # Feather mask edges for seamless compositing
    feather = 10
    mask_feathered_np = cv2.GaussianBlur(
        mask_np.astype(np.float32),
        (feather * 2 + 1, feather * 2 + 1),
        feather / 3,
    )
    mask_feathered_np = np.clip(mask_feathered_np, 0, 255).astype(np.uint8)
    mask_feathered    = Image.fromarray(mask_feathered_np)

    # LaMa may resize internally — snap back to original dimensions
    if result.size != orig_size:
        result = result.resize(orig_size, Image.LANCZOS)

    # Composite: LaMa fill inside mask, original pixels outside
    out = image.copy()
    out.paste(result, (0, 0), mask_feathered)

    log.info(f"Inpaint (LaMa) done → {orig_size}")
    return out
