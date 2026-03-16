# code/ai/img2img/pipelines/canny_control.py
"""
pipe  = FLUX.1-dev + InstantX ControlNet-Canny pipeline (load_flux_controlnet)
args  = Namespace: input, prompt, low_threshold, high_threshold, strength

Canny edge extraction uses OpenCV (already in requirements.txt).
Image dimensions are snapped to multiples of 16 (FLUX requirement).
"""

import cv2
import logging
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def _extract_canny(image: Image.Image, low: int, high: int) -> Image.Image:
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)


def run(pipe, config, args) -> Image.Image:
    image = Image.open(args.input).convert("RGB")
    orig_size = image.size

    from img2img.io_utils import snap_to_flux
    image = snap_to_flux(image)

    d = config.DEFAULTS["canny"]
    low  = getattr(args, "low_threshold",  None) or d["low_threshold"]
    high = getattr(args, "high_threshold", None) or d["high_threshold"]
    canny_image = _extract_canny(image, low, high)

    result = pipe(
        prompt=args.prompt,
        image=image,
        control_image=canny_image,
        controlnet_conditioning_scale=0.7,
        strength=getattr(args, "strength", None) or d["strength"],
        num_inference_steps=getattr(args, "steps", None) or d["steps"],
        guidance_scale=getattr(args, "guidance", None) or d["guidance"],
    ).images[0]

    result = result.resize(orig_size, Image.LANCZOS)
    log.info("Canny ControlNet (FLUX) done.")
    return result
