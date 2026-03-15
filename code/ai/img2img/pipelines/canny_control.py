# code/ai/img2img/pipelines/canny_control.py
"""
pipe  = SDXL + ControlNet-Canny pipeline
args  = Namespace: input, prompt, low_threshold, high_threshold, strength

Canny edge extraction uses OpenCV (already in requirements.txt).
No extra model needed for conditioning extraction.
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

    from img2img.io_utils import resize_to_sdxl
    image = resize_to_sdxl(image)

    low  = getattr(args, "low_threshold",  config.DEFAULTS["canny"]["low_threshold"])
    high = getattr(args, "high_threshold", config.DEFAULTS["canny"]["high_threshold"])
    canny_image = _extract_canny(image, low, high)

    result = pipe(
        prompt=args.prompt,
        negative_prompt=config.NEGATIVE_PROMPT,
        image=image,
        control_image=canny_image,
        controlnet_conditioning_scale=0.75,
        strength=getattr(args, "strength", config.DEFAULTS["canny"]["strength"]),
        num_inference_steps=getattr(args, "steps", config.DEFAULTS["canny"]["steps"]),
        guidance_scale=getattr(args, "guidance", config.DEFAULTS["canny"]["guidance"]),
    ).images[0]

    result = result.resize(orig_size, Image.LANCZOS)
    log.info("Canny ControlNet done.")
    return result
