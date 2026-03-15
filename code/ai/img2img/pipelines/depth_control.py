# code/ai/img2img/pipelines/depth_control.py
"""
pipe  = {"sdxl_controlnet": pipe, "depth": depth_pipe}
args  = Namespace: input, prompt, strength, steps, guidance
"""

import logging
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def run(pipe: dict, config, args) -> Image.Image:
    image = Image.open(args.input).convert("RGB")
    orig_size = image.size

    from img2img.io_utils import resize_to_sdxl
    image = resize_to_sdxl(image)

    # Extract depth map
    depth_result = pipe["depth"](image)
    depth_image = depth_result["depth"]  # PIL grayscale
    depth_image = depth_image.convert("RGB").resize(image.size)

    result = pipe["sdxl_controlnet"](
        prompt=args.prompt,
        negative_prompt=config.NEGATIVE_PROMPT,
        image=image,
        control_image=depth_image,
        controlnet_conditioning_scale=0.75,
        strength=getattr(args, "strength", config.DEFAULTS["depth"]["strength"]),
        num_inference_steps=getattr(args, "steps", config.DEFAULTS["depth"]["steps"]),
        guidance_scale=getattr(args, "guidance", config.DEFAULTS["depth"]["guidance"]),
    ).images[0]

    result = result.resize(orig_size, Image.LANCZOS)
    log.info("Depth ControlNet done.")
    return result
