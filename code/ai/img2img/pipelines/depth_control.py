# code/ai/img2img/pipelines/depth_control.py
"""
pipe  = {"flux_controlnet": FluxControlNetImg2ImgPipeline, "depth": depth_pipe}
args  = Namespace: input, prompt, strength, steps, guidance

Depth map is extracted with Depth-Anything-V2-Small, then used as the
ControlNet conditioning image for FLUX.1-dev.
"""

import logging
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def run(pipe: dict, config, args) -> Image.Image:
    image = Image.open(args.input).convert("RGB")
    orig_size = image.size

    from img2img.io_utils import snap_to_flux
    image = snap_to_flux(image)

    # Extract depth map
    depth_result = pipe["depth"](image)
    depth_image = depth_result["depth"]  # PIL grayscale
    depth_image = depth_image.convert("RGB").resize(image.size)

    d = config.DEFAULTS["depth"]
    result = pipe["flux_controlnet"](
        prompt=args.prompt,
        image=image,
        control_image=depth_image,
        controlnet_conditioning_scale=0.7,
        strength=getattr(args, "strength", None) or d["strength"],
        num_inference_steps=getattr(args, "steps", None) or d["steps"],
        guidance_scale=getattr(args, "guidance", None) or d["guidance"],
    ).images[0]

    result = result.resize(orig_size, Image.LANCZOS)
    log.info("Depth ControlNet (FLUX) done.")
    return result
