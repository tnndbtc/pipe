# code/ai/img2img/pipelines/style_transfer.py
"""
pipe  = FLUX.1-schnell img2img pipeline (load_flux_img2img)
args  = Namespace: input, prompt, strength, steps, guidance

FLUX.1-schnell is a CFG-free distilled model: guidance_scale=0.0, steps=4.
No negative_prompt. Image dimensions are snapped to multiples of 16.
"""

import logging
from PIL import Image

log = logging.getLogger(__name__)


def run(pipe, config, args) -> Image.Image:
    image = Image.open(args.input).convert("RGB")
    orig_size = image.size

    from img2img.io_utils import snap_to_flux
    image = snap_to_flux(image)

    d = config.DEFAULTS["style"]
    strength = getattr(args, "strength", None) or d["strength"]
    result = pipe(
        prompt=args.prompt,
        image=image,
        strength=strength,
        num_inference_steps=getattr(args, "steps", None) or d["steps"],
        guidance_scale=getattr(args, "guidance", None) if getattr(args, "guidance", None) is not None else d["guidance"],
    ).images[0]

    result = result.resize(orig_size, Image.LANCZOS)
    log.info(f"Style transfer (FLUX schnell) done (strength={strength}).")
    return result
