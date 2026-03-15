# code/ai/img2img/pipelines/style_transfer.py
"""
pipe  = SDXL img2img pipeline
args  = Namespace: input, prompt, strength, steps, guidance
"""

import logging
from PIL import Image

log = logging.getLogger(__name__)


def run(pipe, config, args) -> Image.Image:
    image = Image.open(args.input).convert("RGB")
    orig_size = image.size

    from img2img.io_utils import resize_to_sdxl
    image = resize_to_sdxl(image)

    result = pipe(
        prompt=args.prompt,
        negative_prompt=config.NEGATIVE_PROMPT,
        image=image,
        strength=getattr(args, "strength", config.DEFAULTS["style"]["strength"]),
        num_inference_steps=getattr(args, "steps", config.DEFAULTS["style"]["steps"]),
        guidance_scale=getattr(args, "guidance", config.DEFAULTS["style"]["guidance"]),
    ).images[0]

    result = result.resize(orig_size, Image.LANCZOS)
    log.info(f"Style transfer done (strength={getattr(args, 'strength', config.DEFAULTS['style']['strength'])}).")
    return result
