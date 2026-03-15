# code/ai/img2img/pipelines/inpaint.py
"""
pipe  = SDXL Inpainting pipeline (load_sdxl_inpaint)
args  = Namespace: input, mask, prompt, strength, steps, guidance, output
"""

import logging
from PIL import Image

log = logging.getLogger(__name__)


def run(pipe, config, args) -> Image.Image:
    image = Image.open(args.input).convert("RGB")
    mask  = Image.open(args.mask).convert("L")  # white = fill

    orig_size = image.size

    # Resize to SDXL-friendly size (1024-ish)
    from img2img.io_utils import resize_to_sdxl
    image = resize_to_sdxl(image)
    mask  = mask.resize(image.size, Image.NEAREST)

    result = pipe(
        prompt=args.prompt,
        negative_prompt=config.NEGATIVE_PROMPT,
        image=image,
        mask_image=mask,
        strength=getattr(args, "strength", config.DEFAULTS["inpaint"]["strength"]),
        num_inference_steps=getattr(args, "steps", config.DEFAULTS["inpaint"]["steps"]),
        guidance_scale=getattr(args, "guidance", config.DEFAULTS["inpaint"]["guidance"]),
    ).images[0]

    # Restore original resolution
    result = result.resize(orig_size, Image.LANCZOS)
    log.info(f"Inpaint done → {orig_size}")
    return result
