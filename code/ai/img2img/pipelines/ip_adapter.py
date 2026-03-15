# code/ai/img2img/pipelines/ip_adapter.py
"""
pipe  = SDXL img2img with IP-Adapter-Plus attached (load_ip_adapter(load_sdxl_img2img()))
args  = Namespace: input, style_image, prompt, ip_scale, strength, steps, guidance
"""

import logging
from PIL import Image

log = logging.getLogger(__name__)


def run(pipe, config, args) -> Image.Image:
    image       = Image.open(args.input).convert("RGB")
    style_image = Image.open(args.style_image).convert("RGB")
    orig_size = image.size

    from img2img.io_utils import resize_to_sdxl
    image       = resize_to_sdxl(image)
    style_image = style_image.resize(image.size, Image.LANCZOS)

    ip_scale = getattr(args, "ip_scale", config.DEFAULTS["ip_adapter"]["ip_scale"])
    pipe.set_ip_adapter_scale(ip_scale)

    result = pipe(
        prompt=args.prompt,
        negative_prompt=config.NEGATIVE_PROMPT,
        image=image,
        ip_adapter_image=style_image,
        strength=getattr(args, "strength", 0.6),
        num_inference_steps=getattr(args, "steps", config.DEFAULTS["ip_adapter"]["steps"]),
        guidance_scale=getattr(args, "guidance", config.DEFAULTS["ip_adapter"]["guidance"]),
    ).images[0]

    result = result.resize(orig_size, Image.LANCZOS)
    log.info(f"IP-Adapter done (ip_scale={ip_scale}).")
    return result
