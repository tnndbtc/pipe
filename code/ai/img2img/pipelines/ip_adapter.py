# code/ai/img2img/pipelines/ip_adapter.py
"""
pipe  = {"prior": FluxPriorReduxPipeline, "flux": FluxImg2ImgPipeline (FLUX.1-dev)}
args  = Namespace: input, style_image, strength, steps, guidance

Strategy: FLUX Redux prior + FLUX.1-dev img2img.

  Stage 1 — FluxPriorReduxPipeline encodes style_image into visual embeddings
             (prompt_embeds + pooled_prompt_embeds). These capture colour,
             texture, and style far more faithfully than a text description.

  Stage 2 — FluxImg2ImgPipeline (FLUX.1-dev) uses those embeddings as the
             conditioning signal while running img2img on the source image.
             The source composition is preserved via strength; only the regions
             that deviate from the reference style are nudged toward it.

Why img2img instead of pure generation (as in the original Redux spec):
  Pure FluxPipeline with Redux embeddings produces images visually close to
  the reference but loses the source image's pose, layout, and scene structure
  entirely. FluxImg2ImgPipeline keeps the source image as an anchor, so the
  character's pose and the scene's composition are preserved while style,
  lighting, and texture transfer from the reference.

Text prompt is NOT used — Redux visual embeddings replace the text signal.
strength=0.70 is the recommended starting point: enough to inject style
without collapsing the source composition. Lower (0.5) = more source
preserved; higher (0.85) = stronger style transfer.
"""

import logging
from PIL import Image

log = logging.getLogger(__name__)


def run(pipe: dict, config, args) -> Image.Image:
    image       = Image.open(args.input).convert("RGB")
    style_image = Image.open(args.style_image).convert("RGB")
    orig_size   = image.size

    from img2img.io_utils import snap_to_flux
    image = snap_to_flux(image)

    # Stage 1: encode style reference → visual embeddings
    log.info("[ip_adapter] Encoding style reference with FLUX Redux prior...")
    prior_out            = pipe["prior"](style_image)
    prompt_embeds        = prior_out.prompt_embeds
    pooled_prompt_embeds = prior_out.pooled_prompt_embeds

    # Stage 2: img2img conditioned on Redux embeddings
    d        = config.DEFAULTS["ip_adapter"]
    strength = getattr(args, "strength", None) or d["strength"]
    log.info(f"[ip_adapter] FLUX.1-dev img2img (strength={strength})...")
    result = pipe["flux"](
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        image=image,
        strength=strength,
        num_inference_steps=getattr(args, "steps", None) or d["steps"],
        guidance_scale=getattr(args, "guidance", None) or d["guidance"],
    ).images[0]

    result = result.resize(orig_size, Image.LANCZOS)
    log.info("IP-Adapter (FLUX Redux + img2img) done.")
    return result
