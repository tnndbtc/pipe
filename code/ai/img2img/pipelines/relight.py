# code/ai/img2img/pipelines/relight.py
"""
pipe  = None (handled internally based on mode_relight)
args  = Namespace: fg, bg, mode_relight ("local"|"api")

mode_relight="local":
  Uses IC-Light v2 weights via diffusers (ICLightPipeline or custom loading).
  Model: lllyasviel/ic-light — must be downloaded manually (not on HF hub publicly).
  ~2 GB VRAM.

mode_relight="api":
  Calls fal.ai API (same as gen_composite_image.py --mode iclight).
  Requires FAL_KEY environment variable.
  Zero local VRAM.
"""

import logging
import os
from PIL import Image

log = logging.getLogger(__name__)


def _relight_api(fg: Image.Image, bg: Image.Image) -> Image.Image:
    import fal_client
    import base64, io

    def _encode(img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    result = fal_client.run(
        "fal-ai/iclight/v2",
        arguments={
            "foreground_image": _encode(fg),
            "background_image": _encode(bg),
        },
    )
    img_url = result["images"][0]["url"]
    import urllib.request
    with urllib.request.urlopen(img_url) as resp:
        return Image.open(resp).convert("RGBA")


def _relight_local(fg: Image.Image, bg: Image.Image) -> Image.Image:
    # Placeholder: IC-Light local inference via diffusers custom pipeline.
    # Requires ic_light_weights_path set in config.py once weights are obtained.
    # Implementation follows lllyasviel/IC-Light ComfyUI workflow:
    #   1. Composite fg onto bg naively.
    #   2. Run ICLight diffusion pass conditioned on fg RGBA + bg RGB.
    #   3. Return relit composite.
    raise NotImplementedError(
        "IC-Light local mode requires weights from lllyasviel/IC-Light repo. "
        "Set mode_relight='api' to use fal.ai instead."
    )


def run(pipe, config, args) -> Image.Image:
    fg = Image.open(args.fg).convert("RGBA")
    bg = Image.open(args.bg).convert("RGBA")
    mode = getattr(args, "mode_relight", config.DEFAULTS["relight"]["mode_relight"])

    if mode == "api":
        log.info("Relighting via fal.ai IC-Light API...")
        result = _relight_api(fg, bg)
    else:
        log.info("Relighting via local IC-Light...")
        result = _relight_local(fg, bg)

    log.info("Relight done.")
    return result
