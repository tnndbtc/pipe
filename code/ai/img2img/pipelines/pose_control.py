# code/ai/img2img/pipelines/pose_control.py
"""
pipe  = FLUX.1-dev + InstantX ControlNet-Pose pipeline (load_flux_controlnet)
args  = Namespace: input, pose_image (optional), prompt, strength, steps, guidance

Uses FluxControlNetImg2ImgPipeline: the source image is preserved (img2img)
while the pose skeleton conditions the generated content.

If args.pose_image is set: use it directly as conditioning image.
Else: extract pose from args.input using controlnet_aux DwposeDetector.

controlnet_aux dependency: controlnet_aux>=0.0.7
"""

import logging
from PIL import Image

log = logging.getLogger(__name__)


def _extract_pose(image: Image.Image) -> Image.Image:
    from controlnet_aux import DwposeDetector
    detector = DwposeDetector.from_pretrained("yzd-v/DWPose")
    pose_image = detector(image)
    log.info("DwposeDetector extracted pose map.")
    return pose_image


def run(pipe, config, args) -> Image.Image:
    image = Image.open(args.input).convert("RGB")
    orig_size = image.size

    from img2img.io_utils import snap_to_flux
    image = snap_to_flux(image)

    if getattr(args, "pose_image", None):
        control_image = Image.open(args.pose_image).convert("RGB").resize(image.size)
    else:
        control_image = _extract_pose(image)
        control_image = control_image.resize(image.size)

    d = config.DEFAULTS["pose"]
    result = pipe(
        prompt=args.prompt,
        image=image,
        control_image=control_image,
        controlnet_conditioning_scale=0.7,
        strength=getattr(args, "strength", None) or d["strength"],
        num_inference_steps=getattr(args, "steps", None) or d["steps"],
        guidance_scale=getattr(args, "guidance", None) or d["guidance"],
    ).images[0]

    result = result.resize(orig_size, Image.LANCZOS)
    log.info("Pose ControlNet (FLUX) done.")
    return result
