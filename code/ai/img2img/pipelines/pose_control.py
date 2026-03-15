# code/ai/img2img/pipelines/pose_control.py
"""
pipe  = SDXL + ControlNet-OpenPose pipeline
args  = Namespace: input, pose_image (optional), prompt, strength

If args.pose_image is set: use it directly as conditioning image.
Else: extract pose from args.input using controlnet_aux DwposeDetector.

controlnet_aux is a new dependency: add controlnet_aux>=0.0.7 to requirements.txt
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

    from img2img.io_utils import resize_to_sdxl
    image = resize_to_sdxl(image)

    if getattr(args, "pose_image", None):
        control_image = Image.open(args.pose_image).convert("RGB").resize(image.size)
    else:
        control_image = _extract_pose(image)

    result = pipe(
        prompt=args.prompt,
        negative_prompt=config.NEGATIVE_PROMPT,
        image=image,
        control_image=control_image,
        controlnet_conditioning_scale=0.8,
        strength=getattr(args, "strength", config.DEFAULTS["pose"]["strength"]),
        num_inference_steps=getattr(args, "steps", config.DEFAULTS["pose"]["steps"]),
        guidance_scale=getattr(args, "guidance", config.DEFAULTS["pose"]["guidance"]),
    ).images[0]

    result = result.resize(orig_size, Image.LANCZOS)
    log.info("Pose ControlNet done.")
    return result
