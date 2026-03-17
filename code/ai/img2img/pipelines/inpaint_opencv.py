# code/ai/img2img/pipelines/inpaint_opencv.py
"""
pipe  = None  (no model required)
args  = Namespace: input, mask, output
        opencv_method: "telea" (default) | "ns"
        inpaint_radius: int (default 5)
        prompt is intentionally ignored.

Strategy: OpenCV classical inpainting — zero GPU, zero model download.
  Two algorithms, selectable via --opencv-method:

    telea  (default) — Fast Marching Method.
      Propagates texture from the mask boundary inward along shortest paths.
      Sharp edges, very fast (~50ms).  Good for thin/small objects.

    ns  — Navier-Stokes fluid dynamics.
      Smoothly blends surrounding gradients into the masked region.
      Softer result, slower.  Better for smooth gradients (sky, fog).

  Best for: simple uniform textures, small objects, fast previews.
  Falls short of LaMa for large or complex regions — use --engine lama there.
  No GPU or internet connection required.

  Prompt is intentionally ignored.
"""

import logging

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

_CV_FLAGS = {
    "telea": cv2.INPAINT_TELEA,
    "ns":    cv2.INPAINT_NS,
}


def run(pipe, config, args) -> Image.Image:
    image = Image.open(args.input).convert("RGB")
    mask  = Image.open(args.mask).convert("L")   # white = fill region

    orig_size = image.size
    W, H = orig_size

    img_np  = np.array(image)
    mask_np = np.array(mask)

    # Hard binary mask for cv2.inpaint
    binary = (mask_np > 128).astype(np.uint8) * 255

    method = getattr(args, "opencv_method", "telea")
    flag   = _CV_FLAGS.get(method, cv2.INPAINT_TELEA)
    radius = getattr(args, "inpaint_radius", 5)

    coverage = binary.sum() / (W * H * 255)
    log.info(f"[inpaint-opencv] method={method}  radius={radius}  coverage={coverage:.1%}")

    result_np = cv2.inpaint(img_np, binary, radius, flag)

    # Feather composite edge for seamless blending
    feather = 10
    mask_feathered_np = cv2.GaussianBlur(
        mask_np.astype(np.float32),
        (feather * 2 + 1, feather * 2 + 1),
        feather / 3,
    )
    mask_feathered_np = np.clip(mask_feathered_np, 0, 255).astype(np.uint8)

    alpha  = mask_feathered_np.astype(np.float32) / 255.0
    out_np = (
        result_np * alpha[:, :, None] +
        img_np    * (1 - alpha[:, :, None])
    ).astype(np.uint8)

    log.info(f"Inpaint (OpenCV/{method}) done → {orig_size}")
    return Image.fromarray(out_np)
