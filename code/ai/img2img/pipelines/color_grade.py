# code/ai/img2img/pipelines/color_grade.py
"""
pipe  = None
args  = Namespace: input, reference, output

Applies Reinhard (2001) color transfer:
  Match mean and std of each LAB channel from reference → target.
  Fast, deterministic, no model, works well for matching scene mood.
"""

import cv2
import logging
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def _reinhard_transfer(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Both inputs are uint8 RGB. Returns uint8 RGB."""
    src_lab = cv2.cvtColor(source,    cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB).astype(np.float32)

    for ch in range(3):
        src_mean, src_std = src_lab[:, :, ch].mean(), src_lab[:, :, ch].std()
        ref_mean, ref_std = ref_lab[:, :, ch].mean(), ref_lab[:, :, ch].std()
        if src_std < 1e-6:
            continue
        src_lab[:, :, ch] = (src_lab[:, :, ch] - src_mean) * (ref_std / src_std) + ref_mean

    src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(src_lab, cv2.COLOR_LAB2RGB)


def run(pipe, config, args) -> Image.Image:
    source    = np.array(Image.open(args.input).convert("RGB"))
    reference = np.array(Image.open(args.reference).convert("RGB"))

    # Resize reference to source for consistent channel stats
    if reference.shape != source.shape:
        h, w = source.shape[:2]
        reference = cv2.resize(reference, (w, h), interpolation=cv2.INTER_AREA)

    result_np = _reinhard_transfer(source, reference)
    result = Image.fromarray(result_np)
    log.info("Reinhard color grade applied.")
    return result
