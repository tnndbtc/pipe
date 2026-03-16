# code/ai/img2img/pipelines/inpaint.py
"""
pipe  = SDXL Inpainting pipeline (load_sdxl_inpaint) — SUPERSEDED by inpaint_flux.py
args  = Namespace: input, mask, prompt, strength, steps, guidance, output

Strategy: patch-based inpainting with prefill.
  1. Find tight bbox of the mask region.
  2. Crop a patch (bbox + 96px context) from the original image.
  3. Prefill the masked area in the patch using column-from-above texture
     so SDXL refines a chair-free base rather than generating from scratch.
  4. Scale patch so longest side = 1024 (SDXL native resolution).
  5. Run SDXL inpaint at low strength — preserves texture, removes hallucination.
  6. Resize result back to original patch size.
  7. Composite back into the original image with feathered mask (seamless blend).
"""

import logging

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def _opencv_prefill(image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Fill the masked region by cloning pixels from just outside the mask boundary.

    Strategy:
      - For each column in the masked region, copy a strip of pixels from
        immediately ABOVE the mask and tile them downward.
      - Works well for architectural backgrounds (walls, floors) where the
        column above the object contains matching texture.
      - Falls back to a left-strip reference when there is insufficient
        above-content (mask near top edge).
    """
    img_np  = np.array(image.convert("RGB"))
    mask_np = np.array(mask)
    binary  = mask_np > 128

    if not binary.any():
        return image

    ys, xs = np.where(binary)
    mx1, mx2 = int(xs.min()), int(xs.max())
    my1, my2 = int(ys.min()), int(ys.max())

    result  = img_np.copy()
    fill_h  = my2 - my1 + 1

    for x in range(mx1, mx2 + 1):
        col_mask = binary[my1: my2 + 1, x]
        if not col_mask.any():
            continue

        ref_top_rows = my1
        ref_margin   = 50
        ref_avail    = max(0, ref_top_rows - ref_margin)

        if ref_avail >= 20:
            ref = img_np[ref_margin: ref_top_rows, x].astype(np.float32)
        else:
            ref_x = max(0, mx1 - 150)
            ref   = img_np[my1: my2 + 1, ref_x].astype(np.float32)

        repeats = (fill_h // max(len(ref), 1)) + 1
        tiled   = np.tile(ref, (repeats, 1))[:fill_h]

        for c in range(3):
            result[my1: my2 + 1, x, c] = np.where(
                col_mask,
                tiled[:, c].clip(0, 255).astype(np.uint8),
                result[my1: my2 + 1, x, c],
            )

    # Very light seam smoothing only
    blurred = cv2.GaussianBlur(result, (5, 5), 1)
    for c in range(3):
        result[:, :, c] = np.where(binary, blurred[:, :, c], result[:, :, c])

    return Image.fromarray(result)


def run(pipe, config, args) -> Image.Image:
    image = Image.open(args.input).convert("RGB")
    mask  = Image.open(args.mask).convert("L")   # white = fill

    orig_size = image.size
    W, H = orig_size

    # --- Find tight bbox of the masked region ---
    mask_np = np.array(mask)
    binary  = mask_np > 128

    if not binary.any():
        log.warning("[inpaint] Empty mask — returning original image")
        return image

    ys, xs = np.where(binary)
    bx1, bx2 = int(xs.min()), int(xs.max())
    by1, by2 = int(ys.min()), int(ys.max())

    # --- Feather mask edges for seamless compositing ---
    feather = 10
    mask_feathered_np = cv2.GaussianBlur(
        mask_np.astype(np.float32),
        (feather * 2 + 1, feather * 2 + 1),
        feather / 3,
    )
    mask_feathered_np = np.clip(mask_feathered_np, 0, 255).astype(np.uint8)
    mask_feathered    = Image.fromarray(mask_feathered_np)

    # --- Crop patch: bbox + surrounding context ---
    ctx = 96
    px1 = max(0, bx1 - ctx)
    py1 = max(0, by1 - ctx)
    px2 = min(W, bx2 + ctx)
    py2 = min(H, by2 + ctx)

    patch_img          = image.crop((px1, py1, px2, py2))
    patch_mask_binary  = Image.fromarray(mask_np[py1:py2, px1:px2])
    patch_mask_feather = mask_feathered.crop((px1, py1, px2, py2))

    log.info(
        f"[inpaint] Patch: ({px1},{py1})→({px2},{py2})  "
        f"{px2-px1}×{py2-py1}px  "
        f"mask={binary.sum() / (W * H):.1%} of image"
    )

    # --- Prefill the patch (gives SDXL a chair-free base, enables low strength) ---
    if not getattr(args, "no_prefill", False):
        log.info("[inpaint] Pre-fill patch → removing object texture before SDXL refinement")
        patch_img = _opencv_prefill(patch_img, patch_mask_binary)

        if getattr(args, "prefill_only", False):
            log.info("[inpaint] --prefill-only; skipping SDXL")
            result = image.copy()
            result.paste(patch_img, (px1, py1), patch_mask_feather)
            return result

    # --- Scale patch so longest side = 1024 for SDXL ---
    pw, ph    = patch_img.size
    scale     = 1024 / max(pw, ph)
    nw        = max(8, int(pw * scale) // 8 * 8)
    nh        = max(8, int(ph * scale) // 8 * 8)
    patch_sdxl      = patch_img.resize((nw, nh), Image.LANCZOS)
    patch_mask_sdxl = patch_mask_feather.resize((nw, nh), Image.LANCZOS)

    log.info(f"[inpaint] SDXL patch size: {patch_sdxl.size}")

    # --- SDXL inpaint on patch ---
    d = config.DEFAULTS["inpaint"]
    result_patch = pipe(
        prompt=args.prompt,
        negative_prompt=getattr(args, "negative_prompt", None) or config.SDXL_NEGATIVE_PROMPT,
        image=patch_sdxl,
        mask_image=patch_mask_sdxl,
        strength=getattr(args, "strength", None) or d["strength"],
        num_inference_steps=getattr(args, "steps", None) or d["steps"],
        guidance_scale=getattr(args, "guidance", None) or d["guidance"],
    ).images[0]

    # --- Resize result back to original patch dimensions ---
    result_patch_full = result_patch.resize((px2 - px1, py2 - py1), Image.LANCZOS)

    # --- Composite patch back using feathered mask ---
    result = image.copy()
    result.paste(result_patch_full, (px1, py1), patch_mask_feather)

    log.info(f"Inpaint done → {orig_size}")
    return result
