# code/ai/img2img/pipelines/composite.py
"""
Replace gen_composite_image.py.

run(pipe, config, args) where:
  pipe  = {"depth": depth_pipe, "sdxl": sdxl_img2img_pipe | None}
  args  = Namespace with: bg, characters, blend_strength, depth_scale, output

Depth-aware scaling logic ported from gen_composite_image.py (unchanged).
blend_strength=0.0 → pure Pillow composite (same as old script, no GPU beyond depth).
blend_strength>0.0 → composite then SDXL img2img refinement at given strength.
"""

from PIL import Image
import numpy as np
import logging

log = logging.getLogger(__name__)


def _get_depth_scale(depth_map: np.ndarray, foot_y_norm: float,
                     fg_scale: float = 0.70, bg_scale: float = 0.15) -> float:
    """Interpolate character scale from depth value at foot position (0–1 normalised)."""
    depth_at_foot = float(depth_map[int(foot_y_norm * depth_map.shape[0]),
                                    depth_map.shape[1] // 2])
    # depth=0 → foreground (large); depth=1 → background (small)
    return fg_scale + (bg_scale - fg_scale) * depth_at_foot


def _detect_overlap(boxes: list[tuple]) -> list[tuple]:
    """Return list of (i,j) pairs that overlap. box = (x1,y1,x2,y2)."""
    overlapping = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ax1, ay1, ax2, ay2 = boxes[i]
            bx1, by1, bx2, by2 = boxes[j]
            if ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1:
                overlapping.append((i, j))
    return overlapping


def run(pipe: dict, config, args) -> Image.Image:
    if not args.characters:
        raise ValueError(
            "--characters is required for composite mode. "
            "Pass one or more RGBA PNG paths, e.g. --characters char1.png char2.png"
        )

    bg = Image.open(args.bg).convert("RGBA")
    bw, bh = bg.size

    # Characters must be passed in back-to-front order (farther characters first).
    # The caller controls layer order; no automatic sort is applied.
    chars = [Image.open(c).convert("RGBA") for c in args.characters]

    depth_map = None
    if args.depth_scale and pipe.get("depth"):
        result = pipe["depth"](bg.convert("RGB"))
        depth_map = np.array(result["depth"]) / 255.0
        log.info("Depth map obtained for scaling.")

    positions = []
    boxes = []

    for idx, char in enumerate(chars):
        # Determine paste x centre: evenly spaced
        cx = int(bw * (idx + 1) / (len(chars) + 1))
        cy = bh  # anchor bottom of character to bottom of frame

        # Scale character
        if depth_map is not None:
            scale = _get_depth_scale(depth_map, foot_y_norm=0.85)
            target_h = int(bh * scale)
        else:
            target_h = int(bh * 0.50)  # auto-scale fallback

        ratio = target_h / char.size[1]
        target_w = int(char.size[0] * ratio)
        char_resized = char.resize((target_w, target_h), Image.LANCZOS)

        px = cx - target_w // 2
        py = cy - target_h
        positions.append([px, py, char_resized])
        boxes.append((px, py, px + target_w, py + target_h))

    # Nudge overlapping characters horizontally (alternating directions)
    nudge = 20
    for _ in range(50):
        pairs = _detect_overlap(boxes)
        if not pairs:
            break
        for i, j in pairs:
            x1, y1, x2, y2 = boxes[i]
            dx = nudge if i % 2 == 0 else -nudge
            positions[i][0] += dx
            boxes[i] = (boxes[i][0] + dx, y1, boxes[i][2] + dx, y2)

    # Compose
    canvas = bg.copy()
    for px, py, ch in positions:
        canvas.alpha_composite(ch, dest=(px, py))

    log.info(f"Pillow composite done ({len(chars)} character(s)).")

    # Optional SDXL blend pass
    if args.blend_strength and args.blend_strength > 0.0 and pipe.get("sdxl"):
        sdxl = pipe["sdxl"]
        composite_rgb = canvas.convert("RGB")
        result = sdxl(
            prompt=getattr(args, "prompt", "photorealistic historical scene"),
            negative_prompt=config.NEGATIVE_PROMPT,
            image=composite_rgb,
            strength=args.blend_strength,
            num_inference_steps=20,
            guidance_scale=5.0,
        ).images[0]
        # Restore alpha from original composite
        result_rgba = result.convert("RGBA")
        result_rgba.putalpha(canvas.split()[3])
        canvas = result_rgba
        log.info(f"SDXL blend pass done (strength={args.blend_strength}).")

    return canvas
