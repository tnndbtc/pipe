# code/ai/img2img/pipelines/composite.py
"""
Replace gen_composite_image.py.

run(pipe, config, args) where:
  pipe  = {"depth": depth_pipe | None, "flux": flux_img2img_pipe | None}
  args  = Namespace with: bg, characters, char_x, char_y, blend_strength,
          depth_scale, fg_scale, bg_scale, output

Depth-aware scaling:
  Depth map is computed ONCE per background and reused for all characters.
  Each character is sampled at its own (char_x, char_y) foot position — not
  a hardcoded value — so multiple characters at different positions scale
  correctly relative to each other.

blend_strength=0.0 → pure Pillow composite (no GPU beyond depth model).
blend_strength>0.0 → composite then FLUX.1-schnell img2img refinement at given strength.

Layer ordering:
  Characters are sorted back-to-front by char_y (smaller y = higher on screen
  = further away = rendered first). This is automatic; caller does not need to
  pre-sort the --characters list.
"""

from PIL import Image
import numpy as np
import logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Depth helpers
# ---------------------------------------------------------------------------

def _build_depth_map(depth_pipe, bg_rgb: Image.Image) -> np.ndarray:
    """
    Run Depth Anything V2 on the background. Returns a float32 array
    normalised 0-1 (1 = closest to camera), shaped (h, w), resized to
    match the background dimensions.
    """
    result    = depth_pipe(bg_rgb)
    depth_raw = np.array(result["depth"], dtype=np.float32)

    d_min, d_max = depth_raw.min(), depth_raw.max()
    if d_max > d_min:
        norm = (depth_raw - d_min) / (d_max - d_min)
    else:
        norm = np.zeros_like(depth_raw)

    bw, bh = bg_rgb.size
    if norm.shape != (bh, bw):
        tmp  = Image.fromarray((norm * 255).astype(np.uint8))
        tmp  = tmp.resize((bw, bh), Image.BILINEAR)
        norm = np.array(tmp, dtype=np.float32) / 255.0

    return norm


def _sample_depth_scale(depth_norm: np.ndarray, bg_w: int, bg_h: int,
                        char_x: float, char_y: float,
                        fg_scale: float, bg_scale: float) -> float:
    """
    Sample depth at (char_x, char_y) foot position and return a char_scale.
    depth_norm: 1 = foreground (large scale), 0 = background (small scale).
    Formula matches gen_composite_image.py exactly.
    """
    sx = max(0, min(int(bg_w * char_x), bg_w - 1))
    sy = max(0, min(int(bg_h * char_y), bg_h - 1))
    d  = float(depth_norm[sy, sx])
    scale = bg_scale + (fg_scale - bg_scale) * d
    log.info(f"  depth foot=({sx},{sy}) depth={d:.3f} -> scale={scale:.3f}")
    return scale


# ---------------------------------------------------------------------------
# Overlap helpers
# ---------------------------------------------------------------------------

def _boxes_overlap(b1: tuple, b2: tuple) -> bool:
    return not (b1[2] <= b2[0] or b2[2] <= b1[0] or
                b1[3] <= b2[1] or b2[3] <= b1[1])


def _nudge_no_overlap(paste_x: int, paste_y: int, w: int, h: int,
                      placed: list, bg_w: int, step: int = 4) -> int:
    """Shift paste_x horizontally until bbox doesn't overlap any placed box."""
    bbox = (paste_x, paste_y, paste_x + w, paste_y + h)
    if not any(_boxes_overlap(bbox, p) for p in placed):
        return paste_x
    for delta in range(step, bg_w, step):
        for direction in (+1, -1):
            nx    = max(0, min(paste_x + direction * delta, bg_w - w))
            new_b = (nx, paste_y, nx + w, paste_y + h)
            if not any(_boxes_overlap(new_b, p) for p in placed):
                log.info(f"  overlap nudged {direction * delta:+d}px")
                return nx
    log.warning(f"  could not resolve overlap — using original x={paste_x}")
    return paste_x


# ---------------------------------------------------------------------------
# Position / scale helpers
# ---------------------------------------------------------------------------

def _auto_x_positions(n: int) -> list:
    if n == 1:
        return [0.5]
    return [round((i + 1) / (n + 1), 3) for i in range(n)]


def _fill(provided, n: int, default) -> list:
    provided = list(provided or [])
    while len(provided) < n:
        provided.append(default)
    return provided[:n]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(pipe: dict, config, args) -> Image.Image:
    if not args.characters:
        raise ValueError(
            "--characters is required for composite mode. "
            "Pass one or more RGBA PNG paths."
        )

    bg     = Image.open(args.bg).convert("RGBA")
    bg_rgb = bg.convert("RGB")
    bw, bh = bg.size

    n       = len(args.characters)
    fg_scale = getattr(args, "fg_scale", 0.70)
    bg_scale = getattr(args, "bg_scale", 0.15)

    # Resolve per-character positions
    provided_xs = getattr(args, "char_x", None) or []
    provided_ys = getattr(args, "char_y", None) or []
    auto_xs     = _auto_x_positions(n)
    char_xs     = [(provided_xs[i] if i < len(provided_xs) else auto_xs[i]) for i in range(n)]
    char_ys     = _fill(provided_ys, n, 0.92)

    # Build depth map once (reused for all characters)
    depth_norm = None
    if args.depth_scale and pipe.get("depth"):
        log.info("Computing depth map...")
        depth_norm = _build_depth_map(pipe["depth"], bg_rgb)

    # Build per-character dicts
    chars = []
    for i, path in enumerate(args.characters):
        char_x = char_xs[i]
        char_y = char_ys[i]

        if depth_norm is not None:
            scale = _sample_depth_scale(depth_norm, bw, bh,
                                        char_x, char_y, fg_scale, bg_scale)
        else:
            scale = getattr(args, "char_scale", None) or 0.50

        chars.append({
            "img":     Image.open(path).convert("RGBA"),
            "char_x":  char_x,
            "char_y":  char_y,
            "scale":   scale,
        })

    # Sort back-to-front: smaller char_y = higher on screen = further = render first
    chars.sort(key=lambda c: c["char_y"])

    # Composite each character
    canvas       = bg.copy()
    placed_boxes = []

    for c in chars:
        img    = c["img"]
        target_h = int(bh * c["scale"])
        ratio    = target_h / img.size[1]
        target_w = int(img.size[0] * ratio)
        img      = img.resize((target_w, target_h), Image.LANCZOS)

        paste_x = int(bw * c["char_x"]) - target_w // 2
        paste_y = int(bh * c["char_y"]) - target_h
        paste_x = max(0, min(paste_x, bw - target_w))
        paste_y = max(0, min(paste_y, bh - target_h))

        paste_x = _nudge_no_overlap(paste_x, paste_y, target_w, target_h,
                                    placed_boxes, bw)

        log.info(f"  paste ({paste_x},{paste_y}) size {target_w}x{target_h} "
                 f"scale={c['scale']:.3f} x={c['char_x']} y={c['char_y']}")

        canvas.alpha_composite(img, dest=(paste_x, paste_y))
        placed_boxes.append((paste_x, paste_y, paste_x + target_w, paste_y + target_h))

    log.info(f"Pillow composite done ({n} character(s)).")

    # Optional FLUX schnell blend pass
    blend = getattr(args, "blend_strength", 0.0) or 0.0
    if blend > 0.0 and pipe.get("flux"):
        log.info(f"FLUX blend pass (strength={blend})...")
        composite_rgb = canvas.convert("RGB")
        result = pipe["flux"](
            prompt=getattr(args, "prompt", "photorealistic historical scene"),
            image=composite_rgb,
            strength=blend,
            num_inference_steps=4,
            guidance_scale=0.0,
        ).images[0]
        result_rgba = result.convert("RGBA")
        result_rgba.putalpha(canvas.split()[3])
        canvas = result_rgba
        log.info("FLUX blend pass done.")

    return canvas
