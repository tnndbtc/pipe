# code/ai/gen_img2img.py
"""
Main entry point for gen_img2img utility.
Replaces gen_composite_image.py.

Usage: python gen_img2img.py --mode <mode> [mode-specific args]
       python gen_img2img.py --mode <mode> --manifest items.json --output-dir out/

See PART 2 of the plan for full CLI reference.
"""

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path

import torch

# Ensure code/ai/ is on path when called from project root
sys.path.insert(0, str(Path(__file__).parent))

from img2img import config as cfg
from img2img.io_utils import save_image, ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("gen_img2img")

# --- Pipeline dispatch table ---
PIPELINE_MODES = {
    "composite":    "img2img.pipelines.composite",
    "inpaint":      "img2img.pipelines.inpaint",
    "outpaint":     "img2img.pipelines.outpaint",
    "style":        "img2img.pipelines.style_transfer",
    "pose":         "img2img.pipelines.pose_control",
    "depth":        "img2img.pipelines.depth_control",
    "canny":        "img2img.pipelines.canny_control",
    "upscale":      "img2img.pipelines.upscale_diffusion",
    "bg_replace":   "img2img.pipelines.bg_replace",
    "ip_adapter":   "img2img.pipelines.ip_adapter",
    "relight":      "img2img.pipelines.relight",
    "color_grade":  "img2img.pipelines.color_grade",
}


def _validate_args(mode: str, args) -> None:
    """Raise ValueError with a clear message if a required mode-specific arg is missing.
    Called before model loading so the user is not charged 30-60s of VRAM allocation
    for a missing --mask or --characters."""
    _req = {
        "composite":   [("bg",          "--bg"),
                        ("characters",  "--characters")],
        "inpaint":     [("input",       "--input"),
                        ("mask",        "--mask"),
                        ("prompt",      "--prompt")],
        "outpaint":    [("input",       "--input"),
                        ("prompt",      "--prompt")],
        "style":       [("input",       "--input"),
                        ("prompt",      "--prompt")],
        "pose":        [("input",       "--input"),
                        ("prompt",      "--prompt")],
        "depth":       [("input",       "--input"),
                        ("prompt",      "--prompt")],
        "canny":       [("input",       "--input"),
                        ("prompt",      "--prompt")],
        "upscale":     [("input",       "--input")],
        "bg_replace":  [("input",       "--input"),
                        ("bg",          "--bg")],
        "ip_adapter":  [("input",       "--input"),
                        ("style_image", "--style-image"),
                        ("prompt",      "--prompt")],
        "relight":     [("fg",          "--fg"),
                        ("bg",          "--bg")],
        "color_grade": [("input",       "--input"),
                        ("reference",   "--reference")],
    }
    for attr, flag in _req.get(mode, []):
        val = getattr(args, attr, None)
        if not val:
            raise ValueError(f"[{mode}] missing required argument: {flag}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Image-to-image utility (gen_img2img)")
    p.add_argument("--mode",          required=True, choices=list(PIPELINE_MODES))
    p.add_argument("--input",         help="Input image path")
    p.add_argument("--bg",            help="Background image (composite, bg_replace)")
    p.add_argument("--fg",            help="Foreground image (relight)")
    p.add_argument("--characters",    nargs="+", help="Character RGBA PNGs in back-to-front order (farther characters first) (composite mode)")
    p.add_argument("--mask",          help="Inpaint mask (inpaint)")
    p.add_argument("--pose-image",    help="Pose reference image (pose)")
    p.add_argument("--style-image",   help="Style reference image (ip_adapter)")
    p.add_argument("--reference",     help="Color reference image (color_grade)")
    p.add_argument("--prompt",        default="", help="Text prompt")
    p.add_argument("--output",        help="Output path (single image)")
    p.add_argument("--output-dir",    help="Output directory (batch mode)")
    p.add_argument("--manifest",      help="JSON manifest for batch mode")
    p.add_argument("--steps",         type=int)
    p.add_argument("--guidance",      type=float)
    p.add_argument("--strength",      type=float)
    p.add_argument("--blend-strength",type=float, dest="blend_strength")
    p.add_argument("--depth-scale",   action="store_true", dest="depth_scale")
    p.add_argument("--no-depth-scale",action="store_false", dest="depth_scale")
    p.add_argument("--scale",         type=int, help="Upscale factor (upscale mode): 2 or 4 (default 4)")
    p.add_argument("--direction",     choices=["left","right","up","down","all"])
    p.add_argument("--pixels",        type=int, help="Pixels to outpaint")
    p.add_argument("--ip-scale",      type=float, dest="ip_scale")
    p.add_argument("--mode-relight",  choices=["local","api"], dest="mode_relight")
    p.add_argument("--low-threshold", type=int, dest="low_threshold")
    p.add_argument("--high-threshold",type=int, dest="high_threshold")
    p.add_argument("--seed",          type=int)
    p.add_argument("--device",        default=cfg.DEVICE)
    p.add_argument("--no-fp16",       action="store_true", dest="no_fp16")
    # ⚠ OPEN: --no-fp16 is declared but _fp16_kwargs() reads module-level DEVICE only —
    # it does not read args.no_fp16. To take effect, _fp16_kwargs() must be changed to
    # accept a no_fp16: bool parameter and all load_*() functions must thread it through.
    # Until fixed, --no-fp16 is a no-op.
    p.add_argument("--verbose",       action="store_true")
    p.set_defaults(depth_scale=True)
    return p


def load_pipe_for_mode(mode: str, args):
    """Load the correct model(s) for the requested mode. Returns pipe object."""
    from img2img.model_loader import (
        load_sdxl_img2img, load_sdxl_inpaint, load_sdxl_controlnet,
        load_sd_upscaler, load_depth_model, load_ip_adapter,
    )

    if mode == "composite":
        pipe = {"depth": load_depth_model() if args.depth_scale else None}
        if getattr(args, "blend_strength", 0.0) and args.blend_strength > 0.0:
            pipe["sdxl"] = load_sdxl_img2img()
        return pipe

    elif mode in ("inpaint", "outpaint"):
        return load_sdxl_inpaint()

    elif mode == "style":
        return load_sdxl_img2img()

    elif mode == "pose":
        return load_sdxl_controlnet(cfg.MODEL_IDS["controlnet_pose"])

    elif mode == "depth":
        return {"sdxl_controlnet": load_sdxl_controlnet(cfg.MODEL_IDS["controlnet_depth"]),
                "depth": load_depth_model()}

    elif mode == "canny":
        return load_sdxl_controlnet(cfg.MODEL_IDS["controlnet_canny"])

    elif mode == "upscale":
        return load_sd_upscaler()

    elif mode == "bg_replace":
        pipe = {}
        if getattr(args, "blend_strength", 0.0) and args.blend_strength > 0.0:
            pipe["sdxl"] = load_sdxl_img2img()
        return pipe

    elif mode == "ip_adapter":
        return load_ip_adapter(load_sdxl_img2img())

    elif mode in ("relight", "color_grade"):
        return None  # no model; handled inside pipeline

    raise ValueError(f"Unknown mode: {mode}")


def unload_pipe(pipe) -> None:
    from img2img.model_loader import unload_model
    if isinstance(pipe, dict):
        for v in pipe.values():
            unload_model(v)
    else:
        unload_model(pipe)


def run_single(mode: str, pipe, args) -> None:
    import importlib
    module = importlib.import_module(PIPELINE_MODES[mode])

    if args.seed is not None:
        import torch
        torch.manual_seed(args.seed)

    result = module.run(pipe, cfg, args)

    out_path = args.output or _default_output(args, mode)
    save_image(result, out_path)
    log.info(f"[{mode}] Done → {out_path}")


def _default_output(args, mode: str) -> str:
    src = getattr(args, "input", None) or getattr(args, "bg", "output")
    stem = Path(src).stem
    out_dir = cfg.OUTPUT_DIRS.get(mode, "assets/img2img")
    return str(Path(out_dir) / f"{stem}_{mode}.png")


def run_batch(mode: str, pipe, args) -> None:
    with open(args.manifest) as f:
        items = json.load(f)
    out_dir = ensure_dir(args.output_dir or cfg.OUTPUT_DIRS.get(mode, "out"))
    log.info(f"Batch mode: {len(items)} item(s), mode={mode}, output={out_dir}")

    for idx, item in enumerate(items):
        item_id = item.get("item_id") or f"item_{idx:04d}"
        try:
            for k, v in item.items():
                setattr(args, k, v)
            args.output = str(out_dir / f"{item_id}.png")
            run_single(mode, pipe, args)
        except Exception as e:
            log.error(f"[{item_id}] Failed: {e}", exc_info=True)
            continue

    log.info(f"Batch complete: {len(items)} item(s) processed.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    log.info(f"Mode: {args.mode} | Device: {args.device}")
    log.info(f"VRAM profile: {cfg.VRAM_PROFILES[args.mode]}")

    _validate_args(args.mode, args)   # fail fast before model loading

    pipe = load_pipe_for_mode(args.mode, args)

    try:
        if args.manifest:
            run_batch(args.mode, pipe, args)
        else:
            run_single(args.mode, pipe, args)
    finally:
        unload_pipe(pipe)
        log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
