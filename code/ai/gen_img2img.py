# code/ai/gen_img2img.py
"""
Main entry point for the img2img utility.
Replaces gen_composite_image.py for composite mode.

Usage:
    python gen_img2img.py --mode composite --bg bg.png --characters char1.png char2.png
    python gen_img2img.py --mode composite --bg bg.png --characters char1.png --char-x 0.3
    python gen_img2img.py --mode composite --manifest AssetManifest.json --output-dir out/
    python gen_img2img.py --mode inpaint   --input img.png --mask mask.png --prompt "..."
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from img2img import config as cfg
from img2img.io_utils import save_image, ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("gen_img2img")

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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_args(mode: str, args) -> None:
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


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Image-to-image utility (gen_img2img)")
    p.add_argument("--mode", required=True, choices=list(PIPELINE_MODES))

    # --- Inputs ---
    p.add_argument("--input",        help="Input image path")
    p.add_argument("--bg",           help="Background image (composite, bg_replace, relight)")
    p.add_argument("--fg",           help="Foreground image (relight)")
    p.add_argument("--characters",   nargs="+",
                   help="Character RGBA PNGs (composite). Repeat --char-x/--char-y once per character.")
    p.add_argument("--mask",         help="Inpaint mask")
    p.add_argument("--pose-image",   help="Pose reference image (pose)")
    p.add_argument("--style-image",  help="Style reference image (ip_adapter)")
    p.add_argument("--reference",    help="Color reference image (color_grade)")
    p.add_argument("--prompt",       default="", help="Text prompt")

    # --- Per-character positioning (composite mode) ---
    pos = p.add_argument_group("per-character positioning (composite mode)")
    pos.add_argument("--char-x", dest="char_x", type=float, action="append", metavar="X",
                     help="Horizontal anchor 0-1 per character. Repeat once per --characters entry. "
                          "Missing values are auto-spaced.")
    pos.add_argument("--char-y", dest="char_y", type=float, action="append", metavar="Y",
                     help="Vertical anchor for character bottom 0-1 per character. Default: 0.92.")

    # --- Scaling (composite mode) ---
    scale = p.add_argument_group("depth scaling (composite mode)")
    scale.add_argument("--depth-scale",    action="store_true",  dest="depth_scale")
    scale.add_argument("--no-depth-scale", action="store_false", dest="depth_scale")
    scale.add_argument("--fg-scale",  type=float, default=0.70, dest="fg_scale",
                       help="Character size at foreground depth (default: 0.70)")
    scale.add_argument("--bg-scale",  type=float, default=0.15, dest="bg_scale",
                       help="Character size at background depth (default: 0.15)")
    scale.add_argument("--char-scale", type=float, default=None, dest="char_scale",
                       help="Manual fallback scale when depth is disabled")
    scale.add_argument("--blend-strength", type=float, dest="blend_strength",
                       help="SDXL refinement blend strength 0-1 (0 = Pillow only)")
    p.set_defaults(depth_scale=True)

    # --- Output ---
    p.add_argument("--output",     help="Output path (single image)")
    # Accept both --output-dir and --output_dir for server.py compatibility
    p.add_argument("--output-dir", "--output_dir", dest="output_dir",
                   help="Output directory (batch mode)")

    # --- Batch mode ---
    p.add_argument("--manifest",  help="JSON manifest or AssetManifest for batch mode")
    p.add_argument("--asset-id",  dest="asset_id", default=None,
                   help="Process only this asset_id (batch / manifest mode)")

    # --- Generation params ---
    p.add_argument("--steps",          type=int)
    p.add_argument("--guidance",       type=float)
    p.add_argument("--strength",       type=float)
    p.add_argument("--scale",          type=int,
                   help="Upscale factor (upscale mode): 2 or 4 (default 4)")
    p.add_argument("--direction",      choices=["left", "right", "up", "down", "all"])
    p.add_argument("--pixels",         type=int, help="Pixels to outpaint")
    p.add_argument("--ip-scale",       type=float, dest="ip_scale")
    p.add_argument("--mode-relight",   choices=["local", "api"], dest="mode_relight")
    p.add_argument("--low-threshold",  type=int, dest="low_threshold")
    p.add_argument("--high-threshold", type=int, dest="high_threshold")
    p.add_argument("--seed",           type=int)
    p.add_argument("--device",         default=cfg.DEVICE)
    p.add_argument("--no-fp16",        action="store_true", dest="no_fp16",
                   help="Disable FP16 (use FP32). Slower but useful for debugging.")
    p.add_argument("--verbose",        action="store_true")
    return p


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_pipe_for_mode(mode: str, args):
    from img2img.model_loader import (
        load_sdxl_img2img, load_sdxl_inpaint, load_sdxl_controlnet,
        load_sd_upscaler, load_depth_model, load_ip_adapter,
    )
    no_fp16 = getattr(args, "no_fp16", False)

    if mode == "composite":
        pipe = {"depth": load_depth_model() if args.depth_scale else None}
        if getattr(args, "blend_strength", None) and args.blend_strength > 0.0:
            pipe["sdxl"] = load_sdxl_img2img(no_fp16)
        return pipe

    elif mode in ("inpaint", "outpaint"):
        return load_sdxl_inpaint(no_fp16)

    elif mode == "style":
        return load_sdxl_img2img(no_fp16)

    elif mode == "pose":
        return load_sdxl_controlnet(cfg.MODEL_IDS["controlnet_pose"], no_fp16)

    elif mode == "depth":
        return {"sdxl_controlnet": load_sdxl_controlnet(cfg.MODEL_IDS["controlnet_depth"], no_fp16),
                "depth":           load_depth_model()}

    elif mode == "canny":
        return load_sdxl_controlnet(cfg.MODEL_IDS["controlnet_canny"], no_fp16)

    elif mode == "upscale":
        return load_sd_upscaler(no_fp16)

    elif mode == "bg_replace":
        pipe = {}
        if getattr(args, "blend_strength", None) and args.blend_strength > 0.0:
            pipe["sdxl"] = load_sdxl_img2img(no_fp16)
        return pipe

    elif mode == "ip_adapter":
        return load_ip_adapter(load_sdxl_img2img(no_fp16))

    elif mode in ("relight", "color_grade"):
        return None

    raise ValueError(f"Unknown mode: {mode}")


def unload_pipe(pipe) -> None:
    from img2img.model_loader import unload_model
    if isinstance(pipe, dict):
        for v in pipe.values():
            unload_model(v)
    else:
        unload_model(pipe)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(mode: str, pipe, args) -> None:
    import importlib
    module = importlib.import_module(PIPELINE_MODES[mode])

    if args.seed is not None:
        torch.manual_seed(args.seed)

    result  = module.run(pipe, cfg, args)
    out_path = args.output or _default_output(args, mode)
    save_image(result, out_path)
    log.info(f"[{mode}] Done → {out_path}")


def _default_output(args, mode: str) -> str:
    src     = getattr(args, "input", None) or getattr(args, "bg", "output")
    stem    = Path(src).stem
    out_dir = cfg.OUTPUT_DIRS.get(mode, "assets/img2img")
    return str(Path(out_dir) / f"{stem}_{mode}.png")


# ---------------------------------------------------------------------------
# AssetManifest → composite job list
# ---------------------------------------------------------------------------

def _load_asset_manifest_jobs(manifest: dict, out_dir: str,
                               asset_id_filter: str | None) -> list[dict]:
    """
    Convert an AssetManifest (character_packs + backgrounds) into a list of
    composite job dicts compatible with run_batch().
    Mirrors gen_composite_image.py's load_from_manifest() logic exactly.
    """
    bg_ids = [bg["asset_id"] for bg in manifest.get("backgrounds", [])]
    jobs   = []

    for char in manifest.get("character_packs", []):
        char_id   = char["asset_id"]
        char_file = str(Path(out_dir) / f"char-{char_id}-v1-rgba.png")
        ai_prompt = char.get("ai_prompt", "").lower()

        matched_bg = None
        for bg_id in bg_ids:
            keywords = bg_id.replace("bg-", "").split("-")
            if any(kw in ai_prompt for kw in keywords if len(kw) > 4):
                matched_bg = bg_id
                break
        if matched_bg is None and bg_ids:
            matched_bg = bg_ids[0]
        if matched_bg is None:
            continue

        asset_id = f"comp-{char_id}-{matched_bg.replace('bg-', '')}-v1"
        if asset_id_filter and asset_id != asset_id_filter:
            continue

        jobs.append({
            "item_id":    asset_id,
            "bg":         str(Path(out_dir) / f"{matched_bg}-v1.png"),
            "characters": [char_file],
            "char_x":     [0.5],
            "char_y":     [0.92],
        })

    return jobs


# ---------------------------------------------------------------------------
# Batch run
# ---------------------------------------------------------------------------

def run_batch(mode: str, pipe, args) -> None:
    with open(args.manifest, encoding="utf-8") as f:
        data = json.load(f)

    out_dir = ensure_dir(args.output_dir or cfg.OUTPUT_DIRS.get(mode, "out"))

    # Detect AssetManifest vs simple item list
    if isinstance(data, dict) and "character_packs" in data:
        if mode != "composite":
            log.warning(
                "AssetManifest passed but mode=%s — only 'composite' mode supports "
                "AssetManifest auto-pairing. Wrap your items in a JSON array instead.", mode
            )
            return
        items = _load_asset_manifest_jobs(data, str(out_dir),
                                          getattr(args, "asset_id", None))
        log.info(f"AssetManifest → {len(items)} composite job(s)")
    else:
        items = data if isinstance(data, list) else [data]

    log.info(f"Batch: {len(items)} item(s), mode={mode}, output={out_dir}")

    for idx, item in enumerate(items):
        item_id = item.get("item_id") or f"item_{idx:04d}"
        try:
            for k, v in item.items():
                if k != "item_id":
                    setattr(args, k, v)
            args.output = str(out_dir / f"{item_id}.png")
            run_single(mode, pipe, args)
        except Exception as e:
            log.error(f"[{item_id}] Failed: {e}", exc_info=True)

    log.info(f"Batch complete: {len(items)} item(s) processed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    log.info(f"Mode: {args.mode} | Device: {args.device} | FP16: {not args.no_fp16}")
    log.info(f"VRAM profile: {cfg.VRAM_PROFILES[args.mode]}")

    _validate_args(args.mode, args)

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
