#!/usr/bin/env python3
# =============================================================================
# gen_mask.py
# Generate an inpaint mask by asking Claude to locate an object in an image.
# Designed to run on Linux (no GPU required).
#
# Workflow:
#   1. Call `claude -p` with the image path + object name.
#   2. Claude returns a bounding box as JSON fractions (0-1).
#   3. Draw a white filled rectangle on a black canvas → mask.png.
#   4. Optionally refine to pixel-precise edges with SAM (--refine).
#
# Usage:
#   python gen_mask.py --input /full/path/to/image.png --object "chair"
#   python gen_mask.py --input /full/path/to/image.png --object "chair" --refine
#   python gen_mask.py --input /full/path/to/image.png --object "chair" --output /tmp/chair_mask.png
#
# Output (default): <image_dir>/<image_stem>_mask_<object>.png
# Original image is never modified.
#
# Requirements:
#   pip install Pillow numpy
#   pip install segment-anything torch torchvision   (only for --refine)
#   claude CLI must be installed and authenticated
#
# SAM weights for --refine (download once):
#   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
#   Place in ~/.cache/sam/ or pass --sam-checkpoint /path/to/weights.pth
# =============================================================================

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw
import numpy as np

SAM_DEFAULT_CHECKPOINT = Path.home() / ".cache" / "sam" / "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE         = "vit_b"

CLAUDE_PROMPT = (
    "Look at this image: {image_path}\n\n"
    "Find the {object} in the image and return its bounding box.\n"
    "Reply with ONLY a JSON object on a single line, no explanation:\n"
    '  {{"x1": <float>, "y1": <float>, "x2": <float>, "y2": <float>}}\n'
    "Where x1,y1 is the top-left corner and x2,y2 is the bottom-right corner, "
    "expressed as fractions of the image width and height (0.0 to 1.0).\n"
    "If the object is not found, return: {{}}"
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an inpaint mask by asking Claude to locate an object.",
        epilog=(
            "Examples:\n"
            "  python gen_mask.py --input /data/bg.png --object 'chair'\n"
            "  python gen_mask.py --input /data/bg.png --object 'chair' --refine\n"
            "  python gen_mask.py --input /data/bg.png --object 'torch on wall' --padding 10\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",   required=True, help="Full path to input image (not modified).")
    parser.add_argument("--object",  required=True, help="Object to erase, e.g. 'chair', 'torch on wall'.")
    parser.add_argument("--output",  default=None,  help="Output mask path. Default: <img_dir>/<stem>_mask_<object>.png")
    parser.add_argument(
        "--refine", action="store_true",
        help="Refine rect mask to pixel-precise edges using SAM (needs GPU + SAM weights).",
    )
    parser.add_argument(
        "--sam-checkpoint", dest="sam_checkpoint", default=str(SAM_DEFAULT_CHECKPOINT),
        help=f"SAM weights path (default: {SAM_DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--padding", type=int, default=8,
        help="Extra pixels to expand the bounding box on each side (default: 8).",
    )
    parser.add_argument(
        "--claude-bin", dest="claude_bin", default="claude",
        help="Path to claude CLI binary (default: 'claude', assumed on PATH).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Step 1: Ask Claude for bounding box
# ---------------------------------------------------------------------------

def ask_claude_for_bbox(image_path: Path, object_name: str,
                        claude_bin: str) -> dict | None:
    """
    Call `claude -p` with the image path inline in the prompt.
    Returns {"x1", "y1", "x2", "y2"} as floats, or None if not found.
    """
    prompt = CLAUDE_PROMPT.format(
        image_path=str(image_path),
        object=object_name,
    )

    print(f"[CLAUDE] Asking claude to locate '{object_name}'...")
    try:
        result = subprocess.run(
            [claude_bin, "-p", prompt],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError:
        print(f"[ERROR] claude CLI not found at '{claude_bin}'. "
              "Install Claude Code and ensure it is on PATH.")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("[ERROR] claude -p timed out after 60s.")
        sys.exit(1)

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if result.returncode != 0:
        print(f"[ERROR] claude -p exited {result.returncode}")
        if stderr:
            print(f"  stderr: {stderr}")
        sys.exit(1)

    print(f"[CLAUDE] Response: {stdout}")

    # Extract JSON from response (may have extra whitespace or prose)
    json_match = re.search(r'\{[^}]*\}', stdout)
    if not json_match:
        print(f"[ERROR] Could not find JSON in claude response: {stdout}")
        return None

    try:
        bbox = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON: {e}\n  Raw: {json_match.group()}")
        return None

    if not bbox:
        print(f"[WARN] Claude could not find '{object_name}' in the image.")
        return None

    required = {"x1", "y1", "x2", "y2"}
    if not required.issubset(bbox.keys()):
        print(f"[ERROR] Incomplete bbox keys: {bbox}")
        return None

    # Clamp to 0-1
    for k in required:
        bbox[k] = max(0.0, min(1.0, float(bbox[k])))

    print(f"[CLAUDE] Bbox: x1={bbox['x1']:.3f} y1={bbox['y1']:.3f} "
          f"x2={bbox['x2']:.3f} y2={bbox['y2']:.3f}")
    return bbox


# ---------------------------------------------------------------------------
# Step 2a: Rectangular mask
# ---------------------------------------------------------------------------

def make_rect_mask(img_w: int, img_h: int, bbox: dict, padding: int) -> Image.Image:
    """Draw a white rectangle on a black canvas."""
    x1 = max(0,     int(bbox["x1"] * img_w) - padding)
    y1 = max(0,     int(bbox["y1"] * img_h) - padding)
    x2 = min(img_w, int(bbox["x2"] * img_w) + padding)
    y2 = min(img_h, int(bbox["y2"] * img_h) + padding)

    mask = Image.new("L", (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x1, y1, x2, y2], fill=255)
    print(f"[MASK] Rect mask: ({x1},{y1}) → ({x2},{y2})  "
          f"size {x2-x1}x{y2-y1}px  padding={padding}")
    return mask


# ---------------------------------------------------------------------------
# Step 2b: SAM refinement
# ---------------------------------------------------------------------------

def refine_with_sam(image_path: Path, bbox: dict, mask_rect: Image.Image,
                    checkpoint: str) -> Image.Image:
    """
    Use the bounding box centre as a SAM prompt point to get a precise mask.
    Falls back to the rectangular mask if SAM fails.
    """
    try:
        import torch
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError:
        print("[WARN] segment-anything not installed. Falling back to rect mask.")
        print("       pip install segment-anything torch torchvision")
        return mask_rect

    ckpt = Path(checkpoint)
    if not ckpt.exists():
        print(f"[WARN] SAM checkpoint not found: {ckpt}")
        print(f"       Download: wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        print(f"       Place at: {SAM_DEFAULT_CHECKPOINT}")
        print("       Falling back to rect mask.")
        return mask_rect

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SAM] Loading {SAM_MODEL_TYPE} on {device}...")

    sam   = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(ckpt))
    sam.to(device)
    pred  = SamPredictor(sam)

    img_np = np.array(Image.open(str(image_path)).convert("RGB"))
    img_h, img_w = img_np.shape[:2]
    pred.set_image(img_np)

    # Use bbox centre as the positive prompt point
    cx = int(((bbox["x1"] + bbox["x2"]) / 2) * img_w)
    cy = int(((bbox["y1"] + bbox["y2"]) / 2) * img_h)

    # Also pass the bbox as a SAM box prompt for better accuracy
    bx1 = int(bbox["x1"] * img_w)
    by1 = int(bbox["y1"] * img_h)
    bx2 = int(bbox["x2"] * img_w)
    by2 = int(bbox["y2"] * img_h)

    print(f"[SAM] Point prompt: ({cx},{cy})  Box: ({bx1},{by1})→({bx2},{by2})")

    masks, scores, _ = pred.predict(
        point_coords  = np.array([[cx, cy]]),
        point_labels  = np.array([1]),            # 1 = foreground
        box           = np.array([bx1, by1, bx2, by2]),
        multimask_output = True,
    )

    # Pick the mask with the highest confidence score
    best_idx  = int(np.argmax(scores))
    best_mask = masks[best_idx]
    print(f"[SAM] Best mask score: {scores[best_idx]:.3f}  "
          f"coverage: {best_mask.sum()} px")

    mask_img = Image.fromarray((best_mask * 255).astype(np.uint8), mode="L")

    # Clean up VRAM
    del sam, pred
    if device == "cuda":
        torch.cuda.empty_cache()

    return mask_img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    image_path = Path(args.input)
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    # Default output path
    if args.output:
        out_path = Path(args.output)
    else:
        safe_obj = args.object.replace(" ", "_")
        out_path = image_path.parent / f"{image_path.stem}_mask_{safe_obj}.png"

    # Load image dimensions (do not modify the original)
    with Image.open(str(image_path)) as img:
        img_w, img_h = img.size
    print(f"[INPUT] {image_path.name}  {img_w}x{img_h}px")

    # Step 1: Ask Claude for bounding box
    bbox = ask_claude_for_bbox(image_path, args.object, args.claude_bin)
    if bbox is None:
        print("[ERROR] Could not obtain bounding box. Exiting.")
        sys.exit(1)

    # Step 2: Generate mask
    mask = make_rect_mask(img_w, img_h, bbox, args.padding)

    if args.refine:
        print("[SAM] Refining mask with Segment Anything...")
        mask = refine_with_sam(image_path, bbox, mask, args.sam_checkpoint)

    # Save mask
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(str(out_path))
    print(f"[OK] Mask saved → {out_path}")
    print()
    print("NEXT STEP:")
    print(f"  python gen_img2img.py --mode inpaint \\")
    print(f"    --input {image_path} \\")
    print(f"    --mask {out_path} \\")
    print(f"    --prompt \"ancient Egyptian stone floor, dark corner, empty\"")


if __name__ == "__main__":
    main()
