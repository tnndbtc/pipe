# =============================================================================
# gen_composite_image.py
# Composite character RGBA mattes onto background plates.
# Run AFTER gen_character_mattes.py and gen_background_images.py.
# =============================================================================
#
# requirements.txt (pip install before running):
#   Pillow>=10.0.0
#   numpy>=1.24.0            (only needed for --depth-scale)
#   fal-client>=0.5.0        (only needed for --mode iclight)
#   transformers>=4.38.0     (only needed for --depth-scale)
#   torch>=2.1.0             (only needed for --depth-scale)
#
# ---------------------------------------------------------------------------
# Quick test -- one character:
#
#   python gen_composite_image.py --character char.png --background bg.png
#
# Multiple characters on one background -- one call:
#
#   python gen_composite_image.py --background bg.png ^
#     --character char1.png --char-x 0.3 ^
#     --character char2.png --char-x 0.7
#
#   --character / --char-x / --char-y can each be repeated once per character.
#   If fewer --char-x values than characters, remaining characters are spaced evenly.
#   Characters are composited back-to-front (further = rendered first so closer
#   characters naturally appear in front).
#   Overlap between characters is detected and resolved by nudging horizontally.
#
# ---------------------------------------------------------------------------
# Compositing mode (--mode):
#
#   pillow    (default) Pure Pillow alpha-composite. No AI. Instant.
#   iclight   Pillow composite + IC-Light relighting via fal.ai.
#             Requires FAL_KEY env var. Cost ~$0.003-0.005/image.
#             Endpoint: fal-ai/iclight/v2
#
# ---------------------------------------------------------------------------
# Character scaling (priority: depth > --auto-scale > --char-scale > default):
#
#   depth-scale ON by default. Disable with --no-depth-scale.
#   Uses Depth Anything V2 Small to read scene perspective from the background.
#   Each character is sized to match the depth at their foot position.
#   Depth map is computed once per background and reused for all characters.
#   GPU: ~1 GB VRAM, ~0.5s.  CPU fallback: ~5-10s.
#   Model: depth-anything/Depth-Anything-V2-Small-hf
#   --fg-scale FLOAT  character size at foreground (default: 0.70)
#   --bg-scale FLOAT  character size at background (default: 0.15)
#
#   --auto-scale      Fit within 35% bg width and 70% bg height. No AI.
#   --char-scale F    Manual: character height as fraction of bg height.
#
# ---------------------------------------------------------------------------
# Output:
#   One-off:  <bg_dir>/comp_test.png  (or --output path)
#   Batch:    <asset_dir>/<asset_id>.png  (pillow) / <asset_id>_iclight.png
#
# NEXT STEP: run gen_character_animation.py (or gen_upscale.py) on outputs.
# =============================================================================

import argparse
import base64
import io
import json
import os
import time
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# DEFAULTS
# ---------------------------------------------------------------------------
OUTPUT_DIR  = (
    Path(__file__).resolve().parent.parent.parent
    / "projects" / "the-pharaoh-who-defied-death" / "episodes" / "s01e01" / "assets"
)
SCRIPT_NAME = "gen_composite_image"

# Hardcoded scene pairings -- used when no --manifest is given.
# Each scene has ONE character. Multi-character is a one-off feature (--character x2).
COMPOSITES = [
    {
        "asset_id":    "comp-amunhotep-karnak-v1",
        "characters":  [{"file": "char-amunhotep-v1-rgba.png",  "char_x": 0.5, "char_y": 0.92}],
        "background":  "bg-karnak-inner-sanctuary-v1.png",
        "output":      "comp-amunhotep-karnak-v1.png",
        "char_scale":  0.4,
    },
    {
        "asset_id":    "comp-ramesses-karnak-v1",
        "characters":  [{"file": "char-ramesses_ka-v1-rgba.png", "char_x": 0.5, "char_y": 0.92}],
        "background":  "bg-karnak-inner-sanctuary-v1.png",
        "output":      "comp-ramesses-karnak-v1.png",
        "char_scale":  0.4,
    },
    {
        "asset_id":    "comp-amunhotep-archives-v1",
        "characters":  [{"file": "char-amunhotep-v1-rgba.png",  "char_x": 0.35, "char_y": 0.92}],
        "background":  "bg-temple-forbidden-archives-v1.png",
        "output":      "comp-amunhotep-archives-v1.png",
        "char_scale":  0.4,
    },
]

DEPTH_MODEL_ID   = "depth-anything/Depth-Anything-V2-Small-hf"
ICLIGHT_ENDPOINT = "fal-ai/iclight/v2"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Composite character RGBA mattes onto background plates.",
        epilog=(
            "Scaling priority: depth (default) > --auto-scale > --char-scale > hardcoded\n\n"
            "Examples:\n"
            "  # One character (depth-scale on by default)\n"
            "  python gen_composite_image.py --character char.png --background bg.png\n\n"
            "  # Two characters on one background, one output image\n"
            "  python gen_composite_image.py --background bg.png ^\n"
            "    --character char1.png --char-x 0.3 ^\n"
            "    --character char2.png --char-x 0.7\n\n"
            "  # Disable depth scaling\n"
            "  python gen_composite_image.py --character char.png --background bg.png ^\n"
            "    --no-depth-scale --char-scale 0.35\n\n"
            "  # Batch from manifest\n"
            "  python gen_composite_image.py --manifest ../AssetManifest_draft.json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["pillow", "iclight"], default="pillow",
        help="Compositing mode. Default: pillow",
    )
    parser.add_argument("--output_dir", type=str, default=None)

    # --- Scaling ---
    scale_group = parser.add_argument_group(
        "scaling (priority: depth > --auto-scale > --char-scale > default)"
    )
    scale_group.add_argument(
        "--no-depth-scale", dest="depth_scale", action="store_false",
        help="Disable depth-aware scaling (on by default).",
    )
    scale_group.set_defaults(depth_scale=True)
    scale_group.add_argument(
        "--fg-scale", dest="fg_scale", type=float, default=0.70,
        help="Depth scaling: character size at foreground (default: 0.70).",
    )
    scale_group.add_argument(
        "--bg-scale", dest="bg_scale", type=float, default=0.15,
        help="Depth scaling: character size at background (default: 0.15).",
    )
    scale_group.add_argument(
        "--auto-scale", dest="auto_scale", action="store_true",
        help="Fit within 35%% bg width and 70%% bg height. Ignored if depth-scale is on.",
    )
    scale_group.add_argument(
        "--char-scale", dest="char_scale", type=float, default=None,
        help="Manual scale: character height as fraction of bg height.",
    )

    # --- Per-character position (repeatable) ---
    pos_group = parser.add_argument_group(
        "character position (repeat once per --character)"
    )
    pos_group.add_argument(
        "--char-x", dest="char_x", type=float, action="append", metavar="X",
        help="Horizontal anchor 0.0-1.0. Repeat for each character. Missing values auto-spaced.",
    )
    pos_group.add_argument(
        "--char-y", dest="char_y", type=float, action="append", metavar="Y",
        help="Vertical anchor for character bottom 0.0-1.0. Default: 0.92.",
    )

    # --- One-off / multi-char mode ---
    parser.add_argument(
        "--character", type=str, action="append", metavar="FILE",
        help="Character RGBA PNG. Repeat for multiple characters on one background.",
    )
    parser.add_argument(
        "--background", type=str, default=None,
        help="Background PNG. Enables one-off mode.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path. Default: <bg_dir>/comp_test.png",
    )

    # --- Batch mode ---
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to AssetManifest JSON.",
    )
    parser.add_argument(
        "--asset-id", type=str, default=None, dest="asset_id",
        help="Process only this composite asset_id (batch mode).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Manifest loader
# ---------------------------------------------------------------------------
def load_from_manifest(manifest_path: str, asset_id_filter: str | None) -> list[dict]:
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    bg_ids = [bg["asset_id"] for bg in manifest.get("backgrounds", [])]
    composites = []

    for char in manifest.get("character_packs", []):
        char_id   = char["asset_id"]
        char_file = f"char-{char_id}-v1-rgba.png"
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

        composites.append({
            "asset_id":   asset_id,
            "characters": [{"file": char_file, "char_x": 0.5, "char_y": 0.92}],
            "background": f"{matched_bg}-v1.png",
            "output":     f"{asset_id}.png",
            "char_scale": 0.4,
        })

    return composites


# ---------------------------------------------------------------------------
# Depth Anything V2
# ---------------------------------------------------------------------------
def load_depth_model():
    try:
        from transformers import pipeline as hf_pipeline
        import torch
    except ImportError:
        raise ImportError("Run: pip install transformers torch")

    device      = 0 if torch.cuda.is_available() else -1
    device_name = "GPU" if device == 0 else "CPU"
    print(f"[DEPTH] Loading {DEPTH_MODEL_ID} on {device_name}...")
    t0   = time.perf_counter()
    pipe = hf_pipeline(task="depth-estimation", model=DEPTH_MODEL_ID, device=device)
    print(f"[DEPTH] Model ready ({time.perf_counter()-t0:.1f}s)")
    return pipe


def get_depth_map(bg_path: Path, depth_pipe) -> tuple:
    """
    Run depth estimation on the background. Returns (depth_norm_arr, bg_w, bg_h).
    depth_norm_arr is a float32 numpy array normalised 0-1,
    where 1 = closest to camera. Shape: (bg_h, bg_w).
    """
    import numpy as np

    bg_img     = Image.open(str(bg_path)).convert("RGB")
    bg_w, bg_h = bg_img.size
    result     = depth_pipe(bg_img)
    depth_map  = result["depth"]   # PIL Image

    depth_arr  = np.array(depth_map, dtype=np.float32)
    d_min, d_max = depth_arr.min(), depth_arr.max()
    if d_max > d_min:
        norm = (depth_arr - d_min) / (d_max - d_min)
    else:
        norm = np.zeros_like(depth_arr)

    if depth_map.size != (bg_w, bg_h):
        tmp  = Image.fromarray((norm * 255).astype("uint8"))
        tmp  = tmp.resize((bg_w, bg_h), Image.BILINEAR)
        norm = __import__("numpy").array(tmp, dtype="float32") / 255.0

    return norm, bg_w, bg_h


def depth_to_scale(depth_norm_arr, bg_w: int, bg_h: int,
                   char_x: float, char_y: float,
                   fg_scale: float, bg_scale: float) -> float:
    """Sample depth at foot position and return a char_scale."""
    sx = max(0, min(int(bg_w * char_x), bg_w - 1))
    sy = max(0, min(int(bg_h * char_y), bg_h - 1))
    d  = float(depth_norm_arr[sy, sx])
    s  = bg_scale + (fg_scale - bg_scale) * d
    print(f"      [depth] foot=({sx},{sy})  depth={d:.3f}  -> scale={s:.3f}")
    return s


# ---------------------------------------------------------------------------
# Overlap helpers
# ---------------------------------------------------------------------------
def boxes_overlap(b1: tuple, b2: tuple) -> bool:
    """True if two (x1,y1,x2,y2) rectangles overlap."""
    return not (b1[2] <= b2[0] or b2[2] <= b1[0] or
                b1[3] <= b2[1] or b2[3] <= b1[1])


def nudge_no_overlap(paste_x: int, paste_y: int, w: int, h: int,
                     placed: list[tuple], bg_w: int, step: int = 4) -> int:
    """
    Shift paste_x horizontally (alternating left/right) until the bounding box
    (paste_x, paste_y, paste_x+w, paste_y+h) doesn't overlap any box in placed.
    Returns the adjusted paste_x. Warns if resolution is impossible.
    """
    bbox = (paste_x, paste_y, paste_x + w, paste_y + h)
    if not any(boxes_overlap(bbox, p) for p in placed):
        return paste_x

    for delta in range(step, bg_w, step):
        for direction in (+1, -1):
            nx    = paste_x + direction * delta
            nx    = max(0, min(nx, bg_w - w))
            new_b = (nx, paste_y, nx + w, paste_y + h)
            if not any(boxes_overlap(new_b, p) for p in placed):
                print(f"      [overlap] nudged {direction*delta:+d}px to avoid overlap")
                return nx

    print(f"      [WARN] Could not resolve overlap — placing at original x={paste_x}")
    return paste_x


# ---------------------------------------------------------------------------
# Core compositor (multi-character)
# ---------------------------------------------------------------------------
def composite_scene(
    bg_path: Path,
    characters: list[dict],    # [{"path": Path, "char_x": float, "char_y": float, "char_scale": float|None, "auto_scale": bool}]
    existing_canvas: Image.Image | None = None,
) -> Image.Image:
    """
    Paste all characters onto the background (or existing_canvas if provided).

    Characters are rendered back-to-front:
      sorted by char_y ascending = higher on screen = further away = rendered first.
    Overlap between characters is detected and resolved by nudging horizontally.

    Returns the composited RGBA PIL Image.
    """
    canvas = existing_canvas if existing_canvas is not None \
             else Image.open(str(bg_path)).convert("RGBA")
    bg_w, bg_h = canvas.size

    # Sort back-to-front: smaller char_y = higher on screen = farther = render first
    ordered = sorted(characters, key=lambda c: c["char_y"])

    placed_bboxes = []

    for c in ordered:
        char_img = Image.open(str(c["path"])).convert("RGBA")

        # Sizing
        if c.get("auto_scale"):
            scale_by_w = (bg_w * 0.35) / char_img.width
            scale_by_h = (bg_h * 0.70) / char_img.height
            ratio      = min(scale_by_w, scale_by_h)
            target_w   = int(char_img.width  * ratio)
            target_h   = int(char_img.height * ratio)
        else:
            scale    = c["char_scale"] if c.get("char_scale") is not None else 0.4
            target_h = int(bg_h * scale)
            ratio    = target_h / char_img.height
            target_w = int(char_img.width * ratio)

        char_img = char_img.resize((target_w, target_h), Image.LANCZOS)

        # Nominal paste position (bottom-centre anchor)
        paste_x = int(bg_w * c["char_x"]) - target_w // 2
        paste_y = int(bg_h * c["char_y"]) - target_h
        paste_x = max(0, min(paste_x, bg_w - target_w))
        paste_y = max(0, min(paste_y, bg_h - target_h))

        # Resolve overlap with already-placed characters
        paste_x = nudge_no_overlap(paste_x, paste_y, target_w, target_h,
                                   placed_bboxes, bg_w)

        print(f"      paste ({paste_x},{paste_y})  size {target_w}x{target_h}px  "
              f"scale={scale if not c.get('auto_scale') else 'auto':.3f}")

        canvas.paste(char_img, (paste_x, paste_y), char_img)
        placed_bboxes.append((paste_x, paste_y, paste_x + target_w, paste_y + target_h))

    return canvas


# ---------------------------------------------------------------------------
# Resolve per-character scale using depth map (or fallback)
# ---------------------------------------------------------------------------
def resolve_characters(
    char_paths: list[Path],
    char_xs: list[float],
    char_ys: list[float],
    args,
    depth_norm_arr,    # numpy array or None
    bg_w: int,
    bg_h: int,
    fallback_scale: float,
) -> list[dict]:
    """
    Build the character dicts for composite_scene(), resolving scale for each.
    """
    result = []
    for i, path in enumerate(char_paths):
        char_x = char_xs[i]
        char_y = char_ys[i]

        if args.depth_scale and depth_norm_arr is not None:
            scale      = depth_to_scale(depth_norm_arr, bg_w, bg_h,
                                        char_x, char_y, args.fg_scale, args.bg_scale)
            auto_scale = False
        elif args.auto_scale:
            scale      = None
            auto_scale = True
        elif args.char_scale is not None:
            scale      = args.char_scale
            auto_scale = False
        else:
            scale      = fallback_scale
            auto_scale = False

        result.append({
            "path":       path,
            "char_x":     char_x,
            "char_y":     char_y,
            "char_scale": scale,
            "auto_scale": auto_scale,
        })
    return result


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------
def auto_x_positions(n: int) -> list[float]:
    """Evenly space n characters across the frame."""
    if n == 1:
        return [0.5]
    return [round((i + 1) / (n + 1), 3) for i in range(n)]


def fill_positions(provided: list | None, n: int, default: float) -> list[float]:
    """
    Return a list of length n.
    Uses provided values where available, fills remainder with default or auto-spacing.
    """
    provided = provided or []
    result   = list(provided[:n])
    while len(result) < n:
        result.append(default)
    return result


# ---------------------------------------------------------------------------
# IC-Light via fal.ai
# ---------------------------------------------------------------------------
def iclight_relight(composite_img: Image.Image, bg_path: Path) -> Image.Image:
    try:
        import fal_client
    except ImportError:
        raise ImportError("Run: pip install fal-client  then set FAL_KEY env var.")

    if not os.environ.get("FAL_KEY"):
        raise EnvironmentError("FAL_KEY not set. Get a key at https://fal.ai")

    def to_data_uri(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    print(f"    [IC-Light] Uploading to fal.ai ({ICLIGHT_ENDPOINT})...")
    t0     = time.perf_counter()
    result = fal_client.subscribe(
        ICLIGHT_ENDPOINT,
        arguments={
            "foreground_image_url": to_data_uri(composite_img),
            "background_image_url": to_data_uri(Image.open(str(bg_path)).convert("RGB")),
            "prompt":          "cinematic lighting, photorealistic, ancient Egyptian setting",
            "negative_prompt": "blurry, low quality, overexposed",
            "num_inference_steps": 28,
            "guidance_scale":  1.5,
        },
    )
    print(f"    [IC-Light] Done ({time.perf_counter()-t0:.1f}s)")

    import urllib.request
    url = result.get("image", {}).get("url") or result.get("images", [{}])[0].get("url")
    if not url:
        raise RuntimeError(f"Unexpected IC-Light response: {result}")
    with urllib.request.urlopen(url) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def locale_from_manifest_path(path: str) -> str:
    parts = Path(path).stem.split(".")
    return parts[-1] if len(parts) > 1 else "en"


def output_filename(base: str, mode: str) -> str:
    if mode == "pillow":
        return base
    return f"{Path(base).stem}_{mode}.png"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Load depth model once (reused across all scenes / characters)
    depth_pipe = load_depth_model() if args.depth_scale else None

    # Depth map cache: bg_path -> (depth_norm_arr, bg_w, bg_h)
    # Avoids re-running depth estimation when the same bg appears in multiple scenes.
    depth_cache: dict[str, tuple] = {}

    def get_cached_depth(bg_path: Path):
        key = str(bg_path)
        if key not in depth_cache:
            print(f"  [DEPTH] Running depth estimation on {bg_path.name}...")
            depth_cache[key] = get_depth_map(bg_path, depth_pipe)
        return depth_cache[key]

    # ------------------------------------------------------------------
    # One-off / multi-character mode
    # ------------------------------------------------------------------
    if args.character or args.background:
        if not args.character or not args.background:
            print("[ERROR] Provide both --character and --background.")
            return

        bg_path    = Path(args.background)
        char_paths = [Path(c) for c in args.character]

        for p in [bg_path] + char_paths:
            if not p.exists():
                print(f"[ERROR] File not found: {p}")
                return

        n      = len(char_paths)
        char_xs = fill_positions(args.char_x, n, None)  # None = auto-space below
        char_ys = fill_positions(args.char_y, n, 0.92)

        # Auto-space any char_x that wasn't specified
        auto_xs = auto_x_positions(n)
        char_xs = [char_xs[i] if char_xs[i] is not None else auto_xs[i] for i in range(n)]

        if args.output:
            out_path = Path(args.output)
        else:
            suffix   = f"_{args.mode}" if args.mode != "pillow" else ""
            out_path = bg_path.parent / f"comp_test{suffix}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[ONE-OFF] bg={bg_path.name}  characters={n}")
        for i, p in enumerate(char_paths):
            print(f"  char[{i}]: {p.name}  x={char_xs[i]}  y={char_ys[i]}")
        print(f"  -> {out_path}")

        depth_norm_arr, bg_w, bg_h = get_cached_depth(bg_path) if depth_pipe else (None, 0, 0)

        chars = resolve_characters(char_paths, char_xs, char_ys, args,
                                   depth_norm_arr, bg_w, bg_h, fallback_scale=0.4)

        t0        = time.perf_counter()
        composite = composite_scene(bg_path, chars)
        if args.mode == "iclight":
            print("  [ICLIGHT] Sending to fal.ai...")
            composite = iclight_relight(composite, bg_path)
        composite.save(str(out_path), format="PNG")
        print(f"[OK] {out_path.stat().st_size:,} bytes  ({time.perf_counter()-t0:.1f}s)")
        return

    # ------------------------------------------------------------------
    # Batch mode
    # ------------------------------------------------------------------
    locale  = locale_from_manifest_path(args.manifest) if args.manifest else "en"
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR / locale
    out_dir.mkdir(parents=True, exist_ok=True)

    composites = COMPOSITES
    if args.manifest:
        composites = load_from_manifest(args.manifest, args.asset_id)
        if not composites:
            print("[WARN] No composites found in manifest.")
            return
    elif args.asset_id:
        composites = [c for c in composites if c["asset_id"] == args.asset_id]
        if not composites:
            print(f"[WARN] asset_id '{args.asset_id}' not found.")
            return

    scaling_desc = (
        f"depth (fg={args.fg_scale}, bg={args.bg_scale})" if args.depth_scale
        else "auto"    if args.auto_scale
        else f"manual ({args.char_scale})" if args.char_scale
        else "per-scene default"
    )
    print(f"[MODE]    {args.mode}")
    print(f"[SCALING] {scaling_desc}")
    print(f"[OUT DIR] {out_dir}")
    print(f"[SCENES]  {len(composites)}")

    results = []
    total   = len(composites)

    for idx, scene in enumerate(composites, start=1):
        fname    = output_filename(scene["output"], args.mode)
        out_path = out_dir / fname
        bg_path  = out_dir / scene["background"]

        # Build character list from scene definition
        scene_chars = scene["characters"]
        char_paths  = [out_dir / c["file"] for c in scene_chars]
        char_xs     = [args.char_x[0] if args.char_x else c["char_x"] for c in scene_chars]
        char_ys     = [args.char_y[0] if args.char_y else c["char_y"] for c in scene_chars]

        print(f"\n[{idx}/{total}] {scene['asset_id']}")
        for c in scene_chars:
            print(f"  char: {c['file']}")
        print(f"  bg:   {scene['background']}")
        print(f"  out:  {fname}")

        if out_path.exists() and not args.force:
            print(f"  [SKIP] already exists")
            results.append({"asset_id": scene["asset_id"], "mode": args.mode,
                            "output": str(out_path), "size_bytes": out_path.stat().st_size,
                            "status": "skipped"})
            continue

        missing = [p for p in [bg_path] + char_paths if not p.exists()]
        if missing:
            for p in missing:
                print(f"  [SKIP] not found: {p}")
            if any(not (out_dir / c["file"]).exists() for c in scene_chars):
                print(f"  [HINT] Run gen_character_mattes.py first.")
            if not bg_path.exists():
                print(f"  [HINT] Run gen_background_images.py first.")
            results.append({"asset_id": scene["asset_id"], "mode": args.mode,
                            "output": str(out_path), "size_bytes": 0,
                            "status": "skipped",
                            "error": f"Missing: {[str(p) for p in missing]}"})
            continue

        try:
            t0 = time.perf_counter()

            depth_norm_arr, bg_w, bg_h = get_cached_depth(bg_path) if depth_pipe else (None, 0, 0)

            chars = resolve_characters(char_paths, char_xs, char_ys, args,
                                       depth_norm_arr, bg_w, bg_h,
                                       fallback_scale=scene["char_scale"])

            composite = composite_scene(bg_path, chars)

            if args.mode == "iclight":
                print(f"  [ICLIGHT] Sending to fal.ai...")
                composite = iclight_relight(composite, bg_path)

            composite.save(str(out_path), format="PNG")
            size    = out_path.stat().st_size
            elapsed = time.perf_counter() - t0
            print(f"  [OK] {size:,} bytes  ({elapsed:.1f}s)")
            results.append({"asset_id": scene["asset_id"], "mode": args.mode,
                            "output": str(out_path), "size_bytes": size,
                            "status": "success", "time_s": round(elapsed, 1)})

        except Exception as exc:
            print(f"  [ERROR] {exc}")
            results.append({"asset_id": scene["asset_id"], "mode": args.mode,
                            "output": str(out_path), "size_bytes": 0,
                            "status": "failed", "error": str(exc)})

    results_path = out_dir / f"{SCRIPT_NAME}_results.json"
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)

    ok_count    = sum(1 for r in results if r["status"] in ("success", "skipped"))
    total_bytes = sum(r["size_bytes"] for r in results)
    print("\n" + "=" * 60)
    print(f"SUMMARY -- gen_composite_image ({args.mode}, {scaling_desc})")
    print("=" * 60)
    for r in results:
        label  = "OK" if r["status"] == "success" else r["status"].upper()
        time_s = f"  time={r['time_s']:.1f}s" if "time_s" in r else ""
        print(f"  [{label}]  {r['output']}  ({r['size_bytes']:,} bytes){time_s}")
    print(f"\n{ok_count}/{total} completed | {total_bytes:,} bytes total")
    print(f"Results: {results_path}")
    print()
    if args.mode == "pillow":
        print("NEXT STEP: review outputs, then re-run with --mode iclight for AI relighting.")
    else:
        print("NEXT STEP: run gen_character_animation.py to animate the composited scene.")


if __name__ == "__main__":
    main()
