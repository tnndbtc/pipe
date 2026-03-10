# =============================================================================
# gen_upscale.py
# 2x upscale character RGBA PNGs using Real-ESRGAN x4plus so they hold
# quality when composited over 1280x720 background plates.
# Input:  char-*-rgba.png  (512x768, from gen_character_mattes.py)
# Output: char-*-upscaled.png  (1024x1536, RGBA preserved)
# Run AFTER gen_character_mattes.py (or use --auto-matte via gen_upscale.py directly).
# STATUS: VALIDATED -- runs on RTX 4060 8 GB (~2 GB VRAM). Auto-runs
#         gen_character_mattes.py if RGBA inputs are missing.
# =============================================================================
#
# requirements.txt (pip install before running):
#   torch>=2.4.1
#   realesrgan>=0.3.0
#   basicsr>=1.4.2
#   Pillow>=10.0.0
#   numpy>=1.24.0,<2.0.0
#   opencv-python>=4.9.0
#   (weights downloaded directly from GitHub releases -- no huggingface_hub needed)
#
# NOTE: basicsr/realesrgan install can fail on newer environments.
# If `pip install realesrgan basicsr` errors, try:
#   pip install realesrgan basicsr --no-build-isolation
#
# ---------------------------------------------------------------------------
# Hardware Target: NVIDIA RTX 4060 8 GB VRAM
# ---------------------------------------------------------------------------
# Memory-saving techniques:
#   Real-ESRGAN x4plus is a compact RRDB network (~67 MB weights).
#   VRAM usage is ~2 GB -- trivially fits on any GPU.
#
#   Techniques used:
#     1. half=True: loads the model in FP16, halving weight VRAM.
#     2. tile=512, tile_pad=10: processes large images in 512x512 spatial
#        tiles. Without tiling, the intermediate feature maps for a large
#        image can OOM. With tiling, VRAM is capped at ~2 GB regardless
#        of image size.
#     3. outscale=2: upsamples 4x then downsamples to 2x target, which
#        produces better antialiasing than native 2x upscaling.
#     4. Alpha channel is handled separately: RGB and alpha are split,
#        RGB is upscaled, alpha is bilinear-resized, then recombined.
#        This preserves the crisp matte edges from RMBG-1.4.
#     5. torch.cuda.empty_cache() + gc.collect() after each image.
#     6. CPU fallback: tile=0 on CPU (full image, no tiling needed as
#        there is no VRAM limit -- just RAM).
#
# UPGRADE: Background images are already at 1280x720 target resolution.
#   Uncomment the background entries in UPSCALE_TARGETS below if they
#   appear soft after generation.
# ---------------------------------------------------------------------------

import argparse
import gc
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# DEFAULTS -- fully populated; script runs with no CLI flags.
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "projects" / "the-pharaoh-who-defied-death" / "episodes" / "s01e01" / "assets"
SCRIPT_NAME = "gen_upscale"

UPSCALE_TARGETS = [
    # Characters -- input is the RGBA PNG from gen_character_mattes.py
    {
        "asset_id": "char-ramesses_ka-v1",
        "input":    "char-ramesses_ka-v1-rgba.png",
        "output":   "char-ramesses_ka-v1-upscaled.png",
        "scale":    2,   # 512x768 -> 1024x1536
    },
    {
        "asset_id": "char-amunhotep-v1",
        "input":    "char-amunhotep-v1-rgba.png",
        "output":   "char-amunhotep-v1-upscaled.png",
        "scale":    2,
    },
    {
        "asset_id": "char-neferet-v1",
        "input":    "char-neferet-v1-rgba.png",
        "output":   "char-neferet-v1-upscaled.png",
        "scale":    2,
    },
    {
        "asset_id": "char-khamun-v1",
        "input":    "char-khamun-v1-rgba.png",
        "output":   "char-khamun-v1-upscaled.png",
        "scale":    2,
    },
    {
        "asset_id": "char-prisoner-v1",
        "input":    "char-prisoner-v1-rgba.png",
        "output":   "char-prisoner-v1-upscaled.png",
        "scale":    2,
    },
    # Backgrounds -- uncomment if generated images appear soft.
    # Already at 1280x720 target resolution; only upscale if needed.
    # {
    #     "asset_id": "bg-karnak-inner-sanctuary-v1",
    #     "input":    "bg-karnak-inner-sanctuary-v1.png",
    #     "output":   "bg-karnak-inner-sanctuary-v1-upscaled.png",
    #     "scale":    2,   # 1280x720 -> 2560x1440, renderer downscales back
    # },
    # {
    #     "asset_id": "bg-temple-forbidden-archives-v1",
    #     "input":    "bg-temple-forbidden-archives-v1.png",
    #     "output":   "bg-temple-forbidden-archives-v1-upscaled.png",
    #     "scale":    2,
    # },
]

REALESRGAN_FILENAME    = "RealESRGAN_x4plus.pth"
REALESRGAN_GITHUB_URL  = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
    "RealESRGAN_x4plus.pth"
)
REALESRGAN_CACHE_DIR   = Path(__file__).resolve().parent.parent.parent / ".cache" / "realesrgan"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Upscale character RGBA PNGs 2x using Real-ESRGAN x4plus.",
        epilog=(
            "Model used:\n\n"
            "  Real-ESRGAN x4plus    xinntao/Real-ESRGAN  (RealESRGAN_x4plus.pth)\n"
            "                        RRDB network, ~67 MB weights, ~2 GB VRAM.\n"
            "                        tile=512 (default) caps VRAM at ~2 GB for any image size.\n"
            "                        outscale=2: runs 4x model then downsamples to 2x for\n"
            "                        better antialiasing vs a native 2x model.\n"
            "                        Alpha channel preserved (split -> upscale RGB -> resize alpha).\n\n"
            "  This script has no --model flag; Real-ESRGAN x4plus is the only supported model."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tile", type=int, default=512,
        help="Tile size for RealESRGAN (0=no tiling). 512 keeps VRAM under 2 GB."
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to AssetManifest JSON. When given, overrides the hardcoded UPSCALE_TARGETS list.",
    )
    parser.add_argument(
        "--asset-id", type=str, default=None, dest="asset_id",
        help="Process only this asset_id (requires --manifest).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-upscale even if output already exists.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Manifest loader
# ---------------------------------------------------------------------------
def load_from_manifest(manifest_path: str, asset_id_filter):
    """Load upscale target list from AssetManifest JSON (section: character_packs)."""
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    targets = []
    for pack in manifest.get("character_packs", []):
        if asset_id_filter and pack["asset_id"] != asset_id_filter:
            continue
        aid = pack["asset_id"]
        targets.append({
            "asset_id": aid,
            "input":    f"{aid}-rgba.png",
            "output":   f"{aid}-upscaled.png",
            "scale":    2,
        })
    return targets


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
def load_upsampler(tile: int):
    """
    Download Real-ESRGAN x4plus weights from GitHub releases and initialise
    the RealESRGANer inference object.  Weights are cached in .cache/realesrgan/.
    """
    import urllib.request
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    REALESRGAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    weight_path = REALESRGAN_CACHE_DIR / REALESRGAN_FILENAME
    if not weight_path.exists():
        print(f"[MODEL] Downloading {REALESRGAN_FILENAME} from GitHub releases...")
        urllib.request.urlretrieve(REALESRGAN_GITHUB_URL, str(weight_path))
    else:
        print(f"[MODEL] Using cached weights: {weight_path}")
    print(f"[MODEL] Weights at: {weight_path}")

    # RRDBNet architecture for x4plus (23 RRDB blocks)
    rrdb_model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32,
        scale=4,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MODEL] Loading RealESRGAN on {device} (FP16={'yes' if device=='cuda' else 'no/CPU'})...")
    upsampler = RealESRGANer(
        scale=4,
        model_path=str(weight_path),
        model=rrdb_model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=(device == "cuda"),   # FP16 on GPU, FP32 on CPU
        device=device,
    )
    print("[MODEL] RealESRGANer ready.")
    return upsampler


# ---------------------------------------------------------------------------
# Per-image upscaling
# ---------------------------------------------------------------------------
def upscale_image(upsampler, input_path: Path, output_path: Path, target_scale: int) -> int:
    """
    Upscale one image.  Handles RGBA by splitting alpha, upscaling RGB,
    resizing alpha, then recombining.  Returns output file size in bytes.
    """
    img = Image.open(str(input_path))
    has_alpha = img.mode == "RGBA"

    if has_alpha:
        # Split RGB and alpha -- upscale RGB, resize alpha bilinearly
        r, g, b, a = img.split()
        rgb_img = Image.merge("RGB", (r, g, b))
        alpha_np = np.array(a)
    else:
        rgb_img = img.convert("RGB")
        alpha_np = None

    # Convert to BGR numpy array (OpenCV convention used by RealESRGAN)
    rgb_bgr = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)

    # Run Real-ESRGAN at 4x then downscale to target_scale (e.g. 2x)
    # outscale=2 means the returned image is 2x the input, using a 4x
    # internal model for better quality than a native 2x model would give.
    output_bgr, _ = upsampler.enhance(rgb_bgr, outscale=target_scale)

    # Convert back to RGB PIL Image
    output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
    output_h, output_w = output_rgb.shape[:2]
    result = Image.fromarray(output_rgb)

    if has_alpha:
        # Resize alpha mask to match the upscaled dimensions
        alpha_resized = cv2.resize(
            alpha_np, (output_w, output_h), interpolation=cv2.INTER_LANCZOS4
        )
        result = result.convert("RGBA")
        result.putalpha(Image.fromarray(alpha_resized))

    result.save(str(output_path), format="PNG")
    return output_path.stat().st_size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_mattes(manifest_path, output_dir: Path) -> None:
    """
    Auto-run gen_character_mattes.py when RGBA inputs are missing.
    Called before the upscale loop so the user only needs to run gen_upscale.py.
    """
    import subprocess
    import sys

    mattes_script = Path(__file__).resolve().parent / "gen_character_mattes.py"
    if not mattes_script.exists():
        print(f"[AUTO-MATTE] gen_character_mattes.py not found at {mattes_script} -- skipping.")
        return

    print("\n[AUTO-MATTE] RGBA inputs missing -- running gen_character_mattes.py first...")
    cmd = [sys.executable, str(mattes_script), "--output_dir", str(output_dir)]
    if manifest_path:
        cmd += ["--manifest", manifest_path]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[AUTO-MATTE] gen_character_mattes.py exited with errors -- some inputs may still be missing.")
    print()


def locale_from_manifest_path(path: str) -> str:
    """Extract locale from manifest filename.
    'AssetManifest_draft.zh-Hans.json' -> 'zh-Hans'
    'AssetManifest_draft.json'          -> 'en'
    """
    stem = Path(path).stem
    parts = stem.split('.')
    return parts[-1] if len(parts) > 1 else 'en'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    locale = locale_from_manifest_path(args.manifest) if args.manifest else 'en'
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR / locale
    out_dir.mkdir(parents=True, exist_ok=True)

    upscale_targets = UPSCALE_TARGETS
    if args.manifest:
        upscale_targets = load_from_manifest(args.manifest, args.asset_id)
        if not upscale_targets:
            print("[WARN] No matching character_packs in manifest. Nothing to do.")
            return

    # Auto-matte: if any RGBA inputs are missing, run gen_character_mattes.py first
    if any(not (out_dir / t["input"]).exists() for t in upscale_targets):
        ensure_mattes(args.manifest, out_dir)

    upsampler = load_upsampler(args.tile)

    results = []
    total = len(upscale_targets)

    for idx, target in enumerate(upscale_targets, start=1):
        input_path  = out_dir / target["input"]
        output_path = out_dir / target["output"]
        print(f"\n[{idx}/{total}] Upscaling {target['asset_id']} ({target['scale']}x)...")

        if output_path.exists() and not args.force:
            print(f"  [SKIP] {target['output']} already exists")
            results.append({
                "asset_id": target["asset_id"],
                "output": str(output_path),
                "size_bytes": output_path.stat().st_size,
                "status": "skipped",
            })
            continue

        if not input_path.exists():
            print(f"  [SKIP] Input not found: {input_path}")
            print(f"  [HINT] Run gen_character_mattes.py first.")
            results.append({
                "asset_id": target["asset_id"],
                "output": str(output_path),
                "size_bytes": 0,
                "status": "skipped",
                "error": f"Input missing: {input_path}",
            })
            continue

        try:
            # Report input dimensions
            with Image.open(str(input_path)) as probe:
                iw, ih = probe.size
            print(f"  Input:  {iw}x{ih} px  ({input_path.stat().st_size:,} bytes)")

            size = upscale_image(upsampler, input_path, output_path, target["scale"])

            with Image.open(str(output_path)) as probe:
                ow, oh = probe.size
            print(f"  Output: {ow}x{oh} px  ({size:,} bytes)")
            print(f"  [OK] {output_path}")
            results.append({
                "asset_id": target["asset_id"],
                "output": str(output_path),
                "size_bytes": size,
                "input_size": f"{iw}x{ih}",
                "output_size": f"{ow}x{oh}",
                "status": "success",
            })
        except Exception as exc:
            print(f"  [ERROR] {target['asset_id']}: {exc}")
            results.append({
                "asset_id": target["asset_id"],
                "output": str(output_path),
                "size_bytes": 0,
                "status": "failed",
                "error": str(exc),
            })
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    # Free model memory
    del upsampler
    torch.cuda.empty_cache()
    gc.collect()

    # Write manifest
    manifest_path = out_dir / f"{SCRIPT_NAME}_results.json"
    with open(manifest_path, "w") as fh:
        json.dump(results, fh, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY -- gen_upscale")
    print("=" * 60)
    for r in results:
        tag = "OK" if r["status"] == "success" else r["status"].upper()
        dims = f"  {r.get('input_size','?')} -> {r.get('output_size','?')}" if r["status"] == "success" else ""
        print(f"  [{tag}]{dims}  {r['output']}  ({r['size_bytes']:,} bytes)")
    ok_count = sum(1 for r in results if r["status"] in ("success", "skipped"))
    total_bytes = sum(r["size_bytes"] for r in results)
    print(f"\n{ok_count}/{total} completed | {total_bytes:,} bytes total")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
