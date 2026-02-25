# =============================================================================
# gen_character_mattes.py
# Remove backgrounds from character portrait PNGs, producing RGBA images
# with transparent backgrounds ready for compositing over background plates.
# Run AFTER gen_character_images.py.
# =============================================================================
#
# requirements.txt (pip install before running):
#   torch>=2.4.1
#   transformers>=4.40.0
#   torchvision>=0.19.0
#   Pillow>=10.0.0
#   numpy>=1.24.0,<2.0.0
#   huggingface_hub>=0.21.0
#
# ---------------------------------------------------------------------------
# Hardware Target: NVIDIA RTX 4060 8 GB VRAM
# ---------------------------------------------------------------------------
# Memory-saving techniques:
#   RMBG-1.4 is a lightweight BiRefNet segmentation model (~175 MB weights).
#   VRAM usage is ~1-2 GB — the model trivially fits on any GPU.
#
#   Techniques used:
#     1. torch.no_grad() during inference — no gradient graph stored.
#     2. Input resized to 1024×1024 for inference, then mask is resized
#        back to the original image dimensions before applying as alpha.
#     3. torch.cuda.empty_cache() + gc.collect() after each image.
#     4. CPU fallback: if CUDA is unavailable, the model runs on CPU at
#        ~2-5 seconds per image (still acceptable for prototype runs).
#
# NOTE: briaai/RMBG-1.4 requires trust_remote_code=True.
#   The model code is loaded directly from HuggingFace — review the
#   model card at https://huggingface.co/briaai/RMBG-1.4 before running.
#   No licence login required (non-commercial research licence).
#
# ANIMATED VIDEO NOTE:
#   Per-frame alpha matting of char-*-anim.mp4 files is out of scope for
#   this prototype. Phase 2 should apply RMBG-1.4 frame-by-frame using an
#   ffmpeg pipe: ffmpeg -i input.mp4 | [RMBG per frame] | ffmpeg output.mp4
# ---------------------------------------------------------------------------

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# DEFAULTS — fully populated; script runs with no CLI flags.
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("projects/the-pharaoh-who-defied-death/episodes/s01e01/assets")
SCRIPT_NAME = "gen_character_mattes"

CHARACTERS = [
    {
        "asset_id": "char-ramesses_ka-v1",
        "input":    "char-ramesses_ka-v1.png",
        "output":   "char-ramesses_ka-v1-rgba.png",
    },
    {
        "asset_id": "char-amunhotep-v1",
        "input":    "char-amunhotep-v1.png",
        "output":   "char-amunhotep-v1-rgba.png",
    },
    {
        "asset_id": "char-neferet-v1",
        "input":    "char-neferet-v1.png",
        "output":   "char-neferet-v1-rgba.png",
    },
    {
        "asset_id": "char-khamun-v1",
        "input":    "char-khamun-v1.png",
        "output":   "char-khamun-v1-rgba.png",
    },
    {
        "asset_id": "char-prisoner-v1",
        "input":    "char-prisoner-v1.png",
        "output":   "char-prisoner-v1-rgba.png",
    },
]

RMBG_MODEL_ID = "briaai/RMBG-1.4"
# RMBG-1.4 expects images normalised to ImageNet stats at 1024×1024
INFERENCE_SIZE = (1024, 1024)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove backgrounds from character portraits using RMBG-1.4."
    )
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Mask binarisation threshold (0.0–1.0). Lower = keep more of image."
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Path to AssetManifest JSON. When given, overrides the hardcoded CHARACTERS list.",
    )
    parser.add_argument(
        "--asset-id", type=str, default=None, dest="asset_id",
        help="Process only this asset_id (requires --manifest).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Manifest loader
# ---------------------------------------------------------------------------
def load_from_manifest(manifest_path: str, asset_id_filter):
    """Load character list from AssetManifest JSON (section: character_packs)."""
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    chars = []
    for pack in manifest.get("character_packs", []):
        if asset_id_filter and pack["asset_id"] != asset_id_filter:
            continue
        aid = pack["asset_id"]
        chars.append({
            "asset_id": aid,
            "input":    f"{aid}.png",
            "output":   f"{aid}-rgba.png",
        })
    return chars


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
def load_rmbg(device: str):
    """Load RMBG-1.4 from HuggingFace with trust_remote_code=True."""
    from transformers import AutoModelForImageSegmentation

    print(f"[MODEL] Loading {RMBG_MODEL_ID}...")
    model = AutoModelForImageSegmentation.from_pretrained(
        RMBG_MODEL_ID,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    print(f"[MODEL] RMBG-1.4 ready on {device}.")
    return model


# ---------------------------------------------------------------------------
# Preprocessing / postprocessing helpers
# ---------------------------------------------------------------------------
def preprocess(img: Image.Image) -> torch.Tensor:
    """Resize to 1024×1024, normalise, return [1, 3, H, W] tensor."""
    transform = T.Compose([
        T.Resize(INFERENCE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(img.convert("RGB")).unsqueeze(0)


def extract_mask(model_output, orig_w: int, orig_h: int, threshold: float) -> Image.Image:
    """
    Pull the final prediction from the model output, sigmoid it, resize to
    the original image dimensions, and binarise at `threshold`.
    Returns a single-channel PIL Image (mode "L") for use as alpha.
    """
    # RMBG-1.4 returns a list of predictions; use the last (most refined) one
    pred = model_output[-1].sigmoid().squeeze()   # shape: [H, W]
    pred_np = pred.cpu().float().numpy()
    # Scale to [0, 255]
    pred_np = (pred_np * 255).clip(0, 255).astype(np.uint8)
    mask = Image.fromarray(pred_np, mode="L")
    # Resize back to original resolution
    mask = mask.resize((orig_w, orig_h), Image.LANCZOS)
    # Binarise: pixels above threshold → fully opaque, else transparent
    threshold_uint8 = int(threshold * 255)
    mask = mask.point(lambda p: 255 if p >= threshold_uint8 else 0)
    return mask


# ---------------------------------------------------------------------------
# Per-image matting
# ---------------------------------------------------------------------------
def remove_background(
    model,
    input_path: Path,
    output_path: Path,
    device: str,
    threshold: float,
) -> int:
    """
    Run RMBG-1.4 on one image, apply the mask as alpha, save RGBA PNG.
    Returns the output file size in bytes.
    """
    img = Image.open(str(input_path)).convert("RGB")
    orig_w, orig_h = img.size

    # Preprocess → inference
    input_tensor = preprocess(img).to(device)
    with torch.no_grad():
        output = model(input_tensor)

    # Build alpha mask at original resolution
    alpha_mask = extract_mask(output, orig_w, orig_h, threshold)

    # Composite: paste alpha channel onto original RGB
    img_rgba = img.convert("RGBA")
    img_rgba.putalpha(alpha_mask)
    img_rgba.save(str(output_path), format="PNG")

    return output_path.stat().st_size


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    characters = CHARACTERS
    if args.manifest:
        characters = load_from_manifest(args.manifest, args.asset_id)
        if not characters:
            print("[WARN] No matching character_packs in manifest. Nothing to do.")
            return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] Using {device}")

    model = load_rmbg(device)

    results = []
    total = len(characters)

    for idx, char in enumerate(characters, start=1):
        input_path  = out_dir / char["input"]
        output_path = out_dir / char["output"]
        print(f"\n[{idx}/{total}] Matting {char['asset_id']}...")

        if output_path.exists():
            print(f"  [SKIP] {char['output']} already exists")
            results.append({
                "asset_id": char["asset_id"],
                "output": str(output_path),
                "size_bytes": output_path.stat().st_size,
                "status": "skipped",
            })
            continue

        if not input_path.exists():
            print(f"  [SKIP] Input not found: {input_path}")
            print(f"  [HINT] Run gen_character_images.py first.")
            results.append({
                "asset_id": char["asset_id"],
                "output": str(output_path),
                "size_bytes": 0,
                "status": "skipped",
                "error": f"Input missing: {input_path}",
            })
            continue

        try:
            size = remove_background(model, input_path, output_path, device, args.threshold)
            print(f"  [OK] {output_path}  ({size:,} bytes, RGBA)")
            results.append({
                "asset_id": char["asset_id"],
                "output": str(output_path),
                "size_bytes": size,
                "status": "success",
            })
        except Exception as exc:
            print(f"  [ERROR] {char['asset_id']}: {exc}")
            results.append({
                "asset_id": char["asset_id"],
                "output": str(output_path),
                "size_bytes": 0,
                "status": "failed",
                "error": str(exc),
            })
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    # Free model memory before next pipeline step
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Write manifest
    manifest_path = out_dir / f"{SCRIPT_NAME}_results.json"
    with open(manifest_path, "w") as fh:
        json.dump(results, fh, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — gen_character_mattes")
    print("=" * 60)
    for r in results:
        tag = "OK" if r["status"] == "success" else r["status"].upper()
        print(f"  [{tag}]  {r['output']}  ({r['size_bytes']:,} bytes)")
    ok_count = sum(1 for r in results if r["status"] in ("success", "skipped"))
    total_bytes = sum(r["size_bytes"] for r in results)
    print(f"\n{ok_count}/{total} completed | {total_bytes:,} bytes total")
    print(f"Manifest: {manifest_path}")
    print()
    print("NEXT STEP: run gen_upscale.py to 2× upscale the RGBA PNGs.")
    print("NOTE: animated MP4 matting (char-*-anim.mp4) is Phase 2.")


if __name__ == "__main__":
    main()
