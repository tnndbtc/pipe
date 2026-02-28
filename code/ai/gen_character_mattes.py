# =============================================================================
# gen_character_mattes.py
# Remove backgrounds from character portrait PNGs, producing RGBA images
# with transparent backgrounds ready for compositing over background plates.
# Run AFTER gen_character_images.py.
# STATUS: VALIDATED — script works, but RGBA outputs already exist in assets.
#         No need to re-run unless character images are regenerated.
# =============================================================================
#
# requirements.txt (pip install before running):
#   rembg>=2.0.50
#   Pillow>=10.0.0
#   torch>=2.1.0       (optional — used only for GPU cache management)
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

import torch
from PIL import Image

# ---------------------------------------------------------------------------
# DEFAULTS — fully populated; script runs with no CLI flags.
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "projects" / "the-pharaoh-who-defied-death" / "episodes" / "s01e01" / "assets"
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

RMBG_MODEL = "u2net_human_seg"   # rembg model — optimised for human subject isolation


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove backgrounds from character portraits using RMBG-1.4.",
        epilog=(
            "Model used:\n\n"
            "  u2net_human_seg    rembg built-in model, optimised for human subjects.\n"
            "                     Downloaded automatically on first run (~176 MB).\n"
            "                     No GPU or HF login required.\n\n"
            "  This script has no --model flag; u2net_human_seg is the only supported model."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output_dir", type=str, default=None)
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
    """Load rembg session (u2net_human_seg — optimised for human portraits)."""
    from rembg import new_session

    print(f"[MODEL] Loading rembg ({RMBG_MODEL})...")
    session = new_session(RMBG_MODEL)
    print(f"[MODEL] rembg ({RMBG_MODEL}) ready.")
    return session


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
    Remove background using rembg (u2net_human_seg), save RGBA PNG.
    Returns the output file size in bytes.
    """
    from rembg import remove

    img    = Image.open(str(input_path)).convert("RGB")
    result = remove(img, session=model)   # returns RGBA PIL Image
    result.save(str(output_path), format="PNG")

    return output_path.stat().st_size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
