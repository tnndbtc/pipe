# =============================================================================
# placeholder_for_sfx_medium.py
# Documents the GPU requirements for running gen_sfx.py with
# AudioGen medium (facebook/audiogen-medium) instead of the small variant.
#
# This script does NOT generate any assets. Run it to see what GPU is
# needed and what the quality difference looks like.
# =============================================================================

import sys

def main():
    print()
    print("=" * 70)
    print("QUALITY UPGRADE: AudioGen Medium -- Needs ~10-12 GB VRAM")
    print("=" * 70)
    print()
    print("Current (8 GB):  facebook/audiogen-small   (~2-4 GB VRAM)")
    print("Upgrade target:  facebook/audiogen-medium  (~10-12 GB VRAM)")
    print()
    print("WHY MEDIUM DOESN'T FIT ON 8 GB:")
    print("  AudioGen medium is a ~1.5B parameter transformer. In FP32 it")
    print("  requires ~6 GB for weights alone, plus ~4-6 GB of KV-cache")
    print("  activations during the autoregressive generation loop.")
    print("  Peak VRAM during generation exceeds 10 GB -- over the 8 GB budget.")
    print()
    print("QUALITY DIFFERENCE:")
    print("  Small:  Less detailed textures, simpler soundscapes,")
    print("          adequate for prototype review")
    print("  Medium: Richer spatial detail, more convincing layering,")
    print("          better suited for final pipeline output")
    print()
    print("-" * 70)
    print("GPU UPGRADE PATH:")
    print("-" * 70)
    print()
    print("  GPU             VRAM   AudioGen Model")
    print("  --------------- ------ ------------------------------------------")
    print("  RTX 4060 8 GB   8 GB   audiogen-small only (current)")
    print("  RTX 4080 16 GB  16 GB  audiogen-medium [ok]")
    print("  RTX 3090 24 GB  24 GB  audiogen-medium [ok] (headroom for batching)")
    print("  A100 40/80 GB   40 GB  audiogen-medium + large batch sizes")
    print()
    print("-" * 70)
    print("HOW TO SWITCH TO MEDIUM (when you have the GPU):")
    print("-" * 70)
    print()
    print("  Option A -- use the --model flag:")
    print("    python gen_sfx.py --model medium")
    print()
    print("  Option B -- change the default in gen_sfx.py:")
    print("    parser.add_argument('--model', ... default='medium' ...)")
    print()
    print("  No other code changes needed -- gen_sfx.py already supports both.")
    print()
    sys.exit(0)


if __name__ == "__main__":
    main()
