# =============================================================================
# placeholder_for_background_video.py
# Documents the GPU requirements for running gen_background_video.py with
# the originally specified LTX-Video 0.9.8-13B distilled model.
#
# This script does NOT generate any assets. Run it to see what GPU is
# needed and what the upgrade path looks like.
# =============================================================================

import sys

def main():
    print()
    print("=" * 70)
    print("BLOCKED: LTX-Video 0.9.8-13B -- Cannot run on RTX 4060 8 GB")
    print("=" * 70)
    print()
    print("Model:   Lightricks/LTX-Video-0.9.8-13B-distilled")
    print("Params:  13 billion")
    print()
    print("VRAM breakdown:")
    print("  Weights (FP8, 1 byte/param):  ~13 GB   -- already exceeds 8 GB budget")
    print("  Activations (spatiotemporal): ~8-12 GB extra during forward pass")
    print("  Total peak:                   ~24 GB")
    print()
    print("WHY CPU OFFLOAD DOESN'T HELP HERE:")
    print("  enable_model_cpu_offload() reduces *idle* VRAM, but the transformer")
    print("  forward pass for 13B params at 768x432x49 frames still requires")
    print("  ~20+ GB on-GPU simultaneously. The pass OOMs before completing.")
    print("  Sequential offload would yield ~1-3 frames/min (unusable).")
    print()
    print("-" * 70)
    print("CURRENT 8 GB FALLBACK (active in gen_background_video.py):")
    print("-" * 70)
    print()
    print("  Model:   THUDM/CogVideoX-2b")
    print("  VRAM:    ~6 GB (FP16 + CPU offload)")
    print("  Quality: Lower -- 2B param model vs 13B")
    print("  Run:     python gen_background_video.py")
    print()
    print("-" * 70)
    print("GPU UPGRADE PATH:")
    print("-" * 70)
    print()
    print("  GPU             VRAM   Unlocks")
    print("  --------------- ------ ------------------------------------------")
    print("  RTX 4060 8 GB   8 GB   CogVideoX-2b only (current)")
    print("  RTX 4080 16 GB  16 GB  CogVideoX-5B (5B params, better quality)")
    print("  RTX 3090 24 GB  24 GB  LTX-Video 0.9.8-13B FP8 + CPU offload")
    print("  RTX 4090 24 GB  24 GB  LTX-Video 0.9.8-13B FP8 (faster than 3090)")
    print("  A100 40/80 GB   40 GB  LTX-Video full precision, no offloading")
    print()
    print("-" * 70)
    print("INSTALL NOTES FOR LTX-VIDEO (when you have the GPU):")
    print("-" * 70)
    print()
    print("  pip install diffusers>=0.32.0 torch>=2.4.1")
    print()
    print("  In gen_background_video.py, replace the CogVideoXPipeline block with:")
    print()
    print("    from diffusers import LTXPipeline")
    print("    pipe = LTXPipeline.from_pretrained(")
    print('        "Lightricks/LTX-Video-0.9.8-13B-distilled",')
    print("        torch_dtype=torch.bfloat16,")
    print("    )")
    print("    pipe.enable_model_cpu_offload()")
    print()
    print("  Inference steps: 4 (distilled model)")
    print("  Resolution:      768x432 at 8fps")
    print("  HuggingFace:     https://huggingface.co/Lightricks/LTX-Video-0.9.8-13B-distilled")
    print()
    sys.exit(0)


if __name__ == "__main__":
    main()
