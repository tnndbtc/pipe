# code/ai/img2img/config.py

import torch

# --- Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model IDs (HuggingFace hub) ---
MODEL_IDS = {
    # SDXL (kept for ip_adapter mode only)
    "sdxl_base":          "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxl_inpaint":       "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    # Other non-FLUX models
    "sd_x4_upscaler":     "stabilityai/stable-diffusion-x4-upscaler",
    "depth_anything":     "depth-anything/Depth-Anything-V2-Small-hf",
    "rmbg":               "briaai/RMBG-1.4",   # load_rmbg() deleted (Fix 15); rembg manages its own session. Key kept for reference only.
    "ip_adapter_plus":    "h94/IP-Adapter",          # subfolder: sdxl_models
    "iclight_local":      "lllyasviel/ic-light",     # local weights if available
    # FLUX models
    "flux_fill":          "black-forest-labs/FLUX.1-Fill-dev",   # non-commercial
    "flux_schnell":       "black-forest-labs/FLUX.1-schnell",    # Apache 2.0
    "flux_dev":           "black-forest-labs/FLUX.1-dev",        # non-commercial
    "flux_controlnet_pose":  "InstantX/FLUX.1-dev-Controlnet-Pose",   # Apache 2.0
    "flux_controlnet_depth": "InstantX/FLUX.1-dev-Controlnet-Depth",  # Apache 2.0
    "flux_controlnet_canny": "InstantX/FLUX.1-dev-Controlnet-Canny",  # Apache 2.0
    "flux_redux":            "black-forest-labs/FLUX.1-Redux-dev",     # non-commercial
}

# --- VRAM Profiles ---
# Describes peak VRAM for each mode; all verified to fit RTX 4060 8 GB.
VRAM_PROFILES = {
    "composite":       {"peak_gb": 1.0,  "note": "Depth-Anything only; +5 GB with blend"},
    "inpaint":         {"peak_gb": 6.0,  "note": "FLUX.1-Fill-dev 4-bit quant + VAE tiling"},
    "outpaint":        {"peak_gb": 6.0,  "note": "FLUX.1-Fill-dev 4-bit quant + VAE tiling"},
    "style":           {"peak_gb": 6.0,  "note": "FLUX.1-schnell 4-bit quant + VAE tiling"},
    "pose":            {"peak_gb": 7.0,  "note": "FLUX.1-dev + ControlNet 4-bit quant + CPU offload (tight on 8 GB)"},
    "depth":           {"peak_gb": 7.0,  "note": "FLUX.1-dev + ControlNet 4-bit quant + CPU offload (tight on 8 GB)"},
    "canny":           {"peak_gb": 7.0,  "note": "FLUX.1-dev + ControlNet 4-bit quant + CPU offload (tight on 8 GB)"},
    "upscale":         {"peak_gb": 2.0,  "note": "SD x4 Upscaler FP16"},
    "bg_replace":      {"peak_gb": 1.0,  "note": "RMBG-1.4 only; +5 GB with blend"},
    "ip_adapter":      {"peak_gb": 6.5,  "note": "FLUX Redux prior (~600 MB GPU) + FLUX.1-dev img2img NF4 + CPU offload"},
    "relight":         {"peak_gb": 2.0,  "note": "IC-Light local FP16; 0 GB for API mode"},
    "color_grade":     {"peak_gb": 0.0,  "note": "OpenCV Reinhard, no model"},
}

# --- Inference defaults per mode ---
DEFAULTS = {
    "composite":    {"blend_strength": 0.0, "depth_scale": True},
    "inpaint":      {"steps": 50, "guidance": 30.0},
    "outpaint":     {"steps": 50, "guidance": 30.0, "pixels": 256, "direction": "right"},
    "style":        {"steps": 4,  "guidance": 0.0,  "strength": 0.65},
    "pose":         {"steps": 28, "guidance": 3.5,  "strength": 0.80},
    "depth":        {"steps": 28, "guidance": 3.5,  "strength": 0.75},
    "canny":        {"steps": 28, "guidance": 3.5,  "strength": 0.80,
                                                     "low_threshold": 100,
                                                     "high_threshold": 200},
    "upscale":      {"scale": 4},
    "bg_replace":   {"blend_strength": 0.0},
    "ip_adapter":   {"steps": 28, "guidance": 3.5, "strength": 0.70},
    "relight":      {"mode_relight": "api"},   # "local" raises NotImplementedError until IC-Light weights are obtained
    "color_grade":  {},
}

# --- Output directories (relative to project root) ---
OUTPUT_DIRS = {
    "composite":   "assets/composites",
    "inpaint":     "assets/inpaint",
    "outpaint":    "assets/outpaint",
    "style":       "assets/styled",
    "pose":        "assets/pose",
    "depth":       "assets/depth",
    "canny":       "assets/canny",
    "upscale":     "assets/upscaled_diffusion",
    "bg_replace":  "assets/composites",
    "ip_adapter":  "assets/ip_adapted",
    "relight":     "assets/relit",
    "color_grade": "assets/color_graded",
}

# --- Negative prompt for SDXL modes (ip_adapter) ---
# FLUX models do not use negative_prompt — pass config.SDXL_NEGATIVE_PROMPT
# only to pipelines that remain on SDXL.
SDXL_NEGATIVE_PROMPT = (
    "blurry, low quality, deformed, extra limbs, bad anatomy, "
    "watermark, signature, artifacts, noisy, overexposed, underexposed"
)
