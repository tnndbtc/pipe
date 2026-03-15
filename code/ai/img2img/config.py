# code/ai/img2img/config.py

import torch

# --- Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model IDs (HuggingFace hub) ---
MODEL_IDS = {
    "sdxl_base":          "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxl_inpaint":       "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "sd_x4_upscaler":     "stabilityai/stable-diffusion-x4-upscaler",
    "depth_anything":     "depth-anything/Depth-Anything-V2-Small-hf",
    "rmbg":               "briaai/RMBG-1.4",   # load_rmbg() deleted (Fix 15); rembg manages its own session. Key kept for reference only.
    "controlnet_pose":    "thibaud/controlnet-openpose-sdxl-1.0",
    "controlnet_depth":   "diffusers/controlnet-depth-sdxl-1.0",
    "controlnet_canny":   "diffusers/controlnet-canny-sdxl-1.0",
    "ip_adapter_plus":    "h94/IP-Adapter",          # subfolder: sdxl_models
    "iclight_local":      "lllyasviel/ic-light",     # local weights if available
}

# --- VRAM Profiles ---
# Describes peak VRAM for each mode; all verified to fit RTX 4060 8 GB.
VRAM_PROFILES = {
    "composite":       {"peak_gb": 1.0,  "note": "Depth-Anything only; +5 GB with blend"},
    "inpaint":         {"peak_gb": 5.0,  "note": "SDXL Inpaint FP16 + CPU offload"},
    "outpaint":        {"peak_gb": 5.0,  "note": "SDXL Inpaint FP16 + CPU offload"},
    "style":           {"peak_gb": 5.0,  "note": "SDXL img2img FP16 + CPU offload"},
    "pose":            {"peak_gb": 6.5,  "note": "SDXL + ControlNet FP16 + CPU offload"},
    "depth":           {"peak_gb": 6.5,  "note": "SDXL + ControlNet FP16 + CPU offload"},
    "canny":           {"peak_gb": 6.0,  "note": "SDXL + ControlNet FP16 + CPU offload"},
    "upscale":         {"peak_gb": 2.0,  "note": "SD x4 Upscaler FP16"},
    "bg_replace":      {"peak_gb": 1.0,  "note": "RMBG-1.4 only; +5 GB with blend"},
    "ip_adapter":      {"peak_gb": 6.5,  "note": "SDXL + IP-Adapter-Plus FP16 + CPU offload"},
    "relight":         {"peak_gb": 2.0,  "note": "IC-Light local FP16; 0 GB for API mode"},
    "color_grade":     {"peak_gb": 0.0,  "note": "OpenCV Reinhard, no model"},
}

# --- Inference defaults per mode ---
DEFAULTS = {
    "composite":    {"blend_strength": 0.0, "depth_scale": True},
    "inpaint":      {"steps": 30, "guidance": 7.5, "strength": 0.99},
    "outpaint":     {"steps": 30, "guidance": 7.5, "pixels": 256, "direction": "right"},
    "style":        {"steps": 30, "guidance": 7.5, "strength": 0.65},
    "pose":         {"steps": 30, "guidance": 7.5, "strength": 0.80},
    "depth":        {"steps": 30, "guidance": 7.5, "strength": 0.75},
    "canny":        {"steps": 30, "guidance": 7.5, "strength": 0.80,
                                                     "low_threshold": 100,
                                                     "high_threshold": 200},
    "upscale":      {"scale": 4},
    "bg_replace":   {"blend_strength": 0.0},
    "ip_adapter":   {"steps": 30, "guidance": 5.0, "ip_scale": 0.6},
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

# --- Negative prompt used by all SDXL modes ---
NEGATIVE_PROMPT = (
    "blurry, low quality, deformed, extra limbs, bad anatomy, "
    "watermark, signature, artifacts, noisy, overexposed, underexposed"
)
