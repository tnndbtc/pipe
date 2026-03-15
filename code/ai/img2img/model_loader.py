# code/ai/img2img/model_loader.py

import gc
import logging
import torch
from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionUpscalePipeline,
    ControlNetModel,
    StableDiffusionXLControlNetImg2ImgPipeline,
)
from transformers import pipeline as hf_pipeline
from img2img.config import MODEL_IDS, DEVICE

log = logging.getLogger(__name__)


def _fp16_kwargs() -> dict:
    return {"torch_dtype": torch.float16} if DEVICE == "cuda" else {}


def unload_model(pipe) -> None:
    """Delete pipeline and free VRAM."""
    if pipe is None:
        return
    del pipe
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    log.info("Model unloaded, VRAM cleared.")


def load_sdxl_img2img() -> StableDiffusionXLImg2ImgPipeline:
    """SDXL img2img — ~5 GB peak with CPU offload. Used by: style, composite blend."""
    log.info("Loading SDXL img2img...")
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_IDS["sdxl_base"], **_fp16_kwargs()
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    log.info("SDXL img2img ready.")
    return pipe


def load_sdxl_inpaint() -> StableDiffusionXLInpaintPipeline:
    """SDXL Inpainting — ~5 GB peak. Used by: inpaint, outpaint."""
    log.info("Loading SDXL Inpaint...")
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        MODEL_IDS["sdxl_inpaint"], **_fp16_kwargs()
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    log.info("SDXL Inpaint ready.")
    return pipe


def load_sdxl_controlnet(controlnet_id: str) -> StableDiffusionXLControlNetImg2ImgPipeline:
    """SDXL + ControlNet — ~6.5 GB peak. Used by: pose, depth, canny."""
    log.info(f"Loading ControlNet: {controlnet_id}")
    controlnet = ControlNetModel.from_pretrained(controlnet_id, **_fp16_kwargs())
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        MODEL_IDS["sdxl_base"], controlnet=controlnet, **_fp16_kwargs()
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    log.info("SDXL + ControlNet ready.")
    return pipe


def load_sd_upscaler() -> StableDiffusionUpscalePipeline:
    """SD x4 Upscaler — ~2 GB peak. Used by: upscale_diffusion."""
    log.info("Loading SD x4 Upscaler...")
    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        MODEL_IDS["sd_x4_upscaler"], **_fp16_kwargs()
    )
    if DEVICE == "cuda":
        pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    log.info("SD x4 Upscaler ready.")
    return pipe


def load_depth_model():
    """Depth Anything V2 Small — ~1 GB. Used by: composite, depth_control."""
    log.info("Loading Depth Anything V2 Small...")
    depth_pipe = hf_pipeline(
        task="depth-estimation",
        model=MODEL_IDS["depth_anything"],
        device=0 if DEVICE == "cuda" else -1,
    )
    log.info("Depth model ready.")
    return depth_pipe


def load_ip_adapter(sdxl_pipe: StableDiffusionXLImg2ImgPipeline):
    """Attach IP-Adapter-Plus weights to an existing SDXL pipe (~1 GB extra)."""
    log.info("Loading IP-Adapter-Plus...")
    sdxl_pipe.load_ip_adapter(
        MODEL_IDS["ip_adapter_plus"],
        subfolder="sdxl_models",
        weight_name="ip-adapter-plus_sdxl_vit-h.bin",
    )
    log.info("IP-Adapter-Plus attached.")
    return sdxl_pipe
