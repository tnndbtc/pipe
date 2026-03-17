# code/ai/img2img/model_loader.py

import gc
import logging
import torch
from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionUpscalePipeline,
)
from transformers import pipeline as hf_pipeline
from img2img.config import MODEL_IDS, DEVICE

log = logging.getLogger(__name__)


def _fp16_kwargs(no_fp16: bool = False) -> dict:
    if no_fp16 or DEVICE != "cuda":
        return {}
    return {"torch_dtype": torch.float16}


def unload_model(pipe) -> None:
    if pipe is None:
        return
    del pipe
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    log.info("Model unloaded, VRAM cleared.")


def load_sdxl_img2img(no_fp16: bool = False) -> StableDiffusionXLImg2ImgPipeline:
    log.info("Loading SDXL img2img...")
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_IDS["sdxl_base"], **_fp16_kwargs(no_fp16)
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    log.info("SDXL img2img ready.")
    return pipe


def load_sd_upscaler(no_fp16: bool = False) -> StableDiffusionUpscalePipeline:
    log.info("Loading SD x4 Upscaler...")
    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        MODEL_IDS["sd_x4_upscaler"], **_fp16_kwargs(no_fp16)
    )
    if DEVICE == "cuda":
        pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    log.info("SD x4 Upscaler ready.")
    return pipe


def load_depth_model():
    """Depth Anything V2 Small — ~1 GB. No FP16 flag (transformers pipeline manages dtype)."""
    log.info("Loading Depth Anything V2 Small...")
    depth_pipe = hf_pipeline(
        task="depth-estimation",
        model=MODEL_IDS["depth_anything"],
        device=0 if DEVICE == "cuda" else -1,
    )
    log.info("Depth model ready.")
    return depth_pipe


def load_flux_fill() -> "FluxFillPipeline":
    """FLUX.1-Fill-dev with 4-bit BnB quantisation + VAE tiling.
    Fits in ~6 GB VRAM on RTX 4060. Needs HF login (gated model).
    """
    from diffusers import FluxFillPipeline, FluxTransformer2DModel
    from transformers import BitsAndBytesConfig

    log.info("Loading FLUX.1-Fill-dev (4-bit quant + VAE tiling)...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_IDS["flux_fill"],
        subfolder="transformer",
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
    )
    pipe = FluxFillPipeline.from_pretrained(
        MODEL_IDS["flux_fill"],
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()
    log.info("FLUX.1-Fill ready.")
    return pipe


def load_flux_img2img():
    """FLUX.1-schnell img2img with 4-bit BnB quant + VAE tiling.
    guidance_scale=0.0 (CFG-free distilled model). No negative_prompt.
    """
    from diffusers import FluxImg2ImgPipeline, FluxTransformer2DModel
    from transformers import BitsAndBytesConfig

    log.info("Loading FLUX.1-schnell img2img (4-bit quant + VAE tiling)...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_IDS["flux_schnell"],
        subfolder="transformer",
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
    )
    pipe = FluxImg2ImgPipeline.from_pretrained(
        MODEL_IDS["flux_schnell"],
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()
    log.info("FLUX.1-schnell img2img ready.")
    return pipe


def load_flux_controlnet(controlnet_id: str):
    """FLUX.1-dev + InstantX ControlNet with 4-bit BnB quant + VAE tiling.
    Uses FluxControlNetImg2ImgPipeline so the source image is preserved.
    ~7 GB VRAM — tight on 8 GB; reduce controlnet_conditioning_scale if OOM.
    """
    from diffusers import FluxControlNetImg2ImgPipeline, FluxControlNetModel, FluxTransformer2DModel
    from transformers import BitsAndBytesConfig

    log.info(f"Loading FLUX ControlNet: {controlnet_id}")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    controlnet = FluxControlNetModel.from_pretrained(
        controlnet_id,
        torch_dtype=torch.bfloat16,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_IDS["flux_dev"],
        subfolder="transformer",
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
    )
    pipe = FluxControlNetImg2ImgPipeline.from_pretrained(
        MODEL_IDS["flux_dev"],
        transformer=transformer,
        controlnet=controlnet,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()
    log.info("FLUX ControlNet ready.")
    return pipe


def load_flux_redux_prior():
    """FLUX.1-Redux-dev prior — encodes a reference image to visual embeddings.
    ~600 MB VRAM. Stays on GPU alongside the FLUX.1-dev img2img pipe.
    Needs HF login (gated model).
    """
    from diffusers import FluxPriorReduxPipeline

    log.info("Loading FLUX.1-Redux-dev prior...")
    pipe = FluxPriorReduxPipeline.from_pretrained(
        MODEL_IDS["flux_redux"],
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)
    log.info("FLUX Redux prior ready.")
    return pipe


def load_sdxl_inpaint(no_fp16: bool = False):
    """SDXL inpainting pipeline (~5 GB VRAM with FP16).
    Patch-based with prefill — see inpaint.py for strategy details.
    """
    from diffusers import StableDiffusionXLInpaintPipeline

    log.info("Loading SDXL Inpaint...")
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        MODEL_IDS["sdxl_inpaint"], **_fp16_kwargs(no_fp16)
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    log.info("SDXL Inpaint ready.")
    return pipe


def load_lama():
    """LaMa inpainting (simple-lama-inpainting). ~200 MB, CPU or GPU.
    Pure texture continuation — no diffusion, no generation.
    Ideal for architectural backgrounds: stone, brick, tile, wood.

    Install: pip install simple-lama-inpainting
    """
    try:
        from simple_lama_inpainting import SimpleLama
    except ImportError:
        raise ImportError(
            "simple-lama-inpainting is not installed.\n"
            "  pip install simple-lama-inpainting"
        )
    log.info("Loading LaMa inpainting model...")
    lama = SimpleLama()
    log.info("LaMa ready.")
    return lama


def load_flux_dev_img2img():
    """FLUX.1-dev img2img with 4-bit BnB quant + VAE tiling.
    Paired with load_flux_redux_prior() for style-transfer-with-reference.
    guidance_scale=3.5 typical (CFG enabled on dev, unlike schnell).
    Needs HF login (gated model).
    """
    from diffusers import FluxImg2ImgPipeline, FluxTransformer2DModel
    from transformers import BitsAndBytesConfig

    log.info("Loading FLUX.1-dev img2img (4-bit quant + VAE tiling)...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_IDS["flux_dev"],
        subfolder="transformer",
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
    )
    pipe = FluxImg2ImgPipeline.from_pretrained(
        MODEL_IDS["flux_dev"],
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()
    log.info("FLUX.1-dev img2img ready.")
    return pipe
