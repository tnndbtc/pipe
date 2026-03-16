# code/ai/img2img/io_utils.py

import logging
import os
from pathlib import Path
from PIL import Image

log = logging.getLogger(__name__)


def load_image(path: str | Path) -> Image.Image:
    """Load any image as RGBA. Raises FileNotFoundError if missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    img = Image.open(str(p)).convert("RGBA")
    log.debug(f"Loaded image {p} ({img.size[0]}x{img.size[1]})")
    return img


def load_mask(path: str | Path) -> Image.Image:
    """Load mask as grayscale L (0=keep, 255=fill/inpaint region)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Mask not found: {p}")
    mask = Image.open(str(p)).convert("L")
    log.debug(f"Loaded mask {p} ({mask.size[0]}x{mask.size[1]})")
    return mask


def save_image(img: Image.Image, path: str | Path) -> None:
    """Save image to path, creating parent dirs as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(p))
    log.info(f"Saved image → {p}")


def ensure_dir(path: str | Path) -> Path:
    """Create directory and all parents. Return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resize_to_multiple_of_8(img: Image.Image) -> Image.Image:
    """SDXL requires width and height divisible by 8."""
    w, h = img.size
    w = (w // 8) * 8
    h = (h // 8) * 8
    return img.resize((w, h), Image.LANCZOS) if (w, h) != img.size else img


def snap_to_flux(img: Image.Image) -> Image.Image:
    """Snap width and height to nearest multiple of 16 (FLUX VAE requirement)."""
    w, h = img.size
    nw = (w // 16) * 16
    nh = (h // 16) * 16
    return img.resize((nw, nh), Image.LANCZOS) if (nw, nh) != (w, h) else img


def resize_to_sdxl(img: Image.Image, target: int = 1024) -> Image.Image:
    """Resize image so shortest side = target (default 1024), keep aspect, snap to 8."""
    w, h = img.size
    scale = target / min(w, h)
    nw = int(w * scale) // 8 * 8
    nh = int(h * scale) // 8 * 8
    return img.resize((nw, nh), Image.LANCZOS)
