"""
scorer.py — CLIP-based image and video scoring for code/media/http/

All scoring logic lives here. Server calls these functions via
asyncio.to_thread() so the FastAPI event loop stays responsive during
CPU-bound CLIP inference.

Public API
----------
load_clip(cfg)                                         → ClipModel
score_images(clipm, item, img_paths, weights, config)  → list[dict]
score_videos(clipm, item, vid_paths, batch_dir, weights, config) → list[dict]

Backward compatibility
----------------------
Both score_* functions accept a plain string for `item` (old ai_prompt
parameter) — it is promoted to {"ai_prompt": item} automatically.

Image result shape:
    {"path": str, "score": float, "score_detail": {...}}

Video result shape:
    {"path": str, "score": float, "clip_score": float,
     "calmness": float, "score_detail": {...}}
    or on error:
    {"path": str, "score": 0.0, "error": str}

All result lists are sorted by score descending.
"""

from __future__ import annotations

import json
import logging
import subprocess
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

log = logging.getLogger("scorer")

# ---------------------------------------------------------------------------
# Frame-sampling constants (match code/media/test.py)
# ---------------------------------------------------------------------------

VIDEO_SAMPLE_FPS = 0.5   # 1 frame every 2 s
VIDEO_MAX_FRAMES = 24    # cap per video
MOTION_SCALE     = 18.0  # larger → more forgiving calmness for same motion

# ---------------------------------------------------------------------------
# Multi-dimensional scoring constants
# ---------------------------------------------------------------------------

DIMS = ["subjects", "environment", "style", "motion", "technical"]

BASE_WEIGHTS: dict[str, float] = {
    "subjects":    0.40,
    "environment": 0.25,
    "style":       0.20,
    "motion":      0.10,
    "technical":   0.05,
}

# cinematic_role → per-dimension weight deltas
CINEMATIC_ROLE_DELTAS: dict[str, dict[str, float]] = {
    "establish": {
        "subjects":    -0.10,
        "environment": +0.10,
    },
    "hold": {
        "subjects":    +0.10,
        "environment": -0.05,
        "motion":      -0.05,
    },
    "transition": {
        "subjects":    -0.10,
        "style":       -0.05,
        "motion":      +0.15,
    },
    "emotional_support": {
        "subjects":    +0.05,
        "environment": -0.10,
        "style":       +0.10,
        "motion":      -0.05,
    },
    "atmosphere": {
        "subjects":    -0.15,
        "environment": +0.10,
        "style":       +0.10,
        "motion":      -0.05,
    },
}

# motion_level → max calmness threshold (None = no filter)
MOTION_LEVEL_CALM_THRESHOLD: dict[str, float | None] = {
    "none":     0.75,
    "very_low": 0.75,
    "low":      0.60,
    "medium":   0.45,
    "high":     None,
}
MOTION_LEVEL_DEFAULT_THRESHOLD: float = 0.55

# Content-profile image calmness penalty threshold (soft — not a gate).
# None = no calmness penalty for images.  sleep_story enforces calm images;
# documentary/default disable the penalty so ruins/historical photos are not rejected.
IMAGE_PROFILE_CALM_THRESHOLD: dict[str, float | None] = {
    "sleep_story": 0.55,
    "documentary": None,
    "default":     None,
    "action":      None,
}

# lighting → extra style hint string
LIGHTING_STYLE_HINTS: dict[str, str] = {
    "soft_night":       "soft blue moonlight, night time, low key",
    "low_key_soft":     "low key soft lighting, dark atmosphere",
    "warm_dawn":        "warm golden sunrise light, dawn colors",
    "diffused_daylight":"diffused daylight, soft natural light, overcast",
    "bright_open":      "bright open light, high key, clear sky",
    "dramatic":         "dramatic high contrast lighting, strong shadows",
}

# Default named scoring profiles (may be overridden from config.json)
DEFAULT_SCORING_PROFILES: dict[str, dict[str, float]] = {
    "default":     {"subjects": 0.40, "environment": 0.25, "style": 0.20, "motion": 0.10, "technical": 0.05},
    "sleep_story": {"subjects": 0.20, "environment": 0.40, "style": 0.25, "motion": 0.05, "technical": 0.10},
    "documentary": {"subjects": 0.45, "environment": 0.20, "style": 0.15, "motion": 0.15, "technical": 0.05},
    "action":      {"subjects": 0.35, "environment": 0.15, "style": 0.20, "motion": 0.25, "technical": 0.05},
}

# Optional heavy deps — gracefully unavailable
try:
    import cv2          # noqa: PLC0415
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None          # type: ignore[assignment]
    _CV2_AVAILABLE = False
    log.warning("cv2 not available — thumbnail filters and HOG person detection disabled")

try:
    import imagehash                    # noqa: PLC0415
    _IMAGEHASH_AVAILABLE = True
except ImportError:
    imagehash = None                    # type: ignore[assignment]
    _IMAGEHASH_AVAILABLE = False
    log.warning("imagehash not available — pHash deduplication disabled")


# ---------------------------------------------------------------------------
# ClipModel container
# ---------------------------------------------------------------------------

@dataclass
class ClipModel:
    model:      Any
    preprocess: Any
    tokenizer:  Any
    device:     str


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_clip(cfg: dict) -> ClipModel:
    """Load CLIP model once at server startup.  CPU-only."""
    model_name = cfg.get("clip_model",      "ViT-B-32")
    pretrained  = cfg.get("clip_pretrained", "laion2b_s34b_b79k")
    device      = "cpu"

    log.info("Loading CLIP %s / %s on %s …", model_name, pretrained, device)
    try:
        import open_clip  # noqa: PLC0415
        import torch      # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "Missing deps for CLIP. "
            "Run: pip install torch open_clip_torch"
        ) from exc

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    log.info("CLIP loaded: %s / %s on %s", model_name, pretrained, device)

    # Warmup: trigger MKL/BLAS thread-pool init and PyTorch dispatch-cache population
    # so the first real scoring shot isn't penalised by one-time CPU init overhead.
    log.info("Warming up CLIP inference …")
    with torch.no_grad():
        dummy_img = torch.zeros(1, 3, 224, 224)
        dummy_tok = tokenizer(["warmup"])
        model.encode_image(dummy_img)
        model.encode_text(dummy_tok)
    log.info("CLIP warmup done.")

    return ClipModel(model=model, preprocess=preprocess, tokenizer=tokenizer, device=device)


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------

def _resolve_weights(item: dict, config: dict | None) -> dict[str, float]:
    """
    Build the final per-dimension weight dict by:
      1. Starting from BASE_WEIGHTS
      2. Applying the named scoring profile (from config["content_profile"])
      3. Applying cinematic_role deltas from item
      4. Clamping to >= 0 and renormalising to sum 1.0
    """
    cfg = config or {}

    # Step 1: base weights
    weights = dict(BASE_WEIGHTS)

    # Step 2: named profile overrides
    profiles = cfg.get("scoring_profiles", DEFAULT_SCORING_PROFILES)
    profile_name = cfg.get("content_profile", "default")
    profile = profiles.get(profile_name, profiles.get("default", BASE_WEIGHTS))
    weights.update(profile)

    # Step 3: cinematic_role deltas
    role = item.get("cinematic_role", "")
    if role in CINEMATIC_ROLE_DELTAS:
        for dim, delta in CINEMATIC_ROLE_DELTAS[role].items():
            weights[dim] = weights.get(dim, 0.0) + delta

    # Step 4: clamp and renormalise
    weights = {k: max(0.0, v) for k, v in weights.items()}
    total = sum(weights.values()) or 1.0
    weights = {k: v / total for k, v in weights.items()}
    return weights


def _motion_level_threshold(item: dict) -> float | None:
    """Return the calmness rejection threshold for this item's motion_level."""
    ml = item.get("motion_level")
    if ml is None:
        return MOTION_LEVEL_DEFAULT_THRESHOLD
    return MOTION_LEVEL_CALM_THRESHOLD.get(ml, MOTION_LEVEL_DEFAULT_THRESHOLD)


def _image_calmness_threshold(item: dict, cfg: dict) -> float | None:
    """
    Return the image calmness penalty threshold for this item.
    Returns None to disable calmness penalty for images (the default).
    Driven by content_profile: sleep_story enforces a threshold; documentary/default do not.
    Can be overridden per profile in config.json scoring_profiles["image_calmness_threshold"].
    """
    profile = cfg.get("content_profile", "default")
    profiles_cfg = cfg.get("scoring_profiles") or {}
    profile_data = profiles_cfg.get(profile) or {}
    if "image_calmness_threshold" in profile_data:
        return profile_data["image_calmness_threshold"]
    return IMAGE_PROFILE_CALM_THRESHOLD.get(profile)


def _video_calmness_threshold(item: dict, cfg: dict) -> float | None:
    """
    Return the video calmness penalty threshold for this item.
    Falls back to the existing motion_level table (_motion_level_threshold).
    Can be overridden per profile in config.json scoring_profiles["video_calmness_threshold"].
    """
    profile = cfg.get("content_profile", "default")
    profiles_cfg = cfg.get("scoring_profiles") or {}
    profile_data = profiles_cfg.get(profile) or {}
    if "video_calmness_threshold" in profile_data:
        return profile_data["video_calmness_threshold"]
    return _motion_level_threshold(item)


# ---------------------------------------------------------------------------
# CLIP helpers
# ---------------------------------------------------------------------------

def _clip_encode_image(clipm: ClipModel, img_path: Path):
    """Return normalised CLIP image embedding tensor (1 × D) for one image."""
    import torch          # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    img   = Image.open(img_path).convert("RGB")
    img_t = clipm.preprocess(img).unsqueeze(0).to(clipm.device)
    with torch.no_grad():
        img_f = clipm.model.encode_image(img_t)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
    return img_f  # (1, D)


def _clip_encode_text(clipm: ClipModel, text: str):
    """Return normalised CLIP text embedding tensor (1 × D)."""
    import torch  # noqa: PLC0415

    with torch.no_grad():
        tok = clipm.tokenizer([text]).to(clipm.device)
        tf  = clipm.model.encode_text(tok)
        tf  = tf / tf.norm(dim=-1, keepdim=True)
    return tf  # (1, D)


def _clip_scores(clipm: ClipModel, prompt: str, image_paths: list[Path]) -> dict[str, float]:
    """Return {str(path): cosine_similarity} for a single text prompt."""
    if not image_paths:
        return {}

    import torch          # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    with torch.no_grad():
        text_tokens   = clipm.tokenizer([prompt]).to(clipm.device)
        text_features = clipm.model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        scores: dict[str, float] = {}
        for p in image_paths:
            try:
                img   = Image.open(p).convert("RGB")
                img_t = clipm.preprocess(img).unsqueeze(0).to(clipm.device)
                img_f = clipm.model.encode_image(img_t)
                img_f = img_f / img_f.norm(dim=-1, keepdim=True)
                scores[str(p)] = float((img_f @ text_features.T).item())
            except Exception as exc:  # noqa: BLE001
                log.warning("CLIP failed for %s: %s", p.name, exc)
                scores[str(p)] = 0.0

    return scores


def _clip_precompute_text_embeddings(
    clipm: ClipModel,
    hints: dict[str, list[str]],   # dim → list of hint texts
) -> dict[str, list]:              # dim → list of normalised embedding tensors
    """
    Pre-compute and normalise CLIP text embeddings for all hint texts.
    Call once per shot (outside the per-image loop) to avoid redundant encodes.
    """
    import torch  # noqa: PLC0415

    result: dict[str, list] = {}
    with torch.no_grad():
        for dim, texts in hints.items():
            embeddings = []
            for t in texts:
                try:
                    tok = clipm.tokenizer([t]).to(clipm.device)
                    tf  = clipm.model.encode_text(tok)
                    tf  = tf / tf.norm(dim=-1, keepdim=True)
                    embeddings.append(tf)
                except Exception as exc:  # noqa: BLE001
                    log.warning("CLIP text encode failed (%s): %s", t[:40], exc)
            result[dim] = embeddings
    return result


def _clip_score_multidim(
    clipm:   ClipModel,
    img_f,                           # pre-computed image embedding (1×D tensor)
    hints:   dict[str, list[str]],   # dim → list of hint texts
    weights: dict[str, float],
    text_embeddings: dict[str, list] | None = None,  # pre-computed via _clip_precompute_text_embeddings
) -> tuple[float, dict[str, float]]:
    """
    Compute weighted multi-dim CLIP score.

    For each dimension, take the max cosine similarity across its hint texts.
    Pass pre-computed text_embeddings (from _clip_precompute_text_embeddings) to
    avoid redundant encode_text calls when scoring many images for the same shot.
    Returns (combined_score, {dim: score}).
    """
    import torch  # noqa: PLC0415

    dim_scores: dict[str, float] = {}
    with torch.no_grad():
        for dim, texts in hints.items():
            if not texts:
                dim_scores[dim] = 0.0
                continue
            best = 0.0
            # Use pre-computed embeddings when available
            pre = (text_embeddings or {}).get(dim)
            if pre:
                for tf in pre:
                    try:
                        sim = float((img_f @ tf.T).item())
                        if sim > best:
                            best = sim
                    except Exception as exc:  # noqa: BLE001
                        log.warning("CLIP sim failed for dim %s: %s", dim, exc)
            else:
                # Fallback: encode on the fly (slower, used when no pre-computed cache)
                for t in texts:
                    try:
                        tok = clipm.tokenizer([t]).to(clipm.device)
                        tf  = clipm.model.encode_text(tok)
                        tf  = tf / tf.norm(dim=-1, keepdim=True)
                        sim = float((img_f @ tf.T).item())
                        if sim > best:
                            best = sim
                    except Exception as exc:  # noqa: BLE001
                        log.warning("CLIP text encode failed (%s): %s", t[:40], exc)
            dim_scores[dim] = best

    combined = sum(weights.get(d, 0.0) * s for d, s in dim_scores.items())
    return combined, dim_scores


# ---------------------------------------------------------------------------
# Image quality helpers
# ---------------------------------------------------------------------------

def _luma_contrast(img_gray_np) -> tuple[float, float]:
    """Return (mean_luma, rms_contrast) from a uint8 grayscale numpy array."""
    import numpy as np  # noqa: PLC0415

    flat = img_gray_np.astype(np.float32) / 255.0
    return float(flat.mean()), float(flat.std())


def _edge_density(img_gray_np) -> float:
    """Canny edge pixel fraction [0, 1]."""
    if not _CV2_AVAILABLE:
        return 0.0
    edges = cv2.Canny(img_gray_np, 50, 150)
    return float(edges.mean()) / 255.0


def _hog_person_score(img_bgr_np) -> float:
    """
    HOG person detection — DISABLED.

    cv2.HOGDescriptor.detectMultiScale() causes SIGSEGV in the worker process
    on certain thumbnail images (signal — uncatchable by try/except).
    Person detection is a soft quality filter; returning 0.0 is safe.
    """
    return 0.0


def _hue_hist_16(img_bgr_np) -> list[float]:
    """16-bin normalised hue histogram from BGR image.  Returns zeros if cv2 unavailable."""
    if not _CV2_AVAILABLE:
        return [0.0] * 16
    import numpy as np  # noqa: PLC0415

    hsv = cv2.cvtColor(img_bgr_np, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    hist = hist.flatten()
    total = hist.sum() or 1.0
    return [float(v / total) for v in hist]


def _calmness_from_frames(frames: list[Path]) -> float:
    """Calmness in [0, 1].  Higher = calmer (less motion between consecutive frames)."""
    if len(frames) < 2:
        return 1.0
    if not _CV2_AVAILABLE:
        return 1.0

    import numpy as np  # noqa: PLC0415

    diffs: list[float] = []
    prev = None
    for fp in frames:
        img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if prev is not None:
            diffs.append(float(np.mean(cv2.absdiff(img, prev))))
        prev = img

    if not diffs:
        return 1.0
    mean_diff = sum(diffs) / len(diffs)
    return max(0.0, min(1.0, 1.0 - mean_diff / MOTION_SCALE))


# keep old name as alias for internal video use
_calmness = _calmness_from_frames


def _calmness_single(img_path: Path) -> float:
    """
    Estimate 'calmness' for a single image using Laplacian variance.
    High variance → sharp / busy → lower calmness.
    The scale is tuned so a near-uniform image → ~1.0 and a very busy
    image → near 0.0.

    Uses PIL + numpy instead of cv2.imread so unusual formats from
    cultural-heritage sources (TIFF variants, old JPEG) raise Python
    exceptions rather than SIGSEGVing the worker process.
    """
    try:
        import numpy as np      # noqa: PLC0415
        from PIL import Image   # noqa: PLC0415

        with Image.open(img_path) as pil:
            gray = np.array(pil.convert("L"), dtype=np.float64)

        # 5-point Laplacian on interior pixels — same kernel as cv2.Laplacian
        g = gray
        lap = (
            g[:-2, 1:-1] + g[2:, 1:-1] +
            g[1:-1, :-2] + g[1:-1, 2:] -
            4.0 * g[1:-1, 1:-1]
        )
        lap_var = float(lap.var())
        return max(0.0, min(1.0, 1.0 - lap_var / 500.0))
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Thumbnail pass-1.5
# ---------------------------------------------------------------------------

def _download_thumbnail(url: str) -> bytes | None:
    """Download a thumbnail URL; returns raw bytes or None on failure."""
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.read()
    except Exception as exc:  # noqa: BLE001
        log.debug("Thumbnail download failed (%s): %s", url, exc)
        return None


def _thumbnail_filter(
    url: str,
    tf_cfg: dict,
) -> tuple[bool, dict]:
    """
    Download thumbnail, run cheap quality filters.

    Returns (passes: bool, detail: dict).
    detail keys: mean_luma, rms_contrast, edge_density, person_score.
    If download fails → (True, {}) — don't reject.
    """
    import numpy as np      # noqa: PLC0415
    from PIL import Image   # noqa: PLC0415

    raw = _download_thumbnail(url)
    if raw is None:
        return True, {}

    try:
        pil = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        log.debug("Thumbnail decode failed: %s", exc)
        return True, {}

    # Convert to numpy for cv2 operations
    import numpy as np  # noqa: PLC0415

    img_np = np.array(pil)                           # H×W×3 RGB uint8
    gray   = np.array(pil.convert("L"))              # H×W uint8

    # Basic luma / contrast
    mean_luma, rms_contrast = _luma_contrast(gray)

    # Edge density (cv2 Canny)
    ed = _edge_density(gray)

    # HOG person score
    if _CV2_AVAILABLE:
        bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        ps  = _hog_person_score(bgr)
    else:
        bgr = None
        ps  = 0.0

    detail = {
        "mean_luma":    mean_luma,
        "rms_contrast": rms_contrast,
        "edge_density": ed,
        "person_score": ps,
    }

    # Apply filters
    min_luma    = tf_cfg.get("min_luma",              0.05)
    max_luma    = tf_cfg.get("max_luma",              0.85)
    min_contrast= tf_cfg.get("min_contrast",          0.05)
    max_ed      = tf_cfg.get("max_edge_density",      0.15)
    max_ps      = tf_cfg.get("max_person_score",      0.05)

    if mean_luma < min_luma or mean_luma > max_luma:
        log.debug("Thumbnail rejected (luma=%.3f)", mean_luma)
        return False, detail
    if rms_contrast < min_contrast:
        log.debug("Thumbnail rejected (contrast=%.3f)", rms_contrast)
        return False, detail
    if ed > max_ed:
        log.debug("Thumbnail rejected (edge_density=%.3f)", ed)
        return False, detail
    if ps > max_ps:
        log.debug("Thumbnail rejected (person_score=%.3f)", ps)
        return False, detail

    return True, detail


# ---------------------------------------------------------------------------
# pHash deduplication
# ---------------------------------------------------------------------------

def _compute_phash(path: Path):
    """Return imagehash pHash object or None if unavailable / unreadable."""
    if not _IMAGEHASH_AVAILABLE:
        return None
    try:
        from PIL import Image  # noqa: PLC0415

        return imagehash.phash(Image.open(path))
    except Exception as exc:  # noqa: BLE001
        log.debug("pHash failed for %s: %s", path.name, exc)
        return None


def _dedup_phash(results: list[dict], threshold: int) -> list[dict]:
    """
    Remove near-duplicate images (Hamming distance < threshold).
    Within each duplicate cluster keep the highest-scored item.
    Preserves ordering of survivors (score-descending, since results is
    already sorted before this is called).
    """
    if not _IMAGEHASH_AVAILABLE:
        return results

    hashes: list[tuple[int, Any]] = []   # (result_idx, phash)
    for i, r in enumerate(results):
        ph = _compute_phash(Path(r["path"]))
        hashes.append((i, ph))

    kept = [True] * len(results)
    for i in range(len(hashes)):
        if not kept[i] or hashes[i][1] is None:
            continue
        for j in range(i + 1, len(hashes)):
            if not kept[j] or hashes[j][1] is None:
                continue
            dist = hashes[i][1] - hashes[j][1]
            if dist < threshold:
                # results is score-descending, so i is higher-scored → drop j
                kept[j] = False
                if "score_detail" in results[j]:
                    results[j]["score_detail"]["phash_flagged"] = True

    return [r for i, r in enumerate(results) if kept[i]]


def _diversity_top_n(results: list[dict], top_n: int, div_threshold: int) -> list[dict]:
    """
    From score-sorted results, pick up to top_n items while skipping near-
    duplicates (pHash Hamming distance < div_threshold) of already-selected items.
    """
    if not _IMAGEHASH_AVAILABLE or top_n <= 0:
        return results[:top_n] if top_n > 0 else results

    selected: list[dict] = []
    selected_hashes: list[Any] = []

    for r in results:
        if len(selected) >= top_n:
            break
        ph = _compute_phash(Path(r["path"]))
        if ph is not None and selected_hashes:
            dists = [ph - sh for sh in selected_hashes if sh is not None]
            if dists and min(dists) < div_threshold:
                continue   # too similar to an already-selected item
        selected.append(r)
        selected_hashes.append(ph)

    return selected


# ---------------------------------------------------------------------------
# Meta-sidecar writer
# ---------------------------------------------------------------------------

def _write_meta_sidecar(
    path:          Path,
    img_f_list:    list[float] | None,  # CLIP embedding as plain float list
    mean_luma:     float,
    hue_hist:      list[float],
) -> None:
    """Write {path}.meta.json alongside the media file."""
    meta = {
        "hue_hist_16bin": hue_hist,
        "mean_luma":      mean_luma,
        "clip_embedding": img_f_list or [],
    }
    sidecar = Path(str(path) + ".meta.json")
    try:
        sidecar.write_text(json.dumps(meta))
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not write meta sidecar %s: %s", sidecar, exc)


# ---------------------------------------------------------------------------
# Frame extraction (video)
# ---------------------------------------------------------------------------

def _extract_frames(video_path: Path, frames_dir: Path) -> list[Path]:
    """Extract frames at VIDEO_SAMPLE_FPS using ffmpeg. Returns sorted list of jpgs."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_pat = str(frames_dir / "frame_%05d.jpg")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"fps={VIDEO_SAMPLE_FPS}",
        "-frames:v", str(VIDEO_MAX_FRAMES),
        out_pat,
    ]
    log.debug("Extracting frames: %s", video_path.name)
    subprocess.run(cmd, check=True)
    return sorted(frames_dir.glob("frame_*.jpg"))


def _probe_duration(video_path: Path) -> float | None:
    """Return video duration in seconds via ffprobe, or None on failure."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", str(video_path)],
            capture_output=True, text=True, timeout=10,
        )
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Build scoring_hints with lighting injection
# ---------------------------------------------------------------------------

def _resolve_hints(item: dict) -> dict[str, list[str]] | None:
    """
    Return the per-dimension hint dict (dim → list[str]) after injecting the
    lighting style sub-prompt.  Returns None if no scoring_hints are present.
    """
    raw = item.get("scoring_hints")
    if not raw:
        return None

    hints: dict[str, list[str]] = {}
    for dim in DIMS:
        val = raw.get(dim, [])
        if isinstance(val, str):
            val = [val]
        hints[dim] = list(val)

    # Lighting → extra style hint
    lighting = item.get("lighting", "")
    if lighting in LIGHTING_STYLE_HINTS:
        hints.setdefault("style", []).append(LIGHTING_STYLE_HINTS[lighting])

    return hints if any(hints.values()) else None


# ---------------------------------------------------------------------------
# Public scoring functions
# ---------------------------------------------------------------------------

def score_images(
    clip_model: ClipModel,
    item:       dict | str,
    img_paths:  list[Path],
    weights:    dict | None = None,  # legacy param (reserved)
    config:     dict | None = None,
    infos:      dict | None = None,
) -> list[dict]:
    """
    Score images by multi-dimensional CLIP similarity (or fallback to single
    text prompt).

    Parameters
    ----------
    clip_model : ClipModel
    item       : dict with keys used by scorer (ai_prompt, scoring_hints,
                 cinematic_role, motion_level, lighting, _thumbnails …).
                 Pass a plain str for backward compatibility.
    img_paths  : list of local image Paths to score
    weights    : legacy weight dict (ignored when multi-dim mode active)
    config     : server config dict (for scoring_profiles, thumbnail_filters …)

    Returns
    -------
    list[dict] sorted by score descending, each:
        {"path": str, "score": float, "score_detail": {...}}
    """
    # Backward compat: accept plain string for item
    if isinstance(item, str):
        item = {"ai_prompt": item}

    if not img_paths:
        return []

    cfg          = config or {}
    tf_cfg       = cfg.get("thumbnail_filters", {})
    thumbnails   = item.get("_thumbnails") or {}          # path → thumbnail_url
    ai_prompt    = item.get("ai_prompt") or item.get("search_prompt", "")
    hints        = _resolve_hints(item)
    dim_weights  = _resolve_weights(item, cfg)
    img_calm_thresh = _image_calmness_threshold(item, cfg)
    exclude_kws     = [kw.lower() for kw in (item.get("exclude_keywords") or [])]
    min_clip        = cfg.get("min_clip_candidates", 5)
    pflag_thresh = tf_cfg.get("person_flag_threshold", 0.015)
    phash_thresh = cfg.get("phash_dedup_threshold",    8)
    div_thresh   = cfg.get("diversity_phash_threshold", 12)

    log.debug(
        "Scoring %d images | mode=%s | profile=%s | cinematic_role=%s | motion_level=%s",
        len(img_paths),
        "multi_dim" if hints else "fallback",
        cfg.get("content_profile", "default"),
        item.get("cinematic_role", "—"),
        item.get("motion_level", "—"),
    )

    results: list[dict] = []

    # B2: Zero-survivor safeguard — if pool is small, bypass thumbnail pre-filter
    skip_thumbnail = len(img_paths) < (min_clip * 2)
    if skip_thumbnail:
        log.debug(
            "B2 safeguard: only %d image candidates, bypassing thumbnail pre-filter",
            len(img_paths),
        )

    # M3: telemetry counters
    _tel_total     = len(img_paths)
    _tel_thumb_rej = 0
    _tel_clip      = 0

    # Pre-compute text embeddings once per shot (not once per image)
    precomputed_text_emb = _clip_precompute_text_embeddings(clip_model, hints) if hints else {}

    for p in img_paths:
        # ------------------------------------------------------------------
        # Pass-1.5: cheap thumbnail filter
        # ------------------------------------------------------------------
        thumb_detail: dict = {}
        thumb_url = thumbnails.get(str(p)) or thumbnails.get(p.name)
        prefilter_rejected = False

        # Thumbnail pre-filter (B2: bypass if pool is too small)
        if thumb_url:
            if not skip_thumbnail:
                passes, thumb_detail = _thumbnail_filter(thumb_url, tf_cfg)
                if not passes:
                    log.debug("Thumbnail filter rejected %s (kept with score 0)", p.name)
                    prefilter_rejected = True
                    _tel_thumb_rej += 1
            else:
                # Still fetch thumb_detail for luma/contrast metadata even when bypassing
                _, thumb_detail = _thumbnail_filter(thumb_url, tf_cfg)

        # ------------------------------------------------------------------
        # Calmness (single-image Laplacian proxy) — computed always for logging
        # B1: no longer a hard gate; used as a soft penalty after CLIP scoring
        # ------------------------------------------------------------------
        calm = _calmness_single(p)

        # M7: Technical size filter using sidecar metadata (objective — hard gate is OK)
        if not prefilter_rejected:
            min_w = (item.get("search_filters") or {}).get("min_width",  0)
            min_h = (item.get("search_filters") or {}).get("min_height", 0)
            if min_w or min_h:
                meta_entry = (infos or {}).get(str(p)) or {}
                w = meta_entry.get("width",  0)
                h = meta_entry.get("height", 0)
                if (min_w and w and w < min_w) or (min_h and h and h < min_h):
                    log.debug(
                        "Size filter: %s is %dx%d < required %dx%d — skipping",
                        p.name, w, h, min_w, min_h,
                    )
                    prefilter_rejected = True
                    _tel_thumb_rej += 1   # count in same bucket for telemetry

        # If thumbnail or size pre-filter rejected, record score 0 and skip CLIP
        # Calmness is NOT a hard gate (B1) — it becomes a soft penalty below
        if prefilter_rejected:
            mean_luma    = thumb_detail.get("mean_luma",    0.0)
            rms_contrast = thumb_detail.get("rms_contrast", 0.0)
            person_score = thumb_detail.get("person_score") if thumb_detail else None
            results.append({
                "path":         str(p),
                "score":        0.0,
                "score_detail": {
                    "clip_total":   0.0,
                    "clip_dims":    {d: 0.0 for d in DIMS},
                    "calmness":     calm,
                    "mode":         "prefilter_rejected",
                    "mean_luma":    mean_luma,
                    "rms_contrast": rms_contrast,
                    "person_score": person_score,
                    "phash_flagged": False,
                    "error":        "thumbnail_rejected",
                },
            })
            continue

        _tel_clip += 1

        # ------------------------------------------------------------------
        # CLIP scoring
        # ------------------------------------------------------------------
        try:
            img_f = _clip_encode_image(clip_model, p)
        except Exception as exc:  # noqa: BLE001
            log.warning("CLIP image encode failed for %s: %s", p.name, exc)
            continue

        if hints:
            # Multi-dimensional scoring (text embeddings pre-computed once per shot)
            clip_total, clip_dims = _clip_score_multidim(
                clip_model, img_f, hints, dim_weights, precomputed_text_emb)
            mode = "multi_dim"
        else:
            # Fallback: single text prompt
            import torch  # noqa: PLC0415

            if not ai_prompt:
                log.warning("No ai_prompt and no scoring_hints for %s — score=0", p.name)
                clip_total, clip_dims = 0.0, {}
            else:
                with torch.no_grad():
                    tok = clip_model.tokenizer([ai_prompt]).to(clip_model.device)
                    tf  = clip_model.model.encode_text(tok)
                    tf  = tf / tf.norm(dim=-1, keepdim=True)
                    clip_total = float((img_f @ tf.T).item())
                    clip_dims = {}
            mode = "fallback"

        # ------------------------------------------------------------------
        # C1: exclude_keywords soft penalty (metadata signal, not hard gate)
        # ------------------------------------------------------------------
        if exclude_kws and infos:
            meta_entry = infos.get(str(p)) or {}
            meta_text  = " ".join([
                meta_entry.get("title", ""),
                meta_entry.get("description", ""),
                " ".join(meta_entry.get("tags", [])),
            ]).lower()
            n_matches = sum(1 for kw in exclude_kws if kw in meta_text)
            if n_matches:
                excl_penalty = n_matches * cfg.get("exclude_keyword_penalty", 0.05)
                clip_total   = max(0.0, clip_total - excl_penalty)
                log.debug(
                    "exclude_keywords penalty %.3f for %s (%d matches)",
                    excl_penalty, p.name, n_matches,
                )

        # ------------------------------------------------------------------
        # B1: Image calmness soft penalty (never a hard gate for images)
        # ------------------------------------------------------------------
        calmness_penalty = 0.0
        if img_calm_thresh is not None and calm < img_calm_thresh:
            motion_w = dim_weights.get("motion", BASE_WEIGHTS["motion"])
            calmness_penalty = (img_calm_thresh - calm) * motion_w
            log.debug(
                "Image calmness penalty %.3f for %s (calm=%.3f < thresh=%.3f)",
                calmness_penalty, p.name, calm, img_calm_thresh,
            )

        # ------------------------------------------------------------------
        # Build score_detail
        # ------------------------------------------------------------------
        mean_luma    = thumb_detail.get("mean_luma",    0.0)
        rms_contrast = thumb_detail.get("rms_contrast", 0.0)
        person_score = thumb_detail.get("person_score") if thumb_detail else None
        person_flagged = (
            person_score is not None
            and person_score > pflag_thresh
        )

        # If no thumbnail detail, compute basic luma from the image file.
        # Use PIL (not cv2.imread) — cultural-heritage images (TIFF, old JPEG)
        # can SIGSEGV inside OpenCV's native decoders.
        if not thumb_detail:
            try:
                import numpy as np      # noqa: PLC0415
                from PIL import Image   # noqa: PLC0415

                with Image.open(p) as _pil:
                    _gray = np.array(_pil.convert("L"), dtype=np.float32)
                mean_luma    = float(_gray.mean() / 255.0)
                rms_contrast = float(_gray.std() / 255.0)
            except Exception:
                pass

        score_detail: dict = {
            "clip_total":   clip_total,
            "clip_dims":    {d: clip_dims.get(d, 0.0) for d in DIMS},
            "calmness":     calm,
            "calmness_penalty": round(calmness_penalty, 4),
            "mode":         mode,
            "mean_luma":    mean_luma,
            "rms_contrast": rms_contrast,
            "person_score": person_score,
            "phash_flagged": False,  # will be updated by dedup pass (person_flagged is in person_score)
        }

        # Final combined score (B1: calmness is now a soft penalty, not part of primary score)
        if mode == "multi_dim":
            final_score = clip_total
        else:
            # legacy: use weights from config["score_weights"] if present
            w        = weights or cfg.get("score_weights") or {"clip": 0.75, "calmness": 0.25}
            clip_w   = float(w.get("clip",     0.75))
            calm_w   = float(w.get("calmness", 0.25))
            final_score = clip_w * clip_total + calm_w * calm
        # Apply image calmness soft penalty (B1)
        final_score = max(0.0, final_score - calmness_penalty)

        # ------------------------------------------------------------------
        # CLIP embedding + meta sidecar
        # ------------------------------------------------------------------
        try:
            import numpy as np  # noqa: PLC0415

            emb_list  = img_f.squeeze(0).cpu().numpy().tolist()
            hue_hist  = []
            if _CV2_AVAILABLE:
                img_bgr = cv2.imread(str(p))
                if img_bgr is not None:
                    hue_hist = _hue_hist_16(img_bgr)
            _write_meta_sidecar(p, emb_list, mean_luma, hue_hist)
        except Exception as exc:  # noqa: BLE001
            log.debug("Meta sidecar write skipped for %s: %s", p.name, exc)

        results.append({
            "path":         str(p),
            "score":        float(final_score),
            "score_detail": score_detail,
        })

    # Sort by score descending
    results.sort(key=lambda r: r["score"], reverse=True)

    # M3: Filter telemetry — visible in media server logs for every item
    _tel_final = len([r for r in results if r.get("score", 0.0) > 0.0])
    log.info(
        "Scoring telemetry | item=%s | downloaded=%d | prefilter_rejected=%d"
        " | clip_scored=%d | final_nonzero=%d",
        item.get("asset_id", item.get("id", "?")),
        _tel_total,
        _tel_thumb_rej,
        _tel_clip,
        _tel_final,
    )

    # pHash deduplication
    results = _dedup_phash(results, phash_thresh)

    # Diversity-aware top-N (use the full list if no top_n configured)
    top_n = cfg.get("top_n", 0)
    if top_n > 0:
        results = _diversity_top_n(results, top_n, div_thresh)

    # Attach source metadata to each result
    for r in results:
        source = None
        if infos:
            source = infos.get(r["path"])
        if source is None:
            sidecar = Path(str(r["path"]) + ".info.json")
            if sidecar.exists():
                try:
                    source = json.loads(sidecar.read_text())
                except Exception:
                    pass
        if source is not None:
            r["source"] = source

    return results


def score_single_video(
    clip_model: ClipModel,
    video_path: Path,
    frames_dir: Path,
    item:       dict,
    config:     dict | None = None,
) -> dict:
    """
    Score one video: extract frames → CLIP score (multi-dim or fallback) +
    calmness → weighted final score.  Write .meta.json sidecar.

    Used by both remote workers and the local fallback path in score_videos().

    Returns a result dict:
        {"path": str, "score": float, "clip_score": float,
         "calmness": float, "score_detail": {...}}
    or on error:
        {"path": str, "score": 0.0, "clip_score": 0.0, "calmness": 0.0, "error": str}
    """
    cfg         = config or {}
    ai_prompt   = item.get("ai_prompt") or item.get("search_prompt", "")
    hints       = _resolve_hints(item)
    dim_weights = _resolve_weights(item, cfg)
    # Legacy weight fallback
    w       = cfg.get("score_weights") or {"clip": 0.75, "calmness": 0.25}
    clip_w  = float(w.get("clip",     0.75))
    calm_w  = float(w.get("calmness", 0.25))

    vp = video_path
    vid_duration = _probe_duration(vp)

    try:
        frames = _extract_frames(vp, frames_dir)
        if not frames:
            return {
                "path": str(vp), "score": 0.0,
                "clip_score": 0.0, "calmness": 0.0,
                "duration_sec": vid_duration,
                "error": "no_frames",
            }

        calm = _calmness(frames)

        # M2: Video calmness is a soft score penalty, not a hard gate
        vid_calmness_penalty = 0.0
        vid_calm_thresh = _video_calmness_threshold(item, cfg)
        if vid_calm_thresh is not None and calm < vid_calm_thresh:
            log.debug(
                "Video calmness below threshold %s (calm=%.3f < %.3f) — soft penalty applies",
                vp.name, calm, vid_calm_thresh,
            )
            motion_w = dim_weights.get("motion", BASE_WEIGHTS["motion"])
            vid_calmness_penalty = (vid_calm_thresh - calm) * motion_w

        # Score the best frame
        if hints:
            # Pre-compute text embeddings once per video (not once per frame) — same
            # optimisation already applied to images in score_images().
            precomputed_text_emb = _clip_precompute_text_embeddings(clip_model, hints)
            # Multi-dim: pick frame with highest total clip score
            best_total = -1.0
            best_dims:  dict[str, float] = {}
            best_emb:   list[float] = []
            best_frame: Path | None = None
            for fp in frames:
                try:
                    img_f = _clip_encode_image(clip_model, fp)
                    total, dims = _clip_score_multidim(
                        clip_model, img_f, hints, dim_weights, precomputed_text_emb
                    )
                    if total > best_total:
                        best_total = total
                        best_dims  = dims
                        best_frame = fp
                        try:
                            import numpy as np  # noqa: PLC0415
                            best_emb = img_f.squeeze(0).cpu().numpy().tolist()
                        except Exception:  # noqa: BLE001
                            best_emb = []
                except Exception as exc:  # noqa: BLE001
                    log.debug("Frame score error %s: %s", fp.name, exc)

            clip_best = max(0.0, best_total)
            score     = clip_best  # in multi-dim mode the score is the combined dim score
            mode      = "multi_dim"
        else:
            # Fallback: single-prompt across all frames
            frame_scores = _clip_scores(clip_model, ai_prompt, frames)
            clip_best    = max(frame_scores.values()) if frame_scores else 0.0
            best_dims    = {}
            score        = clip_w * clip_best + calm_w * calm
            mode         = "fallback"
            best_frame   = None
            best_emb     = []

        # Meta sidecar for best frame (if known)
        if best_frame is not None and _CV2_AVAILABLE:
            try:
                img_bgr  = cv2.imread(str(best_frame))
                gray_np  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr is not None else None
                luma     = float(gray_np.mean() / 255.0) if gray_np is not None else 0.0
                hue_hist = _hue_hist_16(img_bgr) if img_bgr is not None else [0.0] * 16
                _write_meta_sidecar(vp, best_emb, luma, hue_hist)
            except Exception as exc:  # noqa: BLE001
                log.debug("Meta sidecar write skipped for %s: %s", vp.name, exc)

        # M2: Apply video calmness soft penalty
        score = max(0.0, score - vid_calmness_penalty)

        return {
            "path":         str(vp),
            "score":        float(score),
            "clip_score":   float(clip_best),
            "calmness":     float(calm),
            "duration_sec": vid_duration,
            "score_detail": {
                "clip_total":   float(clip_best),
                "clip_dims":    {d: best_dims.get(d, 0.0) for d in DIMS},
                "calmness":     float(calm),
                "mode":         mode,
                "mean_luma":    0.0,
                "rms_contrast": 0.0,
                "person_score": None,
                "phash_flagged": False,
            },
        }

    except subprocess.CalledProcessError:
        log.warning("ffmpeg failed for %s", vp.name)
        return {
            "path": str(vp), "score": 0.0,
            "clip_score": 0.0, "calmness": 0.0,
            "duration_sec": vid_duration,
            "error": "ffmpeg_failed",
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("Score error for %s: %s", vp.name, exc)
        return {
            "path": str(vp), "score": 0.0,
            "clip_score": 0.0, "calmness": 0.0,
            "duration_sec": vid_duration,
            "error": str(exc),
        }


def score_videos(
    clip_model: ClipModel,
    item:       dict | str,
    vid_paths:  list[Path],
    batch_dir:  Path,
    weights:    dict | None = None,
    config:     dict | None = None,
    infos:      dict | None = None,
) -> list[dict]:
    """
    Score videos: sample frames → multi-dim CLIP (or fallback) + calmness
    → weighted final score.

    Frames are extracted to batch_dir/_frames/<video_stem>/ and cleaned up
    by the caller (server.py) after the full batch is scored.

    Parameters
    ----------
    clip_model : ClipModel
    item       : dict (same schema as score_images).  Plain str accepted for
                 backward compatibility.
    vid_paths  : list of local video Paths
    batch_dir  : working directory for frame extraction
    weights    : legacy weight dict; used in fallback mode
    config     : server config dict

    Returns
    -------
    list[dict] sorted by score descending:
        [{"path": str, "score": float, "clip_score": float,
          "calmness": float, "score_detail": {...}}, ...]
    On per-video error:
        [{"path": str, "score": 0.0, "error": str}, ...]
    """
    # Backward compat: accept plain string for item
    if isinstance(item, str):
        item = {"ai_prompt": item}

    if not vid_paths:
        return []

    cfg          = config or {}
    phash_thresh = cfg.get("phash_dedup_threshold",    8)
    div_thresh   = cfg.get("diversity_phash_threshold", 12)

    log.debug(
        "Scoring %d videos | cinematic_role=%s",
        len(vid_paths),
        item.get("cinematic_role", "—"),
    )

    results: list[dict] = []

    for vp in vid_paths:
        frames_dir = batch_dir / "_frames" / vp.stem
        result = score_single_video(
            clip_model, vp, frames_dir, item, config,
        )
        results.append(result)

    results.sort(key=lambda r: r.get("score", 0.0), reverse=True)

    # pHash dedup on videos is less common but supported
    results = _dedup_phash(results, phash_thresh)

    top_n = cfg.get("top_n", 0)
    if top_n > 0:
        results = _diversity_top_n(results, top_n, div_thresh)

    # Attach source metadata to each result
    for r in results:
        source = None
        if infos:
            source = infos.get(r["path"])
        if source is None:
            sidecar = Path(str(r["path"]) + ".info.json")
            if sidecar.exists():
                try:
                    source = json.loads(sidecar.read_text())
                except Exception:
                    pass
        if source is not None:
            r["source"] = source

    return results
