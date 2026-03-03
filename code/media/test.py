#!/usr/bin/env python3
"""
fetch_and_validate_media.py

Search + download media from Pexels + Pixabay using a prompt in prompt.txt, then run
CPU-only validation/ranking against the same prompt.

What it does:
1) Reads keys from env:
   - PEXELS_API_KEY
   - PIXABAY_API_KEY
2) Reads prompt text from ./prompt.txt
3) For EACH site:
   - downloads 5 videos + 5 images into ./downloads/<site>/(videos|images)/
4) Runs CPU validation:
   - Images: CLIP similarity (prompt vs image)
   - Videos: sample frames (via ffmpeg), CLIP similarity (prompt vs frames),
             plus a simple "calmness" motion metric from frame-to-frame differences
5) Prints every step to STDOUT for traceability and writes ./validation_report.json

Requirements:
- Python 3.10+
- ffmpeg in PATH
- pip install requests pillow numpy torch torchvision open_clip_torch opencv-python

Notes:
- This is a practical baseline validator. You can tune thresholds/weights in config.
- Pexels video downloads are fetched from returned file URLs.
- Pixabay videos come from returned "videos" variants.

Usage:
  export PEXELS_API_KEY="..."
  export PIXABAY_API_KEY="..."
  python fetch_and_validate_media.py [prompt_file]
  python fetch_and_validate_media.py my_prompt.txt

Optional:
  python fetch_and_validate_media.py --prompt prompt.txt --n 5 --out downloads
  python fetch_and_validate_media.py my_prompt.txt --n 5 --out downloads
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# ----------------------------
# Config
# ----------------------------

DEFAULT_N = 5
USER_AGENT = "media-fetcher/1.0 (+offline-validation)"
TIMEOUT = 45

# Validation weights (tune these)
VIDEO_WEIGHT_CLIP = 0.75
VIDEO_WEIGHT_CALM = 0.25

# Frame sampling for videos
VIDEO_SAMPLE_FPS = 0.5  # frames per second to sample (0.5 = 1 frame every 2 seconds)
VIDEO_MAX_FRAMES = 24   # safety cap per video

# Motion calmness (lower motion => calmer)
# We compute mean absolute diff between consecutive grayscale frames and map to [0..1] calmness.
MOTION_SCALE = 18.0  # larger => more forgiving (higher calmness for same motion)

# ----------------------------
# Utilities
# ----------------------------

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def die(msg: str, code: int = 2) -> None:
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)

def read_text(path: Path) -> str:
    if not path.exists():
        die(f"Missing file: {path}")
    return path.read_text(encoding="utf-8").strip()

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def safe_filename(s: str, max_len: int = 140) -> str:
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "", s)
    return s[:max_len] if s else "item"

def http_get(url: str, headers: Optional[dict] = None, stream: bool = False) -> requests.Response:
    hdrs = {"User-Agent": USER_AGENT}
    if headers:
        hdrs.update(headers)
    r = requests.get(url, headers=hdrs, timeout=TIMEOUT, stream=stream)
    return r

def download_file(url: str, out_path: Path, headers: Optional[dict] = None) -> None:
    ensure_dir(out_path.parent)
    log(f"DOWNLOAD -> {out_path}  url={url}")
    r = http_get(url, headers=headers, stream=True)
    r.raise_for_status()
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    with tmp.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 512):
            if chunk:
                f.write(chunk)
    tmp.replace(out_path)
    log(f"DOWNLOADED  bytes={out_path.stat().st_size:,}  sha256={sha256_file(out_path)[:16]}...")

def require_env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        die(f"Env var {name} is not set")
    return v

def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        die("ffmpeg not found in PATH. Install ffmpeg and retry.")
    if shutil.which("ffprobe") is None:
        die("ffprobe not found in PATH. Install ffmpeg (ffprobe) and retry.")

# ----------------------------
# Pexels API
# ----------------------------

def pexels_search_images(api_key: str, query: str, per_page: int) -> List[dict]:
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": api_key}
    query = " ".join(query.splitlines())
    params = {"query": query, "per_page": per_page}
    log(f"Pexels: searching IMAGES  query={query!r} per_page={per_page}")
    r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data.get("photos", [])

def pexels_search_videos(api_key: str, query: str, per_page: int) -> List[dict]:
    url = "https://api.pexels.com/videos/search"
    headers = {"Authorization": api_key}
    query = " ".join(query.splitlines())
    params = {"query": query, "per_page": per_page}
    log(f"Pexels: searching VIDEOS  query={query!r} per_page={per_page}")
    r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data.get("videos", [])

def pexels_pick_image_url(photo: dict) -> Optional[str]:
    # Prefer "large" then "original"
    src = photo.get("src") or {}
    return src.get("large") or src.get("original") or src.get("medium")

def pexels_pick_video_url(video: dict) -> Optional[str]:
    # Pick a reasonable mp4 file with width>=720 if possible, otherwise first mp4
    files = video.get("video_files") or []
    mp4s = [f for f in files if (f.get("file_type") or "").lower() == "video/mp4" and f.get("link")]
    if not mp4s:
        return None
    # Prefer bigger but not insane
    mp4s.sort(key=lambda f: (-(f.get("width") or 0), (f.get("file_size") or 10**18)))
    for f in mp4s:
        if (f.get("width") or 0) >= 720:
            return f["link"]
    return mp4s[0]["link"]

# ----------------------------
# Pixabay API
# ----------------------------

def pixabay_search_images(api_key: str, query: str, per_page: int) -> List[dict]:
    url = "https://pixabay.com/api/"
    query = " ".join(query.splitlines())[:100]
    params = {"key": api_key, "q": query, "image_type": "photo", "per_page": per_page, "safesearch": "true"}
    log(f"Pixabay: searching IMAGES  query={query!r} per_page={per_page}")
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data.get("hits", [])

def pixabay_search_videos(api_key: str, query: str, per_page: int) -> List[dict]:
    url = "https://pixabay.com/api/videos/"
    query = " ".join(query.splitlines())[:100]
    params = {"key": api_key, "q": query, "per_page": per_page, "safesearch": "true"}
    log(f"Pixabay: searching VIDEOS  query={query!r} per_page={per_page}")
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data.get("hits", [])

def pixabay_pick_image_url(hit: dict) -> Optional[str]:
    # Prefer largeImageURL, else webformatURL
    return hit.get("largeImageURL") or hit.get("webformatURL")

def pixabay_pick_video_url(hit: dict) -> Optional[str]:
    vids = hit.get("videos") or {}
    # Prefer "large" then "medium" then "small"
    for k in ("large", "medium", "small", "tiny"):
        v = vids.get(k) or {}
        if v.get("url"):
            return v["url"]
    return None

# ----------------------------
# Validation (CPU) with CLIP
# ----------------------------

@dataclass
class ClipModel:
    model: Any
    preprocess: Any
    tokenizer: Any
    device: str

def load_clip_cpu() -> ClipModel:
    log("Validator: loading CLIP (open_clip_torch) on CPU ...")
    try:
        import torch
        import open_clip
    except Exception as e:
        die(f"Missing deps for CLIP. Install: pip install torch torchvision open_clip_torch\n{e}")

    device = "cpu"
    model_name = "ViT-B-32"
    pretrained = "laion2b_s34b_b79k"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    log(f"Validator: CLIP loaded: {model_name} / {pretrained} on {device}")
    return ClipModel(model=model, preprocess=preprocess, tokenizer=tokenizer, device=device)

def clip_text_image_score(clipm: ClipModel, prompt: str, image_paths: List[Path]) -> Dict[str, float]:
    import torch
    from PIL import Image

    with torch.no_grad():
        text = clipm.tokenizer([prompt]).to(clipm.device)
        text_features = clipm.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        scores: Dict[str, float] = {}
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            img_t = clipm.preprocess(img).unsqueeze(0).to(clipm.device)
            img_features = clipm.model.encode_image(img_t)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            sim = (img_features @ text_features.T).item()
            scores[str(p)] = float(sim)
    return scores

def extract_video_frames(video_path: Path, frames_dir: Path) -> List[Path]:
    ensure_dir(frames_dir)
    # Extract at FPS, cap via -frames:v
    out_pattern = str(frames_dir / "frame_%05d.jpg")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"fps={VIDEO_SAMPLE_FPS}",
        "-frames:v", str(VIDEO_MAX_FRAMES),
        out_pattern
    ]
    log(f"Validator: extracting frames  video={video_path.name} fps={VIDEO_SAMPLE_FPS} max_frames={VIDEO_MAX_FRAMES}")
    subprocess.run(cmd, check=True)
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    log(f"Validator: extracted frames={len(frames)}")
    return frames

def motion_calmness(frames: List[Path]) -> float:
    # Calmness in [0..1], higher is calmer
    try:
        import cv2
        import numpy as np
    except Exception as e:
        die(f"Missing deps for motion metric. Install: pip install opencv-python numpy\n{e}")

    if len(frames) < 2:
        return 1.0

    diffs = []
    prev = None
    for fp in frames:
        img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if prev is not None:
            # mean absolute diff
            dif = float(np.mean(cv2.absdiff(img, prev)))
            diffs.append(dif)
        prev = img
    if not diffs:
        return 1.0
    mean_diff = sum(diffs) / len(diffs)
    calm = max(0.0, min(1.0, 1.0 - (mean_diff / MOTION_SCALE)))
    return calm

def validate_images(clipm: ClipModel, prompt: str, image_paths: List[Path]) -> List[dict]:
    log(f"Validate: scoring {len(image_paths)} images with CLIP ...")
    scores = clip_text_image_score(clipm, prompt, image_paths)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out = [{"path": p, "clip_score": s} for p, s in ranked]
    return out

def validate_videos(clipm: ClipModel, prompt: str, video_paths: List[Path], work_dir: Path) -> List[dict]:
    log(f"Validate: scoring {len(video_paths)} videos (frames+CLIP+motion) ...")
    results = []
    for vp in video_paths:
        frames_dir = work_dir / "frames" / vp.stem
        try:
            frames = extract_video_frames(vp, frames_dir)
            if not frames:
                results.append({"path": str(vp), "clip_score": 0.0, "calmness": 0.0, "final_score": 0.0, "error": "no_frames"})
                continue
            # CLIP on frames: take max or mean; for relevance, max works well
            frame_scores = clip_text_image_score(clipm, prompt, frames)
            clip_best = max(frame_scores.values()) if frame_scores else 0.0
            calm = motion_calmness(frames)

            final = VIDEO_WEIGHT_CLIP * clip_best + VIDEO_WEIGHT_CALM * calm
            results.append({
                "path": str(vp),
                "clip_score": float(clip_best),
                "calmness": float(calm),
                "final_score": float(final),
                "frames": len(frames),
            })
        except subprocess.CalledProcessError as e:
            results.append({"path": str(vp), "clip_score": 0.0, "calmness": 0.0, "final_score": 0.0, "error": f"ffmpeg_failed:{e}"})
        except Exception as e:
            results.append({"path": str(vp), "clip_score": 0.0, "calmness": 0.0, "final_score": 0.0, "error": str(e)})

    results.sort(key=lambda r: r.get("final_score", 0.0), reverse=True)
    return results

# ----------------------------
# Cache helpers
# ----------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv"}

def collect_existing(directory: Path, exts: set) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in exts and not p.suffix.endswith(".part"))

def has_enough(directory: Path, exts: set, n: int) -> bool:
    return len(collect_existing(directory, exts)) >= n

# ----------------------------
# Human-readable report
# ----------------------------

def print_human_report(prompt: str, all_images: List[dict], all_videos: List[dict], out_report: Path) -> None:
    W = 72
    bar  = "─" * W
    dbar = "═" * W

    def trunc(s: str, n: int) -> str:
        return s if len(s) <= n else s[: n - 1] + "…"

    print()
    print(dbar)
    print(f"  MEDIA FETCH RESULTS")
    print(dbar)
    print(f"  Prompt : {trunc(prompt, W - 11)}")
    print(f"  Report : {out_report}")
    print(dbar)

    # --- Images ---
    print()
    print(f"  IMAGES  ({len(all_images)} total, ranked by CLIP relevance)")
    print(bar)
    if not all_images:
        print("  (none)")
    for rank, item in enumerate(all_images, 1):
        score  = item.get("clip_score", 0.0)
        site   = item.get("site", "?")
        cached = "  [cached]" if item.get("cached") else ""
        fname  = Path(item["path"]).name
        bar_w  = int(score * 40)
        meter  = "█" * bar_w + "░" * (40 - bar_w)
        print(f"  #{rank:<3}  {meter}  {score:.4f}  [{site:<7}]{cached}")
        print(f"        {trunc(fname, W - 8)}")
    print(bar)

    # --- Videos ---
    print()
    print(f"  VIDEOS  ({len(all_videos)} total, ranked by relevance + calmness)")
    print(bar)
    if not all_videos:
        print("  (none)")
    for rank, item in enumerate(all_videos, 1):
        final  = item.get("final_score", 0.0)
        clip   = item.get("clip_score", 0.0)
        calm   = item.get("calmness", 0.0)
        site   = item.get("site", "?")
        cached = "  [cached]" if item.get("cached") else ""
        fname  = Path(item["path"]).name
        err    = item.get("error")
        bar_w  = int(final * 40)
        meter  = "█" * bar_w + "░" * (40 - bar_w)
        print(f"  #{rank:<3}  {meter}  {final:.4f}  [{site:<7}]{cached}")
        if err:
            print(f"        clip={clip:.4f}  calm={calm:.3f}  ERROR={err}")
        else:
            print(f"        clip={clip:.4f}  calm={calm:.3f}  frames={item.get('frames','?')}")
        print(f"        {trunc(fname, W - 8)}")
    print(bar)
    print()

# ----------------------------
# Main pipeline
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt_file", nargs="?", default=None, help="Path to prompt file (positional)")
    ap.add_argument("--prompt", default="prompt.txt", help="Path to prompt file (default: prompt.txt)")
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="Download N videos + N images per site (default 5)")
    ap.add_argument("--out", default="downloads", help="Output directory")
    ap.add_argument("--no-validate", action="store_true", help="Skip validation step")
    ap.add_argument("--force", action="store_true", help="Delete cached downloads and redownload everything")
    args = ap.parse_args()
    if args.prompt_file:
        args.prompt = args.prompt_file

    require_ffmpeg()

    prompt_path = Path(args.prompt)
    prompt = read_text(prompt_path)
    if not prompt:
        die("prompt.txt is empty")

    log("=== START ===")
    log(f"Prompt file: {prompt_path.resolve()}")
    log(f"Prompt text: {prompt!r}")

    pexels_key = require_env("PEXELS_API_KEY")
    pixabay_key = require_env("PIXABAY_API_KEY")

    out_root = Path(args.out).resolve()
    ensure_dir(out_root)

    # Wipe cache if --force
    if args.force:
        for site in ("pexels", "pixabay"):
            site_dir = out_root / site
            if site_dir.exists():
                shutil.rmtree(site_dir)
                log(f"--force: removed cache dir {site_dir}")

    report: Dict[str, Any] = {
        "prompt": prompt,
        "download_root": str(out_root),
        "sites": {},
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # ----------------------------
    # Download (or use cache)
    # ----------------------------
    for site in ("pexels", "pixabay"):
        log(f"--- SITE: {site.upper()} ---")
        site_dir = out_root / site
        img_dir = site_dir / "images"
        vid_dir = site_dir / "videos"
        ensure_dir(img_dir)
        ensure_dir(vid_dir)

        downloaded_images: List[Path] = []
        downloaded_videos: List[Path] = []

        # --- Images ---
        if has_enough(img_dir, IMAGE_EXTS, args.n):
            downloaded_images = collect_existing(img_dir, IMAGE_EXTS)[: args.n]
            log(f"{site}: using {len(downloaded_images)} cached images (skip download)")
        else:
            if site == "pexels":
                photos = pexels_search_images(pexels_key, prompt, per_page=max(args.n, 10))
                log(f"Pexels: image hits={len(photos)}")
                for i, ph in enumerate(photos[: args.n]):
                    url = pexels_pick_image_url(ph)
                    if not url:
                        continue
                    pid = str(ph.get("id", i))
                    fname = f"pexels_img_{pid}.jpg"
                    outp = img_dir / fname
                    try:
                        download_file(url, outp)
                        downloaded_images.append(outp)
                    except Exception as e:
                        log(f"Pexels: image download failed id={pid} err={e}")
            else:
                hits = pixabay_search_images(pixabay_key, prompt, per_page=max(args.n, 20))
                log(f"Pixabay: image hits={len(hits)}")
                for i, hit in enumerate(hits[: args.n]):
                    url = pixabay_pick_image_url(hit)
                    if not url:
                        continue
                    hid = str(hit.get("id", i))
                    ext = Path(url.split("?")[0]).suffix or ".jpg"
                    fname = f"pixabay_img_{hid}{ext}"
                    outp = img_dir / fname
                    try:
                        download_file(url, outp)
                        downloaded_images.append(outp)
                    except Exception as e:
                        log(f"Pixabay: image download failed id={hid} err={e}")

        # --- Videos ---
        if has_enough(vid_dir, VIDEO_EXTS, args.n):
            downloaded_videos = collect_existing(vid_dir, VIDEO_EXTS)[: args.n]
            log(f"{site}: using {len(downloaded_videos)} cached videos (skip download)")
        else:
            if site == "pexels":
                vids = pexels_search_videos(pexels_key, prompt, per_page=max(args.n, 10))
                log(f"Pexels: video hits={len(vids)}")
                for i, vd in enumerate(vids[: args.n]):
                    url = pexels_pick_video_url(vd)
                    if not url:
                        continue
                    vid = str(vd.get("id", i))
                    fname = f"pexels_vid_{vid}.mp4"
                    outp = vid_dir / fname
                    try:
                        download_file(url, outp)
                        downloaded_videos.append(outp)
                    except Exception as e:
                        log(f"Pexels: video download failed id={vid} err={e}")
            else:
                hits = pixabay_search_videos(pixabay_key, prompt, per_page=max(args.n, 20))
                log(f"Pixabay: video hits={len(hits)}")
                for i, hit in enumerate(hits[: args.n]):
                    url = pixabay_pick_video_url(hit)
                    if not url:
                        continue
                    hid = str(hit.get("id", i))
                    fname = f"pixabay_vid_{hid}.mp4"
                    outp = vid_dir / fname
                    try:
                        download_file(url, outp)
                        downloaded_videos.append(outp)
                    except Exception as e:
                        log(f"Pixabay: video download failed id={hid} err={e}")

        report["sites"][site] = {
            "images": [str(p) for p in downloaded_images],
            "videos": [str(p) for p in downloaded_videos],
            "counts": {"images": len(downloaded_images), "videos": len(downloaded_videos)},
        }
        log(f"{site}: images={len(downloaded_images)} videos={len(downloaded_videos)}")

    if args.no_validate:
        log("Validation skipped (--no-validate).")
        out_report = out_root / "validation_report.json"
        out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log(f"Wrote report: {out_report}")
        log("=== DONE ===")
        return

    # ----------------------------
    # Validate
    # ----------------------------
    clipm = load_clip_cpu()
    work_dir = out_root / "_work"
    ensure_dir(work_dir)

    validation: Dict[str, Any] = {}
    all_images: List[dict] = []
    all_videos: List[dict] = []

    for site in ("pexels", "pixabay"):
        log(f"--- VALIDATE: {site.upper()} ---")
        site_images = [Path(p) for p in report["sites"][site]["images"]]
        site_videos = [Path(p) for p in report["sites"][site]["videos"]]

        img_scores = validate_images(clipm, prompt, site_images) if site_images else []
        vid_scores = validate_videos(clipm, prompt, site_videos, work_dir=work_dir) if site_videos else []

        # Tag each result with site
        for r in img_scores:
            r["site"] = site
        for r in vid_scores:
            r["site"] = site

        validation[site] = {
            "images_ranked": img_scores,
            "videos_ranked": vid_scores,
        }
        all_images.extend(img_scores)
        all_videos.extend(vid_scores)

    # Sort globally across sites
    all_images.sort(key=lambda r: r.get("clip_score", 0.0), reverse=True)
    all_videos.sort(key=lambda r: r.get("final_score", 0.0), reverse=True)

    report["validation"] = validation
    report["ranked"] = {
        "images": all_images,
        "videos": all_videos,
    }

    out_report = out_root / "validation_report.json"
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"Wrote report: {out_report}")
    log("=== DONE ===")

    print_human_report(prompt, all_images, all_videos, out_report)


if __name__ == "__main__":
    main()
