#!/usr/bin/env python3
"""
test_project_media.py

Minimal smoke test for the media search + CLIP scoring pipeline.

Reads AssetManifest_draft.shared.json from the-rivers-whisper/s01e01,
searches Pexels + Pixabay for each background item using `search_prompt`,
scores candidates with CLIP using the richer `ai_prompt`,
and prints top 5 images + top 5 videos per item with clickable source URLs
for manual verification.

NOTE: This script uses two prompts per background item — matching the actual
manifest schema which /tmp/t1 should be updated to reflect:
  search_prompt — short keywords, sent to Pexels/Pixabay search APIs
  ai_prompt     — rich scene description, used as CLIP text embedding

Usage:
  cd code/media/
  export PEXELS_API_KEY="..."
  export PIXABAY_API_KEY="..."
  python test_project_media.py

Output dir:  code/media/test_project_out/   (wiped on each run)
"""

import json
import shutil
import sys
from pathlib import Path

# ── Import everything from the existing test.py (no code duplication) ────────
sys.path.insert(0, str(Path(__file__).parent))
from test import (
    pexels_search_images, pexels_search_videos,
    pixabay_search_images, pixabay_search_videos,
    pexels_pick_image_url, pexels_pick_video_url,
    pixabay_pick_image_url, pixabay_pick_video_url,
    download_file, load_clip_cpu,
    validate_images, validate_videos,
    log, require_env, require_ffmpeg, ensure_dir,
    IMAGE_EXTS,
)

# ── Config ────────────────────────────────────────────────────────────────────
PIPE_ROOT    = Path(__file__).parent.parent.parent          # repo root
MANIFEST     = (PIPE_ROOT / "projects/the-rivers-whisper/episodes/s01e01"
                          / "AssetManifest_draft.shared.json")
OUT_DIR      = Path(__file__).parent / "test_project_out"

N_IMG        = 8    # image candidates per source  (Pexels + Pixabay → 16 total)
N_VID        = 5    # video candidates per source  (Pexels + Pixabay → 10 total)
TOP_N        = 5    # how many to print per item

# ── Helpers ───────────────────────────────────────────────────────────────────
def pexels_photo_page(photo_id: str) -> str:
    return f"https://www.pexels.com/photo/{photo_id}/"

def pexels_video_page(video_id: str) -> str:
    return f"https://www.pexels.com/video/{video_id}/"

def pixabay_photo_page(photo_id: str) -> str:
    return f"https://pixabay.com/photos/{photo_id}/"

def pixabay_video_page(video_id: str) -> str:
    return f"https://pixabay.com/videos/{video_id}/"

def section(title: str, width: int = 76) -> None:
    print(f"\n{'─'*width}")
    print(f"  {title}")
    print(f"{'─'*width}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    require_ffmpeg()
    pexels_key  = require_env("PEXELS_API_KEY")
    pixabay_key = require_env("PIXABAY_API_KEY")

    manifest    = json.loads(MANIFEST.read_text(encoding="utf-8"))
    backgrounds = manifest["backgrounds"]

    print(f"\n{'='*76}")
    print(f"  MEDIA SEARCH SMOKE TEST")
    print(f"  Project  : {manifest['project_id']}")
    print(f"  Episode  : {manifest['episode_id']}")
    print(f"  Manifest : AssetManifest_draft.shared.json")
    print(f"  Items    : {len(backgrounds)} backgrounds")
    print(f"  Searching: Pexels + Pixabay  |  {N_IMG} img + {N_VID} vid per source")
    print(f"  Scoring  : CLIP ViT-B-32 / laion2b_s34b_b79k (CPU)")
    print(f"{'='*76}")

    # Wipe and recreate output dir
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir()

    # Load CLIP model once for all items
    clipm = load_clip_cpu()

    for bg in backgrounds:
        asset_id      = bg["asset_id"]
        search_prompt = bg["search_prompt"]
        ai_prompt     = bg.get("ai_prompt", search_prompt)

        section(f"BACKGROUND: {asset_id}")
        print(f"  search_prompt : {search_prompt}")
        print(f"  ai_prompt     : {ai_prompt[:110]}...")

        item_dir = OUT_DIR / asset_id
        img_pexels  = item_dir / "images" / "pexels"
        img_pixabay = item_dir / "images" / "pixabay"
        vid_pexels  = item_dir / "videos" / "pexels"
        vid_pixabay = item_dir / "videos" / "pixabay"
        for d in (img_pexels, img_pixabay, vid_pexels, vid_pixabay):
            ensure_dir(d)

        # path → human-readable page URL for manual verification
        img_source: dict[str, str] = {}
        vid_source: dict[str, str] = {}

        # ── Download images ────────────────────────────────────────────────
        log(f"[{asset_id}] Searching images …")

        # Pexels images
        photos = pexels_search_images(pexels_key, search_prompt, per_page=max(N_IMG, 10))
        for i, ph in enumerate(photos[:N_IMG]):
            url = pexels_pick_image_url(ph)
            if not url:
                continue
            pid  = str(ph.get("id", i))
            outp = img_pexels / f"pexels_img_{pid}.jpg"
            try:
                download_file(url, outp)
                img_source[str(outp)] = pexels_photo_page(pid)
            except Exception as e:
                log(f"  pexels img {pid} skip: {e}")

        # Pixabay images
        hits = pixabay_search_images(pixabay_key, search_prompt, per_page=max(N_IMG, 20))
        for i, hit in enumerate(hits[:N_IMG]):
            url = pixabay_pick_image_url(hit)
            if not url:
                continue
            hid  = str(hit.get("id", i))
            ext  = Path(url.split("?")[0]).suffix or ".jpg"
            outp = img_pixabay / f"pixabay_img_{hid}{ext}"
            try:
                download_file(url, outp)
                img_source[str(outp)] = pixabay_photo_page(hid)
            except Exception as e:
                log(f"  pixabay img {hid} skip: {e}")

        # ── Download videos ────────────────────────────────────────────────
        log(f"[{asset_id}] Searching videos …")

        # Pexels videos
        vids = pexels_search_videos(pexels_key, search_prompt, per_page=max(N_VID, 10))
        for i, vd in enumerate(vids[:N_VID]):
            url = pexels_pick_video_url(vd)
            if not url:
                continue
            vid  = str(vd.get("id", i))
            outp = vid_pexels / f"pexels_vid_{vid}.mp4"
            try:
                download_file(url, outp)
                vid_source[str(outp)] = pexels_video_page(vid)
            except Exception as e:
                log(f"  pexels vid {vid} skip: {e}")

        # Pixabay videos
        hits = pixabay_search_videos(pixabay_key, search_prompt, per_page=max(N_VID, 20))
        for i, hit in enumerate(hits[:N_VID]):
            url = pixabay_pick_video_url(hit)
            if not url:
                continue
            hid  = str(hit.get("id", i))
            outp = vid_pixabay / f"pixabay_vid_{hid}.mp4"
            try:
                download_file(url, outp)
                vid_source[str(outp)] = pixabay_video_page(hid)
            except Exception as e:
                log(f"  pixabay vid {hid} skip: {e}")

        # ── CLIP scoring (ai_prompt gives richer semantic embedding) ───────
        log(f"[{asset_id}] Scoring with CLIP (ai_prompt) …")

        all_imgs = [p for p in item_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
        all_vids = list(item_dir.rglob("*.mp4"))

        img_ranked = validate_images(clipm, ai_prompt, all_imgs) if all_imgs else []
        vid_ranked = validate_videos(clipm, ai_prompt, all_vids, work_dir=item_dir) if all_vids else []

        # ── Print results ──────────────────────────────────────────────────
        print(f"\n  ── TOP {TOP_N} IMAGES (CLIP score vs ai_prompt) ──")
        print(f"  {'Rank':<5} {'Score':>7}  {'Source URL'}")
        print(f"  {'─'*4}  {'─'*7}  {'─'*55}")
        for rank, r in enumerate(img_ranked[:TOP_N], 1):
            src = img_source.get(r["path"], Path(r["path"]).name)
            print(f"  #{rank:<4} {r['clip_score']:>7.4f}  {src}")

        print(f"\n  ── TOP {TOP_N} VIDEOS (0.75×CLIP + 0.25×calmness) ──")
        print(f"  {'Rank':<5} {'Score':>7}  {'CLIP':>7}  {'Calm':>6}  {'Source URL'}")
        print(f"  {'─'*4}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*55}")
        for rank, r in enumerate(vid_ranked[:TOP_N], 1):
            if r.get("error"):
                print(f"  #{rank:<4}  ERROR: {r['error']}")
                continue
            src = vid_source.get(r["path"], Path(r["path"]).name)
            print(f"  #{rank:<4} {r['final_score']:>7.4f}  {r['clip_score']:>7.4f}  {r['calmness']:>6.3f}  {src}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*76}")
    print(f"  DONE.  Downloaded files: {OUT_DIR}")
    print(f"  Manual check: click the Source URLs above to verify visual match.")
    print(f"{'='*76}\n")


if __name__ == "__main__":
    main()
