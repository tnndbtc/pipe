#!/usr/bin/env python3
"""
media_plan_auto.py — Auto-generate MediaPlan.json from story sources + VOPlan timing.

Reads story_sources.json in the episode directory (written by export_story.py
alongside _raw.txt), downloads images to media/ subdirectory, and writes
MediaPlan.json with original news images mapped to VO-segment timing.

Only writes if MediaPlan.json does not already exist, so manual overrides in the
VC editor are never clobbered.

Usage:
    python media_plan_auto.py \\
        --episode-dir projects/slug/episodes/s01e01 \\
        [--sources-json /path/to/_sources.json] \\
        [--force]

    Called automatically by POST /api/media_plan_auto_generate in server.py.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

# ── Constants ──────────────────────────────────────────────────────────────────
# Animation cycle applied round-robin across segments.
ANIMATIONS = ["ken_burns", "zoom_in", "pan_lr", "zoom_out", "pan_rl"]
TRANSITION  = "dissolve_short"

# Extra tail silence added after the last VO item (seconds).
VIDEO_TAIL_SEC = 2.5

# Quality gate: skip images whose width attribute (in HTML) is narrower than this.
MIN_IMAGE_WIDTH = 400

# Take at most this many images per source to avoid one outlet dominating visuals.
MAX_IMAGES_PER_SOURCE = 2

# URL path fragments that signal non-content images: avatars, icons, trackers, etc.
_SKIP_RE = re.compile(
    r"(avatar|/icon[s]?/|/logo[s]?/|badge|1x1|pixel|tracker|spacer"
    r"|sprite|/thumb/\d{2,3}x|/thumbnail/\d{2,3}x)",
    re.IGNORECASE,
)

# ── Platform-specific image extraction ────────────────────────────────────────

def extract_images_from_raw_payload(raw_payload: dict, platform: str) -> list[str]:
    """Return a list of image URLs extracted from a raw_payload dict.

    Each platform stores media differently:
    - reddit: media_thumbnail.url  (clean direct field)
    - social (twitter/weibo/bilibili/tiktok/instagram): common thumbnail keys
    - news_rss and everything else: parse <img src> from content[].value HTML
    """
    if not raw_payload:
        return []

    urls: list[str] = []

    if platform == "reddit":
        thumb = raw_payload.get("media_thumbnail") or {}
        if isinstance(thumb, dict) and thumb.get("url"):
            urls.append(thumb["url"])
        for mc in raw_payload.get("media_content") or []:
            if isinstance(mc, dict) and mc.get("url"):
                urls.append(mc["url"])

    elif platform in ("twitter", "weibo", "bilibili", "tiktok", "instagram", "mastodon"):
        for key in ("media_thumbnail", "thumbnail", "image", "cover", "poster"):
            val = raw_payload.get(key)
            if isinstance(val, dict) and val.get("url"):
                urls.append(val["url"])
                break
            if isinstance(val, str) and val.startswith("http"):
                urls.append(val)
                break

    else:
        # news_rss, generic news sites: parse HTML for <img> tags
        urls = _extract_from_html_content(raw_payload)

    return [u for u in urls if _is_quality_url(u)][:MAX_IMAGES_PER_SOURCE]


def _extract_from_html_content(raw_payload: dict) -> list[str]:
    """Parse <img src> from RSS/news raw_payload content HTML."""
    from html.parser import HTMLParser

    class _ImgParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.found: list[str] = []

        def handle_starttag(self, tag: str, attrs: list) -> None:
            if tag != "img":
                return
            d = dict(attrs)
            src = (d.get("src") or d.get("data-src")
                   or d.get("data-lazy-src") or "")
            if not src.startswith("http"):
                return
            # Dimension quality filter: skip images with explicit tiny dimensions
            try:
                w = int(d.get("width") or 0)
                h = int(d.get("height") or 0)
                if w and w < MIN_IMAGE_WIDTH:
                    return
                if h and h < MIN_IMAGE_WIDTH // 2:
                    return
            except (ValueError, TypeError):
                pass
            self.found.append(src)

    parser = _ImgParser()

    # Parse HTML content blocks (RSS feed format)
    for block in raw_payload.get("content") or []:
        html = block.get("value", "") if isinstance(block, dict) else ""
        if html:
            try:
                parser.feed(html)
            except Exception:
                pass

    # Also check top-level thumbnail/image fields (prioritised)
    top_urls: list[str] = []
    for key in ("image", "thumbnail", "media_thumbnail", "cover"):
        val = raw_payload.get(key)
        if isinstance(val, dict):
            u = val.get("url") or val.get("href") or ""
            if u.startswith("http"):
                top_urls.append(u)
        elif isinstance(val, str) and val.startswith("http"):
            top_urls.append(val)

    return top_urls + parser.found


def _is_quality_url(url: str) -> bool:
    """Return True if the URL looks like a real content image."""
    return bool(url) and url.startswith("http") and not _SKIP_RE.search(url)


# ── Image downloader ───────────────────────────────────────────────────────────

def _url_to_filename(url: str, idx: int) -> str:
    """Derive a stable local filename from a URL and sequential index."""
    h   = hashlib.sha1(url.encode()).hexdigest()[:10]
    ext = os.path.splitext(urlparse(url).path)[-1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
        ext = ".jpg"
    return f"src_{idx:02d}_{h}{ext}"


def _download(url: str, dest: Path) -> bool:
    """Download url → dest. Returns True on success."""
    if dest.exists():
        return True   # already cached
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; NewsPipeBot/1.0)",
            "Accept":     "image/webp,image/apng,image/*,*/*;q=0.8",
        })
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = resp.read()
        if len(data) < 1024:          # skip trackers / 1-pixel beacons
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return True
    except Exception as exc:
        print(f"  [media_auto] download failed ({url[:80]}): {exc}")
        return False


# ── Attribution label ──────────────────────────────────────────────────────────

def _source_label(url: str, title: str) -> str:
    """Return a short human-readable source name, e.g. 'Reuters', 'BBC'."""
    # Try "Headline - OutletName" pattern first
    parts = (title or "").rsplit(" - ", 1)
    if len(parts) == 2 and parts[1].strip():
        return parts[1].strip()
    # Fall back to domain slug
    try:
        host = urlparse(url).netloc.lower()
        for prefix in ("www.", "m.", "news.", "rss.", "feeds."):
            host = host.removeprefix(prefix)
        return host.split(".")[0].capitalize()
    except Exception:
        return "Source"


# ── VOPlan reader ──────────────────────────────────────────────────────────────

def _load_voplan(episode_dir: Path) -> dict:
    """Return the first VOPlan.*.json found in the episode directory."""
    for p in sorted(episode_dir.glob("VOPlan.*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            continue
    return {}


def _total_duration(voplan: dict) -> float:
    """Total episode duration = last VO item end_sec + tail silence."""
    items = voplan.get("vo_items") or []
    if not items:
        return 60.0
    last = max((vi.get("end_sec") or 0.0) for vi in items)
    return last + VIDEO_TAIL_SEC


# ── Main generator ─────────────────────────────────────────────────────────────

def generate_media_plan(
    episode_dir: Path,
    sources_json_path: Path | None = None,
    force: bool = False,
) -> tuple[bool, str]:
    """Auto-generate MediaPlan.json for the given episode directory.

    Args:
        episode_dir:       Absolute path to the episode folder.
        sources_json_path: Optional explicit path to story_sources.json.
                           Defaults to <episode_dir>/story_sources.json.
        force:             Overwrite existing MediaPlan.json if True.

    Returns:
        (success: bool, message: str)
    """
    mp_path = episode_dir / "MediaPlan.json"
    if mp_path.exists() and not force:
        return False, "MediaPlan.json already exists — skipped (use force=True to overwrite)"

    # ── Locate sources JSON ───────────────────────────────────────────────────
    if sources_json_path is None:
        sources_json_path = episode_dir / "story_sources.json"
    if not sources_json_path.exists():
        return False, f"story_sources.json not found at {sources_json_path}"

    with open(sources_json_path, encoding="utf-8") as f:
        sources_data = json.load(f)
    sources: list[dict] = sources_data.get("sources") or []
    if not sources:
        return False, "sources list is empty — nothing to download"

    # ── Download images ───────────────────────────────────────────────────────
    media_dir = episode_dir / "media"
    media_dir.mkdir(exist_ok=True)

    downloaded: list[dict] = []   # {local_path, url, source_url, label, attribution}
    img_idx = 0

    for src in sources:
        src_url   = src.get("url", "")
        src_title = src.get("title", "")
        platform  = src.get("platform", "news_rss")

        # image_urls is pre-extracted by export_story.py; raw_payload is fallback
        image_urls: list[str] = (
            src.get("image_urls")
            or extract_images_from_raw_payload(src.get("raw_payload") or {}, platform)
        )
        label = _source_label(src_url, src_title)

        for img_url in image_urls[:MAX_IMAGES_PER_SOURCE]:
            fname = _url_to_filename(img_url, img_idx)
            dest  = media_dir / fname
            if _download(img_url, dest):
                downloaded.append({
                    "local_path":  str(dest),
                    "url":         img_url,
                    "source_url":  src_url,
                    "label":       label,
                    "attribution": f"Source: {label}",
                })
                print(f"  [media_auto] ✓ {label:<20} → {fname}")
                img_idx += 1
                break   # one image per source is enough for quality variety

    if not downloaded:
        return False, "No images downloaded — check source URLs or network"

    # ── Compute timing ────────────────────────────────────────────────────────
    voplan   = _load_voplan(episode_dir)
    total    = _total_duration(voplan)
    n        = len(downloaded)
    hold_sec = round(total / n, 3)

    print(f"  [media_auto] {n} images, total={total:.1f}s → {hold_sec:.1f}s/image")

    # ── Build shot_overrides ──────────────────────────────────────────────────
    shot_overrides: list[dict] = []
    for i, img in enumerate(downloaded):
        shot_overrides.append({
            "type":           "image",
            "url":            img["url"],
            "path":           img["local_path"],
            "hold_sec":       hold_sec,
            "animation_type": ANIMATIONS[i % len(ANIMATIONS)],
            "transition":     "none" if i == 0 else TRANSITION,
            "attribution":    img["attribution"],
        })

    media_plan = {
        "schema_id":      "MediaPlan",
        "schema_version": "1.0",
        "auto_generated": True,
        "shot_overrides": shot_overrides,
    }

    # ── Atomic write ──────────────────────────────────────────────────────────
    tmp = mp_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(media_plan, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(mp_path)

    msg = f"MediaPlan.json written — {n} segments, {hold_sec:.1f}s each"
    print(f"  [media_auto] ✓ {msg}")
    return True, msg


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-generate MediaPlan.json from story_sources.json + VOPlan timing."
    )
    parser.add_argument(
        "--episode-dir", required=True,
        help="Path to episode directory (e.g. projects/slug/episodes/s01e01)"
    )
    parser.add_argument(
        "--sources-json", default=None,
        help="Path to story_sources.json. Defaults to <episode-dir>/story_sources.json"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing MediaPlan.json"
    )
    args = parser.parse_args()

    ep_dir = Path(args.episode_dir)
    if not ep_dir.is_dir():
        print(f"ERROR: episode-dir not found: {ep_dir}", file=sys.stderr)
        sys.exit(1)

    sources_path = Path(args.sources_json) if args.sources_json else None
    ok, msg = generate_media_plan(ep_dir, sources_path, args.force)
    print(f"{'✓' if ok else '✗'} {msg}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
