"""
downloader.py — Pexels + Pixabay media search and download for code/media/http/

All public functions are *synchronous* and designed to be called via
asyncio.to_thread() from server.py so they never block the event loop.

Features
--------
- Per-provider jittered sleep after each API call (respects rate limits)
- 429 exponential backoff with configurable retry delays
- In-process search-response cache (key = source + media_type + query + per_page)
  eliminates duplicate API calls within a batch when items share similar prompts
- prefer-aware: caller passes n_img / n_vid; passing 0 skips that media type
- Query rotation: when item["search_queries"] is set, each query is issued
  independently and results are deduplicated before trimming to n
- Source filters: item["source_filters"] overrides default params per source

Public API
----------
fetch_images(search_prompt, n, output_dir, api_keys, cfg, item=None) → list[tuple[Path, dict]]
fetch_videos(search_prompt, n, output_dir, api_keys, cfg, item=None) → list[tuple[Path, dict]]

output_dir layout (created by this module):
    output_dir/
        pexels/   pexels_img_<id>.jpg   or   pexels_vid_<id>.mp4
        pixabay/  pixabay_img_<id>.<ext>  or  pixabay_vid_<id>.mp4
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger("downloader")

_USER_AGENT = "media-fetcher/1.0 (+offline-pipeline)"


def _slug_to_title(page_url: str) -> str:
    """Extract human-readable title from Pexels/Pixabay page URL slug.
    e.g. https://www.pexels.com/photo/ancient-roman-temple-ruins-in-pompeii-italy-30204931/
         → "Ancient Roman Temple Ruins In Pompeii Italy"
    """
    slug = page_url.rstrip("/").split("/")[-1]
    parts = slug.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        slug = parts[0]
    return slug.replace("-", " ").title()


def _write_info_sidecar(path: Path, info: dict) -> None:
    """Write metadata sidecar alongside a downloaded file. Skips if already exists."""
    sidecar = Path(str(path) + ".info.json")
    if not sidecar.exists():
        try:
            sidecar.write_text(json.dumps(info, ensure_ascii=False))
        except Exception as exc:
            log.warning("Could not write info sidecar %s: %s", sidecar, exc)
_TIMEOUT    = 45  # seconds for API calls and downloads

# In-process search-response cache: (source, media_type, query, per_page) → list[dict]
_search_cache: dict[tuple, list] = {}


# ---------------------------------------------------------------------------
# Low-level HTTP helpers
# ---------------------------------------------------------------------------

def _get(url: str, headers: Optional[dict] = None, **kwargs) -> requests.Response:
    hdrs = {"User-Agent": _USER_AGENT}
    if headers:
        hdrs.update(headers)
    return requests.get(url, headers=hdrs, timeout=_TIMEOUT, **kwargs)


def _download_file(url: str, dest: Path, headers: Optional[dict] = None) -> None:
    """Stream-download url → dest using an atomic .part rename."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = _get(url, headers=headers, stream=True)
    r.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".part")
    with tmp.open("wb") as fh:
        for chunk in r.iter_content(chunk_size=512 * 1024):
            if chunk:
                fh.write(chunk)
    tmp.replace(dest)
    log.debug("Downloaded %s → %s (%d bytes)", url[:80], dest.name, dest.stat().st_size)


def _with_backoff(fn, backoff_seconds: list[int]):
    """
    Call fn().  On HTTP 429, retry after each wait in backoff_seconds.
    Raises the last exception if all retries are exhausted.
    """
    last_exc: Exception | None = None
    for wait in [0] + list(backoff_seconds):
        if wait:
            log.warning("429 received — retrying in %ds …", wait)
            time.sleep(wait)
        try:
            return fn()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                last_exc = exc
                continue
            raise
    raise last_exc  # type: ignore[misc]


def _jitter(cfg: dict, source: str) -> None:
    """Sleep a random delay in [delay_ms_min, delay_ms_max] for the given source."""
    rl = cfg.get("rate_limits", {}).get(source, {})
    lo = float(rl.get("delay_ms_min", 0))
    hi = float(rl.get("delay_ms_max", 0))
    if hi > 0:
        time.sleep(random.uniform(lo, hi) / 1000.0)


# ---------------------------------------------------------------------------
# Query rotation helper
# ---------------------------------------------------------------------------

def _get_queries(search_prompt: str, item: dict | None, n: int) -> list[tuple[str, int]]:
    """
    Returns a list of (query, per_query_n) tuples.

    When item["search_queries"] is a non-empty list, each query gets a budget
    of ceil(n / len(queries)) so that after deduplication we can always fill n.
    Falls back to a single (search_prompt, n) tuple if no search_queries present.
    """
    if item:
        queries = item.get("search_queries") or []
        if queries:
            per_q = math.ceil(n / len(queries))
            return [(q, per_q) for q in queries]
    return [(search_prompt, n)]


# ---------------------------------------------------------------------------
# Source-filter helper
# ---------------------------------------------------------------------------

def _get_source_filters(item: dict | None, source: str) -> dict:
    """
    Return the source-specific filter overrides from item["source_filters"].
    Returns an empty dict if no overrides are defined for this source.
    """
    if item:
        sf = item.get("source_filters") or {}
        return dict(sf.get(source) or {})
    return {}


# ---------------------------------------------------------------------------
# Pexels — images
# ---------------------------------------------------------------------------

def _pexels_search_images(
    api_key: str,
    query: str,
    per_page: int,
    backoff: list[int],
    extra_params: dict | None = None,
) -> list[dict]:
    # Build a stable cache key that includes any extra params
    key = ("pexels", "image", query, per_page, tuple(sorted((extra_params or {}).items())))
    if key in _search_cache:
        return _search_cache[key]

    headers = {"Authorization": api_key}
    q = " ".join(query.splitlines())

    # Always-on Pexels image defaults
    params: dict = {
        "query":       q,
        "per_page":    per_page,
        "orientation": "landscape",
        "size":        "large",
    }
    # Merge caller-supplied overrides (source_filters["pexels"])
    if extra_params:
        params.update(extra_params)

    def call() -> list[dict]:
        r = requests.get(
            "https://api.pexels.com/v1/search",
            headers=headers,
            params=params,
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        return r.json().get("photos", [])

    log.info("Pexels search images  q=%r per_page=%d extra=%s", q, per_page, extra_params)
    result = _with_backoff(call, backoff)
    _search_cache[key] = result
    return result


def _pexels_pick_image_url(photo: dict) -> Optional[str]:
    src = photo.get("src") or {}
    return src.get("large") or src.get("original") or src.get("medium")


# ---------------------------------------------------------------------------
# Pexels — videos
# ---------------------------------------------------------------------------

def _pexels_search_videos(
    api_key: str,
    query: str,
    per_page: int,
    backoff: list[int],
    extra_params: dict | None = None,
) -> list[dict]:
    key = ("pexels", "video", query, per_page, tuple(sorted((extra_params or {}).items())))
    if key in _search_cache:
        return _search_cache[key]

    headers = {"Authorization": api_key}
    q = " ".join(query.splitlines())

    params: dict = {
        "query":    q,
        "per_page": per_page,
    }
    if extra_params:
        params.update(extra_params)

    def call() -> list[dict]:
        r = requests.get(
            "https://api.pexels.com/videos/search",
            headers=headers,
            params=params,
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        return r.json().get("videos", [])

    log.info("Pexels search videos  q=%r per_page=%d extra=%s", q, per_page, extra_params)
    result = _with_backoff(call, backoff)
    _search_cache[key] = result
    return result


def _pexels_pick_video_url(video: dict) -> Optional[str]:
    files = video.get("video_files") or []
    mp4s  = [
        f for f in files
        if (f.get("file_type") or "").lower() == "video/mp4" and f.get("link")
    ]
    if not mp4s:
        return None
    mp4s.sort(key=lambda f: -(f.get("width") or 0))
    for f in mp4s:
        if (f.get("width") or 0) >= 720:
            return f["link"]
    return mp4s[0]["link"]


# ---------------------------------------------------------------------------
# Pixabay — images
# ---------------------------------------------------------------------------

def _pixabay_search_images(
    api_key: str,
    query: str,
    per_page: int,
    backoff: list[int],
    extra_params: dict | None = None,
) -> list[dict]:
    key = ("pixabay", "image", query, per_page, tuple(sorted((extra_params or {}).items())))
    if key in _search_cache:
        return _search_cache[key]

    q = " ".join(query.splitlines())[:100]

    # Always-on Pixabay image defaults
    params: dict = {
        "key":         api_key,
        "q":           q,
        "image_type":  "photo",
        "orientation": "horizontal",
        "safesearch":  "true",
        "order":       "popular",
        "per_page":    per_page,
    }
    # Merge caller-supplied overrides (source_filters["pixabay"])
    if extra_params:
        params.update(extra_params)

    def call() -> list[dict]:
        r = requests.get(
            "https://pixabay.com/api/",
            params=params,
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        return r.json().get("hits", [])

    log.info("Pixabay search images  q=%r per_page=%d extra=%s", q, per_page, extra_params)
    result = _with_backoff(call, backoff)
    _search_cache[key] = result
    return result


def _pixabay_pick_image_url(hit: dict) -> Optional[str]:
    return hit.get("largeImageURL") or hit.get("webformatURL")


# ---------------------------------------------------------------------------
# Pixabay — videos
# ---------------------------------------------------------------------------

def _pixabay_search_videos(
    api_key: str,
    query: str,
    per_page: int,
    backoff: list[int],
    extra_params: dict | None = None,
) -> list[dict]:
    key = ("pixabay", "video", query, per_page, tuple(sorted((extra_params or {}).items())))
    if key in _search_cache:
        return _search_cache[key]

    q = " ".join(query.splitlines())[:100]

    # Always-on Pixabay video defaults
    params: dict = {
        "key":        api_key,
        "q":          q,
        "video_type": "film",
        "safesearch": "true",
        "order":      "popular",
        "per_page":   per_page,
    }
    # Merge caller-supplied overrides (source_filters["pixabay"])
    if extra_params:
        params.update(extra_params)

    def call() -> list[dict]:
        r = requests.get(
            "https://pixabay.com/api/videos/",
            params=params,
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        return r.json().get("hits", [])

    log.info("Pixabay search videos  q=%r per_page=%d extra=%s", q, per_page, extra_params)
    result = _with_backoff(call, backoff)
    _search_cache[key] = result
    return result


def _pixabay_pick_video_url(hit: dict) -> Optional[str]:
    vids = hit.get("videos") or {}
    for tier in ("large", "medium", "small", "tiny"):
        v = vids.get(tier) or {}
        if v.get("url"):
            return v["url"]
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_images(
    search_prompt: str,
    n:             int,
    output_dir:    Path,
    api_keys:      dict,
    cfg:           dict,
    item:          dict | None = None,
) -> list[tuple[Path, dict]]:
    """
    Search and download up to n images into output_dir/<source>/.
    Returns a list of (Path, info_dict) tuples (may be shorter than n × sources
    if some downloads fail or the API returns fewer results).

    Already-downloaded files (by filename) are reused without re-downloading.

    Parameters
    ----------
    search_prompt : str
        Fallback query used when item has no search_queries.
    n : int
        Total desired image count across all queries (deduplicated).
    output_dir : Path
        Base directory; sub-directories per source are created automatically.
    api_keys : dict
        Must contain "pexels" and/or "pixabay" keys.
    cfg : dict
        Server config dict (sources, backoff_seconds, rate_limits, …).
    item : dict | None
        Optional manifest item dict.  Recognised fields:
          - search_queries (list[str]): rotate through multiple queries
          - source_filters (dict):      per-source param overrides
    """
    if n <= 0:
        return []

    backoff  = cfg.get("backoff_seconds", [5, 15, 45])
    sources  = cfg.get("sources", ["pexels", "pixabay"])
    queries  = _get_queries(search_prompt, item, n)

    # Collect all hits across queries; deduplicate by (source, file_id)
    # before trimming to n so that multi-query rotation actually widens coverage.
    seen_ids: set[str] = set()
    saved: list[tuple[Path, dict]]  = []

    # Initialise thumbnail map on item so scorer.py thumbnail pass-1.5 can use it.
    # Maps str(local_dest_path) → thumbnail_url (smaller API-provided preview URL).
    if item is not None:
        item.setdefault("_thumbnails", {})

    for source in sources:
        src_dir = output_dir / source
        src_dir.mkdir(parents=True, exist_ok=True)

        # Source-specific filter overrides from the manifest item
        source_filters = _get_source_filters(item, source)

        for query, per_q in queries:
            # Per-query per_page budget (floor at provider minimums)
            per_page_pexels  = max(per_q, 10)
            per_page_pixabay = max(per_q, 20)

            try:
                if source == "pexels":
                    hits = _pexels_search_images(
                        api_keys.get("pexels", ""),
                        query,
                        per_page_pexels,
                        backoff,
                        extra_params=source_filters or None,
                    )
                    pexels_key = api_keys.get("pexels", "")
                    for i, ph in enumerate(hits[:per_q]):
                        pid  = str(ph.get("id", i))
                        uid  = f"pexels_img_{pid}"
                        if uid in seen_ids:
                            continue
                        seen_ids.add(uid)
                        url  = _pexels_pick_image_url(ph)
                        if not url:
                            continue
                        dest = src_dir / f"{uid}.jpg"
                        # Record thumbnail URL for pass-1.5 pre-filter in scorer.py
                        if item is not None:
                            src = ph.get("src") or {}
                            thumb = src.get("medium") or src.get("small") or src.get("tiny")
                            if thumb:
                                item["_thumbnails"][str(dest)] = thumb
                        info = {
                            "source_site":     "pexels",
                            "asset_page_url":  ph.get("url", ""),
                            "file_url":        url,
                            "title":           ph.get("alt") or _slug_to_title(ph.get("url", "")),
                            "description":     ph.get("alt", ""),
                            "tags":            [],
                            "photographer":    ph.get("photographer", ""),
                            "license_summary": "Pexels License",
                            "license_url":     "https://www.pexels.com/license/",
                            "width":           ph.get("width", 0),
                            "height":          ph.get("height", 0),
                        }
                        if dest.exists():
                            _write_info_sidecar(dest, info)
                            saved.append((dest, info))
                        else:
                            try:
                                _download_file(url, dest, headers={"Authorization": pexels_key})
                                _write_info_sidecar(dest, info)
                                saved.append((dest, info))
                            except Exception as exc:
                                log.warning("pexels img %s skip: %s", pid, exc)
                        _jitter(cfg, source)

                elif source == "pixabay":
                    hits = _pixabay_search_images(
                        api_keys.get("pixabay", ""),
                        query,
                        per_page_pixabay,
                        backoff,
                        extra_params=source_filters or None,
                    )
                    for i, hit in enumerate(hits[:per_q]):
                        hid  = str(hit.get("id", i))
                        uid  = f"pixabay_img_{hid}"
                        if uid in seen_ids:
                            continue
                        seen_ids.add(uid)
                        url  = _pixabay_pick_image_url(hit)
                        if not url:
                            continue
                        ext  = Path(url.split("?")[0]).suffix or ".jpg"
                        dest = src_dir / f"{uid}{ext}"
                        # webformatURL is a ~640px preview — ideal for pass-1.5 thumbnail filter
                        if item is not None:
                            thumb = hit.get("webformatURL") or hit.get("previewURL")
                            if thumb:
                                item["_thumbnails"][str(dest)] = thumb
                        info = {
                            "source_site":     "pixabay",
                            "asset_page_url":  hit.get("pageURL", ""),
                            "file_url":        url,
                            "title":           _slug_to_title(hit.get("pageURL", "")),
                            "description":     "",
                            "tags":            [t.strip() for t in hit.get("tags", "").split(",") if t.strip()],
                            "photographer":    hit.get("user", ""),
                            "license_summary": "Pixabay Content License",
                            "license_url":     "https://pixabay.com/service/license-summary/",
                            "width":           hit.get("imageWidth", 0),
                            "height":          hit.get("imageHeight", 0),
                        }
                        if dest.exists():
                            _write_info_sidecar(dest, info)
                            saved.append((dest, info))
                        else:
                            try:
                                _download_file(url, dest)
                                _write_info_sidecar(dest, info)
                                saved.append((dest, info))
                            except Exception as exc:
                                log.warning("pixabay img %s skip: %s", hid, exc)
                        _jitter(cfg, source)

            except Exception as exc:
                log.warning("fetch_images %s q=%r failed: %s", source, query, exc)

    # Return at most n results
    return saved[:n]


def fetch_videos(
    search_prompt: str,
    n:             int,
    output_dir:    Path,
    api_keys:      dict,
    cfg:           dict,
    item:          dict | None = None,
) -> list[tuple[Path, dict]]:
    """
    Search and download up to n videos into output_dir/<source>/.
    Returns a list of (Path, info_dict) tuples.

    Already-downloaded files are reused without re-downloading.

    Parameters
    ----------
    search_prompt : str
        Fallback query used when item has no search_queries.
    n : int
        Total desired video count across all queries (deduplicated).
    output_dir : Path
        Base directory; sub-directories per source are created automatically.
    api_keys : dict
        Must contain "pexels" and/or "pixabay" keys.
    cfg : dict
        Server config dict (sources, backoff_seconds, rate_limits, …).
    item : dict | None
        Optional manifest item dict.  Recognised fields:
          - search_queries (list[str]): rotate through multiple queries
          - source_filters (dict):      per-source param overrides
    """
    if n <= 0:
        return []

    backoff  = cfg.get("backoff_seconds", [5, 15, 45])
    sources  = cfg.get("sources", ["pexels", "pixabay"])
    queries  = _get_queries(search_prompt, item, n)

    seen_ids: set[str] = set()
    saved: list[tuple[Path, dict]]  = []

    for source in sources:
        src_dir = output_dir / source
        src_dir.mkdir(parents=True, exist_ok=True)

        # Source-specific filter overrides from the manifest item
        source_filters = _get_source_filters(item, source)

        for query, per_q in queries:
            per_page_pexels  = max(per_q, 10)
            per_page_pixabay = max(per_q, 20)

            try:
                if source == "pexels":
                    hits = _pexels_search_videos(
                        api_keys.get("pexels", ""),
                        query,
                        per_page_pexels,
                        backoff,
                        extra_params=source_filters or None,
                    )
                    for i, vd in enumerate(hits[:per_q]):
                        vid  = str(vd.get("id", i))
                        uid  = f"pexels_vid_{vid}"
                        if uid in seen_ids:
                            continue
                        seen_ids.add(uid)
                        url  = _pexels_pick_video_url(vd)
                        if not url:
                            continue
                        dest = src_dir / f"{uid}.mp4"
                        info = {
                            "source_site":     "pexels",
                            "asset_page_url":  vd.get("url", ""),
                            "file_url":        url,
                            "title":           _slug_to_title(vd.get("url", "")),
                            "description":     "",
                            "tags":            [],
                            "photographer":    (vd.get("user") or {}).get("name", ""),
                            "license_summary": "Pexels License",
                            "license_url":     "https://www.pexels.com/license/",
                            "width":           vd.get("width", 0),
                            "height":          vd.get("height", 0),
                        }
                        if dest.exists():
                            _write_info_sidecar(dest, info)
                            saved.append((dest, info))
                        else:
                            try:
                                _download_file(url, dest)
                                _write_info_sidecar(dest, info)
                                saved.append((dest, info))
                            except Exception as exc:
                                log.warning("pexels vid %s skip: %s", vid, exc)
                        _jitter(cfg, source)

                elif source == "pixabay":
                    hits = _pixabay_search_videos(
                        api_keys.get("pixabay", ""),
                        query,
                        per_page_pixabay,
                        backoff,
                        extra_params=source_filters or None,
                    )
                    for i, hit in enumerate(hits[:per_q]):
                        hid  = str(hit.get("id", i))
                        uid  = f"pixabay_vid_{hid}"
                        if uid in seen_ids:
                            continue
                        seen_ids.add(uid)
                        url  = _pixabay_pick_video_url(hit)
                        if not url:
                            continue
                        dest = src_dir / f"{uid}.mp4"
                        # Determine width/height from chosen tier
                        chosen_tier = None
                        for tier in ("large", "medium", "small", "tiny"):
                            v = (hit.get("videos") or {}).get(tier) or {}
                            if v.get("url"):
                                chosen_tier = v
                                break
                        info = {
                            "source_site":     "pixabay",
                            "asset_page_url":  hit.get("pageURL", ""),
                            "file_url":        url,
                            "title":           _slug_to_title(hit.get("pageURL", "")),
                            "description":     "",
                            "tags":            [t.strip() for t in hit.get("tags", "").split(",") if t.strip()],
                            "photographer":    hit.get("user", ""),
                            "license_summary": "Pixabay Content License",
                            "license_url":     "https://pixabay.com/service/license-summary/",
                            "width":           (chosen_tier or {}).get("width", 0),
                            "height":          (chosen_tier or {}).get("height", 0),
                        }
                        if dest.exists():
                            _write_info_sidecar(dest, info)
                            saved.append((dest, info))
                        else:
                            try:
                                _download_file(url, dest)
                                _write_info_sidecar(dest, info)
                                saved.append((dest, info))
                            except Exception as exc:
                                log.warning("pixabay vid %s skip: %s", hid, exc)
                        _jitter(cfg, source)

            except Exception as exc:
                log.warning("fetch_videos %s q=%r failed: %s", source, query, exc)

    # Return at most n results
    return saved[:n]
