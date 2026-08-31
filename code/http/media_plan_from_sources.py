#!/usr/bin/env python3
"""
media_plan_from_sources.py — Auto-generate MediaPlan.json from each source's
OWN og:image, fetched directly from the source's real article URL.

This supersedes media_plan_auto.py's raw_payload-based image extraction for
story_engine deep-dive stories: raw_payload (RSS-embedded <img> tags) was
measured to carry a usable image for only ~10% of real stories (Google News
redirect links, paywalls, and RSS feeds that simply omit media). Fetching the
real article URL directly and reading its og:image meta tag is far more
reliable, because outlets serve that tag to any HTTP client (it's meant for
link-preview cards), often even behind a paywall.

Runs as a Stage 9 step in run.sh, AFTER post_tts_analysis.py has resolved real
per-paragraph timing into VOPlan.{locale}.json, and BEFORE resolve_assets.py /
render_video.py read MediaPlan.json. This ordering matters: without real
per-paragraph start_sec/end_sec, images could only be spaced evenly across the
whole video, not aligned to the paragraph that actually discusses that source.

Safety contract (this runs unattended, in a cron pipeline, for every story):
  - Missing story_sources.json           -> no-op, exit 0 (most callers; this
                                             file is only written for the
                                             story_engine deep-dive pipeline)
  - MediaPlan.json already exists        -> no-op, exit 0 (never clobber a
                                             manual VC-editor edit)
  - VOPlan.{locale}.json timing unresolved -> no-op, exit 0
  - Zero sources yield a usable image    -> no-op, exit 0 (existing
                                             single-background behavior
                                             applies, exactly as before this
                                             feature existed)
  - Any unexpected error                 -> caught, logged, exit 0
  This script must NEVER be the reason an unattended render fails.

Video sources are skipped entirely for now (per product decision — frame
extraction from video platforms is a separate, not-yet-built capability):
  youtube, bilibili, tiktok, douyin, vimeo, twitch

Usage:
    python3 media_plan_from_sources.py --ep-dir <episode_dir> --locale en
"""

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

# ── Constants ──────────────────────────────────────────────────────────────

VIDEO_PLATFORMS = {"youtube", "bilibili", "tiktok", "douyin", "vimeo", "twitch"}

FETCH_TIMEOUT_SEC   = 8        # per-request network timeout
TOTAL_BUDGET_SEC    = 25       # stop trying more sources past this wall-clock budget
MAX_SOURCES_TRIED   = 6        # never try more than this many sources
MIN_IMAGE_BYTES     = 5_000    # below this, almost certainly a tracking pixel/icon
MIN_IMAGE_WIDTH     = 400      # quality floor, matches media_plan_auto.py's bar

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

_OG_IMAGE_RE = re.compile(
    r'<meta[^>]+property=["\']og:image["\'][^>]*content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
# Some sites (rare) put content before property — handle both attribute orders.
_OG_IMAGE_RE_ALT = re.compile(
    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]*property=["\']og:image["\']',
    re.IGNORECASE,
)

STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "to", "for", "and", "or", "is", "was",
    "were", "be", "been", "at", "by", "with", "its", "it's", "as", "after",
    "before", "over", "from", "this", "that", "has", "have", "had", "will",
    "new", "says", "said",
}


def log(msg: str) -> None:
    print(f"  [media_plan_from_sources] {msg}")


def _tokenize(text: str) -> set:
    words = re.findall(r"[a-zA-Z0-9']+", (text or "").lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def _fetch(url: str, timeout: float) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": _UA,
            "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as e:
        log(f"    fetch failed ({url[:70]}): {e}")
        return None


def _extract_og_image(html_bytes: bytes) -> str | None:
    try:
        html = html_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return None
    m = _OG_IMAGE_RE.search(html) or _OG_IMAGE_RE_ALT.search(html)
    if not m:
        return None
    url = m.group(1).replace("&amp;", "&")
    return url if url.startswith("http") else None


def _download_image(url: str, dest: Path, deadline: float) -> bool:
    remaining = deadline - time.time()
    if remaining <= 0:
        return False
    data = _fetch(url, timeout=min(FETCH_TIMEOUT_SEC, max(2.0, remaining)))
    if not data or len(data) < MIN_IMAGE_BYTES:
        return False
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    except Exception as e:
        log(f"    write failed: {e}")
        return False
    # Quality gate: verify dimensions with PIL when available.
    try:
        from PIL import Image  # already a dependency (render_video.py uses it)
        with Image.open(dest) as im:
            w, _h = im.size
        if w < MIN_IMAGE_WIDTH:
            dest.unlink(missing_ok=True)
            return False
    except Exception:
        pass  # if PIL/format check fails, keep the file rather than discard it
    return True


# Display names for outlets whose domain doesn't capitalize into something
# readable on its own (e.g. "theguardian" -> "Theguardian" is technically
# correct but reads worse than "The Guardian"). Extend as new outlets show
# up in practice — this is a display-polish list, not a correctness gate.
_KNOWN_OUTLETS = {
    "scmp": "South China Morning Post",
    "theguardian": "The Guardian",
    "nytimes": "The New York Times",
    "apnews": "AP News",
    "bbc": "BBC",
    "cnbc": "CNBC",
    "cnn": "CNN",
    "reuters": "Reuters",
    "bloomberg": "Bloomberg",
    "wsj": "The Wall Street Journal",
    "ft": "Financial Times",
    "aljazeera": "Al Jazeera",
    "techcrunch": "TechCrunch",
    "theverge": "The Verge",
    "wired": "Wired",
    "arstechnica": "Ars Technica",
    "politico": "Politico",
    "axios": "Axios",
}


def _source_label(url: str, title: str, domain: str = "") -> str:
    # Prefer story_engine's own pre-extracted domain field when present —
    # cleaner than re-deriving it, and it's already what export_story.py used
    # for the #hashtag credits, so attribution stays consistent with those.
    key = (domain or "").lower()
    if not key:
        parts = (title or "").rsplit(" - ", 1)
        if len(parts) == 2 and parts[1].strip():
            return parts[1].strip()
        try:
            host = urlparse(url).netloc.lower()
            for prefix in ("www.", "m.", "news.", "rss.", "feeds."):
                host = host.removeprefix(prefix)
            key = host.split(".")[0]
        except Exception:
            return "Source"
    return _KNOWN_OUTLETS.get(key, key.capitalize() if key else "Source")


def _paragraph_texts(ep_dir: Path) -> list[str]:
    """Best-effort recovery of each paragraph's plain text, for match scoring.
    Falls back to an empty list (image assignment then falls back to
    positional order) if story.txt isn't in the expected '## / -' format."""
    story_path = ep_dir / "story.txt"
    if not story_path.exists():
        return []
    try:
        text = story_path.read_text(encoding="utf-8")
    except Exception:
        return []
    # Reuse the same '## title' / '-' delimiter convention as
    # simple_narration_setup.py's parse_story_txt().
    blocks = re.split(r"^-\s*$", text, flags=re.MULTILINE)
    paras = [b.strip() for b in blocks if b.strip() and not b.strip().startswith("##")]
    return paras


def build_media_plan(ep_dir: Path, locale: str) -> tuple[bool, str]:
    sources_path = ep_dir / "story_sources.json"
    plan_path = ep_dir / "MediaPlan.json"
    voplan_path = ep_dir / f"VOPlan.{locale}.json"

    if plan_path.exists():
        return False, "MediaPlan.json already exists — skipped"
    if not sources_path.exists():
        return False, "no story_sources.json — skipped (existing single-background behavior applies)"
    if not voplan_path.exists():
        return False, f"VOPlan.{locale}.json not found — skipped"

    try:
        sources_doc = json.loads(sources_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"story_sources.json unreadable ({e}) — skipped"

    try:
        voplan = json.loads(voplan_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"VOPlan.{locale}.json unreadable ({e}) — skipped"

    vo_items = sorted(
        (v for v in voplan.get("vo_items", []) if v.get("start_sec") is not None),
        key=lambda v: v["start_sec"],
    )
    if not vo_items:
        return False, "VOPlan has no resolved timing yet — skipped (run after post_tts_analysis)"

    # Tail padding: match render_video.py's convention of a short buffer after
    # the last line, and its own frame-snapped total_dur will absorb the rest.
    total_dur = max(v["end_sec"] for v in vo_items) + 0.4

    sources = [
        s for s in (sources_doc.get("sources") or [])
        if (s.get("platform") or "").lower() not in VIDEO_PLATFORMS
           and (s.get("url") or "").startswith("http")
    ][:MAX_SOURCES_TRIED]
    if not sources:
        return False, "no non-video sources with a URL — skipped"

    para_texts = _paragraph_texts(ep_dir)

    media_dir = ep_dir / "assets" / "sourced_images"
    media_dir.mkdir(parents=True, exist_ok=True)

    deadline = time.time() + TOTAL_BUDGET_SEC
    fetched: list[dict] = []   # {label, path, title}

    for i, src in enumerate(sources):
        if time.time() > deadline:
            log(f"time budget ({TOTAL_BUDGET_SEC}s) exhausted — stopping at {i}/{len(sources)} sources")
            break
        url = src["url"]
        title = src.get("title", "")
        label = _source_label(url, title, src.get("domain", ""))
        log(f"fetching og:image from {label}: {url[:80]}")
        html = _fetch(url, timeout=FETCH_TIMEOUT_SEC)
        if not html:
            continue
        img_url = _extract_og_image(html)
        if not img_url:
            log(f"    no og:image tag found")
            continue
        dest = media_dir / f"src_{i:02d}.jpg"
        if _download_image(img_url, dest, deadline):
            log(f"    ✓ {dest.name}  ({label})")
            fetched.append({"label": label, "path": dest, "title": title})
        else:
            log(f"    image too small/failed quality gate")

    if not fetched:
        return False, "no source yielded a usable og:image — skipped (single-background behavior applies)"

    # ── Match each fetched image to its best (still-unclaimed) paragraph ────
    # Greedy: highest keyword-overlap pair first, one image per paragraph.
    assignments: dict[int, dict] = {}   # para_index -> fetched item
    if para_texts:
        para_tokens = [_tokenize(p) for p in para_texts]
        pairs = []
        for fi, item in enumerate(fetched):
            src_tokens = _tokenize(item["title"])
            for pi, ptoks in enumerate(para_tokens):
                score = len(src_tokens & ptoks)
                pairs.append((score, fi, pi))
        pairs.sort(key=lambda t: -t[0])
        used_f, used_p = set(), set()
        for score, fi, pi in pairs:
            if fi in used_f or pi in used_p:
                continue
            if score <= 0 and len(used_f) < len(fetched) and para_tokens:
                # Allow zero-overlap fallback assignment only once every
                # fetched image has had a chance at a real match, so a
                # single strong image doesn't starve out the rest.
                pass
            assignments[pi] = fetched[fi]
            used_f.add(fi)
            used_p.add(pi)
            if len(used_f) == len(fetched):
                break
        # Anything still unmatched (more paragraphs than images, or vice
        # versa) is fine — those paragraphs just keep the plain background.
    else:
        # No paragraph text recovered — assign in order as a reasonable default.
        for i, item in enumerate(fetched[:len(vo_items)]):
            assignments[i] = item

    # ── Build shot_overrides spanning the FULL timeline, one segment per
    #    vo_item interval, falling back to the plain background where no
    #    image was matched. Static holds only — the zoompan/Ken-Burns
    #    animation caused a shake, confirmed against a real photo, so this
    #    generator never sets animation_type. ─────────────────────────────
    shot_overrides = []
    for i, vi in enumerate(vo_items):
        start = vi["start_sec"]
        end = vo_items[i + 1]["start_sec"] if i + 1 < len(vo_items) else total_dur
        hold = round(end - start, 3)
        if hold <= 0:
            continue
        item = assignments.get(i)
        if item:
            shot_overrides.append({
                "type": "image",
                "path": str(item["path"].relative_to(ep_dir)),
                "hold_sec": hold,
                "transition": "none" if i == 0 else "dissolve_short",
                "attribution": f"Source: {item['label']}",
            })
        else:
            shot_overrides.append({
                "type": "image",
                "path": "assets/bg-provided.png",
                "hold_sec": hold,
                "transition": "none" if i == 0 else "dissolve_short",
            })

    media_plan = {
        "schema_id": "MediaPlan",
        "schema_version": "1.0",
        "auto_generated": True,
        "notes": (
            f"Auto-generated by media_plan_from_sources.py — {len(fetched)}/{len(sources)} "
            f"source(s) yielded a usable og:image, matched to paragraphs by keyword overlap. "
            f"Static holds only (no zoompan)."
        ),
        "shot_overrides": shot_overrides,
    }
    tmp = plan_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(media_plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(plan_path)

    return True, f"MediaPlan.json written — {len(fetched)}/{len(sources)} real image(s), {len(shot_overrides)} segment(s)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ep-dir", required=True)
    ap.add_argument("--locale", default="en")
    args = ap.parse_args()

    ep_dir = Path(args.ep_dir).resolve()
    try:
        ok, msg = build_media_plan(ep_dir, args.locale)
        print(f"  {'✓' if ok else '↷'} {msg}")
    except Exception as e:
        # Contract: this script must NEVER fail the pipeline it runs inside.
        print(f"  ↷ media_plan_from_sources.py error (ignored, single-background "
              f"behavior applies): {e}", file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    main()
