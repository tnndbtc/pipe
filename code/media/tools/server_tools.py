#!/usr/bin/env python3
"""
server_tools.py — CLI tool scripts called by Claude via bash during `media ask`.

Subcommands:
  get_manifest        — fetch AssetManifest.shared.json backgrounds list from pipeline proxy
  list_batches        — list existing media batches for an episode (newest first)
  get_batch_results   — get downloaded files for a specific batch + item (for dedup)
  search_for_shot     — trigger a new media search for a single shot
  append_to_batch     — add more images/videos to an existing batch item
  poll_append         — poll an append operation until done or failed
  delete_batch_images — prune images from a batch that do not match a filter

All subcommands print JSON to stdout and exit 0 on success, 1 on error.

Read tools (get_manifest, list_batches, get_batch_results) call the pipeline proxy.
Write tools (search_for_shot, append_to_batch, poll_append, delete_batch_images)
call the media server directly using MEDIA_API_KEY for authentication.

Environment variables:
  PIPELINE_SERVER_URL  — pipeline proxy base URL (default: http://localhost:8000)
  MEDIA_SERVER_URL     — media server base URL   (default: http://192.168.86.33:8200)
  MEDIA_API_KEY        — API key for X-Api-Key header (media server auth)
"""

import argparse
import json
import math
import os
import sys
import urllib.request
import urllib.error
import urllib.parse

# ── Environment defaults ────────────────────────────────────────────────────

PIPELINE_URL = os.environ.get("PIPELINE_SERVER_URL", "http://localhost:8000").rstrip("/")
MEDIA_URL    = os.environ.get("MEDIA_SERVER_URL",    "http://192.168.86.33:8200").rstrip("/")
MEDIA_KEY    = os.environ.get("MEDIA_API_KEY",       "")

# Sources used for media search. Pexels and Pixabay use their own free licenses
# (not CC) but are free for personal and commercial use without attribution.
# iStock/Getty sponsored results are web-UI only and never returned by the API.
CC_SOURCES = ["wikimedia", "openverse", "europeana", "pexels", "pixabay"]


# ── HTTP helpers ─────────────────────────────────────────────────────────────

def _get(url: str) -> dict:
    """GET url, return parsed JSON. Raises RuntimeError on any network or HTTP error."""
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code} GET {url}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"connection error GET {url}: {e.reason}") from e


def _post(url: str, payload: dict, api_key: str) -> dict:
    """POST JSON payload to url with X-Api-Key header, return parsed JSON."""
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", "X-Api-Key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code} POST {url}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"connection error POST {url}: {e.reason}") from e


def _err(msg: str, code: int = 1):
    print(json.dumps({"error": msg}))
    sys.exit(code)


# ── Subcommand: get_manifest ─────────────────────────────────────────────────

def cmd_get_manifest(args):
    """
    Fetch AssetManifest.shared.json from the pipeline proxy and return
    the backgrounds array so Claude knows the shot structure.

    Output: JSON array [ { asset_id, shot_id, ai_prompt, scene_id, ... }, ... ]
    """
    url = (
        f"{PIPELINE_URL}/api/episode_file"
        f"?slug={urllib.parse.quote(args.project)}"
        f"&ep_id={urllib.parse.quote(args.episode)}"
        f"&file=AssetManifest.shared.json"
    )
    try:
        manifest = _get(url)
    except RuntimeError as e:
        _err(str(e))

    backgrounds = manifest.get("backgrounds")
    if not backgrounds:
        _err("no backgrounds in manifest")

    # Normalise: manifest may store backgrounds as dict keyed by asset_id
    if isinstance(backgrounds, dict):
        backgrounds = list(backgrounds.values())

    print(json.dumps(backgrounds, indent=2))


# ── Subcommand: list_batches ─────────────────────────────────────────────────

def cmd_list_batches(args):
    """
    List all media batches for an episode, newest first.

    Output: JSON array [ { batch_id, status, created_at, item_count }, ... ]
    """
    url = (
        f"{PIPELINE_URL}/api/media_batches"
        f"?slug={urllib.parse.quote(args.project)}"
        f"&ep_id={urllib.parse.quote(args.episode)}"
        f"&server_url={urllib.parse.quote(MEDIA_URL)}"
    )
    try:
        data = _get(url)
    except RuntimeError as e:
        _err(str(e))

    batches = data.get("batches", [])
    # Sort newest first by created_at (ISO string sort works for RFC3339)
    batches.sort(key=lambda b: b.get("created_at", ""), reverse=True)
    print(json.dumps(batches, indent=2))


# ── Subcommand: get_batch_results ────────────────────────────────────────────

def _count_by_source(entries: list) -> dict:
    """
    Count entries by source_site.  Prefers entry["source"]["source_site"];
    falls back to path parsing for relative paths
    (format: {item_id}/{images|videos}/{source}/{filename}).
    """
    counts: dict = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        src = (entry.get("source") or {}).get("source_site", "") \
              if isinstance(entry.get("source"), dict) else ""
        if not src:
            path = entry.get("path", "").replace("\\", "/")
            if path and not path.startswith("/"):
                parts = path.split("/")
                if len(parts) >= 4:
                    src = parts[2]
        if src:
            counts[src] = counts.get(src, 0) + 1
    return counts


def cmd_get_batch_results(args):
    """
    Return the list of already-downloaded file paths for a specific item in a
    batch, plus per-source image/video counts for page-math calculations.

    Output: JSON {
      item_id, images: [...paths], videos: [...paths], batch_status,
      source_counts: {
        images: { "pexels": N, "pixabay": N, ... },
        videos: { "pexels": N, ... }
      }
    }
    """
    url = (
        f"{PIPELINE_URL}/api/media_batch_status"
        f"?batch_id={urllib.parse.quote(args.batch_id)}"
        f"&server_url={urllib.parse.quote(MEDIA_URL)}"
    )
    try:
        state = _get(url)
    except RuntimeError as e:
        _err(str(e))

    batch_status = state.get("status", "unknown")

    # If batch is still running, nothing is ranked yet — return empty with status
    if batch_status != "done":
        print(json.dumps({
            "item_id":      args.item_id,
            "images":       [],
            "videos":       [],
            "batch_status": batch_status,
            "source_counts": {"images": {}, "videos": {}},
        }, indent=2))
        return

    items = state.get("items", {})
    item  = items.get(args.item_id)
    if item is None:
        print(json.dumps({
            "item_id":      args.item_id,
            "images":       [],
            "videos":       [],
            "batch_status": batch_status,
            "source_counts": {"images": {}, "videos": {}},
        }, indent=2))
        return

    # API response fields are "images" and "videos" — arrays of objects with
    # "path", "url", "score", "source". Extract paths and per-source counts.
    img_entries = item.get("images", [])
    vid_entries = item.get("videos", [])
    images = [img["path"] for img in img_entries if "path" in img]
    videos = [vid["path"] for vid in vid_entries if "path" in vid]

    print(json.dumps({
        "item_id":      args.item_id,
        "images":       images,
        "videos":       videos,
        "batch_status": batch_status,
        "source_counts": {
            "images": _count_by_source(img_entries),
            "videos": _count_by_source(vid_entries),
        },
    }, indent=2))


# ── Subcommand: search_for_shot ───────────────────────────────────────────────

def cmd_search_for_shot(args):
    """
    Trigger a new media search for a single shot item.
    Calls the media server DIRECTLY (bypasses the pipeline proxy).

    Sources: Wikimedia, Openverse, Europeana (CC0/CC BY) plus Pexels and Pixabay
    (free for personal and commercial use under their own licenses).

    Output: JSON { batch_id, item_id, status, poll_url }
    """
    if not MEDIA_KEY:
        _err("MEDIA_API_KEY env var is not set — required for media server auth")

    # Step 1: fetch manifest to find the target item
    manifest_url = (
        f"{PIPELINE_URL}/api/episode_file"
        f"?slug={urllib.parse.quote(args.project)}"
        f"&ep_id={urllib.parse.quote(args.episode)}"
        f"&file=AssetManifest.shared.json"
    )
    try:
        manifest = _get(manifest_url)
    except RuntimeError as e:
        _err(f"could not fetch manifest: {e}")

    backgrounds = manifest.get("backgrounds", {})
    if isinstance(backgrounds, dict):
        backgrounds = list(backgrounds.values())

    target = next((b for b in backgrounds if b.get("asset_id") == args.item_id), None)
    if target is None:
        _err(f"item_id '{args.item_id}' not found in manifest backgrounds")

    # Step 2: build minimal single-item manifest with user's query as ai_prompt
    minimal_manifest = {
        "backgrounds": [{**target, "ai_prompt": args.query}]
    }

    # Step 3: POST to media server /batches
    # Do NOT use content_profile for license filtering — it is a CLIP scoring
    # profile only (valid: "default", "sleep_story", "documentary", "action").
    payload = {
        "project":          args.project,
        "episode_id":       args.episode,
        "manifest":         minimal_manifest,
        "sources_override": args.sources if args.sources else CC_SOURCES,
        "n_img":            args.n_img,
        "n_vid":            args.n_vid,
    }

    batches_url = f"{MEDIA_URL}/batches"
    try:
        resp = _post(batches_url, payload, MEDIA_KEY)
    except RuntimeError as e:
        _err(f"media server POST /batches failed: {e}")

    batch_id = resp.get("batch_id")
    if not batch_id:
        _err(f"media server response missing batch_id: {resp}")

    print(json.dumps({
        "batch_id": batch_id,
        "item_id":  args.item_id,
        "status":   resp.get("status", "queued"),
        "poll_url": f"{MEDIA_URL}/batches/{batch_id}",
    }, indent=2))


# Page sizes for Group A sources (those that use simple page=N pagination).
# Must match the page_size constants in downloader.py.
_GROUP_A_PAGE_SIZES = {
    "pexels":    80,
    "pixabay":   200,
    "openverse": 50,
}


# ── Subcommand: append_to_batch ──────────────────────────────────────────────

def cmd_append_to_batch(args):
    """
    Add more images/videos to an existing batch item by triggering an
    additional search without replacing current results.

    For Group A sources (pexels, pixabay, openverse) the tool auto-adjusts
    n_img to skip pages already downloaded: it fetches the current per-source
    image count, then computes n_img = (pages_already_fetched + pages_for_new)
    × page_size so the downloader fetches at least one new page beyond what
    was already seen.

    Calls POST MEDIA_URL/batches/{batch_id}/items/{item_id}/append.

    Output: JSON { "status": "appending", "tmp_batch_id": "...", "poll_url": "..." }
    """
    if not MEDIA_KEY:
        _err("MEDIA_API_KEY env var is not set — required for media server auth")

    # ── Page-math for Group A sources ────────────────────────────────────────
    effective_sources = args.sources if args.sources else CC_SOURCES
    group_a_active    = [s for s in _GROUP_A_PAGE_SIZES if s in effective_sources]
    n_img_adjusted    = args.n_img  # start with user's value (may be None)

    if group_a_active and args.n_img is not None:
        status_url = (
            f"{PIPELINE_URL}/api/media_batch_status"
            f"?batch_id={urllib.parse.quote(args.batch_id)}"
            f"&server_url={urllib.parse.quote(MEDIA_URL)}"
        )
        try:
            state        = _get(status_url)
            batch_status = state.get("status", "unknown")
            item         = state.get("items", {}).get(args.item_id, {})

            # Count existing images per source (only meaningful for done batches)
            source_counts: dict = {}
            if batch_status == "done":
                source_counts = _count_by_source(item.get("images", []))

            # Per-source adjusted n_img; take the maximum across all active
            # Group A sources so every source gets enough pages fetched.
            user_n_img   = args.n_img
            max_adjusted = user_n_img
            for source in group_a_active:
                page_size      = _GROUP_A_PAGE_SIZES[source]
                existing       = source_counts.get(source, 0)
                pages_already  = math.ceil(existing / page_size) if existing > 0 else 0
                pages_for_new  = math.ceil(user_n_img / page_size)
                adjusted       = (pages_already + pages_for_new) * page_size
                max_adjusted   = max(max_adjusted, adjusted)
                print(
                    f"# page-math: {source} existing={existing} "
                    f"pages_skip={pages_already} → n_img {user_n_img}→{adjusted}",
                    file=sys.stderr,
                )
            n_img_adjusted = max_adjusted

        except RuntimeError as e:
            print(
                f"# page-math: could not fetch batch state ({e}), "
                f"using n_img={args.n_img}",
                file=sys.stderr,
            )

    # ── POST to media server ──────────────────────────────────────────────────
    url     = f"{MEDIA_URL}/batches/{args.batch_id}/items/{args.item_id}/append"
    payload = {
        "ai_prompt":     args.prompt,
        "n_img":         n_img_adjusted,
        "n_vid":         args.n_vid,
        "n_img_requested": args.n_img,   # original user count — server caps merge to this
        "n_vid_requested": args.n_vid,   # same for videos
    }
    if args.sources:
        payload["sources_override"] = args.sources
    try:
        result = _post(url, payload, MEDIA_KEY)
    except RuntimeError as e:
        _err(f"append_to_batch POST failed: {e}")

    # Surface the adjustment so Claude can mention it in its response
    if n_img_adjusted != args.n_img and args.n_img is not None:
        result["_n_img_adjusted"] = n_img_adjusted

    print(json.dumps(result, indent=2))


# ── Subcommand: poll_append ───────────────────────────────────────────────────

def cmd_poll_append(args):
    """
    Poll the status of an append operation.
    When the mini-batch is done the server merges results and returns counts.

    Calls GET MEDIA_URL/batches/{batch_id}/items/{item_id}/append/{tmp_batch_id}
    with X-Api-Key authentication.

    Output: JSON { "status": "pending"|"done"|"failed", ... }
    """
    if not MEDIA_KEY:
        _err("MEDIA_API_KEY env var is not set — required for media server auth")

    url = (
        f"{MEDIA_URL}/batches/{args.batch_id}"
        f"/items/{args.item_id}"
        f"/append/{args.tmp_batch_id}"
    )
    # GET with auth header — use urllib.request.Request directly
    req = urllib.request.Request(url, headers={"X-Api-Key": MEDIA_KEY}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        _err(f"HTTP {e.code} GET {url}: {body}")
    except urllib.error.URLError as e:
        _err(f"connection error GET {url}: {e.reason}")

    print(json.dumps(result, indent=2))


# ── Subcommand: delete_batch_images ──────────────────────────────────────────

def cmd_delete_batch_images(args):
    """
    Prune images and/or videos from an existing batch that do not match a filter.

    Sends a FilterSpec to the pipeline proxy → media server, which permanently
    deletes non-matching files from disk and updates batch_state.json.

    Output: JSON {
      batch_id,
      items: {
        item_id: {
          images: {deleted: N, kept: N},
          videos: {deleted: N, kept: N}
        }
      }
    }
    """
    if not MEDIA_KEY:
        _err("MEDIA_API_KEY env var is not set — required for media server auth")

    # Validate and parse --filter
    try:
        filter_spec = json.loads(args.filter)
    except json.JSONDecodeError as e:
        _err(f"--filter is not valid JSON: {e}")

    # Convert optional --item_id to list form expected by the API
    item_ids = [args.item_id] if args.item_id else None

    payload = {
        "batch_id":    args.batch_id,
        "item_ids":    item_ids,
        "filter_spec": filter_spec,
        "server_url":  MEDIA_URL,   # tells proxy which media server to forward to
    }

    url = f"{PIPELINE_URL}/api/media_batch_prune"
    try:
        result = _post(url, payload, MEDIA_KEY)
    except RuntimeError as e:
        _err(f"media_batch_prune failed: {e}")

    # Print full JSON (Claude reads this), then a human-readable summary line
    print(json.dumps(result, indent=2))

    items = result.get("items", {})
    total_img_del = sum(v.get("images", {}).get("deleted", 0) for v in items.values())
    total_img_kpt = sum(v.get("images", {}).get("kept",    0) for v in items.values())
    total_vid_del = sum(v.get("videos", {}).get("deleted", 0) for v in items.values())
    total_vid_kpt = sum(v.get("videos", {}).get("kept",    0) for v in items.values())
    print(
        f"# summary: images deleted={total_img_del} kept={total_img_kpt}"
        f"  |  videos deleted={total_vid_del} kept={total_vid_kpt}"
    )


# ── Subcommand: fetch_by_url ──────────────────────────────────────────────────

def cmd_fetch_by_url(args):
    """
    Download a specific Pexels photo by its page URL and append it to a batch item.
    Calls POST MEDIA_URL/fetch_direct (synchronous — no polling needed).

    Output: JSON { "photo_id": N, "path": "...", "title": "..." }
    """
    if not MEDIA_KEY:
        _err("MEDIA_API_KEY env var is not set — required for media server auth")

    payload = {
        "batch_id": args.batch_id,
        "item_id":  args.item_id,
        "url":      args.url,
    }
    url = f"{MEDIA_URL}/fetch_direct"
    try:
        result = _post(url, payload, MEDIA_KEY)
    except RuntimeError as e:
        _err(f"fetch_by_url failed: {e}")

    print(json.dumps(result, indent=2))
    title = result.get("title", "")
    path  = result.get("path", "")
    print(f"# ✓ Added: {title}  ({path})")


# ── CLI dispatcher ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="server_tools",
        description="Media server tool scripts for `media ask` (called by Claude via bash)",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # get_manifest
    p_gm = sub.add_parser("get_manifest", help="fetch backgrounds from AssetManifest")
    p_gm.add_argument("--project",  required=True, help="project slug")
    p_gm.add_argument("--episode",  required=True, help="episode id (e.g. s01e01)")

    # list_batches
    p_lb = sub.add_parser("list_batches", help="list media batches for episode")
    p_lb.add_argument("--project",  required=True)
    p_lb.add_argument("--episode",  required=True)

    # get_batch_results
    p_gbr = sub.add_parser("get_batch_results", help="get downloaded files in a batch item")
    p_gbr.add_argument("--batch_id", required=True, help="batch id (e.g. b_822b661e)")
    p_gbr.add_argument("--item_id",  required=True, help="asset_id of the background item")

    # search_for_shot
    p_sfs = sub.add_parser("search_for_shot", help="trigger a new media search for a shot")
    p_sfs.add_argument("--project",  required=True)
    p_sfs.add_argument("--episode",  required=True)
    p_sfs.add_argument("--item_id",  required=True, help="asset_id from get_manifest output")
    p_sfs.add_argument("--query",    required=True, help="search terms (overrides ai_prompt)")
    p_sfs.add_argument("--n_img",    type=int, default=None, help="number of images to fetch")
    p_sfs.add_argument("--n_vid",    type=int, default=None, help="number of videos to fetch")
    p_sfs.add_argument("--sources",  nargs="+", default=None,
                       help="override sources list (e.g. pexels pixabay wikimedia)")

    # append_to_batch
    p_atb = sub.add_parser("append_to_batch",
                           help="add more images/videos to an existing batch item")
    p_atb.add_argument("--batch_id", required=True, help="target batch id (e.g. b_822b661e)")
    p_atb.add_argument("--item_id",  required=True, help="asset_id of the background item")
    p_atb.add_argument("--prompt",   required=True, help="search terms for this append run")
    p_atb.add_argument("--n_img",    type=int, default=None, help="number of images to fetch")
    p_atb.add_argument("--n_vid",    type=int, default=None, help="number of videos to fetch")
    p_atb.add_argument("--sources",  nargs="+", default=None,
                       help="override sources list (e.g. pexels pixabay wikimedia)")

    # poll_append
    p_pa = sub.add_parser("poll_append",
                          help="poll an append operation until done or failed")
    p_pa.add_argument("--batch_id",     required=True, help="target batch id")
    p_pa.add_argument("--item_id",      required=True, help="asset_id of the background item")
    p_pa.add_argument("--tmp_batch_id", required=True,
                      help="tmp_batch_id returned by append_to_batch")

    # delete_batch_images
    p_dbi = sub.add_parser("delete_batch_images",
                           help="prune images from a batch that do not match a filter")
    p_dbi.add_argument("--batch_id", required=True,
                       help="batch id (e.g. b_822b661e)")
    p_dbi.add_argument("--item_id",  required=False, default=None,
                       help="if given, only prune this item; else all items in the batch")
    p_dbi.add_argument("--filter",   required=True,
                       help='FilterSpec as JSON string '
                            '(e.g. \'{"exclude_sources": ["pexels"], "min_score": 0.5}\')')

    # fetch_by_url
    p_fbu = sub.add_parser("fetch_by_url",
                           help="download a specific Pexels photo by URL and add to a batch")
    p_fbu.add_argument("--batch_id", required=True, help="target batch id (e.g. b_822b661e)")
    p_fbu.add_argument("--item_id",  required=True, help="asset_id of the target shot")
    p_fbu.add_argument("--url",      required=True,
                       help="Pexels photo page URL "
                            "(e.g. https://www.pexels.com/photo/ruins-7535541/)")

    args = parser.parse_args()

    dispatch = {
        "get_manifest":        cmd_get_manifest,
        "list_batches":        cmd_list_batches,
        "get_batch_results":   cmd_get_batch_results,
        "search_for_shot":     cmd_search_for_shot,
        "append_to_batch":     cmd_append_to_batch,
        "poll_append":         cmd_poll_append,
        "delete_batch_images": cmd_delete_batch_images,
        "fetch_by_url":        cmd_fetch_by_url,
    }
    try:
        dispatch[args.subcommand](args)
    except Exception as e:
        _err(f"unexpected error in {args.subcommand}: {e}")


if __name__ == "__main__":
    main()
