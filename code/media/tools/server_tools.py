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

def cmd_get_batch_results(args):
    """
    Return the list of already-downloaded file paths for a specific item in a
    batch. Used by Claude for deduplication before triggering a new search.

    Output: JSON { item_id, images: [...paths], videos: [...paths], batch_status }
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
        }, indent=2))
        return

    # API response fields are "images" and "videos" — arrays of objects with
    # "path", "url", "score". Extract just the path strings for dedup.
    images = [img["path"] for img in item.get("images", []) if "path" in img]
    videos = [vid["path"] for vid in item.get("videos", []) if "path" in vid]

    print(json.dumps({
        "item_id":      args.item_id,
        "images":       images,
        "videos":       videos,
        "batch_status": batch_status,
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
        "sources_override": CC_SOURCES,
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


# ── Subcommand: append_to_batch ──────────────────────────────────────────────

def cmd_append_to_batch(args):
    """
    Add more images/videos to an existing batch item by triggering an
    additional search without replacing current results.

    Calls POST MEDIA_URL/batches/{batch_id}/items/{item_id}/append.

    Output: JSON { "status": "appending", "tmp_batch_id": "...", "poll_url": "..." }
    """
    if not MEDIA_KEY:
        _err("MEDIA_API_KEY env var is not set — required for media server auth")

    url     = f"{MEDIA_URL}/batches/{args.batch_id}/items/{args.item_id}/append"
    payload = {
        "ai_prompt": args.prompt,
        "n_img":     args.n_img,
        "n_vid":     args.n_vid,
    }
    try:
        result = _post(url, payload, MEDIA_KEY)
    except RuntimeError as e:
        _err(f"append_to_batch POST failed: {e}")

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
    p_sfs.add_argument("--n_img",    type=int, default=15, help="number of images to fetch")
    p_sfs.add_argument("--n_vid",    type=int, default=0,  help="number of videos to fetch")

    # append_to_batch
    p_atb = sub.add_parser("append_to_batch",
                           help="add more images/videos to an existing batch item")
    p_atb.add_argument("--batch_id", required=True, help="target batch id (e.g. b_822b661e)")
    p_atb.add_argument("--item_id",  required=True, help="asset_id of the background item")
    p_atb.add_argument("--prompt",   required=True, help="search terms for this append run")
    p_atb.add_argument("--n_img",    type=int, default=10, help="number of images to fetch")
    p_atb.add_argument("--n_vid",    type=int, default=0,  help="number of videos to fetch")

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

    args = parser.parse_args()

    dispatch = {
        "get_manifest":        cmd_get_manifest,
        "list_batches":        cmd_list_batches,
        "get_batch_results":   cmd_get_batch_results,
        "search_for_shot":     cmd_search_for_shot,
        "append_to_batch":     cmd_append_to_batch,
        "poll_append":         cmd_poll_append,
        "delete_batch_images": cmd_delete_batch_images,
    }
    try:
        dispatch[args.subcommand](args)
    except Exception as e:
        _err(f"unexpected error in {args.subcommand}: {e}")


if __name__ == "__main__":
    main()
