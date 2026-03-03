"""
server.py — Media Search & Rating Service
==========================================

FastAPI server that, given an asset manifest containing a `backgrounds` map,
searches Pexels + Pixabay for every background item, scores each candidate
with CLIP + calmness, and returns top-N ranked candidates per item.

Usage
-----
    cd code/media/http/
    python server.py                                 # uses config.json defaults
    python server.py --host 0.0.0.0 --port 8200

    # or via uvicorn directly:
    uvicorn server:app --host 0.0.0.0 --port 8200

Configuration
-------------
    config.json  (same directory as server.py)
    Required config fields: projects_root
    API keys are NOT stored in config.json — set environment variables instead:
        MEDIA_API_KEY        server authentication key
        PEXELS_API_KEY       required when 'pexels' is in config sources
        PIXABAY_API_KEY      required when 'pixabay' is in config sources
    The server prints ERROR and exits at startup if any required key is missing.

Endpoints
---------
    POST /batches                        start a new batch (authenticated)
    GET  /batches                        list batches for a project/episode (auth)
    GET  /batches/{batch_id}             poll status / get results (auth)
    GET  /files/{path:path}              serve cached media files (no auth)
    GET  /health                         server status (no auth)

Authentication
--------------
    All endpoints except GET /files/ and GET /health require:
        X-Api-Key: <value of MEDIA_API_KEY environment variable>
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import batch_store as bs
import cleanup
import downloader
import scorer
import sequence_ranker

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("server")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_cfg_path = Path(__file__).parent / "config.json"
config: dict = json.loads(_cfg_path.read_text(encoding="utf-8"))

PROJECTS_ROOT = (Path(__file__).parent / config["projects_root"]).resolve()


# ---------------------------------------------------------------------------
# API key loading — keys come from environment variables, never from config
# ---------------------------------------------------------------------------

def _load_api_keys() -> tuple[str, dict[str, str]]:
    """
    Load all required API keys from environment variables.

    Required variables:
      MEDIA_API_KEY        — server authentication key (X-Api-Key header)
      PEXELS_API_KEY       — required when 'pexels' is listed in config sources
      PIXABAY_API_KEY      — required when 'pixabay' is listed in config sources

    Prints a clear ERROR message and exits if any required key is missing.
    Returns (server_api_key, {source: key, ...}).
    """
    errors: list[str] = []

    server_key = os.environ.get("MEDIA_API_KEY", "").strip()
    if not server_key:
        errors.append("MEDIA_API_KEY         (server auth key — sent as X-Api-Key header)")

    source_keys: dict[str, str] = {}
    for source in config.get("sources", []):
        env_name = f"{source.upper()}_API_KEY"
        val = os.environ.get(env_name, "").strip()
        if not val:
            errors.append(f"{env_name:<22}({source} search API key)")
        source_keys[source] = val

    if errors:
        print("", file=sys.stderr)
        print("ERROR: The following required environment variables are not set:", file=sys.stderr)
        for msg in errors:
            print(f"  {msg}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Set them before starting the server, e.g.:", file=sys.stderr)
        print("  export MEDIA_API_KEY=your-secret-key", file=sys.stderr)
        for source in config.get("sources", []):
            print(f"  export {source.upper()}_API_KEY=your-{source}-key", file=sys.stderr)
        print("", file=sys.stderr)
        sys.exit(1)

    return server_key, source_keys


SERVER_API_KEY, API_KEYS = _load_api_keys()

# ---------------------------------------------------------------------------
# Global state (initialised in lifespan)
# ---------------------------------------------------------------------------

clip_model:   scorer.ClipModel | None = None
store:        bs.BatchStore | None    = None
batch_queue:  asyncio.Queue           = asyncio.Queue()

# Per-item download semaphore — sum of max_concurrent across all configured sources
_sem: asyncio.Semaphore | None = None


def _make_semaphore() -> asyncio.Semaphore:
    rl    = config.get("rate_limits", {})
    total = sum(v.get("max_concurrent", 2) for v in rl.values()) or 4
    return asyncio.Semaphore(total)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global clip_model, store, _sem

    log.info("=== Media Server startup ===")

    # Evict stale caches
    ttl = config.get("cache_ttl_days", 7)
    cleanup.evict_old_batches(PROJECTS_ROOT, ttl)

    # Load CLIP (CPU, may take 10-15 s on first run)
    clip_model = await asyncio.to_thread(scorer.load_clip, config)

    # Batch store
    store = bs.BatchStore(PROJECTS_ROOT)
    store.startup_scan()

    # Per-item semaphore
    _sem = _make_semaphore()

    # Queue worker (single coroutine; batches are processed sequentially)
    worker_task = asyncio.create_task(_queue_worker())

    log.info("=== Media Server ready (transport=%s port=%s) ===",
             config.get("transport", "http"), config.get("port", 8200))

    yield

    worker_task.cancel()
    log.info("=== Media Server shutdown ===")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Media Search & Rating Service", version="1.0.0", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def require_auth(x_api_key: str = Header(..., alias="X-Api-Key")) -> None:
    if x_api_key != SERVER_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class BatchRequest(BaseModel):
    project:    str
    episode_id: str
    manifest:   dict | str          # full manifest JSON (object or serialised string)
    top_n:      int = 5


# ---------------------------------------------------------------------------
# POST /batches — start a new batch
# ---------------------------------------------------------------------------

@app.post("/batches", status_code=202)
async def create_batch(body: BatchRequest, _: None = Depends(require_auth)):
    # Accept manifest as object or JSON string
    manifest = body.manifest
    if isinstance(manifest, str):
        try:
            manifest = json.loads(manifest)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"manifest is not valid JSON: {exc}") from exc

    # Prefer manifest["backgrounds"] as a dict keyed by asset_id.
    # Also accept a list (AssetManifest_draft schema) and convert it.
    backgrounds_raw = manifest.get("backgrounds")
    if not backgrounds_raw:
        raise HTTPException(status_code=400, detail="manifest.backgrounds is empty or missing")

    if isinstance(backgrounds_raw, list):
        backgrounds = {item["asset_id"]: item for item in backgrounds_raw if "asset_id" in item}
    elif isinstance(backgrounds_raw, dict):
        backgrounds = backgrounds_raw
    else:
        raise HTTPException(status_code=400, detail="manifest.backgrounds must be a list or dict")

    if not backgrounds:
        raise HTTPException(status_code=400, detail="manifest.backgrounds contains no valid items")

    batch_id = "b_" + uuid.uuid4().hex[:8]
    store.create(batch_id, body.project, body.episode_id, body.top_n, backgrounds)
    await batch_queue.put(batch_id)

    log.info("Queued batch %s  project=%s  episode=%s  items=%d",
             batch_id, body.project, body.episode_id, len(backgrounds))

    return {
        "batch_id":   batch_id,
        "status":     "queued",
        "item_count": len(backgrounds),
        "poll_url":   f"/batches/{batch_id}",
    }


# ---------------------------------------------------------------------------
# GET /batches — list batches for a project/episode
# ---------------------------------------------------------------------------

@app.get("/batches")
async def list_batches(
    project:    str,
    episode_id: str,
    _:          None = Depends(require_auth),
):
    return store.list_for_episode(project, episode_id)


# ---------------------------------------------------------------------------
# GET /batches/{batch_id} — poll / results
# ---------------------------------------------------------------------------

@app.get("/batches/{batch_id}")
async def get_batch(
    batch_id: str,
    top_n:    Optional[int] = None,
    _:        None = Depends(require_auth),
):
    state = store.get(batch_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Batch not found")

    if state["status"] != "done":
        return {
            "batch_id": state["batch_id"],
            "status":   state["status"],
            "progress": state.get("progress", ""),
            "error":    state.get("error"),
        }

    # Build result with sliced top_n and resolved URLs
    effective_n = top_n if top_n is not None else state.get("top_n", 5)
    items_out   = {}

    for item_id, item in state["items"].items():
        imgs = _add_urls(item.get("images_ranked", []), state, item_id)
        vids = _add_urls(item.get("videos_ranked", []), state, item_id)
        items_out[item_id] = {
            "search_prompt": item.get("search_prompt", ""),
            "status":        item.get("status", ""),
            "error":         item.get("error"),
            "images":        imgs[:effective_n],
            "videos":        vids[:effective_n],
        }

    # Compute recommended sequence from .meta.json sidecars (best-effort)
    recommended_sequence: dict | None = None
    try:
        b_dir = store.batch_dir(state["project"], state["episode_id"], state["batch_id"])
        rec_raw = sequence_ranker.compute_recommended_sequence(
            state["items"], b_dir, effective_n,
        )
        if rec_raw:
            recommended_sequence = {}
            for item_id, cand in rec_raw.items():
                if cand:
                    [cand_with_url] = _add_urls([cand], state, item_id)
                    recommended_sequence[item_id] = cand_with_url
    except Exception as _exc:  # noqa: BLE001
        log.warning("recommended_sequence computation failed: %s", _exc)

    return {
        "batch_id":             state["batch_id"],
        "status":               "done",
        "project":              state["project"],
        "episode_id":           state["episode_id"],
        "top_n":                effective_n,
        "created_at":           state.get("created_at"),
        "completed_at":         state.get("completed_at"),
        "items":                items_out,
        "recommended_sequence": recommended_sequence,
    }


# ---------------------------------------------------------------------------
# GET /files/{path} — serve cached media (no auth required)
# ---------------------------------------------------------------------------

@app.get("/files/{file_path:path}")
async def serve_file(file_path: str):
    # Path-traversal protection: resolved path must remain inside PROJECTS_ROOT
    resolved = (PROJECTS_ROOT / file_path).resolve()
    try:
        resolved.relative_to(PROJECTS_ROOT)
    except ValueError:
        raise HTTPException(status_code=403, detail="Path not allowed")

    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(str(resolved))


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status":     "ok",
        "queue_len":  batch_queue.qsize(),
        "transport":  config.get("transport", "http"),
        "config": {
            "candidates_per_source_image": config.get("candidates_per_source_image", 15),
            "candidates_per_source_video": config.get("candidates_per_source_video", 5),
            "sources":                     config.get("sources", []),
            "cache_ttl_days":              config.get("cache_ttl_days", 7),
        },
    }


# ---------------------------------------------------------------------------
# Queue worker — processes batches sequentially
# ---------------------------------------------------------------------------

async def _queue_worker() -> None:
    while True:
        batch_id = await batch_queue.get()
        try:
            await _run_batch(batch_id)
        except Exception as exc:  # noqa: BLE001
            log.exception("Batch %s failed: %s", batch_id, exc)
            store.update(batch_id, status="failed", error=str(exc))
        finally:
            batch_queue.task_done()


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

async def _run_batch(batch_id: str) -> None:
    state       = store.get(batch_id)
    project     = state["project"]
    episode_id  = state["episode_id"]
    backgrounds = state["items"]
    item_count  = len(backgrounds)
    n_img       = config.get("candidates_per_source_image", 15)
    n_vid       = config.get("candidates_per_source_video", 5)

    store.update(batch_id, status="running", progress="starting")
    log.info("Running batch %s  (%d items)", batch_id, item_count)

    batch_dir = store.batch_dir(project, episode_id, batch_id)

    # Shared counter (mutated only from async gather callbacks — safe in asyncio)
    done: list[int] = [0]

    async def _process_item(item_id: str, item: dict) -> None:
        search_prompt = item.get("search_prompt", "")
        ai_prompt     = item.get("ai_prompt") or search_prompt
        prefer        = item.get("prefer", "both")

        item_dir = batch_dir / item_id
        img_dir  = item_dir / "images"
        vid_dir  = item_dir / "videos"

        store.update(batch_id, progress=f"downloading {item_id} ({done[0]+1}/{item_count})")

        try:
            img_paths: list[Path] = []
            vid_paths: list[Path] = []

            async with _sem:
                if prefer != "video":
                    img_paths = await asyncio.to_thread(
                        downloader.fetch_images,
                        search_prompt, n_img, img_dir, API_KEYS, config, item,
                    )
                if prefer != "image":
                    vid_paths = await asyncio.to_thread(
                        downloader.fetch_videos,
                        search_prompt, n_vid, vid_dir, API_KEYS, config, item,
                    )

            store.update(batch_id, progress=f"scoring {item_id} ({done[0]+1}/{item_count})")

            weights        = config.get("score_weights")
            images_ranked  = await asyncio.to_thread(
                scorer.score_images, clip_model, item, img_paths, weights, config,
            )
            videos_ranked  = await asyncio.to_thread(
                scorer.score_videos, clip_model, item, vid_paths, batch_dir, weights, config,
            )

            # Convert absolute paths → relative-to-batch_dir
            images_ranked = _relativise(images_ranked, batch_dir)
            videos_ranked = _relativise(videos_ranked, batch_dir)

            store.update_item(
                batch_id, item_id,
                status="done",
                images_ranked=images_ranked,
                videos_ranked=videos_ranked,
            )
            log.info("  item %s done  imgs=%d  vids=%d",
                     item_id, len(images_ranked), len(videos_ranked))

        except Exception as exc:  # noqa: BLE001
            log.exception("  item %s failed: %s", item_id, exc)
            store.update_item(batch_id, item_id, status="failed", error=str(exc))

        done[0] += 1

    # Run all items concurrently (bounded by _sem)
    await asyncio.gather(*[_process_item(iid, idata) for iid, idata in backgrounds.items()])

    # Delete temp frames directory created by scorer
    frames_dir = batch_dir / "_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir, ignore_errors=True)
        log.debug("Removed temp frames dir for batch %s", batch_id)

    from datetime import datetime, timezone
    store.update(
        batch_id,
        status="done",
        progress="done",
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    log.info("Batch %s complete", batch_id)


# ---------------------------------------------------------------------------
# URL / path helpers
# ---------------------------------------------------------------------------

def _relativise(ranked: list[dict], batch_dir: Path) -> list[dict]:
    """Convert absolute 'path' fields to paths relative to batch_dir."""
    out = []
    for r in ranked:
        r2 = dict(r)
        try:
            r2["path"] = str(Path(r2["path"]).relative_to(batch_dir))
        except (ValueError, KeyError):
            pass
        out.append(r2)
    return out


def _add_urls(ranked: list[dict], state: dict, item_id: str) -> list[dict]:
    """
    Add a 'url' field to each ranked entry based on transport config.

    http  transport: url = <base_url>/files/<project>/episodes/<episode_id>/assets/media/<batch_id>/<rel_path>
                     (base_url empty → relative /files/… path)
    file  transport: url = file://<file_mount_root>/<project>/…/<rel_path>
                     file_mount_root is the NFS mount path as seen from the *consumer*
                     (VC editor / pipeline runner machine).  The media server saves to
                     projects_root (its own view of the same share); the consumer reads
                     the same bytes via file_mount_root.
    """
    transport       = config.get("transport", "http")
    batch_id        = state["batch_id"]
    project         = state["project"]
    episode_id      = state["episode_id"]
    base_url        = config.get("base_url", "").rstrip("/")
    file_mount_root = config.get("file_mount_root", "").rstrip("/")

    out = []
    for r in ranked:
        r2      = dict(r)
        rel     = r2.get("path", "")
        ep_path = f"{project}/episodes/{episode_id}/assets/media/{batch_id}/{rel}"

        if transport == "file" and file_mount_root:
            # Normalise: strip any existing file:// prefix before re-adding it
            mount = file_mount_root
            if mount.startswith("file://"):
                mount = mount[len("file://"):]
            r2["url"] = f"file://{mount}/{ep_path}"
        else:
            r2["url"] = f"{base_url}/files/{ep_path}" if base_url else f"/files/{ep_path}"

        out.append(r2)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="Media Search & Rating Service")
    ap.add_argument("--host", default=config.get("host", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=config.get("port", 8200))
    args = ap.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, reload=False)
