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

    Worker endpoints (distributed scoring — no auth):
    POST /register                       worker self-registration
    GET  /next_job?worker={name}         poll for next scoring task
    POST /result                         submit scoring result
    GET  /workers                        list registered workers + queue status

Authentication
--------------
    All endpoints except GET /files/ and GET /health require:
        X-Api-Key: <value of MEDIA_API_KEY environment variable>
"""

from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
import os
import shutil
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel

import batch_store as bs
import cleanup
import downloader
import job_queue as jq
import scorer
import sequence_ranker

# ---------------------------------------------------------------------------
# Logging — terminal + rotating file (50 MB, 3 backups)
# ---------------------------------------------------------------------------

_LOG_FMT = "%(asctime)s  %(levelname)-8s  %(name)-12s  %(message)s"
_LOG_DIR  = Path(__file__).parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_root_log = logging.getLogger()
_root_log.setLevel(logging.INFO)

_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(logging.Formatter(_LOG_FMT))
_root_log.addHandler(_stream_handler)

_file_handler = logging.handlers.RotatingFileHandler(
    _LOG_DIR / "media_server.log",
    maxBytes=50 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8",
)
_file_handler.setFormatter(logging.Formatter(_LOG_FMT))
_root_log.addHandler(_file_handler)

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

    Optional variables (log warning if missing but do NOT exit):
      WIKIMEDIA_API_TOKEN  — optional; raises quota from 500/hr to 5,000/hr
      OPENVERSE_CLIENT_ID  — optional but strongly recommended
      OPENVERSE_CLIENT_SECRET — optional but strongly recommended
      EUROPEANA_API_KEY    — required when 'europeana' is in sources
      FREESOUND_API_KEY    — required for SFX tab freesound source

    Prints a clear ERROR message and exits if any REQUIRED key is missing.
    Returns (server_api_key, {source: key, ...}).
    """
    # Sources that are strictly required (missing = server exits)
    REQUIRED_SOURCES = {"pexels", "pixabay"}
    # Sources that are optional (missing = warn + skip, do not exit)
    OPTIONAL_SOURCES = {"archive", "europeana"}
    # Sources that have no {SOURCE}_API_KEY at all — auth is handled separately
    # (openverse uses OPENVERSE_CLIENT_ID/SECRET via OPTIONAL_EXTRA;
    #  wikimedia works anonymously with WIKIMEDIA_API_TOKEN as an optional quota upgrade)
    NO_KEY_SOURCES = {"openverse", "wikimedia"}
    # Special optional keys (not tied to a source name directly)
    OPTIONAL_EXTRA = {
        "WIKIMEDIA_API_TOKEN":     "wikimedia_api_token",
        "OPENVERSE_CLIENT_ID":     "openverse_client_id",
        "OPENVERSE_CLIENT_SECRET": "openverse_client_secret",
        "FREESOUND_API_KEY":       "freesound",
    }

    errors: list[str] = []

    server_key = os.environ.get("MEDIA_API_KEY", "").strip()
    if not server_key:
        errors.append("MEDIA_API_KEY         (server auth key — sent as X-Api-Key header)")

    source_keys: dict[str, str] = {}

    for source in config.get("sources", []):
        if source in NO_KEY_SOURCES:
            # These sources don't use a {SOURCE}_API_KEY — skip the check entirely.
            source_keys[source] = ""
            continue
        env_name = f"{source.upper()}_API_KEY"
        val = os.environ.get(env_name, "").strip()
        if not val:
            if source in REQUIRED_SOURCES:
                errors.append(f"{env_name:<22}({source} search API key)")
            elif source in OPTIONAL_SOURCES:
                log.warning("Optional API key %s not set — %s source will be skipped", env_name, source)
            else:
                # Unknown source — treat as required to surface config errors
                errors.append(f"{env_name:<22}({source} search API key)")
        source_keys[source] = val

    # Load optional extras (not in sources list but needed for certain features)
    for env_name, key_name in OPTIONAL_EXTRA.items():
        val = os.environ.get(env_name, "").strip()
        if not val:
            log.debug("Optional env var %s not set", env_name)
        source_keys[key_name] = val

    if errors:
        print("", file=sys.stderr)
        print("ERROR: The following required environment variables are not set:", file=sys.stderr)
        for msg in errors:
            print(f"  {msg}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Set them before starting the server, e.g.:", file=sys.stderr)
        print("  export MEDIA_API_KEY=your-secret-key", file=sys.stderr)
        for source in config.get("sources", []):
            if source in REQUIRED_SOURCES:
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
job_queue:    jq.JobQueue | None      = None

# Per-item download semaphore — sum of max_concurrent across all configured sources
_sem: asyncio.Semaphore | None = None

# Serialises access to the single-batch job_queue when N batches run concurrently
_jq_sem: asyncio.Semaphore | None = None


def _make_semaphore() -> asyncio.Semaphore:
    rl    = config.get("rate_limits", {})
    total = sum(v.get("max_concurrent", 2) for v in rl.values()) or 4
    return asyncio.Semaphore(total)


def _default_batch_workers() -> int:
    """
    Default number of concurrent batch workers.

    Reserves one CPU for the local scoring worker process so it is never
    starved by batch I/O.  Always returns at least 1.
    """
    cpus = os.cpu_count() or 2
    return max(1, cpus - 1)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global clip_model, store, _sem, _jq_sem, job_queue

    log.info("=== Media Server startup ===")

    # Evict stale caches
    ttl = config.get("cache_ttl_days", 7)
    cleanup.evict_old_batches(PROJECTS_ROOT, ttl)

    # Load CLIP (CPU, may take 10-15 s on first run)
    clip_model = await asyncio.to_thread(scorer.load_clip, config)

    # Batch store
    store = bs.BatchStore(PROJECTS_ROOT)
    store.startup_scan()

    # Per-item download semaphore
    _sem = _make_semaphore()

    # Distributed job-queue serialisation semaphore.
    # The global JobQueue singleton only supports one active scoring session
    # at a time (shared _jobs/_results state).  video_scoring_concurrency=1
    # is the safe default; raise it only after upgrading JobQueue to support
    # batch-keyed concurrent state.
    _video_scoring_concurrency = max(1, int(config.get("video_scoring_concurrency", 1)))
    _jq_sem = asyncio.Semaphore(_video_scoring_concurrency)
    if _video_scoring_concurrency > 1:
        log.info("Distributed video scoring concurrency: %d (ensure JobQueue supports concurrent batches)",
                 _video_scoring_concurrency)

    # Job queue for distributed workers
    workers_cfg = config.get("workers", {})
    if workers_cfg.get("enabled", False):
        server_nfs_root = workers_cfg.get("server_nfs_root", config.get("projects_root", "/data/shared"))
        job_queue = jq.JobQueue(server_nfs_root=server_nfs_root)
        log.info("Distributed workers ENABLED (server_nfs_root=%s)", server_nfs_root)
    else:
        log.info("Distributed workers disabled — using local scoring only")

    # Spawn N concurrent batch workers (default: cpu_count - 1, min 1).
    # Each worker independently pulls from the shared batch_queue, so up to N
    # batches can be processed in parallel without any code change to callers.
    n_workers = config.get("max_concurrent_batches", _default_batch_workers())
    n_workers = max(1, int(n_workers))
    worker_tasks = [asyncio.create_task(_queue_worker()) for _ in range(n_workers)]
    log.info("Started %d concurrent batch worker(s)  (max_concurrent_batches=%d)",
             n_workers, n_workers)

    log.info("=== Media Server ready (transport=%s port=%s) ===",
             config.get("transport", "http"), config.get("port", 8200))

    yield

    for t in worker_tasks:
        t.cancel()
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
    project:                str
    episode_id:             str
    manifest:               dict | str          # full manifest JSON (object or serialised string)
    top_n:                  int = 5
    content_profile:        str = "default"     # scoring profile derived from story_format in UI
    n_img:                  Optional[int] = None    # override candidates_per_source_image from config
    n_vid:                  Optional[int] = None    # override candidates_per_source_video from config
    sources_override:       Optional[list] = None   # override sources list from config
    source_limits_override: Optional[dict] = None   # per-source {candidates_images, candidates_videos}


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
    store.create(batch_id, body.project, body.episode_id, body.top_n, backgrounds,
                 content_profile=body.content_profile,
                 n_img=body.n_img, n_vid=body.n_vid,
                 sources_override=body.sources_override,
                 source_limits_override=body.source_limits_override)
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
        # Include per-item progress so the UI can render a live progress table
        def _count_by_source(ranked_list: list) -> dict:
            """Count ranked entries by source name.

            Prefers the structured 'source' field written by the scorer from the
            .info.json sidecar (r["source"]["source_site"]).  Falls back to path
            parsing only for relative paths (absolute paths like /data/shared/…
            would yield a bogus "shared" token at parts[2]).
            """
            counts: dict = {}
            for entry in ranked_list:
                if not isinstance(entry, dict):
                    continue
                # Primary: scorer attaches r["source"] = info dict with source_site
                src_info = entry.get("source")
                src = (src_info or {}).get("source_site", "") if isinstance(src_info, dict) else ""
                if not src:
                    # Fallback: derive from path — only safe for relative paths
                    # (relative = does NOT start with /, so parts[0] is the item_id)
                    path = entry.get("path", "").replace("\\", "/")
                    if path and not path.startswith("/"):
                        parts = path.split("/")
                        # expected: {item_id}/{images|videos}/{source}/{filename}
                        if len(parts) >= 4:
                            src = parts[2]
                if src:
                    counts[src] = counts.get(src, 0) + 1
            return counts

        items_progress = {}
        for iid, it in state["items"].items():
            item_done = it.get("status") == "done"
            items_progress[iid] = {
                "status":          it.get("status", "pending"),
                "phase":           it.get("phase", ""),
                "imgs_downloaded": it.get("imgs_downloaded", 0),
                "vids_downloaded": it.get("vids_downloaded", 0),
                "imgs_scored":     it.get("imgs_scored", 0),
                "vids_scored":     it.get("vids_scored", 0),
                # include final counts + per-source breakdown if item already done
                "total_images":    len(it.get("images_ranked", [])),
                "total_videos":    len(it.get("videos_ranked", [])),
                "img_sources":     _count_by_source(it.get("images_ranked", [])) if item_done else {},
                "vid_sources":     _count_by_source(it.get("videos_ranked", [])) if item_done else {},
            }
        return {
            "batch_id": state["batch_id"],
            "status":   state["status"],
            "progress": state.get("progress", ""),
            "error":    state.get("error"),
            "items":    items_progress,
        }

    # Build result with resolved URLs (all candidates returned; top_n used for sequence ranker)
    effective_n = top_n if top_n is not None else state.get("top_n", 5)
    items_out   = {}

    for item_id, item in state["items"].items():
        imgs = _add_urls(item.get("images_ranked", []), state, item_id)
        vids = _add_urls(item.get("videos_ranked", []), state, item_id)
        items_out[item_id] = {
            "ai_prompt":        item.get("ai_prompt", ""),
            "search_prompt":    item.get("search_prompt", ""),
            "include_keywords": item.get("include_keywords", []),
            "motion_level":     item.get("motion_level", ""),
            "search_filters":   item.get("search_filters", {}),
            "status":           item.get("status", ""),
            "error":            item.get("error"),
            "total_images":     len(imgs),
            "total_videos":     len(vids),
            "images":           imgs,
            "videos":           vids,
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

    elapsed_sec = None
    if state.get("created_at") and state.get("completed_at"):
        from datetime import datetime
        try:
            t0 = datetime.fromisoformat(state["created_at"])
            t1 = datetime.fromisoformat(state["completed_at"])
            elapsed_sec = round((t1 - t0).total_seconds(), 1)
        except Exception:  # noqa: BLE001
            pass

    return {
        "batch_id":             state["batch_id"],
        "status":               "done",
        "project":              state["project"],
        "episode_id":           state["episode_id"],
        "top_n":                effective_n,
        "created_at":           state.get("created_at"),
        "completed_at":         state.get("completed_at"),
        "elapsed_sec":          elapsed_sec,
        "items":                items_out,
        "recommended_sequence": recommended_sequence,
    }


# ---------------------------------------------------------------------------
# POST /batches/{batch_id}/resume — re-queue an interrupted/failed batch
# ---------------------------------------------------------------------------

class ResumeBatchBody(BaseModel):
    source_limits_override: Optional[dict] = None
    sources_override:       Optional[list] = None

@app.post("/batches/{batch_id}/resume", status_code=202)
async def resume_batch(batch_id: str, body: ResumeBatchBody = ResumeBatchBody(),
                       _: None = Depends(require_auth)):
    state = store.get(batch_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Batch not found")
    if state["status"] == "running":
        return {"batch_id": batch_id, "status": "running",
                "poll_url": f"/batches/{batch_id}", "message": "already running"}
    # Update stored source config if the client supplies fresh values.
    # This lets the UI pass current per-source limits even when resuming an
    # old batch that was created before source_limits_override was required.
    if body.source_limits_override:
        store.patch(batch_id, source_limits_override=body.source_limits_override)
    if body.sources_override:
        store.patch(batch_id, sources_override=body.sources_override)
    reset = store.resume(batch_id)
    await batch_queue.put(batch_id)
    log.info("Resuming batch %s  (%d items to process)", batch_id, reset)
    return {"batch_id": batch_id, "status": "queued", "poll_url": f"/batches/{batch_id}"}


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
    n_workers = config.get("max_concurrent_batches", _default_batch_workers())
    return {
        "status":                 "ok",
        "queue_len":              batch_queue.qsize(),
        "transport":              config.get("transport", "http"),
        "max_concurrent_batches": max(1, int(n_workers)),
        "config": {
            "candidates_per_source_image": config.get("candidates_per_source_image", 15),
            "candidates_per_source_video": config.get("candidates_per_source_video", 5),
            "sources":                     config.get("sources", []),
            "cache_ttl_days":              config.get("cache_ttl_days", 7),
        },
    }


# ---------------------------------------------------------------------------
# POST /sfx_search — search Freesound + Openverse Audio for SFX candidates
# ---------------------------------------------------------------------------

class SfxSearchRequest(BaseModel):
    query:        str
    duration_sec: float = 5.0


@app.post("/sfx_search")
async def sfx_search(body: SfxSearchRequest, _: None = Depends(require_auth)):
    """Search Freesound + Openverse Audio for SFX candidates."""
    try:
        candidates = await asyncio.to_thread(
            downloader.fetch_sfx,
            body.query,
            body.duration_sec,
            API_KEYS,
            config,
        )
        return {"candidates": candidates, "count": len(candidates)}
    except Exception as exc:
        log.exception("sfx_search failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# POST /sfx_save — download + save an SFX candidate to project-local storage
# ---------------------------------------------------------------------------

class SfxSaveRequest(BaseModel):
    project:       str
    episode_id:    str
    item_id:       str
    preview_url:   str
    source_site:   str
    attribution:   dict | None = None


@app.post("/sfx_save")
async def sfx_save(body: SfxSaveRequest, _: None = Depends(require_auth)):
    """Download and save an SFX file to project-local storage."""
    import fnmatch as _fnmatch
    import hashlib as _hashlib
    from urllib.parse import urlparse as _urlparse

    # License gate — reject non-commercial-safe licenses before doing any I/O
    from downloader import is_license_acceptable as _lic_ok
    lic_summary = (body.attribution or {}).get("license_summary", "")
    if not _lic_ok(lic_summary):
        raise HTTPException(
            status_code=400,
            detail=f"License '{lic_summary}' is not acceptable for commercial use "
                   f"(accepted: CC0, Public Domain, CC BY)",
        )

    # SSRF protection: validate hostname against allowlist
    SFX_DOWNLOAD_ALLOWLIST = [
        "cdn.freesound.org", "freesound.org",
        "*.openverse.engineering", "cdn.openverse.engineering",
        "*.wikimedia.org",
    ]
    try:
        hostname = _urlparse(body.preview_url).hostname or ""
        if not any(_fnmatch.fnmatch(hostname, p) for p in SFX_DOWNLOAD_ALLOWLIST):
            raise HTTPException(
                status_code=400,
                detail=f"Download rejected: hostname '{hostname}' not in SFX allowlist",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid URL: {exc}")

    # File size guard
    max_bytes = config.get("sfx_max_download_bytes", 20 * 1024 * 1024)
    try:
        head_r = await asyncio.to_thread(
            lambda: __import__("requests").head(body.preview_url, timeout=10,
                                                headers={"User-Agent": "PipedMediaServer/1.0"})
        )
        cl = head_r.headers.get("Content-Length")
        if cl and int(cl) > max_bytes:
            raise HTTPException(status_code=400,
                                detail=f"File too large: {cl} bytes > {max_bytes}")
    except HTTPException:
        raise
    except Exception:
        pass  # HEAD failed; proceed with streaming guard

    # Determine save path
    ep_dir = (PROJECTS_ROOT / body.project / "episodes" / body.episode_id).resolve()
    sfx_dir = ep_dir / "assets" / "sfx"
    sfx_dir.mkdir(parents=True, exist_ok=True)

    url_hash = _hashlib.md5(body.preview_url.encode()).hexdigest()[:8]
    dest = sfx_dir / f"{body.item_id}.mp3"

    # Download
    def _do_download():
        import requests as _req
        r = _req.get(body.preview_url,
                     headers={"User-Agent": "PipedMediaServer/1.0"},
                     stream=True, timeout=45)
        r.raise_for_status()
        total = 0
        tmp = dest.with_suffix(".part")
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(65536):
                if chunk:
                    total += len(chunk)
                    if total > max_bytes:
                        tmp.unlink(missing_ok=True)
                        raise ValueError(f"Download exceeded {max_bytes} bytes")
                    fh.write(chunk)
        tmp.replace(dest)

    try:
        await asyncio.to_thread(_do_download)
    except Exception as exc:
        log.exception("sfx_save download failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    # Write attribution sidecar (.info.json alongside the audio file)
    import json as _json, datetime as _dt
    attr = body.attribution or {}
    sidecar = dest.with_suffix(".info.json")
    info = {
        "source_site":        body.source_site,
        "source_id":          attr.get("source_id", ""),
        "author":             attr.get("author", ""),
        "license_summary":    lic_summary,
        "license_url":        attr.get("license_url", ""),
        "asset_page_url":     attr.get("asset_page_url", ""),
        "attribution_text":   attr.get("attribution_text", ""),
        "attribution_required": lic_summary not in {"CC0", "Public Domain"},
        "original_format":    "mp3",
        "saved_at":           _dt.datetime.utcnow().isoformat() + "Z",
    }
    sidecar.write_text(_json.dumps(info, indent=2, ensure_ascii=False))

    # Patch AssetManifest_draft.shared.json with structured license fields
    manifest_path = ep_dir / "AssetManifest_draft.shared.json"
    if manifest_path.exists():
        import fcntl as _fcntl
        lock_path = str(manifest_path) + ".lock"
        with open(lock_path, "w") as _lf:
            _fcntl.flock(_lf, _fcntl.LOCK_EX)
            try:
                manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
                for sfx in manifest.get("sfx_items", []):
                    if sfx.get("item_id") == body.item_id:
                        sfx["file_path"]            = str(dest.relative_to(ep_dir))
                        sfx["source"]               = body.source_site
                        sfx["source_id"]            = attr.get("source_id", "")
                        sfx["license_summary"]      = lic_summary
                        sfx["license_type"]         = lic_summary   # key for SPDX_MAP in resolve_assets
                        sfx["license_url"]          = attr.get("license_url", "")
                        sfx["attribution_required"] = lic_summary not in {"CC0", "Public Domain"}
                        sfx["attribution_text"]     = attr.get("attribution_text", "")
                        sfx["author"]               = attr.get("author", "")
                        sfx["asset_page_url"]       = attr.get("asset_page_url", "")
                        break
                manifest_path.write_text(
                    _json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            finally:
                _fcntl.flock(_lf, _fcntl.LOCK_UN)

    transport = config.get("transport", "http")
    file_mount_root = config.get("file_mount_root", "").rstrip("/")
    rel = str(dest.relative_to(PROJECTS_ROOT))

    if transport == "file" and file_mount_root:
        url = f"file://{file_mount_root}/{rel}"
    else:
        base_url = config.get("base_url", "").rstrip("/")
        url = f"{base_url}/files/{rel}" if base_url else f"/files/{rel}"

    return {"local_path": str(dest), "url": url, "status": "ok"}


# ---------------------------------------------------------------------------
# Worker endpoints (distributed scoring)
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    name:     str
    hostname: str
    nfs_root: str = "/mnt/shared"


class ResultRequest(BaseModel):
    job_id: str
    worker: str
    result: dict


@app.post("/register")
async def register_worker(body: RegisterRequest):
    if job_queue is None:
        raise HTTPException(status_code=503, detail="Distributed workers not enabled")
    job_queue.register_worker(body.name, body.hostname, body.nfs_root)
    return {"status": "registered", "name": body.name}


@app.get("/next_job")
async def next_job(worker: str = Query(...)):
    if job_queue is None:
        return Response(status_code=204)  # workers not enabled — no work

    # Reject unregistered workers — they need to POST /register first
    if worker not in job_queue._workers:
        return Response(status_code=410)  # Gone — worker must re-register

    # Update last-seen even if no job available
    import time
    job_queue._workers[worker].last_seen = time.monotonic()

    task = job_queue.next_job(worker)
    if task is not None:
        return task  # 200 OK with job payload

    # Queue empty — is a batch active?
    if job_queue.batch_id is not None and not job_queue.all_done():
        # Batch is active but queue temporarily empty (in-flight or requeue pending)
        return Response(status_code=202)

    # No active batch
    return Response(status_code=204)


@app.post("/result")
async def submit_result(body: ResultRequest):
    if job_queue is None:
        raise HTTPException(status_code=503, detail="Distributed workers not enabled")
    accepted = job_queue.submit_result(body.job_id, body.result)
    return {"accepted": accepted, "job_id": body.job_id}


@app.get("/workers")
async def list_workers():
    if job_queue is None:
        return {"workers": [], "enabled": False}
    return {
        "workers": job_queue.get_workers(),
        "enabled": True,
        "batch_id": job_queue.batch_id,
        "queue": {
            "pending":   job_queue.pending_count,
            "in_flight": job_queue.in_flight_count,
            "completed": job_queue.completed_count,
            "total":     job_queue.total_count,
        },
    }


# ---------------------------------------------------------------------------
# Queue worker — N instances run concurrently; each pulls one batch at a time
# ---------------------------------------------------------------------------

async def _queue_worker() -> None:
    """
    Pull batches from batch_queue and run them.

    N copies of this coroutine are started at startup (max_concurrent_batches),
    giving true parallel batch execution bounded by the CPU budget.
    Each instance is independent — no coordination needed between workers
    because asyncio.Queue is inherently safe for multiple concurrent consumers.
    """
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
# Distributed video scoring via job queue
# ---------------------------------------------------------------------------

async def _score_videos_distributed(
    batch_id:  str,
    item:      dict,
    vid_paths: list[Path],
    batch_dir: Path,
    cfg:       dict,
    infos:     dict | None = None,
) -> list[dict]:
    """
    Enqueue video scoring tasks into the job queue and wait for workers
    to complete them.  Returns scored results list (same shape as
    scorer.score_videos).
    """
    # Build tasks for the queue
    tasks = []
    for vp in vid_paths:
        frames_dir = batch_dir / "_frames" / vp.stem
        tasks.append({
            "video_path": str(vp),
            "frames_dir": str(frames_dir),
            "item":       item,
            "config":     {
                k: cfg[k] for k in (
                    "score_weights", "scoring_profiles", "content_profile",
                    "phash_dedup_threshold", "diversity_phash_threshold",
                ) if k in cfg
            },
        })

    workers_cfg    = cfg.get("workers", {})
    timeout_seconds = workers_cfg.get("timeout_seconds", 120)

    # Serialise access to the single-batch JobQueue.  When N batches run
    # concurrently each one waits its turn here; remote scoring workers never
    # see this — they continue to poll /next_job independently.
    async with _jq_sem:
        job_queue.enqueue(batch_id, tasks)
        job_queue.start_reaper(timeout_seconds=timeout_seconds)

        try:
            log.info("Waiting for %d video scoring jobs (batch %s)…",
                     len(tasks), batch_id)
            await job_queue.wait_until_done()
        finally:
            job_queue.stop_reaper()

        # Collect results while still holding the lock (before next batch
        # can call enqueue and reset the queue state)
        raw_results = job_queue.get_results()

    # Sort by score descending
    raw_results.sort(key=lambda r: r.get("score", 0.0), reverse=True)

    # Attach source metadata from infos dict.
    # Workers receive remapped paths (server_nfs_root → worker nfs_root), so
    # r["path"] may differ from the infos keys (which use server_nfs_root).
    # Strategy:
    #   1. Direct lookup (works when worker and server share the same mount path)
    #   2. Reverse-remap each registered worker's nfs_root → server_nfs_root
    #   3. Disk fallback: read sidecar using the server-side path
    _workers_nfs = [w.nfs_root for w in job_queue._workers.values()] if job_queue else []
    for r in raw_results:
        worker_path = r.get("path", "")
        source = infos.get(worker_path) if infos else None

        if source is None and infos:
            # Try reverse-remapping worker path back to server path
            for wnfs in _workers_nfs:
                if worker_path.startswith(wnfs):
                    server_path = job_queue.server_nfs_root + worker_path[len(wnfs):]
                    source = infos.get(server_path)
                    if source is not None:
                        break

        if source is None:
            # Disk fallback: resolve to server-side path and read sidecar
            server_path = worker_path
            if not worker_path.startswith(job_queue.server_nfs_root if job_queue else ""):
                for wnfs in _workers_nfs:
                    if worker_path.startswith(wnfs):
                        server_path = job_queue.server_nfs_root + worker_path[len(wnfs):]
                        break
            sidecar = Path(server_path + ".info.json")
            if sidecar.exists():
                try:
                    source = json.loads(sidecar.read_text())
                except Exception:
                    pass

        if source is not None:
            r["source"] = source

    log.info("Distributed scoring complete: %d results for batch %s",
             len(raw_results), batch_id)
    return raw_results


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

async def _run_batch(batch_id: str) -> None:
    state       = store.get(batch_id)
    project     = state["project"]
    episode_id  = state["episode_id"]
    backgrounds = state["items"]
    item_count  = len(backgrounds)
    n_img       = state.get("n_img") or config.get("candidates_per_source_image", 15)
    n_vid       = state.get("n_vid") or config.get("candidates_per_source_video", 5)

    # Apply per-batch content_profile override (section 30)
    # Use a shallow copy so the global config is never mutated across concurrent batches
    cfg = dict(config)
    cfg["content_profile"] = store.get_content_profile(batch_id)
    # Apply per-batch source overrides from UI settings
    if state.get("sources_override"):
        cfg["sources"] = state["sources_override"]
    # source_limits_override is required — the server has no defaults of its own.
    # Every enabled source must supply candidates_images and candidates_videos.
    source_limits_override = state.get("source_limits_override") or {}
    if not source_limits_override:
        raise HTTPException(status_code=400,
            detail="source_limits_override is required: client must supply per-source candidate counts")
    active_sources = cfg.get("sources", [])
    missing = [
        src for src in active_sources
        if "candidates_images" not in source_limits_override.get(src, {})
        or "candidates_videos" not in source_limits_override.get(src, {})
    ]
    if missing:
        raise HTTPException(status_code=400,
            detail=f"source_limits_override missing candidates_images/candidates_videos for: {missing}")
    cfg["source_limits"] = {
        src: {k: v for k, v in lims.items()
              if k in ("candidates_images", "candidates_videos")}
        for src, lims in source_limits_override.items()
    }

    store.update(batch_id, status="running", progress="starting")
    log.info("Running batch %s  (%d items)", batch_id, item_count)

    batch_dir = store.batch_dir(project, episode_id, batch_id)

    # Check if distributed workers are available
    workers_cfg     = config.get("workers", {})
    _use_distributed = False
    if job_queue is not None and workers_cfg.get("enabled", False):
        grace = workers_cfg.get("fallback_grace_seconds", 10)
        if job_queue.worker_count > 0:
            _use_distributed = True
            log.info("Batch %s: using %d distributed workers",
                     batch_id, job_queue.worker_count)
        elif grace > 0:
            log.info("Batch %s: waiting up to %ds for workers to register…",
                     batch_id, grace)
            deadline = asyncio.get_event_loop().time() + grace
            while asyncio.get_event_loop().time() < deadline:
                if job_queue.worker_count > 0:
                    _use_distributed = True
                    log.info("Batch %s: %d worker(s) registered — using distributed scoring",
                             batch_id, job_queue.worker_count)
                    break
                await asyncio.sleep(1)
            if not _use_distributed:
                log.warning("Batch %s: no workers registered after %ds — falling back to local scoring",
                            batch_id, grace)

    # Shared counter (mutated only from async gather callbacks — safe in asyncio)
    done: list[int] = [0]

    async def _process_item(item_id: str, item: dict) -> None:
        # Skip items already completed (allows resume after server restart)
        if item.get("status") == "done":
            log.info("[%s]  already done — skipping", item_id)
            done[0] += 1
            return

        search_prompt = item.get("search_prompt", "")
        ai_prompt     = item.get("ai_prompt") or search_prompt

        # C3 / M6: map search_filters → prefer + pixabay source_filters
        # "photo" → prefer images but still fetch videos (manifest often over-specifies photo)
        # Only hard-skip the other type when media_type is explicitly "video" or "image"
        sf     = item.get("search_filters") or {}
        mt     = sf.get("media_type", "mixed")
        prefer = {"image": "image", "video": "video"}.get(mt, "both")
        # "photo" maps to "both" (prefer images in scoring, but don't skip video fetch)
        # Use explicit "image" in manifest to hard-skip videos

        # Inject Pixabay dimension constraints into source_filters
        src_filters = dict(item.get("source_filters") or {})
        pbay = dict(src_filters.get("pixabay") or {})
        if sf.get("min_width")  and "min_width"  not in pbay:
            pbay["min_width"]  = sf["min_width"]
        if sf.get("min_height") and "min_height" not in pbay:
            pbay["min_height"] = sf["min_height"]
        if pbay:
            src_filters["pixabay"] = pbay
        # Rebuild item with resolved prefer + source_filters (non-mutating)
        item = {**item, "prefer": prefer, "source_filters": src_filters}

        item_dir = batch_dir / item_id
        img_dir  = item_dir / "images"
        vid_dir  = item_dir / "videos"

        store.update(batch_id, progress=f"downloading {item_id} ({done[0]+1}/{item_count})")
        store.update_item_progress(batch_id, item_id, phase="downloading")

        try:
            img_path_infos: list[tuple] = []
            vid_path_infos: list[tuple] = []

            log.info("[%s]  download start  prompt=%r  n_img=%d  n_vid=%d  prefer=%s",
                     item_id, search_prompt[:80], n_img, n_vid, prefer)
            t_dl = time.perf_counter()

            async with _sem:
                # Fetch images and videos in parallel — they are fully independent.
                async def _noop() -> list:
                    return []

                img_path_infos, vid_path_infos = await asyncio.gather(
                    asyncio.to_thread(
                        downloader.fetch_images,
                        search_prompt, n_img, img_dir, API_KEYS, cfg, item,
                    ) if prefer != "video" else _noop(),
                    asyncio.to_thread(
                        downloader.fetch_videos,
                        search_prompt, n_vid, vid_dir, API_KEYS, cfg, item,
                    ) if prefer != "image" else _noop(),
                )

            log.info("[%s]  download done   imgs=%d  vids=%d  elapsed=%.1fs",
                     item_id, len(img_path_infos), len(vid_path_infos),
                     time.perf_counter() - t_dl)

            img_paths = [p for p, _ in img_path_infos]
            vid_paths = [p for p, _ in vid_path_infos]
            img_infos = {str(p): info for p, info in img_path_infos}
            vid_infos = {str(p): info for p, info in vid_path_infos}

            store.update(batch_id, progress=f"scoring {item_id} ({done[0]+1}/{item_count})")
            store.update_item_progress(
                batch_id, item_id, phase="scoring",
                imgs_downloaded=len(img_path_infos),
                vids_downloaded=len(vid_path_infos),
            )

            log.info("[%s]  scoring start   imgs=%d  vids=%d  mode=%s",
                     item_id, len(img_paths), len(vid_paths),
                     "distributed" if (_use_distributed and vid_paths) else "local")
            t_sc = time.perf_counter()

            weights        = cfg.get("score_weights")
            images_ranked  = await asyncio.to_thread(
                scorer.score_images, clip_model, item, img_paths, weights, cfg,
                img_infos,
            )

            # Video scoring: distributed (job queue) or local fallback
            if _use_distributed and vid_paths:
                videos_ranked = await _score_videos_distributed(
                    batch_id, item, vid_paths, batch_dir, cfg,
                    vid_infos,
                )
            else:
                videos_ranked = await asyncio.to_thread(
                    scorer.score_videos, clip_model, item, vid_paths, batch_dir, weights, cfg,
                    vid_infos,
                )

            log.info("[%s]  scoring done    imgs_ranked=%d  vids_ranked=%d  elapsed=%.1fs",
                     item_id, len(images_ranked), len(videos_ranked),
                     time.perf_counter() - t_sc)

            # Convert absolute paths → relative-to-batch_dir
            images_ranked = _relativise(images_ranked, batch_dir)
            videos_ranked = _relativise(videos_ranked, batch_dir)

            store.update_item_progress(
                batch_id, item_id, phase="done",
                imgs_scored=len(images_ranked),
                vids_scored=len(videos_ranked),
            )
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

        # Guard: if _relativise() failed and path is still absolute,
        # extract the portion after the batch_id directory to avoid
        # building a doubled path like /mount/…/b_xxx//mount/…/b_xxx/…
        if rel.startswith("/"):
            marker = f"/assets/media/{batch_id}/"
            idx    = rel.find(marker)
            if idx != -1:
                rel = rel[idx + len(marker):]

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
