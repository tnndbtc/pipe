"""
server.py — Media Search & Rating Service
==========================================

FastAPI server that, given an asset manifest containing a `backgrounds` map,
searches Pexels + Pixabay (and other sources) for every background item,
scores each candidate with a keyword-match score, and returns top-N ranked
candidates per item.

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
    POST /batches                              start a new batch (authenticated)
    GET  /batches                              list batches for a project/episode (auth)
    GET  /batches/{batch_id}                   poll status / get results (auth)
    POST /batches/{batch_id}/prune             delete images not matching a filter (auth)
    POST /batches/{batch_id}/items/{item_id}/append      append search results (auth)
    GET  /batches/{batch_id}/items/{item_id}/append/{tmp} poll append status (auth)
    POST /ai_ask                               natural-language media operation (auth)
    GET  /files/{path:path}                    serve cached media files (no auth)
    GET  /health                               server status (no auth)

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
import importlib.util
import json
import logging
import logging.handlers
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

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

if "candidates_per_source_image" not in config:
    raise ValueError("candidates_per_source_image missing from config.json")
if "candidates_per_source_video" not in config:
    raise ValueError("candidates_per_source_video missing from config.json")

PROJECTS_ROOT = (Path(__file__).parent / config["projects_root"]).resolve()


# ---------------------------------------------------------------------------
# Load media_ask.build_prompt at module startup (fails fast if file is missing)
# ---------------------------------------------------------------------------

_ask_spec = importlib.util.spec_from_file_location(
    "media_ask",
    Path(__file__).parent.parent / "scripts" / "media_ask.py",
)
_ask_module = importlib.util.module_from_spec(_ask_spec)
_ask_spec.loader.exec_module(_ask_module)
build_prompt = _ask_module.build_prompt  # type: ignore[attr-defined]


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

    # CLIP scoring retired — keyword-match scoring is used instead.
    # clip_model stays None; scorer.load_clip is not called.
    log.info("Keyword-match scoring active — CLIP model not loaded")

    # Batch store
    store = bs.BatchStore(PROJECTS_ROOT)
    store.startup_scan()

    # Load SSRF host lists
    downloader._load_host_lists({"projects_root": str(PROJECTS_ROOT)})

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
    # Also accept a list (AssetManifest schema) and convert it.
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

    # A13: Validate manifest — reject any item containing raw external download URLs
    for item_id, item_data in backgrounds.items():
        if isinstance(item_data, dict):
            for key in ("download_url", "file_url", "url"):
                val = item_data.get(key, "")
                if val and isinstance(val, str) and val.startswith("http"):
                    raise HTTPException(
                        status_code=400,
                        detail=f"manifest.backgrounds[{item_id}] contains raw download URL in '{key}' — "
                               f"only search queries and metadata are accepted",
                    )

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
# POST /hosts/allow — approve a hostname for SSRF allowlist
# ---------------------------------------------------------------------------

class HostAllowRequest(BaseModel):
    hostname: str
    note: str = ""
    source: str = ""


@app.post("/hosts/allow")
async def allow_host(body: HostAllowRequest, _: None = Depends(require_auth)):
    """Add a hostname to the SSRF dynamic allowlist."""
    from datetime import datetime, timezone
    hostname = body.hostname.strip().lower()
    if not hostname:
        raise HTTPException(status_code=400, detail="hostname is required")
    # No wildcards allowed
    if "*" in hostname or "?" in hostname:
        raise HTTPException(status_code=400, detail="wildcards not allowed in hostname")

    meta = {
        "added_ts": datetime.now(timezone.utc).isoformat(),
        "added_by": "user",
        "source": body.source,
        "note": body.note,
        "expires_ts": None,
    }
    downloader._add_allowed_host(hostname, meta, {"projects_root": str(PROJECTS_ROOT)})

    # Audit trail
    _append_audit("allow", hostname, meta)

    return {"status": "allowed", "hostname": hostname}


# ---------------------------------------------------------------------------
# POST /hosts/reject — reject a hostname
# ---------------------------------------------------------------------------

class HostRejectRequest(BaseModel):
    hostname: str


@app.post("/hosts/reject")
async def reject_host(body: HostRejectRequest, _: None = Depends(require_auth)):
    """Add a hostname to the SSRF rejected list."""
    hostname = body.hostname.strip().lower()
    if not hostname:
        raise HTTPException(status_code=400, detail="hostname is required")

    downloader._add_rejected_host(hostname, {"projects_root": str(PROJECTS_ROOT)})

    # Audit trail
    _append_audit("reject", hostname, {})

    return {"status": "rejected", "hostname": hostname}


# ---------------------------------------------------------------------------
# GET /hosts — list current allowed and rejected hosts
# ---------------------------------------------------------------------------

@app.get("/hosts")
async def list_hosts(_: None = Depends(require_auth)):
    """Return current SSRF allowed and rejected host lists."""
    return {
        "allowed": downloader._ssrf_allowed_hosts,
        "rejected": downloader._ssrf_rejected_hosts,
    }


def _append_audit(action: str, hostname: str, meta: dict) -> None:
    """Append an entry to the SSRF audit log."""
    from datetime import datetime, timezone
    audit_path = PROJECTS_ROOT / "_ssrf_audit.jsonl"
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "hostname": hostname,
        "added_by": meta.get("added_by", "user"),
        "source": meta.get("source", ""),
        "note": meta.get("note", ""),
    }
    try:
        with audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        log.warning("Could not append to audit log %s: %s", audit_path, exc)


# ---------------------------------------------------------------------------
# POST /batches/{batch_id}/items/{item_id}/resume — resume downloads for pending hosts (MH-2)
# ---------------------------------------------------------------------------

@app.post("/batches/{batch_id}/items/{item_id}/resume", status_code=202)
async def resume_item_downloads(batch_id: str, item_id: str, _: None = Depends(require_auth)):
    """Re-run download phase for a single item using stored pending candidates.

    Only downloads candidates whose hostnames have been approved since the
    initial batch run. Appends results to existing item results.
    """
    state = store.get(batch_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Batch not found")

    pending_all = state.get("pending_candidates") or {}
    item_candidates = pending_all.get(item_id)
    if not item_candidates:
        raise HTTPException(status_code=404, detail=f"No pending candidates for item {item_id}")

    # Filter: only download candidates whose hostnames are now approved
    approved = []
    still_pending = []
    for cand in item_candidates:
        hostname = cand.get("hostname", "")
        host_status = downloader._is_host_allowed(hostname)
        if host_status in ("static", "dynamic"):
            approved.append(cand)
        elif host_status == "rejected":
            pass  # skip rejected
        else:
            still_pending.append(cand)

    if not approved:
        return {
            "batch_id": batch_id,
            "item_id": item_id,
            "status": "no_approved_candidates",
            "still_pending": len(still_pending),
        }

    # Download approved candidates in a background thread
    cfg = dict(config)
    cfg["content_profile"] = store.get_content_profile(batch_id)
    if state.get("sources_override"):
        cfg["sources"] = state["sources_override"]
    if state.get("source_limits_override"):
        cfg["source_limits"] = {
            src: {k: v for k, v in lims.items()
                  if k in ("candidates_images", "candidates_videos")}
            for src, lims in state["source_limits_override"].items()
        }

    batch_dir = store.batch_dir(state["project"], state["episode_id"], batch_id)

    async def _do_resume():
        from downloader import _download_file, _write_info_sidecar
        downloaded = 0
        for cand in approved:
            url = cand.get("url", "")
            info = cand.get("info", {})
            source = info.get("source_site", "")

            # Determine destination
            source_id = info.get("source_id", cand.get("source_id", ""))
            uid = f"{source}_img_{source_id}"
            ext = Path(url.split("?")[0]).suffix or ".jpg"
            if ext.lower() not in {".jpg", ".jpeg", ".png", ".gif", ".mp4", ".webm"}:
                ext = ".jpg"

            item_dir = batch_dir / item_id / "images" / source
            item_dir.mkdir(parents=True, exist_ok=True)
            dest = item_dir / f"{uid}{ext}"

            try:
                result = await asyncio.to_thread(
                    _download_file, url, dest,
                    headers={}, cfg=cfg, source=source,
                    media_type="image",
                )
                if result is None:  # success
                    _write_info_sidecar(dest, info)
                    downloaded += 1
            except Exception as exc:
                log.warning("Resume download failed for %s: %s", url[:80], exc)

        # Update pending_candidates: remove processed, keep still_pending
        if still_pending:
            pending_all[item_id] = still_pending
        else:
            pending_all.pop(item_id, None)

        # Clean up empty pending_candidates
        if not pending_all:
            store.update(batch_id, pending_candidates=None)
        else:
            store.update(batch_id, pending_candidates=pending_all)

        return downloaded

    downloaded = await _do_resume()

    return {
        "batch_id": batch_id,
        "item_id": item_id,
        "status": "resumed",
        "downloaded": downloaded,
        "still_pending": len(still_pending),
    }


# ---------------------------------------------------------------------------
# POST /batches/{batch_id}/items/{item_id}/select — select an asset (MH-3, MH-4)
# ---------------------------------------------------------------------------

class AssetSelectRequest(BaseModel):
    asset_path: str          # relative path within batch dir
    project: str = ""        # override (usually from batch state)
    episode_id: str = ""     # override (usually from batch state)


@app.post("/batches/{batch_id}/items/{item_id}/select")
async def select_asset(
    batch_id: str, item_id: str,
    body: AssetSelectRequest,
    _: None = Depends(require_auth),
):
    """Select an asset for use in a project. Creates .license.json and
    appends CC BY attribution to youtube_description.txt if needed.
    """
    from datetime import datetime, timezone

    state = store.get(batch_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Batch not found")

    project = body.project or state["project"]
    episode_id = body.episode_id or state["episode_id"]

    # Resolve asset path and find its .info.json sidecar
    batch_dir = store.batch_dir(project, episode_id, batch_id)
    asset_path = (batch_dir / body.asset_path).resolve()

    # Security: must be inside batch_dir
    try:
        asset_path.relative_to(batch_dir)
    except ValueError:
        raise HTTPException(status_code=403, detail="Asset path outside batch directory")

    if not asset_path.is_file():
        raise HTTPException(status_code=404, detail="Asset file not found")

    # Read info sidecar
    sidecar_path = Path(str(asset_path) + ".info.json")
    if not sidecar_path.exists():
        raise HTTPException(status_code=404, detail="Asset .info.json sidecar not found")

    info = json.loads(sidecar_path.read_text(encoding="utf-8"))

    # MH-3: Create .license.json alongside the asset
    now_ts = datetime.now(timezone.utc).isoformat()
    license_data = {
        "asset_file":          asset_path.name,
        "source_site":         info.get("source_site", ""),
        "asset_page_url":      info.get("asset_page_url", ""),
        "file_url":            info.get("file_url", ""),
        "title":               info.get("title", ""),
        "photographer":        info.get("photographer", info.get("author", "")),
        "license_summary":     info.get("license_summary", ""),
        "license_url":         info.get("license_url", ""),
        "attribution_required": info.get("attribution_required", False),
        "attribution_text":    info.get("attribution_text", ""),
        "selected_ts":         now_ts,
        "project":             project,
        "episode":             episode_id,
        "item_id":             item_id,
    }

    license_path = Path(str(asset_path) + ".license.json")
    license_path.write_text(
        json.dumps(license_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Created license file: %s", license_path)

    # MH-4: Append CC BY attribution to youtube_description.txt
    attribution_appended = False
    lic = info.get("license_summary", "")
    if lic.startswith("CC BY"):
        ep_dir = PROJECTS_ROOT / project / "episodes" / episode_id
        ep_dir.mkdir(parents=True, exist_ok=True)
        desc_path = ep_dir / "youtube_description.txt"

        asset_page = info.get("asset_page_url", "")
        title = info.get("title", "")
        photographer = info.get("photographer", info.get("author", ""))
        license_url = info.get("license_url", "")

        attribution_block = (
            f'---\n'
            f'"{title}" by {photographer} is licensed under {lic}.\n'
            f'Source: {asset_page}\n'
            f'License: {license_url}\n'
        )

        # Deduplicate by asset_page_url
        existing = ""
        if desc_path.exists():
            existing = desc_path.read_text(encoding="utf-8")

        if asset_page and asset_page not in existing:
            with desc_path.open("a", encoding="utf-8") as f:
                if existing and not existing.endswith("\n"):
                    f.write("\n")
                f.write(attribution_block + "\n")
            attribution_appended = True
            log.info("Appended CC BY attribution for %s to %s", title, desc_path)

    return {
        "status": "selected",
        "license_path": str(license_path),
        "attribution_appended": attribution_appended,
    }


# ---------------------------------------------------------------------------
# POST /batches/{batch_id}/prune — delete media not matching a filter
# ---------------------------------------------------------------------------

class FilterSpec(BaseModel):
    min_score:       float | None     = None
    max_score:       float | None     = None
    keep_sources:    list[str] | None = None
    exclude_sources: list[str] | None = None
    keep_top_n:      int | None       = None
    title_contains:  str | None       = None  # case-insensitive substring match on title/desc/tags
    media_types:     list[str] | None = None  # ["images"], ["videos"], or None = both


class PruneRequest(BaseModel):
    item_ids:    list[str] | None = None   # if None, prune all items in batch
    filter_spec: FilterSpec


class AiAskRequest(BaseModel):
    prompt:     str
    project:    str
    episode_id: str
    batch_id:   Optional[str] = None      # hints Claude which batch to operate on


class AppendRequest(BaseModel):
    ai_prompt:        str                  # search query for this item
    n_img:            Optional[int] = None # number of new image candidates to fetch
    n_vid:            Optional[int] = None # number of new video candidates to fetch
    sources_override: Optional[list] = None


def _keyword_score(query: str, info: dict) -> float:
    """
    Keyword-match score: fraction of search-query terms found in entry metadata.
    Checks source.title, source.tags, and source.asset_page_url slug (case-insensitive).
    Returns 0.0–1.0.  Returns 1.0 when query is empty or info has no text fields.
    """
    if not query:
        return 1.0
    terms = [t for t in query.lower().split() if t]
    if not terms:
        return 1.0

    parts: list[str] = []

    title = (info.get("title") or "").lower()
    if title:
        parts.append(title)

    tags = info.get("tags") or []
    if tags:
        parts.append(" ".join(str(t).lower() for t in tags))

    url = info.get("asset_page_url") or ""
    if url:
        slug = urlparse(url).path
        slug = re.sub(r"-\d+/?$", "", slug)        # strip trailing numeric ID
        slug = re.sub(r"[/_\-]", " ", slug).lower()
        parts.append(slug)

    if not parts:
        return 1.0   # no metadata to match against — don't penalise

    searchable = " ".join(parts)
    matched = sum(1 for t in terms if t in searchable)
    return round(matched / len(terms), 4)


def _apply_filter(entries: list[dict], fs: FilterSpec) -> list[dict]:
    """
    Apply a FilterSpec to a ranked entry list (images_ranked or videos_ranked).
    Returns the kept entries. All active fields are ANDed together.
    """
    kept = list(entries)

    # Score range
    if fs.min_score is not None:
        kept = [e for e in kept if e.get("score", 0.0) >= fs.min_score]
    if fs.max_score is not None:
        kept = [e for e in kept if e.get("score", 0.0) <= fs.max_score]

    # Source whitelist / blacklist
    # source_site == "" means legacy entry with no source info — never drop by source filter
    if fs.keep_sources is not None:
        kept = [
            e for e in kept
            if (e.get("source") or {}).get("source_site", "") in fs.keep_sources
            or (e.get("source") or {}).get("source_site", "") == ""
        ]
    if fs.exclude_sources is not None:
        kept = [
            e for e in kept
            if (e.get("source") or {}).get("source_site", "") not in fs.exclude_sources
        ]

    # Keyword filter — case-insensitive substring match against title, description, tags
    # Unknown/missing source dict → no fields match → entry is dropped (correct behaviour)
    if fs.title_contains is not None:
        needle = fs.title_contains.lower()

        def _matches(e: dict) -> bool:
            src      = (e.get("source") or {})
            title    = (src.get("title") or "").lower()
            desc     = (src.get("description") or "").lower()
            tags     = src.get("tags") or []
            tags_str = " ".join(tags).lower() if isinstance(tags, list) else str(tags).lower()
            # Also check URL slug — reliable signal for Pexels (no tags) and Pixabay
            url      = src.get("asset_page_url") or ""
            slug     = ""
            if url:
                _slug = urlparse(url).path
                _slug = re.sub(r"-\d+/?$", "", _slug)
                slug  = re.sub(r"[/_\-]", " ", _slug).lower()
            return needle in title or needle in desc or needle in tags_str or needle in slug

        kept = [e for e in kept if _matches(e)]

    # keep_top_n — always re-sort before slicing (never trust pre-sorted order)
    if fs.keep_top_n is not None:
        kept = sorted(kept, key=lambda e: e.get("score", 0.0), reverse=True)
        kept = kept[:fs.keep_top_n]

    return kept


def _delete_entries(entries: list[dict], batch_dir: Path) -> None:
    """Unlink each entry's file and its .info.json sidecar from disk."""
    for entry in entries:
        rel = entry.get("path")
        if not rel:
            continue
        abs_path = (batch_dir / rel).resolve()
        try:
            abs_path.relative_to(batch_dir)
        except ValueError:
            log.warning("prune_batch: skipping path outside batch_dir: %s", abs_path)
            continue
        abs_path.unlink(missing_ok=True)
        Path(str(abs_path) + ".info.json").unlink(missing_ok=True)


@app.post("/batches/{batch_id}/prune")
async def prune_batch(
    batch_id: str,
    body: PruneRequest,
    _: None = Depends(require_auth),
):
    """
    Permanently delete image and/or video files from a batch that do not match
    filter_spec.

    Applies FilterSpec fields (min_score, max_score, keep_sources, exclude_sources,
    title_contains, keep_top_n, media_types) as an AND-chain.  media_types scopes
    the filter to images only, videos only, or both (default).  State is updated
    before files are deleted so batch_state is always consistent.

    Returns: { batch_id, items: { item_id: { images: {deleted, kept},
                                              videos: {deleted, kept} } } }
    """
    # ── Setup ─────────────────────────────────────────────────────────────────
    state = store.get(batch_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Batch not found")

    if state["status"] == "running":
        raise HTTPException(status_code=409,
                            detail="Cannot prune a running batch — wait for it to finish")

    batch_dir  = store.batch_dir(state["project"], state["episode_id"], batch_id)

    target_ids = body.item_ids if body.item_ids is not None else list(state["items"].keys())
    unknown    = [iid for iid in target_ids if iid not in state["items"]]
    if unknown:
        raise HTTPException(status_code=422,
                            detail=f"Unknown item_id(s): {', '.join(unknown)}")

    # ── Step 0: Validate FilterSpec before any deletion ───────────────────────
    fs = body.filter_spec

    if fs.keep_sources and fs.exclude_sources:
        raise HTTPException(status_code=422,
                            detail="keep_sources and exclude_sources are mutually exclusive")
    if fs.min_score is not None and not (0.0 <= fs.min_score <= 1.0):
        raise HTTPException(status_code=422,
                            detail=f"min_score must be in [0.0, 1.0], got {fs.min_score}")
    if fs.max_score is not None and not (0.0 <= fs.max_score <= 1.0):
        raise HTTPException(status_code=422,
                            detail=f"max_score must be in [0.0, 1.0], got {fs.max_score}")
    if fs.min_score is not None and fs.max_score is not None and fs.min_score > fs.max_score:
        raise HTTPException(status_code=422,
                            detail=f"min_score ({fs.min_score}) must be <= max_score ({fs.max_score})")
    if fs.keep_top_n is not None and fs.keep_top_n < 1:
        raise HTTPException(status_code=422,
                            detail=f"keep_top_n must be >= 1, got {fs.keep_top_n}")
    if fs.media_types is not None:
        bad = [m for m in fs.media_types if m not in ("images", "videos")]
        if bad:
            raise HTTPException(status_code=422,
                                detail=f"media_types values must be 'images' or 'videos', got: {bad}")

    # ── Media-type gates ──────────────────────────────────────────────────────
    do_images = fs.media_types is None or "images" in fs.media_types
    do_videos = fs.media_types is None or "videos" in fs.media_types

    # ── Per-item filter loop ───────────────────────────────────────────────────
    items_summary: dict[str, dict] = {}

    for item_id in target_ids:
        item_state  = state["items"][item_id]

        orig_images = list(item_state.get("images_ranked", []))
        orig_videos = list(item_state.get("videos_ranked", []))

        kept_images = _apply_filter(orig_images, fs) if do_images else orig_images
        kept_videos = _apply_filter(orig_videos, fs) if do_videos else orig_videos

        kept_img_paths = {e.get("path") for e in kept_images}
        kept_vid_paths = {e.get("path") for e in kept_videos}
        del_images = [e for e in orig_images if e.get("path") not in kept_img_paths]
        del_videos = [e for e in orig_videos if e.get("path") not in kept_vid_paths]

        # Persist updated ranked lists BEFORE touching disk (state always consistent)
        store.update_item(
            batch_id, item_id,
            status=item_state["status"],   # preserve existing per-item status unchanged
            images_ranked=kept_images,
            videos_ranked=kept_videos,
        )

        # Delete files and sidecars from disk
        _delete_entries(del_images, batch_dir)
        _delete_entries(del_videos, batch_dir)

        items_summary[item_id] = {
            "images": {"deleted": len(del_images), "kept": len(kept_images)},
            "videos": {"deleted": len(del_videos), "kept": len(kept_videos)},
        }

        log.info(
            "prune_batch %s item %s: img_del=%d img_kept=%d vid_del=%d vid_kept=%d filter=%s",
            batch_id, item_id,
            len(del_images), len(kept_images),
            len(del_videos), len(kept_videos),
            fs.model_dump(exclude_none=True),
        )

    return {
        "batch_id": batch_id,
        "items":    items_summary,
    }


# ---------------------------------------------------------------------------
# POST /batches/{batch_id}/items/{item_id}/append — add more results to a batch item
# ---------------------------------------------------------------------------

@app.post("/batches/{batch_id}/items/{item_id}/append", status_code=202)
async def append_to_batch(
    batch_id: str,
    item_id:  str,
    body: AppendRequest,
    _: None = Depends(require_auth),
):
    """
    Trigger an additional media search and append the new results into an
    existing batch item — without replacing current results.

    Creates a temporary mini-batch, runs the normal search+score pipeline on
    it, then merges into the target item when the caller polls the status URL.

    Returns 202 immediately with a poll_url.
    """
    state = store.get(batch_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Batch not found")
    if state["status"] == "running":
        raise HTTPException(status_code=409,
                            detail="Cannot append to a running batch — wait for it to finish")
    if item_id not in state["items"]:
        raise HTTPException(status_code=422, detail=f"Unknown item_id: {item_id}")

    n_img = body.n_img if body.n_img is not None else config.get("candidates_per_source_image")
    n_vid = body.n_vid if body.n_vid is not None else config.get("candidates_per_source_video")
    if n_img is None:
        raise HTTPException(status_code=500, detail="candidates_per_source_image missing from config.json")
    if n_vid is None:
        raise HTTPException(status_code=500, detail="candidates_per_source_video missing from config.json")

    # Build a minimal single-item manifest.  Must use "asset_id" key (NOT "item_id")
    # so that create_batch's list→dict conversion in server.py line 352 picks it up.
    tmp_batch_id = "b_" + uuid.uuid4().hex[:8]
    store.create(
        tmp_batch_id,
        state["project"],
        state["episode_id"],
        top_n            = state.get("top_n", 5),
        backgrounds      = {item_id: {
            **state["items"][item_id],          # inherit scoring_hints, cinematic_role, search_filters, etc.
            "asset_id":      item_id,
            "ai_prompt":     body.ai_prompt,
            "search_prompt": body.ai_prompt,    # worker reads search_prompt for API queries
            "search_queries": None,             # ignore parent's rich queries — use search_prompt verbatim
            "images_ranked": [],                # reset — fresh tmp batch
            "videos_ranked": [],
        }},
        content_profile         = state.get("content_profile", "default"),
        n_img                   = n_img,
        n_vid                   = n_vid,
        sources_override        = body.sources_override or state.get("sources_override"),
        # When n_img/n_vid is explicitly passed by the caller, apply it to
        # each source's candidates_images/candidates_videos so it isn't
        # silently swallowed by an existing source_limits_override on the
        # parent batch.
        source_limits_override  = {
            src: {
                "candidates_images": n_img if body.n_img is not None
                                     else lims.get("candidates_images", n_img),
                "candidates_videos": n_vid if body.n_vid is not None
                                     else lims.get("candidates_videos", n_vid),
            }
            for src, lims in (state.get("source_limits_override") or {}).items()
        } or None,
    )
    await batch_queue.put(tmp_batch_id)

    log.info("append_to_batch: created tmp batch %s for %s/%s", tmp_batch_id, batch_id, item_id)

    return {
        "status":       "appending",
        "tmp_batch_id": tmp_batch_id,
        "poll_url":     f"/batches/{batch_id}/items/{item_id}/append/{tmp_batch_id}",
    }


# ---------------------------------------------------------------------------
# GET /batches/{batch_id}/items/{item_id}/append/{tmp_batch_id} — poll + merge
# ---------------------------------------------------------------------------

@app.get("/batches/{batch_id}/items/{item_id}/append/{tmp_batch_id}")
async def poll_append(
    batch_id:     str,
    item_id:      str,
    tmp_batch_id: str,
    _:            None = Depends(require_auth),
):
    """
    Poll an append operation.  When the mini-batch is done, merges new results
    into the target batch item, moves files, and cleans up the tmp batch.
    """
    state = store.get(batch_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Batch not found")
    if item_id not in state["items"]:
        raise HTTPException(status_code=422, detail=f"Unknown item_id: {item_id}")

    tmp_state = store.get(tmp_batch_id)
    if tmp_state is None:
        # Already merged and cleaned up on a previous poll — return last known state
        raise HTTPException(status_code=404,
                            detail="Tmp batch not found — may have already been merged")

    tmp_status = tmp_state.get("status")

    if tmp_status in ("failed", "interrupted"):
        err = tmp_state.get("error", "unknown error")
        raise HTTPException(status_code=500, detail=f"Append batch failed: {err}")

    if tmp_status != "done":
        return {"status": "pending", "tmp_batch_status": tmp_status}

    # ── Merge ──────────────────────────────────────────────────────────────────
    target_item = state["items"][item_id]
    tmp_item    = tmp_state["items"].get(item_id, {})

    existing_images = list(target_item.get("images_ranked", []))
    existing_videos = list(target_item.get("videos_ranked", []))
    new_images      = list(tmp_item.get("images_ranked", []))
    new_videos      = list(tmp_item.get("videos_ranked", []))

    # Deduplicate by path OR canonical asset_page_url.
    # Path-only dedup fails when the same remote image is downloaded under a
    # different local filename (e.g. same Commons asset found via Wikimedia AND
    # Openverse in separate append searches).  asset_page_url is the stable
    # identity key across all sources (Wikimedia curid URL, Pexels photo page,
    # etc.) and is already used for attribution dedup elsewhere.
    def _img_keys(entries: list[dict]) -> tuple[set[str], set[str]]:
        paths = set()
        urls  = set()
        for e in entries:
            if e.get("path"):
                paths.add(e["path"])
            u = (e.get("source") or {}).get("asset_page_url", "")
            if u:
                urls.add(u)
        return paths, urls

    existing_img_paths, existing_img_urls = _img_keys(existing_images)
    existing_vid_paths, existing_vid_urls = _img_keys(existing_videos)

    def _is_new_img(e: dict) -> bool:
        if e.get("path") in existing_img_paths:
            return False
        u = (e.get("source") or {}).get("asset_page_url", "")
        if u and u in existing_img_urls:
            return False
        return True

    def _is_new_vid(e: dict) -> bool:
        if e.get("path") in existing_vid_paths:
            return False
        u = (e.get("source") or {}).get("asset_page_url", "")
        if u and u in existing_vid_urls:
            return False
        return True

    new_images_deduped = [e for e in new_images if _is_new_img(e)]
    new_videos_deduped = [e for e in new_videos if _is_new_vid(e)]

    # Move files from tmp batch dir to target batch dir (same relative structure)
    target_batch_dir = store.batch_dir(state["project"], state["episode_id"], batch_id)
    tmp_batch_dir    = store.batch_dir(tmp_state["project"], tmp_state["episode_id"], tmp_batch_id)

    def _move_entries(entries: list[dict]) -> list[dict]:
        """Move each entry's file (and .info.json sidecar) into target_batch_dir.
        Paths are relative to batch_dir in both tmp and target, so the relative
        path is preserved verbatim — only the parent directory changes."""
        updated = []
        for entry in entries:
            e2  = dict(entry)
            rel = e2.get("path", "")
            if not rel:
                updated.append(e2)
                continue
            src_abs = (tmp_batch_dir / rel).resolve()
            dst_abs = (target_batch_dir / rel).resolve()
            # Path-traversal guard (mirrors _delete_entries)
            try:
                src_abs.relative_to(tmp_batch_dir)
                dst_abs.relative_to(target_batch_dir)
            except ValueError:
                log.warning("append merge: skipping path outside batch dir: rel=%s", rel)
                continue
            dst_abs.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(src_abs), str(dst_abs))
                src_sidecar = Path(str(src_abs) + ".info.json")
                if src_sidecar.exists():
                    shutil.move(str(src_sidecar), str(Path(str(dst_abs) + ".info.json")))
            except Exception as exc:
                log.warning("append merge: move failed %s → %s: %s", src_abs, dst_abs, exc)
                # path stays valid (file may already be at dst from a previous partial run)
            updated.append(e2)
        return updated

    new_images_moved = _move_entries(new_images_deduped)
    new_videos_moved = _move_entries(new_videos_deduped)

    # Merge and re-sort by score descending
    merged_images = sorted(
        existing_images + new_images_moved,
        key=lambda e: e.get("score", 0.0), reverse=True,
    )
    merged_videos = sorted(
        existing_videos + new_videos_moved,
        key=lambda e: e.get("score", 0.0), reverse=True,
    )

    # Persist state BEFORE cleanup (batch_state is always consistent)
    store.update_item(
        batch_id, item_id,
        status        = target_item.get("status", "done"),
        images_ranked = merged_images,
        videos_ranked = merged_videos,
    )

    # Remove tmp batch from memory + disk (media files already moved)
    store.delete(tmp_batch_id)
    shutil.rmtree(str(tmp_batch_dir), ignore_errors=True)

    log.info(
        "poll_append: merged tmp=%s into %s/%s  +%d imgs +%d vids",
        tmp_batch_id, batch_id, item_id,
        len(new_images_moved), len(new_videos_moved),
    )

    return {
        "status":          "done",
        "images_appended": len(new_images_moved),
        "videos_appended": len(new_videos_moved),
        "images_total":    len(merged_images),
        "videos_total":    len(merged_videos),
    }


# ---------------------------------------------------------------------------
# POST /ai_ask — natural-language media operation via Claude subprocess
# ---------------------------------------------------------------------------

@app.post("/ai_ask")
async def ai_ask(body: AiAskRequest, _: None = Depends(require_auth)):
    """
    Accept a natural-language prompt from a client, build a system prompt
    via media_ask.build_prompt(), and spawn a `claude -p` subprocess to
    execute the appropriate media tool operations.

    Claude calls bash → python server_tools.py <subcommand> → media server REST
    endpoints.  The client never calls Claude directly.

    Returns: { "response": "<Claude's final text>", "error": null | "<stderr>" }
    """
    full_prompt = build_prompt(
        query      = body.prompt,
        project    = body.project,
        episode_id = body.episode_id,
        batch_id   = body.batch_id,
    )

    log.info("ai_ask: prompt=%r  project=%r  episode=%r  batch=%r",
             body.prompt, body.project, body.episode_id, body.batch_id)

    try:
        proc = await asyncio.to_thread(
            subprocess.run,
            ["claude", "-p", "--allowedTools", "Bash", "--output-format", "text"],
            input          = full_prompt,
            capture_output = True,
            text           = True,
            timeout        = 300,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Claude subprocess timed out after 300s")
    except FileNotFoundError:
        raise HTTPException(status_code=500,
                            detail="claude binary not found — ensure it is on PATH")
    except Exception as exc:
        log.exception("ai_ask: subprocess failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    log.info("ai_ask: claude returned rc=%d  stdout_len=%d  response=%r",
             proc.returncode, len(proc.stdout), proc.stdout.strip())

    # Persist last AI ask so Refresh can surface it in the UI
    try:
        import datetime as _dt
        _ai_ask_path = (PROJECTS_ROOT / body.project / "episodes" / body.episode_id
                        / "assets" / "media" / "ai_ask_last.json")
        _ai_ask_path.parent.mkdir(parents=True, exist_ok=True)
        _ai_ask_path.write_text(json.dumps({
            "prompt":    body.prompt,
            "response":  proc.stdout.strip(),
            "error":     proc.stderr.strip() if proc.returncode != 0 else None,
            "timestamp": _dt.datetime.now().isoformat(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as _e:
        log.warning("ai_ask: failed to write ai_ask_last.json: %s", _e)

    return {
        "response": proc.stdout.strip(),
        "error":    proc.stderr.strip() if proc.returncode != 0 else None,
    }


# ---------------------------------------------------------------------------
# GET /ai_ask_last — return last AI ask prompt+response for a project/episode
# ---------------------------------------------------------------------------

@app.get("/ai_ask_last")
async def ai_ask_last(project: str = Query(...), episode_id: str = Query(...),
                      _: None = Depends(require_auth)):
    """Return the last AI ask prompt and response for the given project/episode."""
    path = PROJECTS_ROOT / project / "episodes" / episode_id / "assets" / "media" / "ai_ask_last.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# POST /fetch_direct — download a specific Pexels photo by URL into a batch
# ---------------------------------------------------------------------------

class FetchDirectRequest(BaseModel):
    batch_id: str
    item_id:  str
    url:      str


@app.post("/fetch_direct")
async def fetch_direct(body: FetchDirectRequest, _: None = Depends(require_auth)):
    """
    Download a specific Pexels photo by its page URL and append it to an
    existing batch item's result set (score=1.0, as user-selected).

    Extracts the numeric photo ID from the URL, calls the Pexels single-photo
    API, downloads src.large, writes the file to the standard batch path, and
    appends a result entry to images_ranked via read-modify-write.

    Returns: { "photo_id": int, "path": str, "title": str }
    """
    import re
    import requests as _req

    # ── 1. Extract Pexels photo ID from URL ──────────────────────────────────
    # Supports: https://www.pexels.com/photo/<slug>-<id>/
    m = re.search(r"-(\d+)/?$", body.url.rstrip("/"))
    if not m:
        raise HTTPException(status_code=400, detail="Cannot extract photo ID from URL")
    photo_id = int(m.group(1))

    # ── 2. Validate batch and item; reject if batch is still running ──────────
    state = store.get(body.batch_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Batch not found")
    if state["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail="Cannot add to a running batch — wait for it to finish",
        )
    if body.item_id not in state["items"]:
        raise HTTPException(status_code=404,
                            detail=f"Item '{body.item_id}' not found in batch")

    # ── 3. Call Pexels single-photo API ──────────────────────────────────────
    pexels_key = API_KEYS.get("pexels", "")
    if not pexels_key:
        raise HTTPException(status_code=500,
                            detail="PEXELS_API_KEY not configured on server")

    api_url = f"https://api.pexels.com/v1/photos/{photo_id}"
    try:
        api_resp = await asyncio.to_thread(
            lambda: _req.get(api_url,
                             headers={"Authorization": pexels_key},
                             timeout=15)
        )
    except Exception as exc:
        log.exception("fetch_direct: Pexels API call failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Pexels API unreachable: {exc}")

    if api_resp.status_code == 403:
        raise HTTPException(status_code=500,
                            detail="Pexels API key invalid or unauthorized")
    if api_resp.status_code == 404:
        raise HTTPException(status_code=404, detail="Photo not found on Pexels")
    if api_resp.status_code == 429:
        raise HTTPException(status_code=429,
                            detail="Pexels rate limit hit — retry after a moment")
    if not api_resp.ok:
        raise HTTPException(status_code=502,
                            detail=f"Pexels API error {api_resp.status_code}: "
                                   f"{api_resp.text[:200]}")

    photo   = api_resp.json()
    src     = photo.get("src", {})
    img_url = src.get("large") or src.get("original")
    if not img_url:
        raise HTTPException(status_code=502,
                            detail="Pexels API returned no usable image URL")

    # ── 4. Download to standard batch path ───────────────────────────────────
    batch_dir = store.batch_dir(state["project"], state["episode_id"], body.batch_id)
    dest_dir  = batch_dir / body.item_id / "images" / "pexels"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"pexels_img_{photo_id}.jpg"

    if not dest.exists():
        def _download():
            r = _req.get(img_url,
                         headers={"User-Agent": "PipedMediaServer/1.0"},
                         stream=True, timeout=60)
            r.raise_for_status()
            tmp = dest.with_suffix(".part")
            with tmp.open("wb") as fh:
                for chunk in r.iter_content(65536):
                    if chunk:
                        fh.write(chunk)
            tmp.replace(dest)
        try:
            await asyncio.to_thread(_download)
        except Exception as exc:
            log.exception("fetch_direct: image download failed: %s", exc)
            raise HTTPException(status_code=502,
                                detail=f"Image download failed: {exc}")

    # ── 5. Read-modify-write: append new entry to images_ranked ──────────────
    rel_path  = str(dest.relative_to(batch_dir))
    new_entry = {
        "path":  rel_path,
        "score": 1.0,
        "source": {
            "source_site":    "pexels",
            "asset_page_url": photo.get("url", body.url),
            "photographer":   photo.get("photographer", ""),
            "width":          photo.get("width"),
            "height":         photo.get("height"),
        },
    }

    item_state = state["items"][body.item_id]
    existing   = list(item_state.get("images_ranked", []))
    store.update_item(
        body.batch_id, body.item_id,
        status=item_state["status"],
        images_ranked=existing + [new_entry],
    )

    title = photo.get("alt") or photo.get("url", f"pexels_img_{photo_id}")
    log.info("fetch_direct: added pexels photo %d to %s/%s  path=%s",
             photo_id, body.batch_id, body.item_id, rel_path)

    return {
        "photo_id": photo_id,
        "path":     rel_path,
        "title":    title,
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
            "source_limits":               config.get("source_limits", {}),
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
        "*.openverse.engineering", "cdn.openverse.engineering",  # legacy domain
        "api.openverse.org", "*.openverse.org",                  # current domain
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

    # Patch AssetManifest.shared.json with structured license fields
    manifest_path = ep_dir / "AssetManifest.shared.json"
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
    result: dict | list   # dict for video jobs, list[dict] for image jobs


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



async def _score_images_distributed_batch(
    batch_id:   str,
    items_data: list[tuple[str, dict, list[Path], dict]],
    # each tuple: (item_id, item, img_paths, img_infos)
    cfg:        dict,
) -> dict[str, list[dict]]:
    """
    Enqueue image scoring tasks for ALL items in one batch so all workers
    can run in parallel.  Returns {item_id: scored_list}.
    """
    scoring_config = {
        k: cfg[k] for k in (
            "score_weights", "scoring_profiles", "content_profile",
            "phash_dedup_threshold", "diversity_phash_threshold",
        ) if k in cfg
    }

    tasks = []
    item_id_for_task = []   # parallel list: task index → item_id
    for item_id, item, img_paths, img_infos in items_data:
        if not img_paths:
            continue
        task = {
            "job_type":    "images",
            "item_id":     item_id,       # carried through for result mapping
            "image_paths": [str(p) for p in img_paths],
            "item":        item,
            "config":      scoring_config,
            "infos":       img_infos or {},
        }
        tasks.append(task)
        item_id_for_task.append(item_id)

    if not tasks:
        return {}

    workers_cfg     = cfg.get("workers", {})
    timeout_seconds = workers_cfg.get("timeout_seconds", 120)

    async with _jq_sem:
        job_queue.enqueue(batch_id, tasks)
        job_queue.start_reaper(timeout_seconds=timeout_seconds)

        try:
            log.info(
                "Waiting for %d image scoring jobs across %d items (batch %s)…",
                len(tasks), len(tasks), batch_id,
            )
            await job_queue.wait_until_done()
        finally:
            job_queue.stop_reaper()

        results_by_job = job_queue.get_results_dict()   # {job_id: list[dict]}

    # Map results back to item_id
    _workers_nfs = [w.nfs_root for w in job_queue._workers.values()] if job_queue else []
    out: dict[str, list[dict]] = {}

    for job_id, scored in results_by_job.items():
        task_meta = job_queue._job_tasks.get(job_id, {})
        item_id = task_meta.get("item_id", "")
        if not item_id:
            log.warning("Image result job %s has no item_id in task metadata", job_id)
            continue

        if not isinstance(scored, list):
            log.warning("Unexpected image result shape for job %s item %s", job_id, item_id)
            out[item_id] = []
            continue

        # Reverse-remap worker paths → server paths
        remapped = []
        for r in scored:
            worker_path = r.get("path", "")
            server_path = worker_path
            for wnfs in _workers_nfs:
                if worker_path.startswith(wnfs):
                    server_path = job_queue.server_nfs_root + worker_path[len(wnfs):]
                    break
            remapped.append({**r, "path": server_path})
        out[item_id] = remapped

    log.info(
        "Distributed image scoring complete: %d items scored (batch %s)",
        len(out), batch_id,
    )
    return out


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

async def _run_batch(batch_id: str) -> None:
    state       = store.get(batch_id)
    project     = state["project"]
    episode_id  = state["episode_id"]
    backgrounds = state["items"]
    item_count  = len(backgrounds)
    n_img       = state.get("n_img") or config.get("candidates_per_source_image")
    n_vid       = state.get("n_vid") or config.get("candidates_per_source_video")

    # Apply per-batch content_profile override (section 30)
    # Use a shallow copy so the global config is never mutated across concurrent batches
    cfg = dict(config)
    cfg["content_profile"] = store.get_content_profile(batch_id)
    # Apply per-batch source overrides from UI settings
    if state.get("sources_override"):
        cfg["sources"] = state["sources_override"]
    # Auto-derive source_limits_override when absent: divide n_img/n_vid evenly
    # across all active sources so callers don't need to supply it explicitly.
    source_limits_override = state.get("source_limits_override") or {}
    active_sources = cfg.get("sources", [])
    if not source_limits_override and active_sources:
        per_img = n_img or config.get("candidates_per_source_image", 30)
        per_vid = n_vid or config.get("candidates_per_source_video", 30)
        source_limits_override = {
            src: {"candidates_images": per_img, "candidates_videos": per_vid}
            for src in active_sources
        }
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

    # Distributed workers are no longer used for scoring (keyword-match scoring
    # is synchronous and requires no GPU workers).

    # Shared counters (mutated only from async gather callbacks — safe in asyncio)
    # dl_started: increments as each item begins downloading (for progress display)
    # score_started: increments as each item begins scoring (for progress display)
    # done: increments when an item is fully complete (images + videos scored)
    done:          list[int] = [0]
    dl_started:    list[int] = [0]
    score_started: list[int] = [0]

    # ---------------------------------------------------------------
    # Phase 1 — Download (all items in parallel)
    # ---------------------------------------------------------------

    # dl_data[item_id] = (img_path_infos, vid_path_infos, item_pending, resolved_item)
    dl_data: dict[str, tuple] = {}

    async def _download_item(item_id: str, item: dict) -> tuple:
        """Download images and videos for one item. Return data for scoring."""
        # Skip items already completed (resume support)
        if item.get("status") == "done":
            log.info("[%s]  already done — skipping", item_id)
            done[0] += 1
            return item_id, [], [], [], item

        search_prompt = item.get("search_prompt", "")

        # C3/M6: media_type → prefer
        sf     = item.get("search_filters") or {}
        mt     = sf.get("media_type", "mixed")
        prefer = {"image": "image", "video": "video"}.get(mt, "both")

        # Inject Pixabay dimension constraints
        src_filters = dict(item.get("source_filters") or {})
        pbay = dict(src_filters.get("pixabay") or {})
        if sf.get("min_width")  and "min_width"  not in pbay:
            pbay["min_width"]  = sf["min_width"]
        if sf.get("min_height") and "min_height" not in pbay:
            pbay["min_height"] = sf["min_height"]
        if pbay:
            src_filters["pixabay"] = pbay
        item = {**item, "prefer": prefer, "source_filters": src_filters}

        item_dir = batch_dir / item_id
        img_dir  = item_dir / "images"
        vid_dir  = item_dir / "videos"

        dl_started[0] += 1
        store.update(batch_id, progress=f"downloading {item_id} ({dl_started[0]}/{item_count})")
        store.update_item_progress(batch_id, item_id, phase="downloading")

        try:
            log.info("[%s]  download start  prompt=%r  n_img=%d  n_vid=%d  prefer=%s",
                     item_id, search_prompt[:80], n_img, n_vid, prefer)
            t_dl = time.perf_counter()

            async with _sem:
                async def _fetch_imgs():
                    if prefer == "video":
                        return [], []
                    result = await asyncio.to_thread(
                        downloader.fetch_images,
                        search_prompt, n_img, img_dir, API_KEYS, cfg, item,
                    )
                    if isinstance(result, tuple) and len(result) == 2:
                        return result
                    return result, []

                async def _fetch_vids():
                    if prefer == "image":
                        return [], []
                    result = await asyncio.to_thread(
                        downloader.fetch_videos,
                        search_prompt, n_vid, vid_dir, API_KEYS, cfg, item,
                    )
                    if isinstance(result, tuple) and len(result) == 2:
                        return result
                    return result, []

                (img_path_infos, img_pending), (vid_path_infos, vid_pending) = (
                    await asyncio.gather(_fetch_imgs(), _fetch_vids())
                )

            log.info("[%s]  download done   imgs=%d  vids=%d  elapsed=%.1fs",
                     item_id, len(img_path_infos), len(vid_path_infos),
                     time.perf_counter() - t_dl)

            return item_id, img_path_infos, vid_path_infos, img_pending + vid_pending, item

        except Exception as exc:
            log.exception("  item %s download failed: %s", item_id, exc)
            store.update_item(batch_id, item_id, status="failed", error=str(exc))
            return item_id, [], [], [], item

    download_tasks = [_download_item(iid, idata) for iid, idata in backgrounds.items()]
    download_outputs = await asyncio.gather(*download_tasks)

    # Collect download results; update store progress
    pending_by_item: dict[str, list] = {}   # item_id → pending candidates list
    items_to_score: dict[str, tuple] = {}   # item_id → (img_path_infos, vid_path_infos, item)
    for item_id, img_pi, vid_pi, item_pending, resolved_item in download_outputs:
        if item_pending:
            pending_by_item[item_id] = item_pending
            log.info("[%s]  %d candidates pending host review", item_id, len(item_pending))
        if resolved_item.get("status") == "done":
            continue
        items_to_score[item_id] = (img_pi, vid_pi, resolved_item)
        score_started[0] += 1
        store.update(batch_id,
                     progress=f"scoring {item_id} ({score_started[0]}/{item_count})")
        store.update_item_progress(
            batch_id, item_id, phase="scoring",
            imgs_downloaded=len(img_pi),
            vids_downloaded=len(vid_pi),
        )

    # ---------------------------------------------------------------
    # Phase 2 — Image scoring (keyword-match; no CLIP inference)
    # ---------------------------------------------------------------

    image_results: dict[str, list[dict]] = {}   # item_id → scored list

    for item_id, (img_pi, _, item) in items_to_score.items():
        query  = item.get("ai_prompt") or item.get("search_prompt", "")
        scored = []
        for p, info in img_pi:
            scored.append({
                "path":   str(p),
                "score":  _keyword_score(query, info),
                "source": info,
            })
        scored.sort(key=lambda r: r["score"], reverse=True)
        image_results[item_id] = scored
        log.info("Image keyword scoring: item=%s query=%r scored=%d", item_id, query, len(scored))

    # ---------------------------------------------------------------
    # Phase 3 — Video scoring (keyword-match; no CLIP inference)
    # ---------------------------------------------------------------

    video_results: dict[str, list[dict]] = {}   # item_id → scored list

    for item_id, (_, vid_pi, item) in items_to_score.items():
        query  = item.get("ai_prompt") or item.get("search_prompt", "")
        scored = []
        for p, info in vid_pi:
            scored.append({
                "path":   str(p),
                "score":  _keyword_score(query, info),
                "source": info,
            })
        scored.sort(key=lambda r: r["score"], reverse=True)
        video_results[item_id] = scored
        log.info("Video keyword scoring: item=%s query=%r scored=%d", item_id, query, len(scored))

    # ---------------------------------------------------------------
    # Phase 4 — Save results per item
    # ---------------------------------------------------------------

    for item_id in items_to_score:
        images_ranked = image_results.get(item_id, [])
        videos_ranked = video_results.get(item_id, [])

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
        done[0] += 1

    # Collect pending candidates per-item (keyed by item_id)
    pending_candidates: dict[str, list] = pending_by_item

    if pending_candidates:
        total_pending = sum(len(v) for v in pending_candidates.values())
        log.info("Batch %s: %d candidates across %d items pending host review",
                 batch_id, total_pending, len(pending_candidates))
        store.update(batch_id, pending_candidates=pending_candidates)

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
