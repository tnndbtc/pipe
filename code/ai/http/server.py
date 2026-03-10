"""
AI Asset Generation Server
FastAPI server that accepts asset generation jobs, runs gen_*.py scripts
as subprocesses, and serves results back via HTTP.

Usage:
    python server.py
    # or: uvicorn server:app --host 0.0.0.0 --port 8100
"""
import asyncio
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Optional

import logging

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from job_store import JobStore

logger = logging.getLogger("ai_server")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).parent.parent  # → code/ai/

# asset_type → (script_filename, model_cli_flag)
DISPATCH: dict[str, tuple[str, str]] = {
    "characters":  ("gen_character_images.py",  "--model"),
    "backgrounds": ("gen_background_images.py", "--model"),
    "bg_video":    ("gen_background_video.py",  "--model"),
    "sfx":         ("gen_sfx.py",               "--backend"),
}

# asset_type → manifest section key (for counting total assets)
MANIFEST_SECTION: dict[str, str] = {
    "characters":  "character_packs",
    "backgrounds": "backgrounds",
    "bg_video":    "backgrounds",   # subset; all backgrounds counted
    "sfx":         "sfx_items",
}

ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]|\r')

LOG_TAIL_LINES = 20

# ---------------------------------------------------------------------------
# Config + store initialisation
# ---------------------------------------------------------------------------

_cfg_path = Path(__file__).parent / "config.json"
config: dict = json.loads(_cfg_path.read_text(encoding="utf-8"))
store = JobStore(config.get("job_dir"))
job_queue: asyncio.Queue = asyncio.Queue()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="AI Asset Generation Server", version="1.0.0")


@app.on_event("startup")
async def on_startup() -> None:
    if config.get("offline"):
        logger.warning("[OFFLINE] Server started in offline mode — all job submissions will be rejected.")
    else:
        logger.info("[ONLINE] Server ready — accepting job submissions.")
    store.startup_scan()
    asyncio.create_task(queue_worker())


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def require_auth(x_api_key: str = Header(...)) -> None:
    if x_api_key != config["api_key"]:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class JobRequest(BaseModel):
    manifest:    dict
    asset_types: list[str]
    asset_ids:   Optional[list[str]] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/jobs")
async def create_job(body: JobRequest, x_api_key: str = Header(...)):
    raise_if_bad_key(x_api_key)

    if config.get("offline"):
        logger.warning("[OFFLINE] POST /jobs rejected — asset_types=%s", body.asset_types)
        return {"status": "unavailable", "job_id": None, "total": 0}

    # Validate asset types
    unknown = [t for t in body.asset_types if t not in DISPATCH]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown asset_types: {unknown}")

    # Count total assets
    total = _count_total(body.manifest, body.asset_types, body.asset_ids)

    # Create job
    job_id = str(uuid.uuid4())
    store.create(job_id, total)

    # Write manifest to job dir
    manifest_path = store.job_path(job_id) / "manifest.json"
    manifest_path.write_text(
        json.dumps(body.manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Enqueue
    await job_queue.put((job_id, body))

    return {"job_id": job_id, "status": "queued", "total": total}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str, x_api_key: str = Header(...)):
    raise_if_bad_key(x_api_key)

    state = store.get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Read log tail
    log_path = store.log_path(job_id)
    tail: list[str] = []
    if log_path.exists():
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = lines[-LOG_TAIL_LINES:]
        except Exception:
            pass

    return {**state, "log_tail": tail}


@app.get("/jobs/{job_id}/files/{filename}")
async def download_file(job_id: str, filename: str, x_api_key: str = Header(...)):
    raise_if_bad_key(x_api_key)

    state = store.get(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Prevent path traversal: only allow files listed in state
    if filename not in state.get("files", []):
        raise HTTPException(status_code=404, detail="File not found")

    path = store.out_path(job_id) / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found on disk")

    return StreamingResponse(
        path.open("rb"),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(path.stat().st_size),
        },
    )


@app.get("/health")
async def health():
    if config.get("offline"):
        logger.info("[OFFLINE] GET /health — server is in offline mode")
        return {
            "status":       "offline",
            "gpu":          "N/A",
            "vram_free_gb": 0.0,
            "queue_len":    0,
        }
    gpu_info = _gpu_info()
    return {
        "status":       "ok",
        "gpu":          gpu_info["name"],
        "vram_free_gb": gpu_info["vram_free_gb"],
        "queue_len":    job_queue.qsize(),
    }


# ---------------------------------------------------------------------------
# Queue worker
# ---------------------------------------------------------------------------

async def queue_worker() -> None:
    while True:
        job_id, body = await job_queue.get()
        try:
            await run_job(job_id, body)
        except Exception as exc:
            store.update(job_id, status="failed", errors=[str(exc)])
        finally:
            job_queue.task_done()


# ---------------------------------------------------------------------------
# Job runner
# ---------------------------------------------------------------------------

async def run_job(job_id: str, body: JobRequest) -> None:
    store.update(job_id, status="running")
    manifest_path = store.job_path(job_id) / "manifest.json"
    out_dir       = store.out_path(job_id)
    log_path      = store.log_path(job_id)

    with log_path.open("w", encoding="utf-8") as log_fh:
        for asset_type in body.asset_types:
            script_name, model_flag = DISPATCH[asset_type]
            model = config["models"].get(asset_type, "auto")

            cmd = [
                sys.executable,
                str(SCRIPTS_DIR / script_name),
                "--manifest",   str(manifest_path),
                "--output_dir", str(out_dir),
                model_flag,     model,
            ]

            # sfx: also pass --model (small/medium/auto) separately from --backend
            if asset_type == "sfx":
                cmd += ["--model", config.get("sfx_model", "auto")]

            bg_hint = config.get("bg_hints", {}).get(asset_type)
            if bg_hint:
                cmd += ["--bg-hint", bg_hint]

            if body.asset_ids:
                for aid in body.asset_ids:
                    await _run_subprocess(job_id, cmd + ["--asset-id", aid], log_fh, out_dir)
            else:
                await _run_subprocess(job_id, cmd, log_fh, out_dir)

    # Final state update
    final_files = sorted(
        f.name for f in out_dir.iterdir()
        if f.is_file() and not f.name.endswith(".json")
    )
    final_status = "failed" if store.get(job_id).get("errors") else "done"
    store.update(job_id, status=final_status, files=final_files)


async def _run_subprocess(
    job_id: str,
    cmd: list[str],
    log_fh,
    out_dir: Path,
) -> None:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    timeout_s = config.get("subprocess_timeout_s", 7200)  # default 2 hours
    lines_since_update = 0

    async def _drain():
        nonlocal lines_since_update
        async for raw in proc.stdout:
            line = ANSI_RE.sub("", raw.decode("utf-8", errors="replace"))
            log_fh.write(line)
            log_fh.flush()
            # Periodically update done counter so caller sees progress
            lines_since_update += 1
            if lines_since_update >= 50:
                lines_since_update = 0
                files_now = sorted(
                    f.name for f in out_dir.iterdir()
                    if f.is_file() and not f.name.endswith(".json")
                )
                store.update(job_id, done=len(files_now), files=files_now)
        await proc.wait()

    timed_out = False
    try:
        await asyncio.wait_for(_drain(), timeout=timeout_s)
    except asyncio.TimeoutError:
        timed_out = True
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        msg = f"[SERVER] {Path(cmd[1]).name} timed out after {timeout_s}s — process killed.\n"
        log_fh.write(msg)
        log_fh.flush()
        logger.error("[JOB %s] %s", job_id, msg.strip())

    # Final progress update after subprocess exits
    files_so_far = sorted(
        f.name for f in out_dir.iterdir()
        if f.is_file() and not f.name.endswith(".json")
    )
    state = store.get(job_id)
    extra_errors = []
    if timed_out:
        extra_errors.append(f"{Path(cmd[1]).name} timed out after {timeout_s}s")
    elif proc.returncode != 0:
        extra_errors.append(f"{Path(cmd[1]).name} exited {proc.returncode}")
    store.update(
        job_id,
        done=len(files_so_far),
        files=files_so_far,
        errors=(state.get("errors") or []) + extra_errors,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def raise_if_bad_key(key: Optional[str]) -> None:
    if key != config["api_key"]:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _count_total(manifest: dict, asset_types: list[str], asset_ids: Optional[list[str]]) -> int:
    total = 0
    for asset_type in asset_types:
        section_key = MANIFEST_SECTION.get(asset_type, "")
        section = manifest.get(section_key, [])
        if asset_ids:
            section = [
                item for item in section
                if (item.get("id") if isinstance(item, dict) else item) in asset_ids
            ]
        total += len(section)
    return total


def _gpu_info() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            free_bytes, _ = torch.cuda.mem_get_info(idx)
            return {"name": name, "vram_free_gb": round(free_bytes / 1e9, 1)}
    except Exception:
        pass
    return {"name": "N/A", "vram_free_gb": 0.0}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="AI Asset Generation Server")
    parser.add_argument("--host",    default="192.168.86.27")
    parser.add_argument("--port",    type=int, default=8000)
    parser.add_argument("--offline", action="store_true",
                        help="Start in offline mode: server accepts connections "
                             "but rejects all job submissions with status=unavailable.")
    cli = parser.parse_args()

    if cli.offline:
        config["offline"] = True
        print("[SERVER] Starting in OFFLINE mode — job submissions will be rejected.")

    uvicorn.run(app, host=cli.host, port=cli.port, reload=False)
