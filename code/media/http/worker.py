"""
worker.py — Distributed video scoring worker.

A lightweight polling process that:
  1. Registers with the main media server on startup
  2. Polls GET /next_job for video scoring tasks
  3. Scores each video locally (ffmpeg + CLIP)
  4. Posts the result back via POST /result
  5. Repeats until stopped (SIGTERM / SIGINT)

Usage
-----
    python worker.py --server http://192.168.86.33:8200
    python worker.py --server http://192.168.86.33:8200 --name alma-41
    python worker.py --server http://192.168.86.33:8200 --nfs-root /nfs/media

Workers are stateless — they can be started/stopped at any time.
No configuration change or server restart is needed to add a new worker.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import socket
import sys
import time
from pathlib import Path

import requests

import scorer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("worker")

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_shutdown = False


def _handle_signal(signum, _frame):
    global _shutdown
    log.info("Received signal %s — finishing current job then exiting", signum)
    _shutdown = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT,  _handle_signal)

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(server_url: str, name: str, nfs_root: str) -> None:
    """POST /register to announce this worker to the server."""
    payload = {
        "name":     name,
        "hostname": socket.gethostname(),
        "nfs_root": nfs_root,
    }
    try:
        r = requests.post(f"{server_url}/register", json=payload, timeout=10)
        r.raise_for_status()
        log.info("Registered with server: %s (name=%s, nfs_root=%s)",
                 server_url, name, nfs_root)
    except Exception as exc:
        log.warning("Registration failed (will retry on next poll): %s", exc)


# ---------------------------------------------------------------------------
# Main poll loop
# ---------------------------------------------------------------------------


def poll_loop(
    server_url: str,
    name:       str,
    nfs_root:   str,
    clip_model: scorer.ClipModel,
) -> None:
    """Poll for jobs, score videos, submit results. Runs until shutdown."""

    session = requests.Session()

    while not _shutdown:
        # -- Poll for next job --
        try:
            resp = session.get(
                f"{server_url}/next_job",
                params={"worker": name},
                timeout=30,
            )
        except requests.ConnectionError:
            log.warning("Server unreachable — retrying in 10s")
            time.sleep(10)
            continue
        except Exception as exc:
            log.warning("Poll error: %s — retrying in 5s", exc)
            time.sleep(5)
            continue

        if resp.status_code == 410:
            # Server doesn't recognise us (likely restarted) — re-register
            log.info("Server lost our registration — re-registering")
            register(server_url, name, nfs_root)
            continue

        if resp.status_code == 204:
            # No batch active — back off
            time.sleep(5)
            continue

        if resp.status_code == 202:
            # Batch active but queue temporarily empty (requeues pending)
            time.sleep(1)
            continue

        if resp.status_code != 200:
            log.warning("Unexpected status %d from /next_job — retrying in 5s",
                        resp.status_code)
            time.sleep(5)
            continue

        # -- Got a job --
        job = resp.json()
        job_id     = job["job_id"]
        video_path = Path(job["video_path"])
        frames_dir = Path(job["frames_dir"])
        item       = job["item"]
        job_config = job["config"]

        log.info("Scoring job %s: %s", job_id, video_path.name)

        try:
            result = scorer.score_single_video(
                clip_model, video_path, frames_dir, item, job_config,
            )
        except Exception as exc:
            log.exception("Score failed for job %s: %s", job_id, exc)
            result = {
                "path":       str(video_path),
                "score":      0.0,
                "clip_score": 0.0,
                "calmness":   0.0,
                "error":      str(exc),
            }

        # -- Submit result --
        payload = {
            "job_id": job_id,
            "worker": name,
            "result": result,
        }
        try:
            r = session.post(
                f"{server_url}/result",
                json=payload,
                timeout=30,
            )
            r.raise_for_status()
            log.info("Submitted result for job %s (score=%.3f)",
                     job_id, result.get("score", 0.0))
        except Exception as exc:
            log.error("Failed to submit result for job %s: %s", job_id, exc)

    log.info("Worker shutting down gracefully")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Media scoring worker")
    ap.add_argument(
        "--server", required=True,
        help="URL of the main media server (e.g., http://192.168.86.33:8200)",
    )
    ap.add_argument(
        "--name", default=None,
        help="Human-readable worker name (defaults to hostname)",
    )
    ap.add_argument(
        "--nfs-root", default="/mnt/shared",
        help="Local NFS mount path (default: /mnt/shared)",
    )
    ap.add_argument(
        "--clip-model", default=None,
        help="CLIP model name override (default: from server config)",
    )
    ap.add_argument(
        "--clip-pretrained", default=None,
        help="CLIP pretrained weights override (default: from server config)",
    )
    args = ap.parse_args()

    server_url = args.server.rstrip("/")
    name       = args.name or socket.gethostname()
    nfs_root   = args.nfs_root

    # Load CLIP config from server health endpoint
    clip_cfg = {}
    try:
        r = requests.get(f"{server_url}/health", timeout=10)
        r.raise_for_status()
        health = r.json()
        clip_cfg = health.get("config", {})
        log.info("Connected to server: %s", server_url)
    except Exception as exc:
        log.warning("Could not reach server health endpoint: %s", exc)

    # Allow CLI overrides
    if args.clip_model:
        clip_cfg["clip_model"] = args.clip_model
    if args.clip_pretrained:
        clip_cfg["clip_pretrained"] = args.clip_pretrained

    # Load CLIP model
    log.info("Loading CLIP model (this may take a moment on first run)...")
    clip_model = scorer.load_clip(clip_cfg)

    # Register with server
    register(server_url, name, nfs_root)

    # Start polling
    log.info("Worker '%s' ready — polling %s for jobs", name, server_url)
    poll_loop(server_url, name, nfs_root, clip_model)


if __name__ == "__main__":
    main()
