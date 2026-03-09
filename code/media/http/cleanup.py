"""
cleanup.py — TTL-based eviction of old batch directories.

Called once by server.py at startup.

Scans:
    projects_root/**/assets/media/*/batch_state.json

For each batch directory:
- Reads created_at from batch_state.json (falls back to dir mtime)
- Removes the directory if age > ttl_days

Directory structure within a batch:
    <batch_dir>/
        <item_id>/
            images/<source>/<uid>.jpg           ← preview (Phase 1, always downloaded)
            images/<source>/<uid>.jpg.info.json ← metadata sidecar with file_url + preview_url
            images/<source>/hires/<uid>.jpg     ← high-res (Phase 2, downloaded on demand)
            videos/<source>/<uid>.mp4
            videos/<source>/<uid>.mp4.info.json
            videos/<source>/hires/<uid>.mp4     ← high-res video (Phase 2)

When purging (age > ttl_days): shutil.rmtree() deletes the entire batch directory
including all hires/ subdirectories — no extra handling needed.

When a batch is retained: hires/ subdirectories are preserved alongside their
parent preview files so that resolved manifests continue to reference valid paths.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("cleanup")


def evict_old_batches(projects_root: Path | str, ttl_days: int) -> int:
    """
    Delete batch directories older than ttl_days.

    Returns the number of directories evicted.
    """
    root   = Path(projects_root)
    now_ts = datetime.now(timezone.utc).timestamp()
    cutoff = now_ts - ttl_days * 86_400
    evicted = 0

    for state_file in root.glob("**/assets/media/*/batch_state.json"):
        batch_dir = state_file.parent
        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
            created_str = data.get("created_at", "")
            if created_str:
                created_ts = datetime.fromisoformat(created_str).timestamp()
            else:
                created_ts = state_file.stat().st_mtime

            age_days = (now_ts - created_ts) / 86_400
            if created_ts < cutoff:
                shutil.rmtree(batch_dir, ignore_errors=True)
                log.info("evicted batch %s (age=%.1fd)", batch_dir.name, age_days)
                evicted += 1

        except Exception as exc:  # noqa: BLE001
            log.warning("could not inspect batch dir %s: %s", batch_dir, exc)

    log.info("cleanup: %d batch(es) evicted (ttl=%dd)", evicted, ttl_days)
    return evicted
