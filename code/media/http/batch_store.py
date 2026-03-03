"""
batch_store.py — Persistent batch state for code/media/http/

Mirrors the pattern from code/ai/http/job_store.py but stores batches
in the project/episode directory tree rather than a tmp directory.

Batch state file location:
    <projects_root>/<project>/episodes/<episode_id>/assets/media/<batch_id>/batch_state.json

In-memory dict keyed by batch_id for fast access without disk reads on every poll.
All writes are atomic: temp file + os.replace().

Public API
----------
BatchStore(projects_root)
    .startup_scan()                         → mark interrupted batches failed; load in-memory
    .create(batch_id, project, episode_id, top_n, backgrounds)
    .update(batch_id, **fields)
    .update_item(batch_id, item_id, *, status, images_ranked, videos_ranked, error)
    .get(batch_id)                          → dict | None
    .list_for_episode(project, episode_id)  → list[dict]  (summary dicts)
    .batch_dir(project, episode_id, batch_id) → Path
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("batch_store")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class BatchStore:
    def __init__(self, projects_root: Path | str) -> None:
        self._root = Path(projects_root).resolve()
        self._batches: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def startup_scan(self) -> None:
        """
        Scan all batch_state.json files under projects_root.
        - Loads all batches into the in-memory dict.
        - Marks any "queued" or "running" batch as "failed" (interrupted at shutdown).
        """
        for state_file in self._root.glob("**/assets/media/*/batch_state.json"):
            try:
                data = json.loads(state_file.read_text(encoding="utf-8"))
                bid = data.get("batch_id")
                if not bid:
                    continue

                if data.get("status") in ("queued", "running"):
                    data["status"]     = "failed"
                    data["error"]      = "server_restarted"
                    data["updated_at"] = _now_iso()
                    self._write_atomic(state_file, data)
                    log.info("marked interrupted batch %s as failed", bid)

                self._batches[bid] = data

            except Exception as exc:  # noqa: BLE001
                log.warning("could not load batch state %s: %s", state_file, exc)

        log.info("startup_scan: loaded %d batch(es)", len(self._batches))

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create(
        self,
        batch_id:   str,
        project:    str,
        episode_id: str,
        top_n:      int,
        backgrounds: dict,
    ) -> None:
        """
        Create a new batch directory + batch_state.json.

        backgrounds: dict of {item_id: {search_prompt, ai_prompt, prefer, ...}}
        """
        bd = self.batch_dir(project, episode_id, batch_id)
        bd.mkdir(parents=True, exist_ok=True)

        items: dict[str, dict] = {}
        for item_id, item_data in backgrounds.items():
            items[item_id] = {
                **item_data,        # preserve ALL manifest fields (scoring_hints,
                                    # search_queries, cinematic_role, continuity_hints, …)
                "status":        "pending",
                "error":         None,
                "images_ranked": [],
                "videos_ranked": [],
            }

        now = _now_iso()
        state = {
            "batch_id":     batch_id,
            "status":       "queued",
            "project":      project,
            "episode_id":   episode_id,
            "top_n":        top_n,
            "created_at":   now,
            "updated_at":   now,
            "completed_at": None,
            "progress":     "queued",
            "error":        None,
            "items":        items,
        }

        self._write_atomic(bd / "batch_state.json", state)
        self._batches[batch_id] = state
        log.info("created batch %s  project=%s  episode=%s  items=%d",
                 batch_id, project, episode_id, len(items))

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, batch_id: str, **fields) -> None:
        """Update top-level batch fields and persist atomically."""
        state = self._batches[batch_id]
        state.update(fields)
        state["updated_at"] = _now_iso()
        self._write_atomic(self._state_path(batch_id), state)

    def update_item(
        self,
        batch_id:      str,
        item_id:       str,
        *,
        status:        str,
        images_ranked: list | None = None,
        videos_ranked: list | None = None,
        error:         str | None  = None,
    ) -> None:
        """Update a single per-item record and persist the full batch state."""
        state = self._batches[batch_id]
        item  = state["items"][item_id]

        item["status"] = status
        item["error"]  = error
        if images_ranked is not None:
            item["images_ranked"] = images_ranked
        if videos_ranked is not None:
            item["videos_ranked"] = videos_ranked

        state["updated_at"] = _now_iso()
        self._write_atomic(self._state_path(batch_id), state)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, batch_id: str) -> dict | None:
        data = self._batches.get(batch_id)
        return dict(data) if data is not None else None

    def list_for_episode(self, project: str, episode_id: str) -> list[dict]:
        """
        Return summary dicts for all batches belonging to project/episode_id,
        newest first (by created_at).
        """
        batches = [
            b for b in self._batches.values()
            if b.get("project") == project and b.get("episode_id") == episode_id
        ]
        batches.sort(key=lambda b: b.get("created_at", ""), reverse=True)
        return [_summary(b) for b in batches]

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def batch_dir(self, project: str, episode_id: str, batch_id: str) -> Path:
        return (
            self._root
            / project
            / "episodes"
            / episode_id
            / "assets"
            / "media"
            / batch_id
        )

    def _state_path(self, batch_id: str) -> Path:
        s = self._batches[batch_id]
        return self.batch_dir(s["project"], s["episode_id"], batch_id) / "batch_state.json"

    # ------------------------------------------------------------------
    # Atomic write
    # ------------------------------------------------------------------

    def _write_atomic(self, path: Path, data: dict) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _summary(b: dict) -> dict:
    """Lightweight summary for GET /batches list endpoint."""
    return {
        "batch_id":    b["batch_id"],
        "status":      b["status"],
        "item_count":  len(b.get("items", {})),
        "progress":    b.get("progress", ""),
        "created_at":  b.get("created_at", ""),
    }
