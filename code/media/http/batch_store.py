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
                    data["status"]     = "interrupted"
                    data["error"]      = "server_restarted"
                    data["updated_at"] = _now_iso()
                    self._write_atomic(state_file, data)
                    items_done = sum(
                        1 for it in data.get("items", {}).values()
                        if it.get("status") == "done"
                    )
                    log.info("marked interrupted batch %s (%d items done)", bid, items_done)

                # Prune ranked entries whose files no longer exist on disk.
                # Paths in images_ranked/videos_ranked are relative to the batch directory
                # (the parent of batch_state.json).
                batch_dir = state_file.parent
                pruned = False
                for item in data.get("items", {}).values():
                    for key in ("images_ranked", "videos_ranked"):
                        original = item.get(key)
                        if not original:
                            continue
                        kept = [
                            e for e in original
                            if e.get("path") and (batch_dir / e["path"]).exists()
                        ]
                        if len(kept) != len(original):
                            removed = len(original) - len(kept)
                            log.info(
                                "startup_scan: pruned %d missing %s entr%s from batch %s",
                                removed, key, "y" if removed == 1 else "ies", bid,
                            )
                            item[key] = kept
                            pruned = True
                if pruned:
                    self._write_atomic(state_file, data)

                self._batches[bid] = data

            except Exception as exc:  # noqa: BLE001
                log.warning("could not load batch state %s: %s", state_file, exc)

        log.info("startup_scan: loaded %d batch(es)", len(self._batches))

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create(
        self,
        batch_id:                str,
        project:                 str,
        episode_id:              str,
        top_n:                   int,
        backgrounds:             dict,
        content_profile:         str = "default",
        n_img:                   int | None = None,
        n_vid:                   int | None = None,
        sources_override:        list | None = None,
        source_limits_override:  dict | None = None,
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
            "batch_id":               batch_id,
            "status":                 "queued",
            "project":                project,
            "episode_id":             episode_id,
            "top_n":                  top_n,
            "content_profile":        content_profile,
            "n_img":                  n_img,
            "n_vid":                  n_vid,
            "sources_override":       sources_override,
            "source_limits_override": source_limits_override,
            "created_at":             now,
            "updated_at":      now,
            "completed_at":    None,
            "progress":        "queued",
            "error":           None,
            "items":           items,
        }

        self._write_atomic(bd / "batch_state.json", state)
        self._batches[batch_id] = state
        log.info("created batch %s  project=%s  episode=%s  items=%d",
                 batch_id, project, episode_id, len(items))

    def get_content_profile(self, batch_id: str) -> str:
        """Return the content_profile stored for this batch, defaulting to 'default'."""
        state = self._batches.get(batch_id) or {}
        return state.get("content_profile", "default")

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

    def update_item_progress(
        self,
        batch_id:        str,
        item_id:         str,
        *,
        phase:           str,
        imgs_downloaded: int | None = None,
        vids_downloaded: int | None = None,
        imgs_scored:     int | None = None,
        vids_scored:     int | None = None,
    ) -> None:
        """
        Write lightweight mid-processing progress for a single item.
        Called at phase transitions: 'downloading' → 'scoring' → (update_item 'done').
        """
        state = self._batches[batch_id]
        item  = state["items"][item_id]
        item["phase"] = phase
        if imgs_downloaded is not None:
            item["imgs_downloaded"] = imgs_downloaded
        if vids_downloaded is not None:
            item["vids_downloaded"] = vids_downloaded
        if imgs_scored is not None:
            item["imgs_scored"] = imgs_scored
        if vids_scored is not None:
            item["vids_scored"] = vids_scored
        state["updated_at"] = _now_iso()
        self._write_atomic(self._state_path(batch_id), state)

    def patch(self, batch_id: str, **fields) -> None:
        """Overwrite arbitrary top-level fields in the batch state (e.g. source_limits_override)."""
        state = self._batches[batch_id]
        for k, v in fields.items():
            state[k] = v
        state["updated_at"] = _now_iso()
        self._write_atomic(self._state_path(batch_id), state)

    def resume(self, batch_id: str) -> int:
        """Reset non-done items to pending and mark batch as queued. Returns count to re-run."""
        state = self._batches[batch_id]
        reset = 0
        for item in state["items"].values():
            if item.get("status") != "done":
                item["status"] = "pending"
                item["phase"]  = ""
                item["error"]  = None
                reset += 1
        state["status"]   = "queued"
        state["progress"] = "resuming"
        state["error"]    = None
        state["updated_at"] = _now_iso()
        self._write_atomic(self._state_path(batch_id), state)
        return reset

    def delete(self, batch_id: str) -> None:
        """Remove batch from memory and delete its state file from disk.
        Does NOT delete downloaded media files — caller handles rmtree if needed."""
        state = self._batches.pop(batch_id, None)
        if state is None:
            return
        try:
            path = (
                self.batch_dir(state["project"], state["episode_id"], batch_id)
                / "batch_state.json"
            )
            path.unlink(missing_ok=True)
        except Exception as _del_exc:  # noqa: BLE001
            log.warning("batch_store.delete: could not remove state file for %s: %s",
                        batch_id, _del_exc)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, batch_id: str) -> dict | None:
        data = self._batches.get(batch_id)
        if data is None:
            return None
        # For completed batches, re-read from disk so external edits
        # (e.g. recovered candidates patched into batch_state.json) are
        # picked up without requiring a server restart.
        if data.get("status") in ("done", "failed"):
            try:
                state_file = self._state_path(batch_id)
                fresh = json.loads(state_file.read_text(encoding="utf-8"))
                self._batches[batch_id] = fresh
                return dict(fresh)
            except Exception:  # noqa: BLE001
                pass
        return dict(data)

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
    items = b.get("items", {})
    items_done = sum(1 for it in items.values() if it.get("status") == "done")
    return {
        "batch_id":    b["batch_id"],
        "status":      b["status"],
        "item_count":  len(items),
        "items_done":  items_done,
        "progress":    b.get("progress", ""),
        "created_at":  b.get("created_at", ""),
    }
