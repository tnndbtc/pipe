"""
job_queue.py — Server-side job queue for distributed video scoring.

Wraps asyncio.Queue with batch_id tagging, in-flight tracking, result
collection, stale-job reaping, per-worker path remapping, and a
background reaper task.

Used by server.py to distribute video scoring work to remote workers.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("job_queue")


# ---------------------------------------------------------------------------
# Registered worker info
# ---------------------------------------------------------------------------

@dataclass
class WorkerInfo:
    name: str
    hostname: str
    nfs_root: str            # worker's local NFS mount path
    last_seen: float = 0.0   # time.monotonic() of last activity


# ---------------------------------------------------------------------------
# In-flight job tracking
# ---------------------------------------------------------------------------

@dataclass
class InFlightJob:
    job_id: str
    task: dict               # original task payload (server-side paths)
    worker: str              # name of the worker that grabbed it
    grabbed_at: float        # time.monotonic()


# ---------------------------------------------------------------------------
# JobQueue
# ---------------------------------------------------------------------------

class JobQueue:
    """
    In-memory job queue for one batch at a time.

    Lifecycle:
        jq = JobQueue(server_nfs_root="/data/shared")
        jq.enqueue(batch_id, tasks)
        jq.start_reaper(timeout_seconds=120)
        ...workers poll next_job / submit_result...
        await jq.wait_until_done()
        jq.stop_reaper()
    """

    def __init__(self, server_nfs_root: str = "/data/shared") -> None:
        self.server_nfs_root = server_nfs_root

        self._queue: asyncio.Queue[dict] = asyncio.Queue()
        self._batch_id: str | None = None
        self._total: int = 0
        self._in_flight: dict[str, InFlightJob] = {}   # job_id → InFlightJob
        self._results: dict[str, dict] = {}             # job_id → result dict
        self._done_event: asyncio.Event = asyncio.Event()
        self._reaper_task: asyncio.Task | None = None

        # Registered workers
        self._workers: dict[str, WorkerInfo] = {}       # name → WorkerInfo

    # ------------------------------------------------------------------
    # Worker registration
    # ------------------------------------------------------------------

    def register_worker(self, name: str, hostname: str, nfs_root: str) -> None:
        """Register (or re-register) a worker."""
        self._workers[name] = WorkerInfo(
            name=name, hostname=hostname,
            nfs_root=nfs_root, last_seen=time.monotonic(),
        )
        log.info("Worker registered: %s (host=%s, nfs=%s)", name, hostname, nfs_root)

    def get_workers(self) -> list[dict]:
        """Return worker info dicts for monitoring."""
        now = time.monotonic()
        return [
            {
                "name": w.name,
                "hostname": w.hostname,
                "nfs_root": w.nfs_root,
                "idle_seconds": round(now - w.last_seen, 1),
            }
            for w in self._workers.values()
        ]

    @property
    def worker_count(self) -> int:
        return len(self._workers)

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------

    def enqueue(self, batch_id: str, tasks: list[dict]) -> None:
        """
        Bulk-enqueue video scoring tasks for a new batch.

        Each task dict should contain at minimum:
            video_path, frames_dir, item, config
        (all paths as server-side strings).

        Calling enqueue() discards any prior batch state.
        """
        # Reset state
        self._queue = asyncio.Queue()
        self._batch_id = batch_id
        self._in_flight.clear()
        self._results.clear()
        self._done_event.clear()
        self._total = len(tasks)

        for t in tasks:
            job_id = "j_" + uuid.uuid4().hex[:8]
            t["job_id"] = job_id
            t["batch_id"] = batch_id
            self._queue.put_nowait(t)

        log.info("Enqueued %d jobs for batch %s", len(tasks), batch_id)

    # ------------------------------------------------------------------
    # Job dispatch
    # ------------------------------------------------------------------

    def next_job(self, worker_name: str) -> dict | None:
        """
        Pop the next task from the queue for the given worker.

        Returns the task dict with paths remapped to the worker's nfs_root,
        or None if the queue is empty.
        Marks the job as in-flight.
        """
        if self._queue.empty():
            return None

        try:
            task = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

        job_id = task["job_id"]
        self._in_flight[job_id] = InFlightJob(
            job_id=job_id, task=task,
            worker=worker_name, grabbed_at=time.monotonic(),
        )

        # Update worker last-seen
        if worker_name in self._workers:
            self._workers[worker_name].last_seen = time.monotonic()

        # Remap paths for this worker
        worker_info = self._workers.get(worker_name)
        nfs_root = worker_info.nfs_root if worker_info else "/mnt/shared"
        payload = dict(task)
        payload["video_path"] = self.remap_path(payload["video_path"], nfs_root)
        payload["frames_dir"] = self.remap_path(payload["frames_dir"], nfs_root)

        log.debug("Dispatched job %s to worker %s", job_id, worker_name)
        return payload

    # ------------------------------------------------------------------
    # Result submission
    # ------------------------------------------------------------------

    def submit_result(self, job_id: str, result: dict) -> bool:
        """
        Store the result for a completed job.

        Returns True if accepted, False if the job was already completed
        (late/duplicate result — ignored silently).
        """
        if job_id in self._results:
            log.debug("Ignoring late/duplicate result for job %s", job_id)
            return False

        self._results[job_id] = result
        self._in_flight.pop(job_id, None)

        log.debug("Result received for job %s  (%d/%d done)",
                  job_id, len(self._results), self._total)

        if self.all_done():
            self._done_event.set()

        return True

    # ------------------------------------------------------------------
    # Completion tracking
    # ------------------------------------------------------------------

    def all_done(self) -> bool:
        """True when all enqueued tasks have results."""
        return self._total > 0 and len(self._results) >= self._total

    async def wait_until_done(self) -> None:
        """Block until all jobs are completed."""
        if self.all_done():
            return
        await self._done_event.wait()

    def get_results(self) -> list[dict]:
        """Return all collected results (unordered)."""
        return list(self._results.values())

    @property
    def batch_id(self) -> str | None:
        return self._batch_id

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    @property
    def in_flight_count(self) -> int:
        return len(self._in_flight)

    @property
    def completed_count(self) -> int:
        return len(self._results)

    @property
    def total_count(self) -> int:
        return self._total

    # ------------------------------------------------------------------
    # Stale job reaping
    # ------------------------------------------------------------------

    def requeue_stale(self, timeout_seconds: int = 120) -> int:
        """
        Re-queue jobs grabbed by workers that haven't reported back
        within timeout_seconds.

        Returns the number of re-queued jobs.
        """
        now = time.monotonic()
        stale_ids = [
            jid for jid, inf in self._in_flight.items()
            if (now - inf.grabbed_at) > timeout_seconds
        ]

        for jid in stale_ids:
            inf = self._in_flight.pop(jid)
            # Put the original task (server-side paths) back in the queue
            self._queue.put_nowait(inf.task)
            log.warning(
                "Re-queued stale job %s (worker=%s, age=%.0fs)",
                jid, inf.worker, now - inf.grabbed_at,
            )

        return len(stale_ids)

    # ------------------------------------------------------------------
    # Background reaper
    # ------------------------------------------------------------------

    def start_reaper(self, timeout_seconds: int = 120, interval: float = 30.0) -> None:
        """Launch background task that calls requeue_stale() periodically."""
        if self._reaper_task is not None:
            self._reaper_task.cancel()

        async def _reaper_loop():
            try:
                while True:
                    await asyncio.sleep(interval)
                    n = self.requeue_stale(timeout_seconds)
                    if n:
                        log.info("Reaper re-queued %d stale jobs", n)
            except asyncio.CancelledError:
                pass

        self._reaper_task = asyncio.create_task(_reaper_loop())
        log.debug("Reaper started (timeout=%ds, interval=%.0fs)", timeout_seconds, interval)

    def stop_reaper(self) -> None:
        """Cancel the background reaper task."""
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            self._reaper_task = None
            log.debug("Reaper stopped")

    # ------------------------------------------------------------------
    # Path remapping
    # ------------------------------------------------------------------

    def remap_path(self, path: str, worker_nfs_root: str) -> str:
        """
        Replace the server_nfs_root prefix with the worker's nfs_root.

        Example:
            server_nfs_root = "/data/shared"
            worker_nfs_root = "/mnt/shared"
            path = "/data/shared/proj/ep/video.mp4"
            → "/mnt/shared/proj/ep/video.mp4"
        """
        if path.startswith(self.server_nfs_root):
            return worker_nfs_root + path[len(self.server_nfs_root):]
        return path
