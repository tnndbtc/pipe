"""
JobStore — persistent job state backed by per-job state.json files.
"""
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, job_dir_override: str | None = None):
        base_root = Path(job_dir_override) if job_dir_override else Path(tempfile.gettempdir())
        self.base = base_root / "ai-jobs"
        self._jobs: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup_scan(self) -> None:
        """Load existing jobs; prune old ones; mark interrupted jobs failed."""
        self.base.mkdir(parents=True, exist_ok=True)
        cutoff = datetime.now(timezone.utc).timestamp() - 86400  # 24 h

        for state_file in self.base.glob("*/state.json"):
            try:
                if state_file.stat().st_mtime < cutoff:
                    # Prune entire job dir
                    import shutil
                    shutil.rmtree(state_file.parent, ignore_errors=True)
                    continue

                data = json.loads(state_file.read_text(encoding="utf-8"))
                job_id = data.get("job_id")
                if not job_id:
                    continue

                if data.get("status") == "running":
                    data["status"] = "failed"
                    data["error"] = "Server restarted while job was running"
                    data["updated_at"] = _now_iso()
                    self._write_atomic(state_file, data)

                self._jobs[job_id] = data
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def job_path(self, job_id: str) -> Path:
        return self.base / job_id

    def out_path(self, job_id: str) -> Path:
        return self.base / job_id / "out"

    def log_path(self, job_id: str) -> Path:
        return self.base / job_id / "log.txt"

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(self, job_id: str, total: int) -> dict:
        job_dir = self.job_path(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        self.out_path(job_id).mkdir(parents=True, exist_ok=True)

        now = _now_iso()
        state = {
            "job_id":     job_id,
            "status":     "queued",
            "total":      total,
            "done":       0,
            "files":      [],
            "errors":     [],
            "created_at": now,
            "updated_at": now,
        }
        self._write_atomic(job_dir / "state.json", state)
        self._jobs[job_id] = state
        return state

    def update(self, job_id: str, **fields) -> dict:
        state = self._jobs[job_id]
        state.update(fields)
        state["updated_at"] = _now_iso()
        self._write_atomic(self.job_path(job_id) / "state.json", state)
        return state

    def get(self, job_id: str) -> dict | None:
        data = self._jobs.get(job_id)
        return dict(data) if data is not None else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_atomic(self, path: Path, data: dict) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, path)
