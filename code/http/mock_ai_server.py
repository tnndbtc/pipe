#!/usr/bin/env python3
"""
mock_ai_server.py — Minimal stub that mimics code/ai/http/server.py for
testing fetch_ai_assets.py without a real GPU machine.

Jobs complete instantly and return 1-pixel placeholder files so the full
submit → poll → download → save → index.json flow can be exercised locally.

Usage (terminal 1):
    python3 code/http/mock_ai_server.py

Usage (terminal 2):
    AI_SERVER_URL=http://127.0.0.1:8001 AI_SERVER_KEY=change-me \\
        python3 code/http/fetch_ai_assets.py \\
            --manifest projects/the-pharaoh-who-defied-death/episodes/s01e02/AssetManifest_draft.shared.json \\
            --asset_type characters

    # Single item:
    AI_SERVER_URL=http://127.0.0.1:8001 AI_SERVER_KEY=change-me \\
        python3 code/http/fetch_ai_assets.py \\
            --manifest projects/the-pharaoh-who-defied-death/episodes/s01e02/AssetManifest_draft.shared.json \\
            --asset_type characters --asset-id tahk

    # SFX (42 entries, 19 unique shots):
    AI_SERVER_URL=http://127.0.0.1:8001 AI_SERVER_KEY=change-me \\
        python3 code/http/fetch_ai_assets.py \\
            --manifest projects/the-pharaoh-who-defied-death/episodes/s01e02/AssetManifest_draft.shared.json \\
            --asset_type sfx

    # Wrong key → 401:
    AI_SERVER_URL=http://127.0.0.1:8001 AI_SERVER_KEY=wrong \\
        python3 code/http/fetch_ai_assets.py \\
            --manifest projects/the-pharaoh-who-defied-death/episodes/s01e02/AssetManifest_draft.shared.json \\
            --asset_type characters
"""

import json
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------------------------------------------------------------------------
# Config (must match AI_SERVER_KEY in fetch_ai_assets.py / env var)
# ---------------------------------------------------------------------------

PORT    = 8001
API_KEY = "change-me"
OFFLINE = False   # set to True (or pass --offline) to simulate server offline mode

# ---------------------------------------------------------------------------
# Minimal valid placeholder file bytes
# ---------------------------------------------------------------------------

# 1×1 white PNG
TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
    b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)

# Minimal valid WAV (44-byte header, no audio samples)
TINY_WAV = (
    b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
    b"\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
)

# Minimal valid MP4 (ftyp box only — opens without error in most players)
TINY_MP4 = (
    b"\x00\x00\x00\x14ftypmp42\x00\x00\x00\x00mp42isom"
)

PLACEHOLDER: dict[str, bytes] = {
    ".png": TINY_PNG,
    ".wav": TINY_WAV,
    ".mp4": TINY_MP4,
}

# ---------------------------------------------------------------------------
# Mirrors SECTION_KEY / ID_FIELD / OUTPUT_EXT from fetch_ai_assets.py
# ---------------------------------------------------------------------------

SECTION_KEY = {
    "characters":  "character_packs",
    "backgrounds": "backgrounds",
    "bg_video":    "backgrounds",
    "sfx":         "sfx_items",
}
ID_FIELD = {
    "characters":  "asset_id",
    "backgrounds": "asset_id",
    "bg_video":    "asset_id",
    "sfx":         "shot_id",
}
OUTPUT_EXT = {
    "characters":  ".png",
    "backgrounds": ".png",
    "bg_video":    ".mp4",
    "sfx":         ".wav",
}

# In-memory job store
_jobs: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args) -> None:  # replace default with concise log
        print(f"  [{self.command}] {self.path}")

    # ── auth ────────────────────────────────────────────────────────────────

    def _check_auth(self) -> bool:
        if self.headers.get("X-Api-Key", "") != API_KEY:
            self._send_json(401, {"detail": "Invalid API key"})
            return False
        return True

    # ── response helpers ────────────────────────────────────────────────────

    def _send_json(self, code: int, data: dict) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data: bytes, filename: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    # ── routes ──────────────────────────────────────────────────────────────

    def do_GET(self) -> None:
        if not self._check_auth():
            return

        # GET /health
        if self.path == "/health":
            if OFFLINE:
                self._send_json(200, {"status": "offline", "gpu": "N/A", "vram_free_gb": 0, "queue_len": 0})
            else:
                self._send_json(200, {"status": "ok", "gpu": "Mock GPU (no VRAM needed)", "vram_free_gb": 8.0, "queue_len": 0})
            return

        parts = self.path.strip("/").split("/")

        # GET /jobs/{job_id}
        if len(parts) == 2 and parts[0] == "jobs":
            job_id = parts[1]
            job = _jobs.get(job_id)
            if job is None:
                self._send_json(404, {"detail": "Job not found"})
                return
            self._send_json(200, job)
            return

        # GET /jobs/{job_id}/files/{filename}
        if len(parts) == 4 and parts[0] == "jobs" and parts[2] == "files":
            job_id, filename = parts[1], parts[3]
            job = _jobs.get(job_id)
            if job is None or filename not in job["files"]:
                self._send_json(404, {"detail": "File not found"})
                return
            ext  = "." + filename.rsplit(".", 1)[-1]
            data = PLACEHOLDER.get(ext, TINY_PNG)
            self._send_bytes(data, filename)
            return

        self._send_json(404, {"detail": "Not found"})

    def do_POST(self) -> None:
        if not self._check_auth():
            return

        # POST /jobs
        if self.path == "/jobs":
            if OFFLINE:
                self._send_json(200, {"status": "unavailable", "job_id": None, "total": 0})
                return

            body        = self._read_json()
            manifest    = body.get("manifest", {})
            asset_types = body.get("asset_types", [])

            if not asset_types:
                self._send_json(400, {"detail": "asset_types required"})
                return

            asset_type  = asset_types[0]
            section_key = SECTION_KEY.get(asset_type)
            id_field    = ID_FIELD.get(asset_type, "asset_id")
            ext         = OUTPUT_EXT.get(asset_type, ".bin")

            if section_key is None:
                self._send_json(400, {"detail": f"Unknown asset_type: {asset_type}"})
                return

            entries = manifest.get(section_key, [])

            # Build output filenames that match what the real gen scripts produce.
            # For sfx: sfx_{shot_id}_mock{i}.wav
            #   rsplit("_", 1) correctly extracts shot_id since "mock{i}" has no "_".
            # For others: {asset_id}.{ext}
            files = []
            for i, entry in enumerate(entries):
                eid = entry.get(id_field, f"unknown-{i}")
                if asset_type == "sfx":
                    files.append(f"sfx_{eid}_mock{i}{ext}")
                else:
                    files.append(f"{eid}{ext}")

            job_id = str(uuid.uuid4())
            total  = len(entries)
            now    = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            _jobs[job_id] = {
                "job_id":     job_id,
                "status":     "done",   # instant completion — no real GPU work
                "total":      total,
                "done":       total,
                "files":      files,
                "errors":     [],
                "log_tail":   [f"[mock] {asset_type}: {total} item(s) generated"],
                "created_at": now,
                "updated_at": now,
            }

            print(f"  [JOB]  {job_id}  {asset_type}  {total} items → {files}")
            self._send_json(200, {"job_id": job_id, "status": "queued", "total": total})
            return

        self._send_json(404, {"detail": "Not found"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args   = [a for a in sys.argv[1:] if a != "--offline"]
    OFFLINE = "--offline" in sys.argv  # noqa: F811
    port   = int(args[0]) if args else PORT
    print(f"Mock AI server  →  http://127.0.0.1:{port}")
    print(f"API key         →  {API_KEY!r}")
    print(f"Offline mode    →  {OFFLINE}")
    print(f"Press Ctrl-C to stop.\n")
    try:
        HTTPServer(("127.0.0.1", port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
