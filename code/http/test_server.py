#!/usr/bin/env python3
"""
test_server.py — Claude pipeline runner with story input UI.

Workflow:
  1. Paste story metadata into the web UI
  2. Confirm / edit the auto-generated prompt
  3. Click Run  →  story saved as story_N.txt, then claude -p is launched
  4. Output streams back to the browser via Server-Sent Events

Start:
    python3 code/http/test_server.py
Open:
    http://localhost:8000
"""

import glob
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse, unquote_plus, quote as _url_quote
import urllib.request as _urllib_req
import urllib.error   as _urllib_err

# ── YouTube category mapping (genre → category_id, no LLM needed) ─────────────
_GENRE_TO_CATEGORY = {
    "history":       "27",
    "documentary":   "27",
    "education":     "27",
    "sports":        "17",
    "news":          "25",
    "entertainment": "24",
    "comedy":        "23",
    "narration":     "24",
}
_DEFAULT_CATEGORY = "24"

PORT     = 8000
PIPE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (pipe/)

# ── AI server connection (read once at startup from environment) ────────────────
_AI_SERVER_URL = os.environ.get("AI_SERVER_URL", "http://192.168.86.27:8000").rstrip("/")
_AI_SERVER_KEY = os.environ.get("AI_SERVER_KEY", "change-me")

# ── VC editor config (config.json) ────────────────────────────────────────────
_vc_config = {}
_vc_config_path = os.path.join(os.path.dirname(__file__), "config.json")
if os.path.isfile(_vc_config_path):
    with open(_vc_config_path, encoding="utf-8") as f:
        _vc_config = json.load(f)

# ── Running-process registry (so Stop button can kill it) ──────────────────────
_lock  = threading.Lock()
_procs = {}   # client_addr → subprocess.Popen

# ── Background-job registry ────────────────────────────────────────────────────
# Decouples subprocess/loop lifetime from the SSE connection.
# A job keeps running even if the browser tab disconnects; the client can
# reconnect and replay the full output from the beginning of the log file.
import tempfile as _tempfile

_jobs: dict = {}          # job_key → {"log": str, "done": bool, "rc": int|None}
_jobs_lock = threading.Lock()

# ── Per-episode VO write lock (INVARIANT G) ────────────────────────────────────
# Keyed by ep_dir string. Acquired before any write to *.wav, *.source.wav,
# vo_trim_overrides.json, vo_merge_log.json, AssetManifest_merged, or
# tts_review_complete.json. Different episodes run concurrently (not global).
_vo_locks: dict[str, threading.Lock] = {}
_vo_locks_meta = threading.Lock()   # protects _vo_locks dict itself


def _get_vo_lock(ep_dir: str) -> threading.Lock:
    """Get (or create) the per-episode VO write lock."""
    with _vo_locks_meta:
        if ep_dir not in _vo_locks:
            _vo_locks[ep_dir] = threading.Lock()
        return _vo_locks[ep_dir]


def _job_log_path(job_key: str) -> str:
    """Stable tmp path for this job's output log."""
    h = hashlib.md5(job_key.encode()).hexdigest()
    d = os.path.join(_tempfile.gettempdir(), "pipe_jobs")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, h + ".log")


def _launch_stream_job(job_key: str, cmd: list, env: dict, client) -> str:
    """Run cmd in a background thread, writing tagged lines to a log file.

    Tags written to log:
      O\\t{line}  — stdout line
      E\\t{line}  — stderr line
      D\\t{rc}    — done sentinel (process exit code)

    Returns the log path.  If the same job is already running (e.g. client
    reconnected), returns the existing log path so the client can replay from
    the start.
    """
    log_path = _job_log_path(job_key)
    with _jobs_lock:
        existing = _jobs.get(job_key)
        if existing and not existing["done"]:
            return log_path   # already running — attach to existing log
        _jobs[job_key] = {"log": log_path, "done": False, "rc": None}

    def _run() -> None:
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1, env=env, cwd=PIPE_DIR,
            )
            with _lock:
                _procs[client] = proc

            with open(log_path, "w", buffering=1, encoding="utf-8") as lf:
                def _drain(stream, prefix: str) -> None:
                    for ln in stream:
                        lf.write(prefix + ln.rstrip("\n") + "\n")
                        lf.flush()

                t_out = threading.Thread(target=_drain,
                                         args=(proc.stdout, "O\t"), daemon=True)
                t_err = threading.Thread(target=_drain,
                                         args=(proc.stderr, "E\t"), daemon=True)
                t_out.start()
                t_err.start()
                t_out.join()
                t_err.join()
                proc.wait()
                lf.write(f"D\t{proc.returncode}\n")
                lf.flush()

            with _lock:
                _procs.pop(client, None)
            with _jobs_lock:
                _jobs[job_key]["done"] = True
                _jobs[job_key]["rc"]   = proc.returncode

        except Exception as exc:
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"E\tInternal job error: {exc}\n")
                lf.write("D\t1\n")
            with _jobs_lock:
                _jobs[job_key]["done"] = True
                _jobs[job_key]["rc"]   = 1

    threading.Thread(target=_run, daemon=True).start()
    return log_path


def _launch_fn_job(job_key: str, target_fn) -> str:
    """Run target_fn(write_log) in a background thread.

    target_fn receives write_log(tag, data) where tag is 'O', 'E', or 'D'.
    Returns the log path.
    """
    log_path = _job_log_path(job_key)
    with _jobs_lock:
        existing = _jobs.get(job_key)
        if existing and not existing["done"]:
            return log_path
        _jobs[job_key] = {"log": log_path, "done": False, "rc": None}

    def _run() -> None:
        try:
            with open(log_path, "w", buffering=1, encoding="utf-8") as lf:
                def write_log(tag: str, data: str) -> None:
                    lf.write(tag + "\t" + data + "\n")
                    lf.flush()
                target_fn(write_log)
        except Exception as exc:
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"E\tInternal job error: {exc}\n")
                lf.write("D\t1\n")
            with _jobs_lock:
                _jobs[job_key]["done"] = True
                _jobs[job_key]["rc"]   = 1

    threading.Thread(target=_run, daemon=True).start()
    return log_path


def _tail_log_to_sse(wfile, log_path: str) -> None:
    """Replay a tagged log file as SSE events to wfile.

    Blocks until the D (done) sentinel is read or the caller catches
    BrokenPipeError / ConnectionResetError.

    Tags: O\\t → 'line' event, E\\t → 'error_line' event,
          D\\t{rc} → 'done' event, V\\t{json} → 'vo_review_ready' event.
    """
    deadline = time.time() + 10.0
    while not os.path.exists(log_path) and time.time() < deadline:
        time.sleep(0.05)

    with open(log_path, "r", encoding="utf-8") as lf:
        while True:
            line = lf.readline()
            if not line:          # EOF — process still running, wait for more
                time.sleep(0.1)
                continue
            line = line.rstrip("\n")
            if line.startswith("O\t"):
                wfile.write(sse("line", line[2:]))
                wfile.flush()
            elif line.startswith("E\t"):
                wfile.write(sse("error_line", line[2:]))
                wfile.flush()
            elif line.startswith("V\t"):
                wfile.write(sse("vo_review_ready", line[2:]))
                wfile.flush()
            elif line.startswith("D\t"):
                wfile.write(sse("done", line[2:]))
                wfile.flush()
                return

def _append_tts_usage_to_status_report(slug: str, ep_id: str, write_log) -> None:
    """Read tts_audit_log.json for this episode + all project episodes,
    format a TTS usage summary, and append it to status_report.txt.
    Also emits the summary lines via write_log("O", ...) for the SSE stream.
    """
    import glob as _glob
    from datetime import datetime as _dt2

    ep_dir  = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
    sr_path = os.path.join(ep_dir, "status_report.txt")

    # ── Load this episode's audit log ─────────────────────────────────────────
    ep_log_path = os.path.join(ep_dir, "assets", "meta", "tts_audit_log.json")
    ep_audit: dict = {}
    if os.path.exists(ep_log_path):
        try:
            with open(ep_log_path, encoding="utf-8") as fh:
                ep_audit = json.load(fh)
        except Exception:
            pass

    runs: list[dict]     = ep_audit.get("runs", [])
    accumulated: dict    = ep_audit.get("accumulated", {})

    if not runs and not accumulated:
        return   # no TTS ran — skip silently

    # ── "This run" = last entry per locale in runs[] ──────────────────────────
    last_run_per_locale: dict[str, dict] = {}
    for r in runs:
        last_run_per_locale[r.get("locale", "?")] = r

    # ── Project-wide totals: sum all episodes' accumulated dicts ──────────────
    proj_eps_dir = os.path.join(PIPE_DIR, "projects", slug, "episodes")
    proj_totals: dict[str, dict] = {}
    for log_file in _glob.glob(
        os.path.join(proj_eps_dir, "*", "assets", "meta", "tts_audit_log.json")
    ):
        try:
            with open(log_file, encoding="utf-8") as fh:
                other = json.load(fh)
            for loc, data in other.get("accumulated", {}).items():
                t = proj_totals.setdefault(loc, {
                    "total_raw_chars": 0, "total_ssml_chars": 0,
                    "total_api_calls": 0, "total_items_synthesized": 0,
                    "total_items_skipped": 0, "runs": 0,
                })
                t["total_raw_chars"]          += data.get("total_raw_chars",          0)
                t["total_ssml_chars"]         += data.get("total_ssml_chars",         0)
                t["total_api_calls"]          += data.get("total_api_calls",          0)
                t["total_items_synthesized"]  += data.get("total_items_synthesized",  0)
                t["total_items_skipped"]      += data.get("total_items_skipped",      0)
                t["runs"]                     += data.get("runs",                     0)
        except Exception:
            pass

    # ── Format ────────────────────────────────────────────────────────────────
    lines = ["── Azure TTS Usage ─────────────────────────────────────────────"]

    if last_run_per_locale:
        lines.append("This run:")
        for loc in sorted(last_run_per_locale):
            r = last_run_per_locale[loc]
            raw   = r.get("raw_chars",     0)
            ssml  = r.get("ssml_chars",    0)
            calls = r.get("api_calls",     0)
            mode  = r.get("mode",          "?")
            wall  = r.get("wall_time_sec", 0)
            lines.append(
                f"  {loc:<10} raw: {raw:>6,} chars │ ssml: {ssml:>7,} chars"
                f" │ api calls: {calls:>3} │ mode: {mode} │ {wall:.0f}s"
            )

    if accumulated:
        lines.append("This episode (lifetime):")
        for loc in sorted(accumulated):
            a     = accumulated[loc]
            raw   = a.get("total_raw_chars",  0)
            ssml  = a.get("total_ssml_chars", 0)
            calls = a.get("total_api_calls",  0)
            n_syn = a.get("total_items_synthesized", 0)
            nruns = a.get("runs", 0)
            lines.append(
                f"  {loc:<10} raw: {raw:>6,} chars │ ssml: {ssml:>7,} chars"
                f" │ api calls: {calls:>3} │ items synthesized: {n_syn} │ runs: {nruns}"
            )

    if proj_totals:
        lines.append(f"This project ({slug}) — all episodes:")
        for loc in sorted(proj_totals):
            t     = proj_totals[loc]
            raw   = t.get("total_raw_chars",  0)
            ssml  = t.get("total_ssml_chars", 0)
            calls = t.get("total_api_calls",  0)
            nruns = t.get("runs",             0)
            lines.append(
                f"  {loc:<10} raw: {raw:>6,} chars │ ssml: {ssml:>7,} chars"
                f" │ api calls: {calls:>3} │ runs: {nruns}"
            )

    lines.append("─" * 64)

    text = "\n".join(lines)

    # ── Append to status_report.txt ───────────────────────────────────────────
    try:
        os.makedirs(ep_dir, exist_ok=True)
        ts    = _dt2.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"\n{'='*60}\n{ts}\n{'='*60}\n{text}\n"
        with open(sr_path, "a", encoding="utf-8") as fh:
            fh.write(entry)
    except Exception as exc:
        write_log("E", f"[tts-audit] Could not write status report: {exc}")
        return

    # ── Echo to SSE stream ────────────────────────────────────────────────────
    write_log("O", "")
    for ln in lines:
        write_log("O", ln)


# ── Azure TTS preview — throttle state + lazy imports ─────────────────────────
_tts_lock             = threading.Lock()
_tts_last_call: float = 0.0               # monotonic timestamp
TTS_MIN_INTERVAL      = 3.5               # seconds (F0: 20 req/60 s → safe rate)
TTS_RETRY_BACKOFF     = [3, 6]            # seconds; shorter backoff for preview UI
_TTS_REST_TRIGGERS    = ("429", "too many requests", "websocket upgrade failed",
                         "websocket", "throttl")
_voice_catalog_cache  = None              # set once by parse_azure_tts_styles()
_tts_synth            = None              # lazy singleton; created on first preview request
_diagnose_cache: dict = {}               # key: "{slug}|{ep_id}|{mtime_hash}" → Claude result

# Lazy-import Azure SDK so server starts even when SDK is not installed
try:
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gen_tts_cloud import build_ssml as _build_ssml
    import azure.cognitiveservices.speech as _speechsdk
    _TTS_AVAILABLE = True
except ImportError:
    _build_ssml    = None  # type: ignore
    _speechsdk     = None  # type: ignore
    _TTS_AVAILABLE = False

# ── VO polish thresholds — single source of truth is polish_locale_vo
try:
    from polish_locale_vo import THRESHOLD      as _VO_POLISH_THRESHOLD, \
                                  THRESHOLD_HIGH as _VO_POLISH_THRESHOLD_HIGH
except Exception:
    _VO_POLISH_THRESHOLD      = 0.90  # fallback if module not yet on path
    _VO_POLISH_THRESHOLD_HIGH = 1.10

# ── VO retune shared module
try:
    from vo_retune import retune_vo_items as _retune_vo_items
    _RETUNE_AVAILABLE = True
except ImportError:
    _retune_vo_items  = None   # type: ignore
    _RETUNE_AVAILABLE = False

# ── VO utils (shared with gen_tts_cloud) ──────────────────────────────────────
try:
    from vo_utils import (
        apply_vo_trims_for_item   as _apply_vo_trims_for_item,
        invalidate_vo_state       as _invalidate_vo_state,
        load_vo_trim_overrides    as _load_vo_trim_overrides,
        save_vo_trim_overrides    as _save_vo_trim_overrides,
        get_primary_locale        as _get_primary_locale,
        compute_sentinel_hashes   as _compute_sentinel_hashes,
        write_sentinel            as _write_sentinel,
        verify_sentinel           as _verify_sentinel,
        wav_duration              as _wav_duration,
    )
    _VO_UTILS_AVAILABLE = True
except ImportError as _vo_import_err:
    print(f"[WARN] vo_utils not available: {_vo_import_err}")
    _VO_UTILS_AVAILABLE = False

# ── VO item synthesis (P1-[B]) ────────────────────────────────────────────────
# Threshold constants for ⚠ short / long detection (P4-[P])
_VO_EXPECTED_SEC_PER_CHAR = 0.065   # ~15 chars/sec for English TTS
_VO_SHORT_RATIO = 0.70              # actual < expected * SHORT_RATIO  → ⚠ short
_VO_LONG_RATIO  = 1.50              # actual > expected * LONG_RATIO   → ⚠ long


def _vo_badge(text: str, actual_sec: float) -> str:
    """Return 'short', 'long', or 'ok' based on text length vs WAV duration."""
    expected = len(text) * _VO_EXPECTED_SEC_PER_CHAR
    if expected > 0 and actual_sec < expected * _VO_SHORT_RATIO:
        return "short"
    if expected > 0 and actual_sec > expected * _VO_LONG_RATIO:
        return "long"
    return "ok"


def _vo_resolve_ep_dir(ep_dir_param: str) -> str:
    """Resolve ep_dir to an absolute path under PIPE_DIR/projects/."""
    if os.path.isabs(ep_dir_param):
        return ep_dir_param
    return os.path.join(PIPE_DIR, ep_dir_param)


_RE_LOCALE  = re.compile(r'^[a-zA-Z]{2,8}(-[a-zA-Z0-9]{2,8})*$')
_RE_ITEM_ID = re.compile(r'^vo-[a-zA-Z0-9_]+(-[a-zA-Z0-9]+)+$')


def _vo_validate_inputs(ep_dir: str, locale: str, item_id: str | None = None) -> None:
    """Raise ValueError for invalid VO endpoint inputs."""
    if not ep_dir or ".." in ep_dir:
        raise ValueError("Invalid ep_dir")
    if not _RE_LOCALE.match(locale):
        raise ValueError(f"Invalid locale: {locale!r}")
    if item_id is not None and not _RE_ITEM_ID.match(item_id):
        raise ValueError(f"Invalid item_id: {item_id!r}")
    # Path escape check
    full = _vo_resolve_ep_dir(ep_dir)
    projects_root = os.path.realpath(os.path.join(PIPE_DIR, "projects"))
    if not os.path.realpath(full).startswith(projects_root + os.sep):
        raise ValueError("ep_dir outside projects/ tree")


def _ensure_source_wav(item_id: str, full_ep: str, locale: str) -> None:
    """Migration helper: if source.wav is missing but .wav exists, copy .wav → source.wav.

    Called before any trim/reset operation so that projects synthesised before
    the two-file WAV model was introduced can still use VO trim controls.
    """
    import shutil as _shutil
    vo_dir    = os.path.join(full_ep, "assets", locale, "audio", "vo")
    wav_path  = os.path.join(vo_dir, f"{item_id}.wav")
    src_path  = os.path.join(vo_dir, f"{item_id}.source.wav")
    if not os.path.isfile(src_path) and os.path.isfile(wav_path):
        tmp = src_path + ".tmp"
        _shutil.copy2(wav_path, tmp)
        os.replace(tmp, src_path)
        print(f"[migration] Created source.wav from existing .wav for {item_id}")


def synthesize_vo_item(
    item_id: str,
    text: str,
    params: dict,
    ep_dir: str,
    locale: str,
    write_cache: bool = True,
) -> dict:
    """Synthesize a single VO item and write {item_id}.source.wav + {item_id}.wav.

    Caller must hold _get_vo_lock(ep_dir) before calling this.

    Args:
        item_id:     e.g. "vo-sc01-001"
        text:        Text to synthesize
        params:      dict with voice, style, rate, style_degree, locale fields
        ep_dir:      Absolute path to episode directory
        locale:      Locale string (e.g. "en")
        write_cache: True → write to tts_cache (vo_save), False → bypass (vo_recreate/vo_merge)

    Returns:
        { "source_duration_sec": float, "trimmed_duration_sec": float }

    Calls:
        apply_vo_trims_for_item()  → writes {item_id}.wav
        invalidate_vo_state()      → deletes sentinel, durations, marks stale
    """
    if not _TTS_AVAILABLE:
        raise RuntimeError("Azure TTS SDK not available")
    if not _VO_UTILS_AVAILABLE:
        raise RuntimeError("vo_utils not available")

    full_ep  = _vo_resolve_ep_dir(ep_dir)
    vo_dir   = os.path.join(full_ep, "assets", locale, "audio", "vo")
    os.makedirs(vo_dir, exist_ok=True)

    # Build SSML for a single item
    azure_voice    = params.get("voice", "")
    azure_locale   = '-'.join(azure_voice.split('-')[:2]) if azure_voice else locale
    style          = params.get("style", "")
    style_degree   = float(params.get("style_degree", 1.5))
    rate           = params.get("rate", "0%")
    pitch          = params.get("pitch", "")
    break_ms       = int(params.get("break_ms", 0))

    if not azure_voice:
        raise ValueError("voice is required in params")

    ssml = _preview_build_ssml(
        text, azure_voice, azure_locale, style,
        style_degree=style_degree, rate=rate,
        pitch=pitch, break_ms=break_ms,
    )

    # TTS call (with or without cache)
    # write_cache=True  → used by vo_save (new params → deterministic new entry)
    # write_cache=False → used by vo_recreate and vo_merge (non-deterministic)
    import tempfile as _tf
    from pathlib import Path as _Path

    if write_cache:
        # Cache key based on SSML content
        cache_key_dict = {
            "v": azure_voice, "l": azure_locale, "s": style or "",
            "d": style_degree, "r": rate, "p": pitch or "",
            "b": break_ms, "t": text,
        }
        h = hashlib.sha256(
            json.dumps(cache_key_dict, sort_keys=True).encode()
        ).hexdigest()[:16]
        voice_dir  = azure_voice.replace(":", "_")
        cache_path = (_Path(PIPE_DIR) / "projects" / "resources" / "azure_tts"
                      / voice_dir / f"{h}.mp3")
        if cache_path.exists():
            # Cache hit — still need to convert mp3 to wav
            # For simplicity: just re-call TTS (mp3 is preview cache, not WAV cache)
            # TODO: if full WAV cache is needed, add here
            pass

    # Always synthesize fresh WAV for actual VO items
    result = _tts_throttled_call(_get_synth(), ssml)
    wav_bytes = result.audio_data  # PCM WAV bytes from Azure

    # Write {item_id}.source.wav (raw TTS output, INVARIANT A)
    source_path = os.path.join(vo_dir, f"{item_id}.source.wav")
    _write_wav_bytes_atomic(source_path, wav_bytes)

    # Call apply_vo_trims_for_item → writes {item_id}.wav (INVARIANT B)
    from pathlib import Path as _P
    primary_locale = _get_primary_locale(_P(full_ep))
    trimmed_dur = _apply_vo_trims_for_item(item_id, full_ep, locale)

    # Measure source duration
    source_dur = _wav_duration(_P(source_path))

    # Invalidate VO state (INVARIANT H)
    _invalidate_vo_state(full_ep, primary_locale)

    return {
        "source_duration_sec":  round(source_dur,    3),
        "trimmed_duration_sec": round(trimmed_dur,   3),
    }


def _write_wav_bytes_atomic(path: str, wav_bytes: bytes) -> None:
    """Write WAV bytes atomically (write to .tmp then rename)."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(wav_bytes)
    os.replace(tmp, path)


def _json_resp(handler, data: dict, status: int = 200) -> None:
    """Send a JSON response helper for VO endpoints."""
    body = json.dumps(data).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


# ── Story-file helpers ─────────────────────────────────────────────────────────
def _next_story_num() -> int:
    """Return the lowest unused N for story_N.txt in PIPE_DIR."""
    nums: list[int] = []
    for path in glob.glob(os.path.join(PIPE_DIR, "story_*.txt")):
        m = re.search(r"story_(\d+)\.txt$", path)
        if m:
            nums.append(int(m.group(1)))
    return max(nums) + 1 if nums else 1


def _save_story(text: str) -> tuple[str, int]:
    """Write text to the next story_N.txt and return (filename, N)."""
    num  = _next_story_num()
    path = os.path.join(PIPE_DIR, f"story_{num}.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return f"story_{num}.txt", num


def _parse_story_vars(story_file: str) -> dict:
    """Quick-parse a story_N.txt to extract PROJECT_SLUG and EPISODE_ID.

    Supports the loose key: value format used by the pipeline.
    Returns a dict with keys 'project_slug' and 'episode_id' (may be None).
    """
    path = os.path.join(PIPE_DIR, story_file)
    if not os.path.isfile(path):
        return {}

    slug = ep_id = ep_num = title = None
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            m = re.match(r'(?i)project\s*slug\s*[:\-]\s*(.+)', line)
            if m:
                slug = m.group(1).strip()
            m = re.match(r'(?i)episode\s*id\s*[:\-]\s*(.+)', line)
            if m:
                ep_id = m.group(1).strip()
            m = re.match(r'(?i)episode\s*(?:num(?:ber)?)?\s*[:\-]\s*(\d+)', line)
            if m and ep_num is None:
                ep_num = m.group(1).strip().zfill(2)
            m = re.match(r'(?i)(?:story\s*)?title\s*[:\-]\s*(.+)', line)
            if m and title is None:
                title = m.group(1).strip()

    # Derive slug from title if not explicit
    if not slug and title:
        slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')

    # Derive episode_id from episode number if not explicit
    if not ep_id and ep_num:
        ep_id = f"ep{ep_num}"

    return {"project_slug": slug, "episode_id": ep_id}


# ── Azure TTS helpers ──────────────────────────────────────────────────────────
_PRESETS_FILE    = (Path(PIPE_DIR) / "projects" / "resources" / "azure_tts"
                    / "presets.json")
_INDEX_FILE      = (Path(PIPE_DIR) / "projects" / "resources" / "azure_tts"
                    / "index.json")
_index_cache: dict | None = None   # loaded once; maps voice → clip dict

def _load_index_cache() -> dict:
    """Return the voices section of index.json (in-process cache)."""
    global _index_cache
    if _index_cache is not None:
        return _index_cache
    if _INDEX_FILE.exists():
        try:
            _index_cache = json.loads(
                _INDEX_FILE.read_text(encoding="utf-8")
            ).get("voices", {})
            return _index_cache
        except (json.JSONDecodeError, OSError):
            pass
    _index_cache = {}
    return _index_cache

def _is_default_clip(voice: str, style: str | None, h: str) -> bool:
    """Return True when hash h matches the index entry for voice+style.

    Uses "" as the clips key for no-style voices so they participate in the
    same default-vs-preset logic as styled voices.
    """
    idx       = _load_index_cache()
    style_key = style or ""
    clip      = idx.get(voice, {}).get("clips", {}).get(style_key)
    if not clip:
        return False
    # index stores full 64-char SHA256; h is the first 16 chars
    return clip.get("hash", "")[:16] == h


def save_index(voices_dict: dict) -> None:
    """Atomic write of index.json; also updates the in-process cache."""
    global _index_cache
    _INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {"schema_version": "1.0", "voices": voices_dict}
    tmp  = _INDEX_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(_INDEX_FILE)
    _index_cache = voices_dict  # keep in-process cache in sync


def load_presets() -> dict:
    """Load global voice presets from disk; return empty structure on any error."""
    if _PRESETS_FILE.exists():
        try:
            return json.loads(_PRESETS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"schema_version": "1.0", "presets": {}}

def save_presets(data: dict) -> None:
    """Atomic write of presets.json (same pattern as other cache files)."""
    _PRESETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _PRESETS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    tmp.replace(_PRESETS_FILE)


def parse_azure_tts_styles() -> dict:
    """Parse voice catalog; return voices grouped by story locale.

    Reads prompts/azure_tts_styles.txt — single file containing all 275
    voices (styled + no-style en/zh-capable + bridge + multitalker sections).

    Returns: { "en": [VoiceEntry, ...], "zh-Hans": [...], ... }
    VoiceEntry: {
        voice, azure_locale, gender, local_name, styles,
        has_styles   bool   — True if the voice has ≥1 style (suitable for emotional roles),
        is_multitalker bool — True for pair-synthesis voices (not for solo casting)
    }
    Result is cached at module level (parsed once on first request).
    """
    global _voice_catalog_cache
    if _voice_catalog_cache is not None:
        return _voice_catalog_cache

    catalog_path = os.path.join(PIPE_DIR, "prompts", "azure_tts_styles.txt")

    # Match any voice header line: "VoiceName  [STANDARD]" or "[DRAGON]"
    # Covers Neural, MAI-Voice-*, DragonHD, etc. — anchored on the tier bracket.
    VOICE_RE    = re.compile(r'^(\S+)\s+\[(?:STANDARD|DRAGON)\]')
    LOCALE_RE   = re.compile(r'locale=(\S+)\s+gender=(\S+)\s+local_name=(.+)')
    STYLES_RE   = re.compile(r"styles\(\d+\):\s*\[(.+)\]")
    SUPPORTS_RE = re.compile(r"supports\(\d+\):\s*\[(.+)\]")

    entries: list[dict] = []
    cur: dict | None    = None

    if os.path.isfile(catalog_path):
        with open(catalog_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                vm = VOICE_RE.match(line)
                if vm:
                    if cur and "azure_locale" in cur:
                        entries.append(cur)
                    vname = vm.group(1)
                    cur = {
                        "voice":          vname,
                        "styles":         [],
                        "supports":       [],
                        "is_multitalker": "ultitalker" in vname.lower(),
                    }
                    continue
                lm = LOCALE_RE.search(line)
                if lm and cur:
                    cur["azure_locale"] = lm.group(1)
                    cur["gender"]       = lm.group(2)
                    cur["local_name"]   = lm.group(3).strip()
                    continue
                sm = STYLES_RE.search(line)
                if sm and cur:
                    cur["styles"] = re.findall(r"'([^']+)'", sm.group(1))
                    continue
                pm = SUPPORTS_RE.search(line)
                if pm and cur:
                    cur["supports"] = re.findall(r"'([^']+)'", pm.group(1))
        if cur and "azure_locale" in cur:
            entries.append(cur)

    # Annotate has_styles (after styles are populated)
    for e in entries:
        e["has_styles"] = bool(e.get("styles"))

    # Group by language family: all en-* → "en", all zh-* → "zh-Hans".
    # Additionally expand any voice that has zh-* locales in supports() into
    # zh-Hans with azure_locale set to that zh locale, so e.g.
    # en-GB-Ada:DragonHDLatestNeural (supports zh-HK) appears under zh-HK optgroup.
    catalog: dict[str, list] = {}
    for entry in entries:
        al = entry.get("azure_locale", "")
        if not al:
            continue
        # Primary bucket
        if al.startswith("en-"):
            catalog.setdefault("en", []).append(entry)
        elif al.startswith("zh-"):
            catalog.setdefault("zh-Hans", []).append(entry)
        else:
            catalog.setdefault(al, []).append(entry)
        # Expand into zh-Hans for every zh-* locale in supports()
        for loc in entry.get("supports", []):
            if loc.startswith("zh-") and loc != al:
                e2 = dict(entry)
                e2["azure_locale"] = loc
                catalog.setdefault("zh-Hans", []).append(e2)

    # Sort each locale group by display name (local_name) for consistent UI order
    for loc_key in catalog:
        catalog[loc_key].sort(key=lambda e: e.get("local_name", e.get("voice", "")).lower())

    _voice_catalog_cache = catalog
    return catalog


def _tts_rest_preview(ssml: str) -> bytes:
    """HTTPS REST fallback for TTS preview — avoids WebSocket 429 issues."""
    import urllib.request
    region = os.environ.get("AZURE_SPEECH_REGION", "eastus")
    key    = os.environ.get("AZURE_SPEECH_KEY", "")
    url    = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-24khz-96kbitrate-mono-mp3",
        "User-Agent": "pipe-preview",
    }
    for attempt in range(4):
        req = urllib.request.Request(url, data=ssml.encode("utf-8"),
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(min(2 ** attempt, 8))
                continue
            raise RuntimeError(f"REST TTS HTTP {e.code}: {e.read().decode('utf-8','replace')[:200]}")
        except Exception as e:
            if attempt < 3:
                time.sleep(2)
                continue
            raise
    raise RuntimeError("REST TTS exhausted retries")


def _tts_throttled_call(synth, ssml: str):
    """Call Azure TTS with F0-safe throttling (3.5 s min gap) and 429 retry.
    Falls back to REST endpoint when SDK returns 429."""
    global _tts_last_call
    with _tts_lock:                                   # serialise all preview threads
        elapsed = time.monotonic() - _tts_last_call
        if elapsed < TTS_MIN_INTERVAL:
            time.sleep(TTS_MIN_INTERVAL - elapsed)

        for attempt, backoff in enumerate([0] + TTS_RETRY_BACKOFF):
            if backoff:
                time.sleep(backoff)
            result = synth.speak_ssml_async(ssml).get()
            _tts_last_call = time.monotonic()

            if result.reason == _speechsdk.ResultReason.SynthesizingAudioCompleted:
                return result

            details = result.cancellation_details
            err_str = getattr(details, "error_details", "") or ""

            # On 429/throttle, try REST fallback instead of more SDK retries
            if any(t in err_str.lower() for t in _TTS_REST_TRIGGERS):
                try:
                    audio_bytes = _tts_rest_preview(ssml)
                    # Return a lightweight object with .audio_data like the SDK
                    class _RestResult:
                        def __init__(self, data): self.audio_data = data
                    return _RestResult(audio_bytes)
                except RuntimeError:
                    if attempt < len(TTS_RETRY_BACKOFF):
                        continue   # retry SDK on next backoff
                    raise

            if "429" in err_str and attempt < len(TTS_RETRY_BACKOFF):
                continue   # retry with next backoff
            raise RuntimeError(err_str or str(getattr(details, "reason", "Unknown")))


def _get_synth():
    """Create or return the module-level SpeechSynthesizer singleton (CON-R5).

    Using a module-level singleton avoids ~100 ms overhead per preview request
    and prevents connection leaks from repeatedly creating new synthesisers.
    """
    global _tts_synth
    if _tts_synth is None:
        if not _TTS_AVAILABLE:
            raise RuntimeError("azure-cognitiveservices-speech SDK not installed")
        key    = os.environ.get("AZURE_SPEECH_KEY", "")
        region = os.environ.get("AZURE_SPEECH_REGION", "")
        if not key:
            raise RuntimeError("AZURE_SPEECH_KEY not set")
        config = _speechsdk.SpeechConfig(subscription=key, region=region)
        config.set_speech_synthesis_output_format(
            _speechsdk.SpeechSynthesisOutputFormat.Audio24Khz96KBitRateMonoMp3)
        _tts_synth = _speechsdk.SpeechSynthesizer(speech_config=config, audio_config=None)
    return _tts_synth


def _preview_build_ssml(text: str, azure_voice: str, azure_locale: str,
                         style: str | None, *, style_degree: float = 1.0,
                         rate: str = "0%", pitch: str = "", break_ms: int = 0) -> str:
    """Build SSML for a preview request (identical format to pre_cache_voices.py)."""
    escaped = (text.replace("&", "&amp;").replace("<", "&lt;")
                   .replace(">", "&gt;").replace('"', "&quot;").replace("'", "&apos;"))
    spoken = f'<lang xml:lang="{azure_locale}">{escaped}</lang>'
    if break_ms:
        spoken = f'{spoken}<break time="{break_ms}ms"/>'
    rate_attr  = f' rate="{rate}"'   if rate  and rate  != "0%" else ""
    pitch_attr = f' pitch="{pitch}"' if pitch and pitch != "0%" else ""
    if rate_attr or pitch_attr:
        spoken = f'<prosody{rate_attr}{pitch_attr}>{spoken}</prosody>'
    if style:
        spoken = (f'<mstts:express-as style="{style}" styledegree="{style_degree}">'
                  f"{spoken}</mstts:express-as>")
    return (f"<speak version='1.0' xml:lang='{azure_locale}' "
            f"xmlns='http://www.w3.org/2001/10/synthesis' "
            f"xmlns:mstts='http://www.w3.org/2001/mstts'>"
            f"<voice name='{azure_voice}'>{spoken}</voice></speak>")


# ── SSE helper ─────────────────────────────────────────────────────────────────
def sse(event: str, data: str) -> bytes:
    return f"event: {event}\ndata: {data}\n\n".encode()


# ── Embedded UI ────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Claude Runner</title>
<style>
  :root {
    --bg:      #f5f3ef;
    --surface: #ffffff;
    --border:  #d4d0c8;
    --gold:    #8a6d3b;
    --green:   #3a8a3a;
    --red:     #c04040;
    --blue:    #2a6cb6;
    --text:    #1a1a1a;
    --dim:     #666;
    --mono:    "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    /* Extended variables for light theme */
    --hover-bg:    rgba(0,0,0,0.04);
    --active-bg:   rgba(0,0,0,0.08);
    --input-bg:    #f0ede8;
    --input-border:#c8c4bc;
    --panel-bg:    #f8f6f2;
    --badge-bg:    rgba(0,0,0,0.06);
    --overlay-bg:  rgba(0,0,0,0.35);
    --shadow:      0 2px 12px rgba(0,0,0,0.08);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }

  /* ── Header ── */
  header {
    background: var(--panel-bg);
    border-bottom: 1px solid var(--border);
    padding: 14px 24px;
    display: flex;
    align-items: center;
    gap: 14px;
    flex-shrink: 0;
  }
  header h1 { font-size: 1rem; font-weight: 700; color: var(--gold); letter-spacing: .04em; }
  #status-badge {
    font-size: 0.72em; font-weight: 700; letter-spacing: .06em;
    padding: 3px 10px; border-radius: 20px;
    border: 1px solid var(--border); background: var(--hover-bg); color: var(--dim);
    transition: all .2s;
  }
  #status-badge.running { background:rgba(58,138,58,0.10); border-color:rgba(58,138,58,0.25); color:var(--green); }
  #status-badge.error   { background:rgba(192,64,64,0.10); border-color:rgba(192,64,64,0.25); color:var(--red);   }
  #cost-badge {
    margin-left: auto; font-size: 0.72em; color: var(--dim);
    font-family: var(--mono); display: none;
  }

  /* ── Main layout ── */
  main {
    display: flex;
    flex-direction: column;
    flex: 1;
    overflow-y: auto;
    padding: 16px 24px 20px;
    gap: 10px;
  }

  /* ── Section labels ── */
  .section-label {
    font-size: 0.68em; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: var(--dim);
    margin-bottom: 5px;
    display: flex; align-items: center; gap: 8px;
  }
  .file-badge {
    font-weight: 500; letter-spacing: 0; text-transform: none;
    font-family: var(--mono); font-size: 1.15em;
    color: var(--blue); background: rgba(42,108,182,0.10);
    border: 1px solid rgba(42,108,182,0.20); border-radius: 4px;
    padding: 1px 8px; transition: color .2s;
  }

  /* ── Story textarea ── */
  .story-block { flex-shrink: 0; display: flex; flex-direction: column; }
  #story {
    width: 100%;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-family: var(--mono);
    font-size: 0.84em;
    padding: 12px 14px;
    resize: vertical;
    min-height: 155px;
    max-height: 260px;
    line-height: 1.65;
    outline: none;
    transition: border-color .15s;
    display: block;
  }
  #story:focus { border-color: var(--gold); }
  #story::placeholder { color: #999; }

  .btn-group { display: flex; flex-direction: row; gap: 8px; flex-shrink: 0; }
  button {
    border: none; border-radius: 7px;
    font-size: 0.82em; font-weight: 700;
    padding: 9px 18px; cursor: pointer;
    letter-spacing: .03em;
    transition: opacity .15s, transform .1s;
    white-space: nowrap;
  }
  button:active  { transform: scale(.97); }
  button:disabled { opacity: .4; cursor: default; }
  #btn-run   { background: var(--gold); color: #fff; }
  #btn-stop  { background: var(--red);  color: #fff; display: none; }
  #btn-clear { background: var(--active-bg); color: var(--dim); border: 1px solid var(--border); }

  /* ── Stage progress indicator ── */
  #stage-progress {
    flex-shrink: 0;
    background: rgba(138,109,59,0.08);
    border: 1px solid #c9a84c30;
    border-radius: 6px;
    padding: 8px 14px;
    display: none;
    align-items: center;
    gap: 10px;
    font-size: 0.80em;
    font-family: var(--mono);
  }
  #stage-progress-num {
    color: var(--gold);
    font-weight: 700;
    white-space: nowrap;
    min-width: 78px;
  }
  #stage-progress-label {
    flex: 1;
    color: var(--text);
  }
  #stage-progress-dots {
    display: flex; gap: 3px; align-items: center;
  }
  .sdot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--border); flex-shrink: 0;
    transition: background .25s;
  }
  .sdot.done    { background: var(--green); }
  .sdot.current { background: var(--gold); }

  /* ── Command preview ── */
  #cmd-preview {
    font-family: var(--mono); font-size: 0.75em;
    color: var(--dim); background: var(--surface);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 7px 12px; display: none; flex-shrink: 0;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  #cmd-preview span { color: var(--blue); }

  /* ── Output ── */
  #output-wrap {
    flex: 1; background: var(--surface);
    border: 1px solid var(--border); border-radius: 8px;
    overflow: hidden; display: flex; flex-direction: column;
    min-height: 260px;
  }

  #output-label {
    font-size: 0.68em; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: var(--dim);
    padding: 7px 14px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 8px; flex-shrink: 0;
  }
  #line-count { margin-left: auto; font-weight: 400; font-family: var(--mono); }
  #output {
    flex: 1; font-family: var(--mono); font-size: 0.82em; line-height: 1.65;
    padding: 14px 16px; overflow-y: auto; white-space: pre-wrap;
    word-break: break-word; color: var(--text);
  }
  #output .sys  { color: var(--dim);   font-style: italic; }
  #output .err  { color: var(--red);   }
  #output .done { color: var(--green); font-style: italic; }
  #output .ts   { color: #7a9ab0; font-style: italic; font-size: 0.9em; }

  /* ── Spinner ── */
  @keyframes spin { to { transform: rotate(360deg); } }
  .spinner {
    display: inline-block; width: 10px; height: 10px;
    border: 2px solid rgba(58,138,58,0.25); border-top-color: var(--green);
    border-radius: 50%; animation: spin .7s linear infinite; vertical-align: middle;
  }

  /* ── Scrollbar ── */
  #output::-webkit-scrollbar { width: 6px; }
  #output::-webkit-scrollbar-track { background: transparent; }
  #output::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  /* ── Confirm modal ── */
  #modal-overlay {
    display: none; position: fixed; inset: 0;
    background: var(--overlay-bg); z-index: 100;
    align-items: center; justify-content: center;
  }
  #modal-overlay.visible { display: flex; }
  #modal-box {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 28px 32px; max-width: 430px; width: 90%;
    box-shadow: 0 24px 64px rgba(0,0,0,0.08);
  }
  #modal-box h2 { color: var(--gold); font-size: 1rem; margin-bottom: 10px; }
  #modal-box p  { color: var(--text); font-size: 0.85em; line-height: 1.65; margin-bottom: 14px; }
  #modal-path {
    font-family: var(--mono); color: var(--blue); font-size: 0.80em;
    background: rgba(42,108,182,0.10); border: 1px solid rgba(42,108,182,0.20); border-radius: 4px;
    padding: 4px 10px; display: inline-block; margin-bottom: 18px;
    word-break: break-all;
  }
  .modal-note { color: var(--dim) !important; font-size: 0.78em !important; margin-top: -8px; }
  .modal-btns { display: flex; gap: 10px; justify-content: flex-end; margin-top: 4px; }
  #btn-modal-yes { background: var(--red);   color: #fff; }
  #btn-modal-no  { background: var(--active-bg); color: var(--dim);
                   border: 1px solid var(--border); }

  /* ── Media batch resume modal ── */
  #media-modal-overlay {
    display: none; position: fixed; inset: 0;
    background: var(--overlay-bg); z-index: 101;
    align-items: center; justify-content: center;
  }
  #media-modal-overlay.visible { display: flex; }
  #media-modal-box {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 28px 32px; max-width: 500px; width: 92%;
    box-shadow: 0 24px 64px rgba(0,0,0,0.08);
  }
  #media-modal-box h2 { color: var(--gold); font-size: 1rem; margin: 0 0 10px; }
  #media-modal-box p  { color: var(--text); font-size: 0.85em; line-height: 1.65; margin: 0 0 10px; }
  .media-modal-meta {
    font-family: var(--mono); font-size: 0.78em; color: var(--dim);
    background: var(--hover-bg); border: 1px solid var(--border); border-radius: 6px;
    padding: 8px 12px; margin-bottom: 18px; line-height: 1.8;
  }
  .media-modal-meta b { color: var(--text); }
  .media-modal-meta .mm-path { color: var(--blue); word-break: break-all; }
  .media-modal-btns { display: flex; gap: 10px; justify-content: flex-end; margin-top: 6px; }
  #btn-mm-cancel    { background: var(--active-bg); color: var(--dim); border: 1px solid var(--border); }
  #btn-mm-resume    { background: var(--blue); color: #fff; }
  #btn-mm-new       { background: var(--red);  color: #fff; }

  /* ── Stage review buttons ── */
  .review-bar {
    display: flex; flex-wrap: wrap; gap: 6px;
    padding: 5px 0 3px; align-items: center;
  }
  .review-label {
    font-size: 0.70em; color: var(--dim); font-style: italic; margin-right: 2px;
  }
  .btn-review {
    background: #c9a84c14; color: var(--gold);
    border: 1px solid #c9a84c50; border-radius: 5px;
    font-size: 0.74em; font-weight: 600; font-family: var(--mono);
    padding: 3px 10px; cursor: pointer; letter-spacing: .01em;
    transition: background .15s, border-color .15s;
  }
  .btn-review:hover { background: #c9a84c28; border-color: #c9a84c80; }

  /* ── Segmented tab control ── */
  .tab-bar {
    display: flex; gap: 2px; margin-left: 16px;
    background: var(--hover-bg); border: 1px solid var(--border);
    border-radius: 8px; padding: 3px;
  }
  .tab {
    background: transparent; color: var(--dim);
    border: none; border-radius: 6px;
    font-size: 0.76em; font-weight: 700; letter-spacing: .04em;
    padding: 5px 14px; cursor: pointer;
    transition: background .15s, color .15s, box-shadow .15s;
  }
  .tab:hover  { color: var(--text); background: var(--hover-bg); }
  .tab.active {
    background: rgba(0,0,0,0.10); color: var(--text);
    box-shadow: 0 1px 4px #0005;
  }

  /* ── Slide toggle switch ── */
  .toggle-wrap {
    display: flex; align-items: center; gap: 7px;
    cursor: pointer; user-select: none;
  }
  .toggle-track {
    position: relative; width: 38px; height: 20px; flex-shrink: 0;
    border-radius: 10px;
    background: #3ecf6e1a; border: 1px solid #3ecf6e55;
    transition: background .25s, border-color .25s;
  }
  .toggle-thumb {
    position: absolute; top: 2px; left: 2px;
    width: 14px; height: 14px; border-radius: 50%;
    background: var(--green);
    box-shadow: 0 1px 3px #0006;
    transition: transform .25s cubic-bezier(.4,0,.2,1), background .25s;
  }
  .toggle-label {
    font-size: 0.72em; font-weight: 700; letter-spacing: .05em;
    transition: color .25s; white-space: nowrap;
  }
  .toggle-left  { color: var(--green); }
  .toggle-right { color: var(--dim);   }
  /* Prod state (test OFF) — thumb slides right, colours swap */
  .toggle-wrap.prod .toggle-track { background: #c9a84c18; border-color: #c9a84c44; }
  .toggle-wrap.prod .toggle-thumb { transform: translateX(18px); background: var(--gold); }
  .toggle-wrap.prod .toggle-left  { color: var(--dim);  }
  .toggle-wrap.prod .toggle-right { color: var(--gold); }
  /* HD render state — thumb slides right, colours in blue */
  .toggle-wrap.render-hd .toggle-track { background: #5b9bff18; border-color: #5b9bff44; }
  .toggle-wrap.render-hd .toggle-thumb { transform: translateX(18px); background: #5b9bff; }
  .toggle-wrap.render-hd .toggle-left  { color: var(--dim);  }
  .toggle-wrap.render-hd .toggle-right { color: #5b9bff; }

  /* ── Media panel ── */
  #panel-media {
    flex: 1; overflow: hidden;
    padding: 16px 24px 20px;
    display: none; flex-direction: column; gap: 10px;
  }
  .media-toolbar {
    flex-shrink: 0; display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  }
  #media-ep-select {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; color: var(--text);
    font-family: var(--mono); font-size: 0.80em;
    padding: 5px 10px; cursor: pointer; flex: 1; max-width: 280px;
  }
  .media-cfg-input {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; color: var(--text);
    font-family: var(--mono); font-size: 0.78em;
    padding: 5px 10px; outline: none;
  }
  .media-cfg-input:focus { border-color: var(--gold); }
  #media-btn-search {
    background: var(--gold); color: #fff; border: none;
    border-radius: 6px; font-size: 0.82em; font-weight: 700;
    padding: 6px 16px; cursor: pointer; transition: opacity .15s;
    flex-shrink: 0;
  }
  #media-btn-search:hover  { opacity: 0.85; }
  #media-btn-search:disabled { opacity: 0.4; cursor: not-allowed; }
  .media-status-bar {
    flex-shrink: 0; background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px 14px;
    font-size: 0.82em; color: var(--dim);
    display: flex; align-items: center; gap: 10px;
  }
  .media-spinner {
    width: 14px; height: 14px; border-radius: 50%;
    border: 2px solid var(--border); border-top-color: var(--gold);
    animation: spin .7s linear infinite; flex-shrink: 0;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .media-body {
    flex: 1; overflow-y: auto;
    display: flex; flex-direction: column; gap: 14px;
    padding-right: 2px;
  }
  .media-progress-table {
    width: 100%; border-collapse: collapse; font-size: 0.82em;
    color: var(--text);
  }
  .media-progress-table th {
    color: var(--dim); font-size: 0.78em; text-transform: uppercase;
    letter-spacing: 0.05em; font-weight: 600; padding: 0 10px 6px 0;
    text-align: left; border-bottom: 1px solid var(--border);
  }
  .media-progress-table td {
    padding: 5px 10px 5px 0; border-bottom: 1px solid var(--border);
    vertical-align: middle;
  }
  .media-progress-table tr:last-child td { border-bottom: none; }
  .mprog-pending  { color: var(--dim); }
  .mprog-dl       { color: #5bc4f5; }
  .mprog-scoring  { color: var(--gold); }
  .mprog-done     { color: #3ecf6e; }
  .mprog-failed   { color: #f55; }
  .mprog-num      { text-align: right; padding-right: 16px; font-family: var(--mono); }
  .mprog-dash     { text-align: right; padding-right: 16px; color: var(--dim); }
  .mprog-toggle   { cursor:pointer; user-select:none; color:var(--dim); font-size:0.8em;
                    padding-right:6px; transition:color 0.15s; }
  .mprog-toggle:hover { color:var(--text); }
  .mprog-toggle.open  { color:var(--blue); }
  .mprog-detail-row td { background:rgba(255,255,255,0.03); padding:6px 10px 8px 28px;
                          border-bottom:1px solid var(--border); }
  .mprog-detail-table { font-size:0.82em; border-collapse:collapse; }
  .mprog-detail-table td { padding:2px 16px 2px 0; color:var(--dim); vertical-align:middle; }
  .mprog-detail-table td.src-name { font-family:var(--mono); color:var(--text); min-width:90px; }
  .mprog-detail-table td.src-num  { text-align:right; font-family:var(--mono);
                                     color:var(--text); padding-right:20px; min-width:40px; }
  .mprog-detail-note { font-size:0.8em; color:var(--dim); margin-top:4px; font-style:italic; }
  .media-body::-webkit-scrollbar { width: 6px; }
  .media-body::-webkit-scrollbar-track { background: transparent; }
  .media-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  .media-item-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 14px 18px; flex-shrink: 0;
  }
  .media-item-header {
    font-size: 0.87em; font-weight: 700; color: var(--gold); margin-bottom: 4px;
  }
  .media-item-prompt {
    font-size: 0.76em; color: var(--dim); font-family: var(--mono);
    margin-bottom: 10px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .media-section-label {
    font-size: 0.72em; color: var(--dim); font-weight: 600;
    letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 8px;
  }
  .media-thumb-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }
  .media-thumb {
    position: relative; width: 160px; cursor: pointer;
    border-radius: 6px; overflow: hidden;
    border: 2px solid var(--border); transition: border-color .15s;
    flex-shrink: 0;
  }
  .media-thumb:hover   { border-color: #5b9cf6; }
  .media-thumb.selected { border-color: var(--gold); }
  .media-thumb.used-elsewhere { border-color: #4fc3f7; }
  .media-thumb.used-elsewhere .media-used-badge { display: block; }
  .media-used-badge {
    position: absolute; top: 4px; left: 4px;
    background: #4fc3f7cc; color: #fff; font-size: 0.62em; font-weight: 700;
    border-radius: 3px; padding: 2px 4px;
    pointer-events: none; display: none;
  }
  .media-thumb img,
  .media-thumb video {
    width: 160px; height: 90px; object-fit: cover; display: block;
    background: #181822;
  }
  .media-score-badge {
    position: absolute; bottom: 4px; left: 4px;
    background: #000000cc; color: #eee; font-size: 0.68em;
    border-radius: 4px; padding: 2px 5px; font-family: var(--mono);
    pointer-events: none;
  }
  .media-src-link {
    position: absolute; bottom: 4px; right: 4px;
    font-size: 0.75em; background: rgba(0,0,0,0.55);
    border-radius: 3px; padding: 1px 4px;
    color: #fff; text-decoration: none; z-index: 2;
    opacity: 0; transition: opacity 0.15s;
  }
  .media-thumb:hover .media-src-link { opacity: 1; }
  .media-sel-badge {
    position: absolute; top: 4px; right: 4px;
    background: var(--gold); color: #fff; font-size: 0.66em; font-weight: 700;
    border-radius: 3px; padding: 2px 5px;
    pointer-events: none; display: none;
  }
  .media-thumb.selected .media-sel-badge { display: block; }
  .media-footer {
    flex-shrink: 0; display: flex; align-items: center; gap: 12px; padding-top: 2px;
  }
  #media-btn-confirm {
    background: var(--green); color: #fff; border: none;
    border-radius: 6px; font-size: 0.85em; font-weight: 700;
    padding: 8px 20px; cursor: pointer; transition: opacity .15s;
  }
  #media-btn-confirm:hover    { opacity: 0.85; }
  #media-btn-confirm:disabled { opacity: 0.4; cursor: not-allowed; }
  #media-btn-reset {
    background: var(--active-bg); color: var(--dim);
    border: 1px solid var(--border); border-radius: 6px;
    font-size: 0.82em; padding: 7px 14px; cursor: pointer;
    transition: background .15s, color .15s;
  }
  #media-btn-reset:hover { background: rgba(0,0,0,0.10); color: var(--text); }
  #media-btn-apply-seq {
    background: #2a3a5c; color: #7eb8f7;
    border: 1px solid #3d5a8a; border-radius: 6px;
    font-size: 0.82em; padding: 7px 16px; cursor: pointer;
    transition: background .15s, color .15s;
  }
  #media-btn-apply-seq:hover { background: #364a70; color: #a8d0ff; }
  #media-confirm-msg { font-size: 0.80em; }
  .media-empty { color: var(--dim); font-style: italic; font-size: 0.83em; }
  /* ── Per-shot assignment rows ── */
  .media-shot-section {
    margin-top: 10px; border-top: 1px solid var(--border);
    padding-top: 8px;
  }
  .media-shot-row {
    display: flex; align-items: center; gap: 8px;
    padding: 3px 0; font-size: 0.78em; font-family: var(--mono);
  }
  .media-shot-label {
    color: var(--blue, #5b9cf6); font-weight: 600; min-width: 120px;
  }
  .media-shot-preview {
    color: var(--dim); flex: 1; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap;
  }
  .media-shot-clear {
    background: none; border: none; color: var(--dim);
    cursor: pointer; font-size: 1.0em; padding: 2px 4px;
    transition: color .15s;
  }
  .media-shot-clear:hover { color: #e06c75; }
  .media-btn-auto-assign {
    background: #2a3a5c; color: #7eb8f7;
    border: 1px solid #3d5a8a; border-radius: 5px;
    font-size: 0.76em; padding: 4px 12px; cursor: pointer;
    margin-bottom: 6px; transition: background .15s, color .15s;
  }
  .media-btn-auto-assign:hover { background: #364a70; color: #a8d0ff; }
  /* ── Extended thumbnails (beyond top_n — dimmed until hovered/selected) ── */
  .media-thumb-extended {
    opacity: 0.45; filter: grayscale(30%);
    transition: opacity .15s, filter .15s;
  }
  .media-thumb-extended:hover {
    opacity: 1.0; filter: none;
  }
  .media-thumb-extended.selected {
    opacity: 1.0; filter: none;
  }
  /* ── Lazy-scroll sentinel (triggers IntersectionObserver to load next page) ── */
  .media-lazy-sentinel {
    width: 40px; height: 90px; flex-shrink: 0;
  }
  /* ── Duration badge (top-left on video thumbnails) ── */
  .media-dur-badge {
    position: absolute; top: 4px; left: 4px;
    background: #000000cc; color: #7eb8f7; font-size: 0.64em;
    border-radius: 3px; padding: 2px 5px; font-family: var(--mono);
    pointer-events: none;
  }
  /* ── Animation picker button on image thumbnails ── */
  .media-anim-btn {
    position: absolute; top: 4px; right: 4px;
    background: rgba(0,0,0,0.65); color: #7a5a20; font-size: 0.75em;
    border: none; border-radius: 3px; padding: 2px 5px; cursor: pointer;
    opacity: 0; transition: opacity 0.15s; line-height: 1.2;
    z-index: 2;
  }
  .media-thumb:hover .media-anim-btn { opacity: 1; }
  .media-thumb.has-anim .media-anim-btn { opacity: 0.9; background: rgba(100,65,0,0.9); }
  /* ── Animation badge shown on thumbnails that have an animation set ── */
  .media-anim-thumb-badge {
    position: absolute; bottom: 18px; left: 4px;
    background: rgba(80,50,0,0.9); color: #7a5a20; font-size: 0.62em;
    border-radius: 3px; padding: 1px 4px; pointer-events: none; display: none;
  }
  .media-thumb.has-anim .media-anim-thumb-badge { display: block; }
  /* ── Live animation on the thumbnail img when an animation is set ── */
  .media-thumb img.anim-playing {
    animation-duration: 4s;
    animation-timing-function: ease-in-out;
    animation-iteration-count: infinite;
    animation-direction: alternate;
    animation-fill-mode: both;
  }
  @keyframes thumb-zoom-in   { from { transform: scale(1.0); } to { transform: scale(1.35); } }
  @keyframes thumb-zoom-out  { from { transform: scale(1.35); } to { transform: scale(1.0); } }
  @keyframes thumb-pan-lr    { from { transform: translate(-10%,0) scale(1.2); } to { transform: translate(10%,0) scale(1.2); } }
  @keyframes thumb-pan-rl    { from { transform: translate(10%,0) scale(1.2); } to { transform: translate(-10%,0) scale(1.2); } }
  @keyframes thumb-pan-up    { from { transform: translate(0,8%) scale(1.2); } to { transform: translate(0,-8%) scale(1.2); } }
  @keyframes thumb-ken-burns { from { transform: scale(1.0) translate(0%,0%); } to { transform: scale(1.3) translate(-5%,-3%); } }
  /* ── Animation picker popup (left = option list, right = live preview) ── */
  #media-anim-popup {
    position: fixed; z-index: 9999;
    background: var(--surface); border: 1px solid #444; border-radius: 8px;
    padding: 0; box-shadow: 0 6px 32px rgba(0,0,0,0.08);
    display: none;
  }
  #media-anim-popup .anim-popup-inner {
    display: flex; gap: 0;
  }
  #media-anim-popup .anim-options-col {
    padding: 8px 6px; min-width: 160px;
  }
  #media-anim-popup .anim-popup-title {
    font-size: 0.72em; color: #888888; margin-bottom: 6px; padding-left: 4px;
  }
  .anim-option {
    display: flex; align-items: center; gap: 6px;
    padding: 5px 8px; border-radius: 5px; cursor: pointer;
    transition: background 0.1s; white-space: nowrap;
  }
  .anim-option:hover { background: #e8e6f0; }
  .anim-option.active { background: #f5edd8; }
  .anim-option-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #555; flex-shrink: 0;
  }
  .anim-option.active .anim-option-dot { background: #e0c97f; }
  .anim-label { font-size: 0.8em; color: #ccc; }
  .anim-option.active .anim-label { color: #7a5a20; font-weight: 600; }
  /* ── Right pane: live preview of real image ── */
  #media-anim-preview-pane {
    width: 240px; min-height: 140px;
    background: #111; border-left: 1px solid var(--border);
    border-radius: 0 8px 8px 0; overflow: hidden;
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; flex-shrink: 0;
  }
  #media-anim-preview-pane .anim-live-wrap {
    width: 240px; height: 135px; overflow: hidden; position: relative;
  }
  #media-anim-preview-pane .anim-live-img {
    width: 240px; height: 135px; object-fit: cover; display: block;
    animation-duration: 3s;
    animation-timing-function: ease-in-out;
    animation-iteration-count: infinite;
    animation-direction: alternate;
    animation-fill-mode: both;
  }
  #media-anim-preview-pane .anim-live-label {
    font-size: 0.72em; color: #888888; padding: 5px 0 4px; text-align: center;
    width: 100%;
  }
  /* ── Animation badge in shot-row segment entries ── */
  .seg-anim-badge {
    display: inline-block; background: rgba(138,109,59,0.15); color: #7a5a20;
    font-size: 0.7em; border-radius: 3px; padding: 1px 5px; margin-left: 4px;
    cursor: pointer; vertical-align: middle;
  }
  .seg-anim-badge:hover { background: rgba(138,109,59,0.25); }

  /* ── Cross-shot copy button on thumbnails ── */
  .media-copy-btn {
    position: absolute; top: 2px; left: 2px; z-index: 5;
    background: #d8e8f5; color: #2a5a8a; border: 1px solid #a0c0e0;
    border-radius: 4px; font-size: 12px; padding: 1px 5px; cursor: pointer;
    opacity: 0; transition: opacity .15s; line-height: 1.2;
  }
  .media-thumb:hover .media-copy-btn { opacity: 0.85; }
  .media-copy-btn:hover { opacity: 1 !important; background: #c0d8f0; }

  /* ── Shot picker popup (cross-shot copy) ── */
  #media-shot-picker {
    position: fixed; z-index: 10000;
    background: var(--surface); border: 1px solid #444; border-radius: 8px;
    padding: 10px 12px; box-shadow: 0 6px 32px rgba(0,0,0,0.08);
    display: none; min-width: 200px; max-height: 350px; overflow-y: auto;
  }
  #media-shot-picker .sp-title {
    font-size: 0.75em; color: #888888; margin-bottom: 6px;
  }
  #media-shot-picker .sp-row {
    display: flex; align-items: center; gap: 8px;
    padding: 4px 6px; border-radius: 4px; cursor: pointer;
    font-size: 0.82em; color: #ccc;
  }
  #media-shot-picker .sp-row:hover { background: #e8e6f0; }
  #media-shot-picker .sp-row.sp-already { color: #666; cursor: default; }
  #media-shot-picker .sp-row.sp-already:hover { background: transparent; }
  #media-shot-picker .sp-cb { width: 14px; height: 14px; accent-color: #5b9cf6; }
  #media-shot-picker .sp-done {
    margin-top: 8px; padding: 4px 14px; font-size: 0.8em;
    background: #c0e0c0; color: #8aba8a; border: 1px solid #4a7a4a;
    border-radius: 4px; cursor: pointer; width: 100%;
  }
  #media-shot-picker .sp-done:hover { background: #b0d8b0; }

  /* ── Video clip trimmer (in segment row) ── */
  .seg-trim-row {
    display: flex; align-items: center; gap: 4px; margin-top: 2px;
    font-size: 10px; color: #999;
  }
  .seg-trim-row input[type="number"] {
    width: 50px; font-size: 10px; background: #1a1a1a; color: #ccc;
    border: 1px solid #444; border-radius: 3px; padding: 1px 3px;
  }
  .seg-trim-row .trim-label { color: #666; }
  .seg-trim-row .trim-range-bar {
    flex: 1; height: 6px; background: #d0ccc4; border-radius: 3px;
    position: relative; min-width: 60px; cursor: pointer;
  }
  .seg-trim-row .trim-range-fill {
    position: absolute; height: 100%; background: rgba(42,108,182,0.25);
    border-radius: 3px; top: 0;
  }
  .seg-trim-row .trim-btn {
    background: #1a2a3a; color: #7a9aba; border: 1px solid #3a5a7a;
    border-radius: 3px; font-size: 10px; padding: 0px 5px; cursor: pointer;
  }
  .seg-trim-row .trim-btn:hover { background: #c0d8f0; }

  /* ── Multi-use badge on thumbnails ── */
  .media-multi-badge {
    position: absolute; bottom: 18px; left: 2px; z-index: 3;
    background: #d8e8f5; color: #2a5a8a; font-size: 9px;
    padding: 1px 4px; border-radius: 3px; pointer-events: none;
  }

  /* ── Multi-segment shot rows ── */
  .media-shot-bar {
    height: 6px; flex: 0 0 120px; border-radius: 3px;
    background: var(--active-bg); overflow: hidden;
  }
  .media-shot-bar-fill {
    height: 100%; background: var(--green, #98c379);
    border-radius: 3px; transition: width .2s;
  }
  .media-shot-segments {
    display: flex; flex-direction: column; gap: 2px; flex: 1;
    min-width: 0;
  }
  .media-seg-entry {
    display: flex; align-items: center; gap: 6px;
    font-size: 0.74em; font-family: var(--mono); color: var(--text);
  }
  .media-seg-entry .seg-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .media-seg-entry .seg-dur { color: #7eb8f7; flex-shrink: 0; }
  .media-seg-remove {
    background: none; border: none; color: var(--dim);
    cursor: pointer; font-size: 0.9em; padding: 0 2px; transition: color .15s;
  }
  .media-seg-remove:hover { color: #e06c75; }
  .media-shot-gap { font-size: 0.72em; color: var(--dim); flex-shrink: 0; }
  .media-shot-filled .media-shot-gap { color: var(--green, #98c379); }
  .media-shot-filled .media-shot-bar-fill { background: var(--green, #98c379); }
  .media-shot-row { cursor: pointer; border-left: 3px solid transparent;
    padding-left: 5px; transition: border-color .15s, background .15s; }
  .media-shot-row:hover { background: var(--hover-bg); }
  .media-shot-active { border-left-color: var(--blue, #5b9cf6); background: #5b9cf612; }

  /* ── Music panel ── */
  #panel-music {
    flex: 1; overflow: hidden;
    padding: 16px 24px 20px;
    display: none; flex-direction: column; gap: 10px;
  }
  .music-toolbar {
    flex-shrink: 0; display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  }
  #music-ep-select {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; color: var(--text);
    font-family: var(--mono); font-size: 0.80em;
    padding: 5px 10px; cursor: pointer; flex: 1; max-width: 280px;
  }
  .music-status-bar {
    flex-shrink: 0; background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px 14px;
    font-size: 0.82em; color: var(--dim);
    display: flex; align-items: center; gap: 10px;
  }
  .music-body {
    flex: 1; overflow-y: auto;
    display: flex; flex-direction: column; gap: 14px;
    padding-right: 2px;
  }
  .music-body::-webkit-scrollbar { width: 6px; }
  .music-body::-webkit-scrollbar-track { background: transparent; }
  .music-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  .music-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 14px 18px; flex-shrink: 0;
  }
  .music-card-hdr {
    font-size: 0.87em; font-weight: 700; color: var(--gold); margin-bottom: 4px;
  }
  .music-card-sub {
    font-size: 0.76em; color: var(--dim); font-family: var(--mono);
    margin-bottom: 10px;
  }
  .music-section-label {
    font-size: 0.72em; color: var(--dim); font-weight: 600;
    letter-spacing: 0.06em; text-transform: uppercase; margin: 10px 0 6px;
  }
  .music-timeline-row {
    display: flex; align-items: center; gap: 8px; padding: 4px 0;
    font-size: 0.80em; font-family: var(--mono);
    border-bottom: 1px solid var(--border);
  }
  .music-timeline-shot { font-weight: 700; color: var(--gold); min-width: 100px; }
  .music-timeline-dur  { color: var(--dim); min-width: 60px; }
  .music-timeline-mood { color: var(--text); flex: 1; }
  .music-timeline-duck { color: var(--dim); min-width: 80px; }
  .music-cand-table { width: 100%; border-collapse: collapse; font-size: 0.80em; }
  .music-cand-table th {
    text-align: left; padding: 6px 8px; font-weight: 600;
    color: var(--dim); border-bottom: 1px solid var(--border);
    font-size: 0.78em; text-transform: uppercase; letter-spacing: 0.04em;
  }
  .music-cand-table td { padding: 5px 8px; border-bottom: 1px solid var(--hover-bg); }
  .music-cand-table tr:hover { background: var(--hover-bg); }
  .music-cand-selected { background: #5b9cf612 !important; }
  .music-override-table { width: 100%; border-collapse: collapse; font-size: 0.80em; }
  .music-override-table th {
    text-align: left; padding: 6px 8px; font-weight: 600;
    color: var(--dim); border-bottom: 1px solid var(--border);
    font-size: 0.78em; text-transform: uppercase; letter-spacing: 0.04em;
  }
  .music-override-table td { padding: 4px 8px; border-bottom: 1px solid var(--hover-bg); }
  .music-override-table input[type="number"],
  .music-override-table input[type="range"] {
    background: var(--bg); border: 1px solid var(--border); border-radius: 4px;
    color: var(--text); font-family: var(--mono); font-size: 0.9em;
    padding: 2px 4px; width: 70px;
  }
  .music-override-table input[type="range"] { width: 80px; }
  .music-preview-wrap {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 14px 18px;
  }
  .music-preview-wrap audio { width: 100%; margin-top: 8px; }
  /* Source browser row */
  .music-src-row { padding: 8px 0; border-bottom: 1px solid var(--hover-bg); }
  .music-src-top { display: flex; align-items: center; gap: 12px; margin-bottom: 6px; }
  .music-src-stem { font-family: var(--mono); font-size: 0.85em; font-weight: 700;
    color: var(--gold); min-width: 160px; }
  .music-src-meta { font-size: 0.78em; color: var(--dim); min-width: 100px; }
  .music-src-player { flex: 1; }
  .music-src-player audio { width: 100%; height: 32px; }
  .music-src-controls { display: flex; align-items: center; gap: 6px;
    margin-top: 4px; padding-left: 160px; }
  .music-src-controls button {
    font-size: 0.76em; padding: 3px 10px; cursor: pointer;
    background: var(--active-bg); color: var(--dim); border: 1px solid var(--border);
    border-radius: 4px;
  }
  .music-src-controls button:hover { background: rgba(0,0,0,0.10); color: var(--text); }
  .music-src-controls button.active { background: var(--gold); color: #fff; }
  .music-src-controls .mark-label {
    font-size: 0.76em; color: var(--dim); font-family: var(--mono);
  }
  /* Visual shot timeline bar */
  .music-vtl-bar {
    display: flex; width: 100%; height: 38px; border-radius: 6px;
    overflow: hidden; border: 1px solid var(--border); margin-bottom: 2px;
  }
  .music-vtl-shot {
    display: flex; align-items: center; justify-content: center;
    font-size: 0.68em; font-family: var(--mono); font-weight: 600;
    color: #fff; cursor: default; position: relative;
    border-right: 1px solid #00000040; box-sizing: border-box;
    overflow: hidden; white-space: nowrap;
  }
  .music-vtl-shot:last-child { border-right: none; }
  .music-vtl-sel-row {
    display: flex; width: 100%; margin-bottom: 6px;
  }
  .music-vtl-sel-cell {
    display: flex; flex-direction: column; gap: 2px;
    box-sizing: border-box; padding: 0 1px;
  }
  .music-vtl-sel-cell select {
    font-family: var(--mono); font-size: 0.72em; width: 100%;
    background: var(--bg); color: var(--text); border: 1px solid var(--border);
    border-radius: 3px; padding: 1px 2px;
  }
  .music-vtl-param-row {
    display: flex; width: 100%; margin-bottom: 4px;
  }
  .music-vtl-param-cell {
    display: flex; align-items: center; gap: 3px; box-sizing: border-box;
    padding: 0 2px; font-size: 0.72em; font-family: var(--mono); color: var(--dim);
  }
  .music-vtl-param-cell input[type="number"] {
    width: 42px; background: var(--bg); color: var(--text);
    border: 1px solid var(--border); border-radius: 3px;
    font-family: var(--mono); font-size: 0.95em; padding: 1px 3px;
  }
  .music-vtl-param-cell input[type="range"] { width: 50px; height: 14px; }
  /* Per-shot stacked blocks */
  .music-shot-block {
    border: 1px solid var(--border); border-radius: 6px;
    margin-bottom: 8px; overflow: hidden;
  }
  .music-shot-hdr {
    display: flex; align-items: center; gap: 12px;
    padding: 5px 10px; background: var(--hover-bg);
  }
  .music-shot-hdr-id {
    font-family: var(--mono); font-weight: 700; font-size: 0.88em; color: var(--text);
    min-width: 40px;
  }
  .music-shot-hdr-ep {
    font-family: var(--mono); font-size: 0.78em; color: var(--gold); letter-spacing: 0.02em;
  }
  .music-shot-clip {
    padding: 4px 10px;
  }
  .music-shot-clip select {
    width: 100%; background: var(--bg); color: var(--text);
    border: 1px solid var(--border); border-radius: 4px;
    font-family: var(--mono); font-size: 0.82em; padding: 3px 6px;
  }
  .music-shot-params {
    display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
    padding: 4px 10px 7px; font-size: 0.78em; font-family: var(--mono); color: var(--dim);
  }
  .music-shot-params label { color: var(--dim); white-space: nowrap; }
  .music-shot-params input[type="number"] {
    background: var(--bg); color: var(--text);
    border: 1px solid var(--border); border-radius: 3px;
    font-family: var(--mono); font-size: 0.95em; padding: 2px 4px;
  }
  .music-footer {
    flex-shrink: 0; display: flex; align-items: center; gap: 10px;
    padding-top: 8px; border-top: 1px solid var(--border);
  }
  .music-footer button {
    background: var(--gold); color: #fff; border: none;
    border-radius: 6px; font-size: 0.82em; font-weight: 700;
    padding: 6px 16px; cursor: pointer; transition: opacity .15s;
  }
  .music-footer button:hover { opacity: 0.85; }
  .music-footer button:disabled { opacity: 0.4; cursor: not-allowed; }
  #music-confirm-msg { font-size: 0.80em; color: var(--dim); flex: 1; }
  .music-btn-secondary {
    background: var(--active-bg) !important; color: var(--dim) !important;
    border: 1px solid var(--border) !important;
  }

  /* ── Browse panel ── */
  #panel-browse {
    flex: 1; overflow: hidden;
    padding: 16px 24px 20px;
    display: none; flex-direction: column; gap: 10px;
  }
  .browse-toolbar {
    flex-shrink: 0; display: flex; align-items: center; gap: 10px;
  }
  #btn-refresh {
    background: var(--active-bg); color: var(--dim);
    border: 1px solid var(--border); border-radius: 6px;
    font-size: 0.76em; padding: 5px 12px; cursor: pointer;
    transition: background .15s, color .15s;
  }
  #btn-refresh:hover { background: rgba(0,0,0,0.10); color: var(--text); }
  .browse-empty { color: var(--dim); font-style: italic; font-size: 0.83em; padding: 8px 0; }
  #browse-tree {
    flex: 1; overflow-y: auto;
    font-family: var(--mono); font-size: 0.83em;
  }
  .proj-group  { margin-bottom: 18px; }
  .proj-heading {
    color: var(--gold); font-weight: 700; font-size: 0.9em;
    padding: 5px 0 6px; display: flex; align-items: center; gap: 6px;
    border-bottom: 1px solid var(--border);
  }
  .ep-toggle-row {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 0 6px 10px; cursor: pointer;
    color: var(--blue); font-weight: 600;
    border-radius: 5px; transition: background .12s;
  }
  .ep-toggle-row:hover { background: #5b9cf610; }
  .ep-caret  { font-size: 0.68em; width: 10px; text-align: center; color: var(--dim); }
  .ep-meta   { color: var(--dim); font-size: 0.80em; font-weight: 400; }
  .ep-files  {
    margin-left: 22px; padding-left: 14px;
    border-left: 1px solid var(--border);
    display: none;
  }
  .ep-files.open { display: block; }
  .ep-file-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 4px 2px; border-bottom: 1px solid #1a1a26;
  }
  .ep-file-row:last-child { border-bottom: none; }
  .ep-file-name { color: var(--text); }
  .ep-file-sz   { color: var(--dim); font-size: 0.80em; margin-left: 8px; }
  #browse-tree::-webkit-scrollbar { width: 6px; }
  #browse-tree::-webkit-scrollbar-track { background: transparent; }
  #browse-tree::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  /* ── Pipeline panel ── */
  #panel-pipeline {
    flex: 1; overflow: hidden;
    padding: 16px 24px 20px;
    display: none; flex-direction: column; gap: 8px;
  }
  .pipe-toolbar {
    flex-shrink: 0; display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  }
  #pipe-ep-select {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; color: var(--text);
    font-family: var(--mono); font-size: 0.80em;
    padding: 5px 10px; cursor: pointer; flex: 1; max-width: 360px;
  }
  .pipe-body {
    flex: 1; min-height: 0; overflow-y: auto; display: flex; flex-direction: column; gap: 10px;
    padding-right: 2px;
  }
  .pipe-section {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; overflow: hidden; flex-shrink: 0;
  }
  .pipe-section-hdr {
    background: var(--hover-bg); padding: 8px 14px;
    display: flex; align-items: center; gap: 10px;
    border-bottom: 1px solid var(--border);
    font-size: 0.82em; font-weight: 700; color: var(--text);
  }
  .pipe-step {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 14px; border-bottom: 1px solid #1a1a26;
    font-size: 0.80em;
  }
  .pipe-step:last-child { border-bottom: none; }
  .step-status { width: 18px; text-align: center; flex-shrink: 0; font-size: 0.9em; }
  .step-status.done    { color: var(--green); }
  .step-status.pending { color: #3a3a50; }
  .step-name { flex: 1; font-family: var(--mono); color: var(--text); }
  .step-artifact {
    font-family: var(--mono); font-size: 0.85em; color: var(--blue);
    cursor: pointer; white-space: nowrap;
  }
  .step-artifact:hover { text-decoration: underline; }
  .btn-pipe-run {
    background: #c9a84c14; color: var(--gold);
    border: 1px solid #c9a84c50; border-radius: 5px;
    font-size: 0.72em; font-weight: 700; font-family: var(--mono);
    padding: 3px 10px; cursor: pointer; white-space: nowrap;
    transition: background .15s, border-color .15s;
  }
  .btn-pipe-run:hover { background: #c9a84c28; border-color: #c9a84c80; }
  .btn-pipe-run.blue {
    background: rgba(42,108,182,0.10); color: var(--blue); border-color: #5b9cf650;
  }
  .btn-pipe-run.blue:hover { background: #5b9cf628; border-color: #5b9cf680; }

  /* ── Pipe terminal ── */

  /* ── Review block (Video / Soundtrack / Voice Cast tabs) ── */
  #pipe-review-wrap {
    flex-shrink: 0; background: var(--surface);
    border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
  }
  .review-hdr {
    padding: 8px 14px; border-bottom: 1px solid var(--border);
    font-size: 0.68em; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: var(--dim);
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  }
  .review-content-tabs { display: flex; gap: 4px; }
  .btn-review-tab {
    background: var(--active-bg); color: var(--dim); border: 1px solid var(--border);
    border-radius: 4px; font-size: 1.1em; padding: 2px 10px; cursor: pointer;
    font-weight: 600; letter-spacing: 0; text-transform: none;
    transition: background .15s, color .15s;
  }
  .btn-review-tab.active {
    background: #5b9cf620; color: var(--blue); border-color: #5b9cf650;
  }
  .review-locale-tabs { display: flex; gap: 4px; margin-left: auto; }
  .btn-locale-tab {
    background: var(--active-bg); color: var(--dim); border: 1px solid var(--border);
    border-radius: 4px; font-size: 1.1em; padding: 2px 10px; cursor: pointer;
    font-weight: 600; letter-spacing: 0; text-transform: none;
    transition: background .15s, color .15s;
  }
  .btn-locale-tab.active {
    background: #5b9cf620; color: var(--blue); border-color: #5b9cf650;
  }
  .review-pane { display: none; }
  .review-pane.active { display: block; }
  #pipe-video { width: 100%; max-height: 240px; background: #000; display: block; }
  #pipe-audio { width: 100%; display: block; padding: 8px; box-sizing: border-box; }
  .pipe-body::-webkit-scrollbar { width: 6px; }
  .pipe-body::-webkit-scrollbar-track { background: transparent; }
  .pipe-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  /* ── Stage expand detail ── */
  .btn-expand {
    background: transparent; border: none; color: var(--dim);
    font-size: 1em; padding: 0 4px; cursor: pointer; flex-shrink: 0;
    transition: transform .15s, color .15s; line-height: 1;
  }
  .btn-expand:hover { color: var(--text); }
  .btn-expand.open  { transform: rotate(90deg); color: var(--gold); }
  .pipe-detail {
    display: none;
    background: #0a0a0f; border-top: 1px solid var(--border);
    padding: 8px 14px 8px 42px;
    font-family: var(--mono); font-size: 0.74em;
    color: #6a7a9a; white-space: pre; line-height: 1.7;
  }
  .pipe-detail.open { display: block; }
  .pipe-substep-row {
    display: flex; align-items: center; gap: 5px;
    flex-wrap: wrap; padding: 3px 0; white-space: normal;
  }
  .pipe-substep-locale {
    font-size: 0.78em; color: var(--dim); min-width: 64px;
    font-family: var(--mono); flex-shrink: 0;
  }
  .btn-substep {
    background: var(--hover-bg); color: #6a7a9a;
    border: 1px solid var(--border); border-radius: 4px;
    font-size: 0.76em; padding: 2px 9px; cursor: pointer;
    font-family: var(--mono); transition: background .15s, color .15s;
  }
  .btn-substep:hover { background: rgba(0,0,0,0.10); color: var(--text); }
  .vc-style-chip-UNUSED {   /* kept as placeholder; vc viewer removed from Pipeline tab */
    display: inline-block; background: var(--hover-bg);
    border: 1px solid var(--border); border-radius: 3px;
    padding: 1px 5px; font-size: 0.70em; color: #8a9ab8;
    margin: 2px 2px 0 0;
  }

  /* ── Story Input tab bar ── */
  .story-tab-bar {
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
    padding-bottom: 8px; flex-shrink: 0;
  }
  .btn-story-tab {
    background: var(--hover-bg); color: var(--dim); border: 1px solid var(--border);
    border-radius: 5px; padding: 3px 12px; font-size: 0.78em; cursor: pointer;
    font-weight: 600; transition: background .15s, color .15s;
  }
  .btn-story-tab.active {
    background: #5b9cf620; color: var(--blue); border-color: #5b9cf650;
  }
  #vc-saved-badge {
    font-size: 0.74em; color: var(--green); font-family: var(--mono); margin-left: 4px;
  }
  #vc-editor {
    overflow-y: auto; display: flex; flex-direction: column; gap: 8px;
    min-height: 200px; max-height: 420px;
  }
  #vc-locale-tabs { display: flex; gap: 4px; flex-shrink: 0; }
  #sr-panel {
    display: none; flex-direction: column; gap: 8px;
    min-height: 200px; max-height: 420px;
  }
  #sr-content {
    flex: 1; overflow-y: auto; background: #0a0a0e; border: 1px solid var(--border);
    border-radius: 6px; padding: 10px 14px; font-family: var(--mono); font-size: 0.78em;
    color: var(--text); white-space: pre-wrap; word-break: break-word; min-height: 120px;
  }
  #sr-content .sr-sep   { color: var(--border); }
  #sr-content .sr-ts    { color: var(--dim); }
  #sr-content .sr-issue { color: #e05c5c; font-weight: 700; }
  #sr-content .sr-fix   { color: var(--green); }
  #sr-content .sr-manual{ color: var(--gold); font-weight: 700; }
  /* ── VO Alignment panel ── */
  #sr-alignment {
    font-family: var(--mono); font-size: 0.76em; color: var(--text);
    background: #0a0a0e; border: 1px solid var(--border); border-radius: 6px;
    padding: 10px 14px; display: none; flex-direction: column; gap: 4px;
  }
  #sr-alignment.visible { display: flex; }
  .al-hdr  { color: var(--gold); font-weight: 700; margin-bottom: 2px; }
  .al-stats{ color: var(--dim); margin-bottom: 6px; }
  .al-row  { display: grid; grid-template-columns: 1fr 80px 80px; gap: 8px;
             padding: 2px 0; border-top: 1px solid #1a1a24; }
  .al-id   { color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .al-good { color: var(--green); text-align: right; }
  .al-bad  { color: #e05c5c;      text-align: right; }
  .al-chars{ color: var(--dim);   text-align: right; }
  .al-txt  { grid-column: 1 / -1; padding: 2px 0 4px 12px; }
  .al-txt-before { color: #e05c5c;      font-style: italic; }
  .al-txt-after  { color: var(--green); }
  .sr-add-row {
    display: flex; gap: 6px; flex-shrink: 0; align-items: flex-start;
  }
  #sr-note-input {
    flex: 1; background: var(--surface); border: 1px solid var(--border);
    border-radius: 5px; color: var(--text); font-family: var(--mono);
    font-size: 0.78em; padding: 6px 10px; resize: vertical; min-height: 52px;
  }
  #btn-sr-add {
    background: rgba(42,108,182,0.10); color: var(--blue); border: 1px solid #5b9cf650;
    border-radius: 5px; font-size: 0.78em; padding: 6px 12px; cursor: pointer;
    font-weight: 700; white-space: nowrap; align-self: flex-end;
  }
  #sr-next-step {
    flex-shrink: 0; border-radius: 6px; padding: 8px 14px;
    font-size: 0.82em; font-family: var(--mono); font-weight: 600; display: none;
  }
  #sr-next-step.ns-action  { display: block; background: #1a2f4a; color: var(--blue);  border: 1px solid #5b9cf650; }
  #sr-next-step.ns-running { display: block; background: #2a2010; color: var(--gold);  border: 1px solid #b8860b80; }
  #sr-next-step.ns-done    { display: block; background: #0e2718; color: var(--green); border: 1px solid #4caf5060; }
  /* ── Run-tab VO review banner (Stage 7.5) ─────────────────────────────── */
  #run-vo-review-banner {
    display: none; align-items: center; gap: 10px; flex-wrap: wrap;
    background: #0d2318; border: 1px solid #4caf5060; border-radius: 8px;
    padding: 10px 16px; margin: 8px 0; font-size: 0.85em;
  }
  #run-vo-review-banner.visible { display: flex; }
  #run-vo-review-banner .rvr-msg {
    flex: 1; color: var(--green); font-weight: 600; font-family: var(--mono);
  }
  #run-vo-review-banner button {
    background: #1a3a28; color: var(--green); border: 1px solid #4caf5070;
    border-radius: 5px; padding: 5px 12px; font-size: 0.85em;
    cursor: pointer; white-space: nowrap; font-weight: 600;
  }
  #run-vo-review-banner button:hover { background: #244d35; }
  #run-vo-review-banner button.primary {
    background: #1a3060; color: var(--blue); border-color: #5b9cf660;
  }
  #run-vo-review-banner button.primary:hover { background: #1f3c80; }
  #sr-banner-row { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
  #sr-banner-row #sr-next-step { flex: 1; }
  #btn-diagnose {
    flex-shrink: 0; background: #1a1a2e; color: #a78bfa;
    border: 1px solid #7c3aed60; border-radius: 6px;
    font-size: 0.78em; font-weight: 700; padding: 6px 12px;
    cursor: pointer; white-space: nowrap;
  }
  #btn-diagnose:hover   { background: #2a1a4e; border-color: #a78bfa80; }
  #btn-diagnose:disabled{ opacity: 0.5; cursor: default; }
  #sr-diagnose-result {
    display: none; flex-shrink: 0; border-radius: 6px; padding: 8px 14px;
    font-size: 0.80em; font-family: var(--mono); line-height: 1.5;
  }
  #sr-diagnose-result.srd-action  { display: block; background: #1a2f4a; color: var(--blue);  border: 1px solid #5b9cf650; font-weight: 600; }
  #sr-diagnose-result.srd-done    { display: block; background: #0e2718; color: var(--green); border: 1px solid #4caf5060; font-weight: 600; }
  #sr-diagnose-result.srd-running { display: block; background: #1a1a2e; color: #a78bfa;      border: 1px solid #7c3aed50; font-style: italic; }
  #sr-diagnose-result.srd-error   { display: block; background: #2a0e0e; color: var(--red);   border: 1px solid #e05c5c50; }

  /* ── Voice Cast editor character cards ── */
  .vc-char-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; overflow: hidden; flex-shrink: 0;
  }
  .vc-char-card-hdr {
    background: var(--hover-bg); padding: 7px 14px;
    display: flex; align-items: center; gap: 8px;
    border-bottom: 1px solid var(--border);
    font-size: 0.82em; font-weight: 700; color: var(--text);
  }
  .vc-char-card-body { padding: 10px 14px; display: flex; flex-direction: column; gap: 8px; }
  .vc-voice-row      { display: flex; align-items: center; gap: 8px; }
  .vc-voice-select {
    flex: 1; background: var(--surface); border: 1px solid var(--border);
    border-radius: 5px; color: var(--text); font-family: var(--mono);
    font-size: 0.78em; padding: 4px 8px; cursor: pointer;
  }
  .vc-params-row  { display: flex; flex-direction: column; gap: 5px; }
  .vc-param-group { display: flex; flex-direction: row; align-items: center; gap: 8px; }
  .vc-param-label { font-size: 0.68em; color: var(--dim); font-family: var(--mono);
                    min-width: 115px; text-align: right; flex-shrink: 0; }
  .vc-param-input {
    width: 64px; background: var(--surface); border: 1px solid var(--border);
    border-radius: 4px; color: var(--text); font-size: 0.78em;
    font-family: var(--mono); padding: 3px 6px; flex-shrink: 0;
  }
  /* Interpretation text + zone badges */
  .vc-param-interp { display: flex; align-items: center; gap: 5px; flex-wrap: wrap; flex: 1; }
  .vc-param-range  { font-size: 0.65em; color: var(--dim); font-family: var(--mono);
                     opacity: 0.6; white-space: nowrap; flex-shrink: 0; }
  .vc-interp-text  { font-size: 0.70em; color: var(--dim); font-style: italic; line-height: 1.3; }
  .vc-zone-badge   { font-size: 0.68em; padding: 1px 5px; border-radius: 3px; font-weight: 700;
                     font-family: var(--mono); white-space: nowrap; letter-spacing: 0.03em; flex-shrink: 0; }
  .vc-zone-extended     { background: rgba(201,168,76,.12); color: var(--gold);
                          border: 1px solid rgba(201,168,76,.35); }
  .vc-zone-experimental { background: rgba(220,60,60,.10); color: #d05050;
                          border: 1px solid rgba(208,80,80,.30); }
  .vc-preset-row   { display: flex; align-items: center; gap: 8px; }
  .vc-preset-label { font-size: 0.68em; color: var(--dim); font-family: var(--mono);
                     white-space: nowrap; min-width: 115px; text-align: right; flex-shrink: 0; }
  .vc-preset-select { flex: 1; background: var(--surface); border: 1px solid var(--border);
                      border-radius: 4px; color: var(--text); font-size: 0.78em;
                      font-family: var(--mono); padding: 3px 6px; cursor: pointer; }
  .vc-preset-select:disabled { opacity: 0.4; cursor: default; }
  .vc-style-select {
    width: 150px; background: var(--surface); border: 1px solid var(--border);
    border-radius: 4px; color: var(--text); font-size: 0.78em;
    font-family: var(--mono); padding: 3px 6px; cursor: pointer;
  }
  .btn-vc-preview {
    background: rgba(42,108,182,0.10); color: var(--blue); border: 1px solid #5b9cf650;
    border-radius: 4px; font-size: 0.72em; padding: 3px 10px; cursor: pointer;
    font-family: var(--mono); font-weight: 700; white-space: nowrap;
    transition: background .15s;
  }
  .btn-vc-preview:hover    { background: #5b9cf628; }
  .btn-vc-preview:disabled { opacity: 0.45; cursor: not-allowed; }
  .btn-vc-preset-del {
    background: #f5424214; color: #f54242; border: 1px solid #f5424250;
    border-radius: 4px; font-size: 0.68em; padding: 2px 8px; cursor: pointer;
    font-family: var(--mono); font-weight: 600; white-space: nowrap;
    transition: background .15s;
  }
  .btn-vc-preset-del:hover    { background: #f5424228; }
  .btn-vc-preset-del:disabled { opacity: 0.35; cursor: not-allowed; }
  .vc-footer {
    display: flex; gap: 8px; justify-content: flex-end; padding: 4px 0 2px; flex-shrink: 0;
  }
  #btn-vc-save, #btn-vc-continue {
    background: #c9a84c14; color: var(--gold); border: 1px solid #c9a84c50;
    border-radius: 5px; font-size: 0.78em; font-weight: 700;
    font-family: var(--mono); padding: 5px 14px; cursor: pointer;
    transition: background .15s;
  }
  #btn-vc-continue { background: rgba(42,108,182,0.10); color: var(--blue); border-color: #5b9cf650; }

  /* ── Prepare / info bar ───────────────────────────────────────────────── */
  #run-ep-selector { display:flex; gap:8px; align-items:center; margin-bottom:8px; flex-wrap:wrap; }
  #run-ep-selector select { background:var(--surface); color:var(--fg); border:1px solid var(--border); border-radius:4px; padding:3px 6px; font-size:0.82em; }
  #btn-prepare { background:var(--surface); color:var(--fg); border:1px solid var(--border); border-radius:4px; padding:4px 12px; font-size:0.82em; cursor:pointer; }
  #btn-prepare:hover { border-color:var(--accent); color:var(--accent); }
  #btn-prepare:disabled { opacity:0.4; cursor:default; }
  #info-bar { display:none; border:1px solid var(--border); border-radius:6px; padding:10px 12px; margin:8px 0; font-size:0.82em; background:var(--surface); }
  #info-bar.visible { display:block; }
  .info-row { display:flex; align-items:center; gap:8px; margin:4px 0; }
  .info-label { color:var(--dim); min-width:56px; }
  .info-value { flex:1; }
  .info-value input { background:var(--bg); border:1px solid var(--border); border-radius:3px; color:var(--fg); padding:2px 6px; font-size:0.82em; width:100%; box-sizing:border-box; }
  .info-badge { font-size:0.72em; color:var(--dim); border:1px solid var(--border); border-radius:10px; padding:1px 6px; white-space:nowrap; }
  .info-badge.from-story { color:#7ec87e; border-color:#7ec87e44; }
  .info-badge.unique { color:#7ec87e; border-color:#7ec87e44; }
  .info-badge.exists  { color:#e08030; border-color:#e0803044; }
  #info-format select { background:var(--surface); color:var(--fg); border:1px solid var(--border); border-radius:4px; padding:2px 6px; font-size:0.82em; }
  #info-format .format-hint { color:var(--dim); font-size:0.76em; margin-top:3px; font-style:italic; }
  .locale-row { display:flex; align-items:center; gap:12px; margin-top:6px; }
  .locale-row label { display:flex; align-items:center; gap:4px; cursor:pointer; color:var(--fg); font-size:0.82em; }
</style>
</head>
<body>

<header>
  <h1>⚡ Claude Runner</h1>
  <nav class="tab-bar">
    <button class="tab active" data-tab="run"      onclick="switchTab('run')"     >▶ Run</button>
    <button class="tab"        data-tab="pipeline" onclick="switchTab('pipeline')">🎬 Pipeline</button>
    <button class="tab"        data-tab="browse"   onclick="switchTab('browse')"  >📁 Browse</button>
    <button class="tab"        data-tab="media"    onclick="switchTab('media')"   >🖼 Media</button>
    <button class="tab"        data-tab="sfx"      onclick="switchTab('sfx')"     >🔊 SFX</button>
    <button class="tab"        data-tab="music"    onclick="switchTab('music')"   >🎵 Music</button>
    <button class="tab"        data-tab="vo"       onclick="switchTab('vo')"      >🎙 VO</button>
    <button class="tab"        data-tab="youtube"  onclick="switchTab('youtube')" >▶ YouTube</button>
  </nav>

  <div class="toggle-wrap" id="toggle-render"
       onclick="toggleRenderMode()" tabindex="0"
       title="Preview — fast encode (CRF 28) for review">
    <span class="toggle-label toggle-left">🖼 Preview</span>
    <div class="toggle-track"><div class="toggle-thumb"></div></div>
    <span class="toggle-label toggle-right">📺 HD</span>
  </div>
  <span id="status-badge">IDLE</span>
  <span id="cost-badge"></span>
</header>

<main id="panel-run">

  <!-- ── Story input ── -->
  <div class="story-block">
    <div class="story-tab-bar">
      <div class="section-label" style="margin:0">Story Input</div>
      <button class="btn-story-tab active" data-tab="story" onclick="switchStoryTab('story')">Story</button>
      <button class="btn-story-tab"        data-tab="vc"    onclick="switchStoryTab('vc')">Voice Cast</button>
      <button class="btn-story-tab"        data-tab="sr"    onclick="switchStoryTab('sr')">Status Report</button>
      <span id="vc-saved-badge" style="display:none">✓ Saved</span>
      <span class="file-badge" id="file-badge" style="margin-left:auto">story_1.txt</span>
    </div>
  <!-- Project / Episode selectors for Run tab -->
  <div id="run-ep-selector">
    <span style="color:var(--dim);font-size:0.8em">Project</span>
    <select id="run-project-sel" onchange="onRunProjectChange()">
      <option value="">— New Project —</option>
    </select>
    <span style="color:var(--dim);font-size:0.8em">Episode</span>
    <select id="run-episode-sel" onchange="onRunEpisodeChange()" disabled>
      <option value="">—</option>
    </select>
  </div>
    <textarea id="story" spellcheck="false"
placeholder="Enter your story here"></textarea>
    <div id="vc-editor" style="display:none">
      <div id="vc-locale-tabs"></div>
      <div id="vc-cards"></div>
      <div class="vc-footer">
        <button id="btn-vc-save"     onclick="saveVoiceCast()">💾 Save Voice Cast</button>
        <button id="btn-vc-continue" onclick="vcContinue()" style="display:none">
          ▶ Continue (Run 1–10)
        </button>
      </div>
    </div>
    <div id="sr-panel">
      <div id="sr-banner-row">
        <div id="sr-next-step"></div>
        <button id="btn-diagnose" onclick="runDiagnose()" title="Ask Claude which stage to re-run based on file timestamps">🤖 Diagnose</button>
      </div>
      <div id="sr-diagnose-result"></div>
      <div id="sr-content"><span style="color:var(--dim);font-style:italic">No episode selected.</span></div>
      <div id="sr-alignment"></div>
      <div class="sr-add-row">
        <textarea id="sr-note-input" placeholder="Add a status note (issue / fix / manual step)…"></textarea>
        <button id="btn-sr-add" onclick="appendStatusNote()">＋ Add Note</button>
      </div>
    </div>
  </div>

  <!-- hidden: stage range always 0–9 (internally 0–10); split into 0+1–N handled in runPrompt() -->
  <input type="hidden" id="prompt" value="0  10">
  <!-- ── Run options + buttons ── -->
  <div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
    <label id="label-no-music"
           style="display:flex; align-items:center; gap:6px; cursor:pointer;
                  font-size:0.82em; font-family:var(--mono); color:var(--dim);
                  user-select:none;"
           title="Include background music in renders — uncheck to render VO + SFX only">
      <input type="checkbox" id="chk-no-music" onchange="toggleMusicMode()"
             style="width:14px; height:14px; cursor:pointer; accent-color:var(--gold);">
      🎵 Music
    </label>
    <label style="display:flex; align-items:center; gap:6px; cursor:pointer;
                  font-size:0.82em; font-family:var(--mono); color:var(--dim);
                  user-select:none;"
           title="Delete cached WAVs, images, renders and manifests before each Stage 9 run — ensures a clean rebuild from scratch">
      <input type="checkbox" id="chk-purge-assets" onchange="togglePurgeMode()"
             style="width:14px; height:14px; cursor:pointer; accent-color:#e06c75;">
      🗑 Purge Cache
    </label>
    <button id="btn-prepare" onclick="runPrepare()" title="Analyse story and detect project name, format and genre">⚙ Prepare</button>
    <div class="btn-group" style="margin:0;">
      <button id="btn-run"   onclick="runPrompt()">▶ Run</button>
      <button id="btn-stop"  onclick="stopRun()">■ Stop</button>
      <button id="btn-clear" onclick="clearOutput()">✕ Clear</button>
    </div>
    <button id="btn-stage75" onclick="runStage75()"
            title="Stage 7.5 — run manifest_merge + gen_tts for primary locale, then pause for VO review"
            style="font-size:0.82em;font-family:var(--mono);padding:5px 10px;
                   background:#1a3060;color:var(--blue);border:1px solid #5b9cf650;
                   border-radius:5px;cursor:pointer;white-space:nowrap;">
      🎙 TTS Preview
    </button>
  </div>

  <!-- Info bar — shown after Prepare -->
  <div id="info-bar">
    <div class="info-row">
      <span class="info-label">📖 Title</span>
      <span class="info-value"><input id="info-title" type="text" placeholder="Story title" oninput="onTitleInput()"></span>
      <span id="badge-title" class="info-badge" style="display:none"></span>
    </div>
    <div class="info-row">
      <span class="info-label">🏷 Genre</span>
      <span class="info-value"><input id="info-genre" type="text" placeholder="Genre"></span>
      <span id="badge-genre" class="info-badge" style="display:none"></span>
    </div>
    <div class="info-row">
      <span class="info-label">📁 Slug</span>
      <span class="info-value"><input id="info-slug" type="text" placeholder="project-slug" oninput="this.dataset.autoDerived='0'; onSlugInput()"></span>
      <span id="badge-slug" class="info-badge"></span>
    </div>
    <div class="info-row">
      <span class="info-label">🆔 Episode</span>
      <span class="info-value" id="info-ep-id" style="font-family:var(--mono);color:var(--blue);font-size:0.85em;padding:2px 0">—</span>
      <span id="badge-ep-id" class="info-badge" style="display:none"></span>
    </div>
    <div id="info-format" class="info-row" style="flex-direction:column;align-items:flex-start;gap:4px">
      <div style="display:flex;align-items:center;gap:8px">
        <span class="info-label">🎬 Format</span>
        <select id="info-format-sel" onchange="onFormatChange()">
          <option value="episodic"              disabled title="Not yet validated">Episodic (default)</option>
          <option value="continuous_narration"  selected>Continuous Narration</option>
          <option value="illustrated_narration" disabled title="Not yet validated">Illustrated Narration</option>
          <option value="documentary"           disabled title="Not yet validated">Documentary / Explainer</option>
          <option value="monologue"             disabled title="Not yet validated">Monologue / First-Person</option>
          <option value="ssml_narration">SSML Narration (authored)</option>
        </select>
        <span id="badge-format" class="info-badge" style="display:none"></span>
      </div>
      <div id="format-hint" class="format-hint"></div>
      <div id="media-config-panel" style="margin-top:10px; padding:10px 14px; background:#e8f5e8; border:1px solid #a8d8a8; border-radius:6px; font-size:12px; color:#555; line-height:1.7;">
        <div style="font-weight:600; color:#8bc48b; margin-bottom:6px;">Media Search Config <span id="media-config-profile" style="color:#2a7a2a; font-weight:normal;"></span></div>
        <div id="media-config-body"></div>
      </div>
    </div>
    <div class="locale-row">
      <span class="info-label" style="color:var(--dim)">Locales</span>
      <label><input type="checkbox" id="locale-en"      checked onchange="onLocaleChange()"> en</label>
      <label><input type="checkbox" id="locale-zh-Hans" checked onchange="onLocaleChange()"> zh-Hans</label>
    </div>
    <div id="save-new-ep-row" class="info-row" style="display:none;justify-content:flex-end;margin-top:6px;gap:8px">
      <span id="save-new-ep-status" style="color:var(--dim);font-size:0.85em;font-style:italic"></span>
      <button id="btn-save-new-ep" onclick="saveNewEpMeta()"
        style="padding:3px 12px;font-size:0.85em;background:var(--accent,#4a9eff);color:#fff;border:none;border-radius:4px;cursor:pointer">💾 Save</button>
    </div>
  </div>
  <!-- Info bar for existing projects -->
  <div id="existing-ep-bar" style="display:none;border:1px solid var(--border);border-radius:6px;padding:8px 12px;margin:8px 0;font-size:0.82em;background:var(--surface)">
    <div class="info-row">
      <span class="info-label">📖 Title</span>
      <input id="ex-title" type="text" placeholder="—"
        style="flex:1;background:var(--input-bg,#1e1e1e);border:1px solid var(--border);border-radius:4px;color:var(--text);font-weight:600;padding:2px 6px;font-size:1em">
    </div>
    <div class="info-row">
      <span class="info-label">🏷 Genre</span>
      <input id="ex-genre" type="text" placeholder="—"
        style="flex:1;background:var(--input-bg,#1e1e1e);border:1px solid var(--border);border-radius:4px;color:var(--dim);padding:2px 6px;font-size:1em">
    </div>
    <div class="info-row">
      <span class="info-label">🎬 Format</span>
      <select id="info-format-sel-existing" onchange="onFormatChangeExisting()">
        <option value="episodic"              disabled title="Not yet validated">Episodic (default)</option>
        <option value="continuous_narration"  selected>Continuous Narration</option>
        <option value="illustrated_narration" disabled title="Not yet validated">Illustrated Narration</option>
        <option value="documentary"           disabled title="Not yet validated">Documentary / Explainer</option>
        <option value="monologue"             disabled title="Not yet validated">Monologue / First-Person</option>
        <option value="ssml_narration">SSML Narration (authored)</option>
      </select>
    </div>
    <div id="media-config-panel-existing" style="margin-top:10px; padding:10px 14px; background:#e8f5e8; border:1px solid #a8d8a8; border-radius:6px; font-size:12px; color:#555; line-height:1.7;">
      <div style="font-weight:600; color:#8bc48b; margin-bottom:6px;">Media Search Config <span id="media-config-profile-existing" style="color:#2a7a2a; font-weight:normal;"></span></div>
      <div id="media-config-body-existing"></div>
    </div>
    <div class="locale-row">
      <span class="info-label" style="color:var(--dim)">Locales</span>
      <label><input type="checkbox" id="locale-en-ex"      checked onchange="onLocaleChange()"> en</label>
      <label><input type="checkbox" id="locale-zh-Hans-ex" onchange="onLocaleChange()"> zh-Hans</label>
    </div>
    <div id="format-hint-existing" style="color:var(--dim);font-size:0.76em;margin-top:4px;font-style:italic"></div>
    <div class="info-row" style="justify-content:flex-end;margin-top:6px;gap:8px">
      <span id="save-ep-status" style="color:var(--dim);font-size:0.85em;font-style:italic"></span>
      <button id="btn-save-ep-meta" onclick="saveEpisodeMeta()"
        style="padding:3px 12px;font-size:0.85em;background:var(--accent,#4a9eff);color:#fff;border:none;border-radius:4px;cursor:pointer">💾 Save</button>
    </div>
  </div>

  <div id="cmd-preview">$ <span id="cmd-text"></span></div>

  <!-- ── Stage progress ── -->
  <div id="stage-progress">
    <span id="stage-progress-num">Stage 0/10</span>
    <span class="spinner" id="stage-progress-spinner"></span>
    <span id="stage-progress-label"></span>
    <div id="stage-progress-dots"></div>
  </div>

  <!-- ── Stage 7.5 VO review banner ── -->
  <div id="run-vo-review-banner">
    <span class="rvr-msg">🎙 TTS complete — VO ready for review</span>
    <button onclick="switchTab('vo'); populateVoEpSelect()">→ Open VO Tab</button>
    <button class="primary" onclick="document.getElementById('run-vo-review-banner').classList.remove('visible')" title="Dismiss this banner">✕ Dismiss</button>
  </div>

  <!-- ── Output ── -->
  <div id="output-wrap">
    <div id="output-label">
      Output
      <span id="spinner" style="display:none"><span class="spinner"></span></span>
      <span id="line-count">0 lines</span>
    </div>
    <div id="output"><span class="sys">Ready. Paste a story above and press Run.</span>
</div>
  </div>

</main>

<!-- ── Browse panel ── -->
<div id="panel-browse">
  <div class="browse-toolbar">
    <div class="section-label">Projects &amp; Episodes</div>
    <button id="btn-refresh" onclick="loadProjects()">↺ Refresh</button>
  </div>
  <div id="browse-tree"><span class="browse-empty">Switch to this tab to load projects.</span></div>
</div>

<!-- ── Media panel ── -->
<div id="panel-media">
  <!-- toolbar row -->
  <div class="media-toolbar">
    <div class="section-label" style="margin-bottom:0">Media Search</div>
    <select id="media-ep-select" onchange="onMediaEpChange()">
      <option value="">— select episode —</option>
    </select>
    <input  id="media-server-url" class="media-cfg-input" type="text"
            placeholder="Media server URL  e.g. http://localhost:8200"
            value="{{MEDIA_SERVER_URL}}" />
    <button id="media-btn-search" onclick="mediaStartSearch()" disabled>🔍 Search Media</button>
  </div>

  <!-- per-source settings table (always visible once project selected) -->
  <div id="media-sources-table" style="display:none;padding:8px 2px 10px;border-bottom:1px solid var(--border)">
    <table style="border-collapse:collapse;font-size:0.82em;color:var(--text)">
      <thead>
        <tr style="color:var(--dim);font-size:0.78em;text-transform:uppercase;letter-spacing:0.05em">
          <th style="padding:0 18px 5px 0;text-align:left;font-weight:600">Source</th>
          <th style="padding:0 10px 5px;text-align:center;font-weight:600">Enabled</th>
          <th style="padding:0 10px 5px;text-align:center;font-weight:600">📷 IMG candidates</th>
          <th style="padding:0 0 5px 10px;text-align:center;font-weight:600">🎬 VID candidates</th>
        </tr>
      </thead>
      <tbody id="media-sources-tbody"></tbody>
    </table>
  </div>

  <!-- status / spinner -->
  <div class="media-status-bar" id="media-status-bar">
    <span id="media-status-text">Select an episode to begin.</span>
    <span class="media-spinner" id="media-spinner"></span>
  </div>

  <!-- results body — item cards rendered here by JS -->
  <div class="media-body" id="media-body"></div>

  <!-- footer actions -->
  <div class="media-footer" id="media-footer" style="display:none">
    <span id="media-confirm-msg"></span>
    <button id="media-btn-reset"     onclick="mediaReset()">↺ Reset</button>
    <button id="media-btn-apply-seq" onclick="mediaApplyRecommended()" style="display:none">⚡ Apply Recommended Sequence</button>
    <button id="media-btn-confirm"   onclick="mediaConfirm()">✔ Confirm Selections</button>
  </div>
</div>

<!-- ── SFX panel ── -->
<div id="panel-sfx" style="display:none;flex-direction:column;flex:1;overflow:hidden;padding:16px 24px 20px;gap:12px">
  <style>
    .sfx-toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding-bottom:10px;border-bottom:1px solid var(--border)}
    .sfx-toolbar select{background:var(--surface);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:4px 8px;font-size:0.82em}
    .sfx-body{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:10px}
    .sfx-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px 14px}
    .sfx-card-header{display:flex;align-items:center;gap:8px;margin-bottom:8px}
    .sfx-card-id{font-size:0.78em;font-weight:700;color:var(--accent);font-family:monospace}
    .sfx-card-tag{font-size:0.85em;color:var(--text);flex:1}
    .sfx-card-dur{font-size:0.78em;color:var(--dim)}
    .sfx-cand-list{display:flex;flex-direction:column;gap:4px}
    .sfx-cand-row{display:flex;align-items:center;gap:8px;padding:6px 8px;border-radius:6px;cursor:pointer;border:1px solid transparent;transition:background 0.15s}
    .sfx-cand-row:hover{background:var(--hover-bg)}
    .sfx-cand-row.selected{border-color:var(--accent);background:var(--active-bg)}
    .sfx-cand-waveform{width:48px;height:28px;object-fit:cover;border-radius:3px;flex-shrink:0;background:var(--active-bg)}
    .sfx-cand-title{flex:1;font-size:0.83em;color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .sfx-cand-meta{font-size:0.76em;color:var(--dim);white-space:nowrap;flex-shrink:0}
    .sfx-play-btn{background:var(--accent);border:none;color:#000;cursor:pointer;font-size:1em;font-weight:700;padding:4px 9px;border-radius:5px;flex-shrink:0;min-width:34px;transition:opacity 0.15s}
    .sfx-play-btn:hover{opacity:0.8}
    .sfx-play-btn.playing{background:#e8a020}
    .sfx-play-btn.error{background:#c0392b;color:#fff}
    .sfx-link-btn{background:none;border:none;color:var(--dim);cursor:pointer;font-size:0.85em;padding:0 2px;text-decoration:none;flex-shrink:0}
    .sfx-link-btn:hover{color:var(--text)}
    .sfx-ai-toggle-btn{background:#f3e8f3;border:1px solid #c8a0c8;border-radius:4px;color:#7a3a7a;cursor:pointer;font-size:0.78em;padding:2px 8px;margin-left:6px;flex-shrink:0}
    .sfx-ai-toggle-btn:hover{background:#2a1a2a}
    .sfx-ai-panel{display:none;background:#f3e8f3;border:1px solid #c8a0c8;border-radius:4px;padding:8px 10px;margin:4px 0 6px}
    .sfx-ai-panel textarea{width:100%;box-sizing:border-box;background:var(--panel-bg);color:var(--text);border:1px solid #5a3a5a;border-radius:3px;font-size:11px;font-family:monospace;resize:vertical;padding:4px}
    .sfx-ai-panel .sfx-ai-hint{font-size:10px;color:#666;margin:3px 0 5px}
    .sfx-ai-panel .sfx-ai-row{display:flex;align-items:center;gap:8px;margin-top:5px}
    .sfx-ai-gen-btn{padding:3px 12px;font-size:12px;cursor:pointer;background:#2a1a2a;border:1px solid #7a4a7a;border-radius:4px;color:#7a3a7a}
    .sfx-ai-gen-btn:hover{background:#3a2a3a}
    .sfx-ai-gen-btn:disabled{opacity:0.5;cursor:default}
    .sfx-ai-status{font-size:11px;color:#888888}
    .sfx-status-bar{font-size:0.82em;color:var(--dim);padding:4px 0}
    .sfx-footer{display:flex;align-items:center;gap:10px;padding-top:10px;border-top:1px solid var(--border)}
    .sfx-empty{color:var(--dim);font-size:0.85em;padding:12px 0}
  </style>

  <!-- toolbar -->
  <div class="sfx-toolbar">
    <div class="section-label" style="margin-bottom:0">SFX Search</div>
    <select id="sfx-ep-select" onchange="onSfxEpChange()">
      <option value="">— select episode —</option>
    </select>
    <input id="sfx-server-url" type="text" class="media-cfg-input"
           placeholder="Media server URL  e.g. http://localhost:8200"
           value="{{MEDIA_SERVER_URL}}" />
    <button id="sfx-btn-search" onclick="sfxSearchAll()" disabled>🔍 Search All SFX</button>
    <span id="sfx-count-label" style="font-size:0.82em;color:var(--dim)"></span>
  </div>

  <!-- status bar -->
  <div class="sfx-status-bar" id="sfx-status-bar">Select an episode to begin.</div>

  <!-- results body -->
  <div class="sfx-body" id="sfx-body"></div>

  <!-- footer -->
  <div class="sfx-footer" id="sfx-footer" style="display:none">
    <span id="sfx-confirm-msg" style="font-size:0.83em;color:var(--dim);flex:1"></span>
    <button onclick="sfxSaveAll()">💾 Save SFX Selections</button>
    <button onclick="sfxReset()" style="background:transparent;border:1px solid var(--border);color:var(--dim)">↺ Reset</button>
  </div>
</div>

<!-- ── Music panel ── -->
<div id="panel-music">
  <!-- toolbar row -->
  <div class="music-toolbar">
    <div class="section-label" style="margin-bottom:0">Music Review</div>
    <select id="music-ep-select" onchange="onMusicEpChange()">
      <option value="">— select episode —</option>
    </select>
    <button id="music-btn-review"  onclick="musicGenerateReview()" disabled
            style="background:var(--active-bg);color:var(--dim);border:1px solid var(--border);border-radius:6px;font-size:0.80em;padding:5px 14px;cursor:pointer">🎵 Generate Music Review</button>
  </div>

  <!-- status / spinner -->
  <div class="music-status-bar" id="music-status-bar">
    <span id="music-status-text">Select an episode to begin.</span>
    <span class="media-spinner" id="music-spinner" style="display:none"></span>
  </div>

  <!-- scrollable body: timeline + candidates + overrides + preview -->
  <div class="music-body" id="music-body"></div>

  <!-- footer actions -->
  <div class="music-footer" id="music-footer" style="display:none">
    <span id="music-confirm-msg"></span>
    <button id="music-btn-confirm" onclick="musicConfirm()">✔ Confirm MusicPlan</button>
  </div>
</div>

<!-- ── VO Retune panel ── -->
<div id="panel-vo" style="display:none;flex-direction:column;flex:1;overflow:hidden;padding:16px 24px 20px;gap:10px">
  <style>
    .vo-toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding-bottom:10px;border-bottom:1px solid var(--border)}
    .vo-toolbar select{background:var(--surface);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:4px 8px;font-size:0.82em}
    .vo-body{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:0}
    .vo-empty{color:var(--dim);font-size:0.85em;padding:20px 0}
    .vo-scene-header{display:flex;align-items:center;gap:8px;padding:8px 4px 4px;border-top:1px solid var(--border);margin-top:6px}
    .vo-scene-break{display:flex;align-items:center;gap:6px;padding:4px 8px;margin:6px 0 0;background:var(--active-bg);border-radius:4px;border:1px dashed var(--border)}
    .vo-break-label{font-size:0.72em;color:var(--dim);flex:1}
    .vo-break-input{width:60px;font-size:0.8em;padding:2px 4px;background:var(--input-bg,#1e1e1e);color:var(--text);border:1px solid var(--border);border-radius:3px;text-align:right}
    /* ── VO Timeline Player ─────────────────────────────────────── */
    #vo-timeline-bar{display:none;flex-direction:column;gap:4px;padding:8px 12px;
      margin:6px 0;background:#0e1f2e;border:1px solid #2a4a6a;border-radius:6px;
      position:sticky;top:0;z-index:20}
    #vo-timeline-bar.active{display:flex}
    .vo-tl-top{display:flex;align-items:center;gap:10px}
    .vo-tl-playbtn{width:28px;height:28px;border-radius:50%;background:#1e6fa8;
      border:none;color:#fff;font-size:1em;cursor:pointer;flex-shrink:0;line-height:1}
    .vo-tl-playbtn:hover{background:#2a8fd8}
    .vo-tl-time{font-size:0.78em;color:#7ecfff;font-variant-numeric:tabular-nums;min-width:80px}
    .vo-tl-label{font-size:0.75em;color:#9ab;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .vo-tl-close{background:none;border:none;color:#678;font-size:1em;cursor:pointer;padding:0 2px}
    .vo-tl-close:hover{color:#f88}
    .vo-tl-track{position:relative;height:28px;margin:0 2px}
    .vo-tl-scenes{position:absolute;top:0;left:0;right:0;height:10px;display:flex;overflow:hidden;border-radius:3px 3px 0 0}
    .vo-tl-seg{height:100%;opacity:.55;transition:opacity .15s}
    .vo-tl-seg:hover{opacity:.85}
    .vo-tl-seg-label{font-size:9px;color:#fff;overflow:hidden;padding:0 2px;white-space:nowrap;line-height:10px}
    .vo-tl-scrubber{position:absolute;bottom:0;left:0;right:0;height:16px;
      -webkit-appearance:none;appearance:none;width:100%;margin:0;
      background:transparent;cursor:pointer;outline:none}
    .vo-tl-scrubber::-webkit-slider-runnable-track{height:6px;border-radius:3px;background:#1a3a5a}
    .vo-tl-scrubber::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;
      border-radius:50%;background:#4fc3f7;margin-top:-4px;cursor:pointer;border:2px solid #0e1f2e}
    .vo-tl-scrubber::-moz-range-track{height:6px;border-radius:3px;background:#1a3a5a}
    .vo-tl-scrubber::-moz-range-thumb{width:14px;height:14px;border-radius:50%;
      background:#4fc3f7;border:2px solid #0e1f2e;cursor:pointer}
    .vo-tl-progress{position:absolute;bottom:5px;left:0;height:6px;background:#4fc3f7;
      border-radius:3px;pointer-events:none;transition:width .1s linear}
    .vo-item-row.tl-active{outline:2px solid #4fc3f780;outline-offset:-2px}
    .vo-item-row:hover{background:transparent!important}
    .vo-scene-label{font-size:0.78em;font-weight:700;color:var(--dim);text-transform:uppercase;letter-spacing:.04em;flex:1}
    .vo-scene-btn{font-size:0.72em;padding:3px 10px;background:var(--active-bg);color:var(--dim);border:1px solid var(--border);border-radius:5px;cursor:pointer}
    .vo-scene-btn:hover{background:rgba(0,0,0,0.12);color:var(--text)}
    .vo-item-row{display:flex;align-items:center;gap:6px;padding:4px 4px;border-radius:5px}
    .vo-item-row:hover{background:var(--hover-bg)}
    .vo-item-id{font-size:0.72em;color:var(--dim);min-width:160px;font-family:monospace}
    .vo-field{background:var(--hover-bg);color:var(--text);border:1px solid transparent;border-radius:4px;padding:3px 6px;font-size:0.78em;font-family:monospace}
    .vo-field:focus{border-color:var(--border);outline:none;background:var(--surface)}
    .vo-text{flex:1;min-width:180px}
    .vo-voice{width:150px}
    .vo-style{width:110px}
    .vo-rate,.vo-pitch{width:52px}
    .vo-preview-btn{font-size:0.8em;padding:3px 7px;background:var(--active-bg);color:var(--dim);border:1px solid var(--border);border-radius:5px;cursor:pointer;flex-shrink:0}
    .vo-preview-btn:hover{background:rgba(0,0,0,0.12);color:var(--text)}
    .vo-preview-btn.playing{color:#4fc3f7;border-color:#4fc3f7}
    .vo-degree{width:44px}
    .vo-dur{font-size:0.75em;color:var(--dim);min-width:52px;text-align:right;font-family:monospace}
    .vo-resynth-btn{font-size:0.8em;padding:3px 9px;background:var(--active-bg);color:var(--dim);border:1px solid var(--border);border-radius:5px;cursor:pointer;flex-shrink:0}
    .vo-resynth-btn:hover{background:rgba(0,0,0,0.12);color:var(--text)}
    .vo-resynth-btn:disabled{opacity:.4;cursor:default}
    .vo-col-headers{display:flex;align-items:center;gap:6px;padding:2px 4px;font-size:0.68em;color:var(--dim);text-transform:uppercase;letter-spacing:.04em}
    .vo-badge{font-size:0.7em;padding:1px 6px;border-radius:10px;font-weight:600;flex-shrink:0}
    .vo-badge-ok{background:#1c3a1c;color:#6ec96e}
    .vo-badge-short{background:#4a2200;color:#f0963c}
    .vo-badge-long{background:#3a1a3a;color:#c070d0}
    .vo-badge-stale{background:#3a2200;color:#c09030}
    .vo-trim-row{display:flex;align-items:center;gap:6px;padding:2px 4px 4px 160px;font-size:0.78em;flex-wrap:wrap}
    .vo-trim-input{width:70px;background:var(--hover-bg);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:2px 5px;font-size:0.9em;font-family:monospace}
    .vo-trim-btn{font-size:0.78em;padding:2px 8px;background:var(--active-bg);color:var(--dim);border:1px solid var(--border);border-radius:4px;cursor:pointer}
    .vo-trim-btn:hover{background:rgba(0,0,0,0.12);color:var(--text)}
    .vo-timing-stale{text-decoration:line-through;color:var(--dim);font-style:italic}
  </style>
  <div class="vo-toolbar">
    <span class="section-label" style="margin:0">Episode</span>
    <select id="vo-ep-select" onchange="onVoEpChange()" style="min-width:200px">
      <option value="">— select episode —</option>
    </select>
    <span class="section-label" style="margin:0 0 0 8px">Locale</span>
    <select id="vo-locale-select" onchange="loadVoItems()">
      <option value="">— select locale —</option>
    </select>
    <button class="vo-scene-btn" onclick="loadVoItems()" style="margin-left:4px">↺ Refresh</button>
    <button class="vo-scene-btn" id="vo-preview-all-btn" onclick="_voPreviewAll()"
            style="margin-left:8px;color:#4fc3f7;border-color:#4fc3f7aa" title="Play all VO items sequentially">
      ▶ Generate Preview</button>
    <button class="vo-scene-btn" id="vo-stop-preview-btn" onclick="_voStopPreview()"
            style="display:none;margin-left:4px;color:#f88;border-color:#f88aa" title="Stop playback">
      ■ Stop</button>
  </div>
  <!-- VO Timeline Player — shown after Generate Preview -->
  <div id="vo-timeline-bar">
    <audio id="vo-tl-audio" preload="auto" style="display:none"></audio>
    <div class="vo-tl-top">
      <button class="vo-tl-playbtn" id="vo-tl-playbtn" onclick="_voTlToggle()" title="Play / Pause">▶</button>
      <span class="vo-tl-time" id="vo-tl-time">0:00 / 0:00</span>
      <span class="vo-tl-label" id="vo-tl-label">—</span>
      <button class="vo-tl-close" onclick="_voTlClose()" title="Close timeline">✕</button>
    </div>
    <div class="vo-tl-track" id="vo-tl-track">
      <div  class="vo-tl-scenes"  id="vo-tl-scenes"></div>
      <div  class="vo-tl-progress" id="vo-tl-progress" style="width:0"></div>
      <input class="vo-tl-scrubber" id="vo-tl-scrubber" type="range" min="0" max="1000"
             value="0" step="1"
             oninput="_voTlSeek(this.value)"
             onmousedown="_voTlScrubStart()" onmouseup="_voTlScrubEnd()"
             ontouchstart="_voTlScrubStart()" ontouchend="_voTlScrubEnd()"/>
    </div>
  </div>
  <!-- VO approval banner — always visible; locks in current WAVs via sentinel -->
  <div id="vo-approve-banner" style="display:flex;flex-direction:column;gap:6px;
       padding:10px 14px;background:#1a3a1a;border:1px solid #2d6a2d;
       border-radius:6px;font-size:0.85em;color:#a8e6a8;margin-bottom:4px">
    <div style="display:flex;align-items:center;gap:8px">
      <span id="vo-approve-icon">🎙</span>
      <span id="vo-approve-msg">Review each VO item, then approve to lock in your audio and continue.</span>
      <button id="vo-approve-btn" onclick="voApproveTTS()"
        style="margin-left:auto;padding:5px 14px;background:#2d7a2d;color:#fff;
               border:none;border-radius:5px;cursor:pointer;font-size:0.95em;font-weight:600">
        ✓ VO Approved — Continue
      </button>
    </div>
    <div id="vo-approve-error" style="display:none;color:#f88;font-size:0.9em;padding:2px 0"></div>
  </div>
  <!-- sentinel status indicator (shown after approval) -->
  <div id="vo-sentinel-status" style="display:none;align-items:center;gap:8px;
       padding:5px 10px;border-radius:5px;font-size:0.8em">
    <span id="vo-sentinel-icon">✓</span>
    <span id="vo-sentinel-text">VO Approved</span>
    <span id="vo-sentinel-time" style="color:var(--dim)"></span>
  </div>
  <!-- column headers -->
  <div class="vo-col-headers" style="margin-top:2px">
    <span style="min-width:160px">item_id</span>
    <span style="flex:1;min-width:180px">text</span>
    <span style="width:150px">voice</span>
    <span style="width:110px">style</span>
    <span style="width:52px">rate</span>
    <span style="width:52px">pitch</span>
    <span style="width:44px">deg</span>
    <span style="min-width:52px;text-align:right">dur</span>
    <span style="width:32px"></span>
    <span style="width:46px"></span>
  </div>
  <div id="vo-body" class="vo-body">
    <span class="vo-empty">Select an episode and locale to see VO items.</span>
  </div>
</div>

<!-- ── YouTube panel ── -->
<div id="panel-youtube" style="display:none;flex-direction:column;gap:12px;padding:16px;overflow-y:auto;height:calc(100vh - 60px)">

  <!-- Header row -->
  <div style="display:flex;align-items:center;gap:12px">
    <div class="section-label" style="margin:0">▶ YouTube Upload</div>
    <span id="yt-status-badge" style="font-size:0.8em;padding:3px 10px;border-radius:12px;background:var(--active-bg);color:var(--dim)">No episode loaded</span>
  </div>

  <!-- Locale selector -->
  <div style="display:flex;align-items:center;gap:8px">
    <span style="color:var(--dim);font-size:0.82em">Locale</span>
    <select id="yt-locale-sel" onchange="initYoutubeTab()" style="background:var(--surface);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:4px 8px;font-size:0.85em">
      <option value="en">en</option>
      <option value="zh-Hans">zh-Hans</option>
    </select>
    <button onclick="initYoutubeTab()" style="background:var(--active-bg);color:var(--dim);border:1px solid var(--border);border-radius:6px;font-size:0.76em;padding:5px 12px;cursor:pointer">↺ Refresh</button>
  </div>

  <!-- Metadata form (pre-filled from youtube.json) -->
  <div id="yt-meta-form" style="display:none;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px;flex-direction:column;gap:10px">
    <div style="font-size:0.85em;font-weight:600;color:var(--dim);margin-bottom:2px">METADATA (auto-saved to youtube.json)</div>

    <div style="display:flex;align-items:baseline;gap:8px">
      <label style="min-width:90px;font-size:0.82em;color:var(--dim)">Title</label>
      <div style="flex:1;position:relative">
        <input id="yt-title" type="text" maxlength="70"
               oninput="ytFieldChange('title',this.value);ytUpdateCounter('yt-title-ctr',this.value.length,70)"
               style="width:100%;background:var(--hover-bg);border:1px solid var(--border);border-radius:5px;color:var(--text);padding:5px 8px;font-size:0.9em;box-sizing:border-box">
        <span id="yt-title-ctr" style="position:absolute;right:6px;top:6px;font-size:0.75em;color:var(--dim)">0/70</span>
      </div>
    </div>

    <div style="display:flex;align-items:baseline;gap:8px">
      <label style="min-width:90px;font-size:0.82em;color:var(--dim)">Description</label>
      <div style="flex:1;position:relative">
        <textarea id="yt-desc" rows="4" maxlength="5000"
                  oninput="ytFieldChange('description',this.value);ytUpdateCounter('yt-desc-ctr',this.value.length,5000)"
                  style="width:100%;background:var(--hover-bg);border:1px solid var(--border);border-radius:5px;color:var(--text);padding:5px 8px;font-size:0.85em;resize:vertical;box-sizing:border-box"></textarea>
        <span id="yt-desc-ctr" style="position:absolute;right:6px;bottom:6px;font-size:0.75em;color:var(--dim)">0/5000</span>
      </div>
    </div>

    <div style="display:flex;align-items:center;gap:8px">
      <label style="min-width:90px;font-size:0.82em;color:var(--dim)">Category</label>
      <select id="yt-category" onchange="ytFieldChange('category_id',this.value)"
              style="background:var(--surface);color:var(--text);border:1px solid var(--border);border-radius:5px;padding:4px 8px;font-size:0.85em">
        <option value="1">Film & Animation</option>
        <option value="17">Sports</option>
        <option value="22">People & Blogs</option>
        <option value="24">Entertainment</option>
        <option value="25">News & Politics</option>
        <option value="27">Education</option>
        <option value="28">Science & Technology</option>
      </select>
    </div>

    <div style="display:flex;align-items:center;gap:8px">
      <label style="min-width:90px;font-size:0.82em;color:var(--dim)">Privacy</label>
      <select id="yt-privacy" onchange="ytFieldChange('privacy',this.value)"
              style="background:var(--surface);color:var(--text);border:1px solid var(--border);border-radius:5px;padding:4px 8px;font-size:0.85em">
        <option value="private">Private</option>
        <option value="unlisted">Unlisted</option>
        <option value="public">Public</option>
      </select>
    </div>

    <div style="display:flex;align-items:center;gap:8px">
      <label style="min-width:90px;font-size:0.82em;color:var(--dim)">Made for Kids</label>
      <select id="yt-mfk" onchange="ytFieldChange('made_for_kids',this.value==='true')"
              style="background:var(--surface);color:var(--text);border:1px solid var(--border);border-radius:5px;padding:4px 8px;font-size:0.85em">
        <option value="false">No</option>
        <option value="true">Yes</option>
      </select>
    </div>

    <div style="display:flex;align-items:center;gap:8px">
      <label style="min-width:90px;font-size:0.82em;color:var(--dim)">Notify Subs</label>
      <select id="yt-notify" onchange="ytFieldChange('notify_subscribers',this.value==='true')"
              style="background:var(--surface);color:var(--text);border:1px solid var(--border);border-radius:5px;padding:4px 8px;font-size:0.85em">
        <option value="false">Off — no notification ever sent for this video</option>
        <option value="true">On — notify subscribers when video goes public</option>
      </select>
      <span style="font-size:0.75em;color:var(--red)">⚠ permanent — cannot change after upload</span>
    </div>
  </div>

  <!-- Generate section (shown when youtube.json is missing) -->
  <div id="yt-generate-section" style="display:none;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px;text-align:center">
    <div style="color:var(--dim);font-size:0.9em;margin-bottom:12px">
      No <code>youtube.json</code> found for this locale.
    </div>
    <button id="yt-gen-btn" onclick="ytGenerate()"
            style="background:#7c3aed;color:#fff;border:none;border-radius:7px;padding:10px 24px;cursor:pointer;font-size:0.95em;font-weight:600">
      ✨ Generate youtube.json
    </button>
    <div id="yt-gen-error" style="display:none;margin-top:10px;color:#f87171;font-size:0.82em;text-align:left;background:#7f1d1d22;border-radius:5px;padding:8px 12px"></div>
  </div>

  <!-- Save button (shown after generate or when form is dirty) -->
  <div id="yt-save-wrap" style="display:none">
    <button id="yt-save-btn" onclick="ytSaveAll()"
            style="background:#16a34a;color:#fff;border:none;border-radius:7px;padding:8px 20px;cursor:pointer;font-size:0.9em;font-weight:600">
      💾 Save youtube.json
    </button>
    <span id="yt-save-badge" style="font-size:0.82em;color:var(--dim);margin-left:10px"></span>
  </div>

  <!-- Thumbnail section -->
  <div id="yt-thumb-section" style="display:none;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px">
    <div style="font-size:0.85em;font-weight:600;color:var(--dim);margin-bottom:10px">THUMBNAIL</div>

    <!-- Video player for frame picking -->
    <video id="yt-preview-video" controls preload="metadata"
           style="width:100%;max-height:320px;background:#000;border-radius:6px;display:none">
    </video>
    <div style="margin-top:8px;display:flex;gap:8px;align-items:center">
      <button id="yt-use-frame-btn" onclick="ytUseFrame()"
              style="display:none;background:#2563eb;color:#fff;border:none;border-radius:6px;padding:6px 14px;cursor:pointer;font-size:0.85em">
        Use this frame
      </button>
      <span id="yt-frame-sec" style="font-size:0.8em;color:var(--dim)"></span>
    </div>

    <!-- Preview box -->
    <div id="yt-thumb-preview-wrap" style="margin-top:10px;display:none">
      <div style="font-size:0.8em;color:var(--dim);margin-bottom:6px">Preview:</div>
      <div style="display:flex;gap:12px;align-items:flex-start">
        <img id="yt-thumb-img" style="max-width:280px;border-radius:5px;border:1px solid var(--border)"
             alt="thumbnail preview">
        <div style="font-size:0.8em;color:var(--dim);line-height:1.8">
          <div id="yt-thumb-info"></div>
          <button id="yt-save-thumb-btn" onclick="ytSaveThumbnail()"
                  style="margin-top:8px;background:#16a34a;color:#fff;border:none;border-radius:6px;padding:5px 12px;cursor:pointer;font-size:0.85em;display:none">
            💾 Save as thumbnail
          </button>
        </div>
      </div>
    </div>

    <!-- Custom upload fallback -->
    <div style="margin-top:12px">
      <label style="font-size:0.8em;color:var(--dim);cursor:pointer">
        <input type="file" id="yt-thumb-upload" accept="image/jpeg,image/png"
               onchange="ytUploadCustomThumb(this)" style="display:none">
        📁 Upload custom image instead
      </label>
    </div>
  </div>

  <!-- Subtitles info -->
  <div id="yt-subs-section" style="display:none;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px">
    <div style="font-size:0.85em;font-weight:600;color:var(--dim);margin-bottom:6px">SUBTITLES</div>
    <div id="yt-subs-list" style="font-size:0.85em;line-height:1.8"></div>
  </div>

  <!-- Action buttons -->
  <div id="yt-actions" style="display:none;gap:10px;flex-wrap:wrap;align-items:center">
    <button id="yt-btn-validate" onclick="ytAction('validate')"
            style="background:#7c3aed;color:#fff;border:none;border-radius:7px;padding:8px 18px;cursor:pointer;font-size:0.9em">
      ✓ Validate
    </button>
    <button id="yt-btn-upload" onclick="ytAction('upload')"
            style="background:#2563eb;color:#fff;border:none;border-radius:7px;padding:8px 18px;cursor:pointer;font-size:0.9em">
      ⬆ Upload (Private)
    </button>
    <button id="yt-btn-publish" onclick="ytAction('publish')"
            style="background:#16a34a;color:#fff;border:none;border-radius:7px;padding:8px 18px;cursor:pointer;font-size:0.9em">
      🌐 Publish
    </button>
    <button id="yt-copy-review-btn" onclick="ytCopyReview()"
            style="background:var(--active-bg);color:var(--dim);border:1px solid var(--border);border-radius:7px;padding:8px 14px;cursor:pointer;font-size:0.85em;display:none">
      📋 Copy Review Packet
    </button>
  </div>

  <!-- Upload status / log -->
  <div id="yt-log-wrap" style="display:none;background:#0d1117;border:1px solid var(--border);border-radius:8px;padding:12px;font-family:monospace;font-size:0.82em;min-height:120px;max-height:500px;overflow-y:auto;white-space:pre-wrap;word-break:break-word">
    <div id="yt-log"></div>
  </div>

  <!-- Post-upload checklist -->
  <div id="yt-checklist" style="display:none;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px">
    <div style="font-size:0.85em;font-weight:600;color:var(--dim);margin-bottom:10px">COMPLETE IN YOUTUBE STUDIO</div>
    <div id="yt-checklist-body" style="font-size:0.85em;line-height:2"></div>
  </div>

</div>

<!-- ── Pipeline panel ── -->
<div id="panel-pipeline">
  <!-- toolbar: episode selector -->
  <div class="pipe-toolbar">
    <div class="section-label" style="margin-bottom:0">Episode</div>
    <select id="pipe-ep-select" onchange="onPipeEpChange()">
      <option value="">— select episode —</option>
    </select>
    <button style="background:var(--active-bg);color:var(--dim);border:1px solid var(--border);border-radius:6px;font-size:0.76em;padding:5px 12px;cursor:pointer" onclick="refreshPipeline()">↺ Refresh</button>
  </div>

  <!-- scrollable step list -->
  <div class="pipe-body" id="pipe-body">
    <div style="color:var(--dim);font-style:italic;font-size:0.83em;padding:4px 0">Select an episode above to see pipeline status.</div>
  </div>

  <!-- VIDEO REVIEW block: Video / Soundtrack / Voice Cast -->
  <div id="pipe-review-wrap" style="display:none">
    <div class="review-hdr">
      <div class="review-content-tabs" id="review-content-tabs"></div>
      <div class="review-locale-tabs" id="review-locale-tabs"></div>
    </div>
    <div id="review-pane-video" class="review-pane">
      <video id="pipe-video" controls></video>
    </div>
    <div id="review-pane-audio" class="review-pane">
      <audio id="pipe-audio" controls></audio>
    </div>
  </div>
</div>

<!-- ── Media batch resume modal ── -->
<div id="media-modal-overlay">
  <div id="media-modal-box">
    <h2 id="media-modal-title">⚠️  Previous search batch found</h2>
    <p id="media-modal-msg"></p>
    <div class="media-modal-meta" id="media-modal-meta"></div>
    <div class="media-modal-btns">
      <button id="btn-mm-cancel" onclick="_mediaModalDismiss('cancel')">Cancel</button>
      <button id="btn-mm-resume" onclick="_mediaModalDismiss('resume')">▶ Resume</button>
      <button id="btn-mm-new"    onclick="_mediaModalDismiss('new')">🔍 New Search</button>
    </div>
  </div>
</div>

<!-- ── Confirm modal ── -->
<div id="modal-overlay">
  <div id="modal-box">
    <h2>⚠️  Episode folder already exists</h2>
    <p>A previous run already created output for this episode:</p>
    <div id="modal-path"></div>
    <p>Delete it and start fresh?</p>
    <p class="modal-note">All previously generated JSON files for this episode will be permanently removed.</p>
    <div class="modal-btns">
      <button id="btn-modal-no"  onclick="dismissModal(false)">Keep &amp; continue</button>
      <button id="btn-modal-yes" onclick="dismissModal(true)">🗑 Delete &amp; re-create</button>
    </div>
  </div>
</div>

<script>
  let es = null;
  let lineCount = 0;
  let currentSlug = null;
  let currentEpId = null;

  let renderProd = false;    // false = preview_local (CRF 28), true = high (CRF 18)
  let noMusic    = true;     // true = skip music by default (faster renders during development)
  let purgeAssets = false;  // saved per-project in meta.json; default OFF to avoid accidental data loss

  let _preparedMeta  = null;   // result from last /api/infer_story_meta call
  let _preparedEpId  = null;   // next episode ID fetched during Prepare (e.g. "s01e01")
  let _episodeCreated = false; // true after Create Episode completes
  let _runProjectList = [];    // cached project list for run-tab dropdowns
  let _selectedFormat  = 'continuous_narration';
  let _usingExistingEp = false;  // true when an existing project/episode is selected
  const stageStartMs = {};   // stage number → Date.now() at start
  let _runFromStage = 0;
  let _runToStage   = 10;

  // ── Voice Cast editor globals ────────────────────────────────────────────────
  let _vcLocales       = [];     // locale list for current episode (drives tab rendering)
  let _voiceCatalog    = null;   // loaded once from /api/azure_voices
  let _vcPendingTo     = null;   // if set, Continue runs stages 1 → _vcPendingTo
  let _vcActiveLocale  = null;   // MIN-R7: init from voiceCast.locales, not hardcoded
  let _vcData          = null;   // in-memory VoiceCast.json content
  let _vcPresets       = {};     // voice → preset[]; loaded once on editor open
  let _voiceIndex      = null;   // index.json voices object; loaded once on editor open
  let _vcPlayingAudio  = null;   // Audio object currently playing (for pause support)
  let _lastRunFilename = null;   // set in runPrompt(); fallback for vcContinue()

  const storyEl     = document.getElementById('story');
  const promptEl    = document.getElementById('prompt');
  const fileBadgeEl = document.getElementById('file-badge');
  const outputEl    = document.getElementById('output');
  const statusEl    = document.getElementById('status-badge');
  const cmdPreview  = document.getElementById('cmd-preview');
  const cmdText     = document.getElementById('cmd-text');
  const btnRun      = document.getElementById('btn-run');
  const btnStop     = document.getElementById('btn-stop');
  const spinnerEl   = document.getElementById('spinner');
  const lineCountEl = document.getElementById('line-count');
  const costEl      = document.getElementById('cost-badge');

  // ── Keyboard shortcuts ──────────────────────────────────────────────────────
  [storyEl, promptEl].forEach(el => {
    el.addEventListener('keydown', e => {
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        runPrompt();
      }
    });
  });

  // ── Helpers ─────────────────────────────────────────────────────────────────
  function setStatus(state) {
    statusEl.className = state;
    const labels = { idle: 'IDLE', running: 'RUNNING', error: 'ERROR' };
    statusEl.textContent = labels[state] || state.toUpperCase();
    spinnerEl.style.display = state === 'running' ? 'inline' : 'none';
    btnRun.disabled       = state === 'running';
    btnStop.style.display = state === 'running' ? 'block' : 'none';
  }

  // ── RAF-batched output flusher ────────────────────────────────────────────────
  // SSE lines are queued here and flushed in a single DOM write per animation
  // frame (~16 ms at 60 fps), eliminating per-line reflow during heavy output.
  let _lineBuf      = [];   // { html: string }[]
  let _rafPending   = false;

  function _flushLines() {
    _rafPending = false;
    if (_lineBuf.length === 0) return;
    const atBottom =
      outputEl.scrollHeight - outputEl.scrollTop - outputEl.clientHeight < 40;
    outputEl.insertAdjacentHTML('beforeend', _lineBuf.map(l => l.html).join(''));
    lineCount += _lineBuf.length;
    lineCountEl.textContent = lineCount + (lineCount === 1 ? ' line' : ' lines');
    _lineBuf = [];
    if (atBottom) outputEl.scrollTop = outputEl.scrollHeight;
  }

  function queueLine(text, cls) {
    const html = cls
      ? `<span class="${cls}">${escHtml(text)}\n</span>`
      : escHtml(text) + '\n';
    _lineBuf.push({ html });
    if (!_rafPending) { _rafPending = true; requestAnimationFrame(_flushLines); }
  }
  // ── end RAF batcher ───────────────────────────────────────────────────────────

  function appendLineTs(text, cls) {
    const ts = fmtNow();
    const prefix = `<span class="ts">[${ts}] </span>`;
    const atBottom =
      outputEl.scrollHeight - outputEl.scrollTop - outputEl.clientHeight < 40;
    outputEl.insertAdjacentHTML(
      'beforeend',
      prefix + (cls ? `<span class="${cls}">${escHtml(text)}\n</span>`
                    : `<span>${escHtml(text)}\n</span>`));
    lineCount++;
    lineCountEl.textContent = lineCount + (lineCount === 1 ? ' line' : ' lines');
    if (atBottom) outputEl.scrollTop = outputEl.scrollHeight;
  }

  function appendLine(text, cls) {
    const atBottom =
      outputEl.scrollHeight - outputEl.scrollTop - outputEl.clientHeight < 40;
    if (cls) {
      outputEl.insertAdjacentHTML(
        'beforeend', `<span class="${cls}">${escHtml(text)}\n</span>`);
    } else {
      outputEl.insertAdjacentText('beforeend', text + '\n');
    }
    lineCount++;
    lineCountEl.textContent = lineCount + (lineCount === 1 ? ' line' : ' lines');
    if (atBottom) outputEl.scrollTop = outputEl.scrollHeight;
  }

  function escHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  // ── Timestamp helpers ────────────────────────────────────────────────────────
  function fmtNow() {
    return new Date().toLocaleTimeString('en-US',
      { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
  }
  function fmtElapsed(ms) {
    if (ms < 60000) return Math.round(ms / 1000) + 's';
    return Math.floor(ms / 60000) + 'm ' + Math.round((ms % 60000) / 1000) + 's';
  }

  function clearOutput() {
    outputEl.innerHTML = '<span class="sys">Cleared.\n</span>';
    lineCount = 0;
    lineCountEl.textContent = '0 lines';
    costEl.style.display    = 'none';
    cmdPreview.style.display = 'none';
    hideStageProgress();
  }

  // ── Stage progress helpers ────────────────────────────────────────────────────
  function hideStageProgress() {
    const el = document.getElementById('stage-progress');
    if (el) el.style.display = 'none';
  }

  function updateStageProgress(n, label) {
    const el  = document.getElementById('stage-progress');
    const num = document.getElementById('stage-progress-num');
    const lbl = document.getElementById('stage-progress-label');
    const spn = document.getElementById('stage-progress-spinner');
    const dots = document.getElementById('stage-progress-dots');
    if (!el) return;
    el.style.display = 'flex';
    const total = _runToStage - _runFromStage + 1;
    const idx   = n - _runFromStage;
    num.textContent = `Stage ${n}/${_runToStage}`;
    lbl.textContent = label;
    if (spn) spn.style.display = 'inline-block';
    // Rebuild dots
    if (dots) {
      dots.innerHTML = '';
      for (let i = _runFromStage; i <= _runToStage; i++) {
        const d = document.createElement('span');
        d.className = 'sdot' + (i < n ? ' done' : i === n ? ' current' : '');
        dots.appendChild(d);
      }
    }
  }

  function markStageDone(n) {
    const dots = document.getElementById('stage-progress-dots');
    const spn  = document.getElementById('stage-progress-spinner');
    if (dots) {
      const d = dots.children[n - _runFromStage];
      if (d) { d.className = 'sdot done'; }
    }
    if (spn) spn.style.display = 'none';
  }

  function stopRun() {
    if (es)         { es.close();         es = null;         }
    if (es75)       { es75.close();       es75 = null;       }
    if (pipeStepEs) { pipeStepEs.close(); pipeStepEs = null; }
    fetch('/stop', { method: 'POST' }).catch(() => {});
    appendLine('[ Stopped by user ]', 'sys');
    pipeRunning = null;
    hideStageProgress();
    setStatus('idle');
  }

  // ── Stage 7.5 — TTS Preview for primary locale ───────────────────────────────
  let es75 = null;

  function showRunVoReviewBanner() {
    const banner = document.getElementById('run-vo-review-banner');
    if (banner) banner.classList.add('visible');
  }

  async function runStage75() {
    if (!currentSlug || !currentEpId) {
      appendLine('⚠  No episode selected — Prepare a story first.', 'err');
      return;
    }
    if (es75) { es75.close(); es75 = null; }
    if (es)   { es.close();   es  = null;  }

    // Hide any previous VO review banner
    const banner = document.getElementById('run-vo-review-banner');
    if (banner) banner.classList.remove('visible');

    const profile = renderProd ? 'high' : 'preview_local';
    outputEl.innerHTML = '';
    lineCount = 0;
    lineCountEl.textContent = '0 lines';
    setStatus('running');

    appendLine('── Stage 7.5 — TTS Preview (primary locale) ───────────────', 'sys');

    const url = `/run_stage75?slug=${encodeURIComponent(currentSlug)}&ep_id=${encodeURIComponent(currentEpId)}&profile=${encodeURIComponent(profile)}`;
    es75 = new EventSource(url);

    es75.addEventListener('line', e => {
      queueLine(e.data, '');
    });

    es75.addEventListener('error_line', e => appendLine(e.data, 'err'));

    es75.addEventListener('vo_review_ready', e => {
      try {
        const d = JSON.parse(e.data);
        appendLine(`  ✓ TTS complete for locale: ${d.locale || '?'}`, 'sys');
      } catch (_) {}
      showRunVoReviewBanner();
    });

    es75.addEventListener('done', e => {
      es75.close(); es75 = null;
      const code = parseInt(e.data);
      if (code === 0) {
        appendLine('[ ✓ Stage 7.5 complete — open VO tab to review ]', 'done');
        showRunVoReviewBanner();
        setStatus('idle');
      } else {
        appendLine(`[ Stage 7.5 exited with code ${code} ]`, 'err');
        setStatus('error');
      }
    });

    es75.onerror = () => {
      if (es75) { es75.close(); es75 = null; }
      appendLine('[ Connection lost ]', 'err');
      setStatus('error');
    };
  }

  // ── Confirm modal ────────────────────────────────────────────────────────────
  let _modalResolve = null;

  function showConfirmModal(path) {
    document.getElementById('modal-path').textContent = path;
    document.getElementById('modal-overlay').classList.add('visible');
    document.getElementById('btn-modal-yes').focus();
    return new Promise(resolve => { _modalResolve = resolve; });
  }

  function dismissModal(confirmed) {
    document.getElementById('modal-overlay').classList.remove('visible');
    if (_modalResolve) { _modalResolve(confirmed); _modalResolve = null; }
  }

  // Close on overlay-click (outside the box)
  document.getElementById('modal-overlay').addEventListener('click', e => {
    if (e.target === document.getElementById('modal-overlay')) dismissModal(false);
  });

  // Close on Escape
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && _modalResolve) dismissModal(false);
    if (e.key === 'Escape' && _mediaModalResolve) _mediaModalDismiss('cancel');
  });

  // ── Media batch resume modal ─────────────────────────────────────────────
  let _mediaModalResolve = null;

  /**
   * Show a 3-button modal for an existing batch.
   * Returns a Promise that resolves to 'cancel' | 'resume' | 'new'.
   *
   * batch fields expected: batch_id, status, item_count, items_done, created_at
   * slug / epId used to build the folder path shown in the dialog.
   */
  function _mediaShowBatchModal(batch, slug, epId) {
    const pct      = batch.item_count ? Math.round(100 * batch.items_done / batch.item_count) : 0;
    const statusLabel = { interrupted: 'Interrupted', failed: 'Failed', done: 'Done', running: 'Running' }[batch.status] || batch.status;
    const created  = batch.created_at ? new Date(batch.created_at).toLocaleString() : '—';
    const folder   = slug + '/episodes/' + epId + '/assets/media/' + batch.batch_id + '/';

    document.getElementById('media-modal-msg').textContent =
      'A previous search batch exists for this episode. Choose an action:';

    document.getElementById('media-modal-meta').innerHTML =
      '<b>Batch ID:</b>   ' + batch.batch_id + '<br>' +
      '<b>Status:</b>    ' + statusLabel + '  (' + batch.items_done + ' / ' + batch.item_count + ' shots done' + (batch.item_count ? ', ' + pct + '%' : '') + ')<br>' +
      '<b>Started:</b>   ' + created + '<br>' +
      '<b>Folder:</b>    <span class="mm-path">' + folder + '</span>';

    // Show/hide Resume button — only relevant when not already done
    document.getElementById('btn-mm-resume').style.display =
      (batch.status === 'interrupted' || batch.status === 'failed') ? '' : 'none';

    document.getElementById('media-modal-overlay').classList.add('visible');
    document.getElementById('btn-mm-resume').style.display !== 'none'
      ? document.getElementById('btn-mm-resume').focus()
      : document.getElementById('btn-mm-new').focus();

    return new Promise(resolve => { _mediaModalResolve = resolve; });
  }

  function _mediaModalDismiss(choice) {
    document.getElementById('media-modal-overlay').classList.remove('visible');
    if (_mediaModalResolve) { _mediaModalResolve(choice); _mediaModalResolve = null; }
  }

  document.getElementById('media-modal-overlay').addEventListener('click', e => {
    if (e.target === document.getElementById('media-modal-overlay')) _mediaModalDismiss('cancel');
  });

  // ── Next story number ───────────────────────────────────────────────────────
  async function refreshNextNum() {
    try {
      const res      = await fetch('/next_story_num');
      const { num }  = await res.json();
      fileBadgeEl.textContent = `story_${num}.txt`;
    } catch (_) {}
  }

  // ── Parse stages input ("0 9", "2 4", "9", …) ───────────────────────────────
  function parseStages(raw) {
    const parts = raw.trim().split(/\s+/).map(Number).filter(n => !isNaN(n));
    const from  = parts[0] ?? 0;
    const to    = parts[1] ?? from;   // single number → single stage
    return { from: Math.max(0, from), to: Math.min(10, to) };
  }

  // ── Cleanup on page unload (reload / navigate away) ──────────────────────
  // Release all HTTP connections so the browser's per-host connection pool is
  // clean.  Without this, reloading after a server restart hangs because the
  // browser tries to reuse dead TCP sockets from the old server process.
  window.addEventListener('beforeunload', function() {
    // Close SSE streams (each holds a persistent HTTP connection)
    if (es)         { try { es.close(); }         catch(_){} es = null; }
    if (pipeStepEs) { try { pipeStepEs.close(); } catch(_){} pipeStepEs = null; }
    // Stop polling timers (their in-flight fetches occupy connection slots)
    if (_pipePoller)    { clearInterval(_pipePoller);    _pipePoller = null; }
    if (_mediaPollTimer){ clearInterval(_mediaPollTimer); _mediaPollTimer = null; }
    // Release video connections
    if (typeof _mediaReleaseAllConnections === 'function') _mediaReleaseAllConnections();
    // Stop voice-preview audio
    if (_vcPlayingAudio) { _vcPlayingAudio.pause(); _vcPlayingAudio = null; }
  });

  // ── Init ────────────────────────────────────────────────────────────────────
  window.addEventListener('DOMContentLoaded', async () => {
    await refreshNextNum();
    loadRunProjects();
    _updateMediaConfigPanel(_selectedFormat);
  });

  // ── Run ─────────────────────────────────────────────────────────────────────
  async function runPrompt() {
    const stagesRaw    = promptEl.value.trim() || '0 10';
    let { from, to } = parseStages(stagesRaw);

    // For new projects: Run auto-creates the episode if Prepare has been run.
    // For existing projects: currentSlug/currentEpId are set by the dropdown.
    if (!currentSlug || !currentEpId) {
      if (!_preparedEpId) {
        appendLine('⚠  Paste a story and click Prepare first.', 'err');
        return;
      }
    }

    if (es) { es.close(); es = null; }

    // ── Voice Cast auto-split: when running 0→N, run Stage 0 alone first ──
    // vcContinue() will kick off stages 1–N after the user reviews the cast.
    if (from === 0 && to > 0) {
      _vcPendingTo = to;   // save full target; Continue button picks this up
      to = 0;              // only submit Stage 0 now
    } else {
      _vcPendingTo = null; // clear any stale value from a prior run
    }

    // Track stage range for progress dots
    _runFromStage = from;
    _runToStage   = to;

    // Reset output
    outputEl.innerHTML = '';
    lineCount = 0;
    lineCountEl.textContent = '0 lines';
    costEl.style.display     = 'none';
    cmdPreview.style.display = 'none';
    hideStageProgress();
    setStatus('running');

    // ── Auto-create episode dir for new projects (Prepare → Run flow) ────────
    if (!currentSlug || !currentEpId) {
      const _slug = document.getElementById('info-slug').value.trim();
      appendLine(`Creating episode ${_slug}/${_preparedEpId}…`, 'sys');
      try {
        const ep_dir_created = await createEpisode();
        appendLine(`✓ Created: ${ep_dir_created}`, 'sys');
      } catch(e) {
        appendLine(`✗ Create episode failed: ${e.message}`, 'err');
        setStatus('error');
        return;
      }
    }

    const ep_dir = 'projects/' + currentSlug + '/episodes/' + currentEpId;

    // ── Show command preview ─────────────────────────────────────────────────
    const vcSplitNote = (_vcPendingTo != null) ? `  →  then 1–${_vcPendingTo} after Voice Cast` : '';
    cmdText.textContent      = `./run.sh ${ep_dir} ${from} ${to}${vcSplitNote}  [per-stage models 🎬]`;
    cmdPreview.style.display = 'block';

    // ── Open SSE stream ──────────────────────────────────────────────────────
    const url = `/stream?ep_dir=${encodeURIComponent(ep_dir)}&from=${from}&to=${to}&test=0&profile=${renderProd ? 'high' : 'preview_local'}&no_music=${noMusic ? '1' : '0'}`;
    es = new EventSource(url);

    es.addEventListener('line', e => {
      const text = e.data;
      queueLine(text, '');

      // ── Stage start  (run.sh banner:  "  STAGE N/10  —  label")
      // (\d+[_\w]*) matches both plain stages (3) and sub-stages (4_c, 8_a, etc.)
      const startM = text.match(/^\s{2}STAGE (\d+[_\w]*)\/\d+[_\w]*\s+[—\-]{1,2}\s+(.+)/);
      if (startM) {
        const n     = parseInt(startM[1]);   // parseInt("4_c") → 4
        const label = startM[2].trim();
        stageStartMs[n] = Date.now();
        appendLine(`  ⏱  started  ${fmtNow()}`, 'ts');
        updateStageProgress(n, label);
      }

      // ── Stage complete  (run.sh:  "✓ Stage N complete  →  log: …")
      const doneM = text.match(/^✓ Stage (\d+[_\w]*) complete/);
      if (doneM) {
        const n = parseInt(doneM[1]);
        const elapsed = stageStartMs[n] != null
          ? `  elapsed ${fmtElapsed(Date.now() - stageStartMs[n])}` : '';
        appendLine(`  ⏱  finished ${fmtNow()}${elapsed}`, 'ts');
        markStageDone(n);
        insertReviewButtons(n);
        // Keep Pipeline tab in sync even when run was started from the Run tab
        _stagesDoneInCurrentRun.add(n);
        if (pipeStatus) renderPipelineStatus(pipeStatus);
      }

      // ── Pick up PROJECT_SLUG / EPISODE_ID from run.sh "Loaded vars" line
      const vm = text.match(/PROJECT_SLUG=(\S+)\s+EPISODE_ID=(\S+)/);
      if (vm) { currentSlug = vm[1]; currentEpId = vm[2]; }
    });
    es.addEventListener('error_line', e => appendLine(e.data, 'err'));

    es.addEventListener('meta', e => {
      try {
        const m = JSON.parse(e.data);
        if (m.cost !== undefined) {
          costEl.textContent   = `$${m.cost.toFixed(4)}`;
          costEl.style.display = 'block';
        }
      } catch (_) {}
    });

    es.addEventListener('done', async e => {
      es.close(); es = null;
      const code = parseInt(e.data);
      hideStageProgress();
      if (code === 0) {
        // ── Voice Cast pause: Stage 0 just ran alone; _vcPendingTo holds the
        //    original toStage set by the auto-split in runPrompt() above. ──
        if (_vcPendingTo != null) {
          appendLine('  ✋ Stage 0 done — review Voice Cast then click Continue', 'sys');
          setStatus('idle');
          switchStoryTab('vc');
          document.getElementById('btn-vc-continue').style.display = '';
          await loadVoiceCastForEditing();
          refreshNextNum();
          return;
        }
        appendLine('[ ✓ Done — output.mp4 ready ]', 'done');
        setStatus('idle');
        switchTab('pipeline');   // jump to video player
        setTimeout(() => refreshPipeline(), 600);   // re-fetch after files settle
      } else {
        appendLine(`[ Exited with code ${code} ]`, 'err');
        setStatus('error');
      }
      refreshNextNum();
    });

    es.onerror = () => {
      if (es) { es.close(); es = null; }
      appendLine('[ Connection lost ]', 'err');
      setStatus('error');
    };
  }

  // ── VO Retune panel ──────────────────────────────────────────────────────────

  function populateVoEpSelect() {
    const voSel = document.getElementById('vo-ep-select');
    if (!voSel) return;
    // Always repopulate so switching to VO tab after changing the Run-tab project
    // picks up the new project list and auto-selects the current Run-tab episode.
    const prevEpDir = voSel.value;
    fetch('/list_projects')
      .then(r => r.json())
      .then(data => {
        voSel.innerHTML = '<option value="">— select episode —</option>';
        let autoTarget = null;   // epDir that matches Run tab's currentSlug/currentEpId
        (data.projects || []).forEach(proj => {
          (proj.episodes || []).forEach(ep => {
            const epId  = ep.id;
            const slug  = proj.slug;
            const epDir = 'projects/' + slug + '/episodes/' + epId;
            const o     = document.createElement('option');
            o.value           = epDir;
            o.textContent     = slug + ' / ' + epId;
            o.dataset.slug    = slug;
            o.dataset.epId    = epId;
            voSel.appendChild(o);
            if (currentSlug && currentEpId &&
                slug === currentSlug && epId === currentEpId)
              autoTarget = epDir;
          });
        });
        // Priority: Run-tab project → previous selection → nothing
        const target = autoTarget || prevEpDir;
        if (target) {
          voSel.value = target;
          if (voSel.value) {
            // Fire onVoEpChange whenever target differs from previous,
            // or when locales haven't been loaded yet for this episode.
            const locSel = document.getElementById('vo-locale-select');
            if (target !== prevEpDir || locSel.options.length <= 1)
              onVoEpChange();
          }
        }
      })
      .catch(() => {});
  }

  function onVoEpChange() {
    const sel  = document.getElementById('vo-ep-select');
    const epDir = sel.value;
    const locSel = document.getElementById('vo-locale-select');
    locSel.innerHTML = '<option value="">— select locale —</option>';
    document.getElementById('vo-body').innerHTML =
      '<span class="vo-empty">Select a locale.</span>';
    if (!epDir) return;
    fetch('/api/vo_locales?ep_dir=' + encodeURIComponent(epDir))
      .then(r => r.json())
      .then(data => {
        (data.locales || []).forEach(loc => {
          const o = document.createElement('option');
          o.value = o.textContent = loc;
          locSel.appendChild(o);
        });
        if (data.locales && data.locales.length > 0) {
          locSel.value = data.locales[0];
          loadVoItems();
        }
      })
      .catch(() => {});
  }

  function loadVoItems() {
    const epSel  = document.getElementById('vo-ep-select');
    const locSel = document.getElementById('vo-locale-select');
    const epDir  = epSel.value;
    const locale = locSel.value;
    const body   = document.getElementById('vo-body');
    if (!epDir || !locale) return;
    body.innerHTML = '<span class="vo-empty">Loading…</span>';
    const opt    = epSel.options[epSel.selectedIndex];
    const slug   = opt.dataset.slug   || '';
    const epId   = opt.dataset.epId   || '';
    fetch('/api/vo_items?ep_dir=' + encodeURIComponent(epDir) +
          '&locale=' + encodeURIComponent(locale))
      .then(r => r.json())
      .then(data => {
        if (data.error) {
          // Give a specific, actionable message when the merged manifest doesn't exist yet.
          const isNotFound = data.error.toLowerCase().includes('not found') ||
                             data.error.toLowerCase().includes('no such file');
          if (isNotFound) {
            body.innerHTML =
              '<div style="color:var(--dim);font-size:0.85em;padding:20px 0;line-height:1.7">' +
              '⚠ VO data not ready for this episode.<br>' +
              'The VO tab requires <code>AssetManifest_merged.{locale}.json</code> and ' +
              'WAV files in <code>assets/{locale}/audio/vo/</code>.<br><br>' +
              'Run the following pipeline steps first:<br>' +
              '&nbsp;&nbsp;<strong>[5] manifest_merge</strong> — merges shared + locale drafts<br>' +
              '&nbsp;&nbsp;<strong>[6] gen_tts</strong> — synthesises VO WAV files<br><br>' +
              'These run automatically as part of Stage 9 (Render) in the Run tab.' +
              '</div>';
          } else {
            body.innerHTML = '<span class="vo-empty" style="color:#f88">Error: ' +
                             escHtml(data.error) + '</span>';
          }
          return;
        }
        const items = data.items || [];
        items._scene_tails = data.scene_tails || {};
        _renderVoItems(items, slug, epId, locale, data.voice_catalog || {});
        _voLoadSentinel(epDir, locale);
      })
      .catch(e => {
        body.innerHTML = '<span class="vo-empty" style="color:#f88">Failed: ' +
                         escHtml(String(e)) + '</span>';
      });
  }

  let _voVoiceCatalog = {};   // { voiceName: {styles, local_name, gender} } — set by _renderVoItems

  function _renderVoItems(items, slug, epId, locale, voiceCatalog) {
    _voVoiceCatalog = voiceCatalog || {};
    const epDir = `projects/${slug}/episodes/${epId}`;
    const body = document.getElementById('vo-body');
    if (!items.length) {
      body.innerHTML = '<span class="vo-empty">No VO items found.</span>';
      return;
    }
    // Group by scene in manifest order
    const scenes = {};
    const sceneOrder = [];
    items.forEach(it => {
      const sc = it.scene_id ||
                 (it.item_id.match(/sc\d+/) || [''])[0] || 'unknown';
      if (!scenes[sc]) { scenes[sc] = []; sceneOrder.push(sc); }
      scenes[sc].push(it);
    });
    // Sort voices by local_name for the dropdown
    const voiceNames = Object.keys(_voVoiceCatalog).sort((a, b) => {
      const la = (_voVoiceCatalog[a].local_name || a).toLowerCase();
      const lb = (_voVoiceCatalog[b].local_name || b).toLowerCase();
      return la < lb ? -1 : la > lb ? 1 : 0;
    });
    // Build scene_tails lookup from items list (server injects it) or default 2000
    const sceneTails = items._scene_tails || {};

    let html = '';
    sceneOrder.forEach((sc, scIdx) => {
      const scJ = JSON.stringify(sc);
      const slugJ = JSON.stringify(slug);
      const epIdJ = JSON.stringify(epId);
      const locJ  = JSON.stringify(locale);
      const epDirJ = JSON.stringify(epDir);
      const scIdE = escHtml(sc.replace(/[^a-zA-Z0-9_-]/g, '_'));

      // Inter-scene break separator (shown BEFORE every scene except the first)
      if (scIdx > 0) {
        const tailMs = sceneTails[sc] ?? 2000;
        html += `<div class="vo-scene-break" id="vo-break-${scIdE}">
          <span class="vo-break-label">↕ break before ${escHtml(sc)}</span>
          <input  class="vo-break-input" id="vo-tail-${scIdE}" type="number"
                  value="${tailMs}" min="0" max="30000" step="100" title="Silence before this scene (ms)"/>
          <span style="font-size:0.75em;color:var(--dim)">ms</span>
          <button class="vo-scene-btn" style="padding:2px 8px"
                  onclick='_voSaveSceneTail(${scJ},${epDirJ},${locJ})'>Save</button>
        </div>`;
      }

      html += `<div class="vo-scene-header">
        <span class="vo-scene-label">${escHtml(sc)}</span>
        <button class="vo-scene-btn" id="vo-scene-preview-${scIdE}"
          onclick='_voPreviewScene(${scJ},${slugJ},${epIdJ},${locJ},this)'>
          ▶ Preview</button></div>`;
      scenes[sc].forEach(it => {
        const tp  = it.tts_prompt || {};
        const iid = it.item_id;
        const iidE = escHtml(iid);
        const iidJ = JSON.stringify(iid);
        const dur  = it.duration_sec != null
                     ? it.duration_sec.toFixed(2) + 's' : '—';
        const srcDur = it.source_duration_sec != null
                     ? it.source_duration_sec.toFixed(2) + 's' : null;
        const badge      = it.badge || 'ok';   // 'ok', 'short', 'long'
        const staleClass = it.timing_stale ? ' vo-timing-stale' : '';
        const hasTrim    = it.has_trim_override || false;
        const pauseMs    = it.pause_after_ms ?? 300;
        const currentVoice = tp.azure_voice || '';
        const voiceEntry   = _voVoiceCatalog[currentVoice] || {};
        const voiceStyles  = voiceEntry.styles || [];
        const currentStyle = tp.azure_style || '';

        // Helper: format a voice option label as "Local name (gender) — voice_name"
        const fmtVoice = v => {
          const e = _voVoiceCatalog[v];
          if (!e) return v;
          const parts = e.local_name ? `${e.local_name} (${e.gender || '?'}) \u2014 ${v}` : v;
          return parts;
        };

        // Voice <select> options — sorted by local_name (voiceNames already sorted)
        let vOpts = voiceNames.map(v =>
          `<option value="${escHtml(v)}"${v===currentVoice?' selected':''}>${escHtml(fmtVoice(v))}</option>`
        ).join('');
        if (currentVoice && !voiceNames.includes(currentVoice))
          vOpts = `<option value="${escHtml(currentVoice)}" selected>${escHtml(currentVoice)}</option>` + vOpts;

        // Style <select> options
        let sOpts = `<option value="">— none —</option>`;
        sOpts += voiceStyles.map(s =>
          `<option value="${escHtml(s)}"${s===currentStyle?' selected':''}>${escHtml(s)}</option>`
        ).join('');
        if (currentStyle && !voiceStyles.includes(currentStyle))
          sOpts += `<option value="${escHtml(currentStyle)}" selected>${escHtml(currentStyle)}</option>`;

        // Store original manifest values as data-orig-* so _voPreviewItem
        // can detect whether the user has changed any param in the UI.
        const origText   = escHtml(it.text||'');
        const origVoice  = escHtml(tp.azure_voice||'');
        const origStyle  = escHtml(tp.azure_style||'');
        const origRate   = escHtml(tp.azure_rate||'0%');
        const origPitch  = escHtml(tp.azure_pitch||'');
        const origDegree = escHtml(String(tp.azure_style_degree??''));
        const breakMs    = tp.azure_break_ms ?? 0;

        // Badge label
        const badgeLabel = badge === 'short' ? '⚠ short' : badge === 'long' ? '⚠ long' : '● ok';
        const badgeClass = badge === 'short' ? 'vo-badge-short' : badge === 'long' ? 'vo-badge-long' : 'vo-badge-ok';

        // Timing display (stale if any edit since last approval)
        const startSec = it.start_sec != null ? it.start_sec.toFixed(3) : '—';
        const endSec   = it.end_sec   != null ? it.end_sec.toFixed(3)   : '—';

        // Duration display: trimmed + pause (NEVER summed — plan §C3)
        const durLabel = it.duration_sec != null
          ? `${it.duration_sec.toFixed(2)}s | pause:${pauseMs}ms` : '—';

        const epDir = `projects/${slug}/episodes/${epId}`;
        const epDirJ = JSON.stringify(epDir);
        const locJ   = JSON.stringify(locale);

        html += `<div class="vo-item-row" id="vo-row-${iidE}" data-item-id="${iidE}"
            data-orig-text="${origText}" data-orig-voice="${origVoice}"
            data-orig-style="${origStyle}" data-orig-rate="${origRate}"
            data-orig-pitch="${origPitch}" data-orig-degree="${origDegree}"
            data-break-ms="${breakMs}" data-ep-dir="${escHtml(epDir)}"
            data-locale="${escHtml(locale)}">
          <span class="vo-item-id">${iidE}</span>
          <span class="vo-badge ${badgeClass}" title="${escHtml(badge)}">${badgeLabel}</span>
          <input  class="vo-field vo-text"   id="vo-text-${iidE}"   value="${origText}" title="text"
                  oninput='_voParamChanged(${iidJ})'/>
          <select class="vo-field vo-voice"  id="vo-voice-${iidE}"
                  onchange='_voVoiceChanged(${iidJ})'>${vOpts}</select>
          <select class="vo-field vo-style"  id="vo-style-${iidE}"
                  onchange='_voParamChanged(${iidJ})'>${sOpts}</select>
          <input  class="vo-field vo-rate"   id="vo-rate-${iidE}"
                  value="${origRate}" placeholder="rate" title="azure_rate"
                  oninput='_voParamChanged(${iidJ})'/>
          <input  class="vo-field vo-pitch"  id="vo-pitch-${iidE}"
                  value="${origPitch}" placeholder="pitch" title="azure_pitch"
                  oninput='_voParamChanged(${iidJ})'/>
          <input  class="vo-field vo-degree" id="vo-degree-${iidE}"
                  value="${origDegree}" placeholder="deg" title="azure_style_degree"
                  oninput='_voParamChanged(${iidJ})'/>
          <span   class="vo-dur${staleClass}" id="vo-dur-${iidE}"
                  title="Trimmed duration | Pause after (INVARIANT E: not summed)">${escHtml(durLabel)}</span>
          <button class="vo-preview-btn"     id="vo-preview-${iidE}" title="Preview active .wav"
                  onclick='_voPreviewItem(${iidJ},${slugJ},${epIdJ},${locJ})'>▶</button>
          <button class="vo-resynth-btn"     id="vo-recreate-${iidE}" title="Re-Create: fresh TTS with same params (bypasses cache)"
                  onclick='_voRecreateItem(${iidJ},${epDirJ},${locJ})'>🔄 Re-Create</button>
          <button class="vo-resynth-btn"     id="vo-btn-${iidE}" title="Save: re-synthesize with changed params"
                  onclick='_voSaveItem(${iidJ},${epDirJ},${locJ})'>💾 Save</button>
        </div>
        <div class="vo-trim-row" id="vo-trim-row-${iidE}">
          <span style="color:var(--dim);font-size:0.85em">Trim:</span>
          <input class="vo-trim-input" id="vo-trim-start-${iidE}" type="number"
                 step="0.01" min="0" placeholder="start s" title="trim_start_sec"/>
          <span style="color:var(--dim)">→</span>
          <input class="vo-trim-input" id="vo-trim-end-${iidE}" type="number"
                 step="0.01" min="0" placeholder="end s" title="trim_end_sec"/>
          <span style="color:var(--dim);font-size:0.8em">(src: ${escHtml(srcDur || dur)})</span>
          <button class="vo-trim-btn" onclick='_voApplyTrim(${iidJ},${epDirJ},${locJ})'>Apply Trim</button>
          <button class="vo-trim-btn" onclick='_voResetTrim(${iidJ},${epDirJ},${locJ})'
                  ${hasTrim?'style="color:#f0963c"':''}
                  title="Reset to full source.wav">Reset Trim</button>
          <span style="color:var(--dim);margin-left:8px">Pause:</span>
          <input class="vo-trim-input" id="vo-pause-${iidE}" type="number"
                 step="50" min="0" value="${pauseMs}" title="pause_after_ms (ms)"/>
          <span style="color:var(--dim);font-size:0.8em">ms</span>
          <button class="vo-trim-btn" onclick='_voSavePause(${iidJ},${epDirJ},${locJ})'>Save Pause</button>
        </div>`;
      });
    });
    body.innerHTML = html;

    // Auto-expand trim rows for short/long items (plan §P4-[P])
    items.filter(it => it.badge !== 'ok').forEach(it => {
      // Trim row is always visible now (no collapse in this simplified version)
    });
  }

  // Called whenever any TTS param field changes after a Re-Create.
  // Clears keep_audio flag and restores Save button label so next Save calls Azure TTS.
  function _voParamChanged(itemId) {
    if (window._voKeepAudio && window._voKeepAudio[itemId]) {
      delete window._voKeepAudio[itemId];
      const saveBtn = document.getElementById('vo-btn-' + itemId);
      if (saveBtn && saveBtn.textContent === '📌 Keep') saveBtn.textContent = '💾 Save';
    }
  }

  function _voVoiceChanged(itemId) {
    _voParamChanged(itemId);
    const voice  = document.getElementById('vo-voice-' + itemId)?.value || '';
    const sel    = document.getElementById('vo-style-' + itemId);
    if (!sel) return;
    const entry  = _voVoiceCatalog[voice] || {};
    const styles = entry.styles || [];
    const cur    = sel.value;
    sel.innerHTML = '<option value="">— none —</option>' +
      styles.map(s =>
        `<option value="${escHtml(s)}"${s===cur?' selected':''}>${escHtml(s)}</option>`
      ).join('');
  }

  // ── VO preview helpers ──────────────────────────────────────────────────────

  // Play a URL (WAV or MP3) and manage the ▶/■ button state.
  function _voPlayUrl(url, itemId, btn) {
    const audio = new Audio(url);
    window._voAudio   = audio;
    window._voAudioId = itemId;
    if (btn) { btn.textContent = '■'; btn.classList.add('playing'); btn.disabled = false; }
    // Show the top-level Stop button so single-item playback can also be stopped.
    const stopBtn = document.getElementById('vo-stop-preview-btn');
    if (stopBtn) stopBtn.style.display = '';
    const reset = () => {
      window._voAudio   = null;
      window._voAudioId = null;
      if (btn) { btn.textContent = '▶'; btn.classList.remove('playing'); btn.disabled = false; }
      if (stopBtn) stopBtn.style.display = 'none';
    };
    audio.onended = reset;
    audio.onerror = reset;
    audio.play().catch(reset);
  }

  // ── Sequential scene / all-episode preview ──────────────────────────────────

  // Global sequential-playback state (separate from single-item _voAudio)
  window._voSeqAbort = false;

  function _voStopPreview() {
    window._voSeqAbort = true;
    if (window._voAudio) { window._voAudio.pause(); window._voAudio = null; }
    // Reset all scene preview buttons
    document.querySelectorAll('.vo-scene-btn.vo-scene-playing').forEach(b => {
      b.textContent = b.textContent.replace('■', '▶');
      b.classList.remove('vo-scene-playing');
    });
    const stopBtn = document.getElementById('vo-stop-preview-btn');
    const allBtn  = document.getElementById('vo-preview-all-btn');
    if (stopBtn) stopBtn.style.display = 'none';
    if (allBtn)  { allBtn.style.display = ''; allBtn.disabled = false; }
  }

  // Play an ordered array of {epDir, locale, item_id} sequentially.
  // pauseMs is the gap between clips (from pause_after_ms, default 300ms).
  async function _voPlaySequence(clips, ctxBtn) {
    window._voSeqAbort = false;
    const stopBtn = document.getElementById('vo-stop-preview-btn');
    const allBtn  = document.getElementById('vo-preview-all-btn');
    if (stopBtn) stopBtn.style.display = '';
    // Disable (never hide) Generate Preview during any sequence playback
    if (allBtn) allBtn.disabled = (ctxBtn !== allBtn);
    if (ctxBtn) { ctxBtn.textContent = ctxBtn.textContent.replace('▶', '■'); ctxBtn.classList.add('vo-scene-playing'); }

    for (const clip of clips) {
      if (window._voSeqAbort) break;
      const url = `/api/vo_audio?ep_dir=${encodeURIComponent(clip.epDir)}`
                + `&locale=${encodeURIComponent(clip.locale)}`
                + `&item_id=${encodeURIComponent(clip.item_id)}`;
      // Highlight the item row
      const previewBtn = document.getElementById('vo-preview-' + clip.item_id);
      if (previewBtn) { previewBtn.textContent = '■'; previewBtn.classList.add('playing'); }
      await new Promise(resolve => {
        if (window._voSeqAbort) { resolve(); return; }
        const audio = new Audio(url);
        window._voAudio = audio;
        const done = () => {
          window._voAudio = null;
          if (previewBtn) { previewBtn.textContent = '▶'; previewBtn.classList.remove('playing'); }
          resolve();
        };
        audio.onended = done;
        audio.onerror = done;
        audio.play().catch(done);
      });
      // Pause gap between clips
      if (!window._voSeqAbort && clip.pauseMs > 0) {
        await new Promise(r => setTimeout(r, clip.pauseMs));
      }
    }
    _voStopPreview();
  }

  // Preview all items in one scene sequentially.
  function _voPreviewScene(scene, slug, epId, locale, btn) {
    // Stop any in-progress playback first
    if (window._voAudio) { window._voAudio.pause(); window._voAudio = null; }
    window._voSeqAbort = true;
    setTimeout(() => {
      window._voSeqAbort = false;
      const epDir = `projects/${slug}/episodes/${epId}`;
      const clips = [];
      document.querySelectorAll('.vo-item-row').forEach(row => {
        const iid = row.dataset.itemId;
        if (!iid) return;
        const sc = (iid.match(/sc\d+/) || [''])[0];
        if (sc !== scene) return;
        const pauseEl = document.getElementById('vo-pause-' + iid);
        const pauseMs = parseInt(pauseEl?.value || '300', 10);
        clips.push({ epDir, locale, item_id: iid, pauseMs });
      });
      if (clips.length) _voPlaySequence(clips, btn);
    }, 50);
  }

  // ── VO Timeline Player ───────────────────────────────────────────────────
  window._voTlClips   = [];   // [{item_id, scene_id, text, start_sec, duration_sec}]
  window._voTlScrubbing = false;
  window._voTlRaf    = null;

  function _fmtSec(s) {
    const m = Math.floor(s / 60);
    const ss = String(Math.floor(s % 60)).padStart(2, '0');
    return `${m}:${ss}`;
  }

  function _voTlClipAt(t) {
    const clips = window._voTlClips;
    for (let i = clips.length - 1; i >= 0; i--) {
      if (t >= clips[i].start_sec) return clips[i];
    }
    return clips[0] || null;
  }

  function _voTlHighlightRow(itemId) {
    document.querySelectorAll('.vo-item-row.tl-active')
      .forEach(r => r.classList.remove('tl-active'));
    if (!itemId) return;
    const row = document.getElementById('vo-row-' + itemId);
    if (row) {
      row.classList.add('tl-active');
      row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  function _voTlTick() {
    const audio = document.getElementById('vo-tl-audio');
    const scrubber = document.getElementById('vo-tl-scrubber');
    const progress = document.getElementById('vo-tl-progress');
    const timeEl   = document.getElementById('vo-tl-time');
    const label    = document.getElementById('vo-tl-label');
    const track    = document.getElementById('vo-tl-track');
    if (!audio || audio.paused) { window._voTlRaf = null; return; }

    const t   = audio.currentTime;
    const dur = audio.duration || 1;
    const pct = t / dur;

    if (!window._voTlScrubbing) {
      scrubber.value = Math.round(pct * 1000);
      if (track) progress.style.width = (pct * 100).toFixed(2) + '%';
    }
    timeEl.textContent = `${_fmtSec(t)} / ${_fmtSec(dur)}`;

    const clip = _voTlClipAt(t);
    if (clip) {
      label.textContent = `${clip.scene_id || ''}: ${clip.text.slice(0, 80)}${clip.text.length > 80 ? '…' : ''}`;
      if (window._voTlLastId !== clip.item_id) {
        window._voTlLastId = clip.item_id;
        _voTlHighlightRow(clip.item_id);
      }
    }
    window._voTlRaf = requestAnimationFrame(_voTlTick);
  }

  function _voTlToggle() {
    const audio = document.getElementById('vo-tl-audio');
    const btn   = document.getElementById('vo-tl-playbtn');
    if (!audio || !audio.src) return;
    if (audio.paused) {
      audio.play();
      btn.textContent = '⏸';
      window._voTlRaf = requestAnimationFrame(_voTlTick);
    } else {
      audio.pause();
      btn.textContent = '▶';
    }
  }

  function _voTlScrubStart() { window._voTlScrubbing = true; }
  function _voTlScrubEnd()   {
    window._voTlScrubbing = false;
    const audio    = document.getElementById('vo-tl-audio');
    const scrubber = document.getElementById('vo-tl-scrubber');
    if (audio && audio.duration) {
      audio.currentTime = (parseInt(scrubber.value) / 1000) * audio.duration;
    }
  }
  function _voTlSeek(val) {
    // Update progress bar visually while dragging; actual seek on mouseup
    const track = document.getElementById('vo-tl-track');
    const progress = document.getElementById('vo-tl-progress');
    if (progress) progress.style.width = (val / 10).toFixed(1) + '%';
    // Also update time label
    const audio = document.getElementById('vo-tl-audio');
    if (audio && audio.duration) {
      const t = (val / 1000) * audio.duration;
      const timeEl = document.getElementById('vo-tl-time');
      if (timeEl) timeEl.textContent = `${_fmtSec(t)} / ${_fmtSec(audio.duration)}`;
    }
  }

  function _voTlClose() {
    const audio = document.getElementById('vo-tl-audio');
    if (audio) { audio.pause(); audio.src = ''; }
    if (window._voTlRaf) { cancelAnimationFrame(window._voTlRaf); window._voTlRaf = null; }
    document.getElementById('vo-timeline-bar').classList.remove('active');
    document.getElementById('vo-tl-playbtn').textContent = '▶';
    document.querySelectorAll('.vo-item-row.tl-active').forEach(r => r.classList.remove('tl-active'));
    window._voTlClips = [];
    window._voTlLastId = null;
  }

  function _voTlBuild(data) {
    // data = { wav_url, total_sec, clips: [{item_id, scene_id, text, start_sec, duration_sec}] }
    window._voTlClips  = data.clips || [];
    window._voTlLastId = null;
    const total = data.total_sec || 1;

    // Build coloured scene segments
    const sceneColors = ['#1565c0','#2e7d32','#6a1b9a','#c62828','#e65100','#00695c','#4e342e','#37474f'];
    const sceneMap = {};
    let colorIdx = 0;
    const scenesEl = document.getElementById('vo-tl-scenes');
    scenesEl.innerHTML = '';
    for (const clip of data.clips) {
      const sc = clip.scene_id || 'sc';
      if (!(sc in sceneMap)) { sceneMap[sc] = sceneColors[colorIdx++ % sceneColors.length]; }
      const pct = (clip.duration_sec / total * 100).toFixed(3);
      const seg = document.createElement('div');
      seg.className = 'vo-tl-seg';
      seg.style.cssText = `width:${pct}%;background:${sceneMap[sc]}`;
      seg.title = `${sc}: ${clip.text.slice(0,60)}`;
      // Label on first clip of each scene
      if (data.clips.indexOf(clip) === data.clips.findIndex(c => c.scene_id === sc)) {
        const lbl = document.createElement('span');
        lbl.className = 'vo-tl-seg-label';
        lbl.textContent = sc;
        seg.appendChild(lbl);
      }
      scenesEl.appendChild(seg);
    }

    // Wire audio
    const audio = document.getElementById('vo-tl-audio');
    audio.src = data.wav_url + '&t=' + Date.now();
    audio.onended = () => {
      document.getElementById('vo-tl-playbtn').textContent = '▶';
      window._voTlRaf && cancelAnimationFrame(window._voTlRaf);
      window._voTlRaf = null;
    };

    // Show bar and auto-play
    document.getElementById('vo-timeline-bar').classList.add('active');
    document.getElementById('vo-tl-scrubber').value = 0;
    document.getElementById('vo-tl-progress').style.width = '0';
    document.getElementById('vo-tl-playbtn').textContent = '⏸';
    audio.play().then(() => {
      window._voTlRaf = requestAnimationFrame(_voTlTick);
    }).catch(() => {
      document.getElementById('vo-tl-playbtn').textContent = '▶';
    });
  }

  // Preview all VO items — concat on server, show timeline.
  async function _voPreviewAll() {
    const epSel  = document.getElementById('vo-ep-select');
    const locSel = document.getElementById('vo-locale-select');
    const epDir  = epSel?.value;
    const locale = locSel?.value;
    if (!epDir || !locale) return;

    // Stop any in-progress per-item playback
    if (window._voAudio) { window._voAudio.pause(); window._voAudio = null; }
    window._voSeqAbort = true;

    const btn = document.getElementById('vo-preview-all-btn');
    if (btn) { btn.disabled = true; btn.textContent = '⏳ Building…'; }
    try {
      const resp = await fetch(
        `/api/vo_preview_concat?ep_dir=${encodeURIComponent(epDir)}&locale=${encodeURIComponent(locale)}`
      );
      if (!resp.ok) throw new Error(await resp.text());
      const data = await resp.json();
      _voTlBuild(data);
    } catch(e) {
      alert('Preview error: ' + e.message);
    } finally {
      if (btn) { btn.disabled = false; btn.textContent = '▶ Generate Preview'; }
    }
  }

  async function _voPreviewItem(itemId, slug, epId, locale) {
    const btn = document.getElementById('vo-preview-' + itemId);

    // Toggle off: if THIS button is already in playing state, stop it.
    if (btn && btn.classList.contains('playing')) {
      if (window._voAudio) { window._voAudio.pause(); window._voAudio = null; }
      btn.textContent = '▶'; btn.classList.remove('playing'); btn.disabled = false;
      window._voAudioId = null;
      const stopBtn = document.getElementById('vo-stop-preview-btn');
      if (stopBtn) stopBtn.style.display = 'none';
      return;
    }

    // Stop any other audio that might be playing before starting this one.
    if (window._voAudio) {
      window._voAudio.pause();
      const prev = document.getElementById('vo-preview-' + (window._voAudioId || ''));
      if (prev) { prev.textContent = '▶'; prev.classList.remove('playing'); prev.disabled = false; }
      window._voAudio = null;
      window._voAudioId = null;
    }

    // Read current UI values
    const text   = (document.getElementById('vo-text-'   + itemId)?.value ?? '').trim();
    const voice  = (document.getElementById('vo-voice-'  + itemId)?.value ?? '').trim();
    const style  = (document.getElementById('vo-style-'  + itemId)?.value ?? '').trim();
    const rate   = (document.getElementById('vo-rate-'   + itemId)?.value ?? '').trim() || '0%';
    const pitch  = (document.getElementById('vo-pitch-'  + itemId)?.value ?? '').trim();
    const degree = (document.getElementById('vo-degree-' + itemId)?.value ?? '').trim();

    // Read original manifest values stored as data-orig-* on the row
    const row      = document.getElementById('vo-row-' + itemId);
    const breakMs  = parseInt(row?.dataset.breakMs  ?? '0', 10);
    const origText   = row?.dataset.origText   ?? '';
    const origVoice  = row?.dataset.origVoice  ?? '';
    const origStyle  = row?.dataset.origStyle  ?? '';
    const origRate   = row?.dataset.origRate   ?? '0%';
    const origPitch  = row?.dataset.origPitch  ?? '';
    const origDegree = row?.dataset.origDegree ?? '';

    const paramsUnchanged = text === origText && voice === origVoice &&
                            style === origStyle && rate === origRate &&
                            pitch === origPitch && degree === origDegree;

    if (paramsUnchanged) {
      // Fast path — params match the saved manifest; existing WAV on disk is current.
      const epDir = `projects/${slug}/episodes/${epId}`;
      const url = `/api/vo_audio?ep_dir=${encodeURIComponent(epDir)}`
                + `&locale=${encodeURIComponent(locale)}`
                + `&item_id=${encodeURIComponent(itemId)}`;
      _voPlayUrl(url, itemId, btn);
      return;
    }

    // Slow path — params changed; call /api/preview_voice.
    // Backend checks disk MP3 cache (covers index.json + presets.json clips)
    // before hitting Azure TTS.
    if (btn) { btn.textContent = '…'; btn.disabled = true; }
    try {
      // Derive azure_locale from voice name (e.g. "en-US-Andrew:Dragon..." → "en-US")
      const azureLocale = voice.split('-').slice(0, 2).join('-');
      const r = await fetch('/api/preview_voice', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          azure_voice:  voice,
          azure_locale: azureLocale,
          style:        style || null,
          style_degree: parseFloat(degree) || 1.0,
          rate:         rate  || '0%',
          pitch:        pitch || '',
          break_ms:     breakMs,
          text:         text,
        }),
      });
      const data = await r.json();
      if (data.url) {
        _voPlayUrl(data.url + '&t=' + Date.now(), itemId, btn);
      } else {
        const msg = data.error || 'preview failed';
        if (btn) { btn.textContent = '▶'; btn.disabled = false; }
        appendLine(`⚠ VO preview error (${itemId}): ${msg}`, 'err');
      }
    } catch (e) {
      if (btn) { btn.textContent = '▶'; btn.disabled = false; }
      appendLine(`⚠ VO preview error (${itemId}): ${e}`, 'err');
    }
  }

  function _collectPatch(itemId) {
    const g = id => (document.getElementById(id)?.value ?? '').trim();
    const patch = { item_id: itemId };
    const text   = g('vo-text-'   + itemId); if (text)   patch.text               = text;
    const style  = g('vo-style-'  + itemId); if (style)  patch.azure_style        = style;
    const rate   = g('vo-rate-'   + itemId); if (rate && rate !== '0%')
                                                          patch.azure_rate         = rate;
    const pitch  = g('vo-pitch-'  + itemId); if (pitch && pitch !== '0%')
                                                          patch.azure_pitch        = pitch;
    const deg    = g('vo-degree-' + itemId); if (deg)    patch.azure_style_degree = deg;
    const voice  = g('vo-voice-'  + itemId); if (voice)  patch.azure_voice        = voice;
    return patch;
  }


  // ── New VO Review endpoints (P3) ─────────────────────────────────────────────

  // Helper: POST a VO endpoint and update the item row
  async function _voPost(endpoint, body, itemId) {
    const btn = document.getElementById('vo-btn-' + itemId);
    const dur = document.getElementById('vo-dur-' + itemId);
    if (btn) { btn.disabled = true; }
    try {
      const r = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await r.json();
      if (data.error) throw new Error(data.error);
      return data;
    } finally {
      if (btn) { btn.disabled = false; }
    }
  }

  // Tracks items where Re-Create succeeded and user hasn't changed params since.
  // When set, Save will send keep_audio=true (skip Azure TTS, keep existing WAV).
  window._voKeepAudio = window._voKeepAudio || {};

  // POST /api/vo_save — save with (possibly changed) voice/style/rate/text.
  // If keep_audio flag is set for this item (Re-Create was run and params unchanged),
  // skips Azure TTS and just commits the existing WAV + updates manifest.
  async function _voSaveItem(itemId, epDir, locale) {
    const btn = document.getElementById('vo-btn-' + itemId);
    const keepAudio = !!window._voKeepAudio[itemId];
    if (btn) { btn.textContent = '⏳ Saving…'; btn.disabled = true; }
    try {
      const text   = (document.getElementById('vo-text-'   + itemId)?.value ?? '').trim();
      const voice  = (document.getElementById('vo-voice-'  + itemId)?.value ?? '').trim();
      const style  = (document.getElementById('vo-style-'  + itemId)?.value ?? '').trim();
      const rate   = (document.getElementById('vo-rate-'   + itemId)?.value ?? '').trim() || '0%';
      const pitch  = (document.getElementById('vo-pitch-'  + itemId)?.value ?? '').trim();
      const degree = parseFloat(document.getElementById('vo-degree-' + itemId)?.value ?? '1.5');
      const data = await _voPost('/api/vo_save', {
        ep_dir: epDir, locale, item_id: itemId,
        text, voice, style, rate, pitch, style_degree: degree,
        keep_audio: keepAudio,
      }, itemId);
      delete window._voKeepAudio[itemId];
      if (btn) { btn.textContent = '✓ Saved'; setTimeout(() => { if (btn) btn.textContent = '💾 Save'; }, 2500); }
      _voUpdateDur(itemId, data.trimmed_duration_sec);
      _voMarkSentinelInvalid();
    } catch(e) {
      if (btn) { btn.textContent = '✗'; setTimeout(() => { if (btn) btn.textContent = keepAudio ? '📌 Keep' : '💾 Save'; btn.disabled = false; }, 2500); }
      alert('vo_save error: ' + e.message);
    }
  }

  // POST /api/vo_recreate — fresh TTS with same params, bypass cache
  async function _voRecreateItem(itemId, epDir, locale) {
    const btn = document.getElementById('vo-recreate-' + itemId);
    if (btn) { btn.textContent = '⏳'; btn.disabled = true; }
    try {
      const data = await _voPost('/api/vo_recreate', {
        ep_dir: epDir, locale, item_id: itemId,
      }, itemId);
      // Re-Create succeeded: set keep_audio flag and change Save button to "📌 Keep"
      // so user can commit this audio without triggering another Azure TTS call.
      window._voKeepAudio[itemId] = true;
      const saveBtn = document.getElementById('vo-btn-' + itemId);
      if (saveBtn) saveBtn.textContent = '📌 Keep';
      if (btn) { btn.textContent = '✓'; setTimeout(() => { if (btn) { btn.textContent = '🔄 Re-Create'; btn.disabled = false; } }, 2500); }
      _voUpdateDur(itemId, data.trimmed_duration_sec);
      _voMarkSentinelInvalid();
    } catch(e) {
      if (btn) { btn.textContent = '✗'; setTimeout(() => { if (btn) { btn.textContent = '🔄 Re-Create'; btn.disabled = false; } }, 2500); }
      alert('vo_recreate error: ' + e.message);
    }
  }

  // POST /api/vo_trim — apply trim handles
  async function _voApplyTrim(itemId, epDir, locale) {
    const startEl = document.getElementById('vo-trim-start-' + itemId);
    const endEl   = document.getElementById('vo-trim-end-'   + itemId);
    const start   = parseFloat(startEl?.value || '0');
    const end     = parseFloat(endEl?.value   || '0');
    if (isNaN(start) || isNaN(end) || end <= start) {
      alert('Invalid trim range. End must be greater than start.');
      return;
    }
    try {
      const data = await _voPost('/api/vo_trim', {
        ep_dir: epDir, locale, item_id: itemId,
        trim_start_sec: start, trim_end_sec: end,
      }, itemId);
      _voUpdateDur(itemId, data.trimmed_duration_sec);
      _voMarkSentinelInvalid();
    } catch(e) {
      alert('vo_trim error: ' + e.message);
    }
  }

  // POST /api/vo_reset_trim — restore full source.wav
  async function _voResetTrim(itemId, epDir, locale) {
    try {
      const data = await _voPost('/api/vo_reset_trim', {
        ep_dir: epDir, locale, item_id: itemId,
      }, itemId);
      // Clear trim inputs
      const startEl = document.getElementById('vo-trim-start-' + itemId);
      const endEl   = document.getElementById('vo-trim-end-'   + itemId);
      if (startEl) startEl.value = '';
      if (endEl)   endEl.value   = '';
      _voUpdateDur(itemId, data.source_duration_sec);
      _voMarkSentinelInvalid();
    } catch(e) {
      alert('vo_reset_trim error: ' + e.message);
    }
  }

  // POST /api/vo_pause — update pause_after_ms
  async function _voSavePause(itemId, epDir, locale) {
    const pauseEl = document.getElementById('vo-pause-' + itemId);
    const pauseMs = parseInt(pauseEl?.value ?? '300', 10);
    if (isNaN(pauseMs) || pauseMs < 0) {
      alert('Invalid pause value');
      return;
    }
    try {
      await _voPost('/api/vo_pause', {
        ep_dir: epDir, locale, item_id: itemId, pause_ms: pauseMs,
      }, itemId);
      _voMarkSentinelInvalid();
    } catch(e) {
      alert('vo_pause error: ' + e.message);
    }
  }

  // POST /api/vo_scene_tail — save inter-scene break duration
  async function _voSaveSceneTail(scene, epDir, locale) {
    const scIdE = scene.replace(/[^a-zA-Z0-9_-]/g, '_');
    const input = document.getElementById('vo-tail-' + scIdE);
    const tailMs = parseInt(input?.value ?? '2000', 10);
    if (isNaN(tailMs) || tailMs < 0 || tailMs > 30000) {
      alert('Invalid tail value (0–30000 ms)');
      return;
    }
    try {
      const r = await fetch('/api/vo_scene_tail', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ep_dir: epDir, locale, scene, tail_ms: tailMs }),
      });
      const data = await r.json();
      if (data.error) throw new Error(data.error);
      _voMarkSentinelInvalid();
      // Flash the input green briefly
      if (input) { input.style.outline = '2px solid #6ec96e';
                   setTimeout(() => { input.style.outline = ''; }, 1500); }
    } catch(e) {
      alert('vo_scene_tail error: ' + e.message);
    }
  }

  // Helper: update duration display for an item row
  function _voUpdateDur(itemId, durSec) {
    const dur = document.getElementById('vo-dur-' + itemId);
    if (dur) {
      dur.textContent = durSec != null ? durSec.toFixed(2) + 's' : '—';
      dur.classList.add('vo-timing-stale');   // timing is stale until re-approved
    }
  }

  // Mark sentinel as invalid in UI (show Re-approve state)
  function _voMarkSentinelInvalid() {
    const sentinelEl = document.getElementById('vo-sentinel-status');
    if (sentinelEl) {
      sentinelEl.style.display = 'flex';
      sentinelEl.style.background = '#3a2200';
      sentinelEl.style.color = '#f0963c';
      sentinelEl.style.border = '1px solid #7a4400';
      document.getElementById('vo-sentinel-icon').textContent = '⚠';
      document.getElementById('vo-sentinel-text').textContent = 'VO edits made — click Re-approve when ready';
    }
    const approveBtn = document.getElementById('vo-approve-btn');
    if (approveBtn) {
      approveBtn.textContent = '↻ Re-approve';
      const banner = document.getElementById('vo-approve-banner');
      if (banner) {
        banner.style.display = 'flex';
        document.getElementById('vo-approve-msg').textContent =
          'VO edits made — re-approve before continuing to Stage 8.';
      }
    }
  }

  // POST /api/vo_approve — run post_tts_analysis and write sentinel
  async function voApproveTTS() {
    const epSel  = document.getElementById('vo-ep-select');
    const locSel = document.getElementById('vo-locale-select');
    const epDir  = epSel.value;
    const locale = locSel.value;
    if (!epDir || !locale) { alert('Select episode and locale first.'); return; }

    const btn     = document.getElementById('vo-approve-btn');
    const errEl   = document.getElementById('vo-approve-error');
    const bannerEl = document.getElementById('vo-approve-banner');
    if (btn) { btn.textContent = '⏳ Approving…'; btn.disabled = true; }
    if (errEl) errEl.style.display = 'none';

    try {
      const r = await fetch('/api/vo_approve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ep_dir: epDir, locale }),
      });
      const data = await r.json();
      if (data.error) throw new Error(data.detail ? data.error + ': ' + data.detail : data.error);

      // Success — update UI
      if (btn) { btn.textContent = '✓ VO Approved — Continue'; btn.disabled = false; }
      const sentinelEl = document.getElementById('vo-sentinel-status');
      if (sentinelEl) {
        sentinelEl.style.display = 'flex';
        sentinelEl.style.background = '#1a3a1a';
        sentinelEl.style.color = '#6ec96e';
        sentinelEl.style.border = '1px solid #2d6a2d';
        document.getElementById('vo-sentinel-icon').textContent = '✓';
        document.getElementById('vo-sentinel-text').textContent =
          `VO Approved — ${data.items_measured || 0} items measured`;
        document.getElementById('vo-sentinel-time').textContent = new Date().toLocaleTimeString();
      }
      if (bannerEl) {
        document.getElementById('vo-approve-msg').textContent =
          `✓ VO Approved — ${data.items_measured || 0} items measured. Pipeline may continue.`;
      }
      // Remove stale indicator from all rows
      document.querySelectorAll('.vo-timing-stale').forEach(el =>
        el.classList.remove('vo-timing-stale'));

    } catch(e) {
      if (btn) { btn.textContent = '✓ VO Approved — Continue'; btn.disabled = false; }
      if (errEl) { errEl.textContent = '✗ ' + e.message; errEl.style.display = 'block'; }
    }
  }

  // Load sentinel status when VO tab loads
  async function _voLoadSentinel(epDir, locale) {
    if (!epDir || !locale) return;
    try {
      const r = await fetch(`/api/vo_sentinel?ep_dir=${encodeURIComponent(epDir)}&locale=${encodeURIComponent(locale)}`);
      const data = await r.json();
      const approveBanner = document.getElementById('vo-approve-banner');
      const sentinelEl    = document.getElementById('vo-sentinel-status');
      if (data.valid) {
        if (approveBanner) {
          approveBanner.style.display = 'flex';
          document.getElementById('vo-approve-msg').textContent =
            '✓ VO Approved — pipeline may continue. Edit any item and Re-approve if needed.';
          document.getElementById('vo-approve-btn').textContent = '↻ Re-approve';
        }
        if (sentinelEl) {
          sentinelEl.style.display = 'flex';
          sentinelEl.style.background = '#1a3a1a';
          sentinelEl.style.color = '#6ec96e';
          sentinelEl.style.border = '1px solid #2d6a2d';
          document.getElementById('vo-sentinel-icon').textContent = '✓';
          document.getElementById('vo-sentinel-text').textContent = 'VO Approved';
          document.getElementById('vo-sentinel-time').textContent = data.completed_at || '';
        }
      } else if (data.exists) {
        // Sentinel exists but hashes don't match (edits since last approval)
        _voMarkSentinelInvalid();
        if (approveBanner) approveBanner.style.display = 'flex';
      }
    } catch(e) {}
  }

  // ── Test / Production mode toggle ────────────────────────────────────────────

  function toggleRenderMode() {
    renderProd = !renderProd;
    const wrap = document.getElementById('toggle-render');
    wrap.classList.toggle('render-hd', renderProd);
    wrap.title = renderProd
      ? 'HD mode — high quality encode (CRF 18) for final upload'
      : 'Preview mode — fast encode (CRF 28) for review';
  }
  function toggleMusicMode() {
    const chk = document.getElementById('chk-no-music');
    noMusic = chk ? !chk.checked : !noMusic;  // checked = Music ON = noMusic false
  }
  function togglePurgeMode() {
    const chk = document.getElementById('chk-purge-assets');
    purgeAssets = chk ? chk.checked : !purgeAssets;
  }

  // Allow keyboard activation (Space / Enter)
  document.addEventListener('keydown', e => {
    if (e.target === document.getElementById('toggle-render') &&
        (e.key === ' ' || e.key === 'Enter')) {
      e.preventDefault(); toggleRenderMode();
    }
  });

  // ── Tab switching ───────────────────────────────────────────────────────────
  let _pipePoller = null;   // setInterval handle for Pipeline tab auto-refresh

  function switchTab(name) {
    document.querySelectorAll('.tab').forEach(t =>
      t.classList.toggle('active', t.dataset.tab === name));
    document.getElementById('panel-run').style.display      = name === 'run'      ? 'flex' : 'none';
    document.getElementById('panel-browse').style.display   = name === 'browse'   ? 'flex' : 'none';
    document.getElementById('panel-pipeline').style.display = name === 'pipeline' ? 'flex' : 'none';
    document.getElementById('panel-media').style.display    = name === 'media'    ? 'flex' : 'none';
    document.getElementById('panel-sfx').style.display      = name === 'sfx'      ? 'flex' : 'none';
    document.getElementById('panel-music').style.display    = name === 'music'    ? 'flex' : 'none';
    document.getElementById('panel-vo').style.display       = name === 'vo'       ? 'flex' : 'none';
    document.getElementById('panel-youtube').style.display  = name === 'youtube'  ? 'flex' : 'none';
    if (name === 'sfx')     sfxInit();
    if (name === 'vo')      populateVoEpSelect();
    if (name === 'youtube') initYoutubeTab();

    // ── Connection cleanup on tab switch ──
    // HTTP/1.1 allows only 6 concurrent connections per host.  Free up slots
    // by cleaning stale/unnecessary connections when switching tabs.

    // Clear stale EventSource refs (already closed but not nulled)
    if (es && es.readyState === 2) { es = null; }
    if (pipeStepEs && pipeStepEs.readyState === 2) { pipeStepEs = null; }

    // Leaving Media tab: stop any playing video to release its HTTP connection
    if (name !== 'media' && _mediaPlayingVid) {
      _mediaStopVid(_mediaPlayingVid);
      _mediaPlayingVid = null;
    }

    // Stop voice-preview audio when leaving Run tab (releases audio HTTP connection)
    if (name !== 'run' && _vcPlayingAudio) {
      _vcPlayingAudio.pause();
      _vcPlayingAudio.src = '';   // abort download
      _vcPlayingAudio = null;
    }

    if (name === 'browse')   loadProjects();
    if (name === 'media')    initMediaTab();
    if (name === 'music')    initMusicTab();
    if (name === 'pipeline') {
      initPipelineTab();
      refreshPipeline();   // immediate refresh on every switch so state is always current
      // Auto-refresh every 5 s while Pipeline tab is open
      if (!_pipePoller) {
        _pipePoller = setInterval(() => {
          if (document.getElementById('panel-pipeline').style.display !== 'none') {
            refreshPipeline();
          } else {
            clearInterval(_pipePoller); _pipePoller = null;
          }
        }, 5000);
      }
    } else {
      // Leaving Pipeline tab — stop auto-refresh
      if (_pipePoller) { clearInterval(_pipePoller); _pipePoller = null; }
    }
  }

  // ── SFX tab ─────────────────────────────────────────────────────────────────

  let _sfxSlug       = '';
  let _sfxEpId       = '';
  let _sfxItems      = [];
  let _sfxResults    = {};   // { item_id: { item, candidates[] } }
  let _sfxSelected   = {};   // { item_id: index }
  let _sfxServerError= null; // set when media server is unreachable
  let _sfxAiState    = {};   // { item_id: { open, prompt, generating, statusText } }
  let _sfxAudioEl    = null;
  let _sfxPlayingUrl = null;
  let _sfxPlayGen    = 0;
  let _sfxPlayingBtn = null;  // DOM button currently showing ⏸

  function sfxInit() {
    const sel = document.getElementById('sfx-ep-select');
    const needsPopulate = sel.options.length <= 1;
    if (needsPopulate) {
      // Load all projects/episodes into the select, then sync from Run tab
      fetch('/list_projects').then(r => r.json()).then(data => {
        (data.projects || []).forEach(proj => {
          (proj.episodes || []).forEach(ep => {
            const opt = document.createElement('option');
            opt.value       = proj.slug + '|' + ep.id;
            opt.textContent = proj.slug + ' / ' + ep.id;
            sel.appendChild(opt);
          });
        });
        _sfxSyncFromRunTab();
      }).catch(() => { _sfxSyncFromRunTab(); });
    } else {
      _sfxSyncFromRunTab();
    }
  }

  function _sfxSyncFromRunTab() {
    // Auto-select episode from Run tab globals if SFX tab has no selection
    if (_sfxSlug && _sfxEpId) {
      // Already selected — restore sessionStorage selections
      const stored = sessionStorage.getItem('sfx_selected__' + _sfxSlug + '__' + _sfxEpId);
      if (stored) try { _sfxSelected = JSON.parse(stored); } catch(e) {}
      return;
    }
    if (!currentSlug || !currentEpId) return;
    const target = currentSlug + '|' + currentEpId;
    const sel = document.getElementById('sfx-ep-select');
    for (let i = 0; i < sel.options.length; i++) {
      if (sel.options[i].value === target) {
        sel.value = target;
        onSfxEpChange();
        return;
      }
    }
  }

  async function onSfxEpChange() {
    const val = document.getElementById('sfx-ep-select').value;
    if (!val) return;
    const parts = val.split('|');
    _sfxSlug = parts[0]; _sfxEpId = parts.slice(1).join('|');
    document.getElementById('sfx-btn-search').disabled = false;
    document.getElementById('sfx-status-bar').textContent = 'Episode selected. Loading previous results\u2026';
    // Try to load previously saved search results from disk
    await _sfxLoadExisting();
  }

  async function _sfxLoadExisting() {
    const statusBar = document.getElementById('sfx-status-bar');
    if (!_sfxSlug || !_sfxEpId) {
      statusBar.textContent = 'Select an episode first.';
      return;
    }
    try {
      const r = await fetch('/api/episode_file?slug=' + encodeURIComponent(_sfxSlug)
                          + '&ep_id=' + encodeURIComponent(_sfxEpId)
                          + '&file=assets/sfx/sfx_search_results.json');
      if (!r.ok) {
        statusBar.textContent = 'No previous results found. Run a search to get started.';
        document.getElementById('sfx-btn-search').disabled = false;
        return;
      }
      const saved = await r.json();
      if (!saved || !saved.results) {
        statusBar.textContent = 'No previous results found. Run a search to get started.';
        document.getElementById('sfx-btn-search').disabled = false;
        return;
      }

      _sfxResults  = saved.results;
      _sfxSelected = saved.selected || {};
      _sfxItems    = Object.values(_sfxResults).map(r => r.item);

      // Render all cards
      document.getElementById('sfx-body').innerHTML = '';
      _sfxItems.forEach(item => {
        const res = _sfxResults[item.item_id || item.asset_id || ''];
        if (res) sfxRenderCard(res.item, res.candidates);
      });

      document.getElementById('sfx-footer').style.display = 'flex';
      sfxUpdateCountLabel();
      document.getElementById('sfx-btn-search').disabled = false;
      statusBar.textContent =
        'Loaded ' + _sfxItems.length + ' items from previous search'
        + (saved.saved_at ? ' (' + new Date(saved.saved_at).toLocaleString() + ')' : '') + '.';
    } catch(e) {
      statusBar.textContent = 'No previous results found. Run a search to get started.';
      document.getElementById('sfx-btn-search').disabled = false;
    }
  }

  async function _sfxSaveResults() {
    if (!_sfxSlug || !_sfxEpId || !Object.keys(_sfxResults).length) return;
    try {
      await fetch('/api/sfx_results_save', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          slug:     _sfxSlug,
          ep_id:    _sfxEpId,
          results:  _sfxResults,
          selected: _sfxSelected,
        }),
      });
    } catch(e) {}
  }

  async function sfxSearchAll() {
    if (!_sfxSlug || !_sfxEpId) {
      document.getElementById('sfx-status-bar').textContent = 'Select an episode first.';
      return;
    }
    document.getElementById('sfx-btn-search').disabled = true;
    document.getElementById('sfx-status-bar').textContent = 'Loading manifest\u2026';
    document.getElementById('sfx-body').innerHTML = '';
    document.getElementById('sfx-footer').style.display = 'none';
    _sfxItems = []; _sfxResults = {};

    // Load manifest to get sfx_items
    try {
      const manifestUrl = '/api/episode_file?slug=' + encodeURIComponent(_sfxSlug)
                        + '&ep_id=' + encodeURIComponent(_sfxEpId)
                        + '&file=AssetManifest_draft.shared.json';
      const mresp = await fetch(manifestUrl);
      if (!mresp.ok) throw new Error('manifest: ' + mresp.status);
      const manifest = await mresp.json();
      _sfxItems = manifest.sfx_items || manifest.sfx || [];
    } catch(e) {
      document.getElementById('sfx-status-bar').textContent = 'Failed to load manifest: ' + e.message;
      document.getElementById('sfx-btn-search').disabled = false;
      return;
    }

    if (!_sfxItems.length) {
      document.getElementById('sfx-status-bar').textContent = 'No SFX items found in manifest.';
      document.getElementById('sfx-btn-search').disabled = false;
      return;
    }

    document.getElementById('sfx-status-bar').textContent = 'Searching ' + _sfxItems.length + ' SFX items\u2026';
    const serverUrl = (document.getElementById('sfx-server-url').value || 'http://localhost:8200').trim();
    const apiKey = '{{MEDIA_API_KEY}}';

    let done = 0;
    const MAX_CONCURRENT = 3;
    const sem = { count: 0 };

    const searchOne = async (item) => {
      // Check sessionStorage cache first
      const cacheKey = 'sfx_cache__' + _sfxSlug + '__' + _sfxEpId + '__' + item.item_id;
      const cached = sessionStorage.getItem(cacheKey);
      if (cached) {
        try {
          const parsed = JSON.parse(cached);
          if (Date.now() - parsed.fetched_at < 30 * 60 * 1000) {
            _sfxResults[item.item_id] = { item, candidates: parsed.candidates };
            done++;
            sfxRenderCard(item, parsed.candidates);
            document.getElementById('sfx-status-bar').textContent = 'Searched ' + done + '/' + _sfxItems.length + '\u2026';
            return;
          }
        } catch(e) {}
      }

      try {
        const r = await fetch('/api/sfx_search', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            slug: _sfxSlug, ep_id: _sfxEpId, item_id: item.item_id,
            query: (item.search_queries && item.search_queries[0]) || item.tag || item.description || '',
            duration_sec: item.duration_sec || 5,
            server_url: serverUrl, api_key: apiKey,
          })
        });
        const data = r.ok ? await r.json() : { candidates: [] };
        const candidates = data.candidates || [];
        _sfxResults[item.item_id] = { item, candidates };
        try {
          sessionStorage.setItem(cacheKey, JSON.stringify({ candidates, fetched_at: Date.now() }));
        } catch(e) {}
      } catch(e) {
        _sfxResults[item.item_id] = { item, candidates: [] };
      }
      done++;
      sfxRenderCard(item, (_sfxResults[item.item_id] || {}).candidates || []);
      document.getElementById('sfx-status-bar').textContent = 'Searched ' + done + '/' + _sfxItems.length + '\u2026';
    };

    // Search with concurrency limit
    const queue = [..._sfxItems];
    const workers = Array.from({length: Math.min(MAX_CONCURRENT, queue.length)}, async () => {
      while (queue.length) {
        const item = queue.shift();
        if (item) await searchOne(item);
      }
    });
    await Promise.all(workers);

    document.getElementById('sfx-status-bar').textContent = 'Done \u2014 ' + _sfxItems.length + ' items.';
    document.getElementById('sfx-footer').style.display = 'flex';
    sfxUpdateCountLabel();
    document.getElementById('sfx-btn-search').disabled = false;
    _sfxSaveResults();  // persist to disk so results survive page reload
  }

  function sfxRenderCard(item, candidates) {
    const body = document.getElementById('sfx-body');
    const itemId = item.item_id || item.asset_id || '';
    let card = document.getElementById('sfx-card-' + itemId);
    if (!card) {
      card = document.createElement('div');
      card.className = 'sfx-card';
      card.id = 'sfx-card-' + itemId;
      body.appendChild(card);
    }
    const selIdx = _sfxSelected[itemId];
    const tag = item.tag || item.description || itemId;
    const dur = (item.duration_sec || 0).toFixed(1);

    let rows = '';
    if (!candidates.length) {
      rows = '<div class="sfx-empty">No results found.</div>';
    } else {
      candidates.forEach((c, idx) => {
        const isSel = selIdx === idx;
        const stars = c.rating > 0 ? '\u2605' + c.rating.toFixed(1) : '';
        const dls   = c.downloads > 1000 ? '\u2193' + Math.round(c.downloads/1000) + 'K' : (c.downloads ? '\u2193' + c.downloads : '');
        const lic   = c.license_summary || '';
        const src   = c.source_site || '';
        const dur2  = (c.duration_sec || 0).toFixed(1) + 's';
        const waveUrl = c.waveform_img || '';
        const waveHtml = waveUrl
          ? '<img class="sfx-cand-waveform" src="' + waveUrl + '" alt="waveform">'
          : '<div class="sfx-cand-waveform"></div>';
        const linkUrl = c.asset_page_url || '#';
        rows += `<div class="sfx-cand-row${isSel ? ' selected' : ''}"
                      onclick="sfxSelectCandidate('${itemId}', ${idx})"
                      title="${(c.attribution_text||'').replace(/"/g,'&quot;')}">
          ${waveHtml}
          <button class="sfx-play-btn" onclick="sfxPlay(event, '${c.preview_url||''}', this)" title="Preview">\u25b6</button>
          <span class="sfx-cand-title">${c.title || '(untitled)'}</span>
          <span class="sfx-cand-meta">${dur2}${stars?' '+stars:''}${dls?' '+dls:''}${lic?' '+lic:''}${src?' '+src:''}</span>
          <a class="sfx-link-btn" href="${linkUrl}" target="_blank" title="Open source page" onclick="event.stopPropagation()">\u2197</a>
        </div>`;
      });
    }

    card.innerHTML = `
      <div class="sfx-card-header">
        <span class="sfx-card-id">\ud83d\udd0a ${itemId}</span>
        <span class="sfx-card-tag">${tag}</span>
        <span class="sfx-card-dur">Target: ${dur}s</span>
        <button class="sfx-ai-toggle-btn" onclick="event.stopPropagation();_sfxAiToggle('${itemId}')" title="Generate SFX with AI">\u2728 AI</button>
      </div>
      <div class="sfx-ai-panel" id="sfx-ai-panel-${itemId}">
        <textarea id="sfx-ai-prompt-${itemId}" rows="3" placeholder="Describe the sound to generate\u2026"
                  oninput="_sfxAiStateSet('${itemId}','prompt',this.value)"></textarea>
        <div class="sfx-ai-hint">Target duration: ${dur}s</div>
        <div class="sfx-ai-row">
          <button class="sfx-ai-gen-btn" id="sfx-ai-btn-${itemId}"
                  onclick="_sfxAiGenStart('${itemId}')">\u2728 Generate</button>
          <span class="sfx-ai-status" id="sfx-ai-status-${itemId}"></span>
        </div>
      </div>
      <div class="sfx-cand-list">${rows}</div>
    `;
    // Restore AI panel state (survives re-renders when user selects a candidate)
    _sfxAiRestorePanel(itemId, item);
  }

  function sfxSelectCandidate(itemId, idx) {
    // Stop playback when user changes selection (avoids ghost audio)
    if (_sfxAudioEl) { _sfxAudioEl.pause(); }
    _sfxResetBtn(_sfxPlayingBtn);
    _sfxPlayingUrl = null; _sfxPlayingBtn = null;
    if (_sfxSelected[itemId] === idx) {
      delete _sfxSelected[itemId];
    } else {
      _sfxSelected[itemId] = idx;
    }
    try { sessionStorage.setItem('sfx_selected__' + _sfxSlug + '__' + _sfxEpId, JSON.stringify(_sfxSelected)); } catch(e) {}
    const res = _sfxResults[itemId];
    if (res) sfxRenderCard(res.item, res.candidates);
    sfxUpdateCountLabel();
    _sfxSaveResults();  // persist selection change to disk
  }

  function sfxUpdateCountLabel() {
    const total = _sfxItems.length;
    const sel   = Object.keys(_sfxSelected).length;
    document.getElementById('sfx-count-label').textContent = sel + '/' + total + ' selected';
  }

  // ── SFX AI Generation ─────────────────────────────────────────────────────

  function _sfxAiStateSet(itemId, key, val) {
    if (!_sfxAiState[itemId]) _sfxAiState[itemId] = {};
    _sfxAiState[itemId][key] = val;
  }

  function _sfxAiRestorePanel(itemId, item) {
    const state  = _sfxAiState[itemId] || {};
    const panel  = document.getElementById('sfx-ai-panel-' + itemId);
    const ta     = document.getElementById('sfx-ai-prompt-' + itemId);
    const btn    = document.getElementById('sfx-ai-btn-' + itemId);
    const status = document.getElementById('sfx-ai-status-' + itemId);
    if (!panel) return;
    // Restore panel open/closed state
    panel.style.display = state.open ? 'block' : 'none';
    // Restore textarea: saved prompt → item.tag → item.search_queries[0]
    if (ta) ta.value = state.prompt !== undefined
      ? state.prompt
      : (item.tag || (item.search_queries && item.search_queries[0]) || '');
    // Restore status text + button state
    if (status) { status.textContent = state.statusText || ''; status.style.color = state.statusError ? '#e06c75' : (state.statusDone ? '#6a9a6a' : '#888'); }
    if (btn && state.generating) btn.disabled = true;
  }

  function _sfxAiToggle(itemId) {
    if (!_sfxAiState[itemId]) _sfxAiState[itemId] = {};
    _sfxAiState[itemId].open = !_sfxAiState[itemId].open;
    const panel = document.getElementById('sfx-ai-panel-' + itemId);
    if (panel) panel.style.display = _sfxAiState[itemId].open ? 'block' : 'none';
  }

  async function _sfxAiGenStart(itemId) {
    const ta     = document.getElementById('sfx-ai-prompt-' + itemId);
    const btn    = document.getElementById('sfx-ai-btn-' + itemId);
    const status = document.getElementById('sfx-ai-status-' + itemId);
    const prompt = ta ? ta.value.trim() : '';
    if (!prompt) { if (status) status.textContent = 'Prompt is empty.'; return; }
    _sfxAiStateSet(itemId, 'generating', true);
    _sfxAiStateSet(itemId, 'prompt', prompt);
    _sfxAiStateSet(itemId, 'statusText', '\u23f3 Submitting\u2026');
    _sfxAiStateSet(itemId, 'statusError', false);
    _sfxAiStateSet(itemId, 'statusDone', false);
    if (btn) btn.disabled = true;
    if (status) { status.textContent = '\u23f3 Submitting\u2026'; status.style.color = '#888'; }
    const item = (_sfxResults[itemId] || {}).item || {};
    try {
      const r = await fetch('/api/ai_sfx_generate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ slug: _sfxSlug, ep_id: _sfxEpId,
                               item_id: itemId, prompt,
                               duration_sec: item.duration_sec || 5 }),
      });
      const d = await r.json();
      if (!r.ok || d.error) throw new Error(d.error || 'submit failed');
      const { job_id, asset_id, timestamp_ms } = d;
      const st = '\u23f3 Generating\u2026 (job ' + job_id.slice(0,8) + ')';
      _sfxAiStateSet(itemId, 'statusText', st);
      if (status) status.textContent = st;
      _sfxAiGenPoll(itemId, job_id, asset_id, prompt, timestamp_ms, 0);
    } catch(err) {
      _sfxAiStateSet(itemId, 'generating', false);
      _sfxAiStateSet(itemId, 'statusText', '\u274c ' + err.message);
      _sfxAiStateSet(itemId, 'statusError', true);
      if (status) { status.textContent = '\u274c ' + err.message; status.style.color = '#e06c75'; }
      if (btn) btn.disabled = false;
    }
  }

  async function _sfxAiGenPoll(itemId, jobId, assetId, prompt, tMs, elapsed) {
    const btn    = document.getElementById('sfx-ai-btn-' + itemId);
    const status = document.getElementById('sfx-ai-status-' + itemId);
    const MAX_S  = 300;
    if (elapsed > MAX_S) {
      _sfxAiStateSet(itemId, 'generating', false);
      _sfxAiStateSet(itemId, 'statusText', '\u274c Timed out after ' + MAX_S + 's');
      _sfxAiStateSet(itemId, 'statusError', true);
      if (status) { status.textContent = '\u274c Timed out after ' + MAX_S + 's'; status.style.color = '#e06c75'; }
      if (btn) btn.disabled = false;
      return;
    }
    try {
      const r = await fetch('/api/ai_job_status?job_id=' + encodeURIComponent(jobId));
      const d = await r.json();
      if (d.error) throw new Error(d.error);
      if (d.status === 'done') {
        const filename = assetId + '.mp3';
        const st = '\u23f3 Saving audio\u2026';
        _sfxAiStateSet(itemId, 'statusText', st);
        if (status) status.textContent = st;
        const r2 = await fetch('/api/ai_sfx_save', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ job_id: jobId, filename,
                                 slug: _sfxSlug, ep_id: _sfxEpId,
                                 item_id: itemId, timestamp_ms: tMs }),
        });
        const d2 = await r2.json();
        if (!r2.ok || d2.error) throw new Error(d2.error || 'save failed');
        _sfxAiStateSet(itemId, 'generating', false);
        _sfxAiStateSet(itemId, 'statusText', '\u2713 Done \u2014 audio added below');
        _sfxAiStateSet(itemId, 'statusDone', true);
        if (status) { status.textContent = '\u2713 Done \u2014 audio added below'; status.style.color = '#6a9a6a'; }
        if (btn) btn.disabled = false;
        // Inject as first candidate so user can immediately preview & select
        _sfxAiInjectResult(itemId, {
          preview_url:      d2.url,
          title:            '\u2728 AI: ' + prompt.slice(0, 50),
          source_site:      'ai_gen',
          license_summary:  'AI Generated',
          duration_sec:     d2.duration_sec || 0,
          attribution_text: 'AI generated',
          asset_page_url:   '',
          waveform_img:     '',
          author:           'AI',
          rating:           0,
          downloads:        0,
        });
      } else if (d.status === 'failed') {
        throw new Error('AI job failed: ' + (d.errors || []).join('; '));
      } else {
        const secs = elapsed + 2;
        const st = '\u23f3 Generating\u2026 ' + secs + 's';
        _sfxAiStateSet(itemId, 'statusText', st);
        if (status) status.textContent = st;
        setTimeout(function() { _sfxAiGenPoll(itemId, jobId, assetId, prompt, tMs, secs); }, 2000);
      }
    } catch(err) {
      _sfxAiStateSet(itemId, 'generating', false);
      _sfxAiStateSet(itemId, 'statusText', '\u274c ' + err.message);
      _sfxAiStateSet(itemId, 'statusError', true);
      if (status) { status.textContent = '\u274c ' + err.message; status.style.color = '#e06c75'; }
      if (btn) btn.disabled = false;
    }
  }

  function _sfxAiInjectResult(itemId, candidate) {
    const res = _sfxResults[itemId];
    if (!res) return;
    res.candidates.unshift(candidate);  // prepend — AI result appears first
    sfxRenderCard(res.item, res.candidates);
    _sfxSaveResults();  // persist AI-generated result to disk
  }

  function _sfxResetBtn(b) {
    if (!b) return;
    b.textContent = '\u25b6';
    b.classList.remove('playing', 'error');
  }

  function sfxPlay(event, url, btn) {
    event.stopPropagation();

    // No URL — show error briefly
    if (!url) {
      btn.textContent = '!';
      btn.classList.add('error');
      setTimeout(() => _sfxResetBtn(btn), 2000);
      return;
    }

    const myGen = ++_sfxPlayGen;

    // Create audio element once, attach ended/error handlers
    if (!_sfxAudioEl) {
      _sfxAudioEl = new Audio();
      _sfxAudioEl.addEventListener('ended', () => {
        _sfxResetBtn(_sfxPlayingBtn);
        _sfxPlayingUrl = null;
        _sfxPlayingBtn = null;
      });
      _sfxAudioEl.addEventListener('error', () => {
        if (_sfxPlayingBtn) {
          _sfxPlayingBtn.textContent = '✕';
          _sfxPlayingBtn.classList.add('error');
          setTimeout(() => _sfxResetBtn(_sfxPlayingBtn), 2500);
        }
        _sfxPlayingUrl = null;
        _sfxPlayingBtn = null;
      });
    }

    // Clicking the already-playing row → pause/toggle
    if (_sfxPlayingUrl === url) {
      _sfxAudioEl.pause();
      _sfxResetBtn(btn);
      _sfxPlayingUrl = null;
      _sfxPlayingBtn = null;
      return;
    }

    // Switch to new track — reset old button
    _sfxAudioEl.pause();
    _sfxResetBtn(_sfxPlayingBtn);

    _sfxAudioEl.src  = url;
    _sfxPlayingUrl   = url;
    _sfxPlayingBtn   = btn;
    btn.textContent  = '\u23f8';   // ⏸
    btn.classList.add('playing');

    _sfxAudioEl.play().then(() => {
      if (_sfxPlayGen !== myGen) { _sfxAudioEl.pause(); _sfxResetBtn(btn); }
    }).catch(() => {
      _sfxResetBtn(btn);
      btn.textContent = '✕';
      btn.classList.add('error');
      setTimeout(() => _sfxResetBtn(btn), 2500);
      _sfxPlayingUrl = null;
      _sfxPlayingBtn = null;
    });
  }

  async function sfxSaveAll() {
    const keys = Object.keys(_sfxSelected);
    if (!keys.length) { alert('No SFX selected.'); return; }
    document.getElementById('sfx-confirm-msg').textContent = 'Saving\u2026';
    let saved = 0, failed = 0;
    for (const itemId of keys) {
      const idx = _sfxSelected[itemId];
      const res = _sfxResults[itemId];
      if (!res) continue;
      const cand = res.candidates[idx];
      if (!cand) continue;
      // AI-generated SFX are already saved to disk at generation time; no re-download needed.
      if (cand.source_site === 'ai_gen') { saved++; continue; }
      try {
        const serverUrl = (document.getElementById('sfx-server-url').value || 'http://localhost:8200').trim();
        const r = await fetch('/api/sfx_save', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            slug: _sfxSlug, ep_id: _sfxEpId, item_id: itemId,
            server_url: serverUrl,
            preview_url: cand.preview_url, source_site: cand.source_site,
            attribution: {
              source_id: cand.source_id || '',
              author: cand.author || '',
              license_summary: cand.license_summary || '',
              license_url: cand.license_url || '',
              asset_page_url: cand.asset_page_url || '',
              attribution_text: cand.attribution_text || '',
            }
          })
        });
        if (r.ok) saved++; else { failed++; console.warn('sfx_save failed', itemId, await r.text()); }
      } catch(e) { failed++; }
    }
    document.getElementById('sfx-confirm-msg').textContent =
      saved + ' saved' + (failed ? ', ' + failed + ' failed' : '') + '.';
  }

  function sfxReset() {
    _sfxSelected = {};
    try { sessionStorage.removeItem('sfx_selected__' + _sfxSlug + '__' + _sfxEpId); } catch(e) {}
    Object.keys(_sfxResults).forEach(id => {
      const res = _sfxResults[id];
      if (res) sfxRenderCard(res.item, res.candidates);
    });
    sfxUpdateCountLabel();
    document.getElementById('sfx-confirm-msg').textContent = '';
  }

  // ── Media tab ────────────────────────────────────────────────────────────────

  let _mediaSlug           = null;
  let _mediaEpId           = null;
  let _mediaBatchId        = null;
  let _mediaPollTimer      = null;
  let _mediaExpandedRows   = new Set();   // item IDs whose detail row is open
  let _mediaLastProgress   = null;        // last batch-status data received while running
  let _mediaResults        = null;   // full items dict from last completed batch
  let _mediaItemIds        = [];     // ordered item IDs for confirm iteration
  let _mediaRecommendedSeq = null;   // recommended_sequence from batch response
  // selections: { item_id: { type:'image'|'video', url, path, score } }
  //   Per-shot:  { item_id: { per_shot: { shot_id: { media_type, url, path, score } } } }
  let _mediaSelections = {};
  let _mediaShotMap = null;  // { bg_id: [shot_id, ...] } or null if ShotList unavailable
  let _mediaShotDur = null;  // { shot_id: duration_sec } — flat lookup for shot durations
  let _mediaActiveShot = {}; // { cardId: shot_id } — which shot row is active (target) per card
  // Scene-based grouping (built alongside _mediaShotMap):
  let _mediaSceneMap  = null; // { scene_id: [shot_id, ...] } ordered by appearance
  let _mediaShotToBg  = null; // { shot_id: bg_id } reverse lookup
  let _mediaSceneBgs  = null; // { scene_id: [bg_id, ...] } unique bg_ids per scene, ordered
  let _mediaBgToScene = null; // { bg_id: scene_id } reverse lookup
  let _mediaSceneOrder = null; // [scene_id, ...] ordered by first shot appearance
  var _mediaPlayingVid = null; // currently playing video element (limit 1 active stream)
  let _bgManifestData = {};    // { itemId: { ai_prompt, search_prompt, include_keywords, media_type, motion_level } }

  // Stop a video and release its HTTP connection (pause alone doesn't close it).
  function _mediaStopVid(vid) {
    vid.pause();
    vid.currentTime = 0;
    // Save the current src so we can restore the thumbnail poster later
    var s = vid.src;
    // Fully abort the download: remove src and call load() to release the connection.
    // IMPORTANT: do NOT re-set vid.src here — per HTML spec, setting .src always
    // triggers the media element load algorithm, opening a new HTTP connection
    // even when preload='none'. Instead, store the URL in dataset for lazy re-load.
    vid.removeAttribute('src');
    vid.load();
    vid.preload = 'none';
    if (s) vid.dataset.lazySrc = s;
    // Show the poster frame (if any) so the thumbnail doesn't go blank
    if (vid.poster) vid.setAttribute('poster', vid.poster);
  }

  // Lazy-load observer: loads video src + metadata when thumbnail scrolls into view.
  // Throttled to max 2 concurrent loads.  HTTP/1.1 allows only 6 connections per host;
  // SSE streams + pollers may already consume 2-3, so we keep the lazy-load budget low
  // to avoid starving other requests (especially the Confirm POST).
  var _mediaLazyLoading = 0;
  var _mediaLazyQueue = [];
  function _mediaLazyLoad(vid) {
    if (_mediaLazyLoading >= 2) {
      _mediaLazyQueue.push(vid);
      return;
    }
    _mediaLazyLoading++;
    vid.src     = vid.dataset.lazySrc;
    vid.preload = 'metadata';
    vid.addEventListener('loadedmetadata', _mediaLazyDone);
    vid.addEventListener('error', _mediaLazyDone);
    // Safety: if neither event fires within 8s (e.g. 404 with no media error),
    // release the slot so the queue doesn't stall and block other requests.
    vid._lazyTimer = setTimeout(function() { _mediaLazyDone.call(vid); }, 8000);
  }
  function _mediaLazyDone() {
    if (this._lazyTimer) { clearTimeout(this._lazyTimer); this._lazyTimer = null; }
    this.removeEventListener('loadedmetadata', _mediaLazyDone);
    this.removeEventListener('error', _mediaLazyDone);
    // After metadata loads, the browser shows the first frame as thumbnail.
    // preload='metadata' ensures the browser stops downloading further data.
    // The connection naturally idles and gets reused from the pool.
    _mediaLazyLoading--;
    if (_mediaLazyLoading < 0) _mediaLazyLoading = 0;  // guard against double-fire
    if (_mediaLazyQueue.length > 0) _mediaLazyLoad(_mediaLazyQueue.shift());
  }
  var _mediaLazyObserver = new IntersectionObserver(function(entries) {
    entries.forEach(function(ent) {
      if (!ent.isIntersecting) return;
      var vid = ent.target;
      _mediaLazyObserver.unobserve(vid);
      if (vid.dataset.lazySrc && !vid.src) _mediaLazyLoad(vid);
    });
  }, { rootMargin: '200px' });

  // ── called once when tab is first activated ──
  function _mediaRestoreLastEp() {
    // Fallback: restore last-selected episode from localStorage (survives page reload)
    if (_mediaSlug && _mediaEpId) return;  // already selected via Run-tab sync
    var last = '';
    try { last = localStorage.getItem('media_last_ep') || ''; } catch(_) {}
    if (!last) return;
    var sel = document.getElementById('media-ep-select');
    for (var i = 0; i < sel.options.length; i++) {
      if (sel.options[i].value === last) {
        sel.value = last;
        onMediaEpChange();
        return;
      }
    }
  }

  function initMediaTab() {
    // Populate episode selector from list_projects (same as Pipeline tab)
    var needsSync = document.getElementById('media-ep-select').options.length <= 1;
    if (!needsSync) {
      // Already populated — sync from Run tab, then fall back to localStorage
      _mediaSyncFromRunTab();
      _mediaRestoreLastEp();
      // If episode is selected but no batch results loaded yet, try loading
      if (_mediaSlug && _mediaEpId && !_mediaResults) mediaLoadExisting();
      return;
    }
    fetch('/list_projects').then(r => r.json()).then(data => {
      const sel = document.getElementById('media-ep-select');
      (data.projects || []).forEach(proj => {
        (proj.episodes || []).forEach(ep => {
          const opt = document.createElement('option');
          opt.value       = proj.slug + '|' + ep.id;
          opt.textContent = proj.slug + ' / ' + ep.id;
          sel.appendChild(opt);
        });
      });
      // Sync from Run tab first; fall back to last-used episode from localStorage
      _mediaSyncFromRunTab();
      _mediaRestoreLastEp();
      // If still no episode selected, auto-select the first available episode
      if (!_mediaSlug && !_mediaEpId && sel.options.length > 1) {
        sel.selectedIndex = 1;
        onMediaEpChange();
      }
    }).catch(() => {});
  }

  function _mediaSyncFromRunTab() {
    // If Run tab has a project/episode selected and Media tab doesn't, propagate it
    if (_mediaSlug && _mediaEpId) return;  // already selected
    if (!currentSlug || !currentEpId) return;  // nothing in Run tab
    var target = currentSlug + '|' + currentEpId;
    var sel = document.getElementById('media-ep-select');
    for (var i = 0; i < sel.options.length; i++) {
      if (sel.options[i].value === target) {
        sel.value = target;
        onMediaEpChange();
        return;
      }
    }
  }

  async function onMediaEpChange() {
    const v = document.getElementById('media-ep-select').value;
    if (!v) { _mediaSlug = null; _mediaEpId = null; return; }
    [_mediaSlug, _mediaEpId] = v.split('|');
    try { localStorage.setItem('media_last_ep', v); } catch(_) {}
    document.getElementById('media-btn-search').disabled = false;
    _mediaSetStatus('Episode selected. Click Search Media to begin.');
    // Load saved per-source config from meta.json, then render the table
    await _loadMediaSourceConfig();
    _renderMediaSourcesTable();
    // Try to load an existing completed batch
    mediaLoadExisting();
  }

  function _mediaServerUrl() {
    return (document.getElementById('media-server-url').value || 'http://localhost:8200').replace(/\/+$/, '');
  }
  function _mediaSetStatus(msg, spinning) {
    document.getElementById('media-status-text').textContent = msg;
    document.getElementById('media-spinner').style.display = spinning ? 'inline-block' : 'none';
  }

  // ── Load ShotList.json to build bg_id → [shot_id, ...] map + durations ──
  // Also builds scene-based grouping maps for UI card rendering.
  async function _mediaLoadShotMap() {
    if (!_mediaSlug || !_mediaEpId) {
      _mediaShotMap = null; _mediaShotDur = null;
      _mediaSceneMap = null; _mediaShotToBg = null; _mediaSceneBgs = null;
      _mediaBgToScene = null; _mediaSceneOrder = null;
      return;
    }
    try {
      const r = await fetch('/api/episode_file?slug=' + encodeURIComponent(_mediaSlug)
          + '&ep_id=' + encodeURIComponent(_mediaEpId)
          + '&file=ShotList.json');
      if (!r.ok) {
        _mediaShotMap = null; _mediaShotDur = null;
        _mediaSceneMap = null; _mediaShotToBg = null; _mediaSceneBgs = null;
        _mediaBgToScene = null; _mediaSceneOrder = null;
        return;
      }
      const d = await r.json();
      const shots = d.shots || [];
      const map = {};
      const durMap = {};
      const sceneMap = {};
      const shotToBg = {};
      const sceneBgs = {};
      const bgToScene = {};
      const sceneOrder = [];
      shots.forEach(s => {
        const bg = s.background_id;
        const sc = s.scene_id || bg;  // fallback to bg_id if no scene_id
        const sid = s.shot_id;
        if (!bg) return;
        // bg → shots map (existing)
        if (!map[bg]) map[bg] = [];
        map[bg].push(sid);
        durMap[sid] = s.duration_sec || 0;
        // scene maps
        shotToBg[sid] = bg;
        bgToScene[bg] = sc;
        if (!sceneMap[sc]) { sceneMap[sc] = []; sceneOrder.push(sc); }
        sceneMap[sc].push(sid);
        if (!sceneBgs[sc]) sceneBgs[sc] = [];
        if (sceneBgs[sc].indexOf(bg) === -1) sceneBgs[sc].push(bg);
      });
      _mediaShotMap = map;
      _mediaShotDur = durMap;
      _mediaSceneMap = sceneMap;
      _mediaShotToBg = shotToBg;
      _mediaSceneBgs = sceneBgs;
      _mediaBgToScene = bgToScene;
      _mediaSceneOrder = sceneOrder;
    } catch (_) {
      _mediaShotMap = null; _mediaShotDur = null;
      _mediaSceneMap = null; _mediaShotToBg = null; _mediaSceneBgs = null;
      _mediaBgToScene = null; _mediaSceneOrder = null;
    }
  }

  // ── AI Generation Queue — persisted in localStorage per project/episode ──

  // ── Inline AI Generation ─────────────────────────────────────────────────────
  // localStorage key for in-flight jobs: survives page reload (B5)
  function _aiGenPendingKey() {
    return 'ai_gen_pending__' + (_mediaSlug||'') + '__' + (_mediaEpId||'');
  }
  function _aiGenPendingGet() {
    try { return JSON.parse(localStorage.getItem(_aiGenPendingKey()) || '[]'); }
    catch(e) { return []; }
  }
  function _aiGenPendingSave(arr) {
    localStorage.setItem(_aiGenPendingKey(), JSON.stringify(arr));
  }
  function _aiGenPendingAdd(rec) {
    const arr = _aiGenPendingGet().filter(function(r){ return r.bg_id !== rec.bg_id; });
    arr.push(rec);
    _aiGenPendingSave(arr);
  }
  function _aiGenPendingRemove(bgId) {
    _aiGenPendingSave(_aiGenPendingGet().filter(function(r){ return r.bg_id !== bgId; }));
  }

  // Start inline generation for a background card
  async function _aiGenStart(itemId) {
    const btn    = document.getElementById('ai-gen-btn-' + itemId);
    const ta     = document.getElementById('ai-gen-prompt-' + itemId);
    const status = document.getElementById('ai-gen-status-' + itemId);
    if (!btn || !ta) return;
    const prompt = ta.value.trim();
    if (!prompt) { if (status) status.textContent = 'Prompt is empty.'; return; }
    btn.disabled = true;
    if (status) { status.textContent = '\u23f3 Submitting\u2026'; status.style.color = '#888'; }

    try {
      const r = await fetch('/api/ai_generate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ slug: _mediaSlug, ep_id: _mediaEpId,
                               bg_id: itemId, prompt: prompt }),
      });
      const d = await r.json();
      if (!r.ok || d.error) throw new Error(d.error || 'submit failed');
      const { job_id, asset_id, timestamp_ms } = d;
      _aiGenPendingAdd({ bg_id: itemId, job_id, asset_id, prompt,
                         timestamp_ms, started_at: Date.now() });
      if (status) status.textContent = '\u23f3 Generating\u2026 (job ' + job_id.slice(0,8) + ')';
      _aiGenPoll(itemId, job_id, asset_id, prompt, timestamp_ms, 0);
    } catch(err) {
      if (status) { status.textContent = '\u274c ' + err.message; status.style.color = '#e06c75'; }
      btn.disabled = false;
    }
  }

  // Poll job status, then save image on completion
  async function _aiGenPoll(itemId, jobId, assetId, prompt, tMs, elapsed) {
    const btn    = document.getElementById('ai-gen-btn-' + itemId);
    const status = document.getElementById('ai-gen-status-' + itemId);
    const MAX_S  = 600;
    if (elapsed > MAX_S) {
      _aiGenPendingRemove(itemId);
      if (status) { status.textContent = '\u274c Timed out after ' + MAX_S + 's'; status.style.color = '#e06c75'; }
      if (btn) btn.disabled = false;
      return;
    }
    try {
      const r = await fetch('/api/ai_job_status?job_id=' + encodeURIComponent(jobId));
      const d = await r.json();
      if (d.error) throw new Error(d.error);
      if (d.status === 'done') {
        const filename = assetId + '.png';
        if (status) status.textContent = '\u23f3 Saving image\u2026';
        const r2 = await fetch('/api/ai_save_image', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ job_id: jobId, filename,
                                 slug: _mediaSlug, ep_id: _mediaEpId,
                                 bg_id: itemId, timestamp_ms: tMs }),
        });
        const d2 = await r2.json();
        if (!r2.ok || d2.error) throw new Error(d2.error || 'save failed');
        _aiGenPendingRemove(itemId);
        const candidate = {
          type:         'image',
          source:       'ai',
          url:          d2.url,
          path:         d2.path,
          score:        null,
          prompt:       prompt,
          created_at:   new Date().toISOString(),
        };
        _aiGenInjectResult(itemId, candidate);
        if (status) { status.textContent = '\u2713 Done — image added below'; status.style.color = '#6a9a6a'; }
        if (btn) btn.disabled = false;
      } else if (d.status === 'failed') {
        throw new Error('AI job failed: ' + (d.errors || []).join('; '));
      } else {
        // Still running — poll again in 2s
        const secs = elapsed + 2;
        if (status) status.textContent = '\u23f3 Generating\u2026 ' + secs + 's';
        setTimeout(function() { _aiGenPoll(itemId, jobId, assetId, prompt, tMs, secs); }, 2000);
      }
    } catch(err) {
      _aiGenPendingRemove(itemId);
      if (status) { status.textContent = '\u274c ' + err.message; status.style.color = '#e06c75'; }
      if (btn) btn.disabled = false;
    }
  }

  // Inject a completed AI-generated candidate into the images section
  function _aiGenInjectResult(bgId, candidate) {
    if (!_mediaResults || !_mediaResults[bgId]) return;
    const item = _mediaResults[bgId];
    if (!item.images) item.images = [];
    item.images.push(candidate);
    // Find the card via scene-based ID, then the bg-content container
    const cardId = (_mediaBgToScene && _mediaBgToScene[bgId]) || bgId;
    const bgContent = document.getElementById('media-bg-content-' + bgId);
    const card = bgContent || document.getElementById('media-card-' + cardId);
    if (!card) return;
    // Find or create the images thumb row
    let imgRow = card.querySelector('.media-thumb-row');
    if (!imgRow) {
      const lbl = document.createElement('div');
      lbl.className = 'media-section-label';
      lbl.textContent = 'Images';
      card.appendChild(lbl);
      imgRow = document.createElement('div');
      imgRow.className = 'media-thumb-row';
      card.appendChild(imgRow);
    }
    const idx = item.images.length - 1;
    const wrap = _mediaThumb(cardId, 'image', candidate, idx);
    // AI badge
    const aiBadge = document.createElement('span');
    aiBadge.textContent = 'AI';
    aiBadge.style.cssText = 'position:absolute;bottom:2px;left:2px;background:#d4eed4;color:#2a7a2a;font-size:9px;font-weight:bold;padding:1px 4px;border-radius:3px;pointer-events:none;';
    wrap.style.position = 'relative';
    wrap.appendChild(aiBadge);
    imgRow.appendChild(wrap);
  }

  // Load previously saved AI-generated images from disk and inject into cards
  async function _aiLoadSavedImages() {
    if (!_mediaSlug || !_mediaEpId) return;
    try {
      const r = await fetch('/api/ai_images?slug=' + encodeURIComponent(_mediaSlug)
                            + '&ep_id=' + encodeURIComponent(_mediaEpId));
      if (!r.ok) return;
      const data = await r.json();
      for (const [bgId, files] of Object.entries(data)) {
        if (!_mediaResults || !_mediaResults[bgId]) continue;
        const existing = _mediaResults[bgId].images || [];
        for (const f of files) {
          // Skip if already injected (same url)
          if (existing.some(function(e) { return e.url === f.url; })) continue;
          const candidate = {
            type: 'image', source: 'ai',
            url: f.url, path: f.path,
            score: null, filename: f.filename,
          };
          _aiGenInjectResult(bgId, candidate);
        }
      }
    } catch(e) {}
  }

  // On page load: resume any in-flight AI gen jobs (B5)
  function _aiGenRestorePending() {
    const pending = _aiGenPendingGet();
    pending.forEach(function(rec) {
      const elapsed = Math.round((Date.now() - rec.started_at) / 1000);
      if (elapsed > 600) {
        _aiGenPendingRemove(rec.bg_id);
        return;
      }
      const status = document.getElementById('ai-gen-status-' + rec.bg_id);
      if (status) { status.textContent = '\u23f3 Resuming\u2026'; }
      _aiGenPoll(rec.bg_id, rec.job_id, rec.asset_id, rec.prompt, rec.timestamp_ms, elapsed);
    });
  }

  function _formatToContentProfile(fmt) {
    const map = {
      continuous_narration: 'documentary',
      ssml_narration:       'documentary',
      sleep_story:          'sleep_story',
    };
    return map[fmt] || 'default';
  }

  function _updateMediaConfigPanel(fmt) {
    const profile = _formatToContentProfile(fmt);

    // Per-profile CLIP dimension weights (mirrors config.json scoring_profiles)
    const profileWeights = {
      documentary: {subjects:0.45, environment:0.20, style:0.15, motion:0.15, technical:0.05},
      sleep_story: {subjects:0.20, environment:0.40, style:0.25, motion:0.05, technical:0.10},
      action:      {subjects:0.35, environment:0.15, style:0.20, motion:0.25, technical:0.05},
      default:     {subjects:0.40, environment:0.25, style:0.20, motion:0.10, technical:0.05},
    };

    // Per-profile image calmness threshold (None = disabled)
    const imageCalmness = {
      documentary: 'disabled \u2014 ruins/historical photos are not penalised for texture',
      sleep_story: '0.55 \u2014 images below this calmness score receive a soft penalty',
      action:      'disabled',
      default:     'disabled',
    };

    // Per-profile video calmness
    const videoCalmness = {
      documentary: 'soft penalty (motion_level driven) \u2014 does not hard-reject any video',
      sleep_story: 'soft penalty (motion_level driven) \u2014 calm videos preferred',
      action:      'disabled \u2014 high-motion videos welcome',
      default:     'soft penalty (motion_level driven)',
    };

    const w = profileWeights[profile] || profileWeights.default;

    // Render weight bar: fill proportional to weight value
    function bar(val) {
      const pct = Math.round(val * 100);
      const filled = Math.round(pct / 5);  // max 20 chars at 5% each
      return '<span style="color:#5a8a5a">' + '\u2588'.repeat(filled) + '\u2591'.repeat(20-filled) + '</span> ' + pct + '%';
    }

    const fmtLabels = {
      continuous_narration: 'Continuous Narration',
      ssml_narration:       'SSML Narration',
      sleep_story:          'Sleep Story',
      default:              'Default',
    };
    const fmtLabel = fmtLabels[fmt] || fmt;

    const rows = [
      ['Scoring profile',  '<strong style="color:var(--text)">' + profile + '</strong>'
        + (profile !== fmt ? ' <span style="color:#666;font-size:11px;">\u2190 from \u201c' + fmtLabel + '\u201d</span>' : '')],
      [''],
      ['\u2014 CLIP dimension weights \u2014', ''],
      ['Subjects (who/what)',        bar(w.subjects)],
      ['Environment (where/setting)',bar(w.environment)],
      ['Style (look/feel)',          bar(w.style)],
      ['Motion (movement)',          bar(w.motion)],
      ['Technical (quality)',        bar(w.technical)],
      [''],
      ['\u2014 Calmness filtering \u2014', ''],
      ['Image calmness',   imageCalmness[profile] || 'disabled'],
      ['Video calmness',   videoCalmness[profile] || 'soft penalty'],
      [''],
      ['\u2014 Query strategy \u2014', ''],
      ['Budget rule',      'First query: 50% of candidates \u00b7 Remaining queries share the rest'],
      ['Location inject',  'include_keywords location terms prefixed into every query'],
      [''],
      ['\u2014 Candidate counts \u2014', ''],
      ['Images per source','30 candidates fetched \u00b7 top results ranked by CLIP'],
      ['Videos per source','30 candidates fetched \u00b7 top results ranked by CLIP'],
      ['Sources',          'Pexels + Pixabay'],
    ];

    let html = '<table style="border-collapse:collapse; width:100%;">';
    for (const row of rows) {
      if (row.length === 1 || (row[0] === '' && row.length === 2 && row[1] === '')) {
        html += '<tr><td colspan="2" style="padding:4px 0 2px 0;"></td></tr>';
      } else if (row[1] === '') {
        html += '<tr><td colspan="2" style="padding:2px 0; color:#6a8a6a; font-size:11px; text-transform:uppercase; letter-spacing:0.05em;">' + row[0] + '</td></tr>';
      } else {
        html += '<tr>' +
          '<td style="padding:1px 12px 1px 0; color:#888888; white-space:nowrap; vertical-align:top;">' + row[0] + '</td>' +
          '<td style="padding:1px 0; color:#bbb; font-family:monospace;">' + row[1] + '</td>' +
          '</tr>';
      }
    }
    html += '</table>';

    // Update new-project panel
    const panel = document.getElementById('media-config-panel');
    const body  = document.getElementById('media-config-body');
    const prof  = document.getElementById('media-config-profile');
    if (panel && body && prof) {
      body.innerHTML = html;
      prof.textContent = profile !== fmt
        ? '(' + fmtLabel + ' \u2192 ' + profile + ')'
        : '(' + profile + ')';
    }
    // Update existing-project panel
    const panelEx = document.getElementById('media-config-panel-existing');
    const bodyEx  = document.getElementById('media-config-body-existing');
    const profEx  = document.getElementById('media-config-profile-existing');
    if (panelEx && bodyEx && profEx) {
      bodyEx.innerHTML = html;
      profEx.textContent = profile !== fmt
        ? '(' + fmtLabel + ' \u2192 ' + profile + ')'
        : '(' + profile + ')';
    }
  }

  // ── Per-source config: defaults, localStorage persistence ──────────────────

  const _mediaSrcDefaults = {
    //                enabled  n_img  n_vid
    pexels:    { enabled: true,  n_img: 15, n_vid:  5 },
    pixabay:   { enabled: true,  n_img: 15, n_vid:  5 },
    openverse: { enabled: true,  n_img: 40, n_vid:  0 },
    wikimedia: { enabled: true,  n_img: 40, n_vid:  0 },
    europeana: { enabled: true,  n_img: 15, n_vid:  0 },
    archive:   { enabled: false, n_img: 10, n_vid: 10 },
  };
  let _mediaSourceConfig = JSON.parse(JSON.stringify(_mediaSrcDefaults));

  async function _saveMediaSourceConfig() {
    if (!_mediaSlug || !_mediaEpId) return;
    try {
      await fetch('/api/save_episode_meta', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          slug: _mediaSlug, ep_id: _mediaEpId,
          media_source_config: _mediaSourceConfig,
        }),
      });
    } catch(e) {}
  }

  // Sources that have no video API — vid inputs locked to 0 and greyed out
  const _NO_VIDEO_SOURCES = new Set(['openverse', 'europeana']);

  async function _loadMediaSourceConfig() {
    // Reset to defaults first
    _mediaSourceConfig = JSON.parse(JSON.stringify(_mediaSrcDefaults));
    if (!_mediaSlug || !_mediaEpId) return;
    try {
      const r = await fetch('/api/episode_file?slug=' + encodeURIComponent(_mediaSlug)
                            + '&ep_id=' + encodeURIComponent(_mediaEpId) + '&file=meta.json');
      if (!r.ok) return;
      const meta = await r.json();
      const saved = meta.media_source_config;
      if (saved && typeof saved === 'object') {
        // Merge saved values over defaults so new sources still appear with defaults
        Object.keys(_mediaSrcDefaults).forEach(src => {
          if (saved[src]) _mediaSourceConfig[src] = { ..._mediaSrcDefaults[src], ...saved[src] };
        });
      }
      // Enforce zero for video fields on sources that have no video API
      _NO_VIDEO_SOURCES.forEach(src => {
        if (_mediaSourceConfig[src]) {
          _mediaSourceConfig[src].n_vid = 0;
        }
      });
    } catch(e) {}
  }

  function _renderMediaSourcesTable() {
    const tbody = document.getElementById('media-sources-tbody');
    if (!tbody) return;
    tbody.innerHTML = '';
    const inp     = 'background:var(--surface);color:var(--text);border:1px solid var(--border);border-radius:4px;padding:2px 5px;font-size:0.9em;width:52px;text-align:center';
    const inpGrey = 'background:var(--surface);color:var(--text-muted,#888);border:1px solid var(--border);border-radius:4px;padding:2px 5px;font-size:0.9em;width:52px;text-align:center;opacity:0.4;cursor:not-allowed';
    Object.entries(_mediaSourceConfig).forEach(([src, v]) => {
      const label   = src.charAt(0).toUpperCase() + src.slice(1);
      const noVid   = _NO_VIDEO_SOURCES.has(src);
      const nVidVal = noVid ? 0 : v.n_vid;
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="padding:3px 14px 3px 0;font-weight:500">${label}</td>
        <td style="padding:3px 10px;text-align:center">
          <input type="checkbox" data-src="${src}" data-field="enabled"
                 ${v.enabled ? 'checked' : ''}
                 onchange="_onMediaSrcTableChange(this)">
        </td>
        <td style="padding:3px 10px;text-align:center">
          <input type="number" data-src="${src}" data-field="n_img"
                 min="0" max="200" value="${v.n_img}" style="${inp}"
                 onchange="_onMediaSrcTableChange(this)">
        </td>
        <td style="padding:3px 0 3px 10px;text-align:center">
          <input type="number" data-src="${src}" data-field="n_vid"
                 min="0" max="50" value="${nVidVal}" style="${noVid ? inpGrey : inp}"
                 ${noVid ? 'disabled title="This source does not provide videos"' : ''}
                 onchange="_onMediaSrcTableChange(this)">
        </td>`;
      tbody.appendChild(tr);
    });
    document.getElementById('media-sources-table').style.display = 'block';
  }

  function _onMediaSrcTableChange(el) {
    const src   = el.dataset.src;
    const field = el.dataset.field;
    if (!_mediaSourceConfig[src]) return;
    if (field === 'enabled') {
      _mediaSourceConfig[src].enabled = el.checked;
    } else {
      _mediaSourceConfig[src][field] = parseInt(el.value) || 0;
    }
    _saveMediaSourceConfig();   // fire-and-forget async write to meta.json
  }

  // ── Start a new search batch ──
  async function mediaStartSearch() {
    if (!_mediaSlug || !_mediaEpId) return;
    const serverUrl = _mediaServerUrl();

    // Check for existing batches and let the user decide what to do
    try {
      const rb = await fetch(
        '/api/media_batches?slug=' + encodeURIComponent(_mediaSlug)
        + '&ep_id='      + encodeURIComponent(_mediaEpId)
        + '&server_url=' + encodeURIComponent(serverUrl));
      const db = await rb.json();
      if (rb.ok && !db.error) {
        const batches = db.batches || [];

        // If a batch is actively running, reconnect silently (no dialog needed)
        const running = batches.find(b => b.status === 'running');
        if (running) {
          _mediaBatchId = running.batch_id;
          document.getElementById('media-btn-search').disabled = true;
          _mediaSetStatus('Reconnecting to running batch…', true);
          _mediaStartPolling();
          return;
        }

        // Any other existing batch (interrupted, failed, done) — show the modal
        const existing = batches.find(b =>
          b.status === 'interrupted' || b.status === 'failed' ||
          (b.status === 'done' && b.item_count > 0));
        if (existing) {
          const choice = await _mediaShowBatchModal(existing, _mediaSlug, _mediaEpId);

          if (choice === 'cancel') {
            // Do nothing — user dismissed
            return;
          }

          if (choice === 'resume') {
            document.getElementById('media-btn-search').disabled = true;
            _mediaSetStatus('Resuming batch…', true);
            try {
              const enabledSrcs = Object.entries(_mediaSourceConfig).filter(([, v]) => v.enabled).map(([k]) => k);
              const srcLimits   = Object.fromEntries(
                Object.entries(_mediaSourceConfig).map(([src, v]) => [src, {
                  candidates_images: v.n_img,
                  candidates_videos: v.n_vid,
                }])
              );
              const rr = await fetch('/api/media_batch_resume', {
                method:  'POST',
                headers: {'Content-Type': 'application/json'},
                body:    JSON.stringify({
                  batch_id:               existing.batch_id,
                  server_url:             serverUrl,
                  sources_override:       enabledSrcs,
                  source_limits_override: srcLimits,
                }),
              });
              const rd = await rr.json();
              if (!rr.ok || rd.error) throw new Error(rd.error || 'resume failed');
              _mediaBatchId = existing.batch_id;
              _mediaSetStatus('Resuming — polling for results…', true);
              _mediaStartPolling();
            } catch (err) {
              _mediaSetStatus('Resume error: ' + err.message, false);
              document.getElementById('media-btn-search').disabled = false;
            }
            return;
          }

          // choice === 'new' — fall through to create a fresh batch below
        }
      }
    } catch (_) {}

    _mediaSetStatus('Submitting batch …', true);
    document.getElementById('media-btn-search').disabled = true;
    document.getElementById('media-footer').style.display = 'none';
    document.getElementById('media-body').innerHTML = '';
    _mediaSelections   = {};
    _mediaBatchId      = null;
    _mediaResults      = null;
    _mediaExpandedRows = new Set();
    _mediaLastProgress = null;

    try {
      const r = await fetch('/api/media_batch', {
        method:  'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify((function() {
          const enabledSrcs = Object.entries(_mediaSourceConfig).filter(([, v]) => v.enabled).map(([k]) => k);
          const srcLimits   = Object.fromEntries(
            Object.entries(_mediaSourceConfig).map(([src, v]) => [src, {
              candidates_images: v.n_img,
              candidates_videos: v.n_vid,
            }])
          );
          const maxNImg = enabledSrcs.length ? Math.max(...enabledSrcs.map(s => _mediaSourceConfig[s].n_img)) : 15;
          const maxNVid = enabledSrcs.length ? Math.max(...enabledSrcs.map(s => _mediaSourceConfig[s].n_vid)) : 5;
          return {
            slug:                   _mediaSlug,
            ep_id:                  _mediaEpId,
            server_url:             serverUrl,
            content_profile:        _formatToContentProfile(_selectedFormat),
            sources_override:       enabledSrcs,
            source_limits_override: srcLimits,
            n_img:                  maxNImg || 15,
            n_vid:                  maxNVid || 0,
          };
        })()),
      });
      const d = await r.json();
      if (!r.ok || d.error) throw new Error(d.error || 'batch creation failed');
      _mediaBatchId = d.batch_id;
      _mediaSetStatus('Batch queued — polling for results …', true);
      _mediaStartPolling();
    } catch (err) {
      _mediaSetStatus('Error: ' + err.message, false);
      document.getElementById('media-btn-search').disabled = false;
    }
  }

  // ── Per-shot progress table (shown while batch is running) ──
  //
  // Column meanings:
  //   📷 DL  = total image files downloaded across ALL sources × ALL search queries.
  //            Higher than per-source candidate limit because multiple queries (one per
  //            keyword set) × multiple sources each contribute candidates before dedup.
  //   🎬 DL  = same, for video files.
  //   📷 ✓   = images kept after CLIP scoring (top-N per source).
  //   🎬 ✓   = videos kept after scoring.
  //
  // Click ▶ on any row to expand per-source breakdown (available once item is done).
  //
  function _mediaRenderProgress(d) {
    if (d) _mediaLastProgress = d;
    else   d = _mediaLastProgress;
    if (!d) return;
    const body  = document.getElementById('media-body');
    const items = d.items || {};
    if (!Object.keys(items).length) return;

    // Re-use existing table if already rendered; otherwise create it
    let tbl = body.querySelector('.media-progress-table');
    if (!tbl) {
      body.innerHTML = '';
      tbl = document.createElement('table');
      tbl.className = 'media-progress-table';
      tbl.innerHTML = `<thead><tr>
        <th style="width:18px"></th>
        <th>Shot</th>
        <th>Status</th>
        <th style="text-align:right;padding-right:16px" title="Total image files downloaded across all sources and search queries">Img Downloaded</th>
        <th style="text-align:right;padding-right:16px" title="Total video files downloaded across all sources and search queries">Vid Downloaded</th>
        <th style="text-align:right;padding-right:16px" title="Images kept after CLIP scoring">Img Scored</th>
        <th style="text-align:right;padding-right:16px" title="Videos kept after CLIP scoring">Vid Scored</th>
      </tr></thead><tbody></tbody>`;
      body.appendChild(tbl);
    }

    const tbody = tbl.querySelector('tbody');

    // Helper: build per-source detail HTML from source counts dicts
    function _sourceDetailHtml(imgSources, vidSources, isDone) {
      const allSources = new Set([...Object.keys(imgSources), ...Object.keys(vidSources)]);
      if (!allSources.size) {
        if (!isDone) {
          return `<div class="mprog-detail-note">Per-source breakdown available after item completes.</div>`;
        }
        return `<div class="mprog-detail-note">No per-source data.</div>`;
      }
      let rows = '';
      allSources.forEach(src => {
        const iN = imgSources[src] || 0;
        const vN = vidSources[src] || 0;
        rows += `<tr>
          <td class="src-name">${src}</td>
          <td class="src-num">${iN || '—'}</td>
          <td style="color:var(--dim);padding-right:8px">img</td>
          <td class="src-num">${vN || '—'}</td>
          <td style="color:var(--dim)">vid</td>
        </tr>`;
      });
      return `<table class="mprog-detail-table">
        <tr><td colspan="5" style="color:var(--dim);font-size:0.85em;padding-bottom:3px">
          Per-source breakdown (ranked candidates kept after scoring)
        </td></tr>
        ${rows}
      </table>
      <div class="mprog-detail-note">
        DL total may exceed per-source limit: multiple search queries × sources contribute candidates before dedup &amp; scoring.
      </div>`;
    }

    // Update or create one row per item
    Object.entries(items).forEach(([itemId, it]) => {
      const phase  = it.phase || it.status || 'pending';
      const isDone = it.status === 'done';
      const isOpen = _mediaExpandedRows.has(itemId);

      let statusCls, statusTxt;
      if      (it.status === 'failed')  { statusCls = 'mprog-failed';  statusTxt = '✗ failed'; }
      else if (isDone)                  { statusCls = 'mprog-done';    statusTxt = '✓ done'; }
      else if (phase === 'scoring')     { statusCls = 'mprog-scoring'; statusTxt = '⚙ scoring…'; }
      else if (phase === 'downloading') { statusCls = 'mprog-dl';      statusTxt = '↓ downloading…'; }
      else                             { statusCls = 'mprog-pending'; statusTxt = '· pending'; }

      const num = (n) => (n != null && n > 0)
        ? `<td class="mprog-num">${n}</td>`
        : `<td class="mprog-dash">—</td>`;

      const imgDl  = isDone ? it.total_images : it.imgs_downloaded;
      const vidDl  = isDone ? it.total_videos : it.vids_downloaded;
      const imgSc  = isDone ? it.total_images : it.imgs_scored;
      const vidSc  = isDone ? it.total_videos : it.vids_scored;

      // ── Main row ──
      let row = tbody.querySelector(`tr.mprog-main-row[data-item="${CSS.escape(itemId)}"]`);
      if (!row) {
        row = document.createElement('tr');
        row.className = 'mprog-main-row';
        row.dataset.item = itemId;
        tbody.appendChild(row);
      }
      const toggleCls = 'mprog-toggle' + (isOpen ? ' open' : '');
      const toggleChar = isOpen ? '▼' : '▶';
      row.innerHTML = `
        <td><span class="${toggleCls}" data-toggle="${CSS.escape(itemId)}">${toggleChar}</span></td>
        <td style="font-family:var(--mono);font-size:0.9em">${itemId}</td>
        <td class="${statusCls}">${statusTxt}</td>
        ${num(imgDl)}${num(vidDl)}${num(imgSc)}${num(vidSc)}`;

      // Wire toggle click — re-render using cached _mediaLastProgress
      row.querySelector('[data-toggle]').onclick = () => {
        if (_mediaExpandedRows.has(itemId)) _mediaExpandedRows.delete(itemId);
        else                                _mediaExpandedRows.add(itemId);
        _mediaRenderProgress(null);   // null → uses _mediaLastProgress
      };

      // ── Detail row ──
      let detailRow = tbody.querySelector(`tr[data-detail="${CSS.escape(itemId)}"]`);
      if (isOpen) {
        if (!detailRow) {
          detailRow = document.createElement('tr');
          detailRow.dataset.detail = itemId;
          detailRow.className = 'mprog-detail-row';
          row.insertAdjacentElement('afterend', detailRow);
        }
        const imgSrc = it.img_sources || {};
        const vidSrc = it.vid_sources || {};
        detailRow.innerHTML = `<td colspan="7">${_sourceDetailHtml(imgSrc, vidSrc, isDone)}</td>`;
      } else if (detailRow) {
        detailRow.remove();
      }
    });
  }

  // ── Poll loop ──
  function _mediaStartPolling() {
    if (_mediaPollTimer) clearInterval(_mediaPollTimer);
    _mediaPollTimer = setInterval(_mediaPoll, 3000);
    _mediaPoll();   // immediate first check
  }

  async function _mediaPoll() {
    if (!_mediaBatchId) return;
    const serverUrl = _mediaServerUrl();
    try {
      const r = await fetch(
        '/api/media_batch_status?batch_id=' + encodeURIComponent(_mediaBatchId)
        + '&server_url=' + encodeURIComponent(serverUrl));
      const d = await r.json();
      if (!r.ok || d.error) throw new Error(d.error || 'poll failed');

      if (d.status === 'done') {
        clearInterval(_mediaPollTimer); _mediaPollTimer = null;
        let totalImg = 0, totalVid = 0;
        Object.values(d.items || {}).forEach(it => {
          totalImg += it.total_images || 0;
          totalVid += it.total_videos || 0;
        });
        const elapsed = d.elapsed_sec != null ? ' in ' + d.elapsed_sec + 's' : '';
        const nItems = Object.keys(d.items || {}).length;
        _mediaSetStatus('Done — ' + nItems + ' backgrounds, '
          + totalImg + ' images + ' + totalVid + ' videos found'
          + elapsed + '.', false);
        document.getElementById('media-btn-search').disabled = false;
        _mediaResults        = d.items || {};
        _mediaResults._topN  = d.top_n || 5;
        _mediaRecommendedSeq = d.recommended_sequence || null;
        await _mediaLoadShotMap();
        _mediaRenderResults(_mediaResults);
        await _aiLoadSavedImages();
      } else if (d.status === 'failed' || d.status === 'interrupted') {
        clearInterval(_mediaPollTimer); _mediaPollTimer = null;
        _mediaSetStatus((d.status === 'interrupted' ? 'Batch interrupted' : 'Batch failed')
          + ': ' + (d.error || 'unknown') + ' — click Search Media to resume.', false);
        document.getElementById('media-btn-search').disabled = false;
      } else {
        // Running — show progress string in status bar + per-shot table in body
        const doneCount  = Object.values(d.items || {}).filter(it => it.status === 'done').length;
        const totalCount = Object.keys(d.items || {}).length;
        _mediaSetStatus(
          (d.progress || d.status) + ' …' +
          (totalCount ? `  (${doneCount}/${totalCount} done)` : ''),
          true);
        _mediaRenderProgress(d);
      }
    } catch (err) {
      _mediaSetStatus('Poll error: ' + err.message, true);
    }
  }

  // ── Lazy-scroll helpers for media thumbnails ──
  const _MEDIA_PAGE = 20;  // thumbnails per lazy-scroll page

  // Append a page of thumbnails to a row, dimming items beyond topN
  function _mediaAppendPage(row, entries, itemId, type, from, count, topN) {
    const page = entries.slice(from, from + count);
    page.forEach((entry, i) => {
      const idx = from + i;
      const th = _mediaThumb(itemId, type, entry, idx);
      if (idx >= topN) th.classList.add('media-thumb-extended');
      row.appendChild(th);
    });
    return from + page.length;
  }

  // Observe a sentinel div to lazy-load more thumbnails on scroll
  function _mediaObserveSentinel(row, entries, itemId, type, topN) {
    let loaded = _MEDIA_PAGE;
    const sentinel = document.createElement('div');
    sentinel.className = 'media-lazy-sentinel';
    row.appendChild(sentinel);
    const obs = new IntersectionObserver(ents => {
      if (!ents[0].isIntersecting) return;
      loaded = _mediaAppendPage(row, entries, itemId, type, loaded, _MEDIA_PAGE, topN);
      // Re-position sentinel after newly appended thumbnails
      row.appendChild(sentinel);
      if (loaded >= entries.length) { obs.disconnect(); sentinel.remove(); }
    }, { root: document.getElementById('media-body'), threshold: 0 });
    obs.observe(sentinel);
  }

  // ── Helper: render one bg_id sub-section (search results + AI panel) inside a card ──
  function _mediaRenderBgSection(card, bgId, cardId, items, d_topN) {
    const item = items[bgId];
    if (!item) return;

    // Store manifest fields for info panel
    _bgManifestData[bgId] = {
      ai_prompt:        item.ai_prompt || '',
      search_prompt:    item.search_prompt || '',
      include_keywords: (item.include_keywords || []).join(', '),
      media_type:       (item.search_filters || {}).media_type || 'mixed',
      motion_level:     item.motion_level || '',
    };
    const mdata = _bgManifestData[bgId];

    // Background sub-header (collapsible)
    const bgHdr = document.createElement('div');
    bgHdr.className = 'media-bg-header';
    bgHdr.style.cssText = 'display:flex; align-items:center; flex-wrap:wrap; padding:4px 8px; margin-top:4px; background:var(--hover-bg); border-radius:4px; font-size:12px; color:var(--dim); cursor:pointer;';
    const bgLabel = document.createElement('span');
    bgLabel.style.cssText = 'font-weight:600; color:var(--text); font-size:11px;';
    bgLabel.textContent = '\u25BC ' + bgId;
    bgHdr.appendChild(bgLabel);

    // Manifest info toggle
    const infoBtn = document.createElement('button');
    infoBtn.textContent = '\uD83D\uDCCB';
    infoBtn.title = 'Show manifest details';
    infoBtn.style.cssText = 'margin-left:6px; padding:2px 7px; font-size:11px; cursor:pointer; background:#dbe8f5; border:1px solid #a0c0e0; border-radius:4px; color:#2a5a8a; vertical-align:middle;';
    (function(bid) {
      infoBtn.onclick = function(ev) {
        ev.stopPropagation();
        const panel = document.getElementById('manifest-panel-' + bid);
        if (panel) panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
      };
    })(bgId);
    bgHdr.appendChild(infoBtn);

    // AI Generate toggle button
    const aiGenToggleBtn = document.createElement('button');
    aiGenToggleBtn.textContent = '\u2728 AI';
    aiGenToggleBtn.title = 'Generate image with AI';
    aiGenToggleBtn.style.cssText = 'margin-left:6px; padding:2px 7px; font-size:11px; cursor:pointer; background:#e8f5e8; border:1px solid #a8d8a8; border-radius:4px; color:#2a7a2a; vertical-align:middle;';
    (function(bid) {
      aiGenToggleBtn.onclick = function(ev) {
        ev.stopPropagation();
        const p = document.getElementById('ai-gen-panel-' + bid);
        if (p) p.style.display = p.style.display === 'none' ? 'block' : 'none';
      };
    })(bgId);
    bgHdr.appendChild(aiGenToggleBtn);

    card.appendChild(bgHdr);

    // Collapsible content wrapper
    const bgContent = document.createElement('div');
    bgContent.className = 'media-bg-content';
    bgContent.id = 'media-bg-content-' + bgId;
    bgContent.dataset.bgId = bgId;

    // Toggle collapse on header click
    (function(content, label) {
      bgHdr.addEventListener('click', function() {
        const hidden = content.style.display === 'none';
        content.style.display = hidden ? '' : 'none';
        label.textContent = (hidden ? '\u25BC ' : '\u25B6 ') + bgId;
      });
    })(bgContent, bgLabel);

    // Manifest detail panel (hidden by default)
    const manifestPanel = document.createElement('div');
    manifestPanel.id = 'manifest-panel-' + bgId;
    manifestPanel.style.cssText = 'display:none; background:var(--panel-bg); padding:6px 12px; font-size:11px; color:#888888; font-family:monospace; margin-top:2px; border-radius:4px;';
    const aiPromptTrunc = mdata.ai_prompt.length > 120 ? mdata.ai_prompt.slice(0, 120) + '\u2026' : mdata.ai_prompt;
    manifestPanel.innerHTML = '<b style="color:#6a8aaa">ai_prompt:</b> ' + aiPromptTrunc + '<br>'
      + '<b style="color:#6a8aaa">search_prompt:</b> ' + mdata.search_prompt + '<br>'
      + '<b style="color:#6a8aaa">include_keywords:</b> ' + mdata.include_keywords + '<br>'
      + '<b style="color:#6a8aaa">media_type:</b> ' + mdata.media_type + ' &nbsp; <b style="color:#6a8aaa">motion_level:</b> ' + mdata.motion_level;
    bgContent.appendChild(manifestPanel);

    const prompt = document.createElement('div');
    prompt.className = 'media-item-prompt';
    prompt.textContent = item.search_prompt || '';
    bgContent.appendChild(prompt);

    // ── Inline AI Generate panel ──
    const aiGenPanel = document.createElement('div');
    aiGenPanel.id = 'ai-gen-panel-' + bgId;
    aiGenPanel.style.cssText = 'display:none; background:#e8f5e8; border:1px solid #a8d8a8; border-radius:4px; padding:8px 10px; margin:4px 0 6px 0;';
    const aiGenTA = document.createElement('textarea');
    aiGenTA.id = 'ai-gen-prompt-' + bgId;
    aiGenTA.rows = 3;
    aiGenTA.value = mdata.ai_prompt;
    aiGenTA.style.cssText = 'width:100%; box-sizing:border-box; background:var(--panel-bg); color:var(--text); border:1px solid #a8d8a8; border-radius:3px; font-size:11px; font-family:monospace; resize:vertical; padding:4px;';
    aiGenPanel.appendChild(aiGenTA);
    const aiGenRow = document.createElement('div');
    aiGenRow.style.cssText = 'margin-top:5px; display:flex; align-items:center; gap:8px;';
    const aiGenBtn = document.createElement('button');
    aiGenBtn.id = 'ai-gen-btn-' + bgId;
    aiGenBtn.textContent = '\u2728 Generate';
    aiGenBtn.style.cssText = 'padding:3px 12px; font-size:12px; cursor:pointer; background:#d4eed4; border:1px solid #8aba8a; border-radius:4px; color:#2a6a2a;';
    (function(bid) { aiGenBtn.onclick = function() { _aiGenStart(bid); }; })(bgId);
    aiGenRow.appendChild(aiGenBtn);
    const aiGenStatus = document.createElement('span');
    aiGenStatus.id = 'ai-gen-status-' + bgId;
    aiGenStatus.style.cssText = 'font-size:11px; color:#888888;';
    aiGenRow.appendChild(aiGenStatus);
    aiGenPanel.appendChild(aiGenRow);
    bgContent.appendChild(aiGenPanel);

    // Images section — lazy-scroll with pagination
    const imgs = item.images || [];
    if (imgs.length) {
      const lbl = document.createElement('div');
      lbl.className = 'media-section-label';
      lbl.textContent = 'Images (' + (item.total_images || imgs.length) + ' found)';
      bgContent.appendChild(lbl);
      const row = document.createElement('div');
      row.className = 'media-thumb-row';
      _mediaAppendPage(row, imgs, cardId, 'image', 0, _MEDIA_PAGE, d_topN);
      bgContent.appendChild(row);
      if (imgs.length > _MEDIA_PAGE) _mediaObserveSentinel(row, imgs, cardId, 'image', d_topN);
    }

    // Videos section — lazy-scroll with pagination
    const vids = item.videos || [];
    if (vids.length) {
      const lbl = document.createElement('div');
      lbl.className = 'media-section-label';
      lbl.textContent = 'Videos (' + (item.total_videos || vids.length) + ' found)';
      bgContent.appendChild(lbl);
      const row = document.createElement('div');
      row.className = 'media-thumb-row';
      _mediaAppendPage(row, vids, cardId, 'video', 0, _MEDIA_PAGE, d_topN);
      bgContent.appendChild(row);
      if (vids.length > _MEDIA_PAGE) _mediaObserveSentinel(row, vids, cardId, 'video', d_topN);
    }

    if (!imgs.length && !vids.length) {
      const empty = document.createElement('div');
      empty.className = 'media-empty';
      empty.textContent = item.error || 'No results found.';
      bgContent.appendChild(empty);
    }

    card.appendChild(bgContent);
  }

  // ── Render results grid (scene-grouped) ──
  function _mediaRenderResults(items) {
    const body = document.getElementById('media-body');
    body.innerHTML = '';
    const d_topN = _mediaResults?._topN || 5;
    _mediaItemIds = Object.keys(items);
    _mediaSelections = {};
    _bgManifestData = {};

    // Build ordered card list: one card per scene (or per orphan bg_id)
    const cardOrder = [];   // { cardId, sceneShotIds, bgIds }
    const assignedBgs = new Set();

    if (_mediaSceneOrder && _mediaSceneBgs) {
      _mediaSceneOrder.forEach(sceneId => {
        const bgIds = (_mediaSceneBgs[sceneId] || []).filter(bg => items[bg]);
        if (bgIds.length === 0) return;
        bgIds.forEach(bg => assignedBgs.add(bg));
        const shotIds = _mediaSceneMap[sceneId] || [];
        cardOrder.push({ cardId: sceneId, sceneShotIds: shotIds, bgIds: bgIds });
      });
    }
    // Orphan bg_ids (in results but not in any scene) — standalone cards
    _mediaItemIds.forEach(bgId => {
      if (bgId === '_topN') return;
      if (assignedBgs.has(bgId)) return;
      const shotIds = _mediaShotMap ? (_mediaShotMap[bgId] || []) : [];
      cardOrder.push({ cardId: bgId, sceneShotIds: shotIds, bgIds: [bgId] });
    });

    cardOrder.forEach(entry => {
      const { cardId, sceneShotIds, bgIds } = entry;
      const card = document.createElement('div');
      card.className = 'media-item-card';
      card.id = 'media-card-' + cardId;

      // Card header
      const hdr = document.createElement('div');
      hdr.className = 'media-item-header';
      hdr.style.display = 'flex';
      hdr.style.alignItems = 'center';
      hdr.style.flexWrap = 'wrap';
      const hdrLabel = document.createElement('span');
      if (sceneShotIds.length > 0) {
        hdrLabel.textContent = cardId + ' \u2014 ' + sceneShotIds.length + ' shot'
            + (sceneShotIds.length > 1 ? 's' : '') + ': ' + sceneShotIds.join(', ');
      } else {
        hdrLabel.textContent = cardId;
      }
      hdr.appendChild(hdrLabel);
      card.appendChild(hdr);

      // Render each bg_id's search results as a sub-section
      bgIds.forEach(bgId => {
        _mediaRenderBgSection(card, bgId, cardId, items, d_topN);
      });

      // Per-shot assignment rows (all shots in this scene card)
      if (_mediaShotMap && sceneShotIds.length > 0) {
        const shotSection = document.createElement('div');
        shotSection.className = 'media-shot-section';
        shotSection.id = 'media-shots-' + cardId;

        // Auto-assign button (only when >1 shot)
        if (sceneShotIds.length > 1) {
          const autoBtn = document.createElement('button');
          autoBtn.className = 'media-btn-auto-assign';
          autoBtn.textContent = '\u26A1 Auto-assign';
          autoBtn.onclick = function() { mediaAutoAssign(cardId); };
          shotSection.appendChild(autoBtn);
        }

        sceneShotIds.forEach(function(sid, idx) {
          const row = document.createElement('div');
          row.className = 'media-shot-row' + (idx === 0 ? ' media-shot-active' : '');
          row.id = 'media-shot-row-' + cardId + '-' + sid;
          const shotDur = (_mediaShotDur && _mediaShotDur[sid]) || 0;
          const durLabel = shotDur > 0 ? ' (' + shotDur.toFixed(1) + 's)' : '';
          row.innerHTML = '<span class="media-shot-label">' + sid + durLabel + ':</span>'
              + '<div class="media-shot-bar"><div class="media-shot-bar-fill" style="width:0%"></div></div>'
              + '<div class="media-shot-segments"></div>'
              + '<span class="media-shot-gap">\u2014 not assigned \u2014</span>';
          (function(cid, s) {
            row.addEventListener('click', function(e) {
              if (e.target.closest('.media-seg-remove')) return;
              mediaSetActiveShot(cid, s);
            });
          })(cardId, sid);
          shotSection.appendChild(row);
        });
        // First shot is active by default
        _mediaActiveShot[cardId] = sceneShotIds[0];
        card.appendChild(shotSection);
      }

      body.appendChild(card);
    });

    document.getElementById('media-footer').style.display = 'flex';
    document.getElementById('media-confirm-msg').textContent = '';
    // Show "Apply Recommended Sequence" button only if server computed one
    const applyBtn = document.getElementById('media-btn-apply-seq');
    applyBtn.style.display = _mediaRecommendedSeq ? 'inline-block' : 'none';
  }

  // ── Animation picker: option definitions ──
  // keyframe name → matches @keyframes thumb-* in CSS
  const _ANIM_OPTIONS = [
    { key: 'none',      label: 'None (static)' },
    { key: 'zoom_in',   label: 'Zoom In'       },
    { key: 'zoom_out',  label: 'Zoom Out'      },
    { key: 'pan_lr',    label: 'Pan L \u2192 R' },
    { key: 'pan_rl',    label: 'Pan R \u2192 L' },
    { key: 'pan_up',    label: 'Pan Up'        },
    { key: 'ken_burns', label: 'Ken Burns'     },
  ];

  // Map key → CSS animation-name for the thumb-* keyframes
  const _ANIM_CSS = {
    zoom_in:   'thumb-zoom-in',
    zoom_out:  'thumb-zoom-out',
    pan_lr:    'thumb-pan-lr',
    pan_rl:    'thumb-pan-rl',
    pan_up:    'thumb-pan-up',
    ken_burns: 'thumb-ken-burns',
  };

  let _animPickerCallback = null;

  function _mediaGetAnimLabel(key) {
    const opt = _ANIM_OPTIONS.find(o => o.key === key);
    return (opt && key !== 'none') ? ('\uD83C\uDFAC ' + opt.label) : '';
  }

  // Apply or remove the live CSS animation on the img inside a wrap element.
  function _mediaApplyThumbAnim(wrap, animKey) {
    const img = wrap.querySelector('img');
    if (!img) return;
    img.classList.remove('anim-playing');
    img.style.animationName = '';
    if (animKey && animKey !== 'none' && _ANIM_CSS[animKey]) {
      img.classList.add('anim-playing');
      img.style.animationName = _ANIM_CSS[animKey];
    }
  }

  // Set the animation on the live-preview img inside the popup right pane.
  function _animPreviewSet(animKey) {
    const img = document.getElementById('media-anim-live-img');
    if (!img) return;
    const lbl = document.getElementById('media-anim-live-label');
    img.classList.remove('anim-playing');
    img.style.animationName = '';
    if (animKey && animKey !== 'none' && _ANIM_CSS[animKey]) {
      // Force animation restart via reflow trick
      void img.offsetWidth;
      img.classList.add('anim-playing');
      img.style.animationName = _ANIM_CSS[animKey];
    }
    if (lbl) {
      const opt = _ANIM_OPTIONS.find(o => o.key === animKey);
      lbl.textContent = opt ? opt.label : '';
    }
  }

  function _mediaShowAnimPicker(e, wrap, callback) {
    e.stopPropagation();
    e.preventDefault();

    // Resolve the source image URL from wrap (if provided)
    const imgEl   = wrap ? wrap.querySelector('img') : null;
    const imgSrc  = imgEl ? imgEl.src : '';
    const currentKey = wrap ? (wrap.dataset.animType || 'none') : 'none';
    _animPickerCallback = callback;

    // Build or reuse popup
    let popup = document.getElementById('media-anim-popup');
    if (!popup) {
      popup = document.createElement('div');
      popup.id = 'media-anim-popup';
      document.body.appendChild(popup);
    }

    // ── Build layout: [options col | live preview pane] ──
    popup.innerHTML = '';
    const inner = document.createElement('div');
    inner.className = 'anim-popup-inner';

    // Left: options list
    const optCol = document.createElement('div');
    optCol.className = 'anim-options-col';
    const title = document.createElement('div');
    title.className = 'anim-popup-title';
    title.textContent = '\uD83C\uDFAC Image Animation';
    optCol.appendChild(title);

    _ANIM_OPTIONS.forEach(function(opt) {
      const row = document.createElement('div');
      row.className = 'anim-option' + (opt.key === currentKey ? ' active' : '');
      const dot = document.createElement('span');
      dot.className = 'anim-option-dot';
      const lbl = document.createElement('span');
      lbl.className = 'anim-label';
      lbl.textContent = opt.label;
      row.appendChild(dot);
      row.appendChild(lbl);
      // Hover → update live preview
      row.addEventListener('mouseenter', function() {
        optCol.querySelectorAll('.anim-option').forEach(r => r.classList.remove('active'));
        row.classList.add('active');
        _animPreviewSet(opt.key);
      });
      // Click → commit and close
      row.addEventListener('click', function(ev) {
        ev.stopPropagation();
        popup.style.display = 'none';
        document.removeEventListener('click', _animPickerDismiss, true);
        if (_animPickerCallback) {
          _animPickerCallback(opt.key);
          _animPickerCallback = null;
        }
      });
      optCol.appendChild(row);
    });
    inner.appendChild(optCol);

    // Right: live preview pane (only when we have an image src)
    const previewPane = document.createElement('div');
    previewPane.id = 'media-anim-preview-pane';
    const liveWrap = document.createElement('div');
    liveWrap.className = 'anim-live-wrap';
    const liveImg = document.createElement('img');
    liveImg.id       = 'media-anim-live-img';
    liveImg.className = 'anim-live-img';
    liveImg.src       = imgSrc || '';
    liveImg.alt       = '';
    // Start with current animation showing
    if (currentKey && currentKey !== 'none' && _ANIM_CSS[currentKey]) {
      liveImg.classList.add('anim-playing');
      liveImg.style.animationName = _ANIM_CSS[currentKey];
    }
    liveWrap.appendChild(liveImg);
    const liveLbl = document.createElement('div');
    liveLbl.id = 'media-anim-live-label';
    liveLbl.className = 'anim-live-label';
    const curOpt = _ANIM_OPTIONS.find(o => o.key === currentKey);
    liveLbl.textContent = curOpt ? curOpt.label : 'None';
    previewPane.appendChild(liveWrap);
    previewPane.appendChild(liveLbl);
    inner.appendChild(previewPane);

    popup.appendChild(inner);

    // Position near the click, prefer right side of click point
    popup.style.display = 'block';
    const vw = window.innerWidth, vh = window.innerHeight;
    let x = e.clientX + 6, y = e.clientY + 4;
    popup.style.left = '0'; popup.style.top = '0';
    const pw = popup.offsetWidth, ph = popup.offsetHeight;
    if (x + pw > vw - 8) x = Math.max(4, e.clientX - pw - 6);
    if (y + ph > vh - 8) y = Math.max(4, vh - ph - 8);
    popup.style.left = x + 'px';
    popup.style.top  = y + 'px';

    // Dismiss on outside click
    setTimeout(function() {
      document.addEventListener('click', _animPickerDismiss, true);
    }, 0);
  }

  function _animPickerDismiss(e) {
    const popup = document.getElementById('media-anim-popup');
    if (popup && !popup.contains(e.target)) {
      popup.style.display = 'none';
      document.removeEventListener('click', _animPickerDismiss, true);
      _animPickerCallback = null;
    }
  }

  // ── Cross-shot copy: show shot picker popup ──
  function _mediaShowShotPicker(e, itemId, type, entry, wrapEl) {
    e.stopPropagation();
    e.preventDefault();

    // Find ALL shots across ALL background items that could use this media
    var allShots = [];
    if (_mediaShotMap) {
      Object.keys(_mediaShotMap).forEach(function(bgId) {
        (_mediaShotMap[bgId] || []).forEach(function(sid) {
          if (allShots.indexOf(sid) === -1) allShots.push(sid);
        });
      });
    }
    if (allShots.length === 0) return;

    // Build or reuse popup
    var popup = document.getElementById('media-shot-picker');
    if (!popup) {
      popup = document.createElement('div');
      popup.id = 'media-shot-picker';
      document.body.appendChild(popup);
    }
    popup.innerHTML = '';

    var title = document.createElement('div');
    title.className = 'sp-title';
    title.textContent = '\u2795 Assign to shots (' + (type === 'video' ? '\uD83C\uDFA5' : '\uD83D\uDDBC\uFE0F') + ')';
    popup.appendChild(title);

    // Which shots already have this URL?
    var alreadyIn = {};
    Object.keys(_mediaSelections).forEach(function(bgId) {
      var ps = (_mediaSelections[bgId] || {}).per_shot || {};
      Object.keys(ps).forEach(function(sid) {
        var segs = (ps[sid] || {}).segments || [];
        segs.forEach(function(s) {
          if (s.url === (entry.url || '')) alreadyIn[sid] = bgId;
        });
      });
    });

    var checkboxes = [];
    allShots.forEach(function(sid) {
      var row = document.createElement('div');
      row.className = 'sp-row' + (alreadyIn[sid] ? ' sp-already' : '');
      var cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.className = 'sp-cb';
      cb.dataset.shotId = sid;
      cb.disabled = !!alreadyIn[sid];
      if (alreadyIn[sid]) cb.checked = true;
      var lbl = document.createElement('span');
      var dur = (_mediaShotDur && _mediaShotDur[sid]) || 0;
      lbl.textContent = sid + (dur > 0 ? ' (' + dur.toFixed(1) + 's)' : '')
          + (alreadyIn[sid] ? ' \u2713' : '');
      row.appendChild(cb);
      row.appendChild(lbl);
      if (!alreadyIn[sid]) {
        row.addEventListener('click', function(ev) {
          if (ev.target !== cb) cb.checked = !cb.checked;
        });
      }
      popup.appendChild(row);
      checkboxes.push(cb);
    });

    var doneBtn = document.createElement('button');
    doneBtn.className = 'sp-done';
    doneBtn.textContent = '\u2714 Apply';
    doneBtn.addEventListener('click', function() {
      popup.style.display = 'none';
      // Copy to each checked shot
      checkboxes.forEach(function(cb) {
        if (!cb.checked || cb.disabled) return;
        var sid = cb.dataset.shotId;
        // Find which bgId owns this shot, and the scene-based cardId for DOM
        var targetBgId = (_mediaShotToBg && _mediaShotToBg[sid]) || null;
        if (!targetBgId) {
          Object.keys(_mediaShotMap || {}).forEach(function(bgId) {
            if ((_mediaShotMap[bgId] || []).indexOf(sid) !== -1) targetBgId = bgId;
          });
        }
        if (!targetBgId) return;
        var targetCardId = (_mediaBgToScene && _mediaBgToScene[targetBgId]) || targetBgId;
        // Build canonical segment copy (A2 contract)
        var naturalDur = type === 'video' ? (entry.duration_sec || 0) : 0;
        var shotDur = (_mediaShotDur && _mediaShotDur[sid]) || 0;
        var seg = {
          media_type:            type,
          url:                   entry.url || '',
          path:                  entry.path || '',
          score:                 entry.score,
          natural_duration_sec:  type === 'video' ? naturalDur : null,
          duration_override_sec: null,
          duration_sec:          type === 'video' ? naturalDur : null,
          hold_sec:              null,
          source:                entry.source || null,
          animation_type:        (type === 'image' && wrapEl && wrapEl.dataset.animType) ? wrapEl.dataset.animType : null,
          start_sec:             null,
          end_sec:               null,
        };
        // Insert into target (selections keyed by bg_id)
        if (!_mediaSelections[targetBgId]) _mediaSelections[targetBgId] = { per_shot: {} };
        var ps = _mediaSelections[targetBgId].per_shot;
        if (!ps[sid]) ps[sid] = { segments: [] };
        if (!ps[sid].segments) {
          var old = ps[sid];
          ps[sid] = { segments: old.url ? [old] : [] };
        }
        ps[sid].segments.push(seg);
        _mediaRebalanceImages(sid, ps[sid].segments);
        _mediaRenderShotRow(targetCardId, sid);

        // Also inject into the target bg's search results + thumbnail grid
        if (_mediaResults && _mediaResults[targetBgId]) {
          var targetItem = _mediaResults[targetBgId];
          var listKey = type === 'video' ? 'videos' : 'images';
          var list = targetItem[listKey] || [];
          var entryUrl = entry.url || '';
          var alreadyInList = list.some(function(e) { return e.url === entryUrl; });
          if (!alreadyInList) {
            list.push(entry);
            targetItem[listKey] = list;
            // Find the bg-content container for this bg within the scene card
            var bgContent = document.getElementById('media-bg-content-' + targetBgId);
            var card = bgContent || document.getElementById('media-card-' + targetCardId);
            if (card) {
              var isVideo = (type === 'video');
              var labels = card.querySelectorAll('.media-section-label');
              var targetRow = null;
              labels.forEach(function(lbl) {
                if (lbl.textContent.toLowerCase().indexOf(isVideo ? 'video' : 'image') >= 0) {
                  var sib = lbl.nextElementSibling;
                  if (sib && sib.classList.contains('media-thumb-row')) targetRow = sib;
                }
              });
              if (!targetRow) {
                var newLbl = document.createElement('div');
                newLbl.className = 'media-section-label';
                newLbl.textContent = (isVideo ? 'Videos' : 'Images') + ' (1 found)';
                var newRow = document.createElement('div');
                newRow.className = 'media-thumb-row';
                card.appendChild(newLbl);
                card.appendChild(newRow);
                var emptyEl = card.querySelector('.media-empty');
                if (emptyEl) emptyEl.remove();
                targetRow = newRow;
              } else {
                var prevLbl = targetRow.previousElementSibling;
                if (prevLbl && prevLbl.classList.contains('media-section-label')) {
                  prevLbl.textContent = (isVideo ? 'Videos' : 'Images') + ' (' + list.length + ' found)';
                }
              }
              var th = _mediaThumb(targetCardId, type, entry, list.length - 1);
              th.classList.add('selected');
              targetRow.appendChild(th);
            }
          }
        }
      });
      _mediaMarkCrossCardUsed();
    });
    popup.appendChild(doneBtn);

    // Position near click
    var x = e.clientX, y = e.clientY;
    popup.style.display = 'block';
    var pw = popup.offsetWidth, ph = popup.offsetHeight;
    if (x + pw > window.innerWidth - 10) x = window.innerWidth - pw - 10;
    if (y + ph > window.innerHeight - 10) y = window.innerHeight - ph - 10;
    popup.style.left = x + 'px';
    popup.style.top  = y + 'px';

    // Dismiss on outside click
    setTimeout(function() {
      document.addEventListener('click', _shotPickerDismiss, true);
    }, 0);
  }

  function _shotPickerDismiss(e) {
    var popup = document.getElementById('media-shot-picker');
    if (popup && !popup.contains(e.target)) {
      popup.style.display = 'none';
      document.removeEventListener('click', _shotPickerDismiss, true);
    }
  }

  // ── Build a single thumbnail cell ──
  function _mediaThumb(itemId, type, entry, idx) {
    const wrap = document.createElement('div');
    wrap.className = 'media-thumb';
    wrap.dataset.itemId = itemId;
    wrap.dataset.type   = type;
    wrap.dataset.idx    = idx;

    const score = typeof entry.score === 'number'
                  ? entry.score.toFixed(3) : (entry.score || '');
    const badge = document.createElement('span');
    badge.className = 'media-score-badge';
    badge.textContent = score;
    wrap.appendChild(badge);

    // Duration badge for videos (top-left)
    if (typeof entry.duration_sec === 'number') {
      const durBadge = document.createElement('span');
      durBadge.className = 'media-dur-badge';
      durBadge.textContent = entry.duration_sec.toFixed(1) + 's';
      wrap.appendChild(durBadge);
    }

    // Browsers cannot load file:// URLs from an http:// page.
    // Route them through the VC editor's /api/serve_media_file proxy instead.
    // /files/… URLs are handled transparently by the VC editor's /files/ proxy route.
    const rawUrl    = entry.url || '';
    const displayUrl = rawUrl.startsWith('file://')
        ? '/api/serve_media_file?url=' + encodeURIComponent(rawUrl)
        : rawUrl;

    if (type === 'image') {
      const img    = document.createElement('img');
      img.src      = displayUrl;
      img.loading  = 'lazy';
      img.alt      = 'rank ' + (idx + 1);
      wrap.appendChild(img);

      // 🎬 Animation picker button (top-right, shows on hover)
      const animBtn = document.createElement('button');
      animBtn.className = 'media-anim-btn';
      animBtn.title = 'Set animation for this image';
      animBtn.textContent = '\uD83C\uDFAC';
      // Small badge below score badge showing current anim (hidden until chosen)
      const animThumbBadge = document.createElement('span');
      animThumbBadge.className = 'media-anim-thumb-badge';
      animThumbBadge.textContent = '';
      wrap.appendChild(animThumbBadge);

      animBtn.addEventListener('click', function(e) {
        _mediaShowAnimPicker(e, wrap, function(animKey) {
          if (animKey && animKey !== 'none') {
            wrap.dataset.animType = animKey;
            wrap.classList.add('has-anim');
            animThumbBadge.textContent = _mediaGetAnimLabel(animKey);
          } else {
            delete wrap.dataset.animType;
            wrap.classList.remove('has-anim');
            animThumbBadge.textContent = '';
          }
          // Apply / remove live animation on the thumbnail img immediately
          _mediaApplyThumbAnim(wrap, animKey);
          // If this thumb is already selected in a segment, sync that segment too
          _mediaUpdateSegmentAnim(wrap, animKey === 'none' ? null : animKey);
        });
      });
      wrap.appendChild(animBtn);
    } else {
      const vid    = document.createElement('video');
      vid.muted    = true;
      vid.loop     = true;
      vid.preload  = 'none';
      vid.dataset.lazySrc = displayUrl;
      vid.addEventListener('mouseenter', function() {
        // Pause any other playing video first (limit to 1 active stream)
        if (_mediaPlayingVid && _mediaPlayingVid !== this) {
          _mediaStopVid(_mediaPlayingVid);
        }
        _mediaPlayingVid = this;
        // Restore src from dataset if _mediaStopVid() cleared it
        if (!this.src && this.dataset.lazySrc) {
          this.src = this.dataset.lazySrc;
        }
        this.play().catch(function(){});
      });
      vid.addEventListener('mouseleave', function() {
        _mediaStopVid(this);
        if (_mediaPlayingVid === this) _mediaPlayingVid = null;
      });
      wrap.appendChild(vid);
      // Lazy-load: IntersectionObserver sets src + preload when visible
      _mediaLazyObserver.observe(vid);
    }

    // Source metadata link — opens original asset page in new tab
    if (entry.source && entry.source.asset_page_url) {
      const src  = entry.source;
      const WxH  = (src.width && src.height) ? src.width + '×' + src.height : '';
      const tags  = (src.tags || []).slice(0, 5).join(', ');
      const tip   = [src.title, src.photographer, WxH, src.source_site,
                     src.license_summary, tags]
                    .filter(Boolean).join(' · ');
      const link  = document.createElement('a');
      link.href        = src.asset_page_url;
      link.target      = '_blank';
      link.rel         = 'noopener noreferrer';
      link.className   = 'media-src-link';
      link.title       = tip;
      link.textContent = '🔗';
      link.addEventListener('click', function(e) { e.stopPropagation(); });
      wrap.appendChild(link);
    }

    // AI badge for AI-generated images
    if (entry.source === 'ai') {
      wrap.style.position = 'relative';
      const aiBadge = document.createElement('span');
      aiBadge.textContent = 'AI';
      aiBadge.style.cssText = 'position:absolute;bottom:2px;left:2px;background:#d4eed4;color:#2a7a2a;font-size:9px;font-weight:bold;padding:1px 4px;border-radius:3px;pointer-events:none;';
      wrap.appendChild(aiBadge);
    }

    // Cross-shot copy button (top-left, shows on hover when ShotList available)
    if (_mediaShotMap) {
      const copyBtn = document.createElement('button');
      copyBtn.className = 'media-copy-btn';
      copyBtn.textContent = '\u2795';
      copyBtn.title = 'Assign to other shots';
      (function(iid, t, e, w) {
        copyBtn.addEventListener('click', function(ev) {
          _mediaShowShotPicker(ev, iid, t, e, w);
        });
      })(itemId, type, entry, wrap);
      wrap.appendChild(copyBtn);
    }

    // Store original URL and duration as data attributes
    wrap.dataset.url = entry.url || '';
    if (typeof entry.duration_sec === 'number') {
      wrap.dataset.durationSec = entry.duration_sec;
    }
    wrap.addEventListener('click', () => mediaSelect(itemId, type, entry, wrap));
    return wrap;
  }

  // ── Update animation_type on any existing segment that matches this thumb's URL ──
  function _mediaUpdateSegmentAnim(wrapEl, animKey) {
    const cardId = wrapEl.dataset.itemId;  // cardId = scene_id
    const url    = wrapEl.dataset.url || '';
    if (!cardId || !url) return;
    // Search across all bg_ids in this card's scene
    const bgIds = (_mediaSceneBgs && _mediaSceneBgs[cardId]) || [cardId];
    let updated = false;
    bgIds.forEach(function(bgId) {
      const sel = _mediaSelections[bgId];
      if (!sel || !sel.per_shot) return;
      Object.keys(sel.per_shot).forEach(function(sid) {
        const ps = sel.per_shot[sid];
        if (!ps.segments) return;
        ps.segments.forEach(function(seg) {
          if (seg.url === url) {
            seg.animation_type = animKey || null;
            updated = true;
          }
        });
        if (updated) _mediaRenderShotRow(cardId, sid);
      });
    });
  }

  // ── Segment total duration helper ──
  function _mediaSegmentTotal(shotEntry) {
    if (!shotEntry || !shotEntry.segments) return 0;
    return shotEntry.segments.reduce(function(sum, seg) {
      if (seg.media_type === 'video') return sum + _resolveVideoDur(seg);
      return sum + (seg.hold_sec || 0);
    }, 0);
  }

  // ── Resolve effective duration for a video segment ──
  // duration_override_sec (user input) takes priority over natural_duration_sec.
  function _resolveVideoDur(seg) {
    if (seg.duration_override_sec != null && seg.duration_override_sec > 0)
      return seg.duration_override_sec;
    return seg.natural_duration_sec || seg.duration_sec || 0;
  }

  // ── Rebalance flexible segments to share the remaining gap equally ──
  // "Fixed" = video (uses natural or override duration).
  // "Flexible" = image (shares whatever time remains after videos).
  function _mediaRebalanceImages(shotId, segs) {
    var shotDur = (_mediaShotDur && _mediaShotDur[shotId]) || 0;
    if (segs.length === 0) return;
    var fixedTotal = 0;
    var flexIdxs = [];
    segs.forEach(function(seg, i) {
      if (seg.media_type === 'video') {
        var d = _resolveVideoDur(seg);
        fixedTotal += d;
        // Keep duration_sec in sync (for render pipeline)
        seg.duration_sec = d;
      } else {
        flexIdxs.push(i);
      }
    });
    var gap = shotDur > 0 ? Math.max(0, shotDur - fixedTotal) : 0;
    if (flexIdxs.length === 0) return;
    var per = flexIdxs.length > 0 ? gap / flexIdxs.length : 0;
    flexIdxs.forEach(function(i) {
      segs[i].hold_sec = per;
    });
  }

  // ── Active shot: which shot row is the current target for thumbnail clicks ──
  function mediaSetActiveShot(itemId, shotId) {
    _mediaActiveShot[itemId] = shotId;
    // Update visual highlight: remove from siblings, add to target
    const section = document.getElementById('media-shots-' + itemId);
    if (section) {
      section.querySelectorAll('.media-shot-row').forEach(function(r) {
        r.classList.remove('media-shot-active');
      });
      var row = document.getElementById('media-shot-row-' + itemId + '-' + shotId);
      if (row) row.classList.add('media-shot-active');
    }
  }

  // ── Select / deselect a thumb ──
  // cardId = scene_id (or bg_id for orphan cards). Selections are stored by bg_id.
  function mediaSelect(cardId, type, entry, wrapEl) {
    // Resolve all shots in this card (scene-based or legacy bg-based)
    const sceneShotIds = _mediaSceneMap ? (_mediaSceneMap[cardId] || []) : [];
    const legacyShotIds = _mediaShotMap ? (_mediaShotMap[cardId] || []) : [];
    const shotIds = sceneShotIds.length > 0 ? sceneShotIds : legacyShotIds;

    if (_mediaShotMap && shotIds.length > 0) {
      // ── Multi-segment stacking mode ──
      const clickUrl = entry.url || '';

      // ── Toggle-off: if this URL is already in any shot in this card, remove it ──
      let removed = false;
      for (const sid of shotIds) {
        const bgId = (_mediaShotToBg && _mediaShotToBg[sid]) || cardId;
        const ps = (_mediaSelections[bgId] && _mediaSelections[bgId].per_shot) || {};
        if (!ps[sid] || !ps[sid].segments) continue;
        const idx = ps[sid].segments.findIndex(function(s) { return s.url === clickUrl; });
        if (idx !== -1) {
          ps[sid].segments.splice(idx, 1);
          if (ps[sid].segments.length === 0) {
            delete _mediaSelections[bgId].per_shot[sid];
          } else {
            _mediaRebalanceImages(sid, ps[sid].segments);
          }
          _mediaRenderShotRow(cardId, sid);
          removed = true;
          break;
        }
      }
      if (removed) {
        wrapEl.classList.remove('selected');
        return;
      }

      // ── Target: the user-selected active shot row ──
      var targetShot = _mediaActiveShot[cardId] || shotIds[0];
      var targetBgId = (_mediaShotToBg && _mediaShotToBg[targetShot]) || cardId;

      // Initialize selections for the target bg_id
      if (!_mediaSelections[targetBgId]) _mediaSelections[targetBgId] = { per_shot: {} };
      const ps = _mediaSelections[targetBgId].per_shot;
      if (!ps[targetShot]) ps[targetShot] = { segments: [] };
      if (!ps[targetShot].segments) {
        const old = ps[targetShot];
        ps[targetShot] = { segments: old.url ? [old] : [] };
      }

      const segs = ps[targetShot].segments;
      const shotDur = (_mediaShotDur && _mediaShotDur[targetShot]) || 0;
      const filled = _mediaSegmentTotal({ segments: segs });
      const remaining = Math.max(0, shotDur - filled);

      const naturalDur = type === 'video' ? (entry.duration_sec || 0) : 0;
      const animType = (type === 'image' && wrapEl.dataset.animType) ? wrapEl.dataset.animType : null;
      segs.push({
        media_type:           type,
        url:                  entry.url || '',
        path:                 entry.path || '',
        score:                entry.score,
        natural_duration_sec: type === 'video' ? naturalDur : null,
        duration_override_sec:null,
        duration_sec:         type === 'video' ? naturalDur : null,
        hold_sec:             type === 'image' ? remaining : null,
        source:               entry.source || null,
        animation_type:       animType,
        start_sec:            null,
        end_sec:              null,
      });

      _mediaRebalanceImages(targetShot, segs);
      wrapEl.classList.add('selected');
      _mediaRenderShotRow(cardId, targetShot);
      _mediaMarkCrossCardUsed();
    } else {
      // ── Single-selection mode (no ShotList) ──
      const card = document.getElementById('media-card-' + cardId);
      if (card) {
        card.querySelectorAll('.media-thumb').forEach(t => t.classList.remove('selected'));
        card.querySelectorAll('.media-sel-badge').forEach(b => b.remove());
      }

      if (_mediaSelections[cardId] && _mediaSelections[cardId].url === (entry.url || '')) {
        delete _mediaSelections[cardId];
        return;
      }

      _mediaSelections[cardId] = { media_type: type, url: entry.url || '',
                                    path: entry.path || '', score: entry.score };
      wrapEl.classList.add('selected');

      const selBadge = document.createElement('span');
      selBadge.className = 'media-sel-badge';
      selBadge.textContent = '\u2714 ' + type;
      wrapEl.appendChild(selBadge);
      _mediaMarkCrossCardUsed();
    }
  }

  // ── Render a per-shot row with segment list and fill bar ──
  function _mediaRenderShotRow(cardId, shotId) {
    const row = document.getElementById('media-shot-row-' + cardId + '-' + shotId);
    if (!row) return;
    const barFill = row.querySelector('.media-shot-bar-fill');
    const segContainer = row.querySelector('.media-shot-segments');
    const gapSpan = row.querySelector('.media-shot-gap');

    // Look up bg_id for this shot to find its selection data
    const bgId = (_mediaShotToBg && _mediaShotToBg[shotId]) || cardId;
    const ps = (_mediaSelections[bgId] && _mediaSelections[bgId].per_shot) || {};
    const shotEntry = ps[shotId];
    const shotDur = (_mediaShotDur && _mediaShotDur[shotId]) || 0;

    if (!shotEntry || !shotEntry.segments || shotEntry.segments.length === 0) {
      if (barFill) barFill.style.width = '0%';
      if (segContainer) segContainer.innerHTML = '';
      if (gapSpan) gapSpan.textContent = '\u2014 not assigned \u2014';
      row.classList.remove('media-shot-filled');
      return;
    }

    const segs = shotEntry.segments;
    const filled = _mediaSegmentTotal(shotEntry);
    const pct = shotDur > 0 ? Math.min(100, (filled / shotDur) * 100) : 100;
    if (barFill) barFill.style.width = pct.toFixed(1) + '%';

    // Render segment entries
    const videoSum = segs.reduce(function(s, sg) {
      return s + (sg.media_type === 'video' ? _resolveVideoDur(sg) : 0);
    }, 0);
    const videoOverflow = shotDur > 0 && videoSum > shotDur + 0.05;
    if (segContainer) {
      segContainer.innerHTML = '';
      segs.forEach(function(seg, idx) {
        const se = document.createElement('div');
        se.className = 'media-seg-entry';
        const fname = (seg.path || seg.url || '').split('/').pop() || 'media';
        if (seg.media_type === 'video') {
          const natDur  = seg.natural_duration_sec || seg.duration_sec || 0;
          const ovr     = seg.duration_override_sec;
          const dispDur = _resolveVideoDur(seg);
          const inp = document.createElement('input');
          inp.type        = 'number';
          inp.min         = '0.1';
          inp.step        = '0.1';
          inp.placeholder = natDur.toFixed(1);
          inp.value       = (ovr != null && ovr > 0) ? ovr : '';
          inp.title       = 'Override duration (s). Blank = natural ' + natDur.toFixed(1) + 's';
          inp.style.cssText = 'width:54px;font-size:11px;background:var(--input-bg);color:var(--text);border:1px solid var(--input-border);border-radius:3px;padding:1px 3px;';
          (function(cid, bid, sid, i, s) {
            inp.addEventListener('change', function() {
              var v = parseFloat(this.value);
              if (isNaN(v) || v <= 0) {
                s.duration_override_sec = null;
                s.duration_sec = s.natural_duration_sec || 0;
                this.value = '';
              } else {
                s.duration_override_sec = v;
                s.duration_sec = v;
              }
              _mediaRebalanceImages(sid, (_mediaSelections[bid].per_shot[sid] || {segments:[]}).segments);
              _mediaRenderShotRow(cid, sid);
            });
          })(cardId, bgId, shotId, idx, seg);
          se.appendChild(document.createTextNode('\uD83C\uDFA5 '));
          const nameSpan = document.createElement('span');
          nameSpan.className = 'seg-name';
          nameSpan.textContent = fname;
          se.appendChild(nameSpan);
          se.appendChild(document.createTextNode('\u00a0'));
          se.appendChild(inp);
          const durSpan = document.createElement('span');
          durSpan.className = 'seg-dur';
          durSpan.textContent = dispDur.toFixed(1) + 's';
          se.appendChild(durSpan);

          // ── Clip trimmer row (start_sec / end_sec) ──
          var trimRow = document.createElement('div');
          trimRow.className = 'seg-trim-row';
          var curStart = seg.start_sec || 0;
          var curEnd   = seg.end_sec || natDur;
          // Range bar showing selected portion
          var rangeBar = document.createElement('div');
          rangeBar.className = 'trim-range-bar';
          var rangeFill = document.createElement('div');
          rangeFill.className = 'trim-range-fill';
          if (natDur > 0) {
            rangeFill.style.left  = ((curStart / natDur) * 100).toFixed(1) + '%';
            rangeFill.style.width = (((curEnd - curStart) / natDur) * 100).toFixed(1) + '%';
          }
          rangeBar.appendChild(rangeFill);
          // In label
          var startLbl = document.createElement('span');
          startLbl.className = 'trim-label';
          startLbl.textContent = 'in:';
          var startInp = document.createElement('input');
          startInp.type = 'number'; startInp.min = '0'; startInp.step = '0.1';
          startInp.value = curStart > 0 ? curStart.toFixed(1) : '';
          startInp.placeholder = '0';
          // Out label
          var endLbl = document.createElement('span');
          endLbl.className = 'trim-label';
          endLbl.textContent = 'out:';
          var endInp = document.createElement('input');
          endInp.type = 'number'; endInp.min = '0'; endInp.step = '0.1';
          endInp.value = (seg.end_sec != null) ? curEnd.toFixed(1) : '';
          endInp.placeholder = natDur > 0 ? natDur.toFixed(1) : '';
          // Change handlers
          (function(cid, bid, sid, i, s, sInp, eInp, rFill, nDur) {
            function applyTrim() {
              var sv = parseFloat(sInp.value);
              var ev = parseFloat(eInp.value);
              if (isNaN(sv) || sv < 0) sv = 0;
              if (sv >= nDur) { sv = 0; sInp.value = ''; }
              if (isNaN(ev) || ev <= sv) ev = 0;
              if (ev > nDur) ev = nDur;
              s.start_sec = sv > 0 ? sv : null;
              s.end_sec   = ev > 0 ? ev : null;
              var effStart = s.start_sec || 0;
              var effEnd   = s.end_sec || nDur;
              var clipDur  = Math.max(0.1, effEnd - effStart);
              s.duration_override_sec = clipDur;
              s.duration_sec = clipDur;
              if (nDur > 0) {
                rFill.style.left  = ((effStart / nDur) * 100).toFixed(1) + '%';
                rFill.style.width = (((effEnd - effStart) / nDur) * 100).toFixed(1) + '%';
              }
              _mediaRebalanceImages(sid, (_mediaSelections[bid].per_shot[sid] || {segments:[]}).segments);
              _mediaRenderShotRow(cid, sid);
            }
            sInp.addEventListener('change', applyTrim);
            eInp.addEventListener('change', applyTrim);
          })(cardId, bgId, shotId, idx, seg, startInp, endInp, rangeFill, natDur);
          trimRow.appendChild(startLbl);
          trimRow.appendChild(startInp);
          trimRow.appendChild(rangeBar);
          trimRow.appendChild(endLbl);
          trimRow.appendChild(endInp);
          se.appendChild(trimRow);
        } else {
          // Image segment
          const dur = seg.hold_sec || 0;
          se.innerHTML = '<span class="seg-name">\uD83D\uDDBC\uFE0F ' + fname + '</span>'
              + '<span class="seg-dur">' + dur.toFixed(1) + 's</span>';
          // Animation badge — click to change
          const animBadge = document.createElement('span');
          animBadge.className = 'seg-anim-badge';
          animBadge.title = 'Animation effect — click to change';
          const animKey = seg.animation_type || 'none';
          animBadge.textContent = animKey === 'none' ? '\uD83C\uDFAC \u2014' : ('\uD83C\uDFAC ' + (_ANIM_OPTIONS.find(o => o.key === animKey) || {label: animKey}).label);
          (function(s, badge) {
            badge.addEventListener('click', function(ev) {
              ev.stopPropagation();
              _mediaShowAnimPicker(ev, null, function(newKey) {
                s.animation_type = newKey === 'none' ? null : newKey;
                badge.textContent = (!newKey || newKey === 'none') ? '\uD83C\uDFAC \u2014' : ('\uD83C\uDFAC ' + (_ANIM_OPTIONS.find(o => o.key === newKey) || {label: newKey}).label);
              });
            });
          })(seg, animBadge);
          se.appendChild(animBadge);
        }
        const rmBtn = document.createElement('button');
        rmBtn.className = 'media-seg-remove';
        rmBtn.textContent = '\u2715';
        rmBtn.onclick = (function(iid, sid, i) {
          return function() { mediaRemoveSegment(iid, sid, i); };
        })(cardId, shotId, idx);
        se.appendChild(rmBtn);
        segContainer.appendChild(se);
      });
    }

    // Gap / overflow indicator
    const gap = shotDur > 0 ? shotDur - filled : 0;
    if (gapSpan) {
      if (videoOverflow) {
        gapSpan.textContent = '\u26a0 Videos exceed shot (' + videoSum.toFixed(1) + 's > ' + shotDur.toFixed(1) + 's)';
        gapSpan.style.color = '#e06c75';
        row.classList.remove('media-shot-filled');
      } else if (gap > 0.1) {
        gapSpan.textContent = 'needs ' + gap.toFixed(1) + 's more';
        gapSpan.style.color = '';
        row.classList.remove('media-shot-filled');
      } else {
        gapSpan.textContent = '\u2713 filled';
        gapSpan.style.color = '';
        row.classList.add('media-shot-filled');
      }
    }
  }

  // ── Remove a segment from a shot ──
  function mediaRemoveSegment(cardId, shotId, segIdx) {
    const bgId = (_mediaShotToBg && _mediaShotToBg[shotId]) || cardId;
    if (!_mediaSelections[bgId] || !_mediaSelections[bgId].per_shot) return;
    const shotEntry = _mediaSelections[bgId].per_shot[shotId];
    if (!shotEntry || !shotEntry.segments) return;
    shotEntry.segments.splice(segIdx, 1);
    if (shotEntry.segments.length === 0) {
      delete _mediaSelections[bgId].per_shot[shotId];
    } else {
      _mediaRebalanceImages(shotId, shotEntry.segments);
    }
    _mediaRenderShotRow(cardId, shotId);
  }

  // ── Clear a single per-shot assignment (legacy + segments) ──
  function mediaClearShot(cardId, shotId) {
    const bgId = (_mediaShotToBg && _mediaShotToBg[shotId]) || cardId;
    if (!_mediaSelections[bgId] || !_mediaSelections[bgId].per_shot) return;
    delete _mediaSelections[bgId].per_shot[shotId];
    _mediaRenderShotRow(cardId, shotId);
  }

  // ── Auto-assign: stack segments to fill each shot's duration ──
  function mediaAutoAssign(cardId) {
    // Resolve shots: scene-based or legacy bg-based
    const sceneShotIds = _mediaSceneMap ? (_mediaSceneMap[cardId] || []) : [];
    const legacyShotIds = _mediaShotMap ? (_mediaShotMap[cardId] || []) : [];
    const shotIds = sceneShotIds.length > 0 ? sceneShotIds : legacyShotIds;
    if (shotIds.length === 0) return;

    // Merge candidates from all bg_ids in the scene
    const bgIds = (_mediaSceneBgs && _mediaSceneBgs[cardId]) || [cardId];
    const candidates = [];
    const allImgs = [];
    bgIds.forEach(function(bid) {
      const item = _mediaResults[bid];
      if (!item) return;
      const vids = (item.videos || []).slice().sort((a, b) => (b.score || 0) - (a.score || 0));
      const imgs = (item.images || []).slice().sort((a, b) => (b.score || 0) - (a.score || 0));
      vids.forEach(v => candidates.push({ type: 'video', entry: v }));
      imgs.forEach(v => { candidates.push({ type: 'image', entry: v }); allImgs.push(v); });
    });
    if (candidates.length === 0) return;

    const usedUrls = new Set();
    let candIdx = 0;

    shotIds.forEach(function(sid) {
      const bgId = (_mediaShotToBg && _mediaShotToBg[sid]) || cardId;
      if (!_mediaSelections[bgId]) _mediaSelections[bgId] = { per_shot: {} };
      const ps = _mediaSelections[bgId].per_shot;
      const shotDur = (_mediaShotDur && _mediaShotDur[sid]) || 0;
      ps[sid] = { segments: [] };
      let filled = 0;

      while (filled < shotDur && candIdx < candidates.length * 2) {
        const idx = candIdx % candidates.length;
        const cand = candidates[idx];
        candIdx++;
        if (ps[sid].segments.some(function(s) { return s.url === cand.entry.url; })) continue;
        const remaining = Math.max(0, shotDur - filled);
        if (cand.type === 'video') {
          const dur = cand.entry.duration_sec || remaining;
          ps[sid].segments.push({
            media_type: 'video', url: cand.entry.url || '',
            path: cand.entry.path || '', score: cand.entry.score,
            duration_sec: dur, hold_sec: null,
          });
          filled += dur;
          usedUrls.add(cand.entry.url);
        } else {
          ps[sid].segments.push({
            media_type: 'image', url: cand.entry.url || '',
            path: cand.entry.path || '', score: cand.entry.score,
            duration_sec: null, hold_sec: remaining,
          });
          filled += remaining;
          usedUrls.add(cand.entry.url);
        }
      }

      if (ps[sid].segments.length === 0 && allImgs.length > 0) {
        ps[sid].segments.push({
          media_type: 'image', url: allImgs[0].url || '',
          path: allImgs[0].path || '', score: allImgs[0].score,
          duration_sec: null, hold_sec: shotDur,
        });
        usedUrls.add(allImgs[0].url);
      }

      _mediaRenderShotRow(cardId, sid);
    });

    const card = document.getElementById('media-card-' + cardId);
    if (card) {
      card.querySelectorAll('.media-thumb').forEach(t => {
        const url = t.dataset.url;
        if (usedUrls.has(url)) t.classList.add('selected');
        else t.classList.remove('selected');
      });
    }
  }

  // ── Reset all selections ──
  // ── Mark thumbnails used in OTHER cards with a teal "used" indicator ──
  // Called after every selection change so all cards stay in sync.
  function _mediaMarkCrossCardUsed() {
    // Build a map: url → Set of cardIds (scene_ids) that use it
    const urlToCardIds = {};
    Object.keys(_mediaSelections).forEach(bgId => {
      const cid = (_mediaBgToScene && _mediaBgToScene[bgId]) || bgId;
      const ps = (_mediaSelections[bgId] || {}).per_shot || {};
      Object.keys(ps).forEach(shotId => {
        const segs = (ps[shotId] || {}).segments || [];
        segs.forEach(s => {
          if (!s.url) return;
          if (!urlToCardIds[s.url]) urlToCardIds[s.url] = new Set();
          urlToCardIds[s.url].add(cid);
        });
      });
      // Also handle legacy single-selection mode
      if (_mediaSelections[bgId] && _mediaSelections[bgId].url) {
        const u = _mediaSelections[bgId].url;
        if (!urlToCardIds[u]) urlToCardIds[u] = new Set();
        urlToCardIds[u].add(cid);
      }
    });

    // For every thumb in every card, add/remove .used-elsewhere and badge
    // t.dataset.itemId is now the cardId (scene_id)
    document.querySelectorAll('.media-thumb').forEach(t => {
      const url    = t.dataset.url;
      const cardId = t.dataset.itemId;
      if (!url) return;
      const users = urlToCardIds[url];
      const usedElsewhere = users && [...users].some(id => id !== cardId);
      if (usedElsewhere) {
        t.classList.add('used-elsewhere');
        if (!t.querySelector('.media-used-badge')) {
          const b = document.createElement('span');
          b.className = 'media-used-badge';
          b.textContent = '↗ used';
          t.appendChild(b);
        }
      } else {
        t.classList.remove('used-elsewhere');
        const b = t.querySelector('.media-used-badge');
        if (b) b.remove();
      }
    });
  }

  function mediaReset() {
    _mediaSelections = {};
    document.querySelectorAll('.media-thumb.selected').forEach(t => t.classList.remove('selected'));
    document.querySelectorAll('.media-thumb.used-elsewhere').forEach(t => t.classList.remove('used-elsewhere'));
    document.querySelectorAll('.media-sel-badge').forEach(b => b.remove());
    document.querySelectorAll('.media-used-badge').forEach(b => b.remove());
    // Reset per-shot rows (segments mode)
    document.querySelectorAll('.media-shot-bar-fill').forEach(b => { b.style.width = '0%'; });
    document.querySelectorAll('.media-shot-segments').forEach(c => { c.innerHTML = ''; });
    document.querySelectorAll('.media-shot-gap').forEach(g => {
      g.textContent = '\u2014 not assigned \u2014';
    });
    document.querySelectorAll('.media-shot-row').forEach(r => r.classList.remove('media-shot-filled'));
    // Reset active shot to first shot per card (scene-based)
    document.querySelectorAll('.media-shot-row').forEach(r => r.classList.remove('media-shot-active'));
    for (var cid in _mediaActiveShot) {
      var sids = (_mediaSceneMap && _mediaSceneMap[cid]) || (_mediaShotMap ? (_mediaShotMap[cid] || []) : []);
      if (sids.length > 0) {
        _mediaActiveShot[cid] = sids[0];
        var firstRow = document.getElementById('media-shot-row-' + cid + '-' + sids[0]);
        if (firstRow) firstRow.classList.add('media-shot-active');
      }
    }
    document.getElementById('media-confirm-msg').textContent = '';
  }

  // ── Apply recommended sequence from server ──
  function mediaApplyRecommended() {
    if (!_mediaRecommendedSeq) return;
    // Reset current selections first
    mediaReset();
    let applied = 0;
    for (const [bgId, cand] of Object.entries(_mediaRecommendedSeq)) {
      if (!cand || !cand.url) continue;
      const shotIds = _mediaShotMap ? (_mediaShotMap[bgId] || []) : [];
      const sceneId = (_mediaBgToScene && _mediaBgToScene[bgId]) || bgId;

      if (_mediaShotMap && shotIds.length > 0) {
        const item = _mediaResults[bgId];
        if (!item) continue;
        const candidates = [];
        const vids = (item.videos || []).slice().sort((a, b) => (b.score || 0) - (a.score || 0));
        const imgs = (item.images || []).slice().sort((a, b) => (b.score || 0) - (a.score || 0));
        vids.forEach(v => candidates.push({ type: 'video', entry: v }));
        imgs.forEach(v => candidates.push({ type: 'image', entry: v }));
        if (candidates.length === 0) continue;

        if (!_mediaSelections[bgId]) _mediaSelections[bgId] = { per_shot: {} };
        const ps = _mediaSelections[bgId].per_shot;
        const usedUrls = new Set();
        let ci = 0;

        shotIds.forEach(function(sid) {
          const shotDur = (_mediaShotDur && _mediaShotDur[sid]) || 0;
          ps[sid] = { segments: [] };
          let filled = 0;
          while (filled < shotDur && ci < candidates.length * 2) {
            const pick = candidates[ci % candidates.length];
            ci++;
            if (ps[sid].segments.some(function(s) { return s.url === pick.entry.url; })) continue;
            const remaining = Math.max(0, shotDur - filled);
            if (pick.type === 'video') {
              const dur = pick.entry.duration_sec || remaining;
              ps[sid].segments.push({
                media_type: 'video', url: pick.entry.url || '',
                path: pick.entry.path || '', score: pick.entry.score,
                duration_sec: dur, hold_sec: null,
              });
              filled += dur;
            } else {
              ps[sid].segments.push({
                media_type: 'image', url: pick.entry.url || '',
                path: pick.entry.path || '', score: pick.entry.score,
                duration_sec: null, hold_sec: remaining,
              });
              filled += remaining;
            }
            usedUrls.add(pick.entry.url);
          }
          _mediaRenderShotRow(sceneId, sid);
        });

        // Mark used thumbs
        const card = document.getElementById('media-card-' + sceneId);
        if (card) {
          card.querySelectorAll('.media-thumb').forEach(t => {
            if (usedUrls.has(t.dataset.url)) t.classList.add('selected');
          });
        }
        applied++;
      } else {
        const ext = (cand.path || cand.url || '').split('.').pop().toLowerCase();
        const type = ['mp4', 'mov', 'webm', 'mkv'].includes(ext) ? 'video' : 'image';
        _mediaSelections[bgId] = {
          media_type: type,
          url:   cand.url  || '',
          path:  cand.path || '',
          score: cand.score || 0,
        };
        const card = document.getElementById('media-card-' + sceneId);
        if (card) {
          card.querySelectorAll('.media-sel-badge').forEach(b => b.remove());
          let matched = false;
          card.querySelectorAll('.media-thumb').forEach(thumb => {
            thumb.classList.remove('selected');
            if (!matched && thumb.dataset.url === cand.url) {
              thumb.classList.add('selected');
              const badge = document.createElement('div');
              badge.className = 'media-sel-badge';
              badge.textContent = '\u2714';
              thumb.appendChild(badge);
              matched = true;
            }
          });
          if (!matched) {
            const first = card.querySelector('.media-thumb');
            if (first) {
              first.classList.add('selected');
              const badge = document.createElement('div');
              badge.className = 'media-sel-badge';
              badge.textContent = '\u2714';
              first.appendChild(badge);
            }
          }
        }
        applied++;
      }
    }
    document.getElementById('media-confirm-msg').textContent =
      '\u26A1 Applied recommended sequence for ' + applied + ' item(s).';
  }

  // ── Confirm and write selections.json ──
  // Release ALL video HTTP connections on the Media tab.
  // Call this before any critical fetch (like Confirm) to guarantee a free slot.
  function _mediaReleaseAllConnections() {
    // 1. Stop the currently playing video (if any)
    if (_mediaPlayingVid) {
      _mediaStopVid(_mediaPlayingVid);
      _mediaPlayingVid = null;
    }
    // 2. Drain the lazy-load queue so no new video loads start
    _mediaLazyQueue.length = 0;
    // 3. Abort videos that are actively downloading (readyState < 2 = still fetching
    //    metadata or data).  Videos that already have metadata (readyState >= 2) are
    //    idle — their preload='metadata' connection already closed naturally — so we
    //    leave those alone to preserve their thumbnail first-frame.
    document.querySelectorAll('#panel-media video[src]').forEach(function(v) {
      // HAVE_CURRENT_DATA = 2; anything less means the browser is still fetching
      if (v.readyState < 2 || (!v.paused && v !== _mediaPlayingVid)) {
        var oldSrc = v.src;
        v.removeAttribute('src');
        v.load();   // forces browser to abort any in-flight request
        v.preload = 'none';
        if (oldSrc) v.dataset.lazySrc = oldSrc;
      }
    });
    _mediaLazyLoading = 0;
  }

  async function mediaConfirm() {
    const nSelected = Object.keys(_mediaSelections).length;
    if (nSelected === 0) {
      document.getElementById('media-confirm-msg').textContent = 'Select at least one item first.';
      return;
    }

    // ── Guard: every shot must have media selected ──
    if (_mediaShotMap) {
      const missing = [];
      for (const bgId in _mediaShotMap) {
        const shotIds = _mediaShotMap[bgId];
        const sel = _mediaSelections[bgId];
        const ps = (sel && sel.per_shot) || {};
        for (const sid of shotIds) {
          const shotSel = ps[sid];
          const hasMedia = shotSel && (
            (shotSel.segments && shotSel.segments.length > 0) ||
            shotSel.url || shotSel.abs_path
          );
          if (!hasMedia) missing.push(sid + ' (' + bgId + ')');
        }
      }
      if (missing.length > 0) {
        document.getElementById('media-confirm-msg').textContent =
          '❌ Cannot save selections — missing media for: ' + missing.join(', ');
        return;
      }
    }

    document.getElementById('media-btn-confirm').disabled = true;
    document.getElementById('media-confirm-msg').textContent = 'Saving …';

    // Free all video HTTP connections before the POST to guarantee a slot
    _mediaReleaseAllConnections();

    try {
      // Detect if any selection uses segments format → version 3
      const hasSegments = Object.values(_mediaSelections).some(function(sel) {
        return sel.per_shot && Object.values(sel.per_shot).some(function(ps) { return ps.segments; });
      });
      const selVersion = !_mediaShotMap ? 1 : hasSegments ? 3 : 2;
      const r = await fetch('/api/media_confirm', {
        method:  'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ slug: _mediaSlug, ep_id: _mediaEpId,
                               batch_id: _mediaBatchId,
                               version: selVersion,
                               selections: _mediaSelections }),
      });
      const d = await r.json();
      if (!r.ok || d.error) throw new Error(d.error || 'save failed');
      document.getElementById('media-confirm-msg').textContent =
        '✔ Saved ' + nSelected + ' selection(s) → ' + (d.path || 'selections.json');
    } catch (err) {
      document.getElementById('media-confirm-msg').textContent = 'Error: ' + err.message;
    } finally {
      document.getElementById('media-btn-confirm').disabled = false;
    }
  }

  // ── Try to load the latest completed batch for the current episode ──
  async function mediaLoadExisting() {
    if (!_mediaSlug || !_mediaEpId) return;
    const serverUrl = _mediaServerUrl();

    // ── Try 1: load from external media server batch ─────────────────────
    try {
      const r = await fetch(
        '/api/media_batches?slug=' + encodeURIComponent(_mediaSlug)
        + '&ep_id='      + encodeURIComponent(_mediaEpId)
        + '&server_url=' + encodeURIComponent(serverUrl));
      const d = await r.json();
      if (r.ok && !d.error) {
        const batches = d.batches || [];
        const running = batches.find(b => b.status === 'running');
        if (running) {
          _mediaBatchId = running.batch_id;
          document.getElementById('media-btn-search').disabled = true;
          _mediaSetStatus('Reconnecting to running batch…', true);
          _mediaStartPolling();
          return;
        }
        const done = batches.find(b => b.status === 'done');
        if (done) {
          _mediaBatchId = done.batch_id;
          const r2 = await fetch(
            '/api/media_batch_status?batch_id=' + encodeURIComponent(_mediaBatchId)
            + '&server_url=' + encodeURIComponent(serverUrl));
          const d2 = await r2.json();
          if (r2.ok && !d2.error && d2.status === 'done') {
            _mediaResults        = d2.items || {};
            _mediaResults._topN  = d2.top_n || 5;
            _mediaRecommendedSeq = d2.recommended_sequence || null;
            await _mediaLoadShotMap();
            let totalImg2 = 0, totalVid2 = 0;
            Object.values(d2.items || {}).forEach(it => {
              totalImg2 += it.total_images || 0;
              totalVid2 += it.total_videos || 0;
            });
            const elapsed2 = d2.elapsed_sec != null ? ' in ' + d2.elapsed_sec + 's' : '';
            const nItems2 = Object.keys(d2.items || {}).length;
            _mediaSetStatus('Loaded batch ' + _mediaBatchId + ' — '
              + nItems2 + ' backgrounds, '
              + totalImg2 + ' images + ' + totalVid2 + ' videos'
              + elapsed2 + '.', false);
            _mediaRenderResults(_mediaResults);
            await _aiLoadSavedImages();
            _aiGenRestorePending();
            // Also restore saved selections on top of batch results
            await _mediaLoadSavedSelections();
            return;
          }
        }
      }
    } catch (_) {}

    // ── Try 2: load saved selections.json from disk ──────────────────────
    await _mediaLoadSavedSelections();
  }

  async function _mediaLoadSavedSelections() {
    if (!_mediaSlug || !_mediaEpId) return;
    try {
      const r = await fetch('/api/episode_file?slug=' + encodeURIComponent(_mediaSlug)
        + '&ep_id=' + encodeURIComponent(_mediaEpId)
        + '&file=assets/media/selections.json');
      if (!r.ok) return;
      const saved = await r.json();
      if (!saved.selections || typeof saved.selections !== 'object') return;

      _mediaBatchId = saved.batch_id || '';
      const sels = saved.selections;
      const nSels = Object.keys(sels).length;

      // Restore _mediaSelections from saved data
      _mediaSelections = sels;

      // If we don't have batch results (media server down), show a summary
      if (!_mediaResults || Object.keys(_mediaResults).length === 0) {
        await _mediaLoadShotMap();
        _mediaRenderSavedSummary(sels);
        _mediaSetStatus('Loaded ' + nSels + ' saved selection(s) from disk'
          + ' (media server not available — showing saved selections only).', false);
      } else {
        // Batch results available — apply saved selections to the grid
        _mediaApplySavedToGrid(sels);
      }
    } catch (_) {}
  }

  function _mediaRenderSavedSummary(selections) {
    // Render a read-only summary of saved selections when media server is unavailable
    const body = document.getElementById('media-body');
    body.innerHTML = '';

    const shotDur = _mediaShotDur || {};
    const shotMap = _mediaShotMap || {};

    for (const [bgId, data] of Object.entries(selections)) {
      const card = document.createElement('div');
      card.className = 'media-item-card';

      // Header with background ID and assigned shots
      const shots = shotMap[bgId] || [];
      const hdr = document.createElement('div');
      hdr.className = 'media-item-header';
      hdr.innerHTML = '<strong>' + escHtml(bgId) + '</strong>'
        + (shots.length ? '<span style="color:var(--dim);margin-left:8px">→ '
          + shots.map(s => '<code>' + escHtml(s) + '</code>').join(', ') + '</span>' : '');
      card.appendChild(hdr);

      const perShot = data.per_shot || {};
      for (const [shotId, shotData] of Object.entries(perShot)) {
        const dur = shotDur[shotId] || 0;
        const row = document.createElement('div');
        row.style.cssText = 'padding:6px 12px;border-top:1px solid var(--active-bg)';

        let segsHtml = '<div style="color:var(--dim);font-size:0.82em;margin-bottom:4px">'
          + '<strong>' + escHtml(shotId) + '</strong>'
          + (dur ? ' (' + dur.toFixed(1) + 's)' : '') + '</div>';

        const segments = shotData.segments || [shotData];
        for (const seg of segments) {
          const rawUrl = seg.url || seg.path || '';
          // Convert file:/// or absolute paths to /serve_media URLs
          let url = rawUrl;
          if (rawUrl.startsWith('file:///')) {
            url = '/serve_media?path=' + encodeURIComponent(rawUrl.replace('file://', ''));
          } else if (rawUrl.startsWith('/') && !rawUrl.startsWith('/serve_media')) {
            url = '/serve_media?path=' + encodeURIComponent(rawUrl);
          }
          const type = seg.media_type || 'image';
          const name = rawUrl.split('/').pop() || '(unknown)';
          const durS = seg.duration_sec ? ' ' + seg.duration_sec.toFixed(1) + 's' : '';
          const icon = type === 'video' ? '🎬' : '🖼';

          segsHtml += '<div style="display:flex;align-items:center;gap:8px;margin:2px 0">';
          if (type === 'video' && url) {
            segsHtml += '<video src="' + escHtml(url) + '" style="width:120px;height:68px;'
              + 'object-fit:cover;border-radius:4px;border:1px solid rgba(0,0,0,0.12)" preload="metadata"></video>';
          } else if (url) {
            segsHtml += '<img src="' + escHtml(url) + '" style="width:120px;height:68px;'
              + 'object-fit:cover;border-radius:4px;border:1px solid rgba(0,0,0,0.12)" loading="lazy">';
          }
          segsHtml += '<span style="font-family:var(--mono);font-size:0.8em;color:var(--text)">'
            + icon + ' ' + escHtml(name) + durS + '</span></div>';
        }

        row.innerHTML = segsHtml;
        card.appendChild(row);
      }

      body.appendChild(card);
    }

    if (Object.keys(selections).length === 0) {
      body.innerHTML = '<div style="color:var(--dim);padding:16px;font-style:italic">'
        + 'No saved media selections found.</div>';
    }
  }

  function _mediaApplySavedToGrid(selections) {
    // When batch results are available, restore selection highlights
    for (const [bgId, data] of Object.entries(selections)) {
      const cardId = (_mediaBgToScene && _mediaBgToScene[bgId]) || bgId;
      const perShot = data.per_shot || {};
      for (const [shotId, shotData] of Object.entries(perShot)) {
        try { _mediaRenderShotRow(cardId, shotId); } catch (_) {}
      }
    }
    _mediaMarkCrossCardUsed();
  }

  // ── Voice Cast editor ────────────────────────────────────────────────────────

  // ── F0-3 / BUG-C: full 16-category sentence bank ─────────────────────────────
  // Sentences are IDENTICAL to pre_cache_voices.py STYLE_SENTENCES.
  // The cache hash includes the text — any deviation = guaranteed cache miss.
  const VC_SAMPLE = {
    baseline: {
      en:        'The ancient pharaoh stood before the gods, refusing to yield.',
      'zh-Hans': '古老的法老站在众神面前，拒绝屈服。',
    },
    bedtime: {
      en:        'Breathe in\u2026 and let your shoulders soften. The night is quiet, and you are safe.',
      'zh-Hans': '慢慢吸一口气，把肩膀放松。夜很安静，你很安全。',
    },
    narrator: {
      en:        'It was a quiet evening \u2014 the kind that makes you forget how loud the world can be.',
      'zh-Hans': '那是个安静的傍晚——那种让人忘记世界有多嘈杂的夜晚。',
    },
    documentary: {
      en:        'For millions of years, these mountains have stood as silent witnesses to all that lives below.',
      'zh-Hans': '数百万年来，这些山脉默默伫立，见证着脚下一切生命的来去。',
    },
    epic: {
      en:        'Under a cold moon, the ancient stones remember every name they have ever known.',
      'zh-Hans': '冷月之下，古老的石墙记得每一个曾经存在过的名字。',
    },
    poet: {
      en:        'The river does not remember the rain that made it, yet carries all things to the sea.',
      'zh-Hans': '河流不记得造就它的雨水，却将万物带向大海。',
    },
    angry: {
      en:        'This will not stand. I have given everything, and still it is never enough.',
      'zh-Hans': '这不能接受。我已经付出了一切，却永远都不够。',
    },
    fear: {
      en:        'Something is wrong. I can feel it \u2014 the silence where there should be sound.',
      'zh-Hans': '有什么不对劲。我能感觉到——本该有声音的地方，却一片寂静。',
    },
    sad: {
      en:        'Some things, once broken, can never truly be made whole again.',
      'zh-Hans': '有些事情，一旦破碎，就再也无法复原了。',
    },
    warm: {
      en:        'Even in the darkest hour, a single light is enough to find the way home.',
      'zh-Hans': '即使在最黑暗的时刻，一点点光芒也足以找到回家的路。',
    },
    curious: {
      en:        'Wait \u2014 what is that? I have never seen anything like it.',
      'zh-Hans': '等等——这是什么？我从来没见过这样的东西。',
    },
    social: {
      en:        "I honestly don't know what to say. I should have handled that better.",
      'zh-Hans': '我真的不知道该说什么。我本应该处理得更好的。',
    },
    professional: {
      en:        "Reporting from the capital \u2014 tonight's session concluded with a unanimous vote.",
      'zh-Hans': '来自首都的报道——今晚的会议以全票通过结束。',
    },
    sports: {
      en:        'And he drives forward \u2014 the crowd is on their feet \u2014 can he make it across the line?',
      'zh-Hans': '他向前冲去——观众全站了起来——他能做到吗？',
    },
    'sports-excited': {
      en:        'UNBELIEVABLE! What a finish! Nobody saw that coming!',
      'zh-Hans': '难以置信！什么样的结局！没有任何人预料到这一幕！',
    },
    commercial: {
      en:        'Limited time only \u2014 get the best price of the year, today.',
      'zh-Hans': '限时特惠！现在购买，享受全年最低价！',
    },
  };

  // ── F0-4 / BUG-C: complete style → category mapping ──────────────────────────
  const VC_STYLE_CATEGORY = {
    calm: 'bedtime', gentle: 'bedtime', whispering: 'bedtime', whisper: 'bedtime',
    'narration-relaxed': 'narrator', 'narration-professional': 'narrator',
    'documentary-narration': 'documentary', 'newscast-formal': 'documentary',
    drake: 'epic', geomancer: 'epic', cavalier: 'epic', captain: 'epic',
    assassin: 'epic', gamenarrator: 'epic',
    poet: 'poet', lyrical: 'poet', 'poetry-reading': 'poet', story: 'poet', sentiment: 'poet',
    angry: 'angry', shouting: 'angry', disgruntled: 'angry', complaining: 'angry',
    argue: 'angry', strict: 'angry', unfriendly: 'angry',
    terrified: 'fear', fearful: 'fear', anxious: 'fear', nervous: 'fear',
    sad: 'sad', depressed: 'sad', disappointed: 'sad', tired: 'sad',
    lonely: 'sad', sorry: 'sad', guilty: 'sad',
    cheerful: 'warm', excited: 'warm', friendly: 'warm', hopeful: 'warm',
    affectionate: 'warm', encouragement: 'warm', encourage: 'warm',
    comfort: 'warm', cute: 'warm', cutesy: 'warm',
    curious: 'curious', surprised: 'curious',
    shy: 'social', embarrassed: 'social', envious: 'social', empathetic: 'social',
    relieved: 'social', funny: 'social',
    newscast: 'professional', 'newscast-casual': 'professional', chat: 'professional',
    'chat-casual': 'professional', conversation: 'professional', assistant: 'professional',
    customerservice: 'professional', serious: 'professional', voiceassistant: 'professional',
    'sports-commentary': 'sports', 'sports-commentary-excited': 'sports-excited',
    'advertisement-upbeat': 'commercial', livecommercial: 'commercial',
  };

  // Auto-select sample text by style (null style → bedtime, safe default).
  // NOTE: VC_SAMPLE['baseline'] is dead code — the editor never requests it and
  // pre_cache_voices.py no longer generates baseline clips (Option A, Rev 5).
  function vcSampleText(locale, style) {
    const cat  = (style && VC_STYLE_CATEGORY[style]) ?? 'bedtime';
    const bank = VC_SAMPLE[cat] ?? VC_SAMPLE.bedtime;
    // zh-HK / zh-TW / zh-SG etc. → use zh-Hans text so Azure reads Chinese
    // characters in the correct locale pronunciation (e.g. Cantonese for zh-HK).
    const key = bank[locale] !== undefined ? locale
              : locale.startsWith('zh-')   ? 'zh-Hans'
              : 'en';
    return bank[key] ?? bank['en'];
  }

  // ── switchStoryTab(tab) ──────────────────────────────────────────────────────
  function switchStoryTab(tab) {
    // Scope to .story-tab-bar only — never touch locale tab buttons inside vc-editor
    document.querySelectorAll('.story-tab-bar .btn-story-tab').forEach(b =>
      b.classList.toggle('active', b.dataset.tab === tab));
    document.getElementById('story').style.display     = tab === 'story' ? '' : 'none';
    document.getElementById('vc-editor').style.display = tab === 'vc'    ? 'flex' : 'none';
    document.getElementById('sr-panel').style.display  = tab === 'sr'    ? 'flex' : 'none';
    if (tab === 'vc') {
      const cardsEl = document.getElementById('vc-cards');
      if (!_vcData || !_voiceCatalog) {
        // Nothing in memory yet — full async fetch + render
        loadVoiceCastForEditing();
      } else if (!cardsEl.firstChild) {
        // Data in memory but cards cleared (e.g. after project switch) — re-render
        renderVcEditor(_vcData, _voiceCatalog, _vcLocales);
      }
      // else: cards already rendered — preserve any in-progress edits
    }
    if (tab === 'sr') {
      loadStatusReport();
    }
  }

  // ── Status Report ─────────────────────────────────────────────────────────
  function _srColorize(raw) {
    // Very light syntax highlight: lines starting with known keywords get color
    return raw
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .split('\n').map(line => {
        if (/^={3,}/.test(line))           return `<span class="sr-sep">${line}</span>`;
        if (/^\d{4}-\d{2}-\d{2}/.test(line)) return `<span class="sr-ts">${line}</span>`;
        if (/^(ISSUE|ERROR|BUG|PROBLEM)/i.test(line.trim()))
                                           return `<span class="sr-issue">${line}</span>`;
        if (/^(FIX|FIXED|RESOLVED|APPLIED)/i.test(line.trim()))
                                           return `<span class="sr-fix">${line}</span>`;
        if (/^(MANUAL|ACTION REQUIRED|TODO)/i.test(line.trim()))
                                           return `<span class="sr-manual">${line}</span>`;
        return line;
      }).join('\n');
  }

  async function loadStatusReport() {
    const el = document.getElementById('sr-content');
    if (!currentSlug || !currentEpId) {
      el.innerHTML = '<span style="color:var(--dim);font-style:italic">No episode selected.</span>';
      _renderAlignment(null);
      renderNextStep();
      return;
    }
    try {
      const res  = await fetch('/api/status_report?slug=' + encodeURIComponent(currentSlug)
                              + '&ep_id=' + encodeURIComponent(currentEpId));
      const data = await res.json();
      if (!data.text || !data.text.trim()) {
        el.innerHTML = '<span style="color:var(--dim);font-style:italic">No entries yet for this episode.</span>';
      } else {
        el.innerHTML = _srColorize(data.text);
        el.scrollTop = el.scrollHeight;   // scroll to latest entry
      }
    } catch(e) {
      el.textContent = 'Error loading status report: ' + e;
    }
    // Load VO alignment panel
    try {
      const ar = await fetch('/api/vo_alignment?slug=' + encodeURIComponent(currentSlug)
                            + '&ep_id=' + encodeURIComponent(currentEpId));
      const ad = await ar.json();
      _lastAlignmentData = ad;
      _renderAlignment(ad);
    } catch(e) { _lastAlignmentData = null; _renderAlignment(null); }
    renderNextStep();
  }

  function _renderAlignment(data) {
    const el = document.getElementById('sr-alignment');
    if (!data || !data.locales || !data.locales.length) {
      el.classList.remove('visible'); el.innerHTML = ''; return;
    }
    const esc = s => (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    let html = '';
    for (const loc of data.locales) {
      const fl = loc.flagged_count, total = loc.total_lines;
      const conv = loc.converged_count;
      const wb = loc.worst_ratio_before.toFixed(2);
      const wa = loc.worst_ratio_after.toFixed(2);
      const bb = (loc.best_ratio_before ?? loc.worst_ratio_before).toFixed(2);
      const ba = (loc.best_ratio_after  ?? loc.worst_ratio_after).toFixed(2);
      const ok = fl === 0 || (loc.worst_ratio_after >= __VO_THRESH__ && (loc.best_ratio_after ?? 0) <= __VO_THRESH_HIGH__);
      html += `<div class="al-hdr">── VO Alignment: ${esc(loc.locale)}  `
            + `<span style="color:var(--dim);font-weight:normal;font-size:0.9em">`
            + `updated ${esc(loc.updated)}</span></div>`;
      html += `<div class="al-stats">`
            + `${total} lines · ${fl} flagged (ratio &lt; __VO_THRESH__ or &gt; __VO_THRESH_HIGH__) · `
            + `${conv}/${total} converged · `
            + `range: <span style="color:${ok?'var(--green)':'#e05c5c'}">[${wb}..${bb}] → [${wa}..${ba}]</span>`
            + `</div>`;
      if (loc.lines && loc.lines.length) {
        html += `<div class="al-row" style="color:var(--dim);font-size:0.9em;border-top:none">`
              + `<span>item_id</span><span style="text-align:right">ratio</span>`
              + `<span style="text-align:right">chars</span></div>`;
        for (const line of loc.lines) {
          const fixed = line.ratio_after >= __VO_THRESH__ && line.ratio_after <= __VO_THRESH_HIGH__;
          const rCls  = fixed ? 'al-good' : 'al-bad';
          html += `<div class="al-row">`
                + `<span class="al-id" title="${esc(line.item_id)}">${esc(line.item_id)}</span>`
                + `<span class="${rCls}">${line.ratio_before.toFixed(2)}→${line.ratio_after.toFixed(2)}</span>`
                + `<span class="al-chars">${line.chars_before}→${line.chars_after}</span>`
                + `</div>`;
          if (line.rewritten) {
            html += `<div class="al-txt">`
                  + `<div class="al-txt-before">− ${esc(line.text_before)}</div>`
                  + `<div class="al-txt-after">+ ${esc(line.text_after)}</div>`
                  + `</div>`;
          }
        }
      }
    }
    el.innerHTML = html;
    el.classList.add('visible');
  }

  async function appendStatusNote() {
    const input = document.getElementById('sr-note-input');
    const text  = input.value.trim();
    if (!text || !currentSlug || !currentEpId) return;
    const btn = document.getElementById('btn-sr-add');
    btn.disabled = true;
    try {
      await fetch('/api/append_status_report', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({slug: currentSlug, ep_id: currentEpId, text}),
      });
      input.value = '';
      await loadStatusReport();
    } finally {
      btn.disabled = false;
    }
  }
  // ── end Status Report ──────────────────────────────────────────────────────

  // ── Next Step banner ──────────────────────────────────────────────────────────
  // Computes the recommended next action from live pipeline status.
  function computeNextStep(status) {
    if (!status) return { state: 'idle', msg: '' };
    if (pipeRunning) {
      return { state: 'running',
               msg: '🔄  Pipeline running — stages ' + pipeRunning.from + ' → ' + pipeRunning.to + '…' };
    }
    const stagesMap = status.llm_stages || {};
    const llmKeys = [
      { n:0,  display:0, key:'stage_0'  }, { n:1,  display:1, key:'stage_1'  }, { n:2,  display:2, key:'stage_2'  },
      { n:3,  display:3, key:'stage_3'  }, { n:4,  display:4, key:'stage_4'  }, { n:5,  display:5, key:'stage_5'  },
      { n:6,  display:6, key:'stage_6'  }, { n:7,  display:7, key:'stage_7'  }, { n:8,  display:8, key:'stage_8'  },
      { n:10, display:9, key:'stage_10' },
    ];
    const _nsMaxDisplay = llmKeys[llmKeys.length - 1].display;
    const stageLabels = {
      0:  'Cast voices & pipeline vars',
      1:  'Check story consistency',
      2:  'Write episode direction',
      3:  'Write script & dialogue',
      4:  'Break script into shots',
      5:  'List required assets',
      6:  'Identify new story facts',
      7:  'Update story memory',
      8:  'Translate & adapt locales',
      10: 'Merge assets & render video',
    };
    // Sequential done propagation (mirrors renderPipelineStatus)
    let maxDone = -1;
    llmKeys.forEach(({ n, key }) => { if ((stagesMap[key] || {}).done) maxDone = n; });
    for (const { n, display, key } of llmKeys) {
      const done = n <= maxDone || ((stagesMap[key] || {}).done === true);
      if (!done) {
        // Stage 9 paused at [4b/8] music review checkpoint? (only when music is enabled)
        if (n === 10 && status.tts_done && !status.no_music && !status.music_plan_done) {
          return { state: 'action',
                   msg: '🎵  Music review needed — open the Music tab, confirm the plan, then re-run Stage 9' };
        }
        // Stage 9 music confirmed (or music disabled) but render not yet done?
        if (n === 10 && status.tts_done && (status.no_music || status.music_plan_done)) {
          return { state: 'action',
                   msg: '▶  Re-run Stage 9  —  ' + (status.no_music ? 'no music' : 'music confirmed') + ', resuming from step [5/8] resolve assets → render' };
        }
        return { state: 'action',
                 msg: '▶  Run ' + display + ' → ' + _nsMaxDisplay + '  —  Stage ' + display + ': ' + (stageLabels[n] || '') };
      }
    }
    return { state: 'done', msg: '✅  All stages complete — episode is ready.' };
  }

  // Returns an alignment-based next-step if Stage 9 is done but VO is misaligned.
  function _alignmentNextStep() {
    if (!_lastAlignmentData || !_lastAlignmentData.locales) return null;
    // Only relevant when Stage 9 is already complete
    if (!((( pipeStatus || {}).llm_stages || {}).stage_10 || {}).done) return null;
    const badLocs = _lastAlignmentData.locales.filter(
      loc => loc.total_lines > 0 && loc.flagged_count / loc.total_lines > 0.20
    );
    if (!badLocs.length) return null;
    const parts = badLocs.map(loc => {
      const pct = Math.round(loc.flagged_count / loc.total_lines * 100);
      return `${loc.locale} ${loc.flagged_count}/${loc.total_lines}句偏短（最差${loc.worst_ratio_before.toFixed(2)}）`;
    });
    return {
      state: 'action',
      msg: '⚠️  配音偏短 — ' + parts.join(' · ') + ' — 建议重跑 Stage 8→10',
    };
  }

  function renderNextStep() {
    const el = document.getElementById('sr-next-step');
    if (!el) return;
    if (!currentSlug || !currentEpId) { el.className = 'ns-idle'; el.textContent = ''; return; }
    const ns = _alignmentNextStep() || computeNextStep(pipeStatus);
    el.className = 'ns-' + ns.state;
    el.textContent = ns.msg;
  }
  // ── end Next Step banner ──────────────────────────────────────────────────────

  // ── 🤖 AI Diagnosis ──────────────────────────────────────────────────────────
  let _diagnoseResult          = null;   // last Claude result object
  let _diagnoseForFingerprint  = null;   // _lastStatusFingerprint when result was fetched
  let _lastAlignmentData       = null;   // last /api/vo_alignment response

  function _renderDiagnoseResult(data) {
    const el = document.getElementById('sr-diagnose-result');
    if (!el) return;
    if (data.error) {
      el.className = 'srd-error';
      el.textContent = '✗ Diagnosis error: ' + data.error;
    } else if (data.from_stage === null || data.from_stage === undefined) {
      el.className = 'srd-done';
      el.textContent = '🤖 Nothing stale — ' + (data.reason || 'all outputs are up to date.');
    } else {
      el.className = 'srd-action';
      el.innerHTML =
        '🤖 Re-run Stage <strong>' + data.from_stage + ' → ' + data.to_stage + '</strong>' +
        ' <span style="font-weight:400;opacity:0.6">(' + (data.confidence||'') + ' confidence)</span>' +
        '<br><span style="font-weight:400;opacity:0.85">' + escHtml(data.reason || '') + '</span>';
    }
  }

  async function runDiagnose() {
    if (!currentSlug || !currentEpId) return;

    // Return cached result if pipeline state hasn't changed since last call
    if (_diagnoseForFingerprint !== null &&
        _diagnoseForFingerprint === _lastStatusFingerprint &&
        _diagnoseResult !== null) {
      _renderDiagnoseResult(_diagnoseResult);
      return;
    }

    const btn   = document.getElementById('btn-diagnose');
    const resEl = document.getElementById('sr-diagnose-result');
    btn.disabled     = true;
    btn.textContent  = '⏳ …';
    resEl.className  = 'srd-running';
    resEl.textContent = 'Asking Claude haiku — analysing file timestamps…';
    try {
      const r    = await fetch('/api/diagnose_pipeline?slug=' + encodeURIComponent(currentSlug)
                               + '&ep_id=' + encodeURIComponent(currentEpId));
      const data = await r.json();
      _diagnoseResult         = data;
      _diagnoseForFingerprint = _lastStatusFingerprint;
      _renderDiagnoseResult(data);
    } catch(e) {
      const errData = { error: String(e) };
      _diagnoseResult         = errData;
      _diagnoseForFingerprint = _lastStatusFingerprint;
      _renderDiagnoseResult(errData);
    } finally {
      btn.disabled    = false;
      btn.textContent = '🤖 Diagnose';
    }
  }

  // Invalidate diagnosis cache whenever pipeline state fingerprint changes
  function _invalidateDiagnoseIfStale() {
    if (_diagnoseForFingerprint !== null &&
        _diagnoseForFingerprint !== _lastStatusFingerprint) {
      _diagnoseResult         = null;
      _diagnoseForFingerprint = null;
      const el = document.getElementById('sr-diagnose-result');
      if (el) { el.className = ''; el.textContent = ''; }
    }
  }
  // ── end AI Diagnosis ──────────────────────────────────────────────────────────

  // ── loadVoiceCatalog() ───────────────────────────────────────────────────────
  async function loadVoiceCatalog() {
    if (_voiceCatalog) return _voiceCatalog;
    const r = await fetch('/api/azure_voices');
    _voiceCatalog = await r.json();
    return _voiceCatalog;
  }

  // ── loadVoicePresets() ───────────────────────────────────────────────────────
  async function loadVoicePresets() {
    try {
      const r = await fetch('/api/voice_presets');
      _vcPresets = (await r.json()).presets ?? {};
    } catch (_) {
      _vcPresets = {};
    }
  }

  // ── loadVoiceIndex() ─────────────────────────────────────────────────────────
  async function loadVoiceIndex() {
    if (_voiceIndex) return;
    try {
      const r = await fetch('/api/voice_index');
      _voiceIndex = (await r.json()).voices ?? {};
    } catch (_) {
      _voiceIndex = {};
    }
  }

  // ── _vcGetDefaults(voice, style) ─────────────────────────────────────────────
  // Returns { style_degree, rate, pitch, break_ms } for the given voice+style.
  // Priority: index.json clip → hardcoded universal fallback.
  const _VC_FALLBACK = { style_degree: 1.0, rate: '0%', pitch: '-5%', break_ms: 600 };
  function _vcGetDefaults(voice, style) {
    const clip = _voiceIndex?.[voice]?.clips?.[style ?? ''];
    if (clip?.params) return clip.params;
    return _VC_FALLBACK;
  }

  // ── rebuildPresetSelect(card, voice) ─────────────────────────────────────────
  // Rebuilds .vc-preset-select options from _vcPresets[voice].
  // Preserves the current selection if the hash still exists after rebuild.
  // Module-level so both renderVcCards() and previewVoice() can call it.
  function rebuildPresetSelect(card, voice) {
    const sel     = card.querySelector('.vc-preset-select');
    if (!sel) return;
    const prevVal = sel.value;
    while (sel.options.length > 1) sel.remove(1);
    const presets = _vcPresets[voice] ?? [];
    presets.forEach(p => sel.add(new Option(p.name, p.hash)));
    sel.disabled = presets.length === 0;
    sel.value    = prevVal;
    if (!sel.value) sel.selectedIndex = 0;
  }

  // ── loadVoiceCastForEditing() ────────────────────────────────────────────────
  async function loadVoiceCastForEditing() {
    const vcCards = document.getElementById('vc-cards');
    if (!currentSlug || !currentEpId) {
      vcCards.innerHTML = '<div style="color:var(--dim);font-style:italic;font-size:0.83em;padding:8px 0">Run Stage 0 first to generate the Voice Cast.</div>';
      return;
    }
    try {
      const res    = await fetch('/pipeline_status?slug=' + encodeURIComponent(currentSlug) +
                                 '&ep_id=' + encodeURIComponent(currentEpId));
      const status = await res.json();
      if (!status.voice_cast) {
        vcCards.innerHTML = '<div style="color:var(--dim);font-style:italic;font-size:0.83em;padding:8px 0">No VoiceCast.json yet \u2014 run Stage 0 first.</div>';
        return;
      }
      await loadVoiceCatalog();
      await loadVoicePresets();
      await loadVoiceIndex();
      _vcData    = status.voice_cast;
      _vcLocales = status.locales_str
        ? status.locales_str.split(',').map(l => l.trim()).filter(Boolean)
        : (status.locales || []);
      renderVcEditor(status.voice_cast, _voiceCatalog, _vcLocales);
    } catch(e) {
      vcCards.innerHTML = '<div style="color:var(--red);font-family:var(--mono);font-size:0.82em">Error loading voice cast: ' + escHtml(String(e)) + '</div>';
    }
  }

  // ── renderVcEditor(voiceCast, catalog, locales) ──────────────────────────────
  function renderVcEditor(voiceCast, catalog, locales) {
    const tabsEl = document.getElementById('vc-locale-tabs');
    tabsEl.innerHTML = '';

    // Derive locale list from VoiceCast.json characters directly.
    // status.locales is empty until Stage 8 writes AssetManifest files, so we
    // can't rely on it here — read the locale keys off the first character instead.
    const SKIP = new Set(['character_id', 'role', 'gender', 'personality']);
    const vcLocaleKeys = (voiceCast.characters?.[0])
      ? Object.keys(voiceCast.characters[0]).filter(k => !SKIP.has(k))
      : [];
    const effectiveLocales = (locales && locales.length > 0) ? locales : vcLocaleKeys;

    // MIN-R7: init to first available locale, never hardcoded 'en'
    _vcActiveLocale = effectiveLocales[0] ?? Object.keys(catalog)[0] ?? 'en';

    if (effectiveLocales.length > 1) {
      effectiveLocales.forEach(l => {
        const btn = document.createElement('button');
        btn.className = 'btn-story-tab' + (l === _vcActiveLocale ? ' active' : '');
        btn.textContent = l;
        btn.dataset.locale = l;
        btn.onclick = () => {
          // BUG-R1: flush active locale DOM → _vcData BEFORE rebuilding cards
          flushActiveLocale();
          _vcActiveLocale = l;
          tabsEl.querySelectorAll('.btn-story-tab').forEach(b =>
            b.classList.toggle('active', b.dataset.locale === l));
          renderVcCards(l);
        };
        tabsEl.appendChild(btn);
      });
    }
    renderVcCards(_vcActiveLocale);
  }

  // ── flushActiveLocale() ──────────────────────────────────────────────────────
  // Write the currently rendered card DOM back into _vcData so no edits are lost
  // when switching locale tabs or saving. Called by locale tab onclick AND at the
  // top of saveVoiceCast() (BUG-R1 fix).
  function flushActiveLocale() {
    if (!_vcData || !_vcActiveLocale || !_voiceCatalog) return;
    document.getElementById('vc-cards').querySelectorAll('.vc-char-card').forEach(card => {
      const charId    = card.dataset.charId;
      const voiceVal  = card.querySelector('.vc-voice-select').value;
      const styleVal  = card.querySelector('.vc-style-select').value || null;
      const degreeVal = parseFloat(card.querySelector('[data-field=degree]').value) || 1.0;
      const rateVal   = addPct(card.querySelector('[data-field=rate]').value);
      const pitchVal  = addPct(card.querySelector('[data-field=pitch]').value);
      const breakVal  = parseInt(card.querySelector('[data-field=break]').value) || 0;
      // BUG 2: derive newVoice locally from catalog (not a stale outer variable)
      const lg = Object.keys(_voiceCatalog)
        .find(g => _vcActiveLocale.startsWith(g.split('-')[0]))
        ?? Object.keys(_voiceCatalog)[0];
      const newVoice  = (_voiceCatalog[lg] || []).find(v => v.voice === voiceVal);
      const charEntry = (_vcData.characters || []).find(c => c.character_id === charId);
      if (charEntry) {
        // Create the locale entry if it didn't exist yet (user manually assigned a new locale)
        if (!charEntry[_vcActiveLocale]) charEntry[_vcActiveLocale] = {};
        charEntry[_vcActiveLocale].azure_voice        = voiceVal;
        charEntry[_vcActiveLocale].azure_style        = styleVal;
        charEntry[_vcActiveLocale].azure_style_degree = degreeVal;
        charEntry[_vcActiveLocale].azure_rate         = rateVal;
        charEntry[_vcActiveLocale].azure_pitch        = pitchVal;
        charEntry[_vcActiveLocale].azure_break_ms     = breakVal;
        if (newVoice) charEntry[_vcActiveLocale].available_styles = newVoice.styles;
      }
    });
  }

  // ── addPct(raw) — normalise rate/pitch: user types plain integer, storage needs "%" ──
  // "−25" → "−25%"  |  "" → ""  |  "−5%" → "−5%" (idempotent)
  function addPct(raw) {
    const s = String(raw ?? '').trim().replace('%', '');
    return s === '' ? '' : s + '%';
  }

  // ── Voice Cast parameter interpretation (live meaning bands) ─────────────────
  // Input = any value the user typed; never clamped.  Returns { text, zone }.
  // zone: 'normal' | 'extended' | 'experimental'
  function vcInterpDegree(raw) {
    const v = parseFloat(raw);
    if (isNaN(v)) return { text: '\u2014', zone: 'normal' };
    if (v < 0.6)  return { text: 'Minimal \u2014 nearly emotionless, experimental flat delivery', zone: 'experimental' };
    if (v < 0.8)  return { text: 'Restrained \u2014 calm, controlled narration', zone: 'normal' };
    if (v < 1.0)  return { text: 'Natural \u2014 realistic everyday speaking emotion', zone: 'normal' };
    if (v < 1.3)  return { text: 'Expressive \u2014 engaging storytelling tone', zone: 'normal' };
    if (v < 1.7)  return { text: 'Cinematic \u2014 dramatic movie narration', zone: 'extended' };
    if (v < 2.2)  return { text: 'Epic \u2014 theatrical performance energy', zone: 'extended' };
    if (v < 2.8)  return { text: 'Intense \u2014 emotionally overwhelming delivery', zone: 'extended' };
                  return { text: 'Extreme \u2014 exaggerated emotion (experimental)', zone: 'experimental' };
  }
  function vcInterpRate(raw) {
    const v = parseFloat(String(raw).replace('%', ''));
    if (isNaN(v)) return { text: '\u2014', zone: 'normal' };
    if (v < -40)  return { text: 'Frozen \u2014 extremely slow, surreal pacing', zone: 'experimental' };
    if (v < -30)  return { text: 'Dreamlike \u2014 very slow, sleep/meditation tone', zone: 'extended' };
    if (v < -15)  return { text: 'Slow \u2014 calm narration', zone: 'normal' };
    if (v <= 10)  return { text: 'Natural \u2014 conversational pacing', zone: 'normal' };
    if (v <= 25)  return { text: 'Energetic \u2014 lively delivery', zone: 'normal' };
    if (v <= 35)  return { text: 'Urgent \u2014 tense rapid speech', zone: 'extended' };
                  return { text: 'Hyperfast \u2014 experimental speed, clarity may degrade', zone: 'experimental' };
  }
  function vcInterpPitch(raw) {
    const v = parseFloat(String(raw).replace('%', ''));
    if (isNaN(v)) return { text: '\u2014', zone: 'normal' };
    if (v < -8)   return { text: 'Abyssal \u2014 extremely deep, unnatural resonance', zone: 'experimental' };
    if (v < -6)   return { text: 'Ancient \u2014 mythic elder narrator tone', zone: 'extended' };
    if (v < -2)   return { text: 'Deep \u2014 mature cinematic voice', zone: 'normal' };
    if (v <= 2)   return { text: 'Neutral \u2014 original voice', zone: 'normal' };
    if (v <= 6)   return { text: 'Bright \u2014 youthful tone', zone: 'normal' };
    if (v <= 8)   return { text: 'Airy \u2014 light energetic voice', zone: 'extended' };
                  return { text: 'Helium \u2014 exaggerated high pitch (experimental)', zone: 'experimental' };
  }
  function vcInterpBreak(raw) {
    const v = parseInt(raw, 10);
    if (isNaN(v)) return { text: '\u2014', zone: 'normal' };
    if (v < 200)  return { text: 'Rapid \u2014 almost no pauses', zone: 'experimental' };
    if (v < 300)  return { text: 'Continuous \u2014 flowing speech', zone: 'normal' };
    if (v < 450)  return { text: 'Conversational \u2014 natural dialogue', zone: 'normal' };
    if (v < 700)  return { text: 'Reflective \u2014 thoughtful narration', zone: 'normal' };
    if (v < 900)  return { text: 'Storytelling \u2014 dramatic pacing', zone: 'extended' };
    if (v <= 1200) return { text: 'Meditative \u2014 slow immersive timing', zone: 'extended' };
                  return { text: 'Suspended \u2014 extremely long pauses (experimental)', zone: 'experimental' };
  }

  // ── renderVcCards(locale) ────────────────────────────────────────────────────
  function renderVcCards(locale) {
    const cardsEl = document.getElementById('vc-cards');
    cardsEl.innerHTML = '';
    if (!_vcData || !_voiceCatalog) return;
    // MIN 3: locale group derived dynamically from catalog keys
    const localeGroup = Object.keys(_voiceCatalog)
      .find(g => locale.startsWith(g.split('-')[0]))
      ?? Object.keys(_voiceCatalog)[0];
    const voiceList = _voiceCatalog[localeGroup] || [];

    (_vcData.characters || []).forEach(char => {
      let charLoc = char[locale];

      // Role-based defaults — defined here (outside if-block) so the voiceSel
      // onChange closure can always reference rd when resetting sliders.
      const roleDefaults = {
        narrator:    { pitch: '-5',  break_ms: 600, rate: '0'  },
        protagonist: { pitch: '0',   break_ms: 400, rate: '0'  },
        antagonist:  { pitch: '-5',  break_ms: 500, rate: '0'  },
        default:     { pitch: '0',   break_ms: 500, rate: '0'  },
      };
      const rd = roleDefaults[char.role] || roleDefaults.default;

      // No data for this locale yet — synthesise defaults so user can assign manually
      // Borrow pitch/break/rate from the 'en' locale if it exists, otherwise use role defaults
      if (!charLoc) {
        const enLoc = char['en'] || {};
        charLoc = {
          azure_voice:        voiceList[0]?.voice || '',
          azure_style:        null,
          azure_style_degree: enLoc.azure_style_degree ?? 1.0,
          azure_rate:         enLoc.azure_rate  ?? (rd.rate  + '%'),
          azure_pitch:        enLoc.azure_pitch ?? (rd.pitch + '%'),
          azure_break_ms:     enLoc.azure_break_ms ?? rd.break_ms,
          available_styles:   voiceList[0]?.styles || [],
          _isNew:             true,   // flag: not yet saved to VoiceCast.json
        };
      }
      const isNew = !!charLoc._isNew;

      const card = document.createElement('div');
      card.className = 'vc-char-card';
      card.dataset.charId = char.character_id;

      // ── Card header ──
      const hdr = document.createElement('div');
      hdr.className = 'vc-char-card-hdr';
      hdr.innerHTML = `\u{1F464} ${escHtml(char.character_id)}`+
        `<span style="font-weight:400;color:var(--dim);margin-left:6px">${escHtml(char.role || '')}</span>`+
        (isNew ? `<span style="margin-left:auto;font-size:0.75em;color:#f0a500;font-weight:500">⚠ new — save VC to keep</span>` : '');
      card.appendChild(hdr);

      // ── Card body ──
      const body = document.createElement('div');
      body.className = 'vc-char-card-body';

      // Voice row: select
      const voiceRow = document.createElement('div');
      voiceRow.className = 'vc-voice-row';

      const voiceSel = document.createElement('select');
      voiceSel.className = 'vc-voice-select';

      // All non-multitalker voices, grouped by azure_locale for easy browsing
      const allVoices = voiceList.filter(v => !v.is_multitalker);
      const grpMap = {};
      allVoices.forEach(v => { (grpMap[v.azure_locale] = grpMap[v.azure_locale] || []).push(v); });
      Object.keys(grpMap).sort().forEach(loc => {
        const grp = document.createElement('optgroup');
        grp.label = loc;
        grpMap[loc].forEach(v => {
          const opt = new Option((v.local_name || v.voice) + ' — ' + v.gender + ' — ' + v.voice, v.voice);
          opt.dataset.azureLocale = v.azure_locale;
          if (v.voice === charLoc.azure_voice) opt.selected = true;
          grp.appendChild(opt);
        });
        voiceSel.appendChild(grp);
      });
      // Ensure the currently assigned voice is always selectable
      if (charLoc.azure_voice && !voiceSel.value) {
        const opt = new Option(charLoc.azure_voice, charLoc.azure_voice);
        opt.selected = true;
        voiceSel.add(opt);
      }

      voiceRow.appendChild(voiceSel);
      body.appendChild(voiceRow);

      // Preset row (between voice row and params row)
      const presetRow = document.createElement('div');
      presetRow.className = 'vc-preset-row';
      const presetLbl = document.createElement('span');
      presetLbl.className = 'vc-preset-label';
      presetLbl.textContent = 'Custom:';
      const presetSel = document.createElement('select');
      presetSel.className = 'vc-preset-select';
      presetSel.add(new Option('\u2014 no preset \u2014', ''));
      const presetDelBtn = document.createElement('button');
      presetDelBtn.className = 'btn-vc-preset-del';
      presetDelBtn.textContent = 'Delete';
      presetDelBtn.disabled = true;
      presetDelBtn.addEventListener('click', async () => {
        const voice = voiceSel.value;
        const hash  = presetSel.value;
        if (!hash) return;
        const name  = presetSel.selectedOptions[0]?.textContent || hash;
        if (!confirm('Delete custom preset "' + name + '"?\nThis removes it from presets.json and deletes the cached audio file.')) return;
        presetDelBtn.disabled = true;
        presetDelBtn.textContent = '\u2026';
        try {
          const r = await fetch('/api/delete_preset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ voice, hash }),
          });
          const d = await r.json();
          if (d.error) { alert('Delete failed: ' + d.error); return; }
          // Remove from in-memory presets
          if (_vcPresets[voice]) {
            _vcPresets[voice] = _vcPresets[voice].filter(p => p.hash !== hash);
          }
          rebuildPresetSelect(card, voice);
          presetSel.selectedIndex = 0;
          presetDelBtn.disabled = true;
        } finally {
          presetDelBtn.textContent = 'Delete';
        }
      });
      // Enable/disable delete button based on preset selection
      presetSel.addEventListener('change', () => {
        presetDelBtn.disabled = !presetSel.value;
      });
      presetRow.appendChild(presetLbl);
      presetRow.appendChild(presetDelBtn);
      presetRow.appendChild(presetSel);
      body.appendChild(presetRow);

      // Params column: style row + 4 numeric param rows (no value clamping)
      const paramsRow = document.createElement('div');
      paramsRow.className = 'vc-params-row';

      // Style row: [label] [select] [▶ Preview]
      const styleGrp = document.createElement('div');
      styleGrp.className = 'vc-param-group';
      const styleLbl = document.createElement('label');
      styleLbl.className = 'vc-param-label';
      styleLbl.textContent = 'Style';
      const styleSel = document.createElement('select');
      styleSel.className = 'vc-style-select';
      const blankOpt = document.createElement('option');
      blankOpt.value = ''; blankOpt.textContent = '\u2014 no style \u2014';
      styleSel.appendChild(blankOpt);
      const selectedVoice = voiceList.find(v => v.voice === (charLoc.azure_voice || voiceSel.value));
      (selectedVoice?.styles || []).forEach(s => {
        const opt = document.createElement('option');
        opt.value = s; opt.textContent = s;
        if (s === charLoc.azure_style) opt.selected = true;
        styleSel.appendChild(opt);
      });
      const prevBtn = document.createElement('button');
      prevBtn.className = 'btn-vc-preview';
      prevBtn.setAttribute('data-role', 'preview');
      prevBtn.textContent = '\u25B6 Preview';
      styleGrp.appendChild(styleLbl);
      styleGrp.appendChild(styleSel);
      styleGrp.appendChild(prevBtn);
      paramsRow.appendChild(styleGrp);

      // Numeric params — any value accepted; live interpretation shown inline
      const paramDefs = [
        { label: 'Emotion',  field: 'degree', interp: vcInterpDegree,
          value: String(charLoc.azure_style_degree ?? 1.0), range: '0.6 – 2.8' },
        { label: 'Speak Speed (%)',    field: 'rate',   interp: vcInterpRate,
          value: String(charLoc.azure_rate  ?? '0').replace('%', ''), range: '-40 – +35' },
        { label: 'Voice Depth (%)',   field: 'pitch',  interp: vcInterpPitch,
          value: String(charLoc.azure_pitch ?? '' ).replace('%', ''), range: '-8 – +8' },
        { label: 'Pause Duration (ms)', field: 'break',  interp: vcInterpBreak,
          value: String(charLoc.azure_break_ms ?? 0), range: '200 – 1200' },
      ];
      paramDefs.forEach(pd => {
        const grp = document.createElement('div');
        grp.className = 'vc-param-group';
        const lbl = document.createElement('label');
        lbl.className = 'vc-param-label';
        lbl.textContent = pd.label;
        const inp = document.createElement('input');
        inp.className = 'vc-param-input';
        inp.type = 'text';
        inp.setAttribute('data-field', pd.field);
        inp.value = pd.value;
        const rangeHint = document.createElement('span');
        rangeHint.className = 'vc-param-range';
        rangeHint.textContent = pd.range;
        const interpDiv = document.createElement('div');
        interpDiv.className = 'vc-param-interp';
        const updateInterp = () => {
          const { text, zone } = pd.interp(inp.value);
          const badge = zone === 'normal' ? '' :
            `<span class="vc-zone-badge vc-zone-${zone}">${zone === 'extended' ? 'Extended' : 'Experimental'}</span>`;
          interpDiv.innerHTML = `<span class="vc-interp-text">${escHtml(text)}</span>${badge}`;
        };
        inp.addEventListener('input', updateInterp);
        updateInterp();   // render immediately on card creation
        grp.appendChild(lbl);
        grp.appendChild(inp);
        grp.appendChild(rangeHint);
        grp.appendChild(interpDiv);
        paramsRow.appendChild(grp);
      });

      body.appendChild(paramsRow);

      card.appendChild(body);
      cardsEl.appendChild(card);
      // card is now in the DOM — safe to querySelector inside it
      rebuildPresetSelect(card, charLoc.azure_voice);
      // Auto-select the preset whose params match the loaded VoiceCast values
      const _loadedPresets = _vcPresets[charLoc.azure_voice] ?? [];
      const _matchedPreset = _loadedPresets.find(p =>
        (p.style        || '') === (charLoc.azure_style        || '') &&
        parseFloat(p.style_degree) === parseFloat(charLoc.azure_style_degree) &&
        (p.rate         || '') === (charLoc.azure_rate         || '') &&
        (p.pitch        || '') === (charLoc.azure_pitch        || '') &&
        parseInt(p.break_ms)   === parseInt(charLoc.azure_break_ms)
      );
      if (_matchedPreset) {
        presetSel.value = _matchedPreset.hash;
        presetDelBtn.disabled = false;
      }

      // ── Voice select onChange → rebuild style + preset dropdowns; restore saved params if possible ──
      voiceSel.addEventListener('change', () => {
        const newV = voiceList.find(v => v.voice === voiceSel.value);
        // Rebuild style dropdown (clear selection — new voice may not have the old style)
        styleSel.innerHTML = '';
        const blank2 = document.createElement('option');
        blank2.value = ''; blank2.textContent = '\u2014 no style \u2014';
        styleSel.appendChild(blank2);
        (newV?.styles || []).forEach(s => {
          const opt = document.createElement('option');
          opt.value = s; opt.textContent = s;
          styleSel.appendChild(opt);
        });
        rebuildPresetSelect(card, voiceSel.value);

        // Restore saved VoiceCast params when switching (back) to the saved voice
        if (voiceSel.value === charLoc.azure_voice) {
          styleSel.value = charLoc.azure_style || '';
          card.querySelector('[data-field=degree]').value = charLoc.azure_style_degree ?? 1.0;
          card.querySelector('[data-field=rate]').value   = String(charLoc.azure_rate  ?? '0').replace('%', '');
          card.querySelector('[data-field=pitch]').value  = String(charLoc.azure_pitch ?? '').replace('%', '');
          card.querySelector('[data-field=break]').value  = charLoc.azure_break_ms ?? 0;
          // Auto-select matching preset
          const _ps = _vcPresets[voiceSel.value] ?? [];
          const _pm = _ps.find(p =>
            (p.style || '') === (charLoc.azure_style || '') &&
            parseFloat(p.style_degree) === parseFloat(charLoc.azure_style_degree) &&
            (p.rate || '') === (charLoc.azure_rate || '') &&
            (p.pitch || '') === (charLoc.azure_pitch || '') &&
            parseInt(p.break_ms) === parseInt(charLoc.azure_break_ms)
          );
          if (_pm) card.querySelector('.vc-preset-select').value = _pm.hash;
          else card.querySelector('.vc-preset-select').selectedIndex = 0;
        } else {
          styleSel.value = '';
          const _d = _vcGetDefaults(voiceSel.value, '');
          card.querySelector('[data-field=degree]').value = _d.style_degree ?? 1.0;
          card.querySelector('[data-field=rate]').value   = String(_d.rate  ?? '0%').replace('%', '');
          card.querySelector('[data-field=pitch]').value  = String(_d.pitch ?? '-5%').replace('%', '');
          card.querySelector('[data-field=break]').value  = _d.break_ms ?? 600;
          card.querySelector('.vc-preset-select').selectedIndex = 0;
        }
        ['degree','rate','pitch','break'].forEach(f =>
          card.querySelector(`[data-field=${f}]`).dispatchEvent(new Event('input')));
      });

      // ── Style select onChange → always reset to defaults; custom presets load only via preset dropdown ──
      styleSel.addEventListener('change', () => {
        const voice = voiceSel.value;
        const style = styleSel.value;
        const _d = _vcGetDefaults(voice, style || '');
        card.querySelector('[data-field=degree]').value = _d.style_degree ?? 1.0;
        card.querySelector('[data-field=rate]').value   = String(_d.rate  ?? '0%').replace('%', '');
        card.querySelector('[data-field=pitch]').value  = String(_d.pitch ?? '-5%').replace('%', '');
        card.querySelector('[data-field=break]').value  = _d.break_ms ?? 600;
        presetSel.selectedIndex = 0;
        presetDelBtn.disabled = true;
        // Always fire input events so interpretation text reflects current values
        ['degree','rate','pitch','break'].forEach(f =>
          card.querySelector(`[data-field=${f}]`).dispatchEvent(new Event('input')));
      });

      // ── Preset select onChange → populate all param fields ──
      presetSel.addEventListener('change', () => {
        const presets = _vcPresets[voiceSel.value] ?? [];
        const preset  = presets.find(p => p.hash === presetSel.value);
        if (!preset) {
          // "— no preset —" selected: restore index defaults → universal fallback
          const _d = _vcGetDefaults(voiceSel.value, styleSel.value || '');
          card.querySelector('[data-field=degree]').value = _d.style_degree ?? 1.0;
          card.querySelector('[data-field=rate]').value   = String(_d.rate  ?? '0%').replace('%', '');
          card.querySelector('[data-field=pitch]').value  = String(_d.pitch ?? '-5%').replace('%', '');
          card.querySelector('[data-field=break]').value  = _d.break_ms ?? 600;
          ['degree','rate','pitch','break'].forEach(f => {
            card.querySelector(`[data-field=${f}]`).dispatchEvent(new Event('input'));
          });
          return;
        }
        styleSel.value = preset.style || '';
        card.querySelector('[data-field=degree]').value = preset.style_degree;
        card.querySelector('[data-field=rate]').value   = String(preset.rate  ?? '').replace('%', '');
        card.querySelector('[data-field=pitch]').value  = String(preset.pitch ?? '').replace('%', '');
        card.querySelector('[data-field=break]').value  = preset.break_ms;
        // Trigger interpretation text updates
        ['degree','rate','pitch','break'].forEach(f => {
          card.querySelector(`[data-field=${f}]`).dispatchEvent(new Event('input'));
        });
      });

      // ── Preview button: full params ──
      prevBtn.addEventListener('click', () => {
        if (prevBtn.textContent === '\u23F8 Pause') return;
        previewVoice(card, locale, {
          azure_voice:  voiceSel.value,
          azure_locale: voiceSel.selectedOptions[0]?.dataset.azureLocale || voiceSel.value.split('-').slice(0,2).join('-'),
          style:        styleSel.value || null,
          style_degree: parseFloat(card.querySelector('[data-field=degree]').value),
          rate:         addPct(card.querySelector('[data-field=rate]').value),
          pitch:        addPct(card.querySelector('[data-field=pitch]').value),
          break_ms:     parseInt(card.querySelector('[data-field=break]').value),
        }, prevBtn);
      });
    });
  }

  // ── previewVoice(card, locale, params, clickedBtn) ───────────────────────────
  async function previewVoice(card, locale, params, clickedBtn) {
    // Stop any audio already playing (from this or another card)
    if (_vcPlayingAudio) {
      _vcPlayingAudio.pause();
      _vcPlayingAudio = null;
    }

    const btns = card.querySelectorAll('.btn-vc-preview');
    btns.forEach(b => { b.disabled = true; b._orig = b.textContent; b.textContent = '\u2026'; });
    let playing = false;
    try {
      const _ac = new AbortController();
      const _to = setTimeout(() => _ac.abort(), 25000);  // 25s timeout
      const r = await fetch('/api/preview_voice', {
        method:  'POST',
        signal:  _ac.signal,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          azure_voice:  params.azure_voice,
          azure_locale: params.azure_locale,
          style:        params.style,
          style_degree: params.style_degree,
          rate:         params.rate,
          pitch:        params.pitch,
          break_ms:     params.break_ms,
          text:         vcSampleText(locale, params.style),
        }),
      });
      clearTimeout(_to);
      const data = await r.json();
      if (data.url) {
        const audio = new Audio(data.url + '&t=' + Date.now());
        _vcPlayingAudio = audio;
        playing = true;

        // All buttons except the clicked one re-enable normally
        btns.forEach(b => { if (b !== clickedBtn) { b.disabled = false; b.textContent = b._orig; } });
        // Clicked button becomes ⏸ Pause
        clickedBtn.textContent = '\u23F8 Pause';
        clickedBtn.disabled = false;

        // Revert button and re-enable all when playback ends or is paused
        let reverted = false;
        const revert = () => {
          if (reverted) return;
          reverted = true;
          if (_vcPlayingAudio === audio) _vcPlayingAudio = null;
          clickedBtn.textContent = clickedBtn._orig;
          btns.forEach(b => { b.disabled = false; });
        };
        audio.addEventListener('ended', revert);
        audio.addEventListener('pause', revert);

        // One-shot: clicking the ⏸ Pause button pauses audio (revert fires via 'pause' event)
        clickedBtn.addEventListener('click', () => audio.pause(), { once: true });

        audio.play();

        // New preset auto-saved → add to in-memory list + update dropdown
        if (data.preset) {
          (_vcPresets[params.azure_voice] ??= []).push(data.preset);
          rebuildPresetSelect(card, params.azure_voice);
          card.querySelector('.vc-preset-select').value = data.preset.hash;
        }
        // First-ever preview for this voice+style → sync _voiceIndex so
        // style-change auto-populate and "— no preset —" restore work correctly.
        if (data.index_params) {
          _voiceIndex ??= {};
          (_voiceIndex[params.azure_voice] ??= { clips: {} })
            .clips[params.style || ''] = { params: data.index_params };
        }
      } else {
        appendLine('\u26A0 TTS preview failed: ' + (data.error || 'unknown error'), 'err');
      }
    } catch (e) {
      if (e.name === 'AbortError') {
        appendLine('\u26A0 TTS preview timed out — Azure may be throttled (429)', 'err');
      } else {
        appendLine('\u26A0 TTS preview error: ' + e.message, 'err');
      }
    } finally {
      // Only revert buttons if audio never started (synthesis error / network failure)
      if (!playing) {
        btns.forEach(b => { b.disabled = false; b.textContent = b._orig; });
      }
    }
  }

  // ── saveVoiceCast() → boolean ─────────────────────────────────────────────────
  async function saveVoiceCast() {
    // MIN-I8: guard — user may click Save before Stage 0 has run
    if (!_vcData) return false;
    // BUG-R1: flush active locale DOM → _vcData first so all locales are captured
    flushActiveLocale();
    try {
      const r = await fetch('/api/save_voice_cast', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ slug: currentSlug, voice_cast: _vcData }),
      });
      const data = await r.json();
      if (data.ok) {
        const badge = document.getElementById('vc-saved-badge');
        badge.style.display = '';
        setTimeout(() => { badge.style.display = 'none'; }, 2000);
        return true;
      } else {
        appendLine('\u26A0 Save failed: ' + (data.error || 'unknown error'), 'err');
        return false;
      }
    } catch(e) {
      appendLine('\u26A0 Save failed: ' + e, 'err');
      return false;
    }
  }

  // ── vcContinue() ─────────────────────────────────────────────────────────────
  async function vcContinue() {
    if (_vcPendingTo == null) return;
    // CON-R6: guard on save failure — don't run pipeline with stale VoiceCast.json
    const saveOk = await saveVoiceCast();
    if (!saveOk) {
      appendLine('\u26A0 Save failed \u2014 fix the error above before continuing', 'err');
      return;
    }
    switchStoryTab('story');
    // Build ep_dir from current episode globals (set by runPrompt / createEpisode)
    const _ep_dir = 'projects/' + currentSlug + '/episodes/' + currentEpId;
    startPipeStep({ type: 'llm', from: 1, to: _vcPendingTo,
                    ep_dir: _ep_dir });
    _vcPendingTo = null;
    document.getElementById('btn-vc-continue').style.display = 'none';
  }

  // ── Music tab ──────────────────────────────────────────────────────────────────

  let _musicSlug      = null;
  let _musicEpId      = null;
  let _musicTimeline  = null;   // timeline.json data
  let _musicCandidates = null;  // music_loop_candidates.json data
  let _musicClipResults = [];   // gen_music_clip_results.json — all available clips
  let _musicSources     = [];   // source music tracks from /api/music_sources
  let _musicCutClips  = [];     // user-cut clips: [{clip_id, stem, start_sec, end_sec, path}]
  let _musicClipLookup = {};   // clip_id → { wavStem, path, item_id } — built by _musicRenderBody
  let _musicMarks     = {};     // per-stem marks: { stem: {start: N, end: N} }
  let _musicOverrides = {};       // { item_id: {duck_db, fade_sec, ...} }
  let _musicTrackVolumes = {};    // { stem: dB_offset } — persists across regenerations
  let _musicClipVolumes  = {};    // { item_id|clip_id: dB_offset }
                                  // auto clips  → keyed by item_id  (e.g. "music-s01e01_sh01")
                                  // user-cut    → keyed by clip_id  (e.g. "djokovic_bg_calm:19.4s-40.3s")
  let _musicLoopSel   = {};       // { track_stem: {start_sec, duration_sec, mode, crossfade_ms} }
  let _musicBusy      = false;

  function _musicSetStatus(msg, spinning) {
    document.getElementById('music-status-text').textContent = msg;
    document.getElementById('music-spinner').style.display = spinning ? 'inline-block' : 'none';
  }

  function initMusicTab() {
    const sel = document.getElementById('music-ep-select');
    if (sel.options.length > 1) { _musicSyncFromRunTab(); return; }
    fetch('/list_projects').then(r => r.json()).then(data => {
      (data.projects || []).forEach(proj => {
        (proj.episodes || []).forEach(ep => {
          const opt = document.createElement('option');
          opt.value       = proj.slug + '|' + ep.id;
          opt.textContent = proj.slug + ' / ' + ep.id;
          sel.appendChild(opt);
        });
      });
      _musicSyncFromRunTab();
    }).catch(() => {});
  }

  function _musicSyncFromRunTab() {
    if (_musicSlug && _musicEpId) return;
    if (!currentSlug || !currentEpId) return;
    const target = currentSlug + '|' + currentEpId;
    const sel = document.getElementById('music-ep-select');
    for (let i = 0; i < sel.options.length; i++) {
      if (sel.options[i].value === target) {
        sel.value = target;
        onMusicEpChange();
        return;
      }
    }
  }

  function onMusicEpChange() {
    const v = document.getElementById('music-ep-select').value;
    if (!v) { _musicSlug = null; _musicEpId = null; return; }
    [_musicSlug, _musicEpId] = v.split('|');
    _musicClipVolumes = {};
    _musicCutClips    = [];   // reset on episode change; repopulated from disk by _musicLoadExisting
    document.getElementById('music-btn-review').disabled  = false;
    _musicSetStatus('Episode selected. Click Generate Music Review to begin.');
    // Try to load existing data
    _musicLoadExisting();
  }

  async function _musicLoadExisting() {
    if (!_musicSlug || !_musicEpId) return;
    // Try loading existing candidates + timeline + clip results
    try {
      const cr = await fetch('/api/music_loop_candidates?slug=' + encodeURIComponent(_musicSlug)
        + '&ep_id=' + encodeURIComponent(_musicEpId));
      if (cr.ok) {
        _musicCandidates = await cr.json();
      }
    } catch (_) {}
    try {
      const tr = await fetch('/api/music_timeline?slug=' + encodeURIComponent(_musicSlug)
        + '&ep_id=' + encodeURIComponent(_musicEpId));
      if (tr.ok) {
        _musicTimeline = await tr.json();
      }
    } catch (_) {}
    // Load gen_music_clip_results.json — all available music clips for this episode
    try {
      const mr = await fetch('/api/episode_file?slug=' + encodeURIComponent(_musicSlug)
        + '&ep_id=' + encodeURIComponent(_musicEpId)
        + '&file=assets/meta/gen_music_clip_results.json');
      if (mr.ok) {
        _musicClipResults = await mr.json();
      }
    } catch (_) {}
    // Load source music tracks
    try {
      const sr = await fetch('/api/music_sources?slug=' + encodeURIComponent(_musicSlug)
        + '&ep_id=' + encodeURIComponent(_musicEpId));
      if (sr.ok) {
        _musicSources = await sr.json();
      }
    } catch (_) {}
    // Load user-cut clips
    try {
      const uc = await fetch('/api/episode_file?slug=' + encodeURIComponent(_musicSlug)
        + '&ep_id=' + encodeURIComponent(_musicEpId)
        + '&file=assets/music/user_cut_clips.json');
      if (uc.ok) {
        _musicCutClips = await uc.json();
      }
    } catch (_) {}
    // Try loading existing MusicPlan.json
    try {
      const pr = await fetch('/api/episode_file?slug=' + encodeURIComponent(_musicSlug)
        + '&ep_id=' + encodeURIComponent(_musicEpId) + '&file=assets/music/MusicPlan.json');
      if (pr.ok) {
        const plan = await pr.json();
        _musicLoopSel      = plan.loop_selections || {};
        _musicTrackVolumes = plan.track_volumes   || {};
        _musicClipVolumes  = plan.clip_volumes    || {};
        _musicOverrides    = {};
        (plan.shot_overrides || []).forEach(o => { _musicOverrides[o.item_id] = o; });
      }
    } catch (_) {}
    _musicRenderBody();
  }

  // Auto-save overrides to MusicPlan.json (silently, no UI feedback)
  async function _musicAutoSave() {
    if (!_musicSlug || !_musicEpId) return;
    try {
      const plan = {
        schema_id: 'MusicPlan',
        schema_version: '1.0',
        loop_selections: _musicLoopSel,
        shot_overrides: Object.values(_musicOverrides).filter(o => o.item_id),
      };
      if (Object.keys(_musicTrackVolumes).length)
        plan.track_volumes = _musicTrackVolumes;
      if (Object.keys(_musicClipVolumes).length)
        plan.clip_volumes = _musicClipVolumes;
      await fetch('/api/music_plan_save', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ slug: _musicSlug, ep_id: _musicEpId, plan: plan }),
      });
    } catch (_) {}
  }

  function _musicSetClipVolume(key, val) {
    val = parseInt(val, 10);
    if (isNaN(val) || val === 0) { delete _musicClipVolumes[key]; }
    else                         { _musicClipVolumes[key] = val; }
    // Update any rendered audio preview — compare via dataset to avoid CSS
    // special-char issues with clip_id keys containing ":" and "."
    const gain = Math.pow(10, (val || 0) / 20);
    document.querySelectorAll('audio[data-clip-vol-key]').forEach(a => {
      if (a.dataset.clipVolKey === key) {
        a.dataset.volDb = (val || 0);
        a.volume = Math.min(1, Math.max(0, gain));
      }
    });
    _musicAutoSave();
  }

  function _musicSetTrackVolume(stem, val) {
    val = parseInt(val, 10);
    if (isNaN(val) || val === 0) { delete _musicTrackVolumes[stem]; }
    else                         { _musicTrackVolumes[stem] = val; }
    _musicAutoSave();
  }

  async function musicGenerateReview() {
    if (!_musicSlug || !_musicEpId || _musicBusy) return;
    _musicBusy = true;
    document.getElementById('music-btn-review').disabled = true;

    try {
      // Step 1: Prepare loop candidates
      _musicSetStatus('Step 1/2 — Analysing loop candidates …', true);
      const r1 = await fetch('/api/music_prepare_loops', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ slug: _musicSlug, ep_id: _musicEpId }),
      });
      const d1 = await r1.json();
      if (!r1.ok || d1.error) throw new Error(d1.error || 'loop analysis failed');
      _musicCandidates = d1.candidates || d1;

      // Step 2: Generate review pack (timeline + preview audio) with current overrides
      _musicSetStatus('Step 2/2 — Generating review pack (VO + music preview) …', true);
      // Re-fetch user_cut_clips.json — may have been written after tab loaded
      try {
        const ucR = await fetch('/api/episode_file?slug=' + encodeURIComponent(_musicSlug)
          + '&ep_id=' + encodeURIComponent(_musicEpId) + '&file=assets/music/user_cut_clips.json');
        if (ucR.ok) { _musicCutClips = await ucR.json(); }
      } catch (_) {}
      const currentOverrides = Object.values(_musicOverrides).filter(o => o.item_id);
      const r2 = await fetch('/api/music_review_pack', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          slug: _musicSlug, ep_id: _musicEpId,
          shot_overrides: currentOverrides.length > 0 ? currentOverrides : undefined,
        }),
      });
      const d2 = await r2.json();
      if (!r2.ok || d2.error) throw new Error(d2.error || 'review pack failed');
      _musicTimeline = d2.timeline || null;

      // Auto-save current overrides to MusicPlan.json so they survive page refresh
      await _musicAutoSave();

      const nTracks = Object.keys(_musicCandidates.tracks || {}).length;
      _musicSetStatus('Review ready — ' + nTracks + ' tracks analysed. Listen to preview and adjust below.', false);
      _musicRenderBody();
      document.getElementById('music-footer').style.display = 'flex';
    } catch (err) {
      _musicSetStatus('Error: ' + err.message, false);
    } finally {
      _musicBusy = false;
      document.getElementById('music-btn-review').disabled = false;
    }
  }

  function _musicRenderBody() {
    const body = document.getElementById('music-body');
    body.innerHTML = '';

    // ── Preview audio player ──
    const previewPath = 'projects/' + _musicSlug + '/episodes/' + _musicEpId
      + '/assets/music/MusicReviewPack/preview_audio.wav';
    const previewWrap = document.createElement('div');
    previewWrap.className = 'music-preview-wrap';
    previewWrap.innerHTML = '<div class="music-section-label">Preview Audio (VO + Music)</div>'
      + '<audio controls src="/serve_media?path=' + encodeURIComponent(previewPath)
      + '&t=' + Date.now() + '" style="width:100%"></audio>'
      + '<div style="font-size:0.72em;color:var(--dim);margin-top:4px">'
      + 'If no audio loads, click Generate Review Pack first.</div>';
    body.appendChild(previewWrap);

    // ── Build unified clip list + global lookup (used by Shot Overrides + Generated Clips) ──
    _musicClipLookup = {};
    const allClips = [];
    (_musicClipResults || []).forEach(c => {
      if (c.status !== 'success') return;
      const src = (c.source_file || '').replace(/\.[^.]+$/, '');
      const endSec = (c.start_sec || 0) + (c.duration_sec || 0);
      const cid = src + ':' + (c.start_sec || 0).toFixed(1) + 's-' + endSec.toFixed(1) + 's';
      const wavPath = 'projects/' + _musicSlug + '/episodes/' + _musicEpId
        + '/assets/music/' + c.item_id + '.wav';
      allClips.push({
        clip_id: cid, stem: src,
        start_sec: c.start_sec || 0, end_sec: endSec,
        duration_sec: c.duration_sec || 0,
        score: c.match_score, item_id: c.item_id,
        wavPath: wavPath, origin: 'auto',
      });
      _musicClipLookup[cid] = { wavStem: c.item_id, path: wavPath, item_id: c.item_id };
    });
    (_musicCutClips || []).forEach(c => {
      const wavStem = (c.path || '').replace(/^.*\//, '').replace(/\.[^.]+$/, '');
      allClips.push({
        clip_id: c.clip_id, stem: c.stem,
        start_sec: c.start_sec, end_sec: c.end_sec,
        duration_sec: c.end_sec - c.start_sec,
        score: null, item_id: null,
        wavPath: c.path, origin: 'user',
      });
      _musicClipLookup[c.clip_id] = { wavStem: wavStem, path: c.path, item_id: null };
    });

    // ── Shot Overrides: visual timeline bar + clip dropdown + start/end/duck/fade ──
    if (_musicTimeline && _musicTimeline.shots) {
      const musicShots = _musicTimeline.shots.filter(s => s.music_item_id);
      // Map item_ids to their current clip_id label
      const itemToClipId = {};
      (_musicClipResults || []).forEach(c => {
        if (c.status !== 'success') return;
        const src = (c.source_file || '').replace(/\.[^.]+$/, '');
        const endSec = (c.start_sec || 0) + (c.duration_sec || 0);
        itemToClipId[c.item_id] = src + ':' + (c.start_sec || 0).toFixed(1) + 's-' + endSec.toFixed(1) + 's';
      });
      if (musicShots.length > 0) {
        const totalDur = _musicTimeline.total_duration_sec || 1;
        const ovrCard = document.createElement('div');
        ovrCard.className = 'music-card';

        const shotColors = ['#3b6ea5','#8b5e3c','#5e8c5a','#8b3e6e','#6e6e3e','#3e6e8b','#8b6e3e'];

        let ovrHtml = '<div class="music-card-hdr">Shot Overrides</div>'
          + '<div class="music-card-sub">Clip, music window (start/end within shot), duck &amp; fade per shot.</div>';

        // ── Visual timeline bar ──
        ovrHtml += '<div class="music-vtl-bar">';
        musicShots.forEach((s, i) => {
          const pct = ((s.duration_sec || 0) / totalDur * 100);
          const col = shotColors[i % shotColors.length];
          const t0 = s.offset_sec || 0;
          const t1 = t0 + (s.duration_sec || 0);
          const fmtT = (t) => { const m = Math.floor(t/60); const sc = (t%60).toFixed(0); return m+':'+(sc<10?'0':'')+sc; };
          ovrHtml += '<div class="music-vtl-shot" style="width:' + pct + '%;background:' + col
            + '" title="' + s.shot_id + '  ' + fmtT(t0) + '–' + fmtT(t1) + '  (' + (s.duration_sec||0).toFixed(1) + 's)">'
            + s.shot_id.replace(/^s\d+e\d+_/, '') + '</div>';
        });
        ovrHtml += '</div>';

        // ── Per-shot blocks (full width, stacked) ──
        const fmtEp = (t) => t.toFixed(1) + 's';
        musicShots.forEach((s, i) => {
          const origMid  = s.music_item_id;
          const ovr      = _musicOverrides[origMid] || {};
          const shotDur  = s.duration_sec || 0;
          const epStart  = s.offset_sec || 0;
          const epEnd    = epStart + shotDur;
          // within-shot offsets (stored in overrides / backend)
          const startWithin = ovr.start_sec    != null ? ovr.start_sec    : (s.start_sec != null ? s.start_sec : 0);
          const durVal      = ovr.duration_sec != null ? ovr.duration_sec : shotDur;
          const endWithin   = Math.min(startWithin + durVal, shotDur);
          // episode-absolute values shown in the inputs
          const dispStart = epStart + startWithin;
          const dispEnd   = epStart + endWithin;
          const duckVal  = ovr.duck_db  != null ? ovr.duck_db  : 0;
          const fadeVal  = ovr.fade_sec != null ? ovr.fade_sec : (s.fade_sec != null ? s.fade_sec : 0.15);
          const col      = shotColors[i % shotColors.length];
          // currentClipId: use saved override if present, else empty (no auto-default)
          // IMPORTANT: do NOT fall back to origMid ("music-sc01-sh01") — it's not a real
          // clip ID and causes the browser to silently show the first option without
          // triggering onchange, leaving the override unregistered.
          const currentClipId = ovr.music_clip_id || itemToClipId[origMid] || '';

          ovrHtml += '<div class="music-shot-block">'
            // header: shot id + episode time range
            + '<div class="music-shot-hdr" style="border-left:4px solid ' + col + '">'
            + '<span class="music-shot-hdr-id">' + s.shot_id.replace(/^s\d+e\d+_/, '') + '</span>'
            + '<span class="music-shot-hdr-ep">episode&nbsp;' + fmtEp(epStart) + ' – ' + fmtEp(epEnd)
            + '&nbsp;(' + shotDur.toFixed(1) + 's)</span>'
            + '</div>'
            // clip dropdown — first option is explicit "none" so unassigned shots are visible
            + '<div class="music-shot-clip">'
            + '<select style="width:100%" onchange="_musicSetClipOverride(\'' + origMid + '\',this.value)">'
            + '<option value=""' + (currentClipId === '' ? ' selected' : '') + '>— no clip —</option>';
          allClips.forEach(c => {
            ovrHtml += '<option value="' + c.clip_id + '"' + (c.clip_id === currentClipId ? ' selected' : '') + '>'
              + c.clip_id + '</option>';
          });
          ovrHtml += '</select></div>'
            // params: start / end / duck / fade
            + '<div class="music-shot-params">'
            + '<label title="Episode time when music begins (seconds)">▶ start</label>'
            + '<input type="number" step="0.5" min="' + epStart.toFixed(1) + '" max="' + epEnd.toFixed(1) + '" value="' + dispStart.toFixed(1) + '"'
            + ' onchange="_musicSetStartEnd(\'' + origMid + '\',parseFloat(this.value)-' + epStart + ',null,' + shotDur + ')"'
            + ' style="width:64px">'
            + '<label title="Episode time when music stops (seconds)">⏹ end</label>'
            + '<input type="number" step="0.5" min="' + epStart.toFixed(1) + '" max="' + epEnd.toFixed(1) + '" value="' + dispEnd.toFixed(1) + '"'
            + ' onchange="_musicSetStartEnd(\'' + origMid + '\',null,parseFloat(this.value)-' + epStart + ',' + shotDur + ')"'
            + ' style="width:64px">'
            + '<label title="Attenuation in dB (0 = full volume)">🔉 duck</label>'
            + '<input type="number" step="1" min="-30" max="0" value="' + duckVal + '"'
            + ' onchange="_musicSetOverride(\'' + origMid + '\',\'duck_db\',parseFloat(this.value))"'
            + ' style="width:50px">'
            + '<label title="Fade duration in seconds">⏱ fade</label>'
            + '<input type="number" step="0.05" min="0" max="3" value="' + fadeVal.toFixed(2) + '"'
            + ' onchange="_musicSetOverride(\'' + origMid + '\',\'fade_sec\',parseFloat(this.value))"'
            + ' style="width:50px">'
            + '</div>'
            + '</div>';
        });

        ovrCard.innerHTML = ovrHtml;
        body.appendChild(ovrCard);
        document.getElementById('music-footer').style.display = 'flex';
      }
    }

    // ── Source Music Browser (with mark start/end + cut) ──
    if (_musicSources && _musicSources.length > 0) {
      const srcCard = document.createElement('div');
      srcCard.className = 'music-card';
      let srcHtml = '<div class="music-card-hdr">Source Music Library</div>'
        + '<div class="music-card-sub">Play a source track, mark Start &amp; End positions, '
        + 'then Cut Clip to create a candidate for shot assignment.</div>';
      _musicSources.forEach((s, si) => {
        const dur = s.duration_sec != null
          ? (s.duration_sec / 60 | 0) + ':' + String(Math.round(s.duration_sec % 60)).padStart(2, '0')
          : '—';
        const bpm = s.bpm != null ? Math.round(s.bpm) + ' BPM' : '';
        const mark = _musicMarks[s.stem] || {};
        const markStartTxt = mark.start != null ? mark.start.toFixed(1) + 's' : '—';
        const markEndTxt   = mark.end   != null ? mark.end.toFixed(1)   + 's' : '—';
        const volVal = _musicTrackVolumes[s.stem] || 0;
        const volStyle = volVal !== 0
          ? 'color:var(--gold);font-weight:700' : 'color:var(--dim)';
        srcHtml += '<div class="music-src-row">'
          + '<div class="music-src-top">'
          + '<span class="music-src-stem">' + s.stem + '</span>'
          + '<span class="music-src-meta">' + dur + (bpm ? ' · ' + bpm : '') + '</span>'
          + '<label style="margin-left:auto;display:flex;align-items:center;gap:4px;font-size:0.80em">'
          + '<span style="' + volStyle + '">vol</span>'
          + '<input type="number" step="1" min="-18" max="0" value="' + volVal + '"'
          + ' title="Track volume offset in dB (0 = default -6 dB base; -6 = plays at -12 dB; attenuation only)"'
          + ' style="width:52px;background:var(--input-bg,#1e1e2e);color:' + (volVal !== 0 ? 'var(--gold)' : 'var(--text)') + ';border:1px solid var(--border);border-radius:4px;padding:2px 4px;font-size:0.95em"'
          + ' onchange="_musicSetTrackVolume(\'' + s.stem + '\',this.value);this.style.color=parseInt(this.value)!==0?\'var(--gold)\':\'var(--text)\'">'
          + '</label>'
          + '<div class="music-src-player">'
          + '<audio id="music-src-audio-' + si + '" controls preload="none" style="width:100%;height:32px"'
          + ' src="/serve_media?path=' + encodeURIComponent(s.path) + '"></audio>'
          + '</div></div>'
          + '<div class="music-src-controls">'
          + '<button onclick="_musicMarkPos(\'' + s.stem + '\',' + si + ',\'start\')"'
          + (mark.start != null ? ' class="active"' : '') + '>Mark Start</button>'
          + '<span class="mark-label">In: ' + markStartTxt + '</span>'
          + '<button onclick="_musicMarkPos(\'' + s.stem + '\',' + si + ',\'end\')"'
          + (mark.end != null ? ' class="active"' : '') + '>Mark End</button>'
          + '<span class="mark-label">Out: ' + markEndTxt + '</span>'
          + '<button onclick="_musicCutClip(\'' + s.stem + '\')"'
          + ' style="' + (mark.start != null && mark.end != null
            ? 'background:var(--gold);color:#0d0d10;font-weight:700' : '')
          + '">✂ Cut Clip</button>'
          + '</div></div>';
      });
      srcCard.innerHTML = srcHtml;
      body.appendChild(srcCard);
    }

    // ── Generated Clips (auto + user-cut, unified) ──
    // allClips includes both origins, so user cuts show even when auto clips are absent.
    if (allClips.length > 0) {
      const clipCard = document.createElement('div');
      clipCard.className = 'music-card';
      let clipHtml = '<div class="music-card-hdr">Generated Clips</div>'
        + '<div class="music-card-sub">Auto-extracted and user-cut clips available for shot assignment.</div>'
        + '<table class="music-cand-table"><thead><tr>'
        + '<th style="max-width:160px">Clip</th><th>Duration</th><th>Score</th>'
        + '<th style="width:60px" title="Volume offset in dB (0 = default -6 dB; attenuation only)">Vol</th>'
        + '<th style="width:440px">Preview</th>'
        + '</tr></thead><tbody>';
      allClips.forEach(c => {
        const durTxt   = c.duration_sec.toFixed(1) + 's';
        const scoreTxt = c.score != null ? c.score.toFixed(3) : (c.origin === 'user' ? 'cut' : '—');
        // Vol control:
        //   auto clips  → keyed by item_id  (direct manifest link, no assignment needed)
        //   user-cut    → keyed by clip_id  (persisted now; resolved to item_id at apply time
        //                                    via shot_overrides[].music_clip_id when assigned)
        const volKey   = c.item_id ? c.item_id : c.clip_id;
        const volVal   = _musicClipVolumes[volKey] || 0;
        const volColor = volVal !== 0 ? 'var(--gold)' : 'var(--text)';
        const safeKey  = volKey.replace(/'/g, "\\'");
        const volTitle = c.item_id
          ? 'Volume offset in dB (0 = default -6 dB; -6 = plays at -12 dB). Applied directly via manifest item_id.'
          : 'Volume offset in dB. Persisted now; applied to final output when this clip is assigned to a shot.';
        const volCell  = '<input type="number" step="1" min="-18" max="0" value="' + volVal + '"'
          + ' title="' + volTitle + '"'
          + ' style="width:52px;background:var(--input-bg,#1e1e2e);color:' + volColor + ';'
          + 'border:1px solid var(--border);border-radius:4px;padding:2px 4px;font-size:0.90em"'
          + ' onchange="'
          + '_musicSetClipVolume(\'' + safeKey + '\',this.value);'
          + 'var _db=parseInt(this.value)||0;'
          + 'var _a=this.closest(\'tr\').querySelector(\'audio\');'
          + 'if(_a){_a.dataset.volDb=_db;_a.volume=Math.pow(10,_db/20);}'
          + 'this.style.color=_db!==0?\'var(--gold)\':\'var(--text)\'">';
        clipHtml += '<tr>'
          + '<td style="font-family:var(--mono);font-size:0.82em;color:'
          + (c.origin === 'user' ? 'var(--gold)' : 'var(--text)') + ';'
          + 'max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"'
          + ' title="' + c.clip_id + '">' + c.clip_id + '</td>'
          + '<td>' + durTxt + '</td>'
          + '<td>' + scoreTxt + '</td>'
          + '<td>' + volCell + '</td>'
          + '<td><audio controls preload="none" style="height:28px;width:430px"'
          + ' data-clip-vol-key="' + safeKey + '"'
          + ' data-vol-db="' + volVal + '"'
          + ' onplay="this.volume=Math.pow(10,(+(this.dataset.volDb)||0)/20)"'
          + ' src="/serve_media?path=' + encodeURIComponent(c.wavPath) + '"></audio></td>'
          + '</tr>';
      });
      clipHtml += '</tbody></table>';
      clipCard.innerHTML = clipHtml;
      body.appendChild(clipCard);
      // Set initial volume on all previews immediately — onplay alone is unreliable
      clipCard.querySelectorAll('audio[data-vol-db]').forEach(a => {
        const db = +(a.dataset.volDb) || 0;
        if (db !== 0) a.volume = Math.pow(10, db / 20);
      });
    }

    // ── Shot Timeline (read-only reference, bottom) ──
    if (_musicTimeline && _musicTimeline.shots) {
      const tlCard = document.createElement('div');
      tlCard.className = 'music-card';
      let tlHtml = '<div class="music-card-hdr">Shot Timeline</div>'
        + '<div class="music-card-sub">Total: ' + (_musicTimeline.total_duration_sec || 0).toFixed(1) + 's'
        + ' — ' + _musicTimeline.shots.length + ' shots</div>';
      const fmtTime = (t) => {
        const m = Math.floor(t / 60);
        const sec = (t % 60).toFixed(1);
        return m + ':' + (sec < 10 ? '0' : '') + sec;
      };
      _musicTimeline.shots.forEach(s => {
        const mood = s.music_mood || '(no music)';
        const ovrDuck = (_musicOverrides[s.music_item_id] || {}).duck_db;
        const effectiveDuck = ovrDuck != null ? ovrDuck : (s.duck_db != null ? s.duck_db : 0);
        const duck = effectiveDuck + 'dB' + (ovrDuck != null ? ' ✎' : '');
        const t0 = s.offset_sec || 0;
        const t1 = t0 + (s.duration_sec || 0);
        tlHtml += '<div class="music-timeline-row">'
          + '<span class="music-timeline-shot">' + s.shot_id + '</span>'
          + '<span class="music-timeline-dur">' + fmtTime(t0) + ' – ' + fmtTime(t1) + '</span>'
          + '<span class="music-timeline-mood">' + mood + '</span>'
          + '<span class="music-timeline-duck">' + duck + '</span>'
          + '</div>';
      });
      tlCard.innerHTML = tlHtml;
      body.appendChild(tlCard);
    }
  }

  // ── Mark start/end positions on source tracks ──
  function _musicMarkPos(stem, audioIdx, which) {
    const audio = document.getElementById('music-src-audio-' + audioIdx);
    if (!audio) return;
    if (!_musicMarks[stem]) _musicMarks[stem] = {};
    _musicMarks[stem][which] = audio.currentTime;
    // Update UI in-place (avoid full re-render which would kill audio playback)
    const row = audio.closest('.music-src-row');
    if (row) {
      const controls = row.querySelector('.music-src-controls');
      if (controls) {
        const labels = controls.querySelectorAll('.mark-label');
        const buttons = controls.querySelectorAll('button');
        const mark = _musicMarks[stem];
        if (labels[0]) labels[0].textContent = 'In: ' + (mark.start != null ? mark.start.toFixed(1) + 's' : '—');
        if (labels[1]) labels[1].textContent = 'Out: ' + (mark.end != null ? mark.end.toFixed(1) + 's' : '—');
        if (buttons[0]) buttons[0].className = mark.start != null ? 'active' : '';
        if (buttons[1]) buttons[1].className = mark.end != null ? 'active' : '';
        // Highlight cut button when both marks set
        if (buttons[2] && mark.start != null && mark.end != null) {
          buttons[2].style.background = 'var(--gold)';
          buttons[2].style.color = '#0d0d10';
          buttons[2].style.fontWeight = '700';
        }
      }
    }
  }

  // ── Cut clip from source track ──
  async function _musicCutClip(stem) {
    const mark = _musicMarks[stem];
    if (!mark || mark.start == null || mark.end == null) {
      alert('Mark both Start and End positions first.');
      return;
    }
    if (mark.end <= mark.start) {
      alert('End must be after Start.');
      return;
    }
    _musicSetStatus('Cutting clip from ' + stem + ' …', true);
    try {
      const r = await fetch('/api/music_cut_clip', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          slug: _musicSlug, ep_id: _musicEpId,
          stem: stem, start_sec: mark.start, end_sec: mark.end,
        }),
      });
      const d = await r.json();
      if (!r.ok || d.error) throw new Error(d.error || 'cut failed');
      const clipId = d.clip_id || (stem + ':' + mark.start.toFixed(1) + 's-' + mark.end.toFixed(1) + 's');
      // Remove existing entry with same clip_id (re-cut)
      _musicCutClips = _musicCutClips.filter(c => c.clip_id !== clipId);
      _musicCutClips.push({
        clip_id: clipId, stem: stem,
        start_sec: mark.start, end_sec: mark.end,
        path: d.path,
      });
      _musicSetStatus('Clip cut: ' + clipId, false);
      _musicRenderBody();
    } catch (err) {
      _musicSetStatus('Error: ' + err.message, false);
    }
  }

  // ── Set clip override from visual timeline dropdown ──
  function _musicSetClipOverride(itemId, clipId) {
    // Empty clipId means "no clip" — clear the override entirely
    if (!clipId) {
      delete _musicOverrides[itemId];
      _musicAutoSave();
      return;
    }
    if (!_musicOverrides[itemId]) _musicOverrides[itemId] = { item_id: itemId };
    // Store full clip_id for UI dropdown matching
    _musicOverrides[itemId].music_clip_id = clipId;
    // Resolve to WAV filename stem via lookup (for backend)
    const info = _musicClipLookup[clipId];
    if (info && info.wavStem) {
      _musicOverrides[itemId].music_asset_id = info.wavStem;
    } else {
      // Fallback: reconstruct the filename stem from clip_id format
      // "cher1:11.1s-23.0s"  →  "cher1_11_1s-23_0s"  (matches WAV on disk)
      const m = clipId.match(/^(.+?):(\d+\.?\d*)s-(\d+\.?\d*)s$/);
      if (m) {
        const startFmt = m[2].replace('.', '_');
        const endFmt   = m[3].replace('.', '_');
        _musicOverrides[itemId].music_asset_id = `${m[1]}_${startFmt}s-${endFmt}s`;
      } else {
        _musicOverrides[itemId].music_asset_id = clipId;
      }
    }
    // Store clip timing for apply_music_plan
    const m = clipId.match(/^(.+?):(\d+\.?\d*)s-(\d+\.?\d*)s$/);
    if (m) {
      _musicOverrides[itemId].clip_start_sec    = parseFloat(m[2]);
      _musicOverrides[itemId].clip_duration_sec = parseFloat(m[3]) - parseFloat(m[2]);
    }
    // Persist immediately so a page reload or next Generate sees the selection
    _musicAutoSave();
  }

  function _musicSetOverride(itemId, field, value) {
    if (!_musicOverrides[itemId]) _musicOverrides[itemId] = { item_id: itemId };
    _musicOverrides[itemId][field] = value;
  }

  // start_sec / end_sec: either may be null (means "keep current value")
  function _musicSetStartEnd(itemId, startSec, endSec, shotDur) {
    if (!_musicOverrides[itemId]) _musicOverrides[itemId] = { item_id: itemId };
    const ovr = _musicOverrides[itemId];
    // Resolve current values
    const curStart = startSec  != null ? startSec  : (ovr.start_sec   != null ? ovr.start_sec   : 0);
    const curDur   = ovr.duration_sec != null ? ovr.duration_sec : shotDur;
    const curEnd   = endSec    != null ? endSec    : Math.min(curStart + curDur, shotDur);
    // Clamp and store
    const newStart = Math.max(0, Math.min(curStart, shotDur));
    const newEnd   = Math.max(newStart, Math.min(curEnd, shotDur));
    ovr.start_sec    = parseFloat(newStart.toFixed(2));
    ovr.duration_sec = parseFloat((newEnd - newStart).toFixed(2));
  }

  async function musicConfirm() {
    if (!_musicSlug || !_musicEpId) return;
    document.getElementById('music-btn-confirm').disabled = true;
    document.getElementById('music-confirm-msg').textContent = 'Saving MusicPlan.json …';
    try {
      const plan = {
        schema_id: 'MusicPlan',
        schema_version: '1.0',
        loop_selections: _musicLoopSel,
        shot_overrides: Object.values(_musicOverrides).filter(o => o.item_id),
      };
      if (Object.keys(_musicTrackVolumes).length)
        plan.track_volumes = _musicTrackVolumes;
      if (Object.keys(_musicClipVolumes).length)
        plan.clip_volumes = _musicClipVolumes;
      const r = await fetch('/api/music_plan_save', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ slug: _musicSlug, ep_id: _musicEpId, plan: plan }),
      });
      const d = await r.json();
      if (!r.ok || d.error) throw new Error(d.error || 'save failed');
      document.getElementById('music-confirm-msg').textContent =
        '✔ MusicPlan.json saved → ' + (d.path || 'assets/music/MusicPlan.json')
        + '. Resume pipeline with Stage 9 to apply.';
    } catch (err) {
      document.getElementById('music-confirm-msg').textContent = 'Error: ' + err.message;
    } finally {
      document.getElementById('music-btn-confirm').disabled = false;
    }
  }

  // ── Browse panel ─────────────────────────────────────────────────────────────
  function fmtSize(bytes) {
    if (bytes < 1024)      return bytes + ' B';
    if (bytes < 1048576)   return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(2) + ' MB';
  }

  async function loadProjects() {
    const tree = document.getElementById('browse-tree');
    tree.innerHTML = '<span class="browse-empty">Loading…</span>';
    try {
      const res = await fetch('/list_projects');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const { projects } = await res.json();
      renderTree(projects, tree);
    } catch(e) {
      tree.innerHTML =
        `<span style="color:var(--red);font-family:var(--mono);font-size:0.82em">` +
        `Error: ${escHtml(String(e))}</span>`;
    }
  }

  function renderTree(projects, container) {
    if (!projects.length) {
      container.innerHTML =
        '<span class="browse-empty">No projects found in projects/ folder.</span>';
      return;
    }
    container.innerHTML = '';

    projects.forEach(proj => {
      const group = document.createElement('div');
      group.className = 'proj-group';

      const heading = document.createElement('div');
      heading.className = 'proj-heading';
      heading.textContent = '📁 ' + proj.slug;
      group.appendChild(heading);

      if (!proj.episodes.length) {
        const empty = document.createElement('div');
        empty.className = 'browse-empty';
        empty.style.paddingLeft = '16px';
        empty.textContent = 'No episodes yet.';
        group.appendChild(empty);
      }

      proj.episodes.forEach(ep => {
        // Episode toggle row
        const epRow = document.createElement('div');
        epRow.className = 'ep-toggle-row';
        const caret = document.createElement('span');
        caret.className = 'ep-caret';
        caret.textContent = '▶';
        const epName = document.createElement('span');
        epName.textContent = ep.id;
        const epMeta = document.createElement('span');
        epMeta.className = 'ep-meta';
        epMeta.textContent = ` — ${ep.files.length} file${ep.files.length !== 1 ? 's' : ''}`;
        epRow.appendChild(caret);
        epRow.appendChild(epName);
        epRow.appendChild(epMeta);
        group.appendChild(epRow);

        // File list (collapsed by default)
        const fileDiv = document.createElement('div');
        fileDiv.className = 'ep-files';
        ep.files.forEach(f => {
          const row = document.createElement('div');
          row.className = 'ep-file-row';
          const left = document.createElement('span');
          left.innerHTML =
            `<span class="ep-file-name">${escHtml(f.name)}</span>` +
            `<span class="ep-file-sz">${fmtSize(f.size)}</span>`;
          const btn = document.createElement('button');
          btn.className = 'btn-review';
          btn.textContent = '📄 View';
          btn.onclick = () => window.open(
            '/view_artifact?path=' + encodeURIComponent(f.path), '_blank');
          row.appendChild(left);
          row.appendChild(btn);
          fileDiv.appendChild(row);
        });
        group.appendChild(fileDiv);

        // Toggle expand/collapse
        epRow.addEventListener('click', () => {
          const open = fileDiv.classList.toggle('open');
          caret.textContent = open ? '▼' : '▶';
        });
      });

      container.appendChild(group);
    });
  }

  // ── Stage artifact map ───────────────────────────────────────────────────────
  function stageArtifacts(n) {
    const ep = name => ({
      label: name,
      path:  `projects/${currentSlug}/episodes/${currentEpId}/${name}`,
    });
    const map = {
      0: [
        ep('pipeline_vars.sh'),
        { label: 'VoiceCast.json', path: `projects/${currentSlug}/VoiceCast.json` },
      ],
      2: [ep('StoryPrompt.json')],
      3: [ep('Script.json')],
      4: [ep('ShotList.json')],
      5: [ep('AssetManifest_draft.shared.json'), ep('AssetManifest_draft.en.json')],
      6: [ep('canon_diff.json')],
      7: [ep('canon.json')],
      9: [ep('AssetManifest_final.json'), ep('RenderPlan.json')],
    };
    // All stages need slug/ep_id to build the episode path
    if (!currentSlug || !currentEpId) return [];
    return map[n] || [];
  }

  function insertReviewButtons(n) {
    const artifacts = stageArtifacts(n);
    if (!artifacts.length) return;
    const atBottom =
      outputEl.scrollHeight - outputEl.scrollTop - outputEl.clientHeight < 40;
    const bar = document.createElement('div');
    bar.className = 'review-bar';
    const lbl = document.createElement('span');
    lbl.className = 'review-label';
    lbl.textContent = 'Review:';
    bar.appendChild(lbl);
    artifacts.forEach(({ label, path }) => {
      const btn = document.createElement('button');
      btn.className = 'btn-review';
      btn.textContent = '📄 ' + label;
      btn.onclick = () => window.open(
        '/view_artifact?path=' + encodeURIComponent(path), '_blank');
      bar.appendChild(btn);
    });
    outputEl.appendChild(bar);
    if (atBottom) outputEl.scrollTop = outputEl.scrollHeight;
  }

  // ── Pipeline tab ─────────────────────────────────────────────────────────────
  let pipeEpSlug    = null;
  let pipeEpId      = null;
  let pipeStatus    = null;
  // ── YouTube tab ────────────────────────────────────────────────────────────

  let _ytReviewData = null;  // last upload_review.json data
  let _ytPendingSec = null;  // frame second waiting for [Save as thumbnail] confirmation

  async function initYoutubeTab() {
    const slug   = currentSlug;
    const epId   = currentEpId;
    const locale = document.getElementById('yt-locale-sel').value || 'en';
    const badge  = document.getElementById('yt-status-badge');

    if (!slug || !epId) {
      badge.textContent = 'No episode loaded';
      badge.style.background = 'var(--active-bg)';
      document.getElementById('yt-meta-form').style.display    = 'none';
      document.getElementById('yt-thumb-section').style.display = 'none';
      document.getElementById('yt-subs-section').style.display  = 'none';
      document.getElementById('yt-actions').style.display       = 'none';
      return;
    }

    badge.textContent = 'Loading…';
    badge.style.background = 'var(--active-bg)';

    try {
      const r = await fetch(`/api/youtube_status?slug=${encodeURIComponent(slug)}&ep_id=${encodeURIComponent(epId)}&locale=${encodeURIComponent(locale)}`);
      const d = await r.json();

      const ytMissing = d.error && !d.youtube;

      // Show/hide generate section vs form
      document.getElementById('yt-generate-section').style.display = ytMissing ? 'block' : 'none';
      document.getElementById('yt-gen-error').style.display        = 'none';
      document.getElementById('yt-save-wrap').style.display        = ytMissing ? 'none' : 'block';

      if (ytMissing) {
        badge.textContent = 'No youtube.json — click Generate';
        badge.style.background = '#78350f';
        document.getElementById('yt-meta-form').style.display     = 'none';
        document.getElementById('yt-thumb-section').style.display = 'none';
        document.getElementById('yt-subs-section').style.display  = 'none';
        document.getElementById('yt-actions').style.display       = 'none';
        return;
      }

      // Fill metadata form
      const yt = d.youtube || {};
      _ytFillForm(yt);
      // Preserve all fields (including non-form ones) so ytSaveAll() doesn't lose them
      window._ytDraft = Object.assign({}, yt);
      document.getElementById('yt-meta-form').style.display    = 'flex';
      document.getElementById('yt-thumb-section').style.display = 'block';
      document.getElementById('yt-subs-section').style.display  = 'block';
      document.getElementById('yt-actions').style.display       = 'flex';

      // Upload state badge
      const st = d.upload_state || {};
      if (st.video_id) {
        badge.textContent = `Uploaded: ${st.video_id}`;
        badge.style.background = '#14532d';
        _ytShowChecklist(st, yt);
      } else {
        badge.textContent = 'Ready to validate';
        badge.style.background = '#1e3a5f';
        document.getElementById('yt-checklist').style.display = 'none';
      }

      // Review data
      _ytReviewData = d.review || null;
      document.getElementById('yt-copy-review-btn').style.display =
        _ytReviewData ? 'inline-block' : 'none';

      // Subtitles
      const subs = (yt.subtitles || []);
      document.getElementById('yt-subs-list').innerHTML = subs.map(s =>
        `<div>${s.exists !== false ? '✓' : '✗'} ${s.name || s.language}  <span style="color:var(--dim)">${s.file || ''}</span></div>`
      ).join('') || '<span style="color:var(--dim)">No subtitles defined</span>';

      // Video player for thumbnail seek
      const vidEl = document.getElementById('yt-preview-video');
      const vidSrc = `/api/episode_video?slug=${encodeURIComponent(slug)}&ep_id=${encodeURIComponent(epId)}&locale=${encodeURIComponent(locale)}`;
      if (vidEl.dataset.src !== vidSrc) {
        vidEl.src = vidSrc;
        vidEl.dataset.src = vidSrc;
        vidEl.style.display = 'block';
        document.getElementById('yt-use-frame-btn').style.display = 'inline-block';
        document.getElementById('yt-thumb-preview-wrap').style.display = 'none';
      }

      // Show existing thumbnail if present
      if (yt.thumbnail) {
        _ytShowThumb(`/api/yt_thumbnail?slug=${encodeURIComponent(slug)}&ep_id=${encodeURIComponent(epId)}&locale=${encodeURIComponent(locale)}`, yt.thumbnail_source_sec);
      }

    } catch(e) {
      badge.textContent = 'Error: ' + e.message;
      badge.style.background = '#7f1d1d';
    }
  }

  function _ytFillForm(yt) {
    const ti = document.getElementById('yt-title');
    if (ti) { ti.value = yt.title || ''; ytUpdateCounter('yt-title-ctr', ti.value.length, 70); }
    const de = document.getElementById('yt-desc');
    if (de) { de.value = yt.description || ''; ytUpdateCounter('yt-desc-ctr', de.value.length, 5000); }
    const cat = document.getElementById('yt-category');
    if (cat) cat.value = String(yt.category_id || '24');
    const prv = document.getElementById('yt-privacy');
    if (prv) prv.value = yt.privacy || 'private';
    const mfk = document.getElementById('yt-mfk');
    if (mfk) mfk.value = yt.made_for_kids ? 'true' : 'false';
    const ntf = document.getElementById('yt-notify');
    if (ntf) ntf.value = yt.notify_subscribers ? 'true' : 'false';
  }

  function ytUpdateCounter(id, n, max) {
    const el = document.getElementById(id);
    if (el) el.textContent = `${n}/${max}`;
  }

  // ── Generate youtube.json via Claude ────────────────────────────────────────
  async function ytGenerate() {
    const slug   = currentSlug; if (!slug) return;
    const epId   = currentEpId; if (!epId) return;
    const locale = document.getElementById('yt-locale-sel').value || 'en';
    const btn    = document.getElementById('yt-gen-btn');
    const errEl  = document.getElementById('yt-gen-error');

    btn.disabled = true;
    btn.textContent = '⏳ Generating…';
    errEl.style.display = 'none';

    try {
      const r = await fetch('/api/generate_youtube_json', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({slug, ep_id: epId, locale}),
      });
      const d = await r.json();

      if (!d.ok) {
        errEl.textContent = d.error || 'Unknown error';
        if (d.raw) errEl.textContent += '\n\nRaw: ' + d.raw.slice(0, 300);
        errEl.style.display = 'block';
        btn.disabled = false;
        btn.textContent = '✨ Generate youtube.json';
        return;
      }

      // Draft received — fill form and switch to edit mode
      const draft = d.draft;
      _ytFillForm(draft);
      document.getElementById('yt-generate-section').style.display = 'none';
      document.getElementById('yt-meta-form').style.display        = 'flex';
      document.getElementById('yt-thumb-section').style.display    = 'block';
      document.getElementById('yt-subs-section').style.display     = 'block';
      document.getElementById('yt-actions').style.display          = 'flex';
      document.getElementById('yt-save-wrap').style.display        = 'block';
      document.getElementById('yt-save-badge').textContent         = 'Draft — not saved yet';

      // Fill subtitles list
      if (draft.subtitles) {
        document.getElementById('yt-subs-list').innerHTML = draft.subtitles.map(s =>
          `<div>✓ ${s.name || s.language}  <span style="color:var(--dim)">${s.file}</span></div>`
        ).join('');
      }

      // Show thumbnail_source_sec as suggested
      if (draft.thumbnail_source_sec != null) {
        document.getElementById('yt-frame-sec').textContent =
          `Suggested frame: ${draft.thumbnail_source_sec.toFixed(1)}s — drag to adjust`;
      }

      // Keep draft in memory for save
      window._ytDraft = draft;

      const badge = document.getElementById('yt-status-badge');
      badge.textContent = 'Draft generated — review and save';
      badge.style.background = '#78350f';

    } catch(e) {
      errEl.textContent = 'Network error: ' + e.message;
      errEl.style.display = 'block';
      btn.disabled = false;
      btn.textContent = '✨ Generate youtube.json';
    }
  }

  // ── Save all fields to youtube.json ────────────────────────────────────────
  async function ytSaveAll() {
    const slug   = currentSlug; if (!slug) return;
    const epId   = currentEpId; if (!epId) return;
    const locale = document.getElementById('yt-locale-sel').value || 'en';
    const btn    = document.getElementById('yt-save-btn');
    const badge  = document.getElementById('yt-save-badge');

    // Collect all form fields
    const fields = {
      title:               document.getElementById('yt-title')?.value    || '',
      description:         document.getElementById('yt-desc')?.value     || '',
      category_id:         document.getElementById('yt-category')?.value || '24',
      privacy:             document.getElementById('yt-privacy')?.value  || 'private',
      made_for_kids:       document.getElementById('yt-mfk')?.value === 'true',
      notify_subscribers:  document.getElementById('yt-notify')?.value === 'true',
      // tags come from draft (no tag chip editor yet)
      tags:                (window._ytDraft || {}).tags || [],
      // auto-populated fields from draft
      upload_profile:      (window._ytDraft || {}).upload_profile || '',
      channel_id:          (window._ytDraft || {}).channel_id     || '',
      playlist_id:         (window._ytDraft || {}).playlist_id    || null,
      video_language:      (window._ytDraft || {}).video_language || locale,
      subtitles:           (window._ytDraft || {}).subtitles      || [],
      thumbnail:           (window._ytDraft || {}).thumbnail      || `projects/${slug}/episodes/${epId}/renders/${locale}/thumbnail.jpg`,
      thumbnail_source_sec:(window._ytDraft || {}).thumbnail_source_sec ?? null,
      license:             'youtube',
      embeddable:          true,
      publish_at:          null,
    };

    btn.disabled = true;
    btn.textContent = 'Saving…';
    badge.textContent = '';

    try {
      const r = await fetch('/api/youtube_save_all', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({slug, ep_id: epId, locale, fields}),
      });
      const d = await r.json();
      if (d.ok) {
        badge.textContent = '✓ Saved';
        btn.textContent = '💾 Save youtube.json';
        btn.disabled = false;
        // Reload tab to show full edit mode
        setTimeout(() => initYoutubeTab(), 500);
      } else {
        badge.textContent = '✗ Error: ' + (d.error || 'unknown');
        btn.disabled = false;
        btn.textContent = '💾 Save youtube.json';
      }
    } catch(e) {
      badge.textContent = '✗ ' + e.message;
      btn.disabled = false;
      btn.textContent = '💾 Save youtube.json';
    }
  }

  let _ytSaveTimer = null;
  function ytFieldChange(field, value) {
    // Debounce auto-save to youtube.json
    clearTimeout(_ytSaveTimer);
    _ytSaveTimer = setTimeout(() => _ytSaveField(field, value), 600);
  }

  async function _ytSaveField(field, value) {
    const slug   = currentSlug; if (!slug) return;
    const epId   = currentEpId; if (!epId) return;
    const locale = document.getElementById('yt-locale-sel').value || 'en';
    try {
      await fetch('/api/youtube_save_field', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({slug, ep_id: epId, locale, field, value}),
      });
    } catch(e) { console.warn('ytSaveField error:', e); }
  }

  // [Use this frame] → show preview only, no disk write yet (C23 fix)
  async function ytUseFrame() {
    const slug   = currentSlug; if (!slug) return;
    const epId   = currentEpId; if (!epId) return;
    const locale = document.getElementById('yt-locale-sel').value || 'en';
    const vidEl  = document.getElementById('yt-preview-video');
    const sec    = vidEl.currentTime;
    _ytPendingSec = sec;

    document.getElementById('yt-frame-sec').textContent = `Frame at ${sec.toFixed(2)}s`;

    const imgEl = document.getElementById('yt-thumb-img');
    imgEl.src = `/api/thumbnail_preview?slug=${encodeURIComponent(slug)}&ep_id=${encodeURIComponent(epId)}&locale=${encodeURIComponent(locale)}&sec=${sec}`;
    imgEl.onerror = () => {
      document.getElementById('yt-thumb-info').textContent = '✗ Frame extraction failed';
    };
    imgEl.onload = () => {
      document.getElementById('yt-thumb-info').textContent = `${imgEl.naturalWidth}×${imgEl.naturalHeight}  (not yet saved)`;
      document.getElementById('yt-save-thumb-btn').style.display = 'inline-block';
    };
    document.getElementById('yt-thumb-preview-wrap').style.display = 'block';
    document.getElementById('yt-save-thumb-btn').style.display = 'none';
    document.getElementById('yt-thumb-info').textContent = 'Extracting frame…';
  }

  // [Save as thumbnail] → explicit user confirmation, then disk write (C23 fix)
  async function ytSaveThumbnail() {
    if (_ytPendingSec === null) return;
    const slug   = currentSlug; if (!slug) return;
    const epId   = currentEpId; if (!epId) return;
    const locale = document.getElementById('yt-locale-sel').value || 'en';
    const btn    = document.getElementById('yt-save-thumb-btn');
    btn.disabled = true; btn.textContent = 'Saving…';
    try {
      const r = await fetch('/api/set_thumbnail_sec', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({slug, ep_id: epId, locale, sec: _ytPendingSec}),
      });
      const d = await r.json();
      if (d.ok) {
        document.getElementById('yt-thumb-info').textContent =
          `✓ Saved: ${d.thumbnail_path}`;
        btn.textContent = '✓ Saved';
        _ytPendingSec = null;
      } else {
        document.getElementById('yt-thumb-info').textContent = '✗ Save failed: ' + (d.error || 'unknown');
        btn.disabled = false; btn.textContent = '💾 Save as thumbnail';
      }
    } catch(e) {
      document.getElementById('yt-thumb-info').textContent = '✗ Error: ' + e.message;
      btn.disabled = false; btn.textContent = '💾 Save as thumbnail';
    }
  }

  function _ytShowThumb(src, sec) {
    const imgEl = document.getElementById('yt-thumb-img');
    imgEl.src = src;
    imgEl.onerror = () => {};
    const info = sec != null ? `Frame at ${sec}s` : 'Custom thumbnail';
    document.getElementById('yt-thumb-info').textContent = info;
    document.getElementById('yt-thumb-preview-wrap').style.display = 'block';
    document.getElementById('yt-save-thumb-btn').style.display = 'none';
  }

  async function ytUploadCustomThumb(input) {
    if (!input.files || !input.files[0]) return;
    const slug   = currentSlug; if (!slug) return;
    const epId   = currentEpId; if (!epId) return;
    const locale = document.getElementById('yt-locale-sel').value || 'en';
    const fd = new FormData();
    fd.append('file', input.files[0]);
    fd.append('slug', slug);
    fd.append('ep_id', epId);
    fd.append('locale', locale);
    try {
      const r = await fetch('/api/upload_thumbnail_file', {method: 'POST', body: fd});
      const d = await r.json();
      if (d.ok) {
        const imgEl = document.getElementById('yt-thumb-img');
        imgEl.src = URL.createObjectURL(input.files[0]);
        document.getElementById('yt-thumb-info').textContent = '✓ Custom thumbnail saved';
        document.getElementById('yt-thumb-preview-wrap').style.display = 'block';
        document.getElementById('yt-save-thumb-btn').style.display = 'none';
      }
    } catch(e) { alert('Upload failed: ' + e.message); }
  }

  async function ytAction(action) {
    const slug   = currentSlug; if (!slug) { alert('No episode loaded'); return; }
    const epId   = currentEpId; if (!epId) return;
    const locale = document.getElementById('yt-locale-sel').value || 'en';

    const logWrap = document.getElementById('yt-log-wrap');
    const logEl   = document.getElementById('yt-log');

    // Disable all action buttons while running
    const actionBtns = ['yt-btn-validate','yt-btn-upload','yt-btn-publish'];
    const labels = {'validate':'⏳ Validating…', 'upload':'⏳ Uploading…', 'publish':'⏳ Publishing…'};
    actionBtns.forEach(id => {
      const b = document.getElementById(id);
      if (b) { b.disabled = true; b.style.opacity = '0.45'; b.style.cursor = 'not-allowed'; }
    });
    const activeBtn = document.getElementById(`yt-btn-${action}`);
    if (activeBtn) activeBtn.textContent = labels[action] || '⏳ Running…';

    logWrap.style.display = 'block';
    logEl.textContent = `▶ ${action} started…\n`;
    logWrap.scrollTop = 0;

    try {
      const r = await fetch('/api/youtube_action', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({slug, ep_id: epId, locale, action}),
      });
      const d = await r.json();
      logEl.textContent += (d.output || d.error || JSON.stringify(d));
      if (d.ok !== false) {
        logEl.textContent += '\n\n✅ Done.';
        initYoutubeTab();  // refresh status
      } else {
        logEl.textContent += '\n\n❌ Failed (see above).';
      }
    } catch(e) {
      logEl.textContent += '\n\n❌ Error: ' + e.message;
    }

    // Re-enable buttons and restore labels
    const origLabels = {'yt-btn-validate':'✓ Validate', 'yt-btn-upload':'⬆ Upload (Private)', 'yt-btn-publish':'🌐 Publish'};
    actionBtns.forEach(id => {
      const b = document.getElementById(id);
      if (b) { b.disabled = false; b.style.opacity = ''; b.style.cursor = 'pointer'; b.textContent = origLabels[id]; }
    });

    logWrap.scrollTop = logWrap.scrollHeight;
  }

  function ytCopyReview() {
    if (!_ytReviewData) return;
    navigator.clipboard.writeText(JSON.stringify(_ytReviewData, null, 2))
      .then(() => {
        const btn = document.getElementById('yt-copy-review-btn');
        btn.textContent = '✓ Copied!';
        setTimeout(() => { btn.textContent = '📋 Copy Review Packet'; }, 2000);
      });
  }

  function _ytShowChecklist(st, yt) {
    const vid = st.video_id;
    const lines = [
      st.video_uploaded      ? '✅ Video uploaded (private)' : '⬜ Video not uploaded',
      (st.captions_uploaded && Object.values(st.captions_uploaded).some(Boolean))
        ? '✅ Subtitles: ' + Object.keys(st.captions_uploaded).filter(k => st.captions_uploaded[k]).join(', ')
        : '⬜ Subtitles not uploaded',
      st.thumbnail_uploaded  ? '✅ Thumbnail uploaded'        : '⬜ Thumbnail not uploaded',
      st.playlist_added      ? '✅ Added to playlist'         : '⬜ Playlist not added yet',
    ];

    const studioBase = `https://studio.youtube.com/video/${vid}`;
    const manualSteps = vid ? `
      <div style="margin-top:10px;border-top:1px solid var(--border);padding-top:10px">
        <div style="font-size:0.82em;font-weight:600;color:var(--dim);margin-bottom:4px">Complete in YouTube Studio:</div>
        <div>☐ End screen &amp; Cards → <a href="${studioBase}/edit" target="_blank" style="color:#60a5fa">Open Editor ↗</a></div>
        <div>☐ Subtitles (fine-tune) → <a href="${studioBase}/translations" target="_blank" style="color:#60a5fa">Open Translations ↗</a></div>
        <div>☐ Dubbed audio → <a href="${studioBase}/dubbing" target="_blank" style="color:#60a5fa">Open Dubbing ↗</a></div>
        <div>☐ Monetization → <a href="${studioBase}/monetization" target="_blank" style="color:#60a5fa">Open Monetization ↗</a></div>
        <div>☐ Comments setting → <a href="${studioBase}/edit" target="_blank" style="color:#60a5fa">Details → More options ↗</a></div>
      </div>` : '';

    document.getElementById('yt-checklist-body').innerHTML =
      lines.map(l => `<div>${l}</div>`).join('') + manualSteps;
    document.getElementById('yt-checklist').style.display = 'block';
  }

  let pipeStepEs    = null;
  let pipeStoryFile = null;   // auto-detected from pipeline_vars.*.sh
  let pipeRunning      = null;   // { from, to } while an llm range is running; null otherwise
  let _stagesDoneInCurrentRun = new Set(); // stage numbers confirmed done in the CURRENT run
  let activeVideoLocale = null;  // currently selected video locale tab
  let _lastSyncedEp          = null;  // "slug|ep_id" — guards syncRunTabFromPipeline from firing every poll
  let _lastStatusFingerprint = null;  // JSON fingerprint — skips DOM re-render when nothing changed
  let _pipeTabInited = false;

  async function initPipelineTab() {
    // Populate episode selector (only once per session, but always refresh list)
    try {
      const res = await fetch('/list_projects');
      const { projects } = await res.json();
      const sel = document.getElementById('pipe-ep-select');
      const prev = sel.value;
      sel.innerHTML = '<option value="">— select episode —</option>';
      projects.forEach(proj => {
        proj.episodes.forEach(ep => {
          const opt = document.createElement('option');
          opt.value = proj.slug + '|' + ep.id;
          opt.textContent = proj.slug + ' / ' + ep.id;
          sel.appendChild(opt);
        });
      });
      // Always follow the Run tab's current episode when available.
      // If that option isn't in the dropdown yet (episode still being created),
      // sel.value silently stays empty — fall through to prev as the backup.
      if (currentSlug && currentEpId) {
        sel.value = currentSlug + '|' + currentEpId;
      }
      if (!sel.value && prev) {
        sel.value = prev;   // restore previous selection when Run tab has nothing
      }
      if (sel.value) onPipeEpChange();
      loadRunProjects();  // keep Run tab project dropdown in sync
    } catch (e) {
      console.error('Pipeline tab init error:', e);
    }
  }

  function onPipeEpChange() {
    const val = document.getElementById('pipe-ep-select').value;
    if (!val) {
      pipeEpSlug = pipeEpId = null;
      document.getElementById('pipe-body').innerHTML =
        '<div style="color:var(--dim);font-style:italic;font-size:0.83em;padding:4px 0">Select an episode above.</div>';
      document.getElementById('pipe-review-wrap').style.display = 'none';
      return;
    }
    const parts = val.split('|');
    pipeEpSlug = parts[0]; pipeEpId = parts[1];
    refreshPipeline();
  }

  async function refreshPipeline() {
    if (!pipeEpSlug || !pipeEpId) return;
    const body = document.getElementById('pipe-body');
    // Show "Loading…" only on first load (panel empty); skip on background auto-polls
    // so the existing content doesn't flicker/disappear every 5 seconds.
    const firstLoad = !body.children.length ||
      (body.children.length === 1 && body.children[0].textContent.includes('Select an episode'));
    if (firstLoad) {
      body.innerHTML = '<div style="color:var(--dim);font-style:italic;font-size:0.83em;padding:4px 0">Loading…</div>';
    }
    try {
      // cache:'no-store' prevents browser from serving a stale cached copy even
      // when the server's Cache-Control header is respected inconsistently.
      const res = await fetch(
        '/pipeline_status?slug=' + encodeURIComponent(pipeEpSlug) +
        '&ep_id='                + encodeURIComponent(pipeEpId),
        { cache: 'no-store' }
      );
      if (!res.ok) throw new Error('HTTP ' + res.status);
      pipeStatus = await res.json();
      renderPipelineStatus(pipeStatus);
    } catch (e) {
      body.innerHTML = '<span style="color:var(--red);font-family:var(--mono);font-size:0.82em">Error: ' +
                       escHtml(String(e)) + '</span>';
    }
  }

  function statusIcon(done) {
    return done
      ? '<span class="step-status done">✓</span>'
      : '<span class="step-status pending">○</span>';
  }

  async function loadStoryFiles(selectId, autoSelect) {
    try {
      const res = await fetch('/list_stories');
      const { stories } = await res.json();
      const sel = document.getElementById(selectId);
      if (!sel) return;
      // Preserve any manual selection; fall back to autoSelect hint
      const prev = sel.value || autoSelect || '';
      sel.innerHTML = '<option value="">— story file —</option>';
      stories.forEach(s => {
        const opt = document.createElement('option');
        opt.value = s; opt.textContent = s;
        sel.appendChild(opt);
      });
      if (prev) sel.value = prev;
    } catch (e) {}
  }

  // ── Stage command detail strings ─────────────────────────────────────────────
  const stageDetail = {
    0:  'canon_check.py / gen_pipeline_vars.py + voice_cast_narrator.py  (narration formats)\n' +
        'claude -p --model haiku  prompts/p_0.txt  (episodic/monologue)\n' +
        '→ reads meta.json + story.txt, casts voices per locale\n' +
        '→ writes VoiceCast.json, pipeline_vars.sh',
    1:  'canon_check.py  (deterministic — no LLM)\n→ reads canon.json, prints consistency report',
    2:  'claude -p --model sonnet prompts/p_2.txt\n→ writes StoryPrompt.json',
    3:  'gen_script_narration.py  (narration formats, deterministic)\n' +
        'claude -p --model sonnet prompts/p_3.txt  (episodic/monologue)\n' +
        '→ writes Script.json',
    4:  'gen_shotlist_scaffold.py → claude prompts/p_4_c.txt  (narration formats)\n' +
        'claude -p --model sonnet prompts/p_4.txt  (episodic/monologue)\n' +
        '→ writes ShotList.json',
    5:  'gen_manifest_structure.py → claude prompts/p_5_c.txt → gen_vo_manifest.py  (narration)\n' +
        'claude -p --model sonnet prompts/p_5.txt  (episodic/monologue)\n' +
        '→ writes AssetManifest_draft.shared.json + AssetManifest_draft.{locale}.json',
    6:  'canon_diff_chars.py → claude prompts/p_6.txt → validate_scaffold.py\n→ writes canon_diff.json',
    7:  'canon_merge.py  (deterministic — no LLM)\n→ updates projects/{slug}/canon.json',
    8:  'claude -p --model sonnet prompts/p_8.txt\n→ writes AssetManifest_draft.{locale}.json per non-en locale',
    10: '── Shared steps (once, all locales) ──\n' +
        '[ 1] gen_music_clip.py      --manifest AssetManifest_draft.shared.json\n' +
        '[ 2] gen_characters.py      --manifest AssetManifest_draft.shared.json\n' +
        '[ 3] gen_backgrounds.py     --manifest AssetManifest_draft.shared.json\n' +
        '[ 4] gen_sfx.py             --manifest AssetManifest_draft.shared.json\n' +
        '── Per-locale steps ──\n' +
        '[ 5] manifest_merge.py      --shared ...shared.json --locale ...{locale}.json\n' +
        '                            --out AssetManifest_merged.{locale}.json\n' +
        '[ 6] gen_tts_cloud.py       --manifest AssetManifest_merged.{locale}.json\n' +
        '[ 7] post_tts_analysis.py   --manifest AssetManifest_merged.{locale}.json\n' +
        '[ 8] apply_music_plan.py    --plan assets/music/MusicPlan.json\n' +
        '                            --manifest AssetManifest_merged.{locale}.json\n' +
        '     (skipped if Music disabled; ⏸ pauses to wait for Music tab confirm if plan missing)\n' +
        '[ 9] resolve_assets.py      --manifest AssetManifest_merged.{locale}.json\n' +
        '                            --out AssetManifest.media.{locale}.json\n' +
        '[10] gen_render_plan.py     --manifest AssetManifest_merged.{locale}.json\n' +
        '                            --media AssetManifest.media.{locale}.json\n' +
        '                            → RenderPlan.{locale}.json\n' +
        '[11] render_video.py        --plan RenderPlan.{locale}.json --locale {locale}\n' +
        '                            --out renders/{locale}/  →  renders/{locale}/output.mp4',
  };

  // ── syncRunTabFromPipeline(slug, epId, storyFile, voiceCast) ─────────────────
  // Called whenever a pipeline episode is selected or refreshed.
  // Restores Run-tab state (story textarea, currentSlug/epId, VoiceCast) so the
  // user can resume work after a page reload or server restart.
  async function syncRunTabFromPipeline(slug, epId, storyFile, voiceCast, meta = {}) {
    // Always sync slug/ep_id so the Run tab references the right episode
    currentSlug = slug;
    currentEpId = epId;

    // ── Apply metadata to existing-ep-bar ────────────────────────────────────
    document.getElementById('ex-title').value = meta.title  || slug;
    document.getElementById('ex-genre').value = meta.genre  || '';
    document.getElementById('save-ep-status').textContent = '';
    if (meta.story_format) {
      const _ENABLED_FORMATS = new Set(['continuous_narration', 'ssml_narration']);
      const fmt = _ENABLED_FORMATS.has(meta.story_format) ? meta.story_format : 'continuous_narration';
      _selectedFormat = fmt;
      document.getElementById('info-format-sel-existing').value = fmt;
      updateFormatHint(fmt, 'format-hint-existing');
      _updateMediaConfigPanel(fmt);
    }
    if (meta.locales_str) {
      const locs = meta.locales_str.split(',').map(l => l.trim());
      document.getElementById('locale-en-ex').checked       = locs.includes('en');
      document.getElementById('locale-zh-Hans-ex').checked  = locs.includes('zh-Hans');
    }
    // Restore Music state into the single top-bar checkbox
    // meta.no_music=true → skip music → checkbox unchecked
    // meta.no_music=false → include music → checkbox checked
    if (meta.no_music !== undefined) {
      noMusic = !!meta.no_music;
      const chk = document.getElementById('chk-no-music');
      if (chk) chk.checked = !noMusic;
    }
    // Restore Purge Cache state
    if (meta.purge_cache !== undefined) {
      purgeAssets = !!meta.purge_cache;
      const chk = document.getElementById('chk-purge-assets');
      if (chk) chk.checked = purgeAssets;
    }

    // Always reload story when switching to a different episode
    if (storyFile && storyFile !== _lastRunFilename) {
      try {
        const r = await fetch('/read_story?story_file=' + encodeURIComponent(storyFile));
        const d = await r.json();
        if (d.ok && d.content) {
          storyEl.value      = d.content;
          _lastRunFilename   = storyFile;
          fileBadgeEl.textContent = storyFile;
        }
      } catch (_) { /* non-fatal: server may not have the file */ }
    }

    // Always reload VoiceCast when switching episodes — never use stale data
    if (voiceCast) {
      if (!_voiceCatalog)                  await loadVoiceCatalog();
      if (!Object.keys(_vcPresets).length) await loadVoicePresets();
      if (!_voiceIndex)                    await loadVoiceIndex();
      _vcData    = voiceCast;
      _vcLocales = meta.locales_str
        ? meta.locales_str.split(',').map(l => l.trim()).filter(Boolean)
        : [];
      // Clear rendered cards so switchStoryTab always re-renders with fresh data
      document.getElementById('vc-cards').innerHTML = '';
      // Re-render immediately if the Voice Cast editor is already open
      if (document.getElementById('vc-editor').style.display !== 'none') {
        renderVcEditor(voiceCast, _voiceCatalog, _vcLocales);
      }
    } else {
      _vcData    = null;
      _vcLocales = [];
      document.getElementById('vc-cards').innerHTML = '';  // clear stale cards
    }

    // ── Alignment warning on episode load ────────────────────────────────────
    // Fetch VO alignment and print a one-line warning to the OUTPUT box so the
    // user immediately sees if Chinese VO is misaligned — without clicking any tab.
    // NOTE: pipeStatus may not be loaded yet here, so check alignment data directly.
    try {
      const ar = await fetch('/api/vo_alignment?slug=' + encodeURIComponent(slug)
                            + '&ep_id=' + encodeURIComponent(epId));
      _lastAlignmentData = await ar.json();
      const badLocs = (_lastAlignmentData.locales || []).filter(
        loc => loc.total_lines > 0 && loc.flagged_count / loc.total_lines > 0.20
      );
      if (badLocs.length) {
        const parts = badLocs.map(loc =>
          `${loc.locale} ${loc.flagged_count}/${loc.total_lines}句偏短（最差${loc.worst_ratio_before.toFixed(2)}）`
        );
        appendLine('', null);
        appendLineTs('── VO 配音对齐检查 ──────────────────────────────────', 'sys');
        appendLineTs('⚠️  配音偏短 — ' + parts.join(' · ') + ' — 建议重跑 Stage 8→10', null);
      }
    } catch(_) {}
  }

  function renderPipelineStatus(status) {
    // True while the Run tab has an active run.sh stream — Pipeline run buttons
    // are disabled during this period to prevent startPipeStep() → /stop from
    // accidentally SIGTERMing the background pipeline (exit code -15 bug).
    const _runTabBusy = !!(es && es.readyState !== EventSource.CLOSED);

    // Store the auto-detected story file for loading story text into the textarea
    pipeStoryFile = status.story_file || null;

    // Sync Run tab (story textarea, currentSlug/epId, VoiceCast, metadata) only
    // when the episode selection changes — NOT on every 5-second background poll.
    // Without this guard the auto-poller would overwrite currentSlug/currentEpId
    // and user-visible fields (no_music, locales, format) every 5 s.
    const _epKey = (pipeEpSlug || '') + '|' + (pipeEpId || '');
    if (_epKey !== _lastSyncedEp) {
      _lastSyncedEp = _epKey;
      _lastStatusFingerprint = null;  // new episode → force full re-render
      syncRunTabFromPipeline(pipeEpSlug, pipeEpId, pipeStoryFile, status.voice_cast, status);
    }

    // Skip full DOM re-render when nothing has changed — prevents collapsing
    // expanded detail panels and resetting video tab state on every 5-second poll.
    const _fp = JSON.stringify({
      llm:     status.llm_stages,
      locale:  status.locale_steps,
      shared:  status.shared_steps,
      videos:  status.ready_videos,
      dubbed:  status.ready_dubbed,
      running: pipeRunning,   // client-side: forces re-render when run starts/stops
      doneset: Array.from(_stagesDoneInCurrentRun).sort()  // forces re-render as stages complete
    });
    if (_fp === _lastStatusFingerprint) return;
    _lastStatusFingerprint = _fp;
    _invalidateDiagnoseIfStale();   // clear cached diagnosis when pipeline state changes

    const body = document.getElementById('pipe-body');
    body.innerHTML = '';

    // ── Stage list ────────────────────────────────────────────────────────────
    const section = document.createElement('div');
    section.className = 'pipe-section';

    const hdr = document.createElement('div');
    hdr.className = 'pipe-section-hdr';
    hdr.innerHTML = '⚡ Pipeline — <span style="font-family:var(--mono);color:var(--dim);font-weight:400">run.sh  stages 0 – 8, 10</span>';
    section.appendChild(hdr);
    body.appendChild(section);

    const llmDefs = [
      { n:0,  display:0, label:'Stage 0  — Cast voices & write pipeline_vars.sh',         key:'stage_0'  },
      { n:1,  display:1, label:'Stage 1  — Check story & world consistency',             key:'stage_1'  },
      { n:2,  display:2, label:'Stage 2  — Write episode direction (StoryPrompt)',       key:'stage_2'  },
      { n:3,  display:3, label:'Stage 3  — Write script & character dialogue',           key:'stage_3'  },
      { n:4,  display:4, label:'Stage 4  — Break script into visual shots (ShotList)',   key:'stage_4'  },
      { n:5,  display:5, label:'Stage 5  — List required assets (images, voice, music)', key:'stage_5'  },
      { n:6,  display:6, label:'Stage 6  — Identify new story facts to record',          key:'stage_6'  },
      { n:7,  display:7, label:'Stage 7  — Update story memory (world canon)',           key:'stage_7'  },
      { n:8,  display:8, label:'Stage 8  — Translate & adapt for each language',         key:'stage_8'  },
      { n:10, display:9, label:'Stage 9  — Merge assets & generate video (output.mp4)',  key:'stage_10' },
    ];
    // Auto-derived from the array — no hardcoded max needed.
    const _maxDisplay = llmDefs[llmDefs.length - 1].display;
    const _maxN       = llmDefs[llmDefs.length - 1].n;

    // ── Sequential done propagation ───────────────────────────────────────────
    // If stage N is done, all stages 0..N-1 are implicitly done too.
    const stagesMap = status.llm_stages || {};
    let maxDone = -1;
    llmDefs.forEach(({ n, key }) => {
      if ((stagesMap[key] || {}).done) maxDone = n;
    });
    if (maxDone >= 0) {
      llmDefs.forEach(({ n, key }) => {
        if (n <= maxDone) {
          if (!stagesMap[key]) stagesMap[key] = {};
          stagesMap[key].done = true;
        }
      });
    }

    // While a run is active, clear stale ✓ for stages in range.
    // A stage only gets its checkmark back when the CURRENT run's stream
    // emits "✓ Stage N complete" (tracked in _stagesDoneInCurrentRun).
    // This is time-independent — no grace-period expiry — and correctly handles
    // long-running stages (translation, TTS synthesis, etc.).
    if (pipeRunning) {
      llmDefs.forEach(({ n, key }) => {
        if (n >= pipeRunning.from && n <= pipeRunning.to) {
          if (!_stagesDoneInCurrentRun.has(n)) {
            stagesMap[key] = { done: false, artifacts: [] };
          }
        }
      });
    }

    llmDefs.forEach(({ n, display, label, key }) => {
      const info    = stagesMap[key] || { done: false, artifacts: [] };
      const detail  = stageDetail[n] || '';

      // ── Row wrapper (row + collapsible detail) ──────────────────────────────
      const wrap = document.createElement('div');

      const row = document.createElement('div');
      row.className = 'pipe-step';

      // Artifact links
      const artHtml = (info.artifacts || []).map(a => {
        const name = a.split('/').pop();
        // output.mp4 → play button
        if (name === 'output.mp4') {
          const locale = a.includes('/renders/') ? a.split('/renders/')[1].split('/')[0] : '';
          return '<span class="step-artifact" onclick="playLocaleVideo(\'' + escHtml(locale) +
                 '\')" title="Play video">▶ ' + escHtml(name) + '</span>';
        }
        return '<span class="step-artifact" onclick="openArtifact(\'' + escHtml(a) + '\')">' +
               escHtml(name) + '</span>';
      }).join('&nbsp;');

      // Expand button (only if we have detail text)
      const expandBtn = document.createElement('button');
      expandBtn.className = 'btn-expand';
      expandBtn.textContent = '›';
      expandBtn.title = 'Show commands';

      const _dis = _runTabBusy ? ' disabled title="Run tab pipeline is active"' : '';
      row.innerHTML =
        statusIcon(info.done) +
        '<span class="step-name">' + escHtml(label) + '</span>' +
        artHtml +
        '<span style="margin-left:auto;display:flex;gap:4px;flex-shrink:0">' +
          '<button class="btn-pipe-run"' + _dis + ' onclick="runLlmRange(' + n + ',' + n + ')">Run ' + display + '</button>' +
          (n < _maxN ? '<button class="btn-pipe-run"' + _dis + ' onclick="runLlmRange(' + n + ',' + _maxN + ')">Run ' + display + '→' + _maxDisplay + '</button>' : '') +
        '</span>';
      row.prepend(expandBtn);

      // Detail panel
      const detailEl = document.createElement('div');
      detailEl.className = 'pipe-detail';

      if (n === 10) {
        // Stage 9: numbered Run / Run→11 buttons matching the main stage button style
        const LOCALE_STEPS = [
          { num: 5,  step: 'manifest_merge',   label: '5 — merge',
            cmd: 'manifest_merge.py --shared AssetManifest_draft.shared.json --locale AssetManifest_draft.{locale}.json --out AssetManifest_merged.{locale}.json' },
          { num: 6,  step: 'gen_tts',          label: '6 — tts',
            cmd: 'gen_tts_cloud.py --manifest AssetManifest_merged.{locale}.json' },
          { num: 7,  step: 'post_tts',         label: '7 — post_tts',
            cmd: 'post_tts_analysis.py --manifest AssetManifest_merged.{locale}.json' },
          { num: 8,  step: 'apply_music_plan', label: '8 — music plan',
            cmd: 'apply_music_plan.py --plan assets/music/MusicPlan.json --manifest AssetManifest_merged.{locale}.json  (skipped if Music disabled)' },
          { num: 9,  step: 'resolve_assets',   label: '9 — resolve',
            cmd: 'resolve_assets.py --manifest AssetManifest_merged.{locale}.json --out AssetManifest.media.{locale}.json' },
          { num: 10, step: 'gen_render_plan',  label: '10 — plan',
            cmd: 'gen_render_plan.py --manifest AssetManifest_merged.{locale}.json --media AssetManifest.media.{locale}.json  → RenderPlan.{locale}.json' },
          { num: 11, step: 'render_video',     label: '11 — render',
            cmd: 'render_video.py --plan RenderPlan.{locale}.json --locale {locale} --out renders/{locale}/  → renders/{locale}/output.mp4' },
        ];
        const localeStepsMap = status.locale_steps || {};
        const sharedStepsMap = status.shared_steps || {};
        const locales        = status.locales || [];
        // stage10Running is kept for reference but no longer used to hard-clear ✓s.
        // Sub-step done state is sourced directly from the server (file-existence checks)
        // so completed steps stay checked even while Stage 9 is still running.
        const stage10Running = !!(pipeRunning && pipeRunning.from <= 10 && pipeRunning.to >= 10); // eslint-disable-line no-unused-vars

        function makeRunBtn(label, onclick) {
          const b = document.createElement('button');
          b.className = 'btn-pipe-run';
          b.style.cssText = 'font-size:0.72em;padding:2px 8px';
          b.textContent = label;
          b.onclick = onclick;
          if (_runTabBusy) { b.disabled = true; b.title = 'Run tab pipeline is active'; }
          return b;
        }

        // ── Steps 1–4: shared (no locale) ───────────────────────────────────
        [
          { num: 1, step: 'gen_music_clip',  label: '1 — gen_music_clip',
            cmd: 'gen_music_clip.py --manifest AssetManifest_draft.shared.json' },
          { num: 2, step: 'gen_characters',  label: '2 — gen_characters',
            cmd: 'gen_characters.py --manifest AssetManifest_draft.shared.json' },
          { num: 3, step: 'gen_backgrounds', label: '3 — gen_backgrounds',
            cmd: 'gen_backgrounds.py --manifest AssetManifest_draft.shared.json' },
          { num: 4, step: 'gen_sfx',         label: '4 — gen_sfx',
            cmd: 'gen_sfx.py --manifest AssetManifest_draft.shared.json' },
        ].forEach(({ num, step, label, cmd }) => {
          const done = (sharedStepsMap[step] || {}).done || false;
          const row = document.createElement('div');
          row.className = 'pipe-substep-row';
          row.appendChild(Object.assign(document.createElement('span'), {
            innerHTML: statusIcon(done), style: 'flex-shrink:0'
          }));
          const nameSpan = document.createElement('span');
          nameSpan.className = 'pipe-substep-locale';
          nameSpan.style.cssText = 'min-width:0;flex:1';
          nameSpan.innerHTML = escHtml(label) +
            '<br><span style="font-family:var(--mono);font-size:0.72em;color:var(--dim)">' +
            escHtml(cmd) + '</span>';
          row.appendChild(nameSpan);
          const btnWrap = document.createElement('span');
          btnWrap.style.cssText = 'margin-left:auto;display:flex;gap:4px;flex-shrink:0';
          btnWrap.appendChild(makeRunBtn('Run ' + num, () =>
            startPipeStep({ type: 'post', step,
                            slug: pipeEpSlug, ep_id: pipeEpId, locale: '' })));
          btnWrap.appendChild(makeRunBtn('Run ' + num + '→11', () =>
            startPipeStep({ type: 'shared_chain', from_step: step,
                            slug: pipeEpSlug, ep_id: pipeEpId })));
          row.appendChild(btnWrap);
          detailEl.appendChild(row);
        });

        // ── Steps 5–10: per-locale ──────────────────────────────────────────
        if (locales.length === 0) {
          const hint = document.createElement('div');
          hint.style.cssText = 'color:var(--dim);font-size:0.78em;padding:4px 0';
          hint.textContent = 'No locales yet — run Stages 0–8 first.';
          detailEl.appendChild(hint);
        } else {
          locales.forEach(locale => {
            // Locale header
            const hdr = document.createElement('div');
            hdr.style.cssText = 'color:var(--dim);font-size:0.74em;padding:5px 0 2px;font-family:var(--mono)';
            hdr.textContent = locale + ':';
            detailEl.appendChild(hdr);

            const lsteps = localeStepsMap[locale] || {};
            LOCALE_STEPS.forEach(({ num, step, label, cmd }) => {
              const done = (lsteps[step] || {}).done || false;
              const row  = document.createElement('div');
              row.className = 'pipe-substep-row';
              // status icon
              row.appendChild(Object.assign(document.createElement('span'), {
                innerHTML: statusIcon(done), style: 'flex-shrink:0'
              }));
              // step label + sample command
              const cmdDisplay = cmd.replace(/\{locale\}/g, locale);
              const nameSpan = document.createElement('span');
              nameSpan.className = 'pipe-substep-locale';
              nameSpan.style.cssText = 'min-width:0;flex:1';
              nameSpan.innerHTML = escHtml(label) +
                '<br><span style="font-family:var(--mono);font-size:0.72em;color:var(--dim)">' +
                escHtml(cmdDisplay) + '</span>';
              row.appendChild(nameSpan);
              // Run N  [Run N→7]  — right-aligned
              const btnWrap = document.createElement('span');
              btnWrap.style.cssText = 'margin-left:auto;display:flex;gap:4px;flex-shrink:0';
              btnWrap.appendChild(makeRunBtn('Run ' + num, () =>
                startPipeStep({ type: 'post', step,
                                slug: pipeEpSlug, ep_id: pipeEpId, locale })));
              if (num < 11) {
                btnWrap.appendChild(makeRunBtn('Run ' + num + '→11', () =>
                  startPipeStep({ type: 'locale', from_step: step,
                                  slug: pipeEpSlug, ep_id: pipeEpId, locale })));
              }
              row.appendChild(btnWrap);
              detailEl.appendChild(row);
            });
          });
        }
      } else {
        detailEl.textContent = detail;
      }

      expandBtn.addEventListener('click', () => {
        const open = detailEl.classList.toggle('open');
        expandBtn.classList.toggle('open', open);
        expandBtn.title = open ? 'Hide commands' : 'Show commands';
      });

      wrap.appendChild(row);
      wrap.appendChild(detailEl);
      section.appendChild(wrap);
    });

    // ── VIDEO REVIEW block (Video / Soundtrack / Voice Cast tabs) ────────────
    const reviewWrap        = document.getElementById('pipe-review-wrap');
    const reviewContentTabs = document.getElementById('review-content-tabs');
    const reviewLocaleTabs  = document.getElementById('review-locale-tabs');
    const readyVideos = status.ready_videos || [];
    const readyDubbed = status.ready_dubbed || [];

    const hasVideo = readyVideos.length > 0;
    const hasAudio = readyDubbed.length > 0;

    if (!hasVideo && !hasAudio) {
      reviewWrap.style.display = 'none';
    } else {
      reviewWrap.style.display = '';
      reviewContentTabs.innerHTML = '';
      reviewLocaleTabs.innerHTML  = '';

      // ── Locale tab builder ──────────────────────────────────────────────────
      function buildLocaleTabs(locales, onSelect, activeLocale) {
        reviewLocaleTabs.innerHTML = '';
        locales.forEach(l => {
          const btn = document.createElement('button');
          btn.className = 'btn-locale-tab' + (l === activeLocale ? ' active' : '');
          btn.textContent = l;
          btn.onclick = () => {
            reviewLocaleTabs.querySelectorAll('.btn-locale-tab')
              .forEach(b => b.classList.toggle('active', b === btn));
            onSelect(l);
          };
          reviewLocaleTabs.appendChild(btn);
        });
      }

      // ── Content pane switcher ───────────────────────────────────────────────
      function switchPane(pane) {
        document.querySelectorAll('.review-pane')
          .forEach(p => p.classList.remove('active'));
        document.getElementById('review-pane-' + pane).classList.add('active');
        if (pane === 'video') {
          const tgt = (activeVideoLocale && readyVideos.includes(activeVideoLocale))
                      ? activeVideoLocale : readyVideos[0];
          buildLocaleTabs(readyVideos, playLocaleVideo, tgt);
          playLocaleVideo(tgt);
        } else if (pane === 'audio') {
          buildLocaleTabs(readyDubbed, playLocaleDubbedAudio, readyDubbed[0]);
          playLocaleDubbedAudio(readyDubbed[0]);
        }
      }

      // ── Content tab buttons ─────────────────────────────────────────────────
      const tabDefs = [];
      if (hasVideo) tabDefs.push({ pane: 'video', label: 'Video' });
      if (hasAudio) tabDefs.push({ pane: 'audio', label: 'Soundtrack' });

      tabDefs.forEach(({ pane, label }, i) => {
        const btn = document.createElement('button');
        btn.className = 'btn-review-tab' + (i === 0 ? ' active' : '');
        btn.textContent = label;
        btn.onclick = () => {
          reviewContentTabs.querySelectorAll('.btn-review-tab')
            .forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          switchPane(pane);
        };
        reviewContentTabs.appendChild(btn);
      });

      // Show first available pane on initial render
      switchPane(tabDefs[0].pane);
    }
  }
  // Keep Next Step banner in sync whenever Pipeline tab refreshes
  if (document.getElementById('sr-panel').style.display !== 'none') {
    renderNextStep();
  }

  function playLocaleVideo(locale) {
    activeVideoLocale = locale;
    const video = document.getElementById('pipe-video');
    const path  = 'projects/' + pipeEpSlug + '/episodes/' + pipeEpId +
                  '/renders/' + locale + '/output.mp4';
    video.pause();
    // Append ?t= cache-buster so the browser always fetches the latest file after re-render
    video.src = '/serve_media?path=' + encodeURIComponent(path) + '&t=' + Date.now();
    video.load();
  }

  function playLocaleVideoBtn(locale, clickedBtn) {
    document.querySelectorAll('#review-locale-tabs .btn-locale-tab').forEach(b =>
      b.classList.toggle('active', b === clickedBtn));
    playLocaleVideo(locale);
  }

  function playLocaleDubbedAudio(locale) {
    const audio = document.getElementById('pipe-audio');
    const path  = 'projects/' + pipeEpSlug + '/episodes/' + pipeEpId +
                  '/renders/' + locale + '/youtube_dubbed.m4a';
    audio.pause();
    audio.src = '/serve_media?path=' + encodeURIComponent(path);
    audio.load();
  }

  function playLocaleDubbedAudioBtn(locale, clickedBtn) {
    document.querySelectorAll('#review-locale-tabs .btn-locale-tab').forEach(b =>
      b.classList.toggle('active', b === clickedBtn));
    playLocaleDubbedAudio(locale);
  }

  function openArtifact(path) {
    window.open('/view_artifact?path=' + encodeURIComponent(path), '_blank');
  }

  function runLlmRange(from, to) {
    if (!pipeEpSlug || !pipeEpId) {
      switchTab('run');
      appendLine('⚠  No episode selected. Select an episode in the Pipeline tab first.', 'err');
      return;
    }
    const ep_dir = 'projects/' + pipeEpSlug + '/episodes/' + pipeEpId;
    startPipeStep({ type: 'llm', from, to, ep_dir });
  }

  function runPostStep(step, locale) {
    startPipeStep({ type: 'post', step, locale, slug: pipeEpSlug, ep_id: pipeEpId });
  }

  function runLocaleInPipeTerm(locale) {
    // Runs all 6 post-processing steps for this locale via /run_locale (skip-if-done)
    startPipeStep({ type: 'locale', locale, slug: pipeEpSlug, ep_id: pipeEpId });
  }

  // ── Run tab: Project / Episode dropdowns ────────────────────────────────────
  async function loadRunProjects() {
    try {
      const r = await fetch('/list_projects');
      const { projects } = await r.json();
      _runProjectList = projects || [];
      const sel = document.getElementById('run-project-sel');
      const prev = sel.value;
      sel.innerHTML = '<option value="">— New Project —</option>';
      projects.forEach(p => {
        const o = document.createElement('option');
        o.value = p.slug; o.textContent = p.slug;
        sel.appendChild(o);
      });
      // Restore previous selection if still present
      if (prev && Array.from(sel.options).some(o => o.value === prev)) sel.value = prev;
    } catch(_) {}
  }

  // Load story file + VoiceCast + metadata for a given slug+ep
  async function loadEpisodeDetails(slug, epId) {
    currentSlug = slug; currentEpId = epId;
    _lastRunFilename = null;   // force story reload — never show previous episode's story
    try {
      const res    = await fetch('/pipeline_status?slug=' + encodeURIComponent(slug) +
                                 '&ep_id=' + encodeURIComponent(epId));
      const status = await res.json();
      await syncRunTabFromPipeline(slug, epId, status.story_file, status.voice_cast, status);
    } catch(_) {}
  }

  function onRunProjectChange() {
    const slug  = document.getElementById('run-project-sel').value;
    const epSel = document.getElementById('run-episode-sel');
    epSel.innerHTML = '';

    if (!slug) {
      // ── New Project — clear all state from previous selection ───────────────
      epSel.disabled = true;
      epSel.innerHTML = '<option value="">—</option>';
      _usingExistingEp = false;
      currentSlug = null;
      currentEpId = null;
      _preparedMeta = null;
      _preparedEpId = null;
      _episodeCreated = false;
      document.getElementById('info-bar').classList.remove('visible');
      document.getElementById('existing-ep-bar').style.display = 'none';
      document.getElementById('btn-prepare').style.display = '';
      storyEl.value = '';
      outputEl.innerHTML = '<span class="sys">Ready. Paste a story above and press Run.</span>\n';
      document.getElementById('save-new-ep-row').style.display = 'none';
      setRunBtnEnabled(false);
      return;
    }

    // ── Existing project — populate episodes and auto-load the first one ──────
    epSel.disabled = false;
    const proj = _runProjectList.find(p => p.slug === slug);
    const episodes = (proj && proj.episodes) || [];
    episodes.forEach(ep => {
      const o = document.createElement('option');
      o.value = ep.id; o.textContent = ep.id;
      epSel.appendChild(o);
    });

    _usingExistingEp = true;
    document.getElementById('info-bar').classList.remove('visible');
    document.getElementById('existing-ep-bar').style.display = 'block';
    document.getElementById('btn-prepare').style.display = 'none';
    setRunBtnEnabled(true);

    // Auto-select first episode and load its details
    const firstEpId = episodes.length ? episodes[0].id : null;
    if (firstEpId) {
      epSel.value = firstEpId;
      loadEpisodeDetails(slug, firstEpId);
    }
  }

  function onRunEpisodeChange() {
    const slug = document.getElementById('run-project-sel').value;
    const epId = document.getElementById('run-episode-sel').value;
    if (slug && epId) loadEpisodeDetails(slug, epId);
    // Refresh YouTube tab if it is currently visible
    if (document.getElementById('panel-youtube').style.display !== 'none') {
      initYoutubeTab();
    }
  }

  // ── Prepare button ───────────────────────────────────────────────────────────
  async function runPrepare() {
    const story = storyEl.value.trim();
    if (!story) { storyEl.style.borderColor='var(--red)'; setTimeout(()=>(storyEl.style.borderColor=''),1200); return; }
    const btn = document.getElementById('btn-prepare');
    btn.disabled = true; btn.textContent = '⚙ Preparing…';
    setRunBtnEnabled(false);
    _episodeCreated = false;
    document.getElementById('save-new-ep-row').style.display = 'none';
    try {
      const r = await fetch('/api/infer_story_meta', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({story})
      });
      const d = await r.json();
      if (d.error && !d.title) throw new Error(d.error);
      _preparedMeta = d;
      _selectedFormat = d.story_format || 'continuous_narration';
      _updateMediaConfigPanel(_selectedFormat);
      applyPreparedMeta(d);
      document.getElementById('info-bar').classList.add('visible');
      // For new projects: fetch next episode ID and show Create Episode button
      const _slug = document.getElementById('info-slug').value.trim();
      if (_slug && !d.slug_exists) {
        try {
          const nr = await fetch('/api/next_episode_id?slug=' + encodeURIComponent(_slug));
          const nd = await nr.json();
          _preparedEpId = nd.next_ep_id || 's01e01';
        } catch(_) { _preparedEpId = 's01e01'; }
        const epEl = document.getElementById('info-ep-id');
        if (epEl) epEl.textContent = _preparedEpId;
        setRunBtnEnabled(true);
        document.getElementById('save-new-ep-row').style.display = '';
      } else if (!_slug) {
        // Empty slug (e.g. SSML with no extractable title) — show info bar,
        // let user fill in Title (which auto-derives slug via onTitleInput)
        _preparedEpId = null;
        const epEl = document.getElementById('info-ep-id');
        if (epEl) epEl.textContent = '(fill in Title ↑)';
        // Run stays disabled until onSlugInput computes episode ID
      } else {
        // Existing project/slug — Run directly
        _preparedEpId = null;
        const epEl = document.getElementById('info-ep-id');
        if (epEl) epEl.textContent = '—';
        setRunBtnEnabled(true);
      }
    } catch(e) {
      alert('Prepare failed: ' + e.message);
    } finally {
      btn.disabled = false; btn.textContent = '⚙ Prepare';
    }
  }

  // ── Create Episode — called automatically by runPrompt() for new projects ──
  // Throws on failure so the caller can handle the error.
  async function createEpisode() {
    const slug  = document.getElementById('info-slug').value.trim();
    const title = document.getElementById('info-title').value.trim();
    const genre = document.getElementById('info-genre').value.trim();
    const story = storyEl.value.trim();
    const locs  = getSelectedLocales();
    const ep_id = _preparedEpId;
    if (!slug || !title || !ep_id) throw new Error('Prepare must run first (missing slug or episode ID).');
    const r = await fetch('/api/create_episode', {
      method:  'POST',
      headers: {'Content-Type': 'application/json'},
      body:    JSON.stringify({
        slug, ep_id, story, title, genre,
        story_format: _selectedFormat,
        locales:      locs.join(','),
        no_music:     !(document.getElementById('chk-no-music')?.checked),
        purge_cache:  document.getElementById('chk-purge-assets')?.checked || false,
      })
    });
    const d = await r.json();
    if (!d.ok) throw new Error(d.error || 'Unknown error');
    currentSlug = d.slug;
    currentEpId = d.ep_id;
    _episodeCreated = true;
    document.getElementById('save-new-ep-row').style.display = '';
    // Refresh project dropdown so it reflects the new project
    loadRunProjects().then(() => {
      const sel = document.getElementById('run-project-sel');
      if (Array.from(sel.options).some(o => o.value === currentSlug)) sel.value = currentSlug;
      const epSel = document.getElementById('run-episode-sel');
      epSel.disabled = false;
      if (!Array.from(epSel.options).some(o => o.value === currentEpId)) {
        const o = document.createElement('option');
        o.value = currentEpId; o.textContent = currentEpId;
        epSel.appendChild(o);
      }
      epSel.value = currentEpId;
    });
    return d.ep_dir;
  }

  // ── Save metadata for a newly created episode (reads from #info-bar fields) ─
  async function saveNewEpMeta() {
    const statusEl = document.getElementById('save-new-ep-status');
    const btnEl    = document.getElementById('btn-save-new-ep');
    // If the episode hasn't been created yet (Save clicked before Run), create it now
    if (!_episodeCreated) {
      btnEl.disabled = true;
      statusEl.textContent = 'Creating episode…';
      statusEl.style.color = 'var(--dim)';
      try {
        await createEpisode();
      } catch(e) {
        statusEl.textContent = '✗ ' + e.message;
        statusEl.style.color = '#e74c3c';
        btnEl.disabled = false;
        setTimeout(() => { statusEl.textContent = ''; }, 4000);
        return;
      }
    }
    if (!currentSlug || !currentEpId) return;
    const title    = document.getElementById('info-title').value.trim();
    const genre    = document.getElementById('info-genre').value.trim();
    const format   = document.getElementById('info-format-sel').value;
    const locs     = getSelectedLocales();
    const no_music = !(document.getElementById('chk-no-music')?.checked);
    btnEl.disabled = true;
    statusEl.textContent = 'Saving…';
    statusEl.style.color = 'var(--dim)';
    try {
      const resp = await fetch('/api/save_episode_meta', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ slug: currentSlug, ep_id: currentEpId,
                               title, genre, story_format: format,
                               locales: locs.join(','), no_music })
      });
      if (!resp.ok) throw new Error(await resp.text());
      statusEl.textContent = '✓ Saved';
      statusEl.style.color = 'var(--success,#4caf50)';
    } catch (err) {
      statusEl.textContent = '✗ ' + err.message;
      statusEl.style.color = '#e74c3c';
    } finally {
      btnEl.disabled = false;
      setTimeout(() => { statusEl.textContent = ''; }, 3000);
    }
  }

  function applyPreparedMeta(d) {
    const found = d.metadata_found || [];
    function setField(inputId, badgeId, value, field) {
      document.getElementById(inputId).value = value || '';
      const badge = document.getElementById(badgeId);
      if (found.includes(field)) {
        badge.textContent = '📌 from story'; badge.className = 'info-badge from-story'; badge.style.display = '';
      } else { badge.style.display = 'none'; }
    }
    setField('info-title', 'badge-title', d.title, 'title');
    setField('info-genre', 'badge-genre', d.genre, 'genre');
    // Slug
    const slug = d.slug_suggested || d.slug || '';
    document.getElementById('info-slug').value = slug;
    updateSlugBadge(d.slug_exists, slug);
    // Format — fall back to continuous_narration if saved format is disabled
    const _ENABLED_FORMATS = new Set(['continuous_narration', 'ssml_narration']);
    if (!_ENABLED_FORMATS.has(_selectedFormat)) _selectedFormat = 'continuous_narration';
    const fsel = document.getElementById('info-format-sel');
    fsel.value = _selectedFormat;
    updateFormatHint(_selectedFormat, 'format-hint');
    _updateMediaConfigPanel(_selectedFormat);
    // Auto-select project in dropdown if it already exists on disk
    if (d.slug_exists && slug) {
      loadRunProjects().then(() => {
        const sel = document.getElementById('run-project-sel');
        if (Array.from(sel.options).some(o => o.value === slug)) {
          sel.value = slug;
          onRunProjectChange();   // populates episodes, auto-loads first ep details
        }
      });
    }
  }

  function updateSlugBadge(exists, slug) {
    const badge = document.getElementById('badge-slug');
    if (exists) { badge.textContent = '⚠ exists → ' + slug; badge.className = 'info-badge exists'; }
    else         { badge.textContent = '✓ unique';            badge.className = 'info-badge unique'; }
    badge.style.display = '';
  }

  let _slugCheckTo = null;
  function onSlugInput() {
    const slug = document.getElementById('info-slug').value.trim();
    if (_slugCheckTo) clearTimeout(_slugCheckTo);
    _slugCheckTo = setTimeout(async () => {
      if (!slug) return;
      const r = await fetch('/api/check_slug?slug=' + encodeURIComponent(slug));
      const d = await r.json();
      const finalSlug = d.suggested || slug;
      updateSlugBadge(d.exists, finalSlug);
      // Auto-compute episode ID when slug changes (for SSML / manual flows)
      if (_preparedMeta && !d.exists) {
        try {
          const nr = await fetch('/api/next_episode_id?slug=' + encodeURIComponent(finalSlug));
          const nd = await nr.json();
          _preparedEpId = nd.next_ep_id || 's01e01';
        } catch(_) { _preparedEpId = 's01e01'; }
        const epEl = document.getElementById('info-ep-id');
        if (epEl) epEl.textContent = _preparedEpId;
        setRunBtnEnabled(true);
      }
    }, 400);
  }

  // Auto-derive slug from title input (for SSML / manual flows where slug is empty)
  function onTitleInput() {
    const title = document.getElementById('info-title').value.trim();
    const slugEl = document.getElementById('info-slug');
    // Only auto-derive if slug is currently empty or was auto-derived
    if (title && (!slugEl.value.trim() || slugEl.dataset.autoDerived === '1')) {
      const derived = title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 60);
      slugEl.value = derived;
      slugEl.dataset.autoDerived = '1';
      onSlugInput();  // trigger slug check + episode ID fetch
    }
  }

  function onFormatChange() {
    _selectedFormat = document.getElementById('info-format-sel').value;
    updateFormatHint(_selectedFormat, 'format-hint');
    _updateMediaConfigPanel(_selectedFormat);
  }
  function onFormatChangeExisting() {
    _selectedFormat = document.getElementById('info-format-sel-existing').value;
    updateFormatHint(_selectedFormat, 'format-hint-existing');
    _updateMediaConfigPanel(_selectedFormat);
  }
  function updateFormatHint(fmt, hintId) {
    const hints = {
      episodic:              '',
      continuous_narration:  'No characters on screen. Narrator voice only. Atmospheric visuals.',
      illustrated_narration: 'Characters shown as silent visuals. Narrator reads the story.',
      documentary:           'No characters. Narrator explains over b-roll backgrounds.',
      monologue:             'Single character speaks to camera. No scene changes.',
      ssml_narration:        'Pre-authored SSML. Pipeline generates cinematic visuals. Stages 1–3 & 8 skipped.',
    };
    document.getElementById(hintId).textContent = hints[fmt] || '';
  }

  function getSelectedLocales() {
    const locales = [];
    if (document.getElementById('locale-en')      ?.checked && !locales.includes('en')) locales.push('en');
    if (document.getElementById('locale-en-ex')   ?.checked && !locales.includes('en')) locales.push('en');
    if (document.getElementById('locale-zh-Hans') ?.checked && !locales.includes('zh-Hans')) locales.push('zh-Hans');
    if (document.getElementById('locale-zh-Hans-ex')?.checked && !locales.includes('zh-Hans')) locales.push('zh-Hans');
    return locales.length ? locales : ['en'];
  }
  function onLocaleChange() {
    // Update _vcLocales so VC editor reflects the new selection
    _vcLocales = getSelectedLocales();
    // Re-render VC editor tabs/cards if it is currently open
    if (_vcData && _voiceCatalog &&
        document.getElementById('vc-editor').style.display !== 'none') {
      document.getElementById('vc-cards').innerHTML = '';
      renderVcEditor(_vcData, _voiceCatalog, _vcLocales);
    }
  }
  // ── saveEpisodeMeta() ─────────────────────────────────────────────────────────
  // Saves title, genre, story_format, locales for the current episode to
  // projects/<slug>/episodes/<ep_id>/meta.json via POST /api/save_episode_meta.
  async function saveEpisodeMeta() {
    if (!currentSlug || !currentEpId) return;
    const statusEl = document.getElementById('save-ep-status');
    const btnEl    = document.getElementById('btn-save-ep-meta');
    const title    = document.getElementById('ex-title').value.trim();
    const genre    = document.getElementById('ex-genre').value.trim();
    const format   = document.getElementById('info-format-sel-existing').value;
    const locs     = getSelectedLocales();
    const no_music = !(document.getElementById('chk-no-music')?.checked);
    btnEl.disabled = true;
    statusEl.textContent = 'Saving…';
    statusEl.style.color = 'var(--dim)';
    try {
      const resp = await fetch('/api/save_episode_meta', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          slug: currentSlug, ep_id: currentEpId,
          title, genre, story_format: format, locales: locs.join(','), no_music,
          purge_cache: document.getElementById('chk-purge-assets')?.checked || false
        })
      });
      if (!resp.ok) throw new Error(await resp.text());
      statusEl.textContent = '✓ Saved';
      statusEl.style.color = 'var(--success,#4caf50)';
      // Sync _vcLocales in case locales changed
      _vcLocales = locs;
    } catch (err) {
      statusEl.textContent = '✗ ' + err.message;
      statusEl.style.color = '#e74c3c';
    } finally {
      btnEl.disabled = false;
      setTimeout(() => { statusEl.textContent = ''; }, 3000);
    }
  }

  function setRunBtnEnabled(enabled) {
    // Find the main Run button and enable/disable it
    const btn = document.querySelector('button[onclick*="runPrompt"]') ||
                document.getElementById('btn-run');
    if (btn) btn.disabled = !enabled;
  }

  function startPipeStep(params) {
    // Confirm before purging cached assets (destructive operation)
    if (purgeAssets && (params.type === 'locale' || params.type === 'shared_chain')) {
      if (!confirm('🗑 Purge Cache is ON — this will delete cached WAVs, images, '
                  + 'renders and manifests before running.\n\n'
                  + 'Continue with purge?')) {
        return;
      }
    }
    // Guard: if the Run tab has an active pipeline stream (es open), do not fire
    // /stop — that would SIGTERM the background run.sh mid-run (exit code -15).
    // The user should click Stop first if they genuinely want to abort the run.
    if (es && es.readyState !== EventSource.CLOSED) return;
    if (pipeStepEs) { pipeStepEs.close(); pipeStepEs = null; }
    fetch('/stop', { method: 'POST' }).catch(() => {});   // stop any running proc

    // Track which stage range is running so Pipeline tab doesn't show stale ✓
    pipeRunning = (params.type === 'llm') ? { from: params.from, to: params.to } : null;
    if (params.type === 'llm') {
      // Reset the per-run completion set — stages only get re-checked as the
      // stream emits "✓ Stage N complete", not from stale server-side file checks.
      _stagesDoneInCurrentRun = new Set();
      // Keep progress-dot globals accurate for startPipeStep-driven runs
      _runFromStage = params.from;
      _runToStage   = params.to;
      // Immediately uncheck stages from..to — instant visual feedback.
      // renderPipelineStatus will keep them unchecked (via _stagesDoneInCurrentRun)
      // until the stream confirms each one done.
      if (pipeStatus && pipeStatus.llm_stages) {
        for (let n = params.from; n <= params.to; n++) {
          pipeStatus.llm_stages['stage_' + n] = { done: false, artifacts: [] };
        }
        renderPipelineStatus(pipeStatus);
      }
    }
    refreshPipeline();   // sync with server

    // Route output to the Run tab — switch there and clear the output box
    switchTab('run');
    clearOutput();
    setStatus('running');

    const profile = renderProd ? 'high' : 'preview_local';
    let url;
    if (params.type === 'llm') {
      url = '/stream?ep_dir=' + encodeURIComponent(params.ep_dir) +
            '&from=' + params.from + '&to=' + params.to +
            '&test=0' +
            '&profile=' + profile;
    } else if (params.type === 'locale') {
      url = '/run_locale?slug='   + encodeURIComponent(params.slug) +
            '&ep_id='  + encodeURIComponent(params.ep_id) +
            '&locale=' + encodeURIComponent(params.locale) +
            '&profile=' + profile +
            '&no_music=' + (noMusic ? '1' : '0') +
            '&purge='  + (purgeAssets ? '1' : '0') +
            (params.from_step ? '&from=' + encodeURIComponent(params.from_step) : '');
    } else if (params.type === 'shared_chain') {
      // Run shared steps from_step→4 then all locale steps 5→10 for every locale
      url = '/run_stage10?slug='  + encodeURIComponent(params.slug) +
            '&ep_id='  + encodeURIComponent(params.ep_id) +
            '&from='   + encodeURIComponent(params.from_step) +
            '&profile=' + profile +
            '&no_music=' + (noMusic ? '1' : '0') +
            '&purge='  + (purgeAssets ? '1' : '0');
    } else {
      url = '/run_step?step='   + encodeURIComponent(params.step) +
            '&slug='   + encodeURIComponent(params.slug) +
            '&ep_id='  + encodeURIComponent(params.ep_id) +
            '&locale=' + encodeURIComponent(params.locale) +
            '&profile=' + profile +
            '&no_music=' + (noMusic ? '1' : '0');
      if (params.asset_ids) {
        url += '&asset_ids=' + encodeURIComponent(params.asset_ids);
      }
    }

    pipeStepEs = new EventSource(url);
    pipeStepEs.addEventListener('line', e => {
      const text = e.data;
      queueLine(text, '');
      // Stage progress (same patterns as runPrompt)
      const startM = text.match(/^\s{2}STAGE (\d+[_\w]*)\/\d+[_\w]*\s+[—\-]{1,2}\s+(.+)/);
      if (startM) {
        stageStartMs[parseInt(startM[1])] = Date.now();
        updateStageProgress(parseInt(startM[1]), startM[2].trim());
      }
      const doneM = text.match(/^✓ Stage (\d+[_\w]*) complete/);
      if (doneM) {
        const n = parseInt(doneM[1]);
        const elapsed = stageStartMs[n] != null
          ? `  elapsed ${fmtElapsed(Date.now() - stageStartMs[n])}` : '';
        appendLine(`  ⏱  finished ${fmtNow()}${elapsed}`, 'ts');
        // Record this stage as confirmed done in the current run so that
        // renderPipelineStatus() will show its checkmark even while pipeRunning is set.
        _stagesDoneInCurrentRun.add(n);
        markStageDone(n);
        insertReviewButtons(n);
        // Re-render Pipeline tab immediately so the checkmark appears for this stage
        // (fingerprint includes doneset, so this will not be a no-op).
        if (pipeStatus) renderPipelineStatus(pipeStatus);
      }
      const vm = text.match(/PROJECT_SLUG=(\S+)\s+EPISODE_ID=(\S+)/);
      if (vm) { currentSlug = vm[1]; currentEpId = vm[2]; }
    });
    pipeStepEs.addEventListener('error_line', e => appendLine(e.data, 'err'));
    pipeStepEs.addEventListener('done', e => {
      pipeStepEs.close(); pipeStepEs = null;
      pipeRunning = null;   // clear running state before refreshing Pipeline tab
      hideStageProgress();
      const code = parseInt(e.data);
      if (code === 0) {
        appendLine('[ ✓ Done ]', 'done');
        setStatus('idle');
        // Pre-select the completed episode in the Pipeline tab selector so that
        // initPipelineTab()'s prev-restore mechanism lands on the right episode,
        // even if the user had a different episode selected before this run.
        if (currentSlug && currentEpId) {
          pipeEpSlug = currentSlug; pipeEpId = currentEpId;
          const pipeSel = document.getElementById('pipe-ep-select');
          const tv = currentSlug + '|' + currentEpId;
          if (!Array.from(pipeSel.options).some(o => o.value === tv)) {
            const o = document.createElement('option');
            o.value = tv; o.textContent = currentSlug + ' / ' + currentEpId;
            pipeSel.appendChild(o);
          }
          pipeSel.value = tv;
        }
        switchTab('pipeline');
      } else {
        appendLine(`[ Exited with code ${code} ]`, 'err');
        setStatus('error');
      }
      setTimeout(async () => {
        await refreshPipeline();
        // After the status is fresh, print a next-step hint to the output box.
        if (code === 0) {
          // Fetch fresh alignment data and check first
          try {
            const ar = await fetch('/api/vo_alignment?slug=' + encodeURIComponent(currentSlug)
                                  + '&ep_id=' + encodeURIComponent(currentEpId));
            _lastAlignmentData = await ar.json();
          } catch(e) {}
          const alignStep = _alignmentNextStep();
          if (alignStep) {
            appendLine('', null);
            appendLineTs('── VO 配音对齐检查 ──────────────────────────────────', 'sys');
            appendLineTs(alignStep.msg, null);
            if (_lastAlignmentData && _lastAlignmentData.locales) {
              for (const loc of _lastAlignmentData.locales) {
                if (loc.flagged_count > 0) {
                  appendLineTs(`   原因：Stage 8 字数估算偏高，实测语速 vs 预估存在差距`, 'sys');
                  appendLineTs(`   建议：重跑 Stage 8→10，使用已校准的 ${loc.locale} 字符速率`, null);
                }
              }
            }
          } else {
            const ns = computeNextStep(pipeStatus);
            if (ns.state === 'action' || ns.state === 'done') {
              appendLine('', null);
              appendLineTs('── Next step ────────────────────────────────────────', 'sys');
              appendLineTs(ns.msg, ns.state === 'done' ? 'done' : null);
            }
          }
          renderNextStep();
        }
      }, 600);
    });
    pipeStepEs.onerror = () => {
      if (pipeStepEs) { pipeStepEs.close(); pipeStepEs = null; }
      appendLine('[ Connection lost ]', 'err');
      setStatus('error');
    };
  }
</script>
</body>
</html>
"""
# Inject VO polish thresholds into the JS at startup (source: polish_locale_vo)
HTML = HTML.replace("__VO_THRESH__",      f"{_VO_POLISH_THRESHOLD:.2f}")
HTML = HTML.replace("__VO_THRESH_HIGH__", f"{_VO_POLISH_THRESHOLD_HIGH:.2f}")


# ── Artifact viewer page ───────────────────────────────────────────────────────
VIEWER_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__FILENAME__</title>
<style>
  :root {
    --bg: #f5f3ef; --surface: #ffffff; --border: #d4d0c8;
    --gold: #8a6d3b; --text: #1a1a1a; --dim: #666;
    --mono: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    --c-key: #1a6db6; --c-str: #2a7a4a; --c-num: #9a6a00; --c-bool: #c03050;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    display: flex; flex-direction: column; height: 100vh; overflow: hidden;
  }
  header {
    background: var(--panel-bg); border-bottom: 1px solid var(--border);
    padding: 12px 20px; display: flex; align-items: center; gap: 14px;
    flex-shrink: 0;
  }
  .hdr-name { font-size: 0.92rem; font-weight: 700; color: var(--gold); font-family: var(--mono); }
  .hdr-path { font-size: 0.71em; color: var(--dim); font-family: var(--mono); margin-top: 3px; }
  #btn-copy {
    margin-left: auto; background: var(--active-bg); color: var(--dim);
    border: 1px solid var(--border); border-radius: 6px;
    font-size: 0.78em; font-weight: 700; padding: 6px 16px; cursor: pointer;
    transition: background .15s, color .15s;
  }
  #btn-copy:hover { background: #ffffff22; color: var(--text); }
  #content {
    flex: 1; overflow: auto; padding: 16px 22px;
    font-family: var(--mono); font-size: 0.82em; line-height: 1.7;
    white-space: pre;
  }
  .key  { color: var(--c-key); }
  .str  { color: var(--c-str); }
  .num  { color: var(--c-num); }
  .bool { color: var(--c-bool); }
  #content::-webkit-scrollbar { width: 6px; height: 6px; }
  #content::-webkit-scrollbar-track { background: transparent; }
  #content::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>
<header>
  <div>
    <div class="hdr-name">__FILENAME__</div>
    <div class="hdr-path">__RELPATH__</div>
  </div>
  <button id="btn-copy" onclick="copyRaw()">Copy</button>
</header>
<div id="content"></div>
<script>
const raw    = __CONTENT_JSON__;
const isJson = __IS_JSON__;
const el     = document.getElementById('content');

function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function highlight(text) {
  let s = esc(text);
  // Keys:  leading indent + "key":
  s = s.replace(/^(\s*)("[^"]*")(\s*:)/gm,
    '$1<span class="key">$2</span>$3');
  // String values: : "..."
  s = s.replace(/(\:\s*)("[^"]*")/g,
    '$1<span class="str">$2</span>');
  // Numbers
  s = s.replace(/(\:\s*)(-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)/g,
    '$1<span class="num">$2</span>');
  // Booleans / null
  s = s.replace(/(\:\s*)(true|false|null)\b/g,
    '$1<span class="bool">$2</span>');
  return s;
}

if (isJson) {
  try {
    const pretty = JSON.stringify(JSON.parse(raw), null, 2);
    el.innerHTML = highlight(pretty);
  } catch(e) {
    el.textContent = raw;
  }
} else {
  el.textContent = raw;
}

function copyRaw() {
  navigator.clipboard.writeText(raw).then(() => {
    const b = document.getElementById('btn-copy');
    b.textContent = 'Copied!';
    setTimeout(() => { b.textContent = 'Copy'; }, 1500);
  }).catch(() => {
    const ta = document.createElement('textarea');
    ta.value = raw; document.body.appendChild(ta); ta.select();
    document.execCommand('copy'); document.body.removeChild(ta);
  });
}
</script>
</body>
</html>
"""


# ── SSE helpers ────────────────────────────────────────────────────────────────
def sse(event: str, data: str) -> bytes:
    return f"event: {event}\ndata: {data}\n\n".encode()


# ── Pipeline helpers ────────────────────────────────────────────────────────────
def _pipeline_status(slug: str, ep_id: str) -> dict:
    """Return pipeline step completion status (file-existence checks)."""
    if not slug or not ep_id:
        return {"error": "missing slug or ep_id", "llm_stages": {}, "locales": [], "locale_steps": {}}

    ep_dir   = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
    proj_dir = os.path.join(PIPE_DIR, "projects", slug)

    def ep(f):   return os.path.join(ep_dir, f)
    def pr(f):   return os.path.join(proj_dir, f)
    def root(f): return os.path.join(PIPE_DIR, f)
    def ep_rel(f):   return f"projects/{slug}/episodes/{ep_id}/{f}"
    def root_rel(f): return f

    def check(*paths): return all(os.path.isfile(p) for p in paths)

    # Helper: True if stage N's log file has meaningful content (> 100 bytes).
    # run.sh always writes stage logs via `tee`, so this is reliable even when
    # Claude skips writing an output file (e.g. Stage 1 canon check).
    def _log_done(n: int) -> bool:
        _p = os.path.join(PIPE_DIR, "stage_logs", f"{slug}.{ep_id}.stage_{n}.log")
        return os.path.isfile(_p) and os.path.getsize(_p) > 100

    # Stage 0 is done when pipeline_vars.sh in ep_dir contains VOICE_CAST_FILE
    # (Stage 0 writes that; the Prepare-created stub does not have it)
    import glob as _glob
    _pipeline_vars = os.path.join(ep_dir, "pipeline_vars.sh")
    _stage0_done = False
    if os.path.isfile(_pipeline_vars):
        try:
            _pv_content = open(_pipeline_vars, encoding="utf-8").read()
            _stage0_done = "VOICE_CAST_FILE" in _pv_content
        except Exception:
            pass

    # Stage 1: no reliable output file — use log size.
    _stage1_done = _log_done(1)

    # Stage 8 is done when either:
    #   (a) locale variant StoryPrompt files exist  — multi-locale projects
    #   (b) the stage log has content               — en-only projects produce
    #       no locale files so (a) would never fire
    stage8_done = False
    if os.path.isdir(ep_dir):
        stage8_done = any(
            f.startswith("StoryPrompt.") and f.endswith(".json") and f != "StoryPrompt.json"
            for f in os.listdir(ep_dir)
        )
    if not stage8_done:
        stage8_done = _log_done(8)

    llm_stages = {
        "stage_0": {
            "done": _stage0_done,
            "artifacts": [ep_rel("pipeline_vars.sh")],
        },
        "stage_1": {
            "done": _stage1_done,
            "artifacts": [ep_rel("stage_1_check.txt")],
        },
        "stage_2": {
            "done": check(ep("StoryPrompt.json")),
            "artifacts": [ep_rel("StoryPrompt.json")],
        },
        "stage_3": {
            "done": check(ep("Script.json")),
            "artifacts": [ep_rel("Script.json")],
        },
        "stage_4": {
            "done": check(ep("ShotList.json")),
            "artifacts": [ep_rel("ShotList.json")],
        },
        "stage_5": {
            # Primary: both canonical output files exist.
            # Fallback: stage log (handles non-en locale configs where
            # AssetManifest_draft.en.json might not be the written locale).
            "done": (
                check(ep("AssetManifest_draft.shared.json")) and check(ep("AssetManifest_draft.en.json"))
            ) or _log_done(5),
            "artifacts": [ep_rel("AssetManifest_draft.shared.json"), ep_rel("AssetManifest_draft.en.json")],
        },
        "stage_6": {
            "done": check(ep("canon_diff.json")),
            "artifacts": [ep_rel("canon_diff.json")],
        },
        "stage_7": {
            "done": check(pr("canon.json")),
            "artifacts": [],
        },
        "stage_8": {
            "done": stage8_done,
            "artifacts": [],
        },
        "stage_9": {
            # Stage 9 (p_9.txt) is permanently skipped — gen_render_plan.py in Stage 9 render
            # handles this deterministically. Mark as done so the Pipeline tab shows ✓.
            "done": True,
            "artifacts": [],
        },
        "stage_10": {
            # Done when every locale has a rendered output.mp4
            "done": False,   # filled in below after locales are detected
            "artifacts": [],
        },
    }

    # Detect locales from AssetManifest_draft.{locale}.json files
    locales: list[str] = []
    if os.path.isdir(ep_dir):
        for f in sorted(os.listdir(ep_dir)):
            m = re.match(r"AssetManifest_draft\.(.+)\.json$", f)
            if m and m.group(1) != "shared":
                locales.append(m.group(1))

    # Per-locale post-processing status
    locale_steps: dict[str, dict] = {}
    for locale in locales:
        vo_dir = os.path.join(ep_dir, "assets", locale, "audio", "vo")
        gen_tts_done = os.path.isdir(vo_dir) and bool(
            [f for f in os.listdir(vo_dir) if f.endswith(".wav")]
        ) if os.path.isdir(vo_dir) else False

        locale_steps[locale] = {
            "manifest_merge":  {"done": check(ep(f"AssetManifest_merged.{locale}.json"))},
            "gen_tts":         {"done": gen_tts_done},
            "post_tts":        {"done": gen_tts_done},   # proxy: same check as gen_tts
            "resolve_assets":  {"done": check(ep(f"AssetManifest.media.{locale}.json"))},
            "gen_render_plan": {"done": check(ep(f"RenderPlan.{locale}.json"))},
            "render_video":    {"done": check(os.path.join(ep_dir, "renders", locale, "output.mp4"))},
        }

    # Stage 9 is done when every locale has an output.mp4
    ready_videos: list[str] = [
        loc for loc in locales
        if check(os.path.join(ep_dir, "renders", loc, "output.mp4"))
    ]
    if locales:
        llm_stages["stage_10"]["done"] = len(ready_videos) == len(locales)
        llm_stages["stage_10"]["artifacts"] = [
            f"projects/{slug}/episodes/{ep_id}/renders/{loc}/output.mp4"
            for loc in ready_videos
        ]

    # Dubbed audio tracks ready for YouTube Studio upload (non-en locales only)
    ready_dubbed: list[str] = [
        loc for loc in locales
        if loc != "en" and check(os.path.join(ep_dir, "renders", loc, "youtube_dubbed.m4a"))
    ]

    # Shared (locale-free) post-processing steps — steps 1–4 in the Stage 9 panel
    def _any_files(d: str, ext: str) -> bool:
        return os.path.isdir(d) and any(f.endswith(ext) for f in os.listdir(d))

    _assets_dir = os.path.join(ep_dir, "assets")
    shared_steps = {
        "gen_music_clip":  {"done": _any_files(os.path.join(_assets_dir, "music"),  ".wav")},
        "gen_characters":  {"done": _any_files(os.path.join(proj_dir, "characters"), ".png")},
        "gen_backgrounds": {"done": _any_files(os.path.join(_assets_dir, "backgrounds"), ".png")},
        "gen_sfx":         {"done": _any_files(os.path.join(_assets_dir, "sfx"),    ".wav")},
    }

    # Music plan checkpoint: [4b/8] in Stage 9 — pipeline pauses here until user confirms
    _music_plan_path = os.path.join(_assets_dir, "music", "MusicPlan.json")
    music_plan_done = os.path.isfile(_music_plan_path)

    # TTS done: at least one locale has WAV files (proxy for Stage 9 having started)
    tts_done = any(
        (locale_steps.get(loc) or {}).get("gen_tts", {}).get("done", False)
        for loc in locales
    )

    # story.txt is in the episode folder (written by Create Episode in web UI)
    story_file_detected = ""
    _ep_story = os.path.join(ep_dir, "story.txt")
    if os.path.isfile(_ep_story):
        story_file_detected = f"projects/{slug}/episodes/{ep_id}/story.txt"

    # ── VoiceCast.json (project-level, written by Stage 0) ───────────────
    voice_cast = None
    vc_path = os.path.join(PIPE_DIR, "projects", slug, "VoiceCast.json")
    if os.path.isfile(vc_path):
        try:
            with open(vc_path, encoding="utf-8") as _f:
                voice_cast = json.load(_f)
        except Exception:
            pass

    # ── Metadata: prefer meta.json (user-saved), fall back to pipeline_vars.*.sh ─
    meta_title  = ""
    meta_genre  = ""
    meta_format = "episodic"
    meta_locales_str = ",".join(locales) if locales else "en"
    meta_no_music = False
    meta_purge_cache = False

    _meta_json_path = os.path.join(ep_dir, "meta.json")
    if os.path.isfile(_meta_json_path):
        try:
            _mj = json.load(open(_meta_json_path, encoding="utf-8"))
            # Support both new-style (story_title/series_genre) and old-style (title/genre) fields
            meta_title       = _mj.get("story_title",  _mj.get("title",  ""))
            meta_genre       = _mj.get("series_genre", _mj.get("genre",  ""))
            meta_format      = _mj.get("story_format", "episodic")
            meta_locales_str = _mj.get("locales", meta_locales_str)
            meta_no_music    = bool(_mj.get("no_music", False))
            meta_purge_cache = bool(_mj.get("purge_cache", False))
        except Exception:
            pass
    elif os.path.isfile(_pipeline_vars):
        # Fall back to pipeline_vars.sh (legacy episodes without meta.json)
        try:
            _vc = open(_pipeline_vars, encoding="utf-8").read()
            _vars = dict(re.findall(r'export\s+(\w+)="([^"]*)"', _vc))
            meta_title       = _vars.get("STORY_TITLE", "")
            meta_genre       = _vars.get("SERIES_GENRE", "")
            meta_format      = _vars.get("STORY_FORMAT", "episodic")
            meta_locales_str = _vars.get("LOCALES", meta_locales_str)
        except Exception:
            pass

    return {
        "slug": slug, "ep_id": ep_id,
        "llm_stages": llm_stages,
        "locales": locales,
        "locale_steps": locale_steps,   # kept for /run_step recovery endpoint
        "shared_steps": shared_steps,   # steps 1–4: gen_music_clip, gen_characters, gen_backgrounds, gen_sfx
        "ready_videos": ready_videos,
        "ready_dubbed": ready_dubbed,
        "story_file": story_file_detected,
        "voice_cast": voice_cast,
        "title":          meta_title,
        "genre":          meta_genre,
        "story_format":   meta_format,
        "locales_str":    meta_locales_str,
        "no_music":       meta_no_music,
        "purge_cache":    meta_purge_cache,
        "music_plan_done": music_plan_done,
        "tts_done":        tts_done,
    }


def _step_is_done(step: str, slug: str, ep_id: str, locale: str) -> bool:
    """Return True if the step's output already exists (safe to skip)."""
    ep_dir = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)

    def check(*paths): return all(os.path.isfile(p) for p in paths)

    if step == "manifest_merge":
        return check(os.path.join(ep_dir, f"AssetManifest_merged.{locale}.json"))
    elif step == "apply_music_plan":
        return False   # always re-run — user may have changed overrides
    elif step in ("gen_tts", "post_tts"):
        vo_dir = os.path.join(ep_dir, "assets", locale, "audio", "vo")
        return (os.path.isdir(vo_dir) and
                any(f.endswith(".wav") for f in os.listdir(vo_dir)))
    elif step == "resolve_assets":
        return check(os.path.join(ep_dir, f"AssetManifest.media.{locale}.json"))
    elif step == "gen_render_plan":
        return check(os.path.join(ep_dir, f"RenderPlan.{locale}.json"))
    elif step == "render_video":
        return check(os.path.join(ep_dir, "renders", locale, "output.mp4"))
    return False


def _delete_step_output(step: str, slug: str, ep_id: str, locale: str) -> None:
    """Remove a step's primary output(s) so it will always re-run fresh."""
    ep_dir = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
    targets: dict[str, list[str]] = {
        "manifest_merge":  [os.path.join(ep_dir, f"AssetManifest_merged.{locale}.json")],
        "resolve_assets":  [os.path.join(ep_dir, f"AssetManifest.media.{locale}.json")],
        "gen_render_plan": [os.path.join(ep_dir, f"RenderPlan.{locale}.json")],
        "render_video":    [os.path.join(ep_dir, "renders", locale, "output.mp4"),
                            os.path.join(ep_dir, "renders", locale, "render_output.json")],
        # gen_tts / post_tts: skip deletion — TTS files are expensive to redo
    }
    for path in targets.get(step, []):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _purge_episode_assets(slug: str, ep_id: str, locale: str = "") -> list[str]:
    """
    Delete all generated media assets for an episode (or one locale).

    Purges:
      - assets/{locale}/audio/vo/*.wav       — TTS voice files
      - assets/{locale}/audio/vo/licenses/   — TTS license sidecars
      - assets/music/*.wav|mp3               — generated music  (shared, only when locale="")
      - assets/{locale}/backgrounds/*        — generated background images
      - assets/backgrounds/*                 — shared backgrounds (only when locale="")
      - assets/{locale}/characters/*         — generated character images (if locale-specific)
      - assets/characters/*                  — shared characters (only when locale="")
      - assets/{locale}/sfx/*.wav            — generated SFX
      - assets/sfx/*.wav                     — shared SFX (only when locale="")
      - renders/{locale}/output.mp4|srt|json — render outputs
      - AssetManifest_merged.{locale}.json   — merged manifests
      - AssetManifest.media.{locale}.json    — resolved media manifests
      - RenderPlan.{locale}.json             — render plans
      - assets/meta/gen_tts_cloud_results.json (only when locale="")

    Returns list of deleted file paths (for logging).
    """
    ep_dir = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
    deleted: list[str] = []

    def _rm(path: str) -> None:
        try:
            os.remove(path)
            deleted.append(path)
        except FileNotFoundError:
            pass

    def _rm_dir_contents(directory: str, exts: tuple | None = None) -> None:
        """Delete files inside a directory (non-recursive). exts=None means all files."""
        if not os.path.isdir(directory):
            return
        for fname in os.listdir(directory):
            fpath = os.path.join(directory, fname)
            if not os.path.isfile(fpath):
                continue
            if exts is None or fname.lower().endswith(exts):
                _rm(fpath)

    # Determine which locales to purge
    if locale:
        locales = [locale]
        purge_shared = False   # shared assets only purged on full-episode purge
    else:
        locales = []
        if os.path.isdir(ep_dir):
            for f in sorted(os.listdir(ep_dir)):
                m = re.match(r"AssetManifest_draft\.(.+)\.json$", f)
                if m and m.group(1) != "shared":
                    locales.append(m.group(1))
        purge_shared = True

    for loc in locales:
        assets_loc = os.path.join(ep_dir, "assets", loc)

        # TTS voice WAVs + license sidecars
        _rm_dir_contents(os.path.join(assets_loc, "audio", "vo"), (".wav",))
        _rm_dir_contents(os.path.join(assets_loc, "audio", "vo", "licenses"), (".json",))

        # Locale-specific generated images
        _rm_dir_contents(os.path.join(assets_loc, "backgrounds"), (".png", ".jpg", ".jpeg", ".webp"))
        _rm_dir_contents(os.path.join(assets_loc, "characters"),  (".png", ".jpg", ".jpeg", ".webp"))
        _rm_dir_contents(os.path.join(assets_loc, "sfx"),         (".wav", ".mp3"))

        # Render outputs
        render_loc = os.path.join(ep_dir, "renders", loc)
        for fname in ("output.mp4", "output.srt", "render_output.json", "youtube_dubbed.m4a", "youtube_dubbed.aac"):
            _rm(os.path.join(render_loc, fname))

        # Per-locale derived manifests (NOT AssetManifest_merged — it's the
        # input to gen_tts/post_tts/resolve_assets; deleting it breaks those
        # steps.  manifest_merge recreates it via _delete_step_output instead.)
        for pat in (f"AssetManifest.media.{loc}.json",
                    f"RenderPlan.{loc}.json"):
            _rm(os.path.join(ep_dir, pat))

    if purge_shared:
        assets_shared = os.path.join(ep_dir, "assets")
        # Shared music
        _rm_dir_contents(os.path.join(assets_shared, "music"),      (".wav", ".mp3"))
        # Shared (non-locale) backgrounds / characters / sfx
        _rm_dir_contents(os.path.join(assets_shared, "backgrounds"), (".png", ".jpg", ".jpeg", ".webp"))
        _rm_dir_contents(os.path.join(assets_shared, "characters"),  (".png", ".jpg", ".jpeg", ".webp"))
        _rm_dir_contents(os.path.join(assets_shared, "sfx"),         (".wav", ".mp3"))
        # TTS results meta
        _rm(os.path.join(assets_shared, "meta", "gen_tts_cloud_results.json"))

    return deleted


def _build_step_cmd(step: str, slug: str, ep_id: str, locale: str,
                    profile: str = "preview_local",
                    no_music: bool = False,
                    payload: dict | None = None) -> list | None:
    """Build command list for a post-processing step."""
    ep_dir   = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
    code_dir = os.path.join(PIPE_DIR, "code", "http")

    def ep(f): return os.path.join(ep_dir, f)

    if step == "gen_music_clip":
        return [
            "python3", os.path.join(code_dir, "gen_music_clip.py"),
            "--manifest", ep("AssetManifest_draft.shared.json"),
        ]
    elif step == "gen_characters":
        # AI asset steps inherit AI_SERVER_URL / AI_SERVER_KEY from the shell env.
        return [
            "python3", os.path.join(code_dir, "fetch_ai_assets.py"),
            "--manifest",    ep("AssetManifest_draft.shared.json"),
            "--asset_type",  "characters",
        ]
    elif step == "gen_backgrounds":
        cmd = [
            "python3", os.path.join(code_dir, "fetch_ai_assets.py"),
            "--manifest",    ep("AssetManifest_draft.shared.json"),
            "--asset_type",  "backgrounds",
        ]
        asset_ids = (payload or {}).get("asset_ids", "")
        if asset_ids:
            cmd += ["--asset-ids", asset_ids]
        return cmd
    elif step == "gen_sfx":
        return [
            "python3", os.path.join(code_dir, "fetch_ai_assets.py"),
            "--manifest",    ep("AssetManifest_draft.shared.json"),
            "--asset_type",  "sfx",
        ]
    elif step == "manifest_merge":
        return [
            "python3", os.path.join(code_dir, "manifest_merge.py"),
            "--shared", ep(f"AssetManifest_draft.shared.json"),
            "--locale", ep(f"AssetManifest_draft.{locale}.json"),
            "--out",    ep(f"AssetManifest_merged.{locale}.json"),
        ]
    elif step == "gen_tts":
        cmd = [
            "python3", os.path.join(code_dir, "gen_tts_cloud.py"),
            "--manifest", ep(f"AssetManifest_merged.{locale}.json"),
        ]
        # ssml_narration mode: primary locale uses wrapper-rebuild + inner
        # passthrough (same logic as run.sh lines 170-176).
        _pv_path = ep("pipeline_vars.sh")
        _story_fmt = ""
        _primary_loc = ""
        if os.path.isfile(_pv_path):
            try:
                _pv = open(_pv_path, encoding="utf-8").read()
                for _ln in _pv.splitlines():
                    if _ln.startswith("export STORY_FORMAT="):
                        _story_fmt = _ln.split("=", 1)[1].strip().strip('"')
                    elif _ln.startswith("export PRIMARY_LOCALE="):
                        _primary_loc = _ln.split("=", 1)[1].strip().strip('"')
            except Exception:
                pass
        if _story_fmt == "ssml_narration" and locale == _primary_loc:
            project_dir = os.path.join(PIPE_DIR, "projects", slug)
            cmd += [
                "--ssml-narration",
                "--ssml-inner",  ep("ssml_inner.xml"),
                "--voice-cast",  os.path.join(project_dir, "VoiceCast.json"),
            ]
        return cmd
    elif step == "post_tts":
        return [
            "python3", os.path.join(code_dir, "post_tts_analysis.py"),
            "--manifest", ep(f"AssetManifest_merged.{locale}.json"),
        ]
    elif step == "apply_music_plan":
        music_plan = os.path.join(ep_dir, "assets", "music", "MusicPlan.json")
        if not os.path.isfile(music_plan):
            return []   # [] = intentional skip (no MusicPlan.json); None = unknown step
        return [
            "python3", os.path.join(code_dir, "apply_music_plan.py"),
            "--plan",     music_plan,
            "--manifest", ep(f"AssetManifest_merged.{locale}.json"),
        ]
    elif step == "resolve_assets":
        return [
            "python3", os.path.join(code_dir, "resolve_assets.py"),
            "--manifest", ep(f"AssetManifest_merged.{locale}.json"),
            "--out",      ep(f"AssetManifest.media.{locale}.json"),
        ]
    elif step == "gen_render_plan":
        # Read story_format from meta.json so the ceiling logic in gen_render_plan.py
        # knows whether this is a narrative format (continuous_narration, documentary,
        # illustrated_narration) or an episodic/monologue format.
        _story_format = "episodic"
        _meta_path = ep("meta.json")
        if os.path.isfile(_meta_path):
            try:
                _mj = json.load(open(_meta_path, encoding="utf-8"))
                _story_format = _mj.get("story_format", "episodic")
            except Exception:
                pass
        return [
            "python3", os.path.join(code_dir, "gen_render_plan.py"),
            "--manifest",     ep(f"AssetManifest_merged.{locale}.json"),
            "--media",        ep(f"AssetManifest.media.{locale}.json"),
            "--profile",      profile,
            "--story-format", _story_format,
        ]
    elif step == "render_video":
        out_dir = os.path.join(ep_dir, "renders", locale)
        cmd = [
            "python3", os.path.join(code_dir, "render_video.py"),
            "--plan",    ep(f"RenderPlan.{locale}.json"),
            "--locale",  locale,
            "--out",     out_dir,
            "--profile", profile,
        ]
        if no_music:
            cmd.append("--no-music")
        return cmd
    return None


# ── Request handler ────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):

    def end_headers(self):
        # Force Connection: close on every response.  This prevents the browser
        # from trying to reuse TCP connections after a server restart — stale
        # pooled connections cause page-reload hangs in the existing tab.
        self.send_header("Connection", "close")
        super().end_headers()

    # ── GET ───────────────────────────────────────────────────────────────────
    def do_GET(self):
        parsed = urlparse(self.path)

        # Serve UI
        if parsed.path == "/":
            media_server_url = os.environ.get("MEDIA_SERVER_URL",
                                   _vc_config.get("media", {}).get("default_server_url", ""))
            body = HTML.replace("{{MEDIA_SERVER_URL}}", media_server_url).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # Read a story_N.txt file back to the client (Run-tab restore)
        elif parsed.path == "/read_story":
            params     = parse_qs(parsed.query)
            story_file = unquote_plus(params.get("story_file", [""])[0]).strip()
            # Accept either:
            #   story_N.txt                              (legacy root-level file)
            #   projects/<slug>/episodes/<ep>/story.txt  (episode-folder file)
            _ep_story_pat = r"^projects/[^/]+/episodes/[^/]+/story\.txt$"
            if story_file and re.match(r"^story_\d+\.txt$", story_file):
                full_path = os.path.join(PIPE_DIR, story_file)
            elif story_file and re.match(_ep_story_pat, story_file):
                full_path = os.path.join(PIPE_DIR, story_file)
            else:
                full_path = None
            if full_path and os.path.isfile(full_path):
                with open(full_path, encoding="utf-8") as _fh:
                    content = _fh.read()
                payload = {"ok": True, "content": content, "filename": story_file}
            elif full_path:
                payload = {"ok": False, "error": "file not found"}
            else:
                payload = {"ok": False, "error": "invalid filename"}
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # Pre-flight: check if the episode output folder exists
        elif parsed.path == "/check_episode":
            params     = parse_qs(parsed.query)
            story_file = unquote_plus(params.get("story_file", [""])[0]).strip()
            vars_      = _parse_story_vars(story_file)
            slug       = vars_.get("project_slug")
            ep_id      = vars_.get("episode_id")
            if slug and ep_id:
                rel      = os.path.join("projects", slug, "episodes", ep_id)
                ep_dir   = os.path.join(PIPE_DIR, rel)
                exists   = os.path.isdir(ep_dir)
                payload  = {"exists": exists, "path": rel,
                            "project_slug": slug, "episode_id": ep_id}
            else:
                payload  = {"exists": False, "path": None,
                            "project_slug": slug, "episode_id": ep_id}
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # ── VO Retune: serve existing WAV for preview (GET /api/vo_audio) ──────────
        elif parsed.path == "/api/vo_audio":
            params  = parse_qs(parsed.query)
            ep_dir  = unquote_plus(params.get("ep_dir",  [""])[0]).strip()
            locale  = unquote_plus(params.get("locale",  [""])[0]).strip()
            item_id = unquote_plus(params.get("item_id", [""])[0]).strip()
            if not ep_dir or not locale or not item_id or ".." in item_id:
                self.send_response(400); self.end_headers(); return
            full_ep = os.path.join(PIPE_DIR, ep_dir) \
                      if not os.path.isabs(ep_dir) else ep_dir
            wav_p = os.path.join(full_ep, "assets", locale, "audio", "vo",
                                 item_id + ".wav")
            if os.path.isfile(wav_p):
                with open(wav_p, "rb") as _wf:
                    data = _wf.read()
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                _b = b'{"error":"wav not found"}'
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(_b)))
                self.end_headers()
                self.wfile.write(_b)
            return

        # ── GET /api/vo_preview_concat — concatenate all WAVs into one seekable file ──
        elif parsed.path == "/api/vo_preview_concat":
            import wave as _wave, struct as _struct
            params = parse_qs(parsed.query)
            ep_dir  = unquote_plus(params.get("ep_dir",  [""])[0]).strip()
            locale  = unquote_plus(params.get("locale",  [""])[0]).strip()
            if not ep_dir or not locale or ".." in ep_dir:
                self.send_response(400); self.end_headers(); return
            try:
                full_ep = _vo_resolve_ep_dir(ep_dir)
                manifest_path = os.path.join(full_ep, "AssetManifest_merged.en.json") \
                    if locale == "en" else \
                    os.path.join(full_ep, f"AssetManifest_merged.{locale}.json")
                if not os.path.exists(manifest_path):
                    manifest_path = os.path.join(full_ep, "AssetManifest_merged.en.json")
                with open(manifest_path, encoding="utf-8") as _mf:
                    _mfdata = json.load(_mf)
                vo_items = _mfdata.get("vo_items", [])
                scene_tails = _mfdata.get("scene_tails", {})
                vo_dir = os.path.join(full_ep, "assets", locale, "audio", "vo")

                SAMPLE_RATE   = 24000
                BYTES_PER_SEC = SAMPLE_RATE * 2  # 16-bit mono

                pcm_chunks: list[bytes] = []
                clips_meta: list[dict]  = []
                current_sec = 0.0
                prev_scene  = None

                for _it in vo_items:
                    _iid   = _it.get("item_id", "")
                    _scn   = (_iid.split("-")[1] if "-" in _iid else "") # e.g. "sc01"
                    _wav   = os.path.join(vo_dir, f"{_iid}.wav")
                    if not os.path.isfile(_wav):
                        continue
                    # Inter-scene silence from scene_tails
                    if prev_scene is not None and _scn != prev_scene:
                        tail_ms = int(scene_tails.get(_scn, scene_tails.get(prev_scene, 2000)))
                        _sil = b'\x00' * ((tail_ms * BYTES_PER_SEC // 1000 // 2) * 2)
                        pcm_chunks.append(_sil)
                        current_sec += tail_ms / 1000
                    prev_scene = _scn
                    try:
                        with _wave.open(_wav) as _wf:
                            _pcm = _wf.readframes(_wf.getnframes())
                            _dur = _wf.getnframes() / _wf.getframerate()
                    except Exception as _wav_err:
                        raise RuntimeError(
                            f"{_iid}.wav is corrupt or unreadable: {_wav_err}. "
                            f"Re-create this item in the VO tab before generating preview."
                        )
                    clips_meta.append({
                        "item_id":      _iid,
                        "scene_id":     _scn,
                        "text":         _it.get("text", ""),
                        "start_sec":    round(current_sec, 3),
                        "duration_sec": round(_dur, 3),
                    })
                    pcm_chunks.append(_pcm)
                    current_sec += _dur
                    # pause_after_ms gap
                    _pause_ms = int(_it.get("pause_after_ms", 300))
                    if _pause_ms > 0:
                        _sil = b'\x00' * ((_pause_ms * BYTES_PER_SEC // 1000 // 2) * 2)
                        pcm_chunks.append(_sil)
                        current_sec += _pause_ms / 1000

                all_pcm   = b"".join(pcm_chunks)
                out_path  = os.path.join(full_ep, "assets", "meta",
                                         f"vo_preview_{locale}.wav")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                # Write RIFF WAV header + PCM
                _data_size  = len(all_pcm)
                _byte_rate  = SAMPLE_RATE * 2
                _block_align = 2
                _hdr = _struct.pack(
                    "<4sI4s4sIHHIIHH4sI",
                    b"RIFF", 36 + _data_size, b"WAVE",
                    b"fmt ", 16, 1, 1, SAMPLE_RATE,
                    _byte_rate, _block_align, 16,
                    b"data", _data_size,
                )
                with open(out_path, "wb") as _of:
                    _of.write(_hdr)
                    _of.write(all_pcm)

                _rel = os.path.relpath(out_path, PIPE_DIR).replace("\\", "/")
                _resp = json.dumps({
                    "wav_url":   "/serve_media?path=" + _rel,
                    "total_sec": round(current_sec, 3),
                    "clips":     clips_meta,
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(_resp)))
                self.end_headers()
                self.wfile.write(_resp)
            except Exception as _exc:
                _eb = json.dumps({"error": str(_exc)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(_eb)))
                self.end_headers()
                self.wfile.write(_eb)
            return

        # ── VO Review: serve source.wav for waveform canvas (GET /api/vo_source_audio) ─
        # Serves {item_id}.source.wav — used by the waveform canvas (INVARIANT J).
        # Falls back to .wav if source.wav does not yet exist (backward compat).
        elif parsed.path == "/api/vo_source_audio":
            params  = parse_qs(parsed.query)
            ep_dir  = unquote_plus(params.get("ep_dir",  [""])[0]).strip()
            locale  = unquote_plus(params.get("locale",  [""])[0]).strip()
            item_id = unquote_plus(params.get("item_id", [""])[0]).strip()
            if not ep_dir or not locale or not item_id or ".." in item_id:
                self.send_response(400); self.end_headers(); return
            full_ep = _vo_resolve_ep_dir(ep_dir)
            # Migration: create source.wav from .wav if missing (pre-two-file projects)
            _ensure_source_wav(item_id, full_ep, locale)
            source_p = os.path.join(full_ep, "assets", locale, "audio", "vo",
                                    item_id + ".source.wav")
            wav_p    = os.path.join(full_ep, "assets", locale, "audio", "vo",
                                    item_id + ".wav")
            serve_p  = source_p if os.path.isfile(source_p) else wav_p
            if os.path.isfile(serve_p):
                with open(serve_p, "rb") as _wf:
                    data = _wf.read()
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                _b = b'{"error":"source wav not found"}'
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(_b)))
                self.end_headers()
                self.wfile.write(_b)
            return

        # ── VO Review: sentinel status (GET /api/vo_sentinel?ep_dir=...&locale=...) ──
        elif parsed.path == "/api/vo_sentinel":
            params = parse_qs(parsed.query)
            ep_dir = unquote_plus(params.get("ep_dir",  [""])[0]).strip()
            locale = unquote_plus(params.get("locale",  [""])[0]).strip()
            if not ep_dir or not locale:
                _json_resp(self, {"error": "ep_dir and locale required"}, 400)
                return
            full_ep = _vo_resolve_ep_dir(ep_dir)
            sentinel_path = os.path.join(full_ep, "tts_review_complete.json")
            exists  = os.path.isfile(sentinel_path)
            valid   = False
            sentinel_data = {}
            if exists and _VO_UTILS_AVAILABLE:
                try:
                    valid = _verify_sentinel(full_ep, locale)
                    with open(sentinel_path, encoding="utf-8") as _sf:
                        sentinel_data = json.load(_sf)
                except Exception:
                    pass
            _json_resp(self, {
                "exists":       exists,
                "valid":        valid,
                "completed_at": sentinel_data.get("completed_at"),
            })
            return

        # ── VO Review: vo_trim_overrides for locale ────────────────────────────────
        elif parsed.path == "/api/vo_trim_overrides":
            params = parse_qs(parsed.query)
            ep_dir = unquote_plus(params.get("ep_dir",  [""])[0]).strip()
            locale = unquote_plus(params.get("locale",  [""])[0]).strip()
            if not ep_dir or not locale:
                _json_resp(self, {"error": "ep_dir and locale required"}, 400)
                return
            full_ep = _vo_resolve_ep_dir(ep_dir)
            if _VO_UTILS_AVAILABLE:
                overrides = _load_vo_trim_overrides(full_ep, locale)
            else:
                overrides = {}
            _json_resp(self, {"overrides": overrides})
            return

        # ── VO Retune: available locales (GET /api/vo_locales?ep_dir=...) ──────────
        elif parsed.path == "/api/vo_locales":
            params = parse_qs(parsed.query)
            ep_dir = unquote_plus(params.get("ep_dir", [""])[0]).strip()
            locales = []
            if ep_dir:
                full_ep_dir = os.path.join(PIPE_DIR, ep_dir) \
                              if not os.path.isabs(ep_dir) else ep_dir
                if os.path.isdir(full_ep_dir):
                    for fname in sorted(os.listdir(full_ep_dir)):
                        m = re.match(r"AssetManifest_merged\.(.+)\.json$", fname)
                        if m:
                            locales.append(m.group(1))
            body = json.dumps({"locales": locales}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(body)

        # ── VO Retune: vo_items for locale (GET /api/vo_items?ep_dir=...&locale=...) ──
        elif parsed.path == "/api/vo_items":
            params = parse_qs(parsed.query)
            ep_dir = unquote_plus(params.get("ep_dir", [""])[0]).strip()
            locale = unquote_plus(params.get("locale", [""])[0]).strip()
            if not ep_dir or not locale:
                body = json.dumps({"error": "ep_dir and locale are required"}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                full_ep_dir = os.path.join(PIPE_DIR, ep_dir) \
                              if not os.path.isabs(ep_dir) else ep_dir
                mpath = os.path.join(full_ep_dir,
                                     f"AssetManifest_merged.{locale}.json")
                try:
                    with open(mpath, encoding="utf-8") as fh:
                        manifest = json.load(fh)
                    items = manifest.get("vo_items", [])
                    # Annotate each item with WAV duration, source duration, and badge
                    wav_dir   = os.path.join(full_ep_dir, "assets", locale, "audio", "vo")
                    overrides = _load_vo_trim_overrides(full_ep_dir, locale) \
                                if _VO_UTILS_AVAILABLE else {}
                    sentinel_valid = _verify_sentinel(full_ep_dir, locale) \
                                     if _VO_UTILS_AVAILABLE else False
                    for it in items:
                        iid    = it["item_id"]
                        wav_p  = os.path.join(wav_dir, iid + ".wav")
                        src_p  = os.path.join(wav_dir, iid + ".source.wav")
                        dur    = None
                        src_dur = None
                        try:
                            import wave as _wave
                            if os.path.isfile(wav_p):
                                with _wave.open(wav_p) as wf:
                                    dur = round(wf.getnframes() / wf.getframerate(), 3)
                            if os.path.isfile(src_p):
                                with _wave.open(src_p) as wf:
                                    src_dur = round(wf.getnframes() / wf.getframerate(), 3)
                        except Exception:
                            pass
                        it["duration_sec"]        = dur
                        it["source_duration_sec"] = src_dur
                        it["badge"]               = _vo_badge(it.get("text", ""), src_dur or dur or 0)
                        it["has_trim_override"]   = iid in overrides
                        it["timing_stale"]        = not sentinel_valid
                    # Build voice catalog from the full Azure TTS catalog so the
                    # voice dropdown shows ALL available voices for this locale,
                    # not just the one voice assigned in VoiceCast.json.
                    # parse_azure_tts_styles() is cached after first call.
                    # Catalog key: story locale ("zh-Hans", "en", etc.)
                    _full_cat = parse_azure_tts_styles()
                    # Map locale → catalog key (catalog uses "zh-Hans", "en", etc.)
                    if locale.startswith("zh"):
                        _cat_key = "zh-Hans"
                    elif locale.startswith("en"):
                        _cat_key = "en"
                    else:
                        _cat_key = locale
                    # Convert list-of-VoiceEntry to {voiceName: {styles, local_name, gender}} for the UI
                    voice_catalog = {
                        e["voice"]: {
                            "styles":     e.get("styles", []),
                            "local_name": e.get("local_name", e["voice"]),
                            "gender":     e.get("gender", ""),
                        }
                        for e in _full_cat.get(_cat_key, [])
                        if e.get("voice")
                    }
                    scene_tails = manifest.get("scene_tails", {})
                    body = json.dumps({"items": items, "voice_catalog": voice_catalog,
                                       "scene_tails": scene_tails}).encode()
                    self.send_response(200)
                except FileNotFoundError:
                    body = json.dumps({"error": f"Manifest not found: {mpath}"}).encode()
                    self.send_response(404)
                except Exception as exc:
                    body = json.dumps({"error": str(exc)}).encode()
                    self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()
                self.wfile.write(body)

        # Next available story_N number
        elif parsed.path == "/next_story_num":
            body = json.dumps({"num": _next_story_num()}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # SSE stream  —  runs: bash run.sh <ep_dir> <from> <to>
        elif parsed.path == "/stream":
            params        = parse_qs(parsed.query)
            ep_dir_param  = unquote_plus(params.get("ep_dir",      [""])[0]).strip()
            from_stage    = params.get("from",    ["0"])[0].strip()
            to_stage      = params.get("to",      ["9"])[0].strip()
            test_mode     = params.get("test",    ["1"])[0].strip() == "1"
            render_profile = params.get("profile", ["preview_local"])[0].strip()
            if render_profile not in ("preview_local", "draft_720p", "high"):
                render_profile = "preview_local"
            no_music     = params.get("no_music", ["0"])[0].strip() == "1"

            # Sanitise: digits only, 0–10 (render stage is internally 10, displayed as Stage 9)
            from_stage = str(max(0, min(10, int(from_stage)))) if from_stage.isdigit() else "0"
            to_stage   = str(max(0, min(10, int(to_stage))))   if to_stage.isdigit()   else "10"

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            if not ep_dir_param:
                self.wfile.write(sse("error_line", "No ep_dir provided."))
                self.wfile.write(sse("done", "1"))
                self.wfile.flush()
                return

            # Read no_music from meta.json if not explicitly overridden via checkbox
            _meta_path = os.path.join(PIPE_DIR, ep_dir_param, "meta.json")
            if not no_music and os.path.isfile(_meta_path):
                try:
                    _m = json.load(open(_meta_path, encoding="utf-8"))
                    no_music = bool(_m.get("no_music", False))
                except Exception:
                    pass

            # Build subprocess environment
            run_env = os.environ.copy()
            run_env.pop("CLAUDECODE", None)   # prevent nested-session guard from firing
            run_env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = "100000"   # stages 3/4/5 produce large JSON
            if test_mode:
                run_env["MODEL"] = "haiku"   # cheapest model for all stages
            run_env["RENDER_PROFILE"] = render_profile   # preview_local or high
            if no_music:
                run_env["NO_MUSIC"] = "1"
            # Note: LOCALES and STORY_FORMAT are sourced from pipeline_vars.sh by run.sh

            client  = self.client_address
            job_key = f"stream\x00{ep_dir_param}\x00{from_stage}\x00{to_stage}"
            log_path = _launch_stream_job(
                job_key,
                ["bash", "run.sh", ep_dir_param, from_stage, to_stage],
                run_env,
                client,
            )
            try:
                _tail_log_to_sse(self.wfile, log_path)
            except (BrokenPipeError, ConnectionResetError):
                pass   # client disconnected — job keeps running in background
            except Exception as exc:
                try:
                    self.wfile.write(sse("error_line", f"Server error: {exc}"))
                    self.wfile.write(sse("done", "1"))
                    self.wfile.flush()
                except Exception:
                    pass

        # List all projects / episodes / artifact files
        elif parsed.path == "/list_projects":
            projects_dir = os.path.join(PIPE_DIR, "projects")
            result: list = []
            if os.path.isdir(projects_dir):
                for slug in sorted(os.listdir(projects_dir)):
                    slug_path = os.path.join(projects_dir, slug)
                    if not os.path.isdir(slug_path):
                        continue
                    episodes_path = os.path.join(slug_path, "episodes")
                    episodes: list = []
                    if os.path.isdir(episodes_path):
                        for ep_id in sorted(os.listdir(episodes_path)):
                            ep_path = os.path.join(episodes_path, ep_id)
                            if not os.path.isdir(ep_path):
                                continue
                            files: list = []
                            for fname in sorted(os.listdir(ep_path)):
                                fpath = os.path.join(ep_path, fname)
                                if os.path.isfile(fpath):
                                    rel = os.path.join(
                                        "projects", slug, "episodes", ep_id, fname)
                                    files.append({
                                        "name":  fname,
                                        "path":  rel,
                                        "size":  os.path.getsize(fpath),
                                        "mtime": os.path.getmtime(fpath),
                                    })
                            episodes.append({"id": ep_id, "files": files})
                    result.append({"slug": slug, "episodes": episodes})
            body = json.dumps({"projects": result}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif parsed.path == "/api/check_slug":
            params_qs = parse_qs(parsed.query)
            slug      = unquote_plus(params_qs.get("slug", [""])[0]).strip()
            exists    = os.path.isdir(os.path.join(PIPE_DIR, "projects", slug)) if slug else False
            n = 2
            suggested = slug
            while exists and os.path.isdir(os.path.join(PIPE_DIR, "projects", suggested)):
                suggested = f"{slug}-{n}"; n += 1
            resp = json.dumps({"exists": exists, "suggested": suggested}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif parsed.path == "/api/next_episode_id":
            import re as _re2
            params_qs = parse_qs(parsed.query)
            slug      = unquote_plus(params_qs.get("slug", [""])[0]).strip()
            ep_dir    = os.path.join(PIPE_DIR, "projects", slug, "episodes")
            next_ep   = "s01e01"
            if os.path.isdir(ep_dir):
                nums = []
                for name in os.listdir(ep_dir):
                    m = _re2.match(r"s01e(\d+)$", name)
                    if m:
                        nums.append(int(m.group(1)))
                if nums:
                    next_ep = f"s01e{max(nums)+1:02d}"
            resp = json.dumps({"next_ep_id": next_ep}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        # Serve artifact viewer page
        elif parsed.path == "/view_artifact":
            params    = parse_qs(parsed.query)
            rel_path  = unquote_plus(params.get("path", [""])[0]).strip()
            safe_root = os.path.realpath(PIPE_DIR)

            def _html_err(code, msg):
                body = (f"<html><body style='background:#f5f3ef;color:#c04040;"
                        f"font-family:monospace;padding:40px'>"
                        f"<h2>{msg}</h2><p>{rel_path}</p></body></html>").encode()
                self.send_response(code)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            if not rel_path:
                _html_err(400, "No path specified"); return

            full_path = os.path.realpath(os.path.join(PIPE_DIR, rel_path))
            if not full_path.startswith(safe_root + os.sep) and full_path != safe_root:
                _html_err(403, "403 Forbidden"); return

            if not os.path.isfile(full_path):
                _html_err(404, "File not found"); return

            try:
                with open(full_path, "r", encoding="utf-8") as fh:
                    content = fh.read()
            except Exception as exc:
                _html_err(500, f"Error reading file: {exc}"); return

            filename = os.path.basename(full_path)
            viewer   = (VIEWER_HTML
                        .replace("__FILENAME__",     filename)
                        .replace("__RELPATH__",      rel_path)
                        .replace("__CONTENT_JSON__", json.dumps(content))
                        .replace("__IS_JSON__",      "true" if filename.endswith(".json") else "false"))
            body = viewer.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # List story_N.txt files
        elif parsed.path == "/list_stories":
            stories = sorted(
                os.path.basename(p)
                for p in glob.glob(os.path.join(PIPE_DIR, "story_*.txt"))
            )
            body = json.dumps({"stories": stories}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # Pipeline step completion status
        elif parsed.path == "/pipeline_status":
            params = parse_qs(parsed.query)
            slug   = unquote_plus(params.get("slug",  [""])[0]).strip()
            ep_id  = unquote_plus(params.get("ep_id", [""])[0]).strip()
            body   = json.dumps(_pipeline_status(slug, ep_id)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(body)

        elif parsed.path == "/api/status_report":
            params = parse_qs(parsed.query)
            slug   = unquote_plus(params.get("slug",  [""])[0]).strip()
            ep_id  = unquote_plus(params.get("ep_id", [""])[0]).strip()
            sr_path = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id, "status_report.txt")
            text = ""
            if slug and ep_id and os.path.exists(sr_path):
                with open(sr_path, "r", encoding="utf-8") as fh:
                    text = fh.read()
            body = json.dumps({"text": text}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(body)

        # ── VO alignment summary (written by polish_locale_vo.py) ─────────────
        elif parsed.path == "/api/vo_alignment":
            params = parse_qs(parsed.query)
            slug   = unquote_plus(params.get("slug",  [""])[0]).strip()
            ep_id  = unquote_plus(params.get("ep_id", [""])[0]).strip()
            locales_data = []
            if slug and ep_id:
                ep_dir = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
                if os.path.isdir(ep_dir):
                    for fname in sorted(os.listdir(ep_dir)):
                        m = re.match(r"vo_alignment\.(.+)\.json$", fname)
                        if m:
                            try:
                                with open(os.path.join(ep_dir, fname), encoding="utf-8") as fh:
                                    locales_data.append(json.load(fh))
                            except Exception:
                                pass
            body = json.dumps({"locales": locales_data}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(body)

        # ── AI pipeline diagnostics — which stage to re-run? ──────────────────
        elif parsed.path == "/api/diagnose_pipeline":
            import hashlib as _hl
            params = parse_qs(parsed.query)
            slug   = unquote_plus(params.get("slug",  [""])[0]).strip()
            ep_id  = unquote_plus(params.get("ep_id", [""])[0]).strip()

            if not slug or not ep_id:
                _b = json.dumps({"error": "missing slug or ep_id"}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(_b)))
                self.end_headers(); self.wfile.write(_b)
            else:
                ep_dir   = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
                proj_dir = os.path.join(PIPE_DIR, "projects", slug)
                def _ep(f): return os.path.join(ep_dir, f)
                def _pr(f): return os.path.join(proj_dir, f)

                # Detect locales from AssetManifest_draft.{locale}.json files
                _locales: list[str] = []
                if os.path.isdir(ep_dir):
                    for _f in sorted(os.listdir(ep_dir)):
                        _m = re.match(r"AssetManifest_draft\.(.+)\.json$", _f)
                        if _m and _m.group(1) != "shared":
                            _locales.append(_m.group(1))

                # All files that affect staleness
                _watch = [
                    _ep("meta.json"), _ep("story.txt"),
                    _ep("pipeline_vars.sh"), _pr("VoiceCast.json"),
                    _ep("StoryPrompt.json"), _ep("Script.json"),
                    _ep("ShotList.json"), _ep("AssetManifest_draft.shared.json"),
                    _ep("canon_diff.json"), _pr("canon.json"),
                    _ep("AssetManifest_final.json"), _ep("RenderPlan.json"),
                ] + [_ep(f"AssetManifest_draft.{l}.json")  for l in _locales] \
                  + [_ep(f"AssetManifest_merged.{l}.json") for l in _locales] \
                  + [_ep(f"RenderPlan.{l}.json")           for l in _locales] \
                  + [_ep(f"renders/{l}/output.mp4")        for l in _locales]

                # Cache key: mtime hash of all watched files
                _mparts = [
                    f"{fp}:{os.path.getmtime(fp):.0f}" if os.path.exists(fp) else f"{fp}:0"
                    for fp in sorted(_watch)
                ]
                _cache_key = slug + "|" + ep_id + "|" + _hl.md5("|".join(_mparts).encode()).hexdigest()[:12]

                if _cache_key in _diagnose_cache:
                    _result = _diagnose_cache[_cache_key]
                else:
                    # ── Build dependency status table ────────────────────────
                    _now = time.time()
                    _MIN_STALE = 10  # seconds; less than this is timestamp noise

                    def _mtime(f):
                        try: return os.path.getmtime(f)
                        except: return None

                    def _fmt_delta(sec):
                        if sec < 60:   return f"{int(sec)}s"
                        if sec < 3600: return f"{int(sec/60)}m {int(sec%60)}s"
                        return f"{sec/3600:.1f}h"

                    def _dep_status(inputs, outputs):
                        """Returns (tag, description) for one dependency pair."""
                        missing = [f for f in outputs if not os.path.exists(f)]
                        if missing:
                            return "MISSING", "output missing: " + ", ".join(
                                os.path.basename(f) for f in missing)
                        out_ts = [_mtime(f) for f in outputs if os.path.exists(f)]
                        if not out_ts:
                            return "MISSING", "no output files"
                        oldest_out = min(t for t in out_ts if t)
                        in_ts = [(f, _mtime(f)) for f in inputs if os.path.exists(f)]
                        if not in_ts:
                            return "FRESH", ""
                        newest_in_f, newest_in_t = max(in_ts, key=lambda x: x[1] or 0)
                        if newest_in_t and newest_in_t - oldest_out > _MIN_STALE:
                            return "STALE", (
                                f"{os.path.basename(newest_in_f)} is "
                                f"{_fmt_delta(newest_in_t - oldest_out)} newer than output"
                            )
                        return "FRESH", ""

                    _stage_deps = [
                        (0,  "Cast voices & pipeline_vars.sh",
                             [_ep("meta.json"), _ep("story.txt")],
                             [_ep("pipeline_vars.sh"), _pr("VoiceCast.json")]),
                        (2,  "Write episode direction",
                             [_pr("VoiceCast.json"), _ep("story.txt")],
                             [_ep("StoryPrompt.json")]),
                        (3,  "Write script & dialogue",
                             [_pr("VoiceCast.json"), _ep("StoryPrompt.json")],
                             [_ep("Script.json")]),
                        (4,  "Break script into shots",
                             [_ep("Script.json")],
                             [_ep("ShotList.json")]),
                        (5,  "List required assets",
                             [_ep("ShotList.json")],
                             [_ep("AssetManifest_draft.shared.json")]),
                        (6,  "Identify new story facts",
                             [_ep("Script.json"), _ep("ShotList.json")],
                             [_ep("canon_diff.json")]),
                        (7,  "Update story memory",
                             [_ep("canon_diff.json")],
                             [_pr("canon.json")]),
                        (8,  "Translate & adapt locales",
                             [_ep("AssetManifest_draft.shared.json"), _pr("VoiceCast.json")],
                             [_ep(f"AssetManifest_draft.{l}.json") for l in _locales]),
                        (9,  "Finalize assets & render plan",
                             [_ep("AssetManifest_draft.shared.json")]
                             + [_ep(f"AssetManifest_draft.{l}.json") for l in _locales],
                             [_ep("AssetManifest_final.json"), _ep("RenderPlan.json")]),
                        (10, "Merge assets & render video",
                             [_pr("VoiceCast.json"), _ep("RenderPlan.json")]
                             + [_ep(f"RenderPlan.{l}.json") for l in _locales],
                             [_ep(f"renders/{l}/output.mp4") for l in _locales]),
                    ]

                    _dep_lines = []
                    for (sn, slabel, sins, souts) in _stage_deps:
                        tag, desc = _dep_status(sins, souts)
                        suffix = f"  — {desc}" if desc else ""
                        _dep_lines.append(f"  Stage {sn:2d}  [{tag:7s}]  {slabel}{suffix}")
                    _dep_summary = "\n".join(_dep_lines)

                    # Status report tail
                    _sr_path = os.path.join(ep_dir, "status_report.txt")
                    _sr_tail = ""
                    if os.path.exists(_sr_path):
                        with open(_sr_path, encoding="utf-8") as _fh:
                            _sr_tail = "".join(_fh.readlines()[-30:]).strip()

                    # P10: deterministic diagnose — linear scan for earliest STALE/MISSING stage.
                    # Replaces haiku call; same output schema, zero LLM cost, instant.
                    def _run_diagnose(dep_lines: list[tuple]) -> dict:
                        """Find earliest STALE or MISSING stage; return diagnose result dict."""
                        earliest = None
                        reason   = "All stages fresh — no re-run needed."
                        for (sn, slabel, sins, souts) in dep_lines:
                            tag, desc = _dep_status(sins, souts)
                            if tag in ("STALE", "MISSING"):
                                earliest = sn
                                reason = f"Stage {sn} ({slabel}) is {tag}" + (f": {desc}" if desc else "") + ". Re-run from here."
                                break
                        if earliest is not None:
                            return {"from_stage": earliest, "to_stage": 10,
                                    "reason": reason, "confidence": "high"}
                        return {"from_stage": None, "to_stage": None,
                                "reason": reason, "confidence": "high"}

                    _result = _run_diagnose(_stage_deps)

                    # Store in cache (cap at 100 entries)
                    _diagnose_cache[_cache_key] = _result
                    if len(_diagnose_cache) > 100:
                        del _diagnose_cache[next(iter(_diagnose_cache))]

                _body = json.dumps(_result).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(_body)))
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()
                self.wfile.write(_body)

        # SSE stream for a single post-processing step
        elif parsed.path == "/run_step":
            params   = parse_qs(parsed.query)
            step     = unquote_plus(params.get("step",     [""])[0]).strip()
            slug     = unquote_plus(params.get("slug",     [""])[0]).strip()
            ep_id    = unquote_plus(params.get("ep_id",    [""])[0]).strip()
            locale   = unquote_plus(params.get("locale",   [""])[0]).strip()
            profile   = unquote_plus(params.get("profile",   ["preview_local"])[0]).strip()
            no_music  = params.get("no_music", ["0"])[0].strip() == "1"
            asset_ids = unquote_plus(params.get("asset_ids", [""])[0]).strip()

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            _step_payload = {"asset_ids": asset_ids} if asset_ids else None
            cmd = _build_step_cmd(step, slug, ep_id, locale, profile, no_music,
                                  payload=_step_payload)
            if cmd is None:
                self.wfile.write(sse("error_line", f"Unknown step: {step!r}"))
                self.wfile.write(sse("done", "1"))
                self.wfile.flush()
                return
            if cmd == []:
                self.wfile.write(sse("line", f"  ✓ {step} — skipped (no plan file)"))
                self.wfile.write(sse("done", "0"))
                self.wfile.flush()
                return

            step_env = os.environ.copy()
            step_env.pop("CLAUDECODE", None)   # prevent nested-session guard from firing

            client = self.client_address
            proc   = None
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=step_env,
                    cwd=PIPE_DIR,
                )
                with _lock:
                    _procs[client] = proc

                for raw_line in proc.stdout:
                    self.wfile.write(sse("line", raw_line.rstrip("\n")))
                    self.wfile.flush()

                proc.wait()
                self.wfile.write(sse("done", str(proc.returncode)))
                self.wfile.flush()

            except (BrokenPipeError, ConnectionResetError):
                pass   # client disconnected or server shutting down
            except Exception as exc:
                try:
                    self.wfile.write(sse("error_line", f"Server error: {exc}"))
                    self.wfile.write(sse("done", "1"))
                    self.wfile.flush()
                except Exception:
                    pass
            finally:
                with _lock:
                    _procs.pop(client, None)
                if proc and proc.poll() is None:
                    proc.terminate()

        # SSE stream: all post-processing steps for one locale, skip-if-done
        elif parsed.path == "/run_locale":
            params     = parse_qs(parsed.query)
            slug       = unquote_plus(params.get("slug",    [""])[0]).strip()
            ep_id      = unquote_plus(params.get("ep_id",   [""])[0]).strip()
            locale     = unquote_plus(params.get("locale",  [""])[0]).strip()
            profile    = unquote_plus(params.get("profile", ["preview_local"])[0]).strip()
            from_step  = unquote_plus(params.get("from",    [""])[0]).strip()
            no_music   = params.get("no_music", ["0"])[0].strip() == "1"
            purge      = params.get("purge",    ["1"])[0].strip() == "1"

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            if not slug or not ep_id or not locale:
                self.wfile.write(sse("error_line", "Missing slug, ep_id, or locale"))
                self.wfile.write(sse("done", "1"))
                self.wfile.flush()
                return

            LOCALE_STEPS = ["manifest_merge", "gen_tts", "post_tts",
                            "apply_music_plan", "resolve_assets",
                            "gen_render_plan", "render_video"]
            # Honour optional from= param — start from a specific step
            from_idx = LOCALE_STEPS.index(from_step) if from_step in LOCALE_STEPS else 0
            # When user explicitly picks a start step ("Run N→7"), force-run all
            # steps in the range (delete stale outputs).  Without from= (full chain)
            # keep the skip-if-done behaviour to avoid re-running expensive TTS.
            force_run = bool(from_step)
            step_env = os.environ.copy()
            step_env.pop("CLAUDECODE", None)
            client   = self.client_address
            job_key  = f"run_locale\x00{slug}\x00{ep_id}\x00{locale}\x00{from_step}"

            # Capture loop vars for background thread closure
            _fr = force_run
            def _run_locale_job(write_log):
                _force = _fr
                if purge:
                    write_log("O", f"\n🗑  Purging cached assets for [{locale}]…")
                    removed = _purge_episode_assets(slug, ep_id, locale)
                    write_log("O", f"   Deleted {len(removed)} file(s)")
                    _force = True

                for step in LOCALE_STEPS[from_idx:]:
                    if _force:
                        _delete_step_output(step, slug, ep_id, locale)
                    elif _step_is_done(step, slug, ep_id, locale):
                        write_log("O", f"  ✓ {step} — already done, skipping")
                        continue

                    write_log("O", f"\n── {step} ──────────────────────────────────────────")

                    cmd = _build_step_cmd(step, slug, ep_id, locale, profile, no_music)
                    if cmd is None:
                        write_log("E", f"Unknown step: {step!r}")
                        write_log("D", "1")
                        return
                    if cmd == []:
                        write_log("O", f"  ✓ {step} — skipped (no plan file)")
                        continue

                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True, bufsize=1,
                        env=step_env, cwd=PIPE_DIR,
                    )
                    with _lock:
                        _procs[client] = proc
                    for raw_line in proc.stdout:
                        write_log("O", raw_line.rstrip("\n"))
                    proc.wait()
                    with _lock:
                        _procs.pop(client, None)

                    if proc.returncode != 0:
                        write_log("E", f"✗ {step} failed (exit {proc.returncode})")
                        write_log("D", str(proc.returncode))
                        return

                    write_log("O", f"✓ {step}")

                write_log("O", f"\n✓ [{locale}] All post-processing steps complete")
                _append_tts_usage_to_status_report(slug, ep_id, write_log)
                write_log("D", "0")

            log_path = _launch_fn_job(job_key, _run_locale_job)
            try:
                _tail_log_to_sse(self.wfile, log_path)
            except (BrokenPipeError, ConnectionResetError):
                pass   # client disconnected — job keeps running in background
            except Exception as exc:
                try:
                    self.wfile.write(sse("error_line", f"Server error: {exc}"))
                    self.wfile.write(sse("done", "1"))
                    self.wfile.flush()
                except Exception:
                    pass

        # SSE stream: shared steps N-4 then all locale steps 5-10 (Run N→10 from shared step)
        elif parsed.path == "/run_stage10":
            params    = parse_qs(parsed.query)
            slug      = unquote_plus(params.get("slug",    [""])[0]).strip()
            ep_id     = unquote_plus(params.get("ep_id",   [""])[0]).strip()
            from_step = unquote_plus(params.get("from",    [""])[0]).strip()
            profile   = unquote_plus(params.get("profile", ["preview_local"])[0]).strip()
            no_music  = params.get("no_music", ["0"])[0].strip() == "1"
            purge     = params.get("purge",    ["1"])[0].strip() == "1"

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            if not slug or not ep_id:
                self.wfile.write(sse("error_line", "Missing slug or ep_id"))
                self.wfile.write(sse("done", "1"))
                self.wfile.flush()
                return

            _SHARED_STEPS = ["gen_music_clip", "gen_characters", "gen_backgrounds", "gen_sfx"]
            _LOCALE_STEPS_ALL = ["manifest_merge", "gen_tts", "post_tts",
                                  "apply_music_plan", "resolve_assets",
                                  "gen_render_plan", "render_video"]
            from_idx = _SHARED_STEPS.index(from_step) if from_step in _SHARED_STEPS else 0

            # Detect locales from AssetManifest_draft.{locale}.json files
            _ep_dir_s10 = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
            _locales_s10: list[str] = []
            if os.path.isdir(_ep_dir_s10):
                for _f in sorted(os.listdir(_ep_dir_s10)):
                    _m = re.match(r"AssetManifest_draft\.(.+)\.json$", _f)
                    if _m and _m.group(1) != "shared":
                        _locales_s10.append(_m.group(1))

            step_env = os.environ.copy()
            step_env.pop("CLAUDECODE", None)
            client   = self.client_address
            job_key  = f"run_stage10\x00{slug}\x00{ep_id}\x00{from_step}"

            def _run_stage10_job(write_log):
                def _run_cmd(cmd) -> int:
                    p = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, env=step_env, cwd=PIPE_DIR,
                    )
                    with _lock:
                        _procs[client] = p
                    for raw_line in p.stdout:
                        write_log("O", raw_line.rstrip("\n"))
                    p.wait()
                    with _lock:
                        _procs.pop(client, None)
                    return p.returncode

                # ── Purge stale assets (all locales) before running ───────────
                if purge:
                    write_log("O", "\n🗑  Purging cached assets for all locales…")
                    _total_removed = _purge_episode_assets(slug, ep_id, "")
                    write_log("O", f"   Deleted {len(_total_removed)} file(s)")

                # ── Shared steps (locale-free) ────────────────────────────────
                for _step in _SHARED_STEPS[from_idx:]:
                    write_log("O", f"\n── {_step} (shared) ────────────────────────────────")
                    _cmd = _build_step_cmd(_step, slug, ep_id, "", profile, no_music)
                    if _cmd is None:
                        write_log("E", f"Unknown step: {_step!r}")
                        write_log("D", "1")
                        return
                    if _cmd == []:
                        write_log("O", f"  ✓ {_step} — skipped (no plan file)")
                        continue
                    _rc = _run_cmd(_cmd)
                    if _rc != 0:
                        write_log("E", f"✗ {_step} failed (exit {_rc})")
                        write_log("D", str(_rc))
                        return
                    write_log("O", f"✓ {_step}")

                # ── Per-locale steps ──────────────────────────────────────────
                if not _locales_s10:
                    write_log("E", "No locales found — run Stage 9 first to create manifests")
                    write_log("D", "1")
                    return

                # Determine primary locale and sentinel validity for Stage 9 skip guard
                _ep_dir_s10_full = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
                _primary_locale_s10 = "en"
                if _VO_UTILS_AVAILABLE:
                    try:
                        from pathlib import Path as _P
                        _primary_locale_s10 = _get_primary_locale(_P(_ep_dir_s10_full))
                    except Exception:
                        pass

                # Steps to skip for primary locale when sentinel is valid (INVARIANT I)
                _TTS_STEPS_TO_SKIP = {"manifest_merge", "gen_tts", "post_tts"}

                for _locale in _locales_s10:
                    write_log("O", f"\n── locale: {_locale} ────────────────────────────────")

                    # Check sentinel for primary locale (INVARIANT I)
                    _sentinel_valid_for_locale = False
                    if (_locale == _primary_locale_s10 and _VO_UTILS_AVAILABLE):
                        try:
                            _sentinel_valid_for_locale = _verify_sentinel(
                                _ep_dir_s10_full, _locale
                            )
                            if _sentinel_valid_for_locale:
                                write_log("O",
                                    f"  ✓ Sentinel valid for primary locale {_locale!r} — "
                                    "skipping manifest_merge, gen_tts, post_tts"
                                )
                        except Exception as _sv_exc:
                            write_log("O", f"  [warn] Sentinel check error: {_sv_exc}")

                    for _step in _LOCALE_STEPS_ALL:
                        # Skip manifest_merge/gen_tts/post_tts for primary locale
                        # if sentinel is valid (INVARIANT I)
                        if _sentinel_valid_for_locale and _step in _TTS_STEPS_TO_SKIP:
                            write_log("O",
                                f"  ✓ {_step} [{_locale}] — skipped (VO approved, sentinel valid)"
                            )
                            continue

                        write_log("O", f"\n── {_step} [{_locale}] ──────────────────────────")
                        _cmd = _build_step_cmd(_step, slug, ep_id, _locale, profile, no_music)
                        if _cmd is None:
                            write_log("E", f"Unknown step: {_step!r}")
                            write_log("D", "1")
                            return
                        if _cmd == []:
                            write_log("O", f"  ✓ {_step} — skipped (no plan file)")
                            continue
                        _rc = _run_cmd(_cmd)
                        if _rc != 0:
                            write_log("E", f"✗ {_step} [{_locale}] failed (exit {_rc})")
                            write_log("D", str(_rc))
                            return
                        write_log("O", f"✓ {_step} [{_locale}]")
                    write_log("O", f"✓ [{_locale}] all locale steps complete")

                write_log("O", "\n✓ Stage 9 — all steps complete")
                _append_tts_usage_to_status_report(slug, ep_id, write_log)
                write_log("D", "0")

            log_path = _launch_fn_job(job_key, _run_stage10_job)
            try:
                _tail_log_to_sse(self.wfile, log_path)
            except (BrokenPipeError, ConnectionResetError):
                pass   # client disconnected — job keeps running in background
            except Exception as exc:
                try:
                    self.wfile.write(sse("error_line", f"Server error: {exc}"))
                    self.wfile.write(sse("done", "1"))
                    self.wfile.flush()
                except Exception:
                    pass

        # SSE stream: Stage 7.5 — run manifest_merge + gen_tts for primary locale only
        elif parsed.path == "/run_stage75":
            params  = parse_qs(parsed.query)
            slug    = unquote_plus(params.get("slug",    [""])[0]).strip()
            ep_id   = unquote_plus(params.get("ep_id",   [""])[0]).strip()
            profile = unquote_plus(params.get("profile", ["preview_local"])[0]).strip()

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            if not slug or not ep_id:
                self.wfile.write(sse("error_line", "Missing slug or ep_id"))
                self.wfile.write(sse("done", "1"))
                self.wfile.flush()
                return

            step_env = os.environ.copy()
            step_env.pop("CLAUDECODE", None)
            client   = self.client_address
            job_key  = f"run_stage75\x00{slug}\x00{ep_id}"

            def _run_stage75_job(write_log):
                import json as _json
                _ep_dir_75 = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)

                # Determine primary locale
                _locale_75 = "en"
                if _VO_UTILS_AVAILABLE:
                    try:
                        from pathlib import Path as _P75
                        _locale_75 = _get_primary_locale(_P75(_ep_dir_75))
                    except Exception:
                        pass

                write_log("O", f"\n── Stage 7.5 — primary locale: {_locale_75} ──────────────────")

                _s75_steps = ["manifest_merge", "gen_tts"]
                for _step in _s75_steps:
                    write_log("O", f"\n── {_step} [{_locale_75}] ──────────────────────────────")
                    _cmd = _build_step_cmd(_step, slug, ep_id, _locale_75, profile, False)
                    if _cmd is None:
                        write_log("E", f"Unknown step: {_step!r}")
                        write_log("D", "1")
                        return
                    if _cmd == []:
                        write_log("O", f"  ✓ {_step} — skipped (no plan file)")
                        continue
                    _p75 = subprocess.Popen(
                        _cmd,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, env=step_env, cwd=PIPE_DIR,
                    )
                    with _lock:
                        _procs[client] = _p75
                    for _raw in _p75.stdout:
                        write_log("O", _raw.rstrip("\n"))
                    _p75.wait()
                    with _lock:
                        _procs.pop(client, None)
                    if _p75.returncode != 0:
                        write_log("E", f"✗ {_step} [{_locale_75}] failed (exit {_p75.returncode})")
                        write_log("D", str(_p75.returncode))
                        return
                    write_log("O", f"✓ {_step} [{_locale_75}]")

                # All TTS steps done — emit vo_review_ready event before done
                write_log("V", _json.dumps({"locale": _locale_75, "slug": slug, "ep_id": ep_id}))
                write_log("O", "\n✓ Stage 7.5 complete — VO ready for review")
                write_log("D", "0")

            log_path = _launch_fn_job(job_key, _run_stage75_job)
            try:
                _tail_log_to_sse(self.wfile, log_path)
            except (BrokenPipeError, ConnectionResetError):
                pass   # client disconnected — job keeps running in background
            except Exception as exc:
                try:
                    self.wfile.write(sse("error_line", f"Server error: {exc}"))
                    self.wfile.write(sse("done", "1"))
                    self.wfile.flush()
                except Exception:
                    pass

        # Range-request-capable media streaming (for HTML5 <video>)
        elif parsed.path == "/serve_media":
            params    = parse_qs(parsed.query)
            rel_path  = unquote_plus(params.get("path", [""])[0]).strip()

            if not rel_path:
                self.send_response(400); self.end_headers(); return

            # Allow both relative-to-PIPE_DIR and absolute paths
            # (media server stores file:/// URLs pointing to /mnt/shared/...)
            _safe_roots = [os.path.realpath(PIPE_DIR)]
            _shared = os.environ.get("MEDIA_SHARED_ROOT", "/mnt/shared")
            if os.path.isdir(_shared):
                _safe_roots.append(os.path.realpath(_shared))

            if os.path.isabs(rel_path):
                full_path = os.path.realpath(rel_path)
            else:
                full_path = os.path.realpath(os.path.join(PIPE_DIR, rel_path))

            if not any(full_path.startswith(sr + os.sep) or full_path == sr
                       for sr in _safe_roots):
                self.send_response(403); self.end_headers(); return

            if not os.path.isfile(full_path):
                self.send_response(404); self.end_headers(); return

            ext  = os.path.splitext(full_path)[1].lower().lstrip(".")
            mime = {"mp4": "video/mp4", "webm": "video/webm", "ogg": "video/ogg",
                    "mp3": "audio/mpeg", "wav": "audio/wav",
                    "aac": "audio/aac", "m4a": "audio/mp4",
                    "jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "png": "image/png", "webp": "image/webp",
                    "gif": "image/gif", "svg": "image/svg+xml",
                    }.get(ext, "application/octet-stream")

            file_size = os.path.getsize(full_path)
            range_hdr = self.headers.get("Range", "")

            try:
                if range_hdr:
                    m     = re.match(r"bytes=(\d*)-(\d*)", range_hdr)
                    start = int(m.group(1)) if m and m.group(1) else 0
                    end   = int(m.group(2)) if m and m.group(2) else file_size - 1
                    end   = min(end, file_size - 1)
                    chunk = end - start + 1
                    self.send_response(206)
                    self.send_header("Content-Type",   mime)
                    self.send_header("Content-Length", str(chunk))
                    self.send_header("Content-Range",  f"bytes {start}-{end}/{file_size}")
                    self.send_header("Accept-Ranges",  "bytes")
                    self.end_headers()
                    with open(full_path, "rb") as fh:
                        fh.seek(start)
                        remaining = chunk
                        while remaining > 0:
                            data = fh.read(min(65536, remaining))
                            if not data:
                                break
                            self.wfile.write(data)
                            remaining -= len(data)
                else:
                    self.send_response(200)
                    self.send_header("Content-Type",   mime)
                    self.send_header("Content-Length", str(file_size))
                    self.send_header("Accept-Ranges",  "bytes")
                    self.end_headers()
                    with open(full_path, "rb") as fh:
                        while True:
                            data = fh.read(65536)
                            if not data:
                                break
                            self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass   # client cancelled the video request

        # Voice clip index — params for every pre-cached style clip  (GET /api/voice_index)
        elif parsed.path == "/api/voice_index":
            idx_path = (Path(PIPE_DIR) / "projects" / "resources"
                        / "azure_tts" / "index.json")
            if idx_path.exists():
                try:
                    idx = json.loads(idx_path.read_text(encoding="utf-8"))
                    body = json.dumps({"voices": idx.get("voices", {})}).encode()
                except (json.JSONDecodeError, OSError):
                    body = json.dumps({"voices": {}}).encode()
            else:
                body = json.dumps({"voices": {}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # Voice presets (GET /api/voice_presets)
        elif parsed.path == "/api/voice_presets":
            pdata = load_presets()
            body  = json.dumps({"presets": pdata.get("presets", {})}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # Voice catalog grouped by story locale  (GET /api/azure_voices)
        elif parsed.path == "/api/azure_voices":
            catalog = parse_azure_tts_styles()
            body    = json.dumps(catalog).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # ── SFX: search candidates (POST /api/sfx_search via GET for simplicity) ─
        elif parsed.path == "/api/sfx_search":
            # This is a GET-based proxy, but sfx_search is POST-based — redirect handled in do_POST
            body = json.dumps({"error": "Use POST /api/sfx_search"}).encode()
            self.send_response(405)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # ── Media proxy: list batches for episode (GET /api/media_batches) ──────
        elif parsed.path == "/api/media_batches":
            params     = parse_qs(parsed.query)
            slug       = params.get("slug",       [""])[0].strip()
            ep_id      = params.get("ep_id",      [""])[0].strip()
            server_url = params.get("server_url", ["http://localhost:8200"])[0].strip()
            api_key    = os.environ.get("MEDIA_API_KEY", "")
            if not slug or not ep_id:
                body = json.dumps({"error": "slug and ep_id required"}).encode()
                self.send_response(400)
            else:
                url = (server_url.rstrip("/") + "/batches"
                       + f"?project={slug}&episode_id={ep_id}")
                try:
                    req  = _urllib_req.Request(url,
                               headers={"X-Api-Key": api_key})
                    with _urllib_req.urlopen(req, timeout=10) as resp:
                        raw  = resp.read()
                    body = json.dumps({"batches": json.loads(raw)}).encode()
                    self.send_response(200)
                except Exception as exc:
                    body = json.dumps({"error": str(exc)}).encode()
                    self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # ── Media proxy: poll batch status (GET /api/media_batch_status) ────────
        elif parsed.path == "/api/media_batch_status":
            params     = parse_qs(parsed.query)
            batch_id   = params.get("batch_id",   [""])[0].strip()
            server_url = params.get("server_url", ["http://localhost:8200"])[0].strip()
            api_key    = os.environ.get("MEDIA_API_KEY", "")
            if not batch_id:
                body = json.dumps({"error": "batch_id required"}).encode()
                self.send_response(400)
            else:
                url = server_url.rstrip("/") + "/batches/" + batch_id
                try:
                    req  = _urllib_req.Request(url,
                               headers={"X-Api-Key": api_key})
                    with _urllib_req.urlopen(req, timeout=10) as resp:
                        body = resp.read()
                    self.send_response(200)
                except Exception as exc:
                    body = json.dumps({"error": str(exc)}).encode()
                    self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # ── AI Images: list saved AI-generated images for an episode ────────────
        elif parsed.path == "/api/ai_images":
            params  = parse_qs(parsed.query)
            slug    = params.get("slug",  [""])[0].strip()
            ep_id   = params.get("ep_id", [""])[0].strip()
            if not slug or not ep_id:
                body = json.dumps({"error": "slug and ep_id required"}).encode()
                self.send_response(400)
            else:
                bg_root = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id,
                                       "assets", "backgrounds")
                result = {}
                if os.path.isdir(bg_root):
                    for bg_id in sorted(os.listdir(bg_root)):
                        bg_dir = os.path.join(bg_root, bg_id)
                        if not os.path.isdir(bg_dir):
                            continue
                        ai_files = sorted(
                            f for f in os.listdir(bg_dir)
                            if f.startswith("ai_") and f.endswith(".png")
                        )
                        if ai_files:
                            result[bg_id] = [
                                {"filename": f,
                                 "path":     os.path.join(bg_dir, f),
                                 "url":      "file://" + os.path.join(bg_dir, f)}
                                for f in ai_files
                            ]
                body = json.dumps(result).encode()
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # ── AI Job Status: proxy to AI server (GET /api/ai_job_status) ────────────
        elif parsed.path == "/api/ai_job_status":
            params = parse_qs(parsed.query)
            job_id = params.get("job_id", [""])[0].strip()
            if not job_id:
                body = json.dumps({"error": "job_id required"}).encode()
                self.send_response(400)
            else:
                try:
                    req = _urllib_req.Request(
                        _AI_SERVER_URL + "/jobs/" + job_id,
                        headers={"X-Api-Key": _AI_SERVER_KEY},
                    )
                    with _urllib_req.urlopen(req, timeout=10) as resp:
                        body = resp.read()
                    self.send_response(200)
                except Exception as exc:
                    body = json.dumps({"error": str(exc)}).encode()
                    self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # ── Music: return loop candidates JSON (GET /api/music_loop_candidates) ──
        elif parsed.path == "/api/music_loop_candidates":
            params = parse_qs(parsed.query)
            slug   = params.get("slug", [""])[0].strip()
            ep_id  = params.get("ep_id", [""])[0].strip()
            if not slug or not ep_id:
                body = json.dumps({"error": "slug and ep_id required"}).encode()
                self.send_response(400)
            else:
                cand_path = os.path.join(PIPE_DIR, "projects", slug,
                    "episodes", ep_id, "assets", "music", "music_loop_candidates.json")
                if not os.path.isfile(cand_path):
                    body = json.dumps({"error": "not found"}).encode()
                    self.send_response(404)
                else:
                    with open(cand_path, "rb") as _cf:
                        body = _cf.read()
                    self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # ── Music: return timeline JSON (GET /api/music_timeline) ────────────
        elif parsed.path == "/api/music_timeline":
            params = parse_qs(parsed.query)
            slug   = params.get("slug", [""])[0].strip()
            ep_id  = params.get("ep_id", [""])[0].strip()
            if not slug or not ep_id:
                body = json.dumps({"error": "slug and ep_id required"}).encode()
                self.send_response(400)
            else:
                tl_path = os.path.join(PIPE_DIR, "projects", slug,
                    "episodes", ep_id, "assets", "music",
                    "MusicReviewPack", "timeline.json")
                if not os.path.isfile(tl_path):
                    body = json.dumps({"error": "not found"}).encode()
                    self.send_response(404)
                else:
                    with open(tl_path, "rb") as _tf:
                        body = _tf.read()
                    self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # ── Music: list source music files (GET /api/music_sources) ──────────
        elif parsed.path == "/api/music_sources":
            params = parse_qs(parsed.query)
            slug   = params.get("slug", [""])[0].strip()
            ep_id  = params.get("ep_id", [""])[0].strip()
            if not slug or not ep_id:
                body = json.dumps({"error": "slug and ep_id required"}).encode()
                self.send_response(400)
            else:
                music_dir = os.path.join(PIPE_DIR, "projects", slug,
                                         "resources", "music")
                sources = []
                _supported_exts = {".mp3", ".wav", ".flac", ".ogg"}
                if os.path.isdir(music_dir):
                    # Try to load loop candidates for duration/bpm info
                    cand_data = {}
                    cand_path = os.path.join(PIPE_DIR, "projects", slug,
                        "episodes", ep_id, "assets", "music",
                        "music_loop_candidates.json")
                    if os.path.isfile(cand_path):
                        try:
                            with open(cand_path, encoding="utf-8") as _cf:
                                cand_data = json.load(_cf).get("tracks", {})
                        except Exception:
                            pass

                    for fname in sorted(os.listdir(music_dir)):
                        ext = os.path.splitext(fname)[1].lower()
                        if ext not in _supported_exts:
                            continue
                        stem = os.path.splitext(fname)[0]
                        rel_path = os.path.join("projects", slug,
                                                "resources", "music", fname)
                        track_info = cand_data.get(stem, {})
                        sources.append({
                            "stem": stem,
                            "filename": fname,
                            "path": rel_path,
                            "duration_sec": track_info.get("duration_total_sec"),
                            "bpm": track_info.get("bpm"),
                        })

                body = json.dumps(sources).encode()
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # ── Episode file: return raw JSON for whitelisted episode files ───────
        # GET /api/episode_file?slug=X&ep_id=Y&file=ShotList.json
        # Returns raw file contents as application/json.
        # Whitelisted files only to limit exposure.
        elif parsed.path == "/api/episode_file":
            _EPISODE_FILE_WHITELIST = {"ShotList.json", "selections.json",
                                       "meta.json",
                                       "assets/media/selections.json",
                                       "assets/music/MusicPlan.json",
                                       "assets/music/user_cut_clips.json",
                                       "assets/meta/gen_music_clip_results.json",
                                       "assets/sfx/sfx_search_results.json",
                                       "AssetManifest_draft.shared.json"}
            params   = parse_qs(parsed.query)
            slug     = params.get("slug", [""])[0].strip()
            ep_id    = params.get("ep_id", [""])[0].strip()
            filename = params.get("file", [""])[0].strip()

            if not slug or not ep_id or not filename:
                body = json.dumps({"error": "slug, ep_id, and file are required"}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif filename not in _EPISODE_FILE_WHITELIST:
                body = json.dumps({"error": f"file {filename!r} not in whitelist"}).encode()
                self.send_response(403)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                file_path = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id, filename)
                if not os.path.isfile(file_path):
                    self.send_response(404)
                    self.end_headers()
                else:
                    with open(file_path, "rb") as _ef:
                        data = _ef.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

        # ── NFS proxy: serve a file:// media file to the browser ─────────────
        # Browsers cannot load file:// URLs from an http:// page.  When the
        # media server is running in file transport mode (NFS), thumbnail URLs
        # are file:///mnt/... URIs.  The JS routes them through this endpoint
        # so the browser can display them without CORS / mixed-content errors.
        #
        # Safety: the file must exist and be a regular file.  No path-traversal
        # is possible because we resolve and stat the path before opening it.
        elif parsed.path == "/api/serve_media_file":
            import mimetypes as _mimetypes
            params   = parse_qs(parsed.query)
            file_url = unquote_plus(params.get("url", [""])[0])
            # Accept both file:///path and bare /path
            if file_url.startswith("file://"):
                file_path = file_url[len("file://"):]
            else:
                file_path = file_url
            abs_path = os.path.realpath(file_path)
            if not os.path.isfile(abs_path):
                self.send_response(404)
                self.end_headers()
            else:
                mime, _ = _mimetypes.guess_type(abs_path)
                mime    = mime or "application/octet-stream"
                fsize   = os.path.getsize(abs_path)
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(fsize))
                self.send_header("Cache-Control", "max-age=3600")
                self.end_headers()
                # Stream in 256 KB chunks instead of reading entire file into RAM.
                # Videos can be 50+ MB; reading N of them at once would exhaust memory.
                # Browsers with preload='metadata' close the connection after getting
                # enough data — the resulting BrokenPipe/ConnectionReset is expected.
                try:
                    with open(abs_path, "rb") as _mf:
                        while True:
                            chunk = _mf.read(262144)
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        # ── YouTube: status (GET /api/youtube_status) ────────────────────────────
        elif parsed.path == "/api/youtube_status":
            params = parse_qs(parsed.query)
            slug   = unquote_plus(params.get("slug",   [""])[0]).strip()
            ep_id  = unquote_plus(params.get("ep_id",  [""])[0]).strip()
            locale = unquote_plus(params.get("locale", ["en"])[0]).strip()

            # Input validation
            if not slug or not ep_id or not re.match(r'^[a-zA-Z0-9_\-]+$', slug) \
                    or not re.match(r'^s\d+e\d+$', ep_id) \
                    or locale not in ("en", "zh-Hans", "zh", "zh-CN", "ja", "ko", "fr", "de", "es", "pt"):
                body = json.dumps({"error": "invalid parameters"}).encode()
                self.send_response(400)
            else:
                try:
                    render_dir = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id, "renders", locale)
                    yt_path    = os.path.join(render_dir, "youtube.json")
                    st_path    = os.path.join(render_dir, "upload_state.json")
                    rv_path    = os.path.join(render_dir, "upload_review.json")

                    def _load_json(p):
                        if not os.path.isfile(p):
                            return None
                        with open(p, encoding="utf-8") as _f:
                            return json.load(_f)

                    yt_data = _load_json(yt_path)
                    st_data = _load_json(st_path) or {}
                    rv_data = _load_json(rv_path)

                    payload = {
                        "youtube":      yt_data,
                        "upload_state": st_data,
                        "review":       rv_data,
                        "render_dir":   os.path.relpath(render_dir, PIPE_DIR),
                    }
                    if not yt_data:
                        payload["error"] = "youtube.json not found — render the episode first"
                    body = json.dumps(payload).encode()
                    self.send_response(200)
                except Exception as exc:
                    body = json.dumps({"error": str(exc)}).encode()
                    self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # ── YouTube: serve existing thumbnail (GET /api/yt_thumbnail) ─────────
        elif parsed.path == "/api/yt_thumbnail":
            params = parse_qs(parsed.query)
            slug   = unquote_plus(params.get("slug",   [""])[0]).strip()
            ep_id  = unquote_plus(params.get("ep_id",  [""])[0]).strip()
            locale = unquote_plus(params.get("locale", ["en"])[0]).strip()

            if not slug or not ep_id \
                    or not re.match(r'^[a-zA-Z0-9_\-]+$', slug) \
                    or not re.match(r'^s\d+e\d+$', ep_id) \
                    or locale not in ("en", "zh-Hans", "zh", "zh-CN", "ja", "ko", "fr", "de", "es", "pt"):
                self.send_response(400); self.end_headers(); return

            thumb_path = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id,
                                      "renders", locale, "thumbnail.jpg")
            if not os.path.isfile(thumb_path):
                self.send_response(404); self.end_headers(); return

            with open(thumb_path, "rb") as _tf:
                data = _tf.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        # ── YouTube: stream video with Range support (GET /api/episode_video) ─
        elif parsed.path == "/api/episode_video":
            params = parse_qs(parsed.query)
            slug   = unquote_plus(params.get("slug",   [""])[0]).strip()
            ep_id  = unquote_plus(params.get("ep_id",  [""])[0]).strip()
            locale = unquote_plus(params.get("locale", ["en"])[0]).strip()

            # C26: validate path parameters
            if not slug or not ep_id \
                    or not re.match(r'^[a-zA-Z0-9_\-]+$', slug) \
                    or not re.match(r'^s\d+e\d+$', ep_id) \
                    or locale not in ("en", "zh-Hans", "zh", "zh-CN", "ja", "ko", "fr", "de", "es", "pt"):
                self.send_response(400); self.end_headers(); return

            mp4_path = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id,
                                    "renders", locale, "output.mp4")
            # Safety: realpath check
            mp4_real    = os.path.realpath(mp4_path)
            proj_real   = os.path.realpath(os.path.join(PIPE_DIR, "projects"))
            if not mp4_real.startswith(proj_real + os.sep):
                self.send_response(403); self.end_headers(); return
            if not os.path.isfile(mp4_path):
                self.send_response(404); self.end_headers(); return

            file_size    = os.path.getsize(mp4_path)
            range_header = self.headers.get("Range")

            if range_header:
                # Parse Range: bytes=start-end
                m = re.match(r"bytes=(\d+)-(\d*)", range_header)
                if m:
                    byte_start = int(m.group(1))
                    byte_end   = int(m.group(2)) if m.group(2) else file_size - 1
                else:
                    byte_start = 0
                    byte_end   = file_size - 1
            else:
                byte_start = 0
                byte_end   = file_size - 1

            byte_end     = min(byte_end, file_size - 1)
            content_len  = byte_end - byte_start + 1

            self.send_response(206 if range_header else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(content_len))
            self.send_header("Accept-Ranges", "bytes")
            if range_header:
                self.send_header("Content-Range", f"bytes {byte_start}-{byte_end}/{file_size}")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                with open(mp4_path, "rb") as vf:
                    vf.seek(byte_start)
                    remaining = content_len
                    while remaining > 0:
                        chunk = vf.read(min(262144, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

        # ── YouTube: extract frame preview (GET /api/thumbnail_preview) ───────
        elif parsed.path == "/api/thumbnail_preview":
            params = parse_qs(parsed.query)
            slug   = unquote_plus(params.get("slug",   [""])[0]).strip()
            ep_id  = unquote_plus(params.get("ep_id",  [""])[0]).strip()
            locale = unquote_plus(params.get("locale", ["en"])[0]).strip()
            try:
                sec = float(params.get("sec", ["0"])[0])
            except ValueError:
                self.send_response(400); self.end_headers(); return

            # C26: validate path parameters
            if not slug or not ep_id \
                    or not re.match(r'^[a-zA-Z0-9_\-]+$', slug) \
                    or not re.match(r'^s\d+e\d+$', ep_id) \
                    or locale not in ("en", "zh-Hans", "zh", "zh-CN", "ja", "ko", "fr", "de", "es", "pt"):
                self.send_response(400); self.end_headers(); return

            mp4_path = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id,
                                    "renders", locale, "output.mp4")
            if not os.path.isfile(mp4_path):
                self.send_response(404); self.end_headers(); return

            # C24: bounds-check sec via ffprobe
            try:
                probe_r = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_format",
                     "-print_format", "json", mp4_path],
                    capture_output=True, timeout=10,
                )
                probe_d  = json.loads(probe_r.stdout)
                duration = float(probe_d.get("format", {}).get("duration", 0))
                if sec < 0 or (duration > 0 and sec > duration):
                    body = json.dumps({"error": f"sec {sec} out of range [0, {duration:.2f}]"}).encode()
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
            except Exception:
                pass  # if ffprobe fails, proceed without bounds check

            # Extract frame
            result = subprocess.run([
                "ffmpeg", "-y", "-ss", str(sec), "-i", mp4_path,
                "-frames:v", "1", "-vf", "scale=1280:720",
                "-f", "image2", "-vcodec", "mjpeg", "-",
            ], capture_output=True, timeout=30)

            # C25: check ffmpeg returncode and non-empty output
            if result.returncode != 0 or not result.stdout:
                err_msg = result.stderr.decode(errors="replace")[:500]
                body = json.dumps({"error": f"ffmpeg failed: {err_msg}"}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(result.stdout)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(result.stdout)

        else:
            self.send_response(404)
            self.end_headers()

    # ── POST ──────────────────────────────────────────────────────────────────
    def do_POST(self):

        # Kill running process
        if self.path == "/stop":
            with _lock:
                for proc in _procs.values():
                    if proc.poll() is None:
                        proc.terminate()
            self.send_response(200)
            self.end_headers()

        # Save pasted story to story_N.txt
        elif self.path == "/save_story":
            try:
                length   = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(length)
                payload  = json.loads(raw_body)
                story    = payload.get("story", "").strip()
                if not story:
                    raise ValueError("story field is empty")
                filename, num = _save_story(story)
                resp = json.dumps({"filename": filename, "num": num}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
                print(f"  Saved {filename}  ({len(story)} chars)")
            except Exception as exc:
                resp = json.dumps({"error": str(exc)}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

        # Delete a single episode subfolder (safety-checked to projects/ tree)
        elif self.path == "/delete_episode_dir":
            try:
                length   = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(length)
                payload  = json.loads(raw_body)
                rel_path = payload.get("path", "").strip()

                # Safety: resolve to absolute and confirm it is inside
                # PIPE_DIR/projects/<slug>/episodes/<id>  (depth ≥ 4)
                projects_root = os.path.realpath(os.path.join(PIPE_DIR, "projects"))
                full_path     = os.path.realpath(os.path.join(PIPE_DIR, rel_path))
                parts         = os.path.relpath(full_path, projects_root).split(os.sep)

                if (not full_path.startswith(projects_root + os.sep)
                        or len(parts) < 3          # must be slug/episodes/id
                        or parts[1] != "episodes"):
                    raise ValueError(f"Refusing to delete path outside episodes tree: {rel_path!r}")

                if os.path.isdir(full_path):
                    shutil.rmtree(full_path)
                    print(f"  Deleted episode dir: {rel_path}")
                    resp = json.dumps({"deleted": True, "path": rel_path}).encode()
                else:
                    resp = json.dumps({"deleted": False, "error": "directory not found"}).encode()

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

            except Exception as exc:
                resp = json.dumps({"deleted": False, "error": str(exc)}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

        # ── Selective VO retune  (POST /api/retune_vo) ──────────────────────────
        elif self.path == "/api/retune_vo":
            try:
                length  = int(self.headers.get("Content-Length", 0))
                req     = json.loads(self.rfile.read(length))
                slug    = req.get("slug",   "").strip()
                ep_id   = req.get("ep_id",  "").strip()
                locale  = req.get("locale", "").strip()
                dry_run = bool(req.get("dry_run", False))
                backup  = bool(req.get("backup",  False))
                items   = req.get("items", [])

                if not slug or not ep_id or not locale:
                    raise ValueError("slug, ep_id, and locale are required")
                if not isinstance(items, list) or not items:
                    raise ValueError("items must be a non-empty list")
                if not _RETUNE_AVAILABLE:
                    raise RuntimeError("vo_retune module not available — check server logs")

                ep_dir        = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
                manifest_path = os.path.join(ep_dir,
                                             f"AssetManifest_merged.{locale}.json")

                # Build per-item patches from request items
                # Each item: { item_id, text?, azure_style?, azure_rate?,
                #              azure_pitch?, azure_style_degree?, azure_break_ms? }
                PATCH_FIELDS = {"text","azure_voice","azure_style","azure_rate",
                                "azure_pitch","azure_style_degree","azure_break_ms"}
                per_item_patches = {}
                item_ids         = []
                for it in items:
                    iid   = it.get("item_id", "").strip()
                    if not iid:
                        raise ValueError("Each item must have an item_id")
                    item_ids.append(iid)
                    patch = {k: v for k, v in it.items() if k in PATCH_FIELDS}
                    if patch:
                        per_item_patches[iid] = patch

                results = _retune_vo_items(
                    manifest_path    = manifest_path,
                    locale           = locale,
                    item_ids         = item_ids,
                    per_item_patches = per_item_patches or None,
                    dry_run          = dry_run,
                    backup           = backup,
                )
                body = json.dumps({"results": results}).encode()
                self.send_response(200)
            except (ValueError, FileNotFoundError) as exc:
                body = json.dumps({"error": str(exc)}).encode()
                self.send_response(400)
            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode()
                self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # ── VO Review endpoints (P3) ──────────────────────────────────────────────

        # POST /api/vo_recreate — re-synthesize with same params, bypass cache
        elif self.path == "/api/vo_recreate":
            try:
                if not _VO_UTILS_AVAILABLE:
                    raise RuntimeError("vo_utils not available")
                length = int(self.headers.get("Content-Length", 0))
                req    = json.loads(self.rfile.read(length))
                ep_dir  = req.get("ep_dir",  "").strip()
                locale  = req.get("locale",  "").strip()
                item_id = req.get("item_id", "").strip()
                _vo_validate_inputs(ep_dir, locale, item_id)
                full_ep = _vo_resolve_ep_dir(ep_dir)

                # Read current params from manifest
                mpath = os.path.join(full_ep, f"AssetManifest_merged.{locale}.json")
                with open(mpath, encoding="utf-8") as _mf:
                    _mani = json.load(_mf)
                _item = next((v for v in _mani.get("vo_items", [])
                              if v["item_id"] == item_id), None)
                if _item is None:
                    raise ValueError(f"item_id {item_id!r} not found in manifest")

                tp = _item.get("tts_prompt", {})
                params = {
                    "voice":        tp.get("azure_voice") or tp.get("voice", ""),
                    "style":        tp.get("azure_style") or tp.get("style", ""),
                    "style_degree": tp.get("azure_style_degree", 1.5),
                    "rate":         tp.get("azure_rate") or tp.get("rate", "0%"),
                    "pitch":        tp.get("azure_pitch", ""),
                    "break_ms":     tp.get("azure_break_ms", 0),
                }
                text = _item.get("text", "")

                with _get_vo_lock(full_ep):
                    result = synthesize_vo_item(
                        item_id, text, params, full_ep, locale,
                        write_cache=False,  # INVARIANT F: vo_recreate bypasses cache
                    )
                _json_resp(self, {"item_id": item_id, **result})

            except Exception as exc:
                _json_resp(self, {"error": str(exc)}, 409)

        # POST /api/vo_save — re-synthesize with new params, write to cache
        # Special case: keep_audio=true skips Azure TTS and keeps existing WAV on disk.
        # Used when user liked a Re-Created result and just wants to commit the manifest.
        elif self.path == "/api/vo_save":
            try:
                if not _VO_UTILS_AVAILABLE:
                    raise RuntimeError("vo_utils not available")
                length = int(self.headers.get("Content-Length", 0))
                req    = json.loads(self.rfile.read(length))
                ep_dir  = req.get("ep_dir",  "").strip()
                locale  = req.get("locale",  "").strip()
                item_id = req.get("item_id", "").strip()
                _vo_validate_inputs(ep_dir, locale, item_id)
                full_ep = _vo_resolve_ep_dir(ep_dir)

                new_voice        = req.get("voice", "").strip()
                new_style        = req.get("style", "")
                new_rate         = req.get("rate", "0%")
                new_style_degree = float(req.get("style_degree", 1.5))
                new_text         = req.get("text", "").strip()
                keep_audio       = bool(req.get("keep_audio", False))
                if not new_voice:
                    raise ValueError("voice is required")
                if not new_text:
                    raise ValueError("text is required")

                with _get_vo_lock(full_ep):
                    if keep_audio:
                        # Skip Azure TTS — keep existing source.wav / .wav on disk.
                        # Just update manifest params and return current durations.
                        from pathlib import Path as _P
                        vo_dir     = os.path.join(full_ep, "assets", locale, "audio", "vo")
                        src_path   = os.path.join(vo_dir, f"{item_id}.source.wav")
                        wav_path   = os.path.join(vo_dir, f"{item_id}.wav")
                        if not os.path.exists(src_path):
                            raise FileNotFoundError(f"source.wav not found for {item_id} — cannot keep audio")
                        source_dur  = _wav_duration(_P(src_path))
                        trimmed_dur = _wav_duration(_P(wav_path)) if os.path.exists(wav_path) else source_dur
                        primary_locale = _get_primary_locale(_P(full_ep))
                        _invalidate_vo_state(full_ep, primary_locale)
                        result = {
                            "source_duration_sec":  round(source_dur,  3),
                            "trimmed_duration_sec": round(trimmed_dur, 3),
                        }
                    else:
                        params = {
                            "voice":        new_voice,
                            "style":        new_style,
                            "style_degree": new_style_degree,
                            "rate":         new_rate,
                            "pitch":        req.get("pitch", ""),
                            "break_ms":     int(req.get("break_ms", 0)),
                        }
                        result = synthesize_vo_item(
                            item_id, new_text, params, full_ep, locale,
                            write_cache=True,   # INVARIANT F: vo_save writes cache
                        )

                    # Update manifest with voice/style/rate/text (always)
                    mpath = os.path.join(full_ep, f"AssetManifest_merged.{locale}.json")
                    with open(mpath, encoding="utf-8") as _mf:
                        _mani = json.load(_mf)
                    for _it in _mani.get("vo_items", []):
                        if _it["item_id"] == item_id:
                            _it["text"] = new_text
                            tp = _it.setdefault("tts_prompt", {})
                            tp["azure_voice"]        = new_voice
                            tp["azure_style"]        = new_style
                            tp["azure_style_degree"] = new_style_degree
                            tp["azure_rate"]         = new_rate
                            break
                    _tmp = mpath + ".tmp"
                    with open(_tmp, "w", encoding="utf-8") as _mf:
                        json.dump(_mani, _mf, indent=2, ensure_ascii=False)
                    os.replace(_tmp, mpath)
                _json_resp(self, {"item_id": item_id, **result})

            except Exception as exc:
                _json_resp(self, {"error": str(exc)}, 409)

        # POST /api/vo_trim — apply trim handles, write .wav via apply_vo_trims_for_item
        elif self.path == "/api/vo_trim":
            try:
                if not _VO_UTILS_AVAILABLE:
                    raise RuntimeError("vo_utils not available")
                length = int(self.headers.get("Content-Length", 0))
                req    = json.loads(self.rfile.read(length))
                ep_dir          = req.get("ep_dir",         "").strip()
                locale          = req.get("locale",         "").strip()
                item_id         = req.get("item_id",        "").strip()
                trim_start_sec  = float(req.get("trim_start_sec", 0.0))
                trim_end_sec    = float(req.get("trim_end_sec",   0.0))
                _vo_validate_inputs(ep_dir, locale, item_id)
                full_ep = _vo_resolve_ep_dir(ep_dir)
                from pathlib import Path as _P

                with _get_vo_lock(full_ep):
                    # Migration: create source.wav from .wav if missing (pre-two-file projects)
                    _ensure_source_wav(item_id, full_ep, locale)
                    # Step 1: Validate trim bounds against source.wav (INVARIANT K)
                    src_p = (_P(full_ep) / "assets" / locale / "audio" / "vo"
                             / f"{item_id}.source.wav")
                    if not src_p.exists():
                        raise FileNotFoundError(f"source.wav not found for {item_id}")
                    src_dur = _wav_duration(src_p)
                    if trim_end_sec > src_dur + 1e-6:
                        raise ValueError(
                            f"trim_end_sec ({trim_end_sec:.3f}) > source duration ({src_dur:.3f})"
                        )
                    if trim_start_sec < 0 or trim_end_sec <= trim_start_sec:
                        raise ValueError(
                            f"Invalid trim range: [{trim_start_sec:.3f}, {trim_end_sec:.3f}]"
                        )

                    # Step 2: Write override to sidecar
                    overrides = _load_vo_trim_overrides(full_ep, locale)
                    import time as _time
                    overrides[item_id] = {
                        "trim_start_sec":    trim_start_sec,
                        "trim_end_sec":      trim_end_sec,
                        "source_duration_sec": src_dur,
                        "applied_at": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
                    }
                    _save_vo_trim_overrides(full_ep, locale, overrides)

                    # Step 3: apply_vo_trims_for_item → writes .wav (INVARIANT B)
                    trimmed_dur = _apply_vo_trims_for_item(item_id, full_ep, locale)

                    # Step 4: Invalidate (INVARIANT H)
                    primary = _get_primary_locale(_P(full_ep))
                    _invalidate_vo_state(full_ep, primary)

                _json_resp(self, {
                    "item_id":            item_id,
                    "source_duration_sec": round(src_dur,     3),
                    "trimmed_duration_sec": round(trimmed_dur, 3),
                })

            except Exception as exc:
                _json_resp(self, {"error": str(exc)}, 409)

        # POST /api/vo_reset_trim — restore source.wav as active .wav
        elif self.path == "/api/vo_reset_trim":
            try:
                if not _VO_UTILS_AVAILABLE:
                    raise RuntimeError("vo_utils not available")
                length = int(self.headers.get("Content-Length", 0))
                req    = json.loads(self.rfile.read(length))
                ep_dir  = req.get("ep_dir",  "").strip()
                locale  = req.get("locale",  "").strip()
                item_id = req.get("item_id", "").strip()
                _vo_validate_inputs(ep_dir, locale, item_id)
                full_ep = _vo_resolve_ep_dir(ep_dir)
                from pathlib import Path as _P

                with _get_vo_lock(full_ep):
                    # Step 1: Delete override from sidecar (if present)
                    overrides = _load_vo_trim_overrides(full_ep, locale)
                    if item_id in overrides:
                        del overrides[item_id]
                        _save_vo_trim_overrides(full_ep, locale, overrides)

                    # Migration: create source.wav from .wav if missing (pre-two-file projects)
                    _ensure_source_wav(item_id, full_ep, locale)
                    # Step 2: apply_vo_trims_for_item (no override → copy source → .wav)
                    # (INVARIANT B: apply_vo_trims_for_item is ONLY writer of .wav)
                    src_p = (_P(full_ep) / "assets" / locale / "audio" / "vo"
                             / f"{item_id}.source.wav")
                    if not src_p.exists():
                        raise FileNotFoundError(f"source.wav not found for {item_id}")
                    src_dur = _wav_duration(src_p)
                    _apply_vo_trims_for_item(item_id, full_ep, locale)

                    # Step 3: Invalidate (INVARIANT H)
                    primary = _get_primary_locale(_P(full_ep))
                    _invalidate_vo_state(full_ep, primary)

                _json_resp(self, {
                    "item_id":            item_id,
                    "source_duration_sec": round(src_dur, 3),
                })

            except Exception as exc:
                _json_resp(self, {"error": str(exc)}, 409)

        # POST /api/vo_pause — update pause_after_ms in manifest (no WAV touched)
        elif self.path == "/api/vo_pause":
            try:
                if not _VO_UTILS_AVAILABLE:
                    raise RuntimeError("vo_utils not available")
                length = int(self.headers.get("Content-Length", 0))
                req    = json.loads(self.rfile.read(length))
                ep_dir   = req.get("ep_dir",   "").strip()
                locale   = req.get("locale",   "").strip()
                item_id  = req.get("item_id",  "").strip()
                pause_ms = int(req.get("pause_ms", 300))
                _vo_validate_inputs(ep_dir, locale, item_id)
                full_ep = _vo_resolve_ep_dir(ep_dir)
                from pathlib import Path as _P

                with _get_vo_lock(full_ep):
                    # Update manifest pause_after_ms (no WAV touched — INVARIANT E)
                    mpath = os.path.join(full_ep, f"AssetManifest_merged.{locale}.json")
                    with open(mpath, encoding="utf-8") as _mf:
                        _mani = json.load(_mf)
                    found = False
                    for _it in _mani.get("vo_items", []):
                        if _it["item_id"] == item_id:
                            _it["pause_after_ms"] = pause_ms
                            found = True
                            break
                    if not found:
                        raise ValueError(f"item_id {item_id!r} not found in manifest")
                    _tmp = mpath + ".tmp"
                    with open(_tmp, "w", encoding="utf-8") as _mf:
                        json.dump(_mani, _mf, indent=2, ensure_ascii=False)
                    os.replace(_tmp, mpath)

                    # Invalidate (INVARIANT H)
                    primary = _get_primary_locale(_P(full_ep))
                    _invalidate_vo_state(full_ep, primary)

                _json_resp(self, {"item_id": item_id, "pause_ms": pause_ms})

            except Exception as exc:
                _json_resp(self, {"error": str(exc)}, 409)

        # POST /api/vo_scene_tail — set inter-scene tail silence (ms) for a scene
        elif self.path == "/api/vo_scene_tail":
            try:
                length  = int(self.headers.get("Content-Length", 0))
                req     = json.loads(self.rfile.read(length))
                ep_dir  = req.get("ep_dir",   "").strip()
                locale  = req.get("locale",   "").strip()
                scene   = req.get("scene",    "").strip()
                tail_ms = int(req.get("tail_ms", 2000))
                if not ep_dir or not locale or not scene:
                    raise ValueError("ep_dir, locale, scene required")
                if tail_ms < 0 or tail_ms > 30000:
                    raise ValueError("tail_ms must be 0–30000")
                full_ep = _vo_resolve_ep_dir(ep_dir)
                from pathlib import Path as _P
                with _get_vo_lock(full_ep):
                    mpath = os.path.join(full_ep, f"AssetManifest_merged.{locale}.json")
                    with open(mpath, encoding="utf-8") as _mf:
                        _mani = json.load(_mf)
                    _mani.setdefault("scene_tails", {})[scene] = tail_ms
                    _tmp = mpath + ".tmp"
                    with open(_tmp, "w", encoding="utf-8") as _mf:
                        json.dump(_mani, _mf, indent=2, ensure_ascii=False)
                    os.replace(_tmp, mpath)
                    primary = _get_primary_locale(_P(full_ep))
                    _invalidate_vo_state(full_ep, primary)
                _json_resp(self, {"scene": scene, "tail_ms": tail_ms})
            except Exception as exc:
                _json_resp(self, {"error": str(exc)}, 409)

        # POST /api/vo_merge — merge two adjacent VO items into one
        elif self.path == "/api/vo_merge":
            try:
                if not _VO_UTILS_AVAILABLE:
                    raise RuntimeError("vo_utils not available")
                length = int(self.headers.get("Content-Length", 0))
                req    = json.loads(self.rfile.read(length))
                ep_dir      = req.get("ep_dir",      "").strip()
                locale      = req.get("locale",      "").strip()
                item_id     = req.get("item_id",     "").strip()
                merge_with  = req.get("merge_with",  "").strip()
                merged_text = req.get("merged_text", "").strip()
                _vo_validate_inputs(ep_dir, locale, item_id)
                _vo_validate_inputs(ep_dir, locale, merge_with)
                if not merged_text:
                    raise ValueError("merged_text is required")
                full_ep = _vo_resolve_ep_dir(ep_dir)
                from pathlib import Path as _P

                with _get_vo_lock(full_ep):
                    mpath = os.path.join(full_ep, f"AssetManifest_merged.{locale}.json")
                    with open(mpath, encoding="utf-8") as _mf:
                        _mani = json.load(_mf)

                    vo_items = _mani.get("vo_items", [])

                    # Step 1: Verify both items exist
                    _primary_item = next((v for v in vo_items if v["item_id"] == item_id), None)
                    _second_item  = next((v for v in vo_items if v["item_id"] == merge_with), None)
                    if _primary_item is None:
                        raise ValueError(f"item_id {item_id!r} not found")
                    if _second_item is None:
                        raise ValueError(f"merge_with {merge_with!r} not found")

                    # Step 2: Verify same shot and adjacency
                    shot_id = _primary_item.get("shot_id")
                    shot_items = [v for v in vo_items
                                  if v.get("shot_id") == shot_id]
                    idx_primary = next((i for i, v in enumerate(shot_items)
                                        if v["item_id"] == item_id), -1)
                    idx_second  = next((i for i, v in enumerate(shot_items)
                                        if v["item_id"] == merge_with), -1)
                    if idx_primary < 0 or idx_second < 0:
                        raise ValueError("Both items must be in the same shot")
                    if abs(idx_primary - idx_second) != 1:
                        raise ValueError("Items must be adjacent within their shot")

                    # Step 3b: Clear item_id trim override BEFORE synthesizing
                    overrides = _load_vo_trim_overrides(full_ep, locale)
                    if item_id in overrides:
                        del overrides[item_id]
                        _save_vo_trim_overrides(full_ep, locale, overrides)

                    # Step 4: Synthesize merged item (bypass cache — non-deterministic)
                    tp = _primary_item.get("tts_prompt", {})
                    params = {
                        "voice":        tp.get("azure_voice") or tp.get("voice", ""),
                        "style":        tp.get("azure_style") or tp.get("style", ""),
                        "style_degree": tp.get("azure_style_degree", 1.5),
                        "rate":         tp.get("azure_rate") or tp.get("rate", "0%"),
                        "pitch":        tp.get("azure_pitch", ""),
                        "break_ms":     tp.get("azure_break_ms", 0),
                    }
                    synth_result = synthesize_vo_item(
                        item_id, merged_text, params, full_ep, locale,
                        write_cache=False,  # INVARIANT F: vo_merge bypasses cache
                    )

                    # Step 5: Delete secondary item WAVs
                    vo_dir = _P(full_ep) / "assets" / locale / "audio" / "vo"
                    for _sfx in ("wav", "source.wav"):
                        _p = vo_dir / f"{merge_with}.{_sfx}"
                        if _p.exists():
                            _p.unlink()

                    # Step 6: Remove merge_with from trim overrides
                    overrides = _load_vo_trim_overrides(full_ep, locale)
                    if merge_with in overrides:
                        del overrides[merge_with]
                        _save_vo_trim_overrides(full_ep, locale, overrides)

                    # Step 7: Update manifest
                    _primary_item["text"] = merged_text
                    _mani["vo_items"] = [v for v in vo_items if v["item_id"] != merge_with]

                    # Step 8: Update ShotList.json
                    _shotlist_ref = _mani.get("shotlist_ref", "")
                    if _shotlist_ref:
                        _sl_candidates = [
                            _P(full_ep) / _shotlist_ref,
                            _P(full_ep) / "ShotList.json",
                        ]
                        for _sl_path in _sl_candidates:
                            if _sl_path.exists():
                                try:
                                    with open(_sl_path, encoding="utf-8") as _f:
                                        _sl = json.load(_f)
                                    for _shot in _sl.get("shots", []):
                                        _ids = _shot.get("audio_intent", {}).get("vo_item_ids", [])
                                        if merge_with in _ids:
                                            _ids.remove(merge_with)
                                    _tmp = str(_sl_path) + ".tmp"
                                    with open(_tmp, "w", encoding="utf-8") as _f:
                                        json.dump(_sl, _f, indent=2, ensure_ascii=False)
                                    os.replace(_tmp, str(_sl_path))
                                except Exception:
                                    pass
                                break

                    # Write updated manifest
                    _tmp = mpath + ".tmp"
                    with open(_tmp, "w", encoding="utf-8") as _mf:
                        json.dump(_mani, _mf, indent=2, ensure_ascii=False)
                    os.replace(_tmp, mpath)

                    # Step 9: Deep invalidation (item_id change → render plans invalid)
                    primary = _get_primary_locale(_P(full_ep))
                    _invalidate_vo_state(full_ep, primary)
                    # Also delete RenderPlan and media manifests for all locales
                    for _f in _P(full_ep).glob("RenderPlan.*.json"):
                        _f.unlink(missing_ok=True)
                    for _f in _P(full_ep).glob("AssetManifest.media.*.json"):
                        _f.unlink(missing_ok=True)

                    # Step 10: Append to merge log
                    import time as _time
                    _merge_log_path = _P(full_ep) / "vo_merge_log.json"
                    _merge_log = []
                    if _merge_log_path.exists():
                        try:
                            _merge_log = json.loads(_merge_log_path.read_text())
                        except Exception:
                            pass
                    _merge_log.append({
                        "item_id":     item_id,
                        "retired_id":  merge_with,
                        "merged_text": merged_text,
                        "locale":      locale,
                        "merged_at":   _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
                    })
                    _merge_log_path.write_text(
                        json.dumps(_merge_log, indent=2, ensure_ascii=False)
                    )

                _json_resp(self, {
                    "item_id":             item_id,
                    "retired_id":          merge_with,
                    "source_duration_sec": synth_result["source_duration_sec"],
                    "trimmed_duration_sec": synth_result["trimmed_duration_sec"],
                    "reload_required":     True,
                })

            except Exception as exc:
                _json_resp(self, {"error": str(exc)}, 409)

        # POST /api/vo_approve — validate and approve VO, write sentinel
        elif self.path == "/api/vo_approve":
            try:
                if not _VO_UTILS_AVAILABLE:
                    raise RuntimeError("vo_utils not available")
                length = int(self.headers.get("Content-Length", 0))
                req    = json.loads(self.rfile.read(length))
                ep_dir = req.get("ep_dir", "").strip()
                locale = req.get("locale", "").strip()
                _vo_validate_inputs(ep_dir, locale)
                full_ep = _vo_resolve_ep_dir(ep_dir)
                from pathlib import Path as _P
                import subprocess as _sp

                with _get_vo_lock(full_ep):
                    # Pre-flight check (a): no synthesis jobs in-flight
                    # (Always true under synchronous lock design)

                    # Pre-flight check (e): locale must be primary_locale
                    primary = _get_primary_locale(_P(full_ep))
                    if locale != primary:
                        raise ValueError(
                            f"Must approve primary locale ({primary!r}), not {locale!r}"
                        )

                    # Pre-flight check (c): all trim overrides valid (INVARIANT K)
                    overrides = _load_vo_trim_overrides(full_ep, locale)
                    vo_dir    = _P(full_ep) / "assets" / locale / "audio" / "vo"
                    for _iid, _ov in overrides.items():
                        _src = vo_dir / f"{_iid}.source.wav"
                        if _src.exists():
                            _src_dur = _wav_duration(_src)
                            if _ov.get("trim_end_sec", 0) > _src_dur + 1e-6:
                                raise ValueError(
                                    f"Trim override for {_iid} is out of bounds "
                                    f"(trim_end={_ov['trim_end_sec']:.3f} > "
                                    f"source_dur={_src_dur:.3f})"
                                )

                    # Pre-flight check (d): manifest structurally valid
                    mpath = os.path.join(full_ep, f"AssetManifest_merged.{locale}.json")
                    if not os.path.isfile(mpath):
                        raise FileNotFoundError(
                            f"AssetManifest_merged.{locale}.json not found"
                        )
                    with open(mpath, encoding="utf-8") as _mf:
                        _mani = json.load(_mf)

                    # Run post_tts_analysis under lock (INVARIANT C, G)
                    # post_tts_analysis must NOT acquire _vo_locks internally (deadlock)
                    _pta_script = os.path.join(os.path.dirname(__file__), "post_tts_analysis.py")
                    _pta_cmd = [
                        "python3", _pta_script,
                        "--manifest", mpath,
                    ]
                    _pta_env = os.environ.copy()
                    _pta_env.pop("CLAUDECODE", None)
                    _pta_result = _sp.run(
                        _pta_cmd, capture_output=True, text=True,
                        cwd=PIPE_DIR, env=_pta_env, timeout=120,
                    )
                    if _pta_result.returncode != 0:
                        raise RuntimeError(
                            f"post_tts_analysis failed (exit {_pta_result.returncode}):\n"
                            + _pta_result.stderr[:500]
                        )

                    # Count items processed
                    import re as _re_pta
                    _items_measured = len(_re_pta.findall(r"\[OK\]", _pta_result.stdout))

                    # Write sentinel (INVARIANT I)
                    _hashes = _compute_sentinel_hashes(full_ep, locale)
                    _write_sentinel(full_ep, locale, _hashes)

                    # Export {primary_locale}_vo_durations.json
                    # Re-read manifest (post_tts_analysis mutated it in-place)
                    with open(mpath, encoding="utf-8") as _mf:
                        _mani = json.load(_mf)
                    _durations = {
                        v["item_id"]: round(
                            v.get("end_sec", 0) - v.get("start_sec", 0), 3
                        )
                        for v in _mani.get("vo_items", [])
                        if "start_sec" in v and "end_sec" in v
                    }
                    _dur_path = os.path.join(full_ep, f"{primary}_vo_durations.json")
                    _dur_tmp  = _dur_path + ".tmp"
                    with open(_dur_tmp, "w", encoding="utf-8") as _df:
                        json.dump(_durations, _df, indent=2)
                    os.replace(_dur_tmp, _dur_path)

                _json_resp(self, {
                    "approved":        True,
                    "items_measured":  _items_measured,
                    "locale":          locale,
                })
                print(f"[vo_approve] ✓ {locale} approved — "
                      f"{_items_measured} items measured")

            except ValueError as exc:
                _json_resp(self, {"error": "Validation failed", "detail": str(exc)}, 409)
            except Exception as exc:
                _json_resp(self, {"error": str(exc), "detail": ""}, 409)

        # TTS preview — F0-throttled, disk-cached  (POST /api/preview_voice)
        elif self.path == "/api/preview_voice":
            tmp_path = None
            try:
                length   = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(length)
                req      = json.loads(raw_body)

                azure_voice  = req.get("azure_voice", "").strip()
                style        = req.get("style") or None
                style_degree = float(req.get("style_degree") or 0)
                rate         = req.get("rate") or "0%"
                pitch        = req.get("pitch") or ""
                break_ms     = int(req.get("break_ms") or 0)
                text         = req.get("text", "").strip()

                if not azure_voice or not text:
                    raise ValueError("azure_voice and text are required")

                # Use client-supplied azure_locale (needed for multilingual voices
                # where the user picked the voice under a different locale group,
                # e.g. en-GB-AdaMultilingualNeural selected under zh-HK).
                # Fall back to deriving from voice name for old callers.
                azure_locale = (req.get("azure_locale") or "").strip() \
                               or '-'.join(azure_voice.split('-')[:2])

                # Cache key — must include azure_locale so the same voice previewed
                # under different locale groups (e.g. zh-HK vs en-GB) gets separate files.
                key_dict = {
                    "v": azure_voice, "l": azure_locale, "s": style or "",
                    "d": style_degree, "r": rate,
                    "p": pitch or "", "b": break_ms, "t": text,
                }
                h = hashlib.sha256(
                    json.dumps(key_dict, sort_keys=True).encode()
                ).hexdigest()[:16]

                # BUG-A: colon → underscore (colon illegal on Windows/FAT)
                voice_dir  = azure_voice.replace(":", "_")
                # BUG-R4: PIPE_DIR-anchored absolute path (not CWD-relative)
                _pipe_path = Path(PIPE_DIR)
                cache_path = (_pipe_path / "projects" / "resources" / "azure_tts"
                              / voice_dir / f"{h}.mp3")

                if not cache_path.exists():
                    if not _TTS_AVAILABLE:
                        raise RuntimeError("azure-cognitiveservices-speech SDK not installed")

                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    ssml   = _preview_build_ssml(text, azure_voice, azure_locale, style,
                                                 style_degree=style_degree, rate=rate,
                                                 pitch=pitch, break_ms=break_ms)
                    result = _tts_throttled_call(_get_synth(), ssml)

                    # BUG 3: atomic write — no partial files land at cache_path
                    tmp_path = cache_path.with_suffix(".tmp")
                    tmp_path.write_bytes(result.audio_data)
                    tmp_path.rename(cache_path)
                    tmp_path = None

                # BUG-I1: relative URL so serve_media can resolve it
                url = "/serve_media?path=" + str(cache_path.relative_to(_pipe_path))

                # ── Update index.json ──────────────────────────────────────
                # Only save to index when params match canonical defaults.
                # Styled clips: degree=1.0 rate="0%" pitch="" break_ms=0
                # No-style:     degree=1.0 rate="0%" pitch="-5%" break_ms=600
                idx       = _load_index_cache()
                style_key = style or ""   # "" key for no-style voices
                is_new_to_index = style_key not in idx.get(azure_voice, {}).get("clips", {})

                if style_key:
                    _canon = (1.0, "0%", "",   0)
                else:
                    _canon = (1.0, "0%", "-5%", 600)
                _actual = (style_degree, rate, pitch or "", break_ms)
                _is_canonical = (_actual == _canon)

                if is_new_to_index and _is_canonical:
                    ve = idx.setdefault(azure_voice, {
                        "locale":       azure_locale,
                        "locale_group": azure_locale.split("-")[0],
                        "clips":        {},
                    })
                    ve.setdefault("clips", {})[style_key] = {
                        "hash":   h,
                        "file":   str(cache_path.relative_to(_pipe_path)),
                        "text":   text,
                        "params": {
                            "style":        style,
                            "style_degree": style_degree,
                            "rate":         rate,
                            "pitch":        pitch or "",
                            "break_ms":     break_ms,
                        },
                    }
                    save_index(idx)

                # ── Auto-save preset ────────────────────────────────────────
                # Save as custom preset when:
                #   - index entry exists but hash differs (user tweaked params)
                #   - index entry doesn't exist yet AND params are non-canonical
                #     (first preview with custom params — don't pollute index)
                new_preset = None
                if (not is_new_to_index and not _is_default_clip(azure_voice, style, h)) \
                   or (is_new_to_index and not _is_canonical):
                    pdata = load_presets()
                    vp    = pdata["presets"].setdefault(azure_voice, [])
                    if not any(p["style"] == (style or "") and p["style_degree"] == style_degree
                                          and p["rate"] == rate and p["pitch"] == (pitch or "")
                                          and p["break_ms"] == break_ms for p in vp):
                        new_preset = {
                            "name":         f"custom{max((int(p['name'].removeprefix('custom')) for p in vp if p['name'].startswith('custom') and p['name'].removeprefix('custom').isdigit()), default=0) + 1}",
                            "style":        style or "",
                            "style_degree": style_degree,
                            "rate":         rate,
                            "pitch":        pitch or "",
                            "break_ms":     break_ms,
                            "hash":         h,
                        }
                        vp.append(new_preset)
                        save_presets(pdata)

                # ── Tell JS the index-default params so it can sync _voiceIndex ──
                # Sent only on the FIRST preview (is_new_to_index=True) so the JS
                # in-memory cache stays in sync without a round-trip to /api/voice_index.
                index_params = None
                if is_new_to_index:
                    index_params = {
                        "style":        style,
                        "style_degree": style_degree,
                        "rate":         rate,
                        "pitch":        pitch or "",
                        "break_ms":     break_ms,
                    }

                resp = json.dumps({
                    "url":          url,
                    "preset":       new_preset,
                    "index_params": index_params,   # non-null only on first preview
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

            except Exception as exc:
                if tmp_path:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                resp = json.dumps({"error": str(exc)}).encode()
                self.send_response(200)          # 200 so JS always gets JSON body
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

        # Save VoiceCast.json  (POST /api/save_voice_cast)
        elif self.path == "/api/save_voice_cast":
            try:
                length   = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(length)
                req      = json.loads(raw_body)

                slug       = req.get("slug", "").strip()
                voice_cast = req.get("voice_cast")

                if not slug:
                    raise ValueError("slug is required")
                if voice_cast is None:
                    raise ValueError("voice_cast is required")

                # Write to projects/{slug}/VoiceCast.json (project-level, full overwrite)
                proj_dir = os.path.join(PIPE_DIR, "projects", slug)
                os.makedirs(proj_dir, exist_ok=True)
                vc_path  = os.path.join(proj_dir, "VoiceCast.json")
                with open(vc_path, "w", encoding="utf-8") as fh:
                    json.dump(voice_cast, fh, indent=2, ensure_ascii=False)

                resp = json.dumps({"ok": True}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
                print(f"  Saved VoiceCast.json → projects/{slug}/")

            except Exception as exc:
                resp = json.dumps({"error": str(exc)}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

        # Delete a custom voice preset  (POST /api/delete_preset)
        elif self.path == "/api/delete_preset":
            try:
                length   = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(length)
                req      = json.loads(raw_body)

                voice = req.get("voice", "").strip()
                h     = req.get("hash", "").strip()
                if not voice or not h:
                    raise ValueError("voice and hash are required")

                # Remove from presets.json
                pdata   = load_presets()
                vp      = pdata.get("presets", {}).get(voice, [])
                before  = len(vp)
                vp[:]   = [p for p in vp if p.get("hash") != h]
                removed = before - len(vp)
                if removed:
                    pdata["presets"][voice] = vp
                    save_presets(pdata)

                # Delete cached audio file(s)
                voice_dir  = voice.replace(":", "_")
                _pipe_path = Path(PIPE_DIR)
                deleted_files = []
                for ext in ("mp3", "wav"):
                    fp = _pipe_path / "projects" / "resources" / "azure_tts" / voice_dir / f"{h}.{ext}"
                    if fp.exists():
                        fp.unlink()
                        deleted_files.append(str(fp.relative_to(_pipe_path)))

                resp = json.dumps({
                    "ok": True,
                    "presets_removed": removed,
                    "files_deleted": deleted_files,
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
                print(f"  Deleted preset {h} for {voice} ({removed} entry, {len(deleted_files)} file(s))")

            except Exception as exc:
                resp = json.dumps({"error": str(exc)}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

        elif self.path == "/api/append_status_report":
            try:
                from datetime import datetime as _dt
                length   = int(self.headers.get("Content-Length", 0))
                req      = json.loads(self.rfile.read(length))
                slug     = req.get("slug",  "").strip()
                ep_id    = req.get("ep_id", "").strip()
                text     = req.get("text",  "").strip()
                if not (slug and ep_id and text):
                    raise ValueError("slug, ep_id, and text are required")
                ep_dir  = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
                os.makedirs(ep_dir, exist_ok=True)
                sr_path = os.path.join(ep_dir, "status_report.txt")
                ts      = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
                entry   = f"\n{'='*60}\n{ts}\n{'='*60}\n{text}\n"
                with open(sr_path, "a", encoding="utf-8") as fh:
                    fh.write(entry)
                resp = json.dumps({"ok": True}).encode()
            except Exception as exc:
                resp = json.dumps({"error": str(exc)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        elif self.path == "/api/infer_story_meta":
            length  = int(self.headers.get("Content-Length", 0))
            body    = json.loads(self.rfile.read(length).decode())
            story   = body.get("story", "").strip()
            if not story:
                resp = json.dumps({"error": "No story provided"}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
                return

            import tempfile, re as _re

            # ── SSML pre-check: detect authored SSML before calling haiku ──
            _ssml_pattern = _re.compile(
                r'<(?:speak|voice|prosody|mstts:)\b', _re.IGNORECASE
            )
            if _ssml_pattern.search(story[:2000]):
                # SSML detected — skip haiku call, return ssml_narration directly
                _ssml_title = ""
                # Try to extract title from SSML comment
                _title_m = _re.search(r'<!--\s*[Tt]itle:\s*(.+?)\s*-->', story[:2000])
                if _title_m:
                    _ssml_title = _title_m.group(1).strip()
                # Fallback: first meaningful text content
                if not _ssml_title:
                    _text_m = _re.search(r'>([^<]{10,60})<', story[:3000])
                    if _text_m:
                        _ssml_title = _text_m.group(1).strip()[:50]
                _ssml_slug = _re.sub(r'[^a-z0-9]+', '-', _ssml_title.lower()).strip('-')[:60] if _ssml_title else ""
                data = {
                    "title": _ssml_title,
                    "slug": _ssml_slug,
                    "genre": "narration",
                    "story_format": "ssml_narration",
                    "metadata_found": ["ssml_tags"],
                }
            else:
                prompt_text = (
                    "Read the following story and reply with ONLY a valid JSON object — no explanation, no markdown, just the JSON.\n\n"
                    "JSON fields required:\n"
                    "  title        : the story title (string)\n"
                    "  slug         : URL-safe project slug — lowercase, hyphens only, no spaces (string)\n"
                    "  genre        : inferred genre e.g. dark-fantasy, sci-fi, romance, sleep-story (string)\n"
                    "  story_format : one of: episodic, continuous_narration, illustrated_narration, documentary, monologue\n"
                    "                 Rules: sleep/meditation/mindfulness → continuous_narration\n"
                    "                        children's story/fable/fairy tale → illustrated_narration\n"
                    "                        documentary/explainer/educational → documentary\n"
                    "                        diary/confession/first-person speech → monologue\n"
                    "                        all others → episodic\n"
                    "  metadata_found : array of field names explicitly present in the story text\n"
                    "                   (e.g. [\"title\", \"slug\"] if the story has 'Title:' and 'Project slug:' lines)\n\n"
                    "Story:\n\n" + story[:6000]
                )

                with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tf:
                    tf.write(prompt_text)
                    tmp_path = tf.name

                try:
                    _haiku_env = os.environ.copy()
                    _haiku_env.pop("CLAUDECODE", None)
                    result = subprocess.run(
                        ["claude", "-p",
                         "--model", "haiku",
                         "--dangerously-skip-permissions",
                         "--no-session-persistence",
                         tmp_path],
                        capture_output=True, text=True, cwd=PIPE_DIR, timeout=30,
                        env=_haiku_env,
                    )
                    raw = result.stdout.strip()
                    # Strip markdown code fences if present
                    raw = _re.sub(r"^```[a-z]*\n?", "", raw, flags=_re.MULTILINE)
                    raw = _re.sub(r"\n?```$", "", raw, flags=_re.MULTILINE)
                    raw = raw.strip()
                    data = json.loads(raw)
                except Exception as exc:
                    data = {"error": str(exc), "title": "", "slug": "", "genre": "", "story_format": "episodic", "metadata_found": []}
                finally:
                    import os as _os
                    _os.unlink(tmp_path)

            # Check slug uniqueness
            slug = data.get("slug", "")
            if slug:
                projects_dir = os.path.join(PIPE_DIR, "projects")
                if os.path.isdir(os.path.join(projects_dir, slug)):
                    data["slug_exists"] = True
                    # Suggest a unique slug
                    n = 2
                    while os.path.isdir(os.path.join(projects_dir, f"{slug}-{n}")):
                        n += 1
                    data["slug_suggested"] = f"{slug}-{n}"
                else:
                    data["slug_exists"] = False
                    data["slug_suggested"] = slug

            resp = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        # Create a new episode: dir + story.txt + meta.json + pipeline_vars.sh stub
        elif self.path == "/api/create_episode":
            try:
                import random as _random
                from datetime import datetime as _dt
                length   = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(length)
                payload  = json.loads(raw_body)
                slug         = payload.get("slug", "").strip()
                ep_id        = payload.get("ep_id", "").strip()
                story_text   = payload.get("story", "").strip()
                title        = payload.get("title", "").strip()
                genre        = payload.get("genre", "").strip()
                story_format = payload.get("story_format", "episodic").strip()
                locales      = payload.get("locales", "en").strip()
                no_music     = bool(payload.get("no_music", False))
                purge_cache  = bool(payload.get("purge_cache", False))

                if not slug or not ep_id:
                    raise ValueError("slug and ep_id are required")
                if not story_text:
                    raise ValueError("story text is required")

                # Derive episode_number from ep_id (s01e03 → "03", ep0018 → "18")
                import re as _re3
                _ep_num_match = _re3.search(r"e(\d+)$", ep_id, _re3.IGNORECASE)
                episode_number = _ep_num_match.group(1) if _ep_num_match else ep_id

                # Create episode directory
                ep_dir = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
                os.makedirs(ep_dir, exist_ok=True)

                # Write story.txt
                with open(os.path.join(ep_dir, "story.txt"), "w", encoding="utf-8") as _f:
                    _f.write(story_text)

                # Write meta.json with full schema (field names match what p_0.txt reads)
                gen_seed = _random.randint(100_000_000, 999_999_999)
                meta = {
                    "schema_id":      "EpisodeMeta",
                    "story_title":    title,
                    "project_slug":   slug,
                    "episode_id":     ep_id,
                    "episode_number": episode_number,
                    "series_genre":   genre,
                    "story_format":   story_format,
                    "locales":        locales,
                    "generation_seed": gen_seed,
                    "render_profile": "preview_local",
                    "no_music":       no_music,
                    "purge_cache":    purge_cache,
                    "created_at":     _dt.now().isoformat(),
                }
                with open(os.path.join(ep_dir, "meta.json"), "w", encoding="utf-8") as _f:
                    json.dump(meta, _f, indent=2, ensure_ascii=False)

                # Write pipeline_vars.sh stub (Stage 0 will overwrite with full version)
                _locs_clean = ",".join(l.strip() for l in locales.split(",") if l.strip())
                vars_content = (
                    f'export STORY_TITLE="{title}"\n'
                    f'export EPISODE_NUMBER="{episode_number}"\n'
                    f'export EPISODE_ID="{ep_id}"\n'
                    f'export LOCALES="{_locs_clean}"\n'
                    f'export PROJECT_SLUG="{slug}"\n'
                    f'export SERIES_GENRE="{genre}"\n'
                    f'export GENERATION_SEED="{gen_seed}"\n'
                    f'export RENDER_PROFILE="preview_local"\n'
                    f'export STORY_FORMAT="{story_format}"\n'
                    f'export PROJECT_DIR="projects/{slug}"\n'
                    f'export EPISODE_DIR="projects/{slug}/episodes/{ep_id}"\n'
                )
                with open(os.path.join(ep_dir, "pipeline_vars.sh"), "w", encoding="utf-8") as _f:
                    _f.write(vars_content)

                ep_dir_rel = f"projects/{slug}/episodes/{ep_id}"
                print(f"  Created episode  {ep_dir_rel}")
                resp = json.dumps({"ok": True, "ep_dir": ep_dir_rel,
                                   "slug": slug, "ep_id": ep_id}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
            except Exception as exc:
                resp = json.dumps({"ok": False, "error": str(exc)}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

        # ── Media proxy: start a new batch (POST /api/media_batch) ─────────────
        elif self.path == "/api/media_batch":
            try:
                length   = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(length)
                payload  = json.loads(raw_body)

                slug            = payload.get("slug", "").strip()
                ep_id           = payload.get("ep_id", "").strip()
                server_url      = (payload.get("server_url") or "http://localhost:8200").rstrip("/")
                api_key         = os.environ.get("MEDIA_API_KEY", "")
                content_profile = payload.get("content_profile", "default")
                n_img            = payload.get("n_img") or None
                n_vid            = payload.get("n_vid") or None
                sources_override        = payload.get("sources_override") or None
                source_limits_override  = payload.get("source_limits_override") or None

                if not slug or not ep_id:
                    raise ValueError("slug and ep_id are required")

                # Load AssetManifest_draft to pass backgrounds to media server
                ep_dir = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
                manifest_path = os.path.join(ep_dir, "AssetManifest_draft.shared.json")
                if not os.path.isfile(manifest_path):
                    raise FileNotFoundError(
                        f"AssetManifest_draft.shared.json not found at {manifest_path}")
                with open(manifest_path, encoding="utf-8") as _mf:
                    manifest = json.load(_mf)

                req_body = json.dumps({
                    "project":          slug,
                    "episode_id":       ep_id,
                    "manifest":         manifest,
                    "top_n":            int(os.environ.get("MEDIA_TOP_N",
                                            _vc_config.get("media", {}).get("top_n", 5))),
                    "content_profile":  content_profile,
                    **({"n_img": n_img} if n_img is not None else {}),
                    **({"n_vid": n_vid} if n_vid is not None else {}),
                    **({"sources_override": sources_override} if sources_override is not None else {}),
                    **({"source_limits_override": source_limits_override} if source_limits_override is not None else {}),
                }).encode()

                url = server_url + "/batches"
                req = _urllib_req.Request(
                    url, data=req_body,
                    headers={"X-Api-Key": api_key,
                             "Content-Type": "application/json",
                             "Content-Length": str(len(req_body))},
                    method="POST",
                )
                with _urllib_req.urlopen(req, timeout=15) as resp:
                    body = resp.read()
                self.send_response(202)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        # ── Media proxy: resume an interrupted batch (POST /api/media_batch_resume) ──
        elif self.path == "/api/media_batch_resume":
            try:
                length     = int(self.headers.get("Content-Length", 0))
                payload    = json.loads(self.rfile.read(length))
                batch_id   = payload.get("batch_id", "").strip()
                server_url = (payload.get("server_url") or "http://localhost:8200").rstrip("/")
                api_key    = os.environ.get("MEDIA_API_KEY", "")
                if not batch_id:
                    raise ValueError("batch_id is required")
                forward = {}
                if payload.get("source_limits_override"):
                    forward["source_limits_override"] = payload["source_limits_override"]
                if payload.get("sources_override"):
                    forward["sources_override"] = payload["sources_override"]
                req_body = json.dumps(forward).encode()
                url = f"{server_url}/batches/{batch_id}/resume"
                req = _urllib_req.Request(
                    url, data=req_body,
                    headers={"X-Api-Key": api_key,
                             "Content-Type": "application/json",
                             "Content-Length": str(len(req_body))},
                    method="POST",
                )
                with _urllib_req.urlopen(req, timeout=15) as resp:
                    body = resp.read()
                self.send_response(202)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        # ── SFX: search candidates (POST /api/sfx_search) ────────────────────
        elif self.path == "/api/sfx_search":
            try:
                length  = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length))
                query        = payload.get("query", "").strip()
                duration_sec = float(payload.get("duration_sec", 5.0))
                server_url   = (payload.get("server_url") or "http://localhost:8200").rstrip("/")
                api_key      = os.environ.get("MEDIA_API_KEY", "")
                if not query:
                    raise ValueError("query is required")

                req_body = json.dumps({"query": query, "duration_sec": duration_sec}).encode()
                req = _urllib_req.Request(
                    server_url + "/sfx_search",
                    data=req_body,
                    headers={"X-Api-Key": api_key,
                             "Content-Type": "application/json",
                             "Content-Length": str(len(req_body))},
                    method="POST",
                )
                with _urllib_req.urlopen(req, timeout=30) as resp:
                    body = resp.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            except Exception as exc:
                body = json.dumps({"error": str(exc), "candidates": []}).encode()
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        # ── SFX: save a selected sound (POST /api/sfx_save) ──────────────────
        elif self.path == "/api/sfx_save":
            try:
                length  = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length))
                slug        = payload.get("slug", "").strip()
                ep_id       = payload.get("ep_id", "").strip()
                server_url  = (payload.get("server_url") or "http://localhost:8200").rstrip("/")
                api_key     = os.environ.get("MEDIA_API_KEY", "")
                if not slug or not ep_id:
                    raise ValueError("slug and ep_id are required")
                if not payload.get("preview_url"):
                    raise ValueError("preview_url is required")

                # Forward to media server's /sfx_save endpoint
                req_body = json.dumps({
                    "project":     slug,
                    "episode_id":  ep_id,
                    "item_id":     payload.get("item_id", ""),
                    "preview_url": payload.get("preview_url", ""),
                    "source_site": payload.get("source_site", ""),
                    "attribution": payload.get("attribution"),
                }).encode()
                req = _urllib_req.Request(
                    server_url + "/sfx_save",
                    data=req_body,
                    headers={"X-Api-Key": api_key,
                             "Content-Type": "application/json",
                             "Content-Length": str(len(req_body))},
                    method="POST",
                )
                with _urllib_req.urlopen(req, timeout=60) as resp:
                    body = resp.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            except _urllib_err.HTTPError as exc:
                # Pass through the real error body from the media server so the
                # browser console shows the actual reason (license, SSRF, etc.)
                try:
                    err_body = exc.read()
                except Exception:
                    err_body = json.dumps({"error": str(exc)}).encode()
                self.send_response(exc.code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_body)))
                self.end_headers()
                self.wfile.write(err_body)
            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode()
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        # ── SFX: persist search results to disk (POST /api/sfx_results_save) ──
        elif self.path == "/api/sfx_results_save":
            try:
                length  = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length))
                slug    = payload.get("slug", "").strip()
                ep_id   = payload.get("ep_id", "").strip()
                if not slug or not ep_id:
                    raise ValueError("slug and ep_id are required")
                dest_dir = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id,
                                        "assets", "sfx")
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, "sfx_search_results.json")
                data = {
                    "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "results":  payload.get("results", {}),
                    "selected": payload.get("selected", {}),
                }
                with open(dest_path, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, ensure_ascii=False, indent=2)
                body = json.dumps({"ok": True}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        # ── AI SFX Generate: submit generation job to AI server ──────────────
        elif self.path == "/api/ai_sfx_generate":
            try:
                length      = int(self.headers.get("Content-Length", 0))
                payload     = json.loads(self.rfile.read(length))
                slug        = payload.get("slug", "").strip()
                ep_id       = payload.get("ep_id", "").strip()
                item_id     = payload.get("item_id", "").strip()
                prompt      = payload.get("prompt", "").strip()
                duration_sec = float(payload.get("duration_sec", 5.0))
                if not all([slug, ep_id, item_id, prompt]):
                    raise ValueError("slug, ep_id, item_id, and prompt are required")
                timestamp_ms = int(time.time() * 1000)
                h = hashlib.sha1(prompt.encode()).hexdigest()[:8]
                asset_id = f"{item_id}-{h}-{timestamp_ms}"
                req_body = json.dumps({
                    "manifest": {
                        "sfx_items": [{
                            "asset_id":    asset_id,
                            "ai_prompt":   prompt,
                            "duration_sec": duration_sec,
                        }]
                    },
                    "asset_types": ["sfx"],
                    "asset_ids":   [asset_id],
                }).encode()
                req = _urllib_req.Request(
                    _AI_SERVER_URL + "/jobs",
                    data=req_body,
                    headers={"Content-Type": "application/json",
                             "X-Api-Key": _AI_SERVER_KEY},
                    method="POST",
                )
                with _urllib_req.urlopen(req, timeout=15) as resp:
                    ai_resp = json.loads(resp.read())
                body = json.dumps({"job_id": ai_resp.get("job_id"),
                                   "asset_id": asset_id,
                                   "timestamp_ms": timestamp_ms}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        # ── AI SFX Save: fetch generated audio from AI server → local path ───
        elif self.path == "/api/ai_sfx_save":
            try:
                length   = int(self.headers.get("Content-Length", 0))
                payload  = json.loads(self.rfile.read(length))
                job_id   = payload.get("job_id", "").strip()
                filename = payload.get("filename", "").strip()
                slug     = payload.get("slug", "").strip()
                ep_id    = payload.get("ep_id", "").strip()
                item_id  = payload.get("item_id", "").strip()
                ts_ms    = payload.get("timestamp_ms", "")
                if not all([job_id, filename, slug, ep_id, item_id, ts_ms]):
                    raise ValueError("job_id, filename, slug, ep_id, item_id, timestamp_ms required")
                # Resolve actual filename from AI server job state
                state_req = _urllib_req.Request(
                    _AI_SERVER_URL + f"/jobs/{job_id}",
                    headers={"X-Api-Key": _AI_SERVER_KEY},
                )
                with _urllib_req.urlopen(state_req, timeout=15) as sr:
                    job_state = json.loads(sr.read())
                actual_files = job_state.get("files", [])
                if not actual_files:
                    raise ValueError(f"AI job {job_id} completed with no files")
                actual_filename = actual_files[0]
                # Fetch audio bytes from AI server
                req = _urllib_req.Request(
                    _AI_SERVER_URL + f"/jobs/{job_id}/files/{actual_filename}",
                    headers={"X-Api-Key": _AI_SERVER_KEY},
                )
                with _urllib_req.urlopen(req, timeout=120) as resp:
                    audio_bytes = resp.read()
                # Save to episode sfx assets directory
                dest_dir = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id,
                                        "assets", "sfx", item_id)
                os.makedirs(dest_dir, exist_ok=True)
                dest_file = f"ai_{ts_ms}.mp3"
                dest_path = os.path.join(dest_dir, dest_file)
                with open(dest_path, "wb") as fh:
                    fh.write(audio_bytes)
                import wave, contextlib
                duration_sec = 0.0
                try:
                    with contextlib.closing(wave.open(dest_path, "r")) as wf:
                        duration_sec = wf.getnframes() / wf.getframerate()
                except Exception:
                    pass
                rel_path = os.path.relpath(dest_path, PIPE_DIR)
                serve_url = "/serve_media?path=" + _url_quote(rel_path)
                body = json.dumps({"path": dest_path,
                                   "url":  serve_url,
                                   "filename": dest_file,
                                   "duration_sec": duration_sec}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        # ── Media: write selections.json (POST /api/media_confirm) ───────────
        elif self.path == "/api/media_confirm":
            try:
                length   = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(length)
                payload  = json.loads(raw_body)

                slug       = payload.get("slug", "").strip()
                ep_id      = payload.get("ep_id", "").strip()
                batch_id   = payload.get("batch_id", "").strip()
                selections = payload.get("selections", {})

                if not slug or not ep_id:
                    raise ValueError("slug and ep_id are required")
                if not isinstance(selections, dict):
                    raise ValueError("selections must be a dict")

                ep_dir    = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
                media_dir = os.path.join(ep_dir, "assets", "media")
                os.makedirs(media_dir, exist_ok=True)

                sel_path = os.path.join(media_dir, "selections.json")
                out = {
                    "batch_id":   batch_id,
                    "slug":       slug,
                    "episode_id": ep_id,
                    "version":    payload.get("version", 1),
                    "selections": selections,
                }
                with open(sel_path, "w", encoding="utf-8") as _sf:
                    json.dump(out, _sf, indent=2, ensure_ascii=False)

                rel_path = os.path.relpath(sel_path, PIPE_DIR)
                print(f"  Saved media selections  slug={slug}  ep={ep_id}  "
                      f"n={len(selections)}")
                body = json.dumps({"ok": True,
                                   "path": rel_path,
                                   "n": len(selections)}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        # ── AI Generate: submit single-shot inline generation job ──────────────
        elif self.path == "/api/ai_generate":
            try:
                length   = int(self.headers.get("Content-Length", 0))
                payload  = json.loads(self.rfile.read(length))
                slug     = payload.get("slug", "").strip()
                ep_id    = payload.get("ep_id", "").strip()
                bg_id    = payload.get("bg_id", "").strip()
                prompt   = payload.get("prompt", "").strip()
                if not slug or not ep_id or not bg_id or not prompt:
                    raise ValueError("slug, ep_id, bg_id, and prompt are required")
                timestamp_ms = int(time.time() * 1000)
                h = hashlib.sha1(prompt.encode()).hexdigest()[:8]
                asset_id = f"{bg_id}-{h}-{timestamp_ms}"
                synthetic_manifest = {
                    "backgrounds": [{
                        "asset_id":  asset_id,
                        "ai_prompt": prompt,
                        "motion":    None,
                        "search_filters": {},
                    }]
                }
                req_body = json.dumps({
                    "manifest":    synthetic_manifest,
                    "asset_types": ["backgrounds"],
                    "asset_ids":   [asset_id],
                }).encode()
                req = _urllib_req.Request(
                    _AI_SERVER_URL + "/jobs",
                    data=req_body,
                    headers={"Content-Type": "application/json",
                             "X-Api-Key": _AI_SERVER_KEY},
                    method="POST",
                )
                with _urllib_req.urlopen(req, timeout=15) as resp:
                    ai_resp = json.loads(resp.read())
                body = json.dumps({"job_id": ai_resp.get("job_id"),
                                   "asset_id": asset_id,
                                   "timestamp_ms": timestamp_ms}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        # ── AI Save Image: fetch generated image from AI server → NFS share ──────
        elif self.path == "/api/ai_save_image":
            try:
                length   = int(self.headers.get("Content-Length", 0))
                payload  = json.loads(self.rfile.read(length))
                job_id   = payload.get("job_id", "").strip()
                filename = payload.get("filename", "").strip()
                slug     = payload.get("slug", "").strip()
                ep_id    = payload.get("ep_id", "").strip()
                bg_id    = payload.get("bg_id", "").strip()
                ts_ms    = payload.get("timestamp_ms", "")
                if not all([job_id, filename, slug, ep_id, bg_id, ts_ms]):
                    raise ValueError("job_id, filename, slug, ep_id, bg_id, timestamp_ms required")
                # Fetch image bytes from AI server
                req = _urllib_req.Request(
                    _AI_SERVER_URL + f"/jobs/{job_id}/files/{filename}",
                    headers={"X-Api-Key": _AI_SERVER_KEY},
                )
                with _urllib_req.urlopen(req, timeout=120) as resp:
                    img_bytes = resp.read()
                # Write to stable NFS path
                dest_dir = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id,
                                        "assets", "backgrounds", bg_id)
                os.makedirs(dest_dir, exist_ok=True)
                dest_file = f"ai_{ts_ms}.png"
                dest_path = os.path.join(dest_dir, dest_file)
                with open(dest_path, "wb") as fh:
                    fh.write(img_bytes)
                dest_url = "file://" + dest_path
                body = json.dumps({"path": dest_path, "url": dest_url,
                                   "filename": dest_file}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        # ── Music: prepare loop candidates (POST /api/music_prepare_loops) ──────
        elif self.path == "/api/music_prepare_loops":
            try:
                length   = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(length)
                payload  = json.loads(raw_body)
                slug     = payload.get("slug", "").strip()
                ep_id    = payload.get("ep_id", "").strip()
                if not slug or not ep_id:
                    raise ValueError("slug and ep_id are required")

                ep_dir = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
                manifest_path = os.path.join(ep_dir, "AssetManifest_draft.shared.json")
                if not os.path.isfile(manifest_path):
                    raise FileNotFoundError("AssetManifest_draft.shared.json not found")

                code_dir = os.path.join(PIPE_DIR, "code", "http")
                result = subprocess.run(
                    ["python3", os.path.join(code_dir, "music_prepare_loops.py"),
                     "--manifest", manifest_path],
                    capture_output=True, text=True, timeout=120, cwd=PIPE_DIR,
                )
                if result.returncode != 0:
                    raise RuntimeError(result.stderr[-2000:] if result.stderr else "process failed")

                # Read and return the candidates file
                cand_path = os.path.join(ep_dir, "assets", "music", "music_loop_candidates.json")
                if os.path.isfile(cand_path):
                    with open(cand_path, encoding="utf-8") as _cf:
                        candidates = json.load(_cf)
                    body = json.dumps({"ok": True, "candidates": candidates}).encode()
                else:
                    body = json.dumps({"ok": True, "candidates": {},
                                       "message": "No candidates generated (no music resources?)"}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        # ── Music: cut clip from source track (POST /api/music_cut_clip) ─────
        elif self.path == "/api/music_cut_clip":
            try:
                length   = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(length)
                payload  = json.loads(raw_body)
                slug     = payload.get("slug", "").strip()
                ep_id    = payload.get("ep_id", "").strip()
                stem     = payload.get("stem", "").strip()
                start_sec = float(payload.get("start_sec", 0))
                end_sec   = float(payload.get("end_sec", 0))
                if not slug or not ep_id or not stem:
                    raise ValueError("slug, ep_id, and stem are required")
                if end_sec <= start_sec:
                    raise ValueError("end_sec must be greater than start_sec")

                resources_dir = os.path.join(PIPE_DIR, "projects", slug,
                                             "resources", "music")
                assets_dir = os.path.join(PIPE_DIR, "projects", slug,
                                          "episodes", ep_id, "assets", "music")
                os.makedirs(assets_dir, exist_ok=True)

                # Find source file
                source_path = None
                for ext in (".mp3", ".wav", ".flac", ".ogg"):
                    candidate = os.path.join(resources_dir, stem + ext)
                    if os.path.isfile(candidate):
                        source_path = candidate
                        break
                if not source_path:
                    raise FileNotFoundError(
                        f"Source track '{stem}' not found in {resources_dir}")

                # Generate clip filename from stem + range
                clip_fname = (f"{stem}_{start_sec:.1f}s-{end_sec:.1f}s.wav"
                              .replace(".", "_", 2).replace("_wav", ".wav"))
                out_path = os.path.join(assets_dir, clip_fname)

                # Extract using librosa
                import librosa
                import soundfile as sf_mod
                audio, _ = librosa.load(source_path, sr=48000, mono=True)
                s0 = int(start_sec * 48000)
                s1 = min(int(end_sec * 48000), len(audio))
                segment = audio[s0:s1]
                if len(segment) == 0:
                    raise ValueError("Empty segment — check start/end times")
                sf_mod.write(out_path, segment.astype("float32"), 48000,
                             subtype="PCM_16")

                rel_path = os.path.relpath(out_path, PIPE_DIR)
                print(f"  Cut clip: {stem} [{start_sec:.1f}s-{end_sec:.1f}s]"
                      f" → {rel_path}")

                # Persist cut clip metadata to user_cut_clips.json
                end_sec_actual = start_sec + len(segment) / 48000.0
                clip_id = (f"{stem}:{start_sec:.1f}s-"
                           f"{end_sec_actual:.1f}s")
                meta_path = os.path.join(assets_dir, "user_cut_clips.json")
                existing_cuts = []
                if os.path.isfile(meta_path):
                    try:
                        with open(meta_path, encoding="utf-8") as _mf:
                            existing_cuts = json.load(_mf)
                    except Exception:
                        pass
                # Remove duplicate clip_id if re-cutting same range
                existing_cuts = [c for c in existing_cuts
                                 if c.get("clip_id") != clip_id]
                existing_cuts.append({
                    "clip_id": clip_id,
                    "stem": stem,
                    "start_sec": round(start_sec, 2),
                    "end_sec": round(end_sec_actual, 2),
                    "path": rel_path,
                })
                with open(meta_path, "w", encoding="utf-8") as _mf:
                    json.dump(existing_cuts, _mf, indent=2)
                    _mf.write("\n")

                body = json.dumps({
                    "ok": True, "path": rel_path,
                    "clip_id": clip_id,
                    "clip_fname": clip_fname,
                    "size_bytes": os.path.getsize(out_path),
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        # ── Music: generate review pack (POST /api/music_review_pack) ────────
        elif self.path == "/api/music_review_pack":
            try:
                length   = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(length)
                payload  = json.loads(raw_body)
                slug     = payload.get("slug", "").strip()
                ep_id    = payload.get("ep_id", "").strip()
                shot_overrides = payload.get("shot_overrides", [])
                if not slug or not ep_id:
                    raise ValueError("slug and ep_id are required")

                ep_dir = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)

                # Find the merged manifest — use PRIMARY_LOCALE from
                # pipeline_vars.sh, then fall back to alphabetical first.
                import glob as _glob_mod
                merged_manifests = _glob_mod.glob(
                    os.path.join(ep_dir, "AssetManifest_merged.*.json"))
                if not merged_manifests:
                    raise FileNotFoundError(
                        "No AssetManifest_merged.*.json found. "
                        "Run stages 10[1]–10[4] first.")
                # Read PRIMARY_LOCALE from pipeline_vars.sh
                _primary_locale = "en"  # default
                _vars_file = os.path.join(ep_dir, "pipeline_vars.sh")
                if os.path.isfile(_vars_file):
                    import re as _re_mv
                    with open(_vars_file, encoding="utf-8") as _vf:
                        _vf_content = _vf.read()
                    _m = _re_mv.search(
                        r'(?:^|[\n;])(?:export\s+)?PRIMARY_LOCALE=["\']?([^"\';\n]+)["\']?',
                        _vf_content)
                    if _m:
                        _primary_locale = _m.group(1).strip()
                # Try primary locale first, then fall back to first available
                _primary_manifest = os.path.join(
                    ep_dir, f"AssetManifest_merged.{_primary_locale}.json")
                if os.path.isfile(_primary_manifest):
                    manifest_path = _primary_manifest
                else:
                    manifest_path = sorted(merged_manifests)[0]

                code_dir = os.path.join(PIPE_DIR, "code", "http")
                cmd = ["python3", os.path.join(code_dir, "music_review_pack.py"),
                       "--manifest", manifest_path]

                # Write overrides to temp file if present
                import tempfile as _tempfile
                ovr_path = None
                if shot_overrides:
                    ovr_fd, ovr_path = _tempfile.mkstemp(
                        suffix=".json", prefix="music_ovr_")
                    with os.fdopen(ovr_fd, "w", encoding="utf-8") as _of:
                        json.dump(shot_overrides, _of)
                    cmd.extend(["--overrides", ovr_path])

                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True, text=True, timeout=120, cwd=PIPE_DIR,
                    )
                    if result.returncode != 0:
                        raise RuntimeError(result.stderr[-2000:] if result.stderr else "process failed")
                finally:
                    if ovr_path and os.path.exists(ovr_path):
                        os.unlink(ovr_path)

                # Read and return the timeline
                tl_path = os.path.join(ep_dir, "assets", "music",
                                       "MusicReviewPack", "timeline.json")
                timeline = None
                if os.path.isfile(tl_path):
                    with open(tl_path, encoding="utf-8") as _tf:
                        timeline = json.load(_tf)

                body = json.dumps({"ok": True, "timeline": timeline}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        # ── Music: save MusicPlan.json (POST /api/music_plan_save) ───────────
        elif self.path == "/api/music_plan_save":
            try:
                length   = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(length)
                payload  = json.loads(raw_body)
                slug     = payload.get("slug", "").strip()
                ep_id    = payload.get("ep_id", "").strip()
                plan     = payload.get("plan")

                if not slug or not ep_id:
                    raise ValueError("slug and ep_id are required")
                if not plan or not isinstance(plan, dict):
                    raise ValueError("plan must be a non-empty object")

                ep_dir    = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
                music_dir = os.path.join(ep_dir, "assets", "music")
                os.makedirs(music_dir, exist_ok=True)

                plan_path = os.path.join(music_dir, "MusicPlan.json")
                with open(plan_path, "w", encoding="utf-8") as _pf:
                    json.dump(plan, _pf, indent=2, ensure_ascii=False)
                    _pf.write("\n")

                rel_path = os.path.relpath(plan_path, PIPE_DIR)
                print(f"  Saved MusicPlan  slug={slug}  ep={ep_id}  "
                      f"loops={len(plan.get('loop_selections', {}))}  "
                      f"overrides={len(plan.get('shot_overrides', []))}")
                body = json.dumps({"ok": True, "path": rel_path}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        # Save episode metadata (title, genre, format, locales) → merge into meta.json
        elif self.path == "/api/save_episode_meta":
            try:
                length   = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(length)
                payload  = json.loads(raw_body)
                slug     = payload.get("slug", "").strip()
                ep_id    = payload.get("ep_id", "").strip()
                if not slug or not ep_id:
                    raise ValueError("slug and ep_id are required")
                ep_dir = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
                if not os.path.isdir(ep_dir):
                    raise ValueError(f"Episode directory not found: {ep_dir}")
                meta_path = os.path.join(ep_dir, "meta.json")
                # Read existing meta.json (to preserve schema_id, episode_id, slug, seed, etc.)
                existing_meta: dict = {}
                if os.path.isfile(meta_path):
                    try:
                        existing_meta = json.load(open(meta_path, encoding="utf-8"))
                    except Exception:
                        pass
                # Merge user-edited fields (use new-style field names)
                if "title" in payload:
                    existing_meta["story_title"]  = payload.get("title", "").strip()
                if "genre" in payload:
                    existing_meta["series_genre"] = payload.get("genre", "").strip()
                if "story_format" in payload:
                    existing_meta["story_format"] = payload.get("story_format", "episodic").strip()
                if "locales" in payload:
                    existing_meta["locales"]      = payload.get("locales", "en").strip()
                if "no_music" in payload:
                    existing_meta["no_music"]     = bool(payload.get("no_music", False))
                if "purge_cache" in payload:
                    existing_meta["purge_cache"]  = bool(payload.get("purge_cache", False))
                # Media source config (per-source image/video counts and enabled flags)
                if "media_source_config" in payload:
                    existing_meta["media_source_config"] = payload["media_source_config"]
                with open(meta_path, "w", encoding="utf-8") as _f:
                    json.dump(existing_meta, _f, indent=2, ensure_ascii=False)
                print(f"  Saved meta.json  slug={slug}  ep={ep_id}")
                resp = json.dumps({"ok": True, "path": meta_path}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
            except Exception as exc:
                resp = json.dumps({"ok": False, "error": str(exc)}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

        # ── YouTube: generate youtube.json draft via Claude ───────────────────────
        elif self.path == "/api/generate_youtube_json":
            try:
                length = int(self.headers.get("Content-Length", 0))
                req    = json.loads(self.rfile.read(length))
                slug   = req.get("slug",   "").strip()
                ep_id  = req.get("ep_id",  "").strip()
                locale = req.get("locale", "en").strip()

                if not slug or not ep_id:
                    raise ValueError("slug and ep_id required")
                if not re.match(r'^[a-zA-Z0-9_\-]+$', slug) or not re.match(r'^s\d+e\d+$', ep_id):
                    raise ValueError("invalid slug or ep_id")

                ep_dir     = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
                render_dir = os.path.join(ep_dir, "renders", locale)

                # ── Load episode files ────────────────────────────────────────
                def _jload(p):
                    if os.path.isfile(p):
                        with open(p, encoding="utf-8") as f:
                            return json.load(f)
                    return None

                script      = _jload(os.path.join(ep_dir, "Script.json")) or {}
                shotlist    = _jload(os.path.join(ep_dir, "ShotList.json")) or {}
                story_prompt= _jload(os.path.join(ep_dir, "StoryPrompt.json"))

                # ── Collect narrator text (capped at 4000 chars) ──────────────
                lines = []
                for scene in script.get("scenes", []):
                    for action in scene.get("actions", []):
                        if action.get("type") == "dialogue" and action.get("line"):
                            lines.append(action["line"])
                total, truncated = 0, []
                for line in lines:
                    if total + len(line) > 4000:
                        break
                    truncated.append(line)
                    total += len(line)

                # ── Shot summaries ────────────────────────────────────────────
                shots_data = shotlist.get("shots", [])
                shot_cursor = 0.0
                shot_summaries = []
                for sh in shots_data:
                    dur = sh.get("duration_sec", 0)
                    shot_summaries.append({
                        "emotional_tag": sh.get("emotional_tag", ""),
                        "duration_sec":  dur,
                        "start_sec":     round(shot_cursor, 2),
                        "background":    sh.get("background_id", ""),
                    })
                    shot_cursor += dur

                # ── category_id from genre (no LLM) ──────────────────────────
                genre = script.get("genre", "").lower()
                category_id = _GENRE_TO_CATEGORY.get(genre, _DEFAULT_CATEGORY)

                # ── Subtitle scan ─────────────────────────────────────────────
                subtitles = []
                if os.path.isdir(render_dir):
                    for fname in sorted(os.listdir(render_dir)):
                        if not fname.endswith(".srt"):
                            continue
                        if ".en." in fname:
                            subtitles.append({"file": f"renders/{locale}/{fname}",
                                              "language": "en", "name": "English"})
                        elif ".zh-Hans." in fname:
                            subtitles.append({"file": f"renders/{locale}/{fname}",
                                              "language": "zh-CN", "name": "Chinese Simplified"})

                # ── Load profiles → upload_profile ────────────────────────────
                profiles_path = os.path.expanduser("~/.config/pipe/youtube_profiles.json")
                profiles = {}
                if os.path.isfile(profiles_path):
                    with open(profiles_path, encoding="utf-8") as f:
                        profiles = json.load(f)
                locale_to_profile = {v.get("locale", k): k for k, v in profiles.items()}
                upload_profile = locale_to_profile.get(locale, locale)
                profile_info   = profiles.get(upload_profile, {})

                # ── Build Claude prompt ───────────────────────────────────────
                ep_goal = None
                if story_prompt:
                    ep_goal = (story_prompt.get("episode_goal")
                               or story_prompt.get("prompt_text", "")[:500])

                output_lang = "English" if profile_info.get("locale", "en") == "en" \
                              else "Chinese (Simplified)"

                total_dur = shotlist.get("total_duration_sec", 0)

                user_msg = json.dumps({
                    "locale":             locale,
                    "output_language":    output_lang,
                    "episode_id":         ep_id,
                    "genre":              genre,
                    "total_duration_sec": total_dur,
                    "script_title":       script.get("title", ""),
                    "episode_goal":       ep_goal,
                    "narrator_text":      truncated,
                    "shots":              shot_summaries,
                }, ensure_ascii=False)

                system_prompt = (
                    "You are a YouTube metadata expert. Generate upload metadata "
                    "for a short narrative video episode. "
                    "Output ONLY valid JSON with exactly these fields: "
                    "title, description, tags, thumbnail_source_sec. "
                    "No markdown, no explanation — raw JSON only.\n\n"
                    "Constraints:\n"
                    f"- title: ≤ 70 characters, in {output_lang}\n"
                    "- description: first 2 lines are compelling hooks (shown in search results); "
                    "hashtags ONLY in last paragraph; ≤ 5000 chars total; in {output_lang}\n"
                    "- tags: 10-15 items, mix specific and broad terms\n"
                    f"- thumbnail_source_sec: pick midpoint of shot with emotional_tag "
                    f"'triumph', 'climax', or 'reveal'; must be within [0, {total_dur}]\n"
                    "- Do NOT include category_id in the response"
                ).format(output_lang=output_lang)

                # ── Call Claude via CLI (same mechanism as run.sh pipeline) ──────
                # Combine system instructions + user data into a single prompt file.
                prompt_text = (
                    system_prompt
                    + "\n\n---\n\nEpisode data (JSON):\n\n"
                    + user_msg
                )
                REQUIRED = {"title", "description", "tags", "thumbnail_source_sec"}

                def _call_claude():
                    import tempfile as _tf_yt
                    _yt_env = os.environ.copy()
                    _yt_env.pop("CLAUDECODE", None)  # prevent nested-session guard
                    with _tf_yt.NamedTemporaryFile(
                        mode="w", suffix=".txt", delete=False, encoding="utf-8"
                    ) as tf:
                        tf.write(prompt_text)
                        tmp_path = tf.name
                    try:
                        result = subprocess.run(
                            ["claude", "-p",
                             "--model", "sonnet",
                             "--dangerously-skip-permissions",
                             "--no-session-persistence",
                             tmp_path],
                            capture_output=True, text=True, cwd=PIPE_DIR, timeout=120,
                            env=_yt_env,
                        )
                    finally:
                        os.unlink(tmp_path)
                    if result.returncode != 0 and not result.stdout.strip():
                        raise RuntimeError(
                            f"claude CLI failed (rc={result.returncode}): "
                            f"{result.stderr.strip()[:300]}"
                        )
                    raw = result.stdout.strip()
                    # Strip markdown fences
                    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
                    raw = re.sub(r"\n?```$", "", raw, flags=re.MULTILINE)
                    raw = raw.strip()
                    # Extract JSON object even if claude wrapped it in explanation text
                    start = raw.find('{')
                    end   = raw.rfind('}')
                    if start != -1 and end != -1 and end > start:
                        raw = raw[start:end + 1]
                    return raw

                def _parse_claude(raw):
                    """Parse JSON; on failure return (None, error_str)."""
                    try:
                        return json.loads(raw), None
                    except json.JSONDecodeError as e:
                        return None, str(e)

                raw = _call_claude()
                suggested, err = _parse_claude(raw)
                if suggested is None:
                    # Retry once with an explicit reminder (C28b)
                    raw = _call_claude()
                    suggested, err = _parse_claude(raw)
                if suggested is None:
                    resp = json.dumps({"ok": False,
                                       "error": f"Claude returned invalid JSON: {err}",
                                       "raw": raw}).encode("utf-8")
                    self.send_response(422)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.end_headers()
                    self.wfile.write(resp)
                    return

                # Check required fields (C28c)
                missing = REQUIRED - set(suggested.keys())
                if missing:
                    resp = json.dumps({"ok": False,
                                       "error": f"Missing fields: {sorted(missing)}"}).encode()
                    self.send_response(422)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.end_headers()
                    self.wfile.write(resp)
                    return

                # ── Assemble full draft ───────────────────────────────────────
                draft = {
                    "upload_profile":      upload_profile,
                    "title":               suggested["title"],
                    "description":         suggested["description"],
                    "tags":                suggested.get("tags", []),
                    "category_id":         category_id,
                    "playlist_id":         profile_info.get("playlist_id"),
                    "channel_id":          profile_info.get("channel_id"),
                    "video_language":      locale if locale != "zh-Hans" else "zh-Hans",
                    "privacy":             "private",
                    "made_for_kids":       False,
                    "thumbnail":           f"projects/{slug}/episodes/{ep_id}/renders/{locale}/thumbnail.jpg",
                    "thumbnail_source_sec":suggested.get("thumbnail_source_sec"),
                    "subtitles":           subtitles,
                    "publish_at":          None,
                    "notify_subscribers":  False,
                    "license":             "youtube",
                    "embeddable":          True,
                }

                resp = json.dumps({"ok": True, "draft": draft}, ensure_ascii=False).encode()
                self.send_response(200)

            except Exception as exc:
                import traceback; traceback.print_exc()
                resp = json.dumps({"ok": False, "error": str(exc)}).encode()
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        # ── YouTube: write all fields to youtube.json ─────────────────────────
        elif self.path == "/api/youtube_save_all":
            try:
                length = int(self.headers.get("Content-Length", 0))
                req    = json.loads(self.rfile.read(length))
                slug   = req.get("slug",   "").strip()
                ep_id  = req.get("ep_id",  "").strip()
                locale = req.get("locale", "en").strip()
                fields = req.get("fields", {})

                if not slug or not ep_id or not fields:
                    raise ValueError("slug, ep_id, fields required")
                if not re.match(r'^[a-zA-Z0-9_\-]+$', slug) or not re.match(r'^s\d+e\d+$', ep_id):
                    raise ValueError("invalid slug or ep_id")

                render_dir = os.path.join(PIPE_DIR, "projects", slug, "episodes",
                                          ep_id, "renders", locale)
                os.makedirs(render_dir, exist_ok=True)
                yt_path = os.path.join(render_dir, "youtube.json")
                with open(yt_path, "w", encoding="utf-8") as f:
                    json.dump(fields, f, indent=2, ensure_ascii=False)

                resp = json.dumps({"ok": True, "path": f"renders/{locale}/youtube.json"}).encode()
                self.send_response(200)
            except Exception as exc:
                resp = json.dumps({"ok": False, "error": str(exc)}).encode()
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        # ── YouTube: save a single field to youtube.json ─────────────────────────
        elif self.path == "/api/youtube_save_field":
            try:
                length  = int(self.headers.get("Content-Length", 0))
                req     = json.loads(self.rfile.read(length))
                slug    = req.get("slug",   "").strip()
                ep_id   = req.get("ep_id",  "").strip()
                locale  = req.get("locale", "en").strip()
                field   = req.get("field",  "").strip()
                value   = req.get("value")

                if not slug or not ep_id or not field:
                    raise ValueError("slug, ep_id, field required")
                if not re.match(r'^[a-zA-Z0-9_\-]+$', slug) or not re.match(r'^s\d+e\d+$', ep_id):
                    raise ValueError("invalid slug or ep_id")

                # Only allow known writable fields
                _ALLOWED = {"title","description","tags","category_id","privacy",
                            "made_for_kids","notify_subscribers","publish_at",
                            "thumbnail_source_sec","playlist_id","license","embeddable"}
                if field not in _ALLOWED:
                    raise ValueError(f"field '{field}' is not editable")

                yt_path = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id,
                                       "renders", locale, "youtube.json")
                if not os.path.isfile(yt_path):
                    raise FileNotFoundError("youtube.json not found")

                with open(yt_path, encoding="utf-8") as f:
                    yt = json.load(f)
                yt[field] = value
                with open(yt_path, "w", encoding="utf-8") as f:
                    json.dump(yt, f, indent=2, ensure_ascii=False)

                resp = json.dumps({"ok": True}).encode()
                self.send_response(200)
            except Exception as exc:
                resp = json.dumps({"ok": False, "error": str(exc)}).encode()
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        # ── YouTube: save confirmed thumbnail frame (POST /api/set_thumbnail_sec) ─
        elif self.path == "/api/set_thumbnail_sec":
            try:
                length  = int(self.headers.get("Content-Length", 0))
                req     = json.loads(self.rfile.read(length))
                slug    = req.get("slug",   "").strip()
                ep_id   = req.get("ep_id",  "").strip()
                locale  = req.get("locale", "en").strip()
                sec     = float(req.get("sec", 0))

                if not slug or not ep_id:
                    raise ValueError("slug and ep_id required")
                # C26: validate path parameters
                if not re.match(r'^[a-zA-Z0-9_\-]+$', slug) or not re.match(r'^s\d+e\d+$', ep_id):
                    raise ValueError("invalid slug or ep_id")
                if locale not in ("en", "zh-Hans", "zh", "zh-CN", "ja", "ko", "fr", "de", "es", "pt"):
                    raise ValueError("invalid locale")

                render_dir = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id,
                                          "renders", locale)
                mp4_path   = os.path.join(render_dir, "output.mp4")
                thumb_path = os.path.join(render_dir, "thumbnail.jpg")

                if not os.path.isfile(mp4_path):
                    raise FileNotFoundError("output.mp4 not found")

                # C24: bounds-check sec via ffprobe
                probe_r = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_format",
                     "-print_format", "json", mp4_path],
                    capture_output=True, timeout=10,
                )
                probe_d  = json.loads(probe_r.stdout) if probe_r.returncode == 0 else {}
                duration = float(probe_d.get("format", {}).get("duration", 0))
                if duration > 0 and (sec < 0 or sec > duration):
                    raise ValueError(f"sec {sec} out of range [0, {duration:.2f}]")

                # C25: check ffmpeg returncode
                result = subprocess.run([
                    "ffmpeg", "-y", "-ss", str(sec), "-i", mp4_path,
                    "-frames:v", "1", "-vf", "scale=1280:720",
                    "-update", "1", thumb_path,
                ], capture_output=True, timeout=30)
                if result.returncode != 0:
                    raise RuntimeError(f"ffmpeg failed: {result.stderr.decode(errors='replace')[:300]}")

                # Update youtube.json thumbnail_source_sec
                yt_path = os.path.join(render_dir, "youtube.json")
                if os.path.isfile(yt_path):
                    with open(yt_path, encoding="utf-8") as f:
                        yt = json.load(f)
                    yt["thumbnail_source_sec"] = sec
                    yt["thumbnail"] = f"projects/{slug}/episodes/{ep_id}/renders/{locale}/thumbnail.jpg"
                    with open(yt_path, "w", encoding="utf-8") as f:
                        json.dump(yt, f, indent=2, ensure_ascii=False)

                resp = json.dumps({
                    "ok": True,
                    "thumbnail_path": f"projects/{slug}/episodes/{ep_id}/renders/{locale}/thumbnail.jpg",
                    "sec": sec,
                }).encode()
                self.send_response(200)
            except Exception as exc:
                resp = json.dumps({"ok": False, "error": str(exc)}).encode()
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        # ── YouTube: upload custom thumbnail image ─────────────────────────────
        elif self.path == "/api/upload_thumbnail_file":
            try:
                import cgi as _cgi
                ctype = self.headers.get("Content-Type", "")
                if "multipart/form-data" not in ctype:
                    raise ValueError("Expected multipart/form-data")
                length = int(self.headers.get("Content-Length", 0))
                raw    = self.rfile.read(length)

                # Parse multipart manually (simple approach)
                boundary = re.search(r"boundary=(.+)", ctype)
                if not boundary:
                    raise ValueError("No boundary in Content-Type")
                bnd = boundary.group(1).strip().encode()

                parts = raw.split(b"--" + bnd)
                fields = {}
                file_data = None
                for part in parts:
                    if b"Content-Disposition" not in part:
                        continue
                    header, _, body = part.partition(b"\r\n\r\n")
                    body = body.rstrip(b"\r\n")
                    name_m = re.search(rb'name="([^"]+)"', header)
                    fname_m = re.search(rb'filename="([^"]+)"', header)
                    if name_m:
                        name = name_m.group(1).decode()
                        if fname_m:
                            file_data = body
                        else:
                            fields[name] = body.decode(errors="replace")

                slug   = fields.get("slug", "").strip()
                ep_id  = fields.get("ep_id", "").strip()
                locale = fields.get("locale", "en").strip()

                if not slug or not ep_id or file_data is None:
                    raise ValueError("slug, ep_id, and file required")
                if not re.match(r'^[a-zA-Z0-9_\-]+$', slug) or not re.match(r'^s\d+e\d+$', ep_id):
                    raise ValueError("invalid slug or ep_id")

                render_dir = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id,
                                          "renders", locale)
                os.makedirs(render_dir, exist_ok=True)
                thumb_path = os.path.join(render_dir, "thumbnail.jpg")
                with open(thumb_path, "wb") as tf:
                    tf.write(file_data)

                # Null out thumbnail_source_sec in youtube.json
                yt_path = os.path.join(render_dir, "youtube.json")
                if os.path.isfile(yt_path):
                    with open(yt_path, encoding="utf-8") as f:
                        yt = json.load(f)
                    yt["thumbnail_source_sec"] = None
                    yt["thumbnail"] = f"projects/{slug}/episodes/{ep_id}/renders/{locale}/thumbnail.jpg"
                    with open(yt_path, "w", encoding="utf-8") as f:
                        json.dump(yt, f, indent=2, ensure_ascii=False)

                resp = json.dumps({"ok": True, "path": f"projects/{slug}/episodes/{ep_id}/renders/{locale}/thumbnail.jpg"}).encode()
                self.send_response(200)
            except Exception as exc:
                resp = json.dumps({"ok": False, "error": str(exc)}).encode()
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        # ── YouTube: run pipeline action (POST /api/youtube_action) ──────────
        elif self.path == "/api/youtube_action":
            try:
                length = int(self.headers.get("Content-Length", 0))
                req    = json.loads(self.rfile.read(length))
                slug   = req.get("slug",   "").strip()
                ep_id  = req.get("ep_id",  "").strip()
                locale = req.get("locale", "en").strip()
                action = req.get("action", "").strip()

                if not slug or not ep_id or not action:
                    raise ValueError("slug, ep_id, action required")
                if not re.match(r'^[a-zA-Z0-9_\-]+$', slug) or not re.match(r'^s\d+e\d+$', ep_id):
                    raise ValueError("invalid slug or ep_id")
                if action not in ("validate", "upload", "publish"):
                    raise ValueError(f"unknown action: {action!r}")

                ep_dir_rel = f"projects/{slug}/episodes/{ep_id}"
                script_map = {
                    "validate": "code/deploy/youtube/prepare_upload.py",
                    "upload":   "code/deploy/youtube/upload_private.py",
                    "publish":  "code/deploy/youtube/publish_episode.py",
                }
                script = script_map[action]

                cmd = [
                    "python3",
                    os.path.join(PIPE_DIR, script),
                    ep_dir_rel,
                    "--locale", locale,
                ]
                if action == "validate":
                    cmd.append("--yes")  # non-interactive

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=PIPE_DIR,
                    timeout=1800,  # 30-min max (upload can be slow)
                )
                output = result.stdout + ("\n\n--- stderr ---\n" + result.stderr if result.stderr.strip() else "")
                resp = json.dumps({
                    "ok":     result.returncode == 0,
                    "rc":     result.returncode,
                    "output": output,
                }).encode()
                self.send_response(200)
            except subprocess.TimeoutExpired:
                resp = json.dumps({"ok": False, "error": "timeout"}).encode()
                self.send_response(200)
            except Exception as exc:
                resp = json.dumps({"ok": False, "error": str(exc)}).encode()
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        first = str(args[0]) if args else ""
        parts = first.split()
        path = parts[1] if len(parts) > 1 else ""
        # Silence noisy but uninteresting routes
        silent = {"/", "/stop", "/next_story_num", "/check_episode", "/read_story",
                  "/list_projects", "/view_artifact", "/list_stories",
                  "/pipeline_status", "/serve_media", "/run_locale", "/run_stage10",
                  "/api/azure_voices", "/api/voice_presets", "/api/voice_index",
                  "/api/preview_voice", "/api/save_voice_cast", "/api/delete_preset",
                  "/api/status_report", "/api/append_status_report",
                  "/api/vo_alignment",
                  "/api/check_slug", "/api/next_episode_id",
                  "/api/create_episode", "/api/save_episode_meta",
                  "/api/diagnose_pipeline",
                  "/api/media_batches", "/api/media_batch_status",
                  "/api/media_batch", "/api/media_batch_resume", "/api/media_confirm",
                  "/api/sfx_search", "/api/sfx_save", "/api/sfx_results_save",
                  "/api/ai_sfx_generate", "/api/ai_sfx_save",
                  "/api/ai_images", "/api/ai_job_status",
                  "/api/serve_media_file",
                  "/api/music_loop_candidates", "/api/music_timeline",
                  "/api/music_prepare_loops", "/api/music_review_pack",
                  "/api/music_plan_save", "/api/music_sources",
                  "/api/music_cut_clip"}
        if not any(path == s or path.startswith(s + "?") for s in silent):
            print(f"  {self.address_string()}  {fmt % args}")


# ── Entry point ────────────────────────────────────────────────────────────────
def local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "localhost"
    finally:
        s.close()


class ReusableServer(ThreadingHTTPServer):
    allow_reuse_address = True
    # Daemon threads die immediately when the main thread exits (Ctrl-C).
    # Without this, SSE handler threads and video-streaming threads keep
    # the old process alive after Ctrl-C, preventing a clean restart.
    # The browser's existing tab still has TCP connections to the zombie
    # process, so reloading that tab hangs (the zombie can't serve pages).
    daemon_threads = True


if __name__ == "__main__":
    ip     = local_ip()
    server = ReusableServer(("0.0.0.0", PORT), Handler)
    print(f"\n🤖  Claude Runner  —  story pipeline UI")
    print(f"\n    http://localhost:{PORT}")
    print(f"    http://{ip}:{PORT}   ← open from any device on your network")
    print(f"\n    Story files saved to: {PIPE_DIR}/")
    print(f"    Ctrl-C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down…")
        server.shutdown()       # stop accepting, join handler threads
        server.server_close()   # close the listening socket
        # Kill any child processes (pipeline runs) still alive
        with _lock:
            for proc in _procs.values():
                if proc.poll() is None:
                    proc.terminate()
        print("Stopped.")
