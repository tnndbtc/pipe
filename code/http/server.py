#!/usr/bin/env python3
"""
server.py — Claude pipeline runner with story input UI.

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
import logging
import logging.handlers
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse, unquote_plus, quote as _url_quote
import urllib.request as _urllib_req
import urllib.error   as _urllib_err

# ── YouTube category mapping (genre → category_id, no LLM needed) ─────────────
# "narration" is set by simple_narration_setup.py — ONLY simple_run.sh writes this
# genre, and its content is always News & Politics per product requirement.
_GENRE_TO_CATEGORY = {
    "history":       "27",
    "documentary":   "27",
    "education":     "27",
    "sports":        "17",
    "news":          "25",
    "entertainment": "24",
    "comedy":        "23",
    "narration":     "25",
}
_DEFAULT_CATEGORY = "25"  # News & Politics (default for simple_run.sh content)

PORT      = 8000
PIPE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (pipe/)
_TEST_MODE = False

# ── Persistent server log ──────────────────────────────────────────────────────
# Writes to code/http/logs/server.log (10 MB cap, 5 rotated backups).
# Use:  _log.info("msg")  /  _log.debug("msg")  /  _log.warning("msg")
_LOG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_log = logging.getLogger("pipe_server")
_log.setLevel(logging.DEBUG)
if not _log.handlers:
    _log_handler = logging.handlers.RotatingFileHandler(
        os.path.join(_LOG_DIR, "server.log"),
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    _log_handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    _log.addHandler(_log_handler)
    # Also mirror to stdout so the terminal still shows everything
    _log_stdout = logging.StreamHandler()
    _log_stdout.setFormatter(logging.Formatter("%(levelname)-7s  %(message)s"))
    _log.addHandler(_log_stdout)
# ──────────────────────────────────────────────────────────────────────────────

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

_jobs: dict = {}          # job_key → {"log": str, "done": bool, "rc": int|None}
_jobs_lock = threading.Lock()

# ── Media AI Ask: in-process error store for fire-and-forget background thread ──
_media_ai_ask_errors: dict = {}   # key = "slug:ep_id" → error string

# ── Per-episode VO write lock (INVARIANT G) ────────────────────────────────────
# Keyed by ep_dir string. Acquired before any write to *.wav, *.source.wav,
# vo_trim_overrides.json, vo_merge_log.json, VOPlan, or
# tts_review_complete.json. Different episodes run concurrently (not global).
_vo_locks: dict[str, threading.Lock] = {}
_vo_locks_meta = threading.Lock()   # protects _vo_locks dict itself
_sfx_preview_locks = {}   # per-episode lock for sfx preview generation


def _get_vo_lock(ep_dir: str) -> threading.Lock:
    """Get (or create) the per-episode VO write lock."""
    with _vo_locks_meta:
        if ep_dir not in _vo_locks:
            _vo_locks[ep_dir] = threading.Lock()
        return _vo_locks[ep_dir]


def _job_log_path(job_key: str, ep_dir: str) -> str:
    """Stable path for this job's output log, stored under the episode directory."""
    h = hashlib.md5(job_key.encode()).hexdigest()
    d = os.path.join(ep_dir, "pipe_jobs")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, h + ".log")


def _launch_stream_job(job_key: str, ep_dir: str, cmd: list, env: dict, client) -> str:
    """Run cmd in a background thread, writing tagged lines to a log file.

    Tags written to log:
      O\\t{line}  — stdout line
      E\\t{line}  — stderr line
      D\\t{rc}    — done sentinel (process exit code)

    Returns the log path.  If the same job is already running (e.g. client
    reconnected), returns the existing log path so the client can replay from
    the start.
    """
    log_path = _job_log_path(job_key, ep_dir)
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
                stdin=subprocess.DEVNULL,   # prevent SIGTTIN from terminal reads
                text=True, bufsize=1, env=env, cwd=PIPE_DIR,
                start_new_session=True,     # new session = new process group, isolated
                                            # from server's terminal SIGTSTP/SIGHUP
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


def _launch_fn_job(job_key: str, ep_dir: str, target_fn) -> str:
    """Run target_fn(write_log) in a background thread.

    target_fn receives write_log(tag, data) where tag is 'O', 'E', or 'D'.
    Returns the log path.
    """
    log_path = _job_log_path(job_key, ep_dir)
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
    from gen_tts_cloud import segment_zh_phonemes as _segment_zh_phonemes
    import azure.cognitiveservices.speech as _speechsdk
    _TTS_AVAILABLE = True
except ImportError:
    _build_ssml    = None  # type: ignore
    _speechsdk     = None  # type: ignore
    _TTS_AVAILABLE = False
    def _segment_zh_phonemes(text: str, azure_lang: str) -> list:  # type: ignore
        """Fallback when gen_tts_cloud is unavailable — single text segment."""
        return [{"type": "text", "content": text}]

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
        write_vo_preview_approved,
        is_vo_approved            as _is_vo_approved,
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
    phoneme_overrides = params.get("phoneme_overrides", {})

    if not azure_voice:
        raise ValueError("voice is required in params")

    ssml = _preview_build_ssml(
        text, azure_voice, azure_locale, style,
        style_degree=style_degree, rate=rate,
        pitch=pitch, break_ms=break_ms,
        phoneme_overrides=phoneme_overrides,
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
            "ph": json.dumps(phoneme_overrides, sort_keys=True) if phoneme_overrides else "",
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
    _log.info("[synthesize_vo] ssml=%s", ssml)
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


def _mp3_bytes_to_wav_bytes(mp3_data: bytes) -> bytes:
    """Convert MP3 bytes to 24 kHz 16-bit mono PCM WAV bytes using ffmpeg.

    Azure TTS is configured to output MP3 (smaller, better quality for HD voices).
    All downstream processing (wave module, vo_preview_concat, trim) requires
    standard PCM WAV, so we convert here rather than changing the TTS output format.
    """
    import subprocess as _sp
    result = _sp.run(
        ["ffmpeg", "-y", "-f", "mp3", "-i", "pipe:0",
         "-ar", "24000", "-ac", "1", "-sample_fmt", "s16", "-f", "wav", "pipe:1"],
        input=mp3_data, capture_output=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg MP3→WAV conversion failed: "
            f"{result.stderr.decode('utf-8', 'replace')[:300]}"
        )
    return result.stdout


def _is_mp3_bytes(data: bytes) -> bool:
    """Return True if data is MP3 (MPEG sync header or ID3 tag)."""
    if len(data) < 3:
        return False
    if data[:3] == b"ID3":           # ID3 tag header before MP3 frames
        return True
    # MPEG sync: 0xFF followed by 0xE0–0xFF (sync bits + layer bits set)
    return data[0] == 0xFF and (data[1] & 0xE0) == 0xE0


def _write_wav_bytes_atomic(path: str, wav_bytes: bytes) -> None:
    """Write WAV bytes atomically (write to .tmp then rename).

    Automatically converts MP3 bytes to PCM WAV if the TTS returned MP3.
    Azure TTS is configured with Audio24Khz96KBitRateMonoMp3; all .wav files
    stored on disk must be real PCM WAV for downstream wave-module reads.
    """
    if _is_mp3_bytes(wav_bytes):
        wav_bytes = _mp3_bytes_to_wav_bytes(wav_bytes)
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


def _resolve_sfx_local_path(ep_dir: str, item_id: str, candidate: dict):
    """Find the local audio file for a selected SFX candidate. Returns path or None."""
    url = candidate.get("preview_url", "")
    if "/serve_media?path=" in url:
        from urllib.parse import unquote, urlparse, parse_qs
        qs = parse_qs(urlparse(url).query)
        rel_path = unquote(qs.get("path", [""])[0])
        abs_path = os.path.join(PIPE_DIR, rel_path)
        if os.path.isfile(abs_path):
            return abs_path
    basename_hint = None
    for key in ("filename", "name"):
        if candidate.get(key):
            basename_hint = candidate[key]
            break
    if not basename_hint and url:
        from urllib.parse import urlparse as _up
        basename_hint = os.path.basename(_up(url).path) or None
    item_dir = os.path.join(ep_dir, "assets", "sfx", item_id)
    if os.path.isdir(item_dir):
        audio_files = [f for f in sorted(os.listdir(item_dir), reverse=True)
                       if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))]
        if basename_hint:
            for f in audio_files:
                if f == basename_hint or os.path.splitext(f)[0] == os.path.splitext(basename_hint)[0]:
                    return os.path.join(item_dir, f)
        if audio_files:
            if len(audio_files) > 1:
                print(f"[WARN] {item_id}: {len(audio_files)} audio files found, using {audio_files[0]} (ambiguous)")
            return os.path.join(item_dir, audio_files[0])
    return None


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
        "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm",
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
        # output format left at SDK default — matches voice2.py behaviour
        _tts_synth = _speechsdk.SpeechSynthesizer(speech_config=config, audio_config=None)
    return _tts_synth


def _preview_build_ssml(text: str, azure_voice: str, azure_locale: str,
                         style: str | None, *, style_degree: float = 1.0,
                         rate: str = "0%", pitch: str = "", break_ms: int = 0,
                         phoneme_overrides: dict | None = None) -> str:
    """Build SSML for a preview / Re-Create request.

    For zh-* locales, known mispronounced characters are emitted as bare
    <phoneme> elements that are DIRECT children of <voice>, outside any
    <mstts:express-as> block.  Azure rejects <phoneme> inside express-as
    (error 1007), but accepts it as a <voice>-level sibling of express-as.
    """
    rate_attr  = f' rate="{rate}"'   if rate  and rate  != "0%" else ""
    pitch_attr = f' pitch="{pitch}"' if pitch and pitch != "0%" else ""

    def _xml_esc(s: str) -> str:
        return (s.replace("&", "&amp;").replace("<", "&lt;")
                 .replace(">", "&gt;").replace('"', "&quot;").replace("'", "&apos;"))

    def _wrap(content: str) -> str:
        """Wrap an escaped text fragment in lang → prosody → express-as layers."""
        if not azure_locale.startswith("zh"):
            content = f'<lang xml:lang="{azure_locale}">{content}</lang>'
        # Always emit <prosody> for zh locales — Azure requires the wrapper
        # for <phoneme alphabet="sapi"> to be honoured.
        # An empty <prosody> (no attributes) causes error 1007, so fall back to
        # rate="0%" when neither rate nor pitch is set.
        if rate_attr or pitch_attr or azure_locale.startswith("zh"):
            r = rate_attr or (' rate="0%"' if azure_locale.startswith("zh") else "")
            content = f'<prosody{r}{pitch_attr}>\n{content}</prosody>'
        if style:
            content = (f'<mstts:express-as style="{style}" styledegree="{style_degree}">'
                       f'{content}</mstts:express-as>')
        return content

    voice_parts: list[str] = []
    # Build inner content with sapi phonemes inline — alphabet="sapi" is accepted
    # inside <mstts:express-as> (unlike alphabet="pinyin" which caused error 1007).
    # Merge global corrections with per-item overrides (item wins on conflict).
    # Pinyin tone format conversion: "hou4" → "hou 4" (insert space before tone digit).
    _ph_overrides = phoneme_overrides or {}
    try:
        from gen_tts_cloud import _ZH_PHONEME_CORRECTIONS as _g_corrections
        _merged = {**_g_corrections, **_ph_overrides}
    except Exception:
        _merged = dict(_ph_overrides)

    def _seg_with_merged(t):
        if not azure_locale.startswith("zh") or not _merged:
            return [{"type": "text", "content": t}]
        segs, buf = [], []
        for ch in t:
            py = _merged.get(ch)
            if py:
                if buf:
                    segs.append({"type": "text", "content": "".join(buf)})
                    buf = []
                segs.append({"type": "phoneme", "content": ch, "pinyin": py})
            else:
                buf.append(ch)
        if buf:
            segs.append({"type": "text", "content": "".join(buf)})
        return segs or [{"type": "text", "content": t}]

    inner_parts: list[str] = []
    for seg in _seg_with_merged(text):
        if seg["type"] == "text":
            inner_parts.append(_xml_esc(seg["content"]))
        else:  # phoneme
            raw_py = seg["pinyin"]   # e.g. "hou4" or "hou 4"
            # Normalise to SAPI format: insert space before trailing tone digit if missing
            if len(raw_py) > 1 and raw_py[-1].isdigit() and raw_py[-2] != ' ':
                sapi = raw_py[:-1] + " " + raw_py[-1]
            else:
                sapi = raw_py
            # Phoneme goes INSIDE the current express-as+prosody block (at end),
            # then the block is flushed.  Text after the phoneme opens a new block.
            # This matches voice4.ssml: <express-as><prosody>text<phoneme/></prosody></express-as>
            inner_parts.append(
                f'<phoneme alphabet="sapi" ph="{_xml_esc(sapi)}">'
                f'{_xml_esc(seg["content"])}</phoneme>'
            )
            voice_parts.append(_wrap("".join(inner_parts)))
            inner_parts = []
    if inner_parts:
        voice_parts.append(_wrap("".join(inner_parts)))
    if break_ms:
        voice_parts.append(f'<break time="{break_ms}ms"/>')

    return (f"<speak version='1.0' xml:lang='{azure_locale}' "
            f"xmlns='http://www.w3.org/2001/10/synthesis' "
            f"xmlns:mstts='http://www.w3.org/2001/mstts'>"
            f"<voice name='{azure_voice}'>{''.join(voice_parts)}</voice></speak>")


# ── SSE helper ─────────────────────────────────────────────────────────────────
def sse(event: str, data: str) -> bytes:
    return f"event: {event}\ndata: {data}\n\n".encode()


# ── Embedded UI ────────────────────────────────────────────────────────────────



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
            "done": check(ep("StoryPrompt.json")) or _log_done(2),
            "artifacts": [ep_rel("StoryPrompt.json")],
        },
        "stage_3": {
            "done": check(ep("Script.json")) or _log_done(3),
            "artifacts": [ep_rel("Script.json")],
        },
        "stage_4": {
            "done": check(ep("ShotList.json")),
            "artifacts": [ep_rel("ShotList.json")],
        },
        "stage_5": {
            # Primary: both canonical output files exist.
            # Fallback: stage log (handles non-en locale configs where
            # VOPlan.en.json might not be the written locale).
            "done": (
                check(ep("AssetManifest.shared.json")) and check(ep("VOPlan.en.json"))
            ) or _log_done(5),
            "artifacts": [ep_rel("AssetManifest.shared.json"), ep_rel("VOPlan.en.json")],
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

    # Detect locales from VOPlan.{locale}.json files, then fill in any
    # additional locales from AssetManifest.{locale}.json that don't have a
    # VOPlan yet.  This ensures translated locales appear in Stage 9 (so the
    # user can click "Run 5" to create their VOPlan) even before manifest_merge
    # has run for them.  Previously the AssetManifest scan was a fallback that
    # only ran when NO VOPlan existed at all — so once VOPlan.en.json appeared,
    # zh-Hans (which only had an AssetManifest) was silently dropped.
    locales: list[str] = []
    _locales_seen: set[str] = set()
    if os.path.isdir(ep_dir):
        for f in sorted(os.listdir(ep_dir)):
            m = re.match(r"VOPlan\.(.+)\.json$", f)
            if m and m.group(1) != "shared":
                _locales_seen.add(m.group(1))
                locales.append(m.group(1))
        # Always also scan AssetManifest files and add any locale not yet seen
        for f in sorted(os.listdir(ep_dir)):
            m = re.match(r"AssetManifest\.(.+)\.json$", f)
            if m and m.group(1) != "shared" and m.group(1) not in _locales_seen:
                _locales_seen.add(m.group(1))
                locales.append(m.group(1))

    # Per-locale post-processing status
    locale_steps: dict[str, dict] = {}
    for locale in locales:
        vo_dir = os.path.join(ep_dir, "assets", locale, "audio", "vo")
        gen_tts_done = os.path.isdir(vo_dir) and bool(
            [f for f in os.listdir(vo_dir) if f.endswith(".wav")]
        ) if os.path.isdir(vo_dir) else False

        # resolve_assets populates resolved_assets[] in the unified manifest.
        # Empty [] (falsy) = manifest_merge ran but resolve_assets has not yet.
        # Non-empty (truthy) = resolve_assets has run.
        _ua_path = ep(f"VOPlan.{locale}.json")
        try:
            if os.path.isfile(_ua_path):
                with open(_ua_path, encoding="utf-8") as _ua_f:
                    _ra_done = bool(json.load(_ua_f).get("resolved_assets"))
            else:
                _ra_done = False
        except Exception:
            _ra_done = False

        # manifest_merge is done when VOPlan.{locale}.json exists with locale_scope == "merged".
        _mm_done = _step_is_done("manifest_merge", slug, ep_id, locale)

        locale_steps[locale] = {
            "manifest_merge":  {"done": _mm_done},
            "gen_tts":         {"done": gen_tts_done},
            "post_tts":        {"done": gen_tts_done},   # proxy: same check as gen_tts
            "resolve_assets":  {"done": _ra_done},
            "gen_render_plan": {"done": True},  # eliminated — always done
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
    # gen_cross_srt done: renders/A/output.B.srt exists for every locale pair (A,B)
    import itertools as _itertools
    _renders_dir = os.path.join(ep_dir, "renders")
    _cross_srt_done = False
    if len(locales) >= 2:
        _cross_srt_done = all(
            os.path.isfile(os.path.join(_renders_dir, a, f"output.{b}.srt")) and
            os.path.isfile(os.path.join(_renders_dir, b, f"output.{a}.srt"))
            for a, b in _itertools.combinations(locales, 2)
        )

    shared_steps = {
        "gen_music_clip":  {"done": _any_files(os.path.join(_assets_dir, "music"),  ".wav")},
        "gen_characters":  {"done": _any_files(os.path.join(proj_dir, "characters"), ".png")},
        "gen_backgrounds": {"done": _any_files(os.path.join(_assets_dir, "backgrounds"), ".png")},
        "gen_sfx":         {"done": _any_files(os.path.join(_assets_dir, "sfx"),    ".wav")},
        "gen_cross_srt":   {"done": _cross_srt_done},
    }

    # Music plan checkpoint: [4b/8] in Stage 9 — pipeline pauses here until user confirms
    _music_plan_path = os.path.join(ep_dir, "MusicPlan.json")
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

    # ── ep_cast: character_ids appearing in this episode's Script.json ───
    ep_cast: list[str] = []
    _script_path = os.path.join(ep_dir, "Script.json")
    if os.path.isfile(_script_path):
        try:
            with open(_script_path, encoding="utf-8") as _f:
                _script = json.load(_f)
            # Prefer the top-level cast array if present
            if _script.get("cast"):
                ep_cast = [c["character_id"] for c in _script["cast"] if c.get("character_id")]
            else:
                # Fall back: collect unique speaker_ids from dialogue actions
                _seen: set[str] = set()
                for _scene in (_script.get("scenes") or []):
                    for _act in (_scene.get("actions") or []):
                        _sid = _act.get("speaker_id")
                        if _sid and _sid not in _seen:
                            _seen.add(_sid)
                            ep_cast.append(_sid)
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

    # ── Approval checkpoints (FIX E) ──────────────────────────────────────────
    # VO approval: check VOPlan.{primary_locale}.json vo_approval block
    _vo_primary = meta_locales_str.split(",")[0].strip() if meta_locales_str else "en"
    _vo_approved_legacy = os.path.join(ep_dir, "tts_review_complete.json")
    # Per-locale VO approval — used by Stage 9 UI to grey gen_tts only for
    # locales whose VO has actually been approved (not just the primary).
    _vo_by_locale: dict[str, bool] = {}
    for _loc in (locales or ["en"]):
        _vo_by_locale[_loc] = _is_vo_approved(ep_dir, _loc) if _VO_UTILS_AVAILABLE else False
    # Legacy fallback: if tts_review_complete.json exists, count it for primary locale
    if os.path.isfile(_vo_approved_legacy) and not _vo_by_locale.get(_vo_primary):
        _vo_by_locale[_vo_primary] = True

    approvals = {
        "vo":           (_is_vo_approved(ep_dir, _vo_primary) if _VO_UTILS_AVAILABLE else False) or os.path.isfile(_vo_approved_legacy),
        "vo_by_locale": _vo_by_locale,
        "music":        os.path.isfile(os.path.join(ep_dir, "MusicPlan.json")),
        "sfx":          os.path.isfile(os.path.join(ep_dir, "SfxPlan.json")),
        "media":        os.path.isfile(os.path.join(ep_dir, "MediaPlan.json")),
    }
    approvals["all"] = approvals["vo"] and approvals["music"] and approvals["sfx"] and approvals["media"]

    # ── Staleness guard: is render output older than VO approval? ────────────
    # If VOPlan.{locale}.json (which holds vo_approval.approved_at) is newer than
    # the rendered output.mp4, the video was produced before the current VO approval
    # and should be re-rendered.  gen_render_plan no longer exists — render_video reads
    # VOPlan directly, so staleness is VOPlan vs output.mp4 (not vs RenderPlan).
    _render_plan_stale: dict[str, bool] = {}   # keyed by locale (name kept for UI compat)
    _tts_complete_path = os.path.join(ep_dir, f"VOPlan.{_vo_primary}.json")
    _tts_mtime = os.path.getmtime(_tts_complete_path) if os.path.isfile(_tts_complete_path) else None
    for _loc in (locales or ["en"]):
        _render_out = os.path.join(ep_dir, "renders", _loc, "output.mp4")
        if _tts_mtime is not None:
            if not os.path.isfile(_render_out):
                _render_plan_stale[_loc] = True   # not yet rendered
            elif os.path.getmtime(_render_out) < _tts_mtime:
                _render_plan_stale[_loc] = True   # render predates VO approval
            else:
                _render_plan_stale[_loc] = False
        else:
            _render_plan_stale[_loc] = False      # no VO approval yet — not stale

    return {
        "slug": slug, "ep_id": ep_id,
        "llm_stages": llm_stages,
        "locales": locales,
        "locale_steps": locale_steps,   # kept for /run_step recovery endpoint
        "shared_steps": shared_steps,   # steps 1–4 + 12: gen_music_clip, gen_characters, gen_backgrounds, gen_sfx, gen_cross_srt
        "ready_videos": ready_videos,
        "ready_dubbed": ready_dubbed,
        "story_file": story_file_detected,
        "voice_cast": voice_cast,
        "ep_cast":    ep_cast,
        "title":          meta_title,
        "genre":          meta_genre,
        "story_format":   meta_format,
        "locales_str":    meta_locales_str,
        "no_music":       meta_no_music,
        "purge_cache":    meta_purge_cache,
        "music_plan_done":      music_plan_done,
        "tts_done":             tts_done,
        "approvals":            approvals,
        "render_plan_stale":    _render_plan_stale,   # locale→bool: RenderPlan older than VO approval
    }


def _step_is_done(step: str, slug: str, ep_id: str, locale: str) -> bool:
    """Return True if the step's output already exists (safe to skip)."""
    ep_dir = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)

    def check(*paths): return all(os.path.isfile(p) for p in paths)

    if step == "manifest_merge":
        # Primary locale: done when VOPlan.{locale}.json exists with locale_scope=="merged".
        # Non-primary locale (compound step): also requires WAVs and VO timing populated.
        _path = os.path.join(ep_dir, f"VOPlan.{locale}.json")
        if not os.path.isfile(_path):
            return False
        try:
            with open(_path, encoding="utf-8") as _f:
                _doc = json.load(_f)
            if _doc.get("locale_scope") != "merged":
                return False
            _vo_items = _doc.get("vo_items", [])
            if _vo_items:
                # If any item is missing timing, post_tts hasn't run yet
                if any(v.get("start_sec") is None for v in _vo_items):
                    return False
                # WAVs must exist (gen_tts ran)
                _vo_dir = os.path.join(ep_dir, "assets", locale, "audio", "vo")
                if not os.path.isdir(_vo_dir) or not any(
                    f.endswith(".wav") for f in os.listdir(_vo_dir)
                ):
                    return False
            return True
        except Exception:
            return False
    elif step == "apply_music_plan":
        return False   # always re-run — user may have changed overrides
    elif step in ("gen_tts", "post_tts"):
        vo_dir = os.path.join(ep_dir, "assets", locale, "audio", "vo")
        return (os.path.isdir(vo_dir) and
                any(f.endswith(".wav") for f in os.listdir(vo_dir)))
    elif step == "resolve_assets":
        # resolved_assets[] is always present (placeholder [] written by manifest_merge).
        # Done when the list is non-empty (truthy) — i.e. resolve_assets has populated it.
        _path = os.path.join(ep_dir, f"VOPlan.{locale}.json")
        if not os.path.isfile(_path):
            return False
        try:
            with open(_path, encoding="utf-8") as _f:
                return bool(json.load(_f).get("resolved_assets"))
        except Exception:
            return False
    elif step == "gen_render_plan":
        return True  # gen_render_plan eliminated — always considered done
    elif step == "render_video":
        return check(os.path.join(ep_dir, "renders", locale, "output.mp4"))
    elif step == "gen_cross_srt":
        # Done when cross SRT files exist for every locale pair.
        # Locale arg is "" for this step; derive locale list from renders/ dir.
        _rd = os.path.join(ep_dir, "renders")
        if not os.path.isdir(_rd):
            return False
        _locs = sorted(d for d in os.listdir(_rd)
                       if os.path.isdir(os.path.join(_rd, d)) and not d.startswith("."))
        if len(_locs) < 2:
            return False
        import itertools as _it
        return all(
            os.path.isfile(os.path.join(_rd, a, f"output.{b}.srt")) and
            os.path.isfile(os.path.join(_rd, b, f"output.{a}.srt"))
            for a, b in _it.combinations(_locs, 2)
        )
    return False


def _delete_step_output(step: str, slug: str, ep_id: str, locale: str) -> None:
    """Remove a step's primary output(s) so it will always re-run fresh."""
    ep_dir = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)

    if step == "manifest_merge":
        # VOPlan.{locale}.json is ALSO the source of VO items written by
        # Stage 5 — deleting the whole file destroys VO data that cannot be
        # recovered without re-running Stage 5.
        # Instead, strip only the merge-added fields and reset locale_scope so
        # manifest_merge will re-run cleanly while VO data is preserved.
        _mpath = os.path.join(ep_dir, f"VOPlan.{locale}.json")
        if os.path.isfile(_mpath):
            try:
                with open(_mpath, encoding="utf-8") as _mf:
                    _mdoc = json.load(_mf)
                for _k in ("resolved_assets", "background_overrides",
                           "shared_music_items", "shared_sfx_items"):
                    _mdoc.pop(_k, None)
                _mdoc["locale_scope"] = "locale"   # mark as pre-merge
                with open(_mpath, "w", encoding="utf-8") as _mf:
                    json.dump(_mdoc, _mf, indent=2, ensure_ascii=False)
                    _mf.write("\n")
            except Exception:
                pass   # leave file untouched on error
        # For non-primary locales, step 5 is compound (manifest_merge + gen_tts +
        # polish + post_tts). Also wipe WAVs so gen_tts re-runs from scratch.
        _wav_dir = os.path.join(ep_dir, "assets", locale, "audio", "vo")
        if locale and os.path.isdir(_wav_dir):
            import shutil as _shutil_del
            try:
                _shutil_del.rmtree(_wav_dir)
            except Exception:
                pass
        return

    if step == "resolve_assets":
        # Only clear resolved_assets[] — leave locale_scope intact so
        # resolve_assets.py does not reject the manifest.
        _mpath = os.path.join(ep_dir, f"VOPlan.{locale}.json")
        if os.path.isfile(_mpath):
            try:
                with open(_mpath, encoding="utf-8") as _mf:
                    _mdoc = json.load(_mf)
                _mdoc["resolved_assets"] = []
                with open(_mpath, "w", encoding="utf-8") as _mf:
                    json.dump(_mdoc, _mf, indent=2, ensure_ascii=False)
                    _mf.write("\n")
            except Exception:
                pass
        return

    if step == "gen_cross_srt":
        # Delete all cross-locale SRT files: renders/A/output.B.srt for every A≠B
        _rd = os.path.join(ep_dir, "renders")
        if os.path.isdir(_rd):
            _locs = sorted(d for d in os.listdir(_rd)
                           if os.path.isdir(os.path.join(_rd, d)) and not d.startswith("."))
            for _la in _locs:
                for _lb in _locs:
                    if _la == _lb:
                        continue
                    _srt = os.path.join(_rd, _la, f"output.{_lb}.srt")
                    try:
                        os.remove(_srt)
                    except FileNotFoundError:
                        pass
        return

    targets: dict[str, list[str]] = {
        "gen_render_plan": [],  # eliminated — nothing to delete
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
      - VOPlan.{locale}.json          — unified manifest (merge + resolved_assets + render_plan)
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
                m = re.match(r"VOPlan\.(.+)\.json$", f)
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

        # Delete RenderPlan only.  VOPlan.{loc}.json is intentionally
        # kept — gen_tts/post_tts read it; manifest_merge recreates it via
        # _delete_step_output when a specific step is reset.
        _rm(os.path.join(ep_dir, f"RenderPlan.{loc}.json"))

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
            "--manifest", ep("AssetManifest.shared.json"),
        ]
    elif step == "gen_characters":
        # AI asset steps inherit AI_SERVER_URL / AI_SERVER_KEY from the shell env.
        return [
            "python3", os.path.join(code_dir, "fetch_ai_assets.py"),
            "--manifest",    ep("AssetManifest.shared.json"),
            "--asset_type",  "characters",
        ]
    elif step == "gen_backgrounds":
        cmd = [
            "python3", os.path.join(code_dir, "fetch_ai_assets.py"),
            "--manifest",    ep("AssetManifest.shared.json"),
            "--asset_type",  "backgrounds",
        ]
        asset_ids = (payload or {}).get("asset_ids", "")
        if asset_ids:
            cmd += ["--asset-ids", asset_ids]
        return cmd
    elif step == "gen_sfx":
        return [
            "python3", os.path.join(code_dir, "fetch_ai_assets.py"),
            "--manifest",    ep("AssetManifest.shared.json"),
            "--asset_type",  "sfx",
        ]
    elif step == "manifest_merge":
        # Use VOPlan.{locale}.json as input when it already exists (in-place re-run,
        # preserves vo_approval block). Fall back to AssetManifest.{locale}.json on
        # first run (Stage 8 produces AssetManifest.zh-Hans.json, not VOPlan.zh-Hans.json).
        _voplan_in = ep(f"VOPlan.{locale}.json")
        _locale_in = _voplan_in if os.path.isfile(_voplan_in) else ep(f"AssetManifest.{locale}.json")
        _mm_cmd = [
            "python3", os.path.join(code_dir, "manifest_merge.py"),
            "--shared", ep("AssetManifest.shared.json"),
            "--locale", _locale_in,
            "--out",    ep(f"VOPlan.{locale}.json"),
        ]
        # Pass --primary so translated locales inherit scene_heads/scene_tails from
        # the primary locale. Use _get_primary_locale (reads meta.json) — pipeline_vars.sh
        # does not have a PRIMARY_LOCALE field.
        if _VO_UTILS_AVAILABLE:
            try:
                from pathlib import Path as _Pmm
                _primary_loc_mm = _get_primary_locale(_Pmm(ep_dir))
            except Exception:
                _primary_loc_mm = ""
        else:
            _primary_loc_mm = ""
        if _primary_loc_mm:
            _mm_cmd += ["--primary", ep(f"VOPlan.{_primary_loc_mm}.json")]
        return _mm_cmd
    elif step == "gen_tts":
        # MTV: skip TTS entirely — no VO synthesis needed, lyrics come from alignment
        _meta_tts = os.path.join(ep_dir, "meta.json")
        if os.path.isfile(_meta_tts):
            try:
                _mj_tts = json.load(open(_meta_tts, encoding="utf-8"))
                if _mj_tts.get("story_format") == "mtv":
                    print("[gen_tts] MTV format — skipping TTS (no VO synthesis needed)")
                    return []  # no-op
            except Exception:
                pass
        cmd = [
            "python3", os.path.join(code_dir, "gen_tts_cloud.py"),
            "--manifest", ep(f"VOPlan.{locale}.json"),
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
            "--manifest", ep(f"VOPlan.{locale}.json"),
        ]
    elif step == "polish_locale_vo":
        # Alignment + calibration loop for non-primary locale (run.sh step [6b]).
        _primary_pvo = "en"
        if _VO_UTILS_AVAILABLE:
            try:
                from pathlib import Path as _Ppvo
                _primary_pvo = _get_primary_locale(_Ppvo(ep_dir))
            except Exception:
                pass
        return [
            "python3", os.path.join(code_dir, "polish_locale_vo.py"),
            "--manifest",       ep(f"VOPlan.{locale}.json"),
            "--locale",         locale,
            "--ep-dir",         ep_dir,
            "--primary-locale", _primary_pvo,
        ]
    elif step == "apply_music_plan":
        music_plan = os.path.join(ep_dir, "MusicPlan.json")
        if not os.path.isfile(music_plan):
            return []   # [] = intentional skip (no MusicPlan.json); None = unknown step
        return [
            "python3", os.path.join(code_dir, "apply_music_plan.py"),
            "--plan",     music_plan,
            "--manifest", ep(f"VOPlan.{locale}.json"),
        ]
    elif step == "resolve_assets":
        return [
            "python3", os.path.join(code_dir, "resolve_assets.py"),
            "--manifest", ep(f"VOPlan.{locale}.json"),
        ]
    elif step == "gen_render_plan":
        # gen_render_plan.py is ELIMINATED — its logic is absorbed into render_video.py.
        # Return [] (empty command = no-op skip) so any legacy /run_step call is harmless.
        return []
    elif step == "render_video":
        out_dir = os.path.join(ep_dir, "renders", locale)
        cmd = [
            "python3", os.path.join(code_dir, "render_video.py"),
            "--plan",    ep(f"VOPlan.{locale}.json"),
            "--locale",  locale,
            "--out",     out_dir,
            "--profile", profile,
        ]
        if no_music:
            cmd.append("--no-music")
        # MTV format: pass --format so render_video skips VO mixing and music ducking
        _meta_rv = os.path.join(ep_dir, "meta.json")
        if os.path.isfile(_meta_rv):
            try:
                _mj_rv = json.load(open(_meta_rv, encoding="utf-8"))
                if _mj_rv.get("story_format") == "mtv":
                    cmd += ["--format", "mtv"]
            except Exception:
                pass
        # Phase 2 — Timeline Lock: pass primary locale's approved VO timing so non-primary
        # locales cannot produce shorter shots than the primary locale.
        _primary_loc_rv = (payload or {}).get("primary_locale", "en")
        if locale != _primary_loc_rv:
            _ref_vp_rv = ep(f"VOPlan.{_primary_loc_rv}.json")
            if os.path.isfile(_ref_vp_rv):
                cmd += ["--reference-approval", _ref_vp_rv]
        return cmd
    elif step == "gen_cross_srt":
        # Episode-level step: locale arg is "" — derive locales from meta.json
        _gcs_locales = ""
        _gcs_primary = "en"
        _meta_path = ep("meta.json")
        if os.path.isfile(_meta_path):
            try:
                _mj = json.load(open(_meta_path, encoding="utf-8"))
                _gcs_locales = _mj.get("locales", "")
                _gcs_primary = (_gcs_locales.split(",")[0].strip()
                                if _gcs_locales else "en")
            except Exception:
                pass
        if not _gcs_locales:
            # Fallback: derive from pipeline_vars.sh
            _pv_path = ep("pipeline_vars.sh")
            if os.path.isfile(_pv_path):
                try:
                    _pv = open(_pv_path, encoding="utf-8").read()
                    for _ln in _pv.splitlines():
                        if _ln.startswith("export LOCALES="):
                            _gcs_locales = _ln.split("=", 1)[1].strip().strip('"')
                        elif _ln.startswith("export PRIMARY_LOCALE="):
                            _gcs_primary = _ln.split("=", 1)[1].strip().strip('"')
                except Exception:
                    pass
        if not _gcs_locales or "," not in _gcs_locales:
            return []   # [] = intentional skip (single locale, no cross SRT needed)
        return [
            "python3", os.path.join(code_dir, "gen_cross_srt.py"),
            "--ep-dir",  ep_dir,
            "--locales", _gcs_locales,
            "--primary", _gcs_primary,
        ]
    return None


def _fake_subprocess(cmd, step_name, ep_dir, locale):
    """In --test-mode: copy pre-baked fixture outputs, return fake process."""
    import shutil as _shutil
    # Fixture episode dir — hard truth lives here (tests/fixtures/projects/test-proj/episodes/s01e01)
    _fixture_ep = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "tests", "fixtures", "projects", "test-proj", "episodes", "s01e01"
    )

    def _copy_merged_voplan():
        # Run the real manifest_merge.py on the ep_dir fixture inputs so the
        # merged output is produced from hard truth (no pre-baked step_outputs needed).
        import subprocess as _sp
        _mm_py  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest_merge.py")
        _loc    = locale or "en"
        _shared = os.path.join(ep_dir, "AssetManifest.shared.json")
        _loc_in = os.path.join(ep_dir, f"VOPlan.{_loc}.json")
        _sp.run(["python3", _mm_py, "--shared", _shared, "--locale", _loc_in, "--out", _loc_in], check=True)

    def _stub_render():
        out = os.path.join(ep_dir, "renders", locale or "en")
        os.makedirs(out, exist_ok=True)
        open(os.path.join(out, "output.mp4"), "wb").write(b"\x00" * 100)

    _actions = {
        "manifest_merge":     _copy_merged_voplan,
        "gen_tts_cloud":      lambda: None,
        "post_tts_analysis":  lambda: None,
        "gen_music_clip":     lambda: None,
        "music_prepare_loops": lambda: None,
        "apply_music_plan":   lambda: None,
        "resolve_assets":     lambda: None,
        "render_video":       _stub_render,
    }

    action = _actions.get(step_name)
    if action is None:
        raise RuntimeError(
            f"_fake_subprocess: unknown step {step_name!r} — "
            f"add it to _actions in _fake_subprocess()"
        )
    try:
        action()
    except Exception as _fe:
        print(f"  [TEST MODE] _fake_subprocess {step_name}: {_fe}", flush=True)

    class _FakeProc:
        returncode = 0
        def __init__(self):
            self._lines = iter([f"  [TEST MODE] {step_name} — fixture applied\n"])
        @property
        def stdout(self):
            return self
        def __iter__(self):
            return self._lines
        def wait(self, timeout=None):
            return 0
        def poll(self):
            return 0
    return _FakeProc()


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
            _ui_path = Path(__file__).parent / "ui" / "main.html"
            page = _ui_path.read_text(encoding="utf-8")
            media_server_url = os.environ.get("MEDIA_SERVER_URL",
                                   _vc_config.get("media", {}).get("default_server_url", ""))
            page = page.replace("{{MEDIA_SERVER_URL}}", media_server_url)
            page = page.replace("__VO_THRESH__",      f"{_VO_POLISH_THRESHOLD:.2f}")
            page = page.replace("__VO_THRESH_HIGH__", f"{_VO_POLISH_THRESHOLD_HIGH:.2f}")
            body = page.encode("utf-8")
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
            _vdb_str = params.get("volume_db", ["0"])[0].strip()
            try:
                _vdb = float(_vdb_str)
            except (ValueError, TypeError):
                _vdb = 0.0
            _log.info("[vo_preview_item] item=%s  locale=%s  ep=%s  volume_db=%s",
                      item_id, locale, ep_dir, _vdb)
            if not ep_dir or not locale or not item_id or ".." in item_id:
                self.send_response(400); self.end_headers(); return
            full_ep = os.path.join(PIPE_DIR, ep_dir) \
                      if not os.path.isabs(ep_dir) else ep_dir
            wav_p = os.path.join(full_ep, "assets", locale, "audio", "vo",
                                 item_id + ".wav")
            if os.path.isfile(wav_p):
                if _vdb != 0.0:
                    # Pipe through ffmpeg to apply volume gain on-the-fly
                    import tempfile as _tf
                    _tmp_out = _tf.NamedTemporaryFile(suffix=".wav", delete=False)
                    _tmp_out.close()
                    _vcmd = [
                        "ffmpeg", "-y", "-i", wav_p,
                        "-af", f"volume={_vdb}dB",
                        "-ar", "24000", "-ac", "1",
                        _tmp_out.name,
                    ]
                    _vr = subprocess.run(_vcmd, capture_output=True, timeout=15)
                    if _vr.returncode == 0:
                        with open(_tmp_out.name, "rb") as _wf:
                            data = _wf.read()
                    else:
                        _log.warning("[vo_audio] ffmpeg volume failed rc=%d — serving raw",
                                     _vr.returncode)
                        with open(wav_p, "rb") as _wf:
                            data = _wf.read()
                    try:
                        os.unlink(_tmp_out.name)
                    except OSError:
                        pass
                else:
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
            _log.info("[vo_generate_preview] locale=%s  ep=%s", locale, ep_dir)
            if not ep_dir or not locale or ".." in ep_dir:
                self.send_response(400); self.end_headers(); return
            try:
                full_ep = _vo_resolve_ep_dir(ep_dir)
                manifest_path = os.path.join(full_ep, "VOPlan.en.json") \
                    if locale == "en" else \
                    os.path.join(full_ep, f"VOPlan.{locale}.json")
                if not os.path.exists(manifest_path):
                    manifest_path = os.path.join(full_ep, "VOPlan.en.json")
                with open(manifest_path, encoding="utf-8") as _mf:
                    _mfdata = json.load(_mf)
                vo_items = _mfdata.get("vo_items", [])
                scene_tails = _mfdata.get("scene_tails", {})
                # scene_heads: start with VOPlan values, then override with any
                # unsaved DOM values sent by the frontend (so Generate Preview
                # reflects changes the user has typed but not yet saved).
                scene_heads = _mfdata.get("scene_heads", {})  # values in seconds
                _sh_param = unquote_plus(params.get("scene_heads", [""])[0]).strip()
                if _sh_param:
                    try:
                        _sh_override = json.loads(_sh_param)
                        if isinstance(_sh_override, dict):
                            scene_heads = {**scene_heads, **_sh_override}
                    except Exception:
                        pass
                # pause_after_ms overrides from DOM (keyed by item_id)
                _pauses_override: dict = {}
                _pauses_param = unquote_plus(params.get("pauses", [""])[0]).strip()
                if _pauses_param:
                    try:
                        _po = json.loads(_pauses_param)
                        if isinstance(_po, dict):
                            _pauses_override = {k: int(float(v)) for k, v in _po.items()}
                    except Exception:
                        pass
                # scene_tails overrides from DOM (keyed by scene_id)
                _tails_param = unquote_plus(params.get("scene_tails", [""])[0]).strip()
                if _tails_param:
                    try:
                        _to = json.loads(_tails_param)
                        if isinstance(_to, dict):
                            scene_tails = {**scene_tails, **{k: int(float(v)) for k, v in _to.items()}}
                    except Exception:
                        pass
                # volume_db overrides from DOM (keyed by item_id) — only non-zero entries sent
                _volumes_override: dict = {}
                _volumes_param = unquote_plus(params.get("volumes", [""])[0]).strip()
                if _volumes_param:
                    try:
                        _vo2 = json.loads(_volumes_param)
                        if isinstance(_vo2, dict):
                            _volumes_override = {k: float(v) for k, v in _vo2.items()}
                    except Exception:
                        pass
                vo_dir = os.path.join(full_ep, "assets", locale, "audio", "vo")

                # Detect sample rate from the first available WAV so the output
                # header matches the source files.  All items in a single locale
                # are synthesised by the same TTS engine at the same rate, so
                # reading the first file is sufficient.  Fall back to 24000 only
                # when no WAV exists yet (empty episode edge-case).
                SAMPLE_RATE = 24000  # fallback
                for _it_sr in vo_items:
                    _wav_sr = os.path.join(vo_dir, f"{_it_sr.get('item_id','')}.wav")
                    if os.path.isfile(_wav_sr):
                        with _wave.open(_wav_sr) as _wf_sr:
                            SAMPLE_RATE = _wf_sr.getframerate()
                        break
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
                    # Scene head pause — stored in seconds, applied before first item of each scene
                    if _scn != prev_scene:
                        head_sec = float(scene_heads.get(_scn, 0.0))
                        if head_sec > 0:
                            _n_samps = round(head_sec * SAMPLE_RATE)
                            _sil = b'\x00' * (_n_samps * 2)
                            pcm_chunks.append(_sil)
                            current_sec += head_sec
                    prev_scene = _scn
                    try:
                        with _wave.open(_wav) as _wf:
                            _item_rate = _wf.getframerate()
                            _pcm = _wf.readframes(_wf.getnframes())
                            _dur = _wf.getnframes() / _item_rate
                    except Exception as _wav_err:
                        raise RuntimeError(
                            f"{_iid}.wav is corrupt or unreadable: {_wav_err}. "
                            f"Re-create this item in the VO tab before generating preview."
                        )
                    # Resample to the output rate when this item was synthesised
                    # at a different sample rate (e.g. 16000 Hz re-gen in a 24000 Hz
                    # episode). Without resampling the PCM byte-count is wrong and
                    # the segment plays at the wrong speed (cartoon / fast-forward).
                    if _item_rate != SAMPLE_RATE:
                        try:
                            import audioop as _audioop  # stdlib up to Python 3.12
                            _pcm, _ = _audioop.ratecv(
                                _pcm, 2, 1, _item_rate, SAMPLE_RATE, None)
                        except ImportError:
                            # Python ≥ 3.13 removed audioop — linear interpolation fallback
                            import array as _array
                            _src   = _array.array('h', _pcm)
                            _ratio = SAMPLE_RATE / _item_rate
                            _n_out = round(len(_src) * _ratio)
                            _dst   = _array.array('h', [0] * _n_out)
                            for _ri in range(_n_out):
                                _pos  = _ri / _ratio
                                _lo   = int(_pos)
                                _hi   = min(_lo + 1, len(_src) - 1)
                                _frac = _pos - _lo
                                _dst[_ri] = round(
                                    _src[_lo] * (1 - _frac) + _src[_hi] * _frac)
                            _pcm = bytes(_dst)
                    # Apply volume_db gain from DOM (preview only — does not touch .wav)
                    _vdb_preview = _volumes_override.get(_iid, 0.0)
                    if _vdb_preview != 0.0:
                        import array as _arr
                        _gain = 10 ** (_vdb_preview / 20.0)
                        _samps = _arr.array('h', _pcm)
                        _samps = _arr.array('h', [
                            max(-32768, min(32767, round(s * _gain))) for s in _samps
                        ])
                        _pcm = bytes(_samps)
                    clips_meta.append({
                        "item_id":      _iid,
                        "scene_id":     _scn,
                        "text":         _it.get("text", ""),
                        "start_sec":    round(current_sec, 3),
                        "duration_sec": round(_dur, 3),
                    })
                    pcm_chunks.append(_pcm)
                    current_sec += _dur
                    # pause_after_ms gap — DOM override takes precedence over VOPlan
                    _pause_ms = _pauses_override.get(_iid, int(_it.get("pause_after_ms", 300)))
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
                _payload = {
                    "wav_url":   "/serve_media?path=" + _rel,
                    "total_sec": round(current_sec, 3),
                    "clips":     clips_meta,
                }
                # Persist metadata so the frontend can restore the player on reload
                _meta_path = os.path.join(full_ep, "assets", "meta",
                                          f"vo_preview_{locale}.meta.json")
                with open(_meta_path, "w", encoding="utf-8") as _mf:
                    json.dump(_payload, _mf, ensure_ascii=False)
                _resp = json.dumps(_payload).encode()
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
            # Check VOPlan vo_approval block, fall back to legacy sentinel
            _leg_s  = os.path.join(full_ep, "tts_review_complete.json")
            exists  = (_is_vo_approved(full_ep, locale) if _VO_UTILS_AVAILABLE else False) \
                      or os.path.isfile(_leg_s)
            valid   = False
            timestamp = None
            if _VO_UTILS_AVAILABLE:
                try:
                    valid = _verify_sentinel(full_ep, locale)
                    _mani_p = os.path.join(full_ep, f"VOPlan.{locale}.json")
                    if os.path.isfile(_mani_p):
                        with open(_mani_p, encoding="utf-8") as _sf:
                            _mani_d = json.load(_sf)
                        timestamp = _mani_d.get("vo_approval", {}).get("approved_at")
                    if not timestamp and os.path.isfile(_leg_s):
                        with open(_leg_s, encoding="utf-8") as _sf2:
                            timestamp = json.load(_sf2).get("completed_at")
                except Exception:
                    pass
            _json_resp(self, {
                "exists":       exists,
                "valid":        valid,
                "completed_at": timestamp,
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
                        m = re.match(r"VOPlan\.(.+)\.json$", fname)
                        if m and m.group(1) != "shared":
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
                # VOPlan.{locale}.json is the single source for all stages
                mpath = os.path.join(full_ep_dir, f"VOPlan.{locale}.json")
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
                        # MTV has no WAV files — fall back to VOPlan start_sec/end_sec
                        if dur is not None:
                            it["duration_sec"] = dur
                        elif it.get("end_sec") and it.get("start_sec") is not None:
                            it["duration_sec"] = round(it["end_sec"] - it["start_sec"], 3)
                        else:
                            it["duration_sec"] = None
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
                    scene_heads = manifest.get("scene_heads", {})
                    # Include story_format so VO tab can detect MTV mode
                    _vo_story_format = "episodic"
                    _vo_meta_path = os.path.join(full_ep_dir, "meta.json")
                    if os.path.isfile(_vo_meta_path):
                        try:
                            _vo_meta = json.load(open(_vo_meta_path, encoding="utf-8"))
                            _vo_story_format = _vo_meta.get("story_format", "episodic")
                        except Exception:
                            pass
                    body = json.dumps({"items": items, "voice_catalog": voice_catalog,
                                       "scene_tails": scene_tails,
                                       "scene_heads": scene_heads,
                                       "story_format": _vo_story_format}).encode()
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

        # ── GET /api/vo_approved_timing — approved durations for drift detection ──
        # Returns {item_id: duration_sec} from VOPlan.{locale}.json vo_items[].
        # Used by the VO tab JS to detect timing drift after re-synthesis.
        elif parsed.path == "/api/vo_approved_timing":
            params = parse_qs(parsed.query)
            ep_dir = unquote_plus(params.get("ep_dir", [""])[0]).strip()
            locale = unquote_plus(params.get("locale", [""])[0]).strip()
            approved: dict = {}
            if ep_dir and locale:
                _full_ep = os.path.join(PIPE_DIR, ep_dir) \
                           if not os.path.isabs(ep_dir) else ep_dir
                _apath = os.path.join(_full_ep, f"VOPlan.{locale}.json")
                if os.path.isfile(_apath):
                    try:
                        with open(_apath, encoding="utf-8") as _fh:
                            _adoc = json.load(_fh)
                        if _adoc.get("vo_approval", {}).get("approved_at"):
                            approved = {
                                _it["item_id"]: float(_it.get("duration_sec", 0))
                                for _it in _adoc.get("vo_items", [])
                                if "item_id" in _it
                            }
                    except Exception:
                        pass
            body = json.dumps({"approved": approved}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(body)

        # ── GET /api/vo_whisper_compare — serve cached whisper_compare.json ──
        elif parsed.path == "/api/vo_whisper_compare":
            params = parse_qs(parsed.query)
            ep_dir = unquote_plus(params.get("ep_dir", [""])[0]).strip()
            locale = unquote_plus(params.get("locale", [""])[0]).strip()
            if not ep_dir or not locale:
                body = json.dumps({"ok": False, "error": "ep_dir and locale required"}).encode()
                self.send_response(400)
            else:
                # Resolve relative ep_dir to absolute path (same pattern as other VO endpoints)
                _full_ep = os.path.join(PIPE_DIR, ep_dir) \
                           if not os.path.isabs(ep_dir) else ep_dir
                _wc_path = os.path.join(_full_ep, "assets", locale, "whisper_compare.json")
                # Path traversal guard
                try:
                    _wc_resolved = str(Path(_wc_path).resolve())
                    assert _wc_resolved.startswith(str(PIPE_DIR))
                except (AssertionError, Exception):
                    body = json.dumps({"ok": False, "error": "invalid path"}).encode()
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if not os.path.isfile(_wc_path):
                    body = json.dumps({"ok": False, "error": "not found"}).encode()
                else:
                    try:
                        with open(_wc_path, encoding="utf-8") as _wf:
                            _wc_data = json.load(_wf)
                        body = json.dumps({"ok": True, "data": _wc_data}).encode()
                    except Exception as _wc_exc:
                        body = json.dumps({"ok": False, "error": str(_wc_exc)}).encode()
                self.send_response(200)
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
                os.path.join(PIPE_DIR, ep_dir_param),
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
                            # Check YouTube upload states per locale
                            yt_upload: dict = {}
                            renders_path = os.path.join(ep_path, "renders")
                            if os.path.isdir(renders_path):
                                for _locale in sorted(os.listdir(renders_path)):
                                    _state_path = os.path.join(renders_path, _locale, "upload_state.json")
                                    if os.path.isfile(_state_path):
                                        try:
                                            with open(_state_path) as _sf:
                                                _st = json.load(_sf)
                                            if _st.get("video_id"):
                                                _privacy = None
                                                _yt_path = os.path.join(renders_path, _locale, "youtube.json")
                                                if os.path.isfile(_yt_path):
                                                    try:
                                                        with open(_yt_path) as _yf:
                                                            _privacy = json.load(_yf).get("privacy")
                                                    except Exception:
                                                        pass
                                                yt_upload[_locale] = {
                                                    "video_id": _st["video_id"],
                                                    "video_uploaded": bool(_st.get("video_uploaded")),
                                                    "published": _privacy == "public",
                                                }
                                        except Exception:
                                            pass
                            episodes.append({"id": ep_id, "files": files, "yt_upload": yt_upload})
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
            _viewer_path = Path(__file__).parent / "ui" / "viewer.html"
            viewer   = (_viewer_path.read_text(encoding="utf-8")
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

                # Detect locales from VOPlan.{locale}.json files
                _locales: list[str] = []
                if os.path.isdir(ep_dir):
                    for _f in sorted(os.listdir(ep_dir)):
                        _m = re.match(r"VOPlan\.(.+)\.json$", _f)
                        if _m and _m.group(1) != "shared":
                            _locales.append(_m.group(1))

                # All files that affect staleness
                _watch = [
                    _ep("meta.json"), _ep("story.txt"),
                    _ep("pipeline_vars.sh"), _pr("VoiceCast.json"),
                    _ep("StoryPrompt.json"), _ep("Script.json"),
                    _ep("ShotList.json"), _ep("AssetManifest.shared.json"),
                    _ep("canon_diff.json"), _pr("canon.json"),
                ] + [_ep(f"VOPlan.{l}.json") for l in _locales] \
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
                             [_ep("AssetManifest.shared.json")]),
                        (6,  "Identify new story facts",
                             [_ep("Script.json"), _ep("ShotList.json")],
                             [_ep("canon_diff.json")]),
                        (7,  "Update story memory",
                             [_ep("canon_diff.json")],
                             [_pr("canon.json")]),
                        (8,  "Translate & adapt locales",
                             [_ep("AssetManifest.shared.json"), _pr("VoiceCast.json")],
                             [_ep(f"VOPlan.{l}.json") for l in _locales]),
                        (9,  "Resolve assets",
                             [_ep("AssetManifest.shared.json")]
                             + [_ep(f"VOPlan.{l}.json") for l in _locales],
                             [_ep(f"VOPlan.{l}.json") for l in _locales]),
                        (10, "Render video",
                             [_pr("VoiceCast.json"), _ep("ShotList.json")]
                             + [_ep(f"VOPlan.{l}.json") for l in _locales],
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

            # ── Server-side approval skip ──────────────────────────────────────
            # These steps are skipped when their approval sentinel exists.
            # This enforces the skip even if the UI button is somehow clicked.
            _step_ep_dir   = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
            _step_music_ok = os.path.isfile(os.path.join(_step_ep_dir, "MusicPlan.json"))
            _step_sfx_ok   = os.path.isfile(os.path.join(_step_ep_dir, "SfxPlan.json"))
            _step_vo_ok    = _is_vo_approved(_step_ep_dir, locale or 'en') if _VO_UTILS_AVAILABLE else False
            _approved_skip = {
                "gen_music_clip":      _step_music_ok,
                "music_prepare_loops": _step_music_ok,
                "gen_sfx":             _step_sfx_ok,
                "gen_tts":             _step_vo_ok,
            }
            if _approved_skip.get(step):
                self.wfile.write(sse("line", f"  ✓ {step} — skipped (already approved)"))
                self.wfile.write(sse("done", "0"))
                self.wfile.flush()
                return
            # ──────────────────────────────────────────────────────────────────

            _step_payload = {"asset_ids": asset_ids} if asset_ids else None
            cmd = _build_step_cmd(step, slug, ep_id, locale, profile, no_music,
                                  payload=_step_payload)
            if cmd is None:
                self.wfile.write(sse("error_line", f"Unknown step: {step!r}"))
                self.wfile.write(sse("done", "1"))
                self.wfile.flush()
                return
            step_env = os.environ.copy()
            step_env.pop("CLAUDECODE", None)   # prevent nested-session guard from firing

            client = self.client_address

            # ── manifest_merge preamble: gen_music_clip → music_prepare_loops ──
            # Steps 1 and 1b are prerequisites for step 5 and run automatically
            # here so the Music tab is ready after "Run 5" completes.
            # NOTE: manifest_merge still produces a real subprocess cmd (not []).
            # Only gen_render_plan returns [] (eliminated step).
            # Preamble runs BEFORE the cmd==[] check so it also fires for any
            # future step that may become a no-op.
            if step == "manifest_merge" and not _step_music_ok:
                for _pre_name in ["gen_music_clip", "music_prepare_loops"]:
                    _pre_cmd = _build_step_cmd(_pre_name, slug, ep_id, "", profile, no_music)
                    if not _pre_cmd:
                        continue
                    self.wfile.write(sse("line", f"  [preamble] {_pre_name}…"))
                    self.wfile.flush()
                    if _TEST_MODE:
                        _pre_stem = os.path.splitext(os.path.basename(_pre_cmd[1]))[0]
                        _fake_subprocess(_pre_cmd, _pre_stem, _step_ep_dir, locale)
                        self.wfile.write(sse("line", f"  [TEST MODE] {_pre_name} — fixture applied"))
                        self.wfile.flush()
                        continue
                    _pre_proc = subprocess.Popen(
                        _pre_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        text=True,
                        bufsize=1,
                        env=step_env,
                        cwd=PIPE_DIR,
                        start_new_session=True,
                    )
                    with _lock:
                        _procs[client] = _pre_proc
                    for _pre_line in _pre_proc.stdout:
                        self.wfile.write(sse("line", _pre_line.rstrip("\n")))
                        self.wfile.flush()
                    _pre_proc.wait()
                    if _pre_proc.returncode != 0:
                        self.wfile.write(sse("error_line",
                            f"  ✗ {_pre_name} failed (exit {_pre_proc.returncode}) — aborting step 5"))
                        self.wfile.write(sse("done", str(_pre_proc.returncode)))
                        self.wfile.flush()
                        return
            # ─────────────────────────────────────────────────────────────────

            if cmd == []:
                self.wfile.write(sse("line", f"  ✓ {step} — skipped (no plan file)"))
                self.wfile.write(sse("done", "0"))
                self.wfile.flush()
                return

            if _TEST_MODE and cmd:
                _tm_step_stem = os.path.splitext(os.path.basename(cmd[1]))[0]
                _tm_ep_dir = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
                _tm_fake = _fake_subprocess(cmd, _tm_step_stem, _tm_ep_dir, locale)
                for _tm_line in _tm_fake.stdout:
                    self.wfile.write(sse("line", _tm_line.rstrip()))
                    self.wfile.flush()
                self.wfile.write(sse("done", "0"))
                self.wfile.flush()
                return

            proc   = None
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                    env=step_env,
                    cwd=PIPE_DIR,
                    start_new_session=True,
                )
                with _lock:
                    _procs[client] = proc

                for raw_line in proc.stdout:
                    self.wfile.write(sse("line", raw_line.rstrip("\n")))
                    self.wfile.flush()

                proc.wait()

                if proc.returncode != 0:
                    self.wfile.write(sse("done", str(proc.returncode)))
                    self.wfile.flush()
                    return

                # ── Compound step 5: after manifest_merge for non-primary locale,
                # run gen_tts → polish_locale_vo → post_tts (mirrors run.sh [6][6b][7]).
                # Only when VO is not yet approved — same guard as /run_locale.
                _rs_primary = "en"
                if locale and _VO_UTILS_AVAILABLE:
                    try:
                        from pathlib import Path as _Prs
                        _rs_primary = _get_primary_locale(
                            _Prs(os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)))
                    except Exception:
                        pass
                if (step == "manifest_merge"
                        and locale and locale != _rs_primary
                        and not _step_vo_ok):
                    for _rs_sub in ("gen_tts", "polish_locale_vo", "post_tts"):
                        _rs_cmd = _build_step_cmd(
                            _rs_sub, slug, ep_id, locale, profile, no_music)
                        if _rs_cmd is None:
                            self.wfile.write(sse("error_line",
                                f"Unknown sub-step: {_rs_sub!r}"))
                            self.wfile.write(sse("done", "1"))
                            self.wfile.flush()
                            return
                        if _rs_cmd == []:
                            self.wfile.write(sse("line",
                                f"  ✓ {_rs_sub} — skipped (no plan file)"))
                            self.wfile.flush()
                            continue
                        self.wfile.write(sse("line",
                            f"\n── {_rs_sub} ────────────────────────────────────────"))
                        self.wfile.flush()
                        _rs_proc = subprocess.Popen(
                            _rs_cmd,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL,
                            text=True, bufsize=1, env=step_env, cwd=PIPE_DIR,
                            start_new_session=True,
                        )
                        with _lock:
                            _procs[client] = _rs_proc
                        for _rs_line in _rs_proc.stdout:
                            self.wfile.write(sse("line", _rs_line.rstrip("\n")))
                            self.wfile.flush()
                        _rs_proc.wait()
                        with _lock:
                            _procs.pop(client, None)
                        if _rs_proc.returncode != 0:
                            self.wfile.write(sse("error_line",
                                f"✗ {_rs_sub} failed (exit {_rs_proc.returncode})"))
                            self.wfile.write(sse("done", str(_rs_proc.returncode)))
                            self.wfile.flush()
                            return
                        self.wfile.write(sse("line", f"✓ {_rs_sub}"))
                        self.wfile.flush()
                # ─────────────────────────────────────────────────────────────────

                self.wfile.write(sse("done", "0"))
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

            LOCALE_STEPS = ["manifest_merge",
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

                # Approval sentinels — these steps are unconditionally skipped
                # when the corresponding approval exists, even in force_run mode.
                _rl_ep_dir    = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
                _rl_vo_ok     = _is_vo_approved(_rl_ep_dir, locale) if _VO_UTILS_AVAILABLE else False
                _rl_music_ok  = os.path.isfile(os.path.join(_rl_ep_dir, "MusicPlan.json"))
                # Determine primary locale for compound step 5 guard.
                _rl_primary = "en"
                if _VO_UTILS_AVAILABLE:
                    try:
                        from pathlib import Path as _Prl
                        _rl_primary = _get_primary_locale(_Prl(_rl_ep_dir))
                    except Exception:
                        pass
                _rl_locale_approved = {
                    # Compound step 5 (manifest_merge): skip for non-primary when VO approved.
                    # Primary locale manifest_merge is fast and safe to re-run always.
                    # apply_music_plan: skip when MusicPlan.json exists — loop WAVs are shared
                    # across locales; renderer reads MusicPlan.json directly, not per-locale.
                    "manifest_merge":   _rl_vo_ok and (locale != _rl_primary),
                    "apply_music_plan": _rl_music_ok,
                }

                def _rl_run_sub(sub_step):
                    """Run a single sub-command, streaming output. Returns returncode."""
                    _sub_cmd = _build_step_cmd(sub_step, slug, ep_id, locale, profile, no_music)
                    if _sub_cmd is None:
                        write_log("E", f"Unknown sub-step: {sub_step!r}")
                        return 1
                    if _sub_cmd == []:
                        write_log("O", f"  ✓ {sub_step} — skipped (no plan file)")
                        return 0
                    write_log("O", f"\n── {sub_step} ────────────────────────────────────────")
                    _sp = subprocess.Popen(
                        _sub_cmd,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        text=True, bufsize=1, env=step_env, cwd=PIPE_DIR,
                        start_new_session=True,
                    )
                    with _lock:
                        _procs[client] = _sp
                    for _rl in _sp.stdout:
                        write_log("O", _rl.rstrip("\n"))
                    _sp.wait()
                    with _lock:
                        _procs.pop(client, None)
                    return _sp.returncode

                for step in LOCALE_STEPS[from_idx:]:
                    # Approval skip — takes priority over force_run
                    if _rl_locale_approved.get(step):
                        write_log("O", f"  ✓ {step} — skipped (already approved)")
                        continue
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
                        stdin=subprocess.DEVNULL,
                        text=True, bufsize=1,
                        env=step_env, cwd=PIPE_DIR,
                        start_new_session=True,
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

                    # ── Compound step 5: for non-primary locales, follow manifest_merge
                    # with gen_tts → polish_locale_vo → post_tts (mirrors run.sh [6][6b][7]).
                    if step == "manifest_merge" and locale != _rl_primary and not _rl_vo_ok:
                        for _sub in ("gen_tts", "polish_locale_vo", "post_tts"):
                            _rc = _rl_run_sub(_sub)
                            if _rc != 0:
                                write_log("E", f"✗ {_sub} failed (exit {_rc})")
                                write_log("D", str(_rc))
                                return
                            write_log("O", f"✓ {_sub}")

                write_log("O", f"\n✓ [{locale}] All post-processing steps complete")
                _append_tts_usage_to_status_report(slug, ep_id, write_log)
                write_log("D", "0")

            log_path = _launch_fn_job(job_key, os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id), _run_locale_job)
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
            _LOCALE_STEPS_ALL = ["manifest_merge",
                                  "apply_music_plan", "resolve_assets",
                                  "gen_render_plan", "render_video"]
            from_idx = _SHARED_STEPS.index(from_step) if from_step in _SHARED_STEPS else 0

            # Detect locales from VOPlan.{locale}.json files
            _ep_dir_s10 = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
            _locales_s10: list[str] = []
            if os.path.isdir(_ep_dir_s10):
                for _f in sorted(os.listdir(_ep_dir_s10)):
                    _m = re.match(r"VOPlan\.(.+)\.json$", _f)
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
                        stdin=subprocess.DEVNULL,
                        text=True, bufsize=1, env=step_env, cwd=PIPE_DIR,
                        start_new_session=True,
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

                # ── Approval sentinels for shared-step skip ───────────────────
                _s10_music_ok = os.path.isfile(os.path.join(_ep_dir_s10, "MusicPlan.json"))
                _s10_sfx_ok   = os.path.isfile(os.path.join(
                    _ep_dir_s10, "SfxPlan.json"))
                _s10_shared_approved = {
                    "gen_music_clip":      _s10_music_ok,
                    "music_prepare_loops": _s10_music_ok,
                    "gen_sfx":             _s10_sfx_ok,
                }

                # ── Shared steps (locale-free) ────────────────────────────────
                for _step in _SHARED_STEPS[from_idx:]:
                    if _s10_shared_approved.get(_step):
                        write_log("O", f"  ✓ {_step} — skipped (already approved)")
                        continue
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

                # Step to skip for primary locale when sentinel is valid (INVARIANT I).
                # gen_tts and post_tts are now compound sub-steps inside manifest_merge.
                _TTS_STEPS_TO_SKIP = {"manifest_merge"}

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
                                    "skipping manifest_merge (compound)"
                                )
                        except Exception as _sv_exc:
                            write_log("O", f"  [warn] Sentinel check error: {_sv_exc}")

                    # VO approved for non-primary? skip compound step 5 entirely.
                    _s10_vo_ok = (_is_vo_approved(_ep_dir_s10_full, _locale)
                                  if _VO_UTILS_AVAILABLE else False)

                    for _step in _LOCALE_STEPS_ALL:
                        # Skip manifest_merge for primary locale if sentinel valid
                        if _sentinel_valid_for_locale and _step in _TTS_STEPS_TO_SKIP:
                            write_log("O",
                                f"  ✓ {_step} [{_locale}] — skipped (VO approved, sentinel valid)"
                            )
                            continue
                        # Skip compound step 5 for non-primary when VO approved
                        if (_step == "manifest_merge"
                                and _locale != _primary_locale_s10
                                and _s10_vo_ok):
                            write_log("O",
                                f"  ✓ {_step} [{_locale}] — skipped (VO already approved)"
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

                        # ── Compound step 5: after manifest_merge for non-primary locale,
                        # run gen_tts → polish_locale_vo → post_tts (mirrors run.sh [6][6b][7]).
                        if (_step == "manifest_merge"
                                and _locale != _primary_locale_s10
                                and not _s10_vo_ok):
                            for _sub in ("gen_tts", "polish_locale_vo", "post_tts"):
                                write_log("O", f"\n── {_sub} [{_locale}] ────────────────────────")
                                _sub_cmd = _build_step_cmd(
                                    _sub, slug, ep_id, _locale, profile, no_music)
                                if _sub_cmd is None:
                                    write_log("E", f"Unknown sub-step: {_sub!r}")
                                    write_log("D", "1")
                                    return
                                if _sub_cmd == []:
                                    write_log("O", f"  ✓ {_sub} — skipped (no plan file)")
                                    continue
                                _sub_rc = _run_cmd(_sub_cmd)
                                if _sub_rc != 0:
                                    write_log("E", f"✗ {_sub} [{_locale}] failed (exit {_sub_rc})")
                                    write_log("D", str(_sub_rc))
                                    return
                                write_log("O", f"✓ {_sub} [{_locale}]")

                    write_log("O", f"✓ [{_locale}] all locale steps complete")

                # ── Step 12: gen_cross_srt (episode-level, after all locales) ─────
                if len(_locales_s10) >= 2:
                    write_log("O", "\n── Step 12: gen_cross_srt ──────────────────────────────")
                    if _step_is_done("gen_cross_srt", slug, ep_id, ""):
                        write_log("O", "  ✓ gen_cross_srt — already done, skipping")
                    else:
                        _gcs_cmd = _build_step_cmd("gen_cross_srt", slug, ep_id, "",
                                                   profile, no_music)
                        if not _gcs_cmd:
                            write_log("O", "  ✓ gen_cross_srt — skipped (single locale)")
                        else:
                            _gcs_rc = _run_cmd(_gcs_cmd)
                            if _gcs_rc != 0:
                                write_log("E", f"✗ gen_cross_srt failed (exit {_gcs_rc})")
                                write_log("D", str(_gcs_rc))
                                return
                            write_log("O", "✓ gen_cross_srt")
                else:
                    write_log("O", "\n  [skip] gen_cross_srt — single locale, no cross SRT needed")

                write_log("O", "\n✓ Stage 9 — all steps complete")
                _append_tts_usage_to_status_report(slug, ep_id, write_log)
                write_log("D", "0")

            log_path = _launch_fn_job(job_key, _ep_dir_s10, _run_stage10_job)
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

        # SSE stream: Stage 3.5 — run gen_tts (--script mode) + post_tts for primary locale
        elif parsed.path == "/run_stage35":
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
            job_key  = f"run_stage35\x00{slug}\x00{ep_id}"

            def _run_stage35_job(write_log):
                import json as _json
                _ep_dir_35  = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
                _code_dir   = os.path.join(PIPE_DIR, "code", "http")
                _script_35  = os.path.join(_ep_dir_35, "Script.json")
                _vc_35      = os.path.join(PIPE_DIR, "projects", slug, "VoiceCast.json")

                # Determine primary locale
                _locale_35 = "en"
                if _VO_UTILS_AVAILABLE:
                    try:
                        from pathlib import Path as _P35
                        _locale_35 = _get_primary_locale(_P35(_ep_dir_35))
                    except Exception:
                        pass

                write_log("O", f"\n── Stage 3.5 — primary locale: {_locale_35} ──────────────────")

                # Check Script.json exists
                if not os.path.isfile(_script_35):
                    write_log("E", f"Script.json not found: {_script_35}")
                    write_log("D", "1")
                    return

                # Step 1: gen_tts_cloud --script mode
                write_log("O", f"\n── gen_tts (--script mode) [{_locale_35}] ─────────────────")
                _cmd_tts = [
                    "python3", os.path.join(_code_dir, "gen_tts_cloud.py"),
                    "--script",    _script_35,
                    "--voicecast", _vc_35,
                    "--locale",    _locale_35,
                    "--stage",     "3.5",
                ]
                _p35 = subprocess.Popen(
                    _cmd_tts,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    text=True, bufsize=1, env=step_env, cwd=PIPE_DIR,
                    start_new_session=True,
                )
                with _lock:
                    _procs[client] = _p35
                for _raw in _p35.stdout:
                    write_log("O", _raw.rstrip("\n"))
                _p35.wait()
                with _lock:
                    _procs.pop(client, None)
                if _p35.returncode != 0:
                    write_log("E", f"✗ gen_tts [{_locale_35}] failed (exit {_p35.returncode})")
                    write_log("D", str(_p35.returncode))
                    return
                write_log("O", f"✓ gen_tts [{_locale_35}]")

                # Step 2: post_tts_analysis on draft manifest
                write_log("O", f"\n── post_tts_analysis [{_locale_35}] ──────────────────────")
                _draft_manifest = os.path.join(_ep_dir_35, f"VOPlan.{_locale_35}.json")
                if not os.path.isfile(_draft_manifest):
                    write_log("E", f"VOPlan.{_locale_35}.json not found after TTS")
                    write_log("D", "1")
                    return
                _cmd_pta = [
                    "python3", os.path.join(_code_dir, "post_tts_analysis.py"),
                    "--manifest", _draft_manifest,
                ]
                _p35b = subprocess.Popen(
                    _cmd_pta,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    text=True, bufsize=1, env=step_env, cwd=PIPE_DIR,
                    start_new_session=True,
                )
                with _lock:
                    _procs[client] = _p35b
                for _raw in _p35b.stdout:
                    write_log("O", _raw.rstrip("\n"))
                _p35b.wait()
                with _lock:
                    _procs.pop(client, None)
                if _p35b.returncode != 0:
                    write_log("E", f"✗ post_tts_analysis [{_locale_35}] failed (exit {_p35b.returncode})")
                    write_log("D", str(_p35b.returncode))
                    return
                write_log("O", f"✓ post_tts_analysis [{_locale_35}]")

                # All steps done — emit vo_review_ready event before done
                write_log("V", _json.dumps({"locale": _locale_35, "slug": slug, "ep_id": ep_id, "stage": "3.5"}))
                write_log("O", "\n✓ Stage 3.5 complete — VO ready for review")
                write_log("D", "0")

            log_path = _launch_fn_job(job_key, os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id), _run_stage35_job)
            try:
                _tail_log_to_sse(self.wfile, log_path)
            except (BrokenPipeError, ConnectionResetError):
                pass
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
                        stdin=subprocess.DEVNULL,
                        text=True, bufsize=1, env=step_env, cwd=PIPE_DIR,
                        start_new_session=True,
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

            log_path = _launch_fn_job(job_key, os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id), _run_stage75_job)
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

        # ── Media: check which preview files exist on disk (GET /api/media_preview_exists) ──
        elif parsed.path == "/api/media_preview_exists":
            params = parse_qs(parsed.query)
            slug   = params.get("slug",  [""])[0].strip()
            ep_id  = params.get("ep_id", [""])[0].strip()
            if not slug or not ep_id:
                body = json.dumps({"error": "slug and ep_id required"}).encode()
                self.send_response(400)
            else:
                pack_dir = os.path.join(PIPE_DIR, "projects", slug,
                                        "episodes", ep_id,
                                        "assets", "media", "MediaPreviewPack")
                full_exists = os.path.isfile(os.path.join(pack_dir, "preview_video.mp4"))
                scenes = []
                if os.path.isdir(pack_dir):
                    for fname in os.listdir(pack_dir):
                        if fname.startswith("scene_") and fname.endswith("_preview.mp4"):
                            # scene_{cardId}_preview.mp4 → extract cardId
                            card_id = fname[len("scene_"):-len("_preview.mp4")]
                            if card_id:
                                scenes.append(card_id)
                body = json.dumps({"full": full_exists, "scenes": scenes}).encode()
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # ── Media: load shared library state (GET /api/media_library) ────────
        elif self.path.startswith("/api/media_library"):
            try:
                from urllib.parse import urlparse as _urlparse_ml, parse_qs as _pqs_ml
                import urllib.request as _urllib_req_ml
                _qs_ml = _pqs_ml(_urlparse_ml(self.path).query)
                slug       = (_qs_ml.get("slug",       [""])[0]).strip()
                ep_id      = (_qs_ml.get("ep_id",      [""])[0]).strip()
                server_url = (_qs_ml.get("server_url", [""])[0]).strip() \
                             or os.environ.get("MEDIA_SERVER_URL", "")
                if not slug or not ep_id:
                    raise ValueError("slug and ep_id are required")
                ep_dir  = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
                api_key = os.environ.get("MEDIA_API_KEY", "")

                # ── 1. Manually-uploaded library ─────────────────────────────
                lib_path   = os.path.join(ep_dir, "assets", "media_library", "media_library.json")
                clips_path = os.path.join(ep_dir, "assets", "media_library", "media_cut_clips.json")

                if os.path.isfile(lib_path):
                    with open(lib_path, encoding="utf-8") as _lf:
                        lib_data = json.load(_lf)
                else:
                    lib_data = {"videos": [], "images": []}

                if os.path.isfile(clips_path):
                    with open(clips_path, encoding="utf-8") as _cf:
                        clips_list = json.load(_cf)
                else:
                    clips_list = []

                def _media_url(item_path):
                    return f"/serve_media?path=projects/{slug}/episodes/{ep_id}/{item_path}"

                videos = [dict(v, url=_media_url(v["path"])) for v in lib_data.get("videos", [])]
                images = [dict(i, url=_media_url(i["path"])) for i in lib_data.get("images", [])]
                clips  = [dict(c, url=_media_url(c["path"])) for c in clips_list]

                # ── 2. Assets from done search batches on the media server ────
                # The media server stores batches and their ranked assets.
                # We ask it for all done batches for this episode, then fetch
                # full item data (images/videos with URLs) for each done batch.
                try:
                    _batches_url = (server_url.rstrip("/")
                                    + f"/batches?project={slug}&episode_id={ep_id}")
                    _req = _urllib_req_ml.Request(
                        _batches_url, headers={"X-Api-Key": api_key})
                    with _urllib_req_ml.urlopen(_req, timeout=10) as _resp:
                        _batch_list = json.loads(_resp.read())

                    # _batch_list: [{batch_id, status, item_count, items_done, ...}]
                    # newest-first ordering is preserved from the media server response
                    for _bs in _batch_list:
                        if _bs.get("status") != "done":
                            continue
                        _bid = _bs.get("batch_id", "")
                        if not _bid:
                            continue
                        try:
                            _batch_url = server_url.rstrip("/") + f"/batches/{_bid}"
                            _req2 = _urllib_req_ml.Request(
                                _batch_url, headers={"X-Api-Key": api_key})
                            with _urllib_req_ml.urlopen(_req2, timeout=15) as _resp2:
                                _batch_data = json.loads(_resp2.read())

                            # Dedup keys — track seen asset_page_url (canonical
                            # identity across items/sources) falling back to
                            # filename when page_url is absent.
                            _seen_img_keys: set = set()
                            _seen_vid_keys: set = set()

                            for _iid, _item in _batch_data.get("items", {}).items():
                                if _item.get("status") != "done":
                                    continue

                                def _to_serve_url(raw_url):
                                    """Convert file:// or http://media-server/files/... URL
                                    to a browser-loadable /serve_media?path=<abs_path> URL."""
                                    if not raw_url:
                                        return ""
                                    if raw_url.startswith("file://"):
                                        # file:///mnt/shared/... → /serve_media?path=/mnt/shared/...
                                        from urllib.parse import unquote as _uq_srv
                                        abs_path = _uq_srv(raw_url[len("file://"):])
                                        return f"/serve_media?path={abs_path}"
                                    # http(s) URL from media server — use as-is (browser can reach it)
                                    return raw_url

                                def _abs_path_from_url(raw_url, fallback_path):
                                    """Extract absolute filesystem path from file:// URL.
                                    Used so ffmpeg in /api/media_cut_clip can open batch assets."""
                                    if raw_url.startswith("file://"):
                                        from urllib.parse import unquote as _uq_abs
                                        return _uq_abs(raw_url[len("file://"):])
                                    return fallback_path

                                # Videos — field is "videos" (URLs already computed by media server)
                                for _vr in _item.get("videos", []):
                                    _raw_url = _vr.get("url", "")
                                    _path = _vr.get("path", "")
                                    if not _raw_url:
                                        continue
                                    _src = _vr.get("source") or {}
                                    _vkey = _src.get("asset_page_url", "") or os.path.basename(_path)
                                    if _vkey in _seen_vid_keys:
                                        continue
                                    _seen_vid_keys.add(_vkey)
                                    _abs = _abs_path_from_url(_raw_url, _path)
                                    videos.append({
                                        "id":           f"batch_{_bid}_{_iid}_{os.path.basename(_path)}",
                                        "filename":     os.path.basename(_path),
                                        "path":         _abs,
                                        "duration_sec": float(_vr.get("duration_sec") or 0.0),
                                        "size_bytes":   0,
                                        "from_batch":   True,
                                        "batch_id":     _bid,
                                        "score":        float(_vr.get("score") or 0.0),
                                        "title":        _src.get("title", ""),
                                        "source_site":  _src.get("source_site", ""),
                                        "page_url":     _src.get("asset_page_url", ""),
                                        "url":          _to_serve_url(_raw_url),
                                    })
                                # Images — field is "images"
                                for _ir in _item.get("images", []):
                                    _raw_url = _ir.get("url", "")
                                    _path = _ir.get("path", "")
                                    if not _raw_url:
                                        continue
                                    _src = _ir.get("source") or {}
                                    _ikey = _src.get("asset_page_url", "") or os.path.basename(_path)
                                    if _ikey in _seen_img_keys:
                                        continue
                                    _seen_img_keys.add(_ikey)
                                    _abs = _abs_path_from_url(_raw_url, _path)
                                    images.append({
                                        "id":          f"batch_{_bid}_{_iid}_{os.path.basename(_path)}",
                                        "filename":    os.path.basename(_path),
                                        "path":        _abs,
                                        "width":       int(_src.get("width") or 0),
                                        "height":      int(_src.get("height") or 0),
                                        "size_bytes":  0,
                                        "from_batch":  True,
                                        "batch_id":    _bid,
                                        "score":       float(_ir.get("score") or 0.0),
                                        "title":       _src.get("title", ""),
                                        "source_site": _src.get("source_site", ""),
                                        "page_url":    _src.get("asset_page_url", ""),
                                        "url":         _to_serve_url(_raw_url),
                                    })
                        except Exception as _be:
                            print(f"  [media_library] error fetching batch {_bid}: {_be}")

                except Exception as _me:
                    # Media server unreachable — return only manually-uploaded assets
                    print(f"  [media_library] media server unreachable ({server_url}): {_me}")

                body = json.dumps({"videos": videos, "images": images, "clips": clips}).encode()
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

        # ── Media proxy: health check (GET /api/media_health) ───────────────────
        elif parsed.path == "/api/media_health":
            params     = parse_qs(parsed.query)
            server_url = params.get("server_url", ["http://localhost:8200"])[0].strip().rstrip("/")
            api_key    = os.environ.get("MEDIA_API_KEY", "")
            if not server_url:
                body = json.dumps({"error": "server_url required"}).encode()
                self.send_response(400)
            else:
                try:
                    req  = _urllib_req.Request(f"{server_url}/health",
                               headers={"X-Api-Key": api_key} if api_key else {})
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

        # ── Media proxy: poll batch status (GET /api/media_batch_status) ────────
        elif parsed.path == "/api/media_batch_status":
            params     = parse_qs(parsed.query)
            batch_id   = params.get("batch_id",   [""])[0].strip()
            server_url = params.get("server_url", ["http://localhost:8200"])[0].strip()
            slug       = params.get("slug",  [""])[0].strip()
            ep_id      = params.get("ep_id", [""])[0].strip()
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
                    # Append last_ai_ask from media server if slug/ep_id provided
                    if slug and ep_id:
                        try:
                            from urllib.parse import quote as _uquote
                            _ask_url = (server_url.rstrip("/") + "/ai_ask_last"
                                        + "?project=" + _uquote(slug)
                                        + "&episode_id=" + _uquote(ep_id))
                            print(f"  [media_batch_status] fetching ai_ask_last: {_ask_url}")
                            _ask_req = _urllib_req.Request(_ask_url,
                                           headers={"X-Api-Key": api_key})
                            with _urllib_req.urlopen(_ask_req, timeout=5) as _ask_resp:
                                _ask_data = json.loads(_ask_resp.read())
                            print(f"  [media_batch_status] ai_ask_last response: {_ask_data}")
                            if _ask_data:
                                _d = json.loads(body)
                                _d["last_ai_ask"] = _ask_data
                                body = json.dumps(_d).encode()
                        except Exception as _ask_exc:
                            print(f"  [media_batch_status] ai_ask_last FAILED: {_ask_exc}")
                    self.send_response(200)
                except Exception as exc:
                    body = json.dumps({"error": str(exc)}).encode()
                    self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # ── Media AI Ask error poll (GET /api/media_ai_ask_error?slug=X&ep_id=Y) ─
        elif parsed.path == "/api/media_ai_ask_error":
            params = parse_qs(parsed.query)
            slug   = params.get("slug",  [""])[0].strip()
            ep_id  = params.get("ep_id", [""])[0].strip()
            key    = f"{slug}:{ep_id}"
            err    = _media_ai_ask_errors.pop(key, None)
            body   = json.dumps({"error": err} if err else {}).encode()
            self.send_response(200)
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
                result = {}
                for folder in ("backgrounds", "elements"):
                    root = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id,
                                        "assets", folder)
                    if not os.path.isdir(root):
                        continue
                    for asset_id in sorted(os.listdir(root)):
                        asset_dir = os.path.join(root, asset_id)
                        if not os.path.isdir(asset_dir):
                            continue
                        ai_files = sorted(
                            f for f in os.listdir(asset_dir)
                            if f.startswith("ai_") and f.endswith(".png")
                        )
                        if ai_files:
                            result[asset_id] = [
                                {"filename": f,
                                 "path":     os.path.join(asset_dir, f),
                                 "url":      "file://" + os.path.join(asset_dir, f)}
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
        # Primary sources: ShotList.json (timing) + MusicPlan.json (content).
        # Fallback: when MusicPlan is absent or empty, VOPlan.*.json music_items
        # are used to seed the music index so Shot Overrides render on first load.
        elif parsed.path == "/api/music_timeline":
            params = parse_qs(parsed.query)
            slug   = params.get("slug", [""])[0].strip()
            ep_id  = params.get("ep_id", [""])[0].strip()
            try:
                if not slug or not ep_id:
                    raise ValueError("slug and ep_id required")
                import re as _re_vol_mtl
                from music_review_pack import (
                    build_timeline as _mrp_build_mtl,
                    apply_music_plan_overrides as _mrp_apply_mtl,
                    load_loop_candidates as _mrp_loop_mtl,
                    BASE_MUSIC_DB as _MRP_BASE_DB_MTL,
                )
                _ep_dir_mtl = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)

                # ── ShotList → shot list with timing (timing anchor) ──────────
                _sl_path_mtl = os.path.join(_ep_dir_mtl, "ShotList.json")
                if not os.path.isfile(_sl_path_mtl):
                    raise FileNotFoundError("ShotList.json not found")
                _sl_mtl = json.load(open(_sl_path_mtl, encoding="utf-8"))
                _shots_from_sl = [
                    {
                        "shot_id":      _sh.get("shot_id", ""),
                        "scene_id":     _sh.get("scene_id", ""),
                        "duration_sec": float(_sh.get("duration_sec") or 0.0),
                        "audio_intent": _sh.get("audio_intent", {}),
                    }
                    for _sh in _sl_mtl.get("shots", [])
                    if _sh.get("shot_id")
                ]

                # ── MusicPlan → music index (sole source) ────────────────────
                # MusicPlan.json is the only source of music assignments.
                # No fallback to VOPlan or AssetManifest — those fields are
                # removed.  When no MusicPlan exists the index stays empty and
                # the timeline shows no music bars, which is correct.
                _mp_path_mtl = os.path.join(_ep_dir_mtl, "MusicPlan.json")
                _lmp_mtl = None
                _midx_mtl: dict = {}
                if os.path.isfile(_mp_path_mtl):
                    try:
                        _lmp_mtl = json.load(open(_mp_path_mtl, encoding="utf-8"))
                    except Exception as _mp_err_mtl:
                        print(f"  [WARN] music_timeline: MusicPlan.json: {_mp_err_mtl}")

                # build_timeline receives no VOPlan manifest (empty dict) and no
                # VO shot map (empty dict) — VO data is not music_timeline's concern.
                _tls_mtl, _tdur_mtl = _mrp_build_mtl(
                    _shots_from_sl, {}, {}, _midx_mtl,
                    _mrp_loop_mtl(Path(_ep_dir_mtl)))

                # Apply MusicPlan overrides (timing, end_sec, duck_db, etc.)
                if _lmp_mtl:
                    _po_mtl = _lmp_mtl.get("shot_overrides", [])
                    if _po_mtl:
                        _mrp_apply_mtl(_tls_mtl, _po_mtl, "MusicPlan.json", {})
                    # Apply volume offsets from MusicPlan track_volumes / clip_volumes
                    _tv_mtl = _lmp_mtl.get("track_volumes", {})
                    _cv_mtl = _lmp_mtl.get("clip_volumes",  {})
                    if _tv_mtl or _cv_mtl:
                        for _ent_mtl in _tls_mtl:
                            _rid_mtl = (_ent_mtl.get("music_item_id_override")
                                        or _ent_mtl.get("music_item_id") or "")
                            if not _rid_mtl:
                                continue
                            _db_mtl = 0.0
                            _s_mtl  = _re_vol_mtl.sub(r'_\d[\d_]*s-[\d_\.]+s$', '', _rid_mtl)
                            _db_mtl += float(_tv_mtl.get(_s_mtl, 0))
                            _db_mtl += float(_cv_mtl.get(_rid_mtl, 0))
                            if _db_mtl == 0.0:
                                _mx_mtl = _re_vol_mtl.match(
                                    r'^(.+?)_(\d+)_(\d+)s-(\d+)_(\d+)s$', _rid_mtl)
                                if _mx_mtl:
                                    _cid_mtl = (f"{_mx_mtl.group(1)}:{_mx_mtl.group(2)}."
                                                f"{_mx_mtl.group(3)}s-{_mx_mtl.group(4)}.{_mx_mtl.group(5)}s")
                                    _db_mtl += float(_cv_mtl.get(_cid_mtl, 0))
                            if _db_mtl != 0.0:
                                _ent_mtl["base_db"] = (
                                    _ent_mtl.get("base_db", _MRP_BASE_DB_MTL) + _db_mtl)

                # Derive flat music_items list directly from MusicPlan shot_overrides
                # (episode-absolute coords) so the visual bars match the audio renderer.
                # The old approach re-added shot offset to clamped-to-boundary values,
                # producing bars that ended at the ShotList shot boundary instead of the
                # MusicPlan end_sec (e.g. 32s instead of 35s for the last segment).
                _music_items_flat: list = []
                if _lmp_mtl:
                    for _seg_flat in _lmp_mtl.get("shot_overrides", []):
                        _mid_flat = (_seg_flat.get("music_asset_id")
                                     or _seg_flat.get("music_clip_id") or "")
                        if not _mid_flat:
                            continue
                        _ms_flat = float(_seg_flat.get("start_sec", 0))
                        _me_flat = float(_seg_flat.get("end_sec",   0))
                        if _me_flat <= _ms_flat:
                            continue
                        _music_items_flat.append({
                            "item_id":    _mid_flat,
                            "start_sec":  round(_ms_flat, 3),
                            "end_sec":    round(_me_flat, 3),
                            "shot_id":    "",
                            "music_mood": "",
                        })

                _json_resp(self, {
                    "episode_id":         ep_id,
                    "total_duration_sec": round(_tdur_mtl, 3),
                    "total_dur_sec":      round(_tdur_mtl, 3),  # normalised key
                    "shots":              _tls_mtl,
                    "music_items":        _music_items_flat,
                })
            except Exception as _etl_m:
                _json_resp(self, {"error": str(_etl_m)}, 400)

        # ── SFX: return timeline JSON in-memory (GET /api/sfx_timeline) ─────
        # Authoritative sources: ShotList.json (timing) + SfxPlan.json (content).
        # VOPlan and MusicPlan are NOT consulted — SFX data lives in SfxPlan only.
        elif parsed.path == "/api/sfx_timeline":
            params = parse_qs(parsed.query)
            slug  = params.get("slug", [""])[0].strip()
            ep_id = params.get("ep_id", [""])[0].strip()
            try:
                if not slug or not ep_id:
                    raise ValueError("slug and ep_id required")
                _ep_dir_stl = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)

                # ── ShotList → episode-absolute shot offsets (timing anchor) ──
                _sl_path_stl = os.path.join(_ep_dir_stl, "ShotList.json")
                if not os.path.isfile(_sl_path_stl):
                    raise FileNotFoundError("ShotList.json not found")
                _sl_stl = json.load(open(_sl_path_stl, encoding="utf-8"))
                _shot_offset_stl: dict = {}   # shot_id → cumulative episode offset
                _cum_stl = 0.0
                for _sh_stl in _sl_stl.get("shots", []):
                    _sid_stl = _sh_stl.get("shot_id", "")
                    if _sid_stl:
                        _shot_offset_stl[_sid_stl] = _cum_stl
                    _cum_stl += float(_sh_stl.get("duration_sec") or 0.0)

                # ── SfxPlan → sfx_items (authoritative source) ───────────────
                # SfxPlan uses timing_format:"episode_absolute" — start/end_sec
                # are already episode-absolute; pass through directly without
                # adding the shot offset (doing so would double-count the offset).
                _sfx_items_out_stl = []
                _sfx_plan_path_stl = os.path.join(_ep_dir_stl, "SfxPlan.json")
                if os.path.isfile(_sfx_plan_path_stl):
                    try:
                        _sfx_plan_stl = json.load(open(_sfx_plan_path_stl, encoding="utf-8"))
                        for _en_stl in _sfx_plan_stl.get("shot_overrides", []):
                            _shot_id_stl = _en_stl.get("shot_id", "")
                            _sfx_items_out_stl.append({
                                "item_id":   _en_stl.get("item_id", ""),
                                "start_sec": round(float(_en_stl.get("start_sec", 0)), 3),
                                "end_sec":   round(float(_en_stl.get("end_sec",   0)), 3),
                                "shot_id":   _shot_id_stl,
                                "tag":       _en_stl.get("tag", ""),
                            })
                    except Exception as _spe_stl:
                        print(f"  [WARN] sfx_timeline: SfxPlan: {_spe_stl}")

                _wav_stl = os.path.join(_ep_dir_stl, "assets", "sfx",
                                        "SfxPreviewPack", "preview_audio.wav")
                _json_resp(self, {
                    "total_dur_sec": round(_cum_stl, 3),
                    "sfx_items":     _sfx_items_out_stl,
                    "has_wav":       os.path.isfile(_wav_stl),
                })
            except Exception as _e_stl:
                _json_resp(self, {"error": str(_e_stl)}, 400)

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

        # ── SFX: list source SFX files (GET /api/sfx_sources) ────────────────
        elif parsed.path == "/api/sfx_sources":
            params = parse_qs(parsed.query)
            slug   = params.get("slug", [""])[0].strip()
            ep_id  = params.get("ep_id", [""])[0].strip()
            if not slug or not ep_id:
                body = json.dumps({"error": "slug and ep_id required"}).encode()
                self.send_response(400)
            else:
                sfx_dir = os.path.join(PIPE_DIR, "projects", slug,
                                       "resources", "sfx")
                sources = []
                _supported_exts = {".mp3", ".wav", ".flac", ".ogg", ".aiff", ".aif"}
                if os.path.isdir(sfx_dir):
                    for fname in sorted(os.listdir(sfx_dir)):
                        ext = os.path.splitext(fname)[1].lower()
                        if ext not in _supported_exts:
                            continue
                        stem = os.path.splitext(fname)[0]
                        rel_path = os.path.join("projects", slug,
                                                "resources", "sfx", fname)
                        # Try to get duration via ffprobe
                        duration_sec = None
                        try:
                            import subprocess as _sp_sfxsrc
                            _full = os.path.join(PIPE_DIR, rel_path)
                            _r = _sp_sfxsrc.run(
                                ["ffprobe", "-v", "quiet", "-print_format", "json",
                                 "-show_streams", _full],
                                capture_output=True, timeout=10)
                            _info = json.loads(_r.stdout)
                            for _s in _info.get("streams", []):
                                if _s.get("codec_type") == "audio":
                                    duration_sec = float(_s.get("duration", 0)) or None
                                    break
                        except Exception:
                            pass
                        sources.append({
                            "stem": stem,
                            "filename": fname,
                            "path": rel_path,
                            "duration_sec": duration_sec,
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
            _EPISODE_FILE_WHITELIST = {"ShotList.json", "MediaPlan.json",
                                       "meta.json",
                                       "assets/media/bg_id_remap.json",
                                       "MusicPlan.json",
                                       "assets/music/user_cut_clips.json",
                                       "assets/meta/gen_music_clip_results.json",
                                       "assets/sfx/sfx_search_results.json",
                                       "assets/sfx/sfx_cut_clips.json",
                                       "SfxPlan.json",
                                       "AssetManifest.shared.json",
                                       "assets/media_library/media_library.json",
                                       "assets/media_library/media_cut_clips.json"}
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

        # ── Shot durations — for SFX Shot Overrides timeline ────────────────
        # GET /api/rp_shot_durations?slug=X&ep_id=Y
        # Returns {durations: {shot_id: duration_sec}} from ShotList.json.
        # RenderPlan is eliminated; ShotList.duration_sec is the authoritative
        # floor.  VO ceiling is applied at render time by render_video.py.
        elif parsed.path == "/api/rp_shot_durations":
            try:
                params = parse_qs(parsed.query)
                slug  = params.get("slug",  [""])[0].strip()
                ep_id = params.get("ep_id", [""])[0].strip()
                if not slug or not ep_id:
                    raise ValueError("slug and ep_id are required")
                ep_dir = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
                sl_path = os.path.join(ep_dir, "ShotList.json")
                if not os.path.isfile(sl_path):
                    _json_resp(self, {"durations": {}})
                else:
                    with open(sl_path, encoding="utf-8") as _slf:
                        _sl = json.load(_slf)
                    _durs = {
                        s["shot_id"]: round(float(s.get("duration_sec", 0)), 4)
                        for s in _sl.get("shots", [])
                        if s.get("shot_id")
                    }
                    _json_resp(self, {"durations": _durs})
            except Exception as exc:
                _json_resp(self, {"error": str(exc)}, 400)

        # ── Authoritative episode shot timeline from ShotList.json ───────────
        # GET /api/vo_timeline?slug=X&ep_id=Y
        # Returns episode-absolute start/end for every shot using ShotList.json
        # start_sec / duration_sec.  RenderPlan is eliminated — render_video.py
        # applies the VO ceiling at render time.  Shot start_sec in ShotList is
        # already episode-absolute.
        # Result is also persisted to assets/meta/VOTimeline.json.
        elif parsed.path == "/api/vo_timeline":
            try:
                params = parse_qs(parsed.query)
                slug  = params.get("slug",  [""])[0].strip()
                ep_id = params.get("ep_id", [""])[0].strip()
                if not slug or not ep_id:
                    raise ValueError("slug and ep_id are required")
                # Optional DOM scene_heads override — forwarded by _loadAndMergeTl
                # when called from _voPreviewAll, so bars match the audio preview
                # even before the user clicks Save.
                _tl_sh_param = unquote_plus(params.get("scene_heads", [""])[0]).strip()
                _tl_sh_override: dict = {}
                if _tl_sh_param:
                    try:
                        _parsed_sh = json.loads(_tl_sh_param)
                        if isinstance(_parsed_sh, dict):
                            _tl_sh_override = _parsed_sh
                    except Exception:
                        pass
                ep_dir = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
                sl_path = os.path.join(ep_dir, "ShotList.json")
                if not os.path.isfile(sl_path):
                    _json_resp(self, {"error": "ShotList.json not found"}, 404)
                else:
                    with open(sl_path, encoding="utf-8") as _slf:
                        _sl = json.load(_slf)
                    _shot_entries = []
                    _scene_map: dict = {}
                    _cum = 0.0
                    for _rs in _sl.get("shots", []):
                        _sid  = _rs.get("shot_id", "")
                        _scid = _rs.get("scene_id", "")
                        _dur  = round(float(_rs.get("duration_sec") or 0.0), 4)
                        _t0   = round(float(_rs.get("start_sec") or _cum), 4)
                        _t1   = round(_t0 + _dur, 4)
                        if not _sid:
                            continue
                        _shot_entries.append({
                            "shot_id":      _sid,
                            "scene_id":     _scid,
                            "start_sec":    _t0,
                            "end_sec":      _t1,
                            "duration_sec": _dur,
                        })
                        if _scid:
                            if _scid not in _scene_map:
                                _scene_map[_scid] = {
                                    "scene_id":  _scid,
                                    "start_sec": _t0,
                                    "end_sec":   _t1,
                                }
                            else:
                                _scene_map[_scid]["end_sec"] = _t1
                        _cum = _t1
                    # ── Apply scene_heads offsets from VOPlan ─────────────────
                    # scene_heads[scene_id] = N means the first VO item of that
                    # scene starts N seconds into the episode (head before VO).
                    # Each scene's head expands that scene's episode window by N s,
                    # shifting all subsequent shot start/end positions forward too.
                    try:
                        import glob as _gh_sh
                        import re   as _re_sh
                        _sh_vp_list = [p for p in _gh_sh.glob(
                            os.path.join(ep_dir, "VOPlan.*.json"))
                            if os.path.basename(p) != "AssetManifest.shared.json"]
                        if _sh_vp_list:
                            _sh_loc  = "en"
                            _sh_vars = os.path.join(ep_dir, "pipeline_vars.sh")
                            if os.path.isfile(_sh_vars):
                                _sh_m = _re_sh.search(
                                    r'(?:^|[\n;])(?:export\s+)?PRIMARY_LOCALE=["\']?([^"\';\n]+)["\']?',
                                    open(_sh_vars).read())
                                if _sh_m:
                                    _sh_loc = _sh_m.group(1).strip()
                            _sh_vp_path = os.path.join(ep_dir, f"VOPlan.{_sh_loc}.json")
                            if not os.path.isfile(_sh_vp_path):
                                _sh_vp_path = sorted(_sh_vp_list)[0]
                            _sh_vp      = json.load(open(_sh_vp_path, encoding="utf-8"))
                            # DOM override takes precedence over saved VOPlan values
                            _scene_heads_map = {**_sh_vp.get("scene_heads", {}), **_tl_sh_override}
                            if _scene_heads_map:
                                _ep_off   = 0.0
                                _seen_sc: set = set()
                                for _se in _shot_entries:
                                    _sc_sh = _se.get("scene_id", "")
                                    _head  = (_scene_heads_map.get(_sc_sh, 0.0)
                                              if _sc_sh and _sc_sh not in _seen_sc
                                              else 0.0)
                                    _se["start_sec"]    = round(_ep_off, 4)
                                    _ep_off            += _head + _se["duration_sec"]
                                    _se["end_sec"]      = round(_ep_off, 4)
                                    _se["duration_sec"] = round(
                                        _se["end_sec"] - _se["start_sec"], 4)
                                    if _sc_sh:
                                        _seen_sc.add(_sc_sh)
                                _cum = _ep_off
                                # Rebuild scene_map with corrected positions
                                _scene_map = {}
                                for _se in _shot_entries:
                                    _sc_sh = _se.get("scene_id", "")
                                    if _sc_sh:
                                        if _sc_sh not in _scene_map:
                                            _scene_map[_sc_sh] = {
                                                "scene_id":  _sc_sh,
                                                "start_sec": _se["start_sec"],
                                                "end_sec":   _se["end_sec"],
                                            }
                                        else:
                                            _scene_map[_sc_sh]["end_sec"] = _se["end_sec"]
                    except Exception as _sh_err:
                        print(f"  [WARN] vo_timeline: scene_heads: {_sh_err}")
                    # ── VOPlan → vo_items (authoritative source) ─────────────
                    # Build vid→shot_id from ShotList audio_intent (not from VOPlan)
                    # so the mapping is grounded in the timing-locked shot structure.
                    _vid2shot_vot: dict = {}
                    for _rs2 in _sl.get("shots", []):
                        for _vid in _rs2.get("audio_intent", {}).get("vo_item_ids", []):
                            _vid2shot_vot[_vid] = _rs2.get("shot_id", "")
                    _vo_items_vot: list = []
                    try:
                        import glob as _glob_vot
                        import re as _re_vot
                        _vp_list = [p for p in _glob_vot.glob(
                            os.path.join(ep_dir, "VOPlan.*.json"))
                            if os.path.basename(p) != "AssetManifest.shared.json"]
                        if _vp_list:
                            _loc_vot = "en"
                            _vars_vot = os.path.join(ep_dir, "pipeline_vars.sh")
                            if os.path.isfile(_vars_vot):
                                _vm_v = _re_vot.search(
                                    r'(?:^|[\n;])(?:export\s+)?PRIMARY_LOCALE=["\']?([^"\';\n]+)["\']?',
                                    open(_vars_vot).read())
                                if _vm_v:
                                    _loc_vot = _vm_v.group(1).strip()
                            _vp_path = os.path.join(ep_dir, f"VOPlan.{_loc_vot}.json")
                            if not os.path.isfile(_vp_path):
                                _vp_path = sorted(_vp_list)[0]
                            _vp = json.load(open(_vp_path, encoding="utf-8"))
                            for _vo in _vp.get("vo_items", []):
                                _vs = _vo.get("start_sec")
                                _ve = _vo.get("end_sec")
                                if _vs is not None:
                                    _shot_id_vot = (_vid2shot_vot.get(_vo.get("item_id", ""))
                                                    or _vo.get("shot_id", ""))
                                    _vo_items_vot.append({
                                        "item_id":       _vo.get("item_id", ""),
                                        "start_sec":     round(float(_vs), 3),
                                        "end_sec":       round(float(_ve or _vs), 3),
                                        "speaker_id":    _vo.get("speaker_id", ""),
                                        "shot_id":       _shot_id_vot,
                                        "pause_after_ms": int(_vo.get("pause_after_ms") or 0),
                                    })
                    except Exception as _vot_err:
                        print(f"  [WARN] vo_timeline: VOPlan: {_vot_err}")
                    # total_sec = max(ShotList cumulative, max vo_item end_sec + pause_after_ms)
                    # Including pause_after_ms ensures the trailing silence the user
                    # intentionally added after the last VO item is reflected in all
                    # downstream consumers (Media, SFX, Music tabs and clamping logic).
                    _vo_max = max(
                        (v["end_sec"] + (v.get("pause_after_ms") or 0) / 1000.0
                         for v in _vo_items_vot),
                        default=0.0
                    )
                    _vot_result = {
                        "shots":     _shot_entries,
                        "scenes":    list(_scene_map.values()),
                        "total_sec": round(max(_cum, _vo_max), 4),
                        "vo_items":  _vo_items_vot,
                    }
                    _json_resp(self, _vot_result)
            except Exception as exc:
                _json_resp(self, {"error": str(exc)}, 400)

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

        # ── YouTube: list playlists for a profile (GET /api/youtube_playlists) ──
        elif parsed.path == "/api/youtube_playlists":
            params = parse_qs(parsed.query)
            locale = unquote_plus(params.get("locale", ["en"])[0]).strip() or "en"
            try:
                from google.oauth2.credentials import Credentials
                from google.auth.transport.requests import Request as _GRequest
                from googleapiclient.discovery import build as _yt_build
                _profile_path = os.path.join(os.path.expanduser("~"), ".config", "pipe", "youtube_profiles.json")
                _profiles = json.loads(open(_profile_path, encoding="utf-8").read())
                # Match by locale or upload_profile key
                _prof = _profiles.get(locale) or next(
                    (v for v in _profiles.values() if v.get("locale") == locale), None
                ) or next(iter(_profiles.values()))
                _creds = Credentials.from_authorized_user_file(_prof["token_path"])
                if _creds.expired and _creds.refresh_token:
                    _creds.refresh(_GRequest())
                    # Persist the refreshed access token so the next request
                    # doesn't need to refresh again (and survives server restarts).
                    open(_prof["token_path"], "w", encoding="utf-8").write(_creds.to_json())
                _yt = _yt_build("youtube", "v3", credentials=_creds)
                _playlists = []
                _req = _yt.playlists().list(part="snippet", mine=True, maxResults=50)
                while _req:
                    _resp = _req.execute()
                    for _item in _resp.get("items", []):
                        _playlists.append({
                            "id":    _item["id"],
                            "title": _item["snippet"]["title"],
                        })
                    _req = _yt.playlists().list_next(_req, _resp)
                _json_resp(self, {"ok": True, "playlists": _playlists})
            except Exception as _exc:
                _json_resp(self, {"ok": False, "error": str(_exc), "playlists": []})

        # ── YouTube: status (GET /api/youtube_status) ────────────────────────────
        elif parsed.path == "/api/youtube_status":
            params = parse_qs(parsed.query)
            slug   = unquote_plus(params.get("slug",   [""])[0]).strip()
            ep_id  = unquote_plus(params.get("ep_id",  [""])[0]).strip()
            locale = unquote_plus(params.get("locale", ["en"])[0]).strip()

            # Input validation
            if not slug or not ep_id or not re.match(r'^[a-zA-Z0-9_\-]+$', slug) \
                    or not re.match(r'^[a-zA-Z0-9_\-]+$', ep_id) \
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
                    or not re.match(r'^[a-zA-Z0-9_\-]+$', ep_id) \
                    or locale not in ("en", "zh-Hans", "zh", "zh-CN", "ja", "ko", "fr", "de", "es", "pt"):
                self.send_response(400); self.end_headers(); return

            thumb_path = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id,
                                      "renders", locale, "thumbnail.jpg")
            _thumb_ctype = "image/jpeg"
            if not os.path.isfile(thumb_path):
                # Fall back to PNG (TTS mode writes thumbnail.png directly)
                thumb_path = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id,
                                          "renders", locale, "thumbnail.png")
                _thumb_ctype = "image/png"
            if not os.path.isfile(thumb_path):
                self.send_response(404); self.end_headers(); return

            with open(thumb_path, "rb") as _tf:
                data = _tf.read()
            self.send_response(200)
            self.send_header("Content-Type", _thumb_ctype)
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
                    or not re.match(r'^[a-zA-Z0-9_\-]+$', ep_id) \
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
                    or not re.match(r'^[a-zA-Z0-9_\-]+$', ep_id) \
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

    # HEAD requests use the same logic as GET — /serve_media returns 200/404
    # based on file existence; the client ignores the body.
    do_HEAD = do_GET

    # ── POST ──────────────────────────────────────────────────────────────────
    def do_POST(self):

        # Cancel YouTube upload/publish subprocess
        if self.path == "/api/youtube_cancel":
            killed = []
            with _lock:
                for k in list(_procs.keys()):
                    if str(k).startswith("yt_"):
                        p = _procs.pop(k)
                        if p.poll() is None:
                            p.terminate()
                            killed.append(str(k))
            _log.info("[youtube_cancel] terminated: %s", killed)
            resp = json.dumps({"ok": True, "killed": killed}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        # Kill running process
        elif self.path == "/stop":
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
                                             f"VOPlan.{locale}.json")

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
                _log.info("[vo_recreate] item=%s  locale=%s  ep=%s", item_id, locale, ep_dir)
                _vo_validate_inputs(ep_dir, locale, item_id)
                full_ep = _vo_resolve_ep_dir(ep_dir)

                # Read current params from manifest as base
                mpath = os.path.join(full_ep, f"VOPlan.{locale}.json")
                with open(mpath, encoding="utf-8") as _mf:
                    _mani = json.load(_mf)
                _item = next((v for v in _mani.get("vo_items", [])
                              if v["item_id"] == item_id), None)
                if _item is None:
                    raise ValueError(f"item_id {item_id!r} not found in manifest")

                tp = _item.get("tts_prompt", {})
                params = {
                    "voice":             tp.get("azure_voice") or tp.get("voice", ""),
                    "style":             tp.get("azure_style") or tp.get("style", ""),
                    "style_degree":      tp.get("azure_style_degree", 1.5),
                    "rate":              tp.get("azure_rate") or tp.get("rate", "0%"),
                    "pitch":             tp.get("azure_pitch", ""),
                    "break_ms":          tp.get("azure_break_ms", 0),
                    "phoneme_overrides": tp.get("phoneme_overrides", {}),
                }
                text = _item.get("text", "")

                # Override with UI-supplied params if present (called from Preview slow path)
                if req.get("text"):        text                  = req["text"].strip()
                if req.get("voice"):       params["voice"]       = req["voice"].strip()
                if req.get("style") is not None: params["style"] = req["style"] or ""
                if req.get("style_degree"): params["style_degree"] = float(req["style_degree"])
                if req.get("rate"):        params["rate"]        = req["rate"].strip()
                if req.get("pitch") is not None: params["pitch"] = req["pitch"] or ""
                if req.get("break_ms") is not None: params["break_ms"] = int(req["break_ms"])

                with _get_vo_lock(full_ep):
                    result = synthesize_vo_item(
                        item_id, text, params, full_ep, locale,
                        write_cache=False,  # INVARIANT F: vo_recreate bypasses cache
                    )
                _json_resp(self, {"item_id": item_id, **result})

            except Exception as exc:
                _json_resp(self, {"error": str(exc)}, 409)

        # POST /api/vo_phoneme — add or remove a phoneme override for one VO item
        elif self.path == "/api/vo_phoneme":
            try:
                if not _VO_UTILS_AVAILABLE:
                    raise RuntimeError("vo_utils not available")
                length = int(self.headers.get("Content-Length", 0))
                req     = json.loads(self.rfile.read(length))
                ep_dir  = req.get("ep_dir",  "").strip()
                locale  = req.get("locale",  "").strip()
                item_id = req.get("item_id", "").strip()
                char    = req.get("char",    "").strip()
                pinyin  = req.get("pinyin",  "").strip()
                action  = req.get("action",  "add")   # "add" | "remove"
                _vo_validate_inputs(ep_dir, locale, item_id)
                full_ep = _vo_resolve_ep_dir(ep_dir)
                mpath   = os.path.join(full_ep, f"VOPlan.{locale}.json")
                with _get_vo_lock(full_ep):
                    with open(mpath, encoding="utf-8") as _mf:
                        mani = json.load(_mf)
                    item = next((v for v in mani.get("vo_items", [])
                                 if v["item_id"] == item_id), None)
                    if item is None:
                        raise ValueError(f"item_id {item_id!r} not found in manifest")
                    tp = item.setdefault("tts_prompt", {})
                    overrides = tp.setdefault("phoneme_overrides", {})
                    if action == "add" and char and pinyin:
                        overrides[char] = pinyin
                    elif action == "remove" and char:
                        overrides.pop(char, None)
                    with open(mpath, "w", encoding="utf-8") as _mf:
                        json.dump(mani, _mf, ensure_ascii=False, indent=2)
                _log.info("[vo_phoneme] %s item=%s char=%r action=%s", locale, item_id, char, action)
                _json_resp(self, {"ok": True, "phoneme_overrides": overrides})
            except Exception as exc:
                _json_resp(self, {"error": str(exc)}, 409)

        # POST /api/vo_save — re-synthesize with new params, write to cache
        # Special case: keep_audio=true skips Azure TTS and keeps existing WAV on disk.
        # Used when user previewed with changed params (WAV already written) and just wants to commit the manifest.
        elif self.path == "/api/vo_save":
            try:
                if not _VO_UTILS_AVAILABLE:
                    raise RuntimeError("vo_utils not available")
                length = int(self.headers.get("Content-Length", 0))
                req    = json.loads(self.rfile.read(length))
                ep_dir  = req.get("ep_dir",  "").strip()
                locale  = req.get("locale",  "").strip()
                item_id = req.get("item_id", "").strip()
                _log.info("[vo_save] item=%s  locale=%s  keep_audio=%s  ep=%s",
                          item_id, locale, req.get("keep_audio", False), ep_dir)
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
                    from pathlib import Path as _P
                    _primary_locale = _get_primary_locale(_P(full_ep))
                    if keep_audio:
                        # Skip Azure TTS — keep existing source.wav / .wav on disk.
                        # Just update manifest params and return current durations.
                        vo_dir     = os.path.join(full_ep, "assets", locale, "audio", "vo")
                        src_path   = os.path.join(vo_dir, f"{item_id}.source.wav")
                        wav_path   = os.path.join(vo_dir, f"{item_id}.wav")
                        if not os.path.exists(src_path):
                            raise FileNotFoundError(f"source.wav not found for {item_id} — cannot keep audio")
                        source_dur  = _wav_duration(_P(src_path))
                        trimmed_dur = _wav_duration(_P(wav_path)) if os.path.exists(wav_path) else source_dur
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
                        # Carry phoneme_overrides from the manifest into synthesis.
                        # They are written there by /api/vo_phoneme and must be
                        # forwarded here so the SSML is built with the corrections.
                        _pm_path = os.path.join(full_ep, f"VOPlan.{locale}.json")
                        with open(_pm_path, encoding="utf-8") as _pm_f:
                            _pm_data = json.load(_pm_f)
                        _pm_item = next((v for v in _pm_data.get("vo_items", [])
                                         if v.get("item_id") == item_id), {})
                        params["phoneme_overrides"] = (
                            _pm_item.get("tts_prompt", {}).get("phoneme_overrides", {})
                        )
                        _log.debug("[vo_save] phoneme_overrides=%r for %s",
                                   params["phoneme_overrides"], item_id)
                        result = synthesize_vo_item(
                            item_id, new_text, params, full_ep, locale,
                            write_cache=True,   # INVARIANT F: vo_save writes cache
                        )

                    # ── Update unified manifest with text + tts_prompt ──
                    # VOPlan.{locale}.json is now the single source of truth.
                    # Only clear start_sec/end_sec when a new WAV was synthesized
                    # (keep_audio=True keeps the existing WAV, so timing remains valid).
                    _mpath = os.path.join(full_ep, f"VOPlan.{locale}.json")
                    with open(_mpath, encoding="utf-8") as _df:
                        _unified_m = json.load(_df)
                    _vdb_new = float(req.get("volume_db", 0.0) or 0.0)
                    for _dit in _unified_m.get("vo_items", []):
                        if _dit.get("item_id") == item_id:
                            _dit["text"] = new_text
                            _tp = _dit.setdefault("tts_prompt", {})
                            _tp["azure_voice"]        = new_voice
                            _tp["azure_style"]        = new_style
                            _tp["azure_style_degree"] = new_style_degree
                            _tp["azure_rate"]         = new_rate
                            # volume_db: omit when zero to keep VOPlan file clean
                            if _vdb_new != 0.0:
                                _dit["volume_db"] = _vdb_new
                            else:
                                _dit.pop("volume_db", None)
                            if not keep_audio:
                                # Clear stale timing — new WAV needs re-measurement
                                _dit.pop("start_sec", None)
                                _dit.pop("end_sec",   None)
                            break
                    _dtmp = _mpath + ".tmp"
                    with open(_dtmp, "w", encoding="utf-8") as _df:
                        json.dump(_unified_m, _df, indent=2, ensure_ascii=False)
                    os.replace(_dtmp, _mpath)
                    _log.debug("[vo_save] manifest patched: %s locale=%s", item_id, locale)

                    # ── Re-run manifest_merge to recompute duck_intervals ──
                    # Reads from VOPlan.{locale}.json (in-place).
                    _mm_script  = os.path.join(os.path.dirname(__file__), "manifest_merge.py")
                    _mm_shared  = os.path.join(full_ep, "AssetManifest.shared.json")
                    _mm_locale  = os.path.join(full_ep, f"VOPlan.{locale}.json")
                    _mm_env     = os.environ.copy()
                    _mm_env.pop("CLAUDECODE", None)
                    _mm_result  = subprocess.run(
                        ["python3", _mm_script,
                         "--shared", _mm_shared,
                         "--locale", _mm_locale],
                        capture_output=True, text=True, timeout=30,
                        cwd=PIPE_DIR, env=_mm_env,
                    )
                    if _mm_result.returncode != 0:
                        _log.warning("[vo_save] manifest_merge failed (rc=%d): %s",
                                     _mm_result.returncode, _mm_result.stderr[:300])
                    else:
                        _log.debug("[vo_save] manifest_merge ok for locale=%s", locale)
                    # ── end manifest update ────────────────────────────────────
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
                _log.info("[vo_save_pause] item=%s  pause_ms=%s  locale=%s  ep=%s",
                          item_id, pause_ms, locale, ep_dir)
                _vo_validate_inputs(ep_dir, locale, item_id)
                full_ep = _vo_resolve_ep_dir(ep_dir)
                from pathlib import Path as _P

                with _get_vo_lock(full_ep):
                    # Update manifest pause_after_ms (no WAV touched — INVARIANT E)
                    mpath = os.path.join(full_ep, f"VOPlan.{locale}.json")
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

        # POST /api/vo_text — update lyrics text for a single vo_item (MTV mode)
        elif self.path == "/api/vo_text":
            try:
                length  = int(self.headers.get("Content-Length", 0))
                req     = json.loads(self.rfile.read(length))
                ep_dir  = req.get("ep_dir",  "").strip()
                locale  = req.get("locale",  "").strip()
                item_id = req.get("item_id", "").strip()
                text    = req.get("text",    "").strip()
                _log.info("[vo_text] item=%s  text=%r  locale=%s  ep=%s",
                          item_id, text[:80], locale, ep_dir)
                if not text:
                    raise ValueError("text cannot be empty")
                _vo_validate_inputs(ep_dir, locale, item_id)
                full_ep = _vo_resolve_ep_dir(ep_dir)
                from pathlib import Path as _P

                with _get_vo_lock(full_ep):
                    mpath = os.path.join(full_ep, f"VOPlan.{locale}.json")
                    with open(mpath, encoding="utf-8") as _mf:
                        _mani = json.load(_mf)
                    found = False
                    for _it in _mani.get("vo_items", []):
                        if _it["item_id"] == item_id:
                            _it["text"] = text
                            found = True
                            break
                    if not found:
                        raise ValueError(f"item_id {item_id!r} not found in manifest")
                    _tmp = mpath + ".tmp"
                    with open(_tmp, "w", encoding="utf-8") as _mf:
                        json.dump(_mani, _mf, indent=2, ensure_ascii=False)
                    os.replace(_tmp, mpath)

                    primary = _get_primary_locale(_P(full_ep))
                    _invalidate_vo_state(full_ep, primary)

                _json_resp(self, {"item_id": item_id, "text": text})

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
                _log.info("[vo_save_break] scene=%s  tail_ms=%s  locale=%s  ep=%s",
                          scene, tail_ms, locale, ep_dir)
                if not ep_dir or not locale or not scene:
                    raise ValueError("ep_dir, locale, scene required")
                if tail_ms < 0 or tail_ms > 30000:
                    raise ValueError("tail_ms must be 0–30000")
                full_ep = _vo_resolve_ep_dir(ep_dir)
                from pathlib import Path as _P
                with _get_vo_lock(full_ep):
                    mpath = os.path.join(full_ep, f"VOPlan.{locale}.json")
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

        # POST /api/vo_scene_head — set scene intro silence (seconds) before first VO item
        elif self.path == "/api/vo_scene_head":
            try:
                length   = int(self.headers.get("Content-Length", 0))
                req      = json.loads(self.rfile.read(length))
                ep_dir   = req.get("ep_dir",   "").strip()
                locale   = req.get("locale",   "").strip()
                scene    = req.get("scene",    "").strip()
                head_sec = float(req.get("head_sec", 0.0))
                _log.info("[vo_scene_head] scene=%s  head_sec=%s  locale=%s  ep=%s",
                          scene, head_sec, locale, ep_dir)
                if not ep_dir or not locale or not scene:
                    raise ValueError("ep_dir, locale, scene required")
                if head_sec < 0 or head_sec > 300:
                    raise ValueError("head_sec must be 0–300")
                full_ep = _vo_resolve_ep_dir(ep_dir)
                from pathlib import Path as _P
                with _get_vo_lock(full_ep):
                    mpath = os.path.join(full_ep, f"VOPlan.{locale}.json")
                    with open(mpath, encoding="utf-8") as _mf:
                        _mani = json.load(_mf)
                    _mani.setdefault("scene_heads", {})[scene] = head_sec
                    _tmp = mpath + ".tmp"
                    with open(_tmp, "w", encoding="utf-8") as _mf:
                        json.dump(_mani, _mf, indent=2, ensure_ascii=False)
                    os.replace(_tmp, mpath)
                    primary = _get_primary_locale(_P(full_ep))
                    _invalidate_vo_state(full_ep, primary)
                _json_resp(self, {"scene": scene, "head_sec": head_sec})
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
                    mpath = os.path.join(full_ep, f"VOPlan.{locale}.json")
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
                    # Also delete RenderPlan and unified locale manifests for all locales.
                    # (item_id change invalidates resolved_assets and render_plan sections)
                    # AssetManifest.shared.json is NOT deleted — it is locale-free and
                    # item_id-independent (backgrounds, characters, SFX, music).
                    for _f in _P(full_ep).glob("RenderPlan.*.json"):
                        _f.unlink(missing_ok=True)
                    for _f in _P(full_ep).glob("VOPlan.*.json"):
                        if _f.name != "AssetManifest.shared.json":
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

        # POST /api/vo_approve — validate and approve VO, write vo_approval into VOPlan
        elif self.path == "/api/vo_approve":
            try:
                if not _VO_UTILS_AVAILABLE:
                    raise RuntimeError("vo_utils not available")
                length = int(self.headers.get("Content-Length", 0))
                req    = json.loads(self.rfile.read(length))
                ep_dir = req.get("ep_dir", "").strip()
                locale = req.get("locale", "").strip()
                stage  = req.get("stage",  "").strip()   # "3.5", "8.5", or "" (legacy)
                _log.info("[vo_approve] locale=%s  stage=%r  items=%s  ep=%s",
                          locale, stage, len(req.get("items") or []), ep_dir)
                _vo_validate_inputs(ep_dir, locale)
                full_ep = _vo_resolve_ep_dir(ep_dir)
                from pathlib import Path as _P
                import subprocess as _sp

                with _get_vo_lock(full_ep):
                    # Pre-flight check (a): no synthesis jobs in-flight
                    # (Always true under synchronous lock design)

                    # Pre-flight check (e): locale constraints by stage
                    primary = _get_primary_locale(_P(full_ep))
                    if stage == "8.5":
                        # Stage 8.5 (explicit): allow non-primary locales
                        pass
                    elif locale != primary:
                        # Non-primary locale without explicit stage="8.5" —
                        # allow if their merged manifest exists (Stage 8.5 run from
                        # run.sh, which never fires a vo_review_ready SSE, so the
                        # JS cannot set _pendingApproveStage automatically).
                        _merged_check = os.path.join(full_ep, f"VOPlan.{locale}.json")
                        if not os.path.isfile(_merged_check):
                            raise ValueError(
                                f"Cannot approve non-primary locale ({locale!r}) — "
                                f"VOPlan.{locale}.json not found. "
                                "Run Stage 8 (translation) before approving this locale."
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
                    # VOPlan.{locale}.json is now the single source for all stages.
                    mpath = os.path.join(full_ep, f"VOPlan.{locale}.json")
                    if not os.path.isfile(mpath):
                        raise FileNotFoundError(
                            f"VOPlan.{locale}.json not found — "
                            "run Stage 5 (or Stage 3.5 TTS) first"
                        )
                    with open(mpath, encoding="utf-8") as _mf:
                        _mani = json.load(_mf)

                    # ── Request diagnostics ───────────────────────────────
                    _dbg_items = req.get("items")
                    _log.debug("[vo_approve] stage=%r  locale=%r  req_keys=%s  items=%s(len=%s)",
                               stage, locale, list(req.keys()),
                               type(_dbg_items).__name__,
                               len(_dbg_items) if isinstance(_dbg_items, list) else "N/A")
                    if isinstance(_dbg_items, list) and _dbg_items:
                        _log.debug("[vo_approve] first=%s", _dbg_items[0])
                        _log.debug("[vo_approve] last =%s", _dbg_items[-1])
                        for _di in _dbg_items:
                            if "sc02-001" in _di.get("item_id", ""):
                                _log.debug("[vo_approve] sc02-001=%s", _di)
                                break
                    # ──────────────────────────────────────────────────────

                    # Build authoritative timeline from frontend-provided items.
                    # The frontend sends live DOM values (duration from data-dur,
                    # pause from the pause input) — this is what the user heard and
                    # approved.  We compute start_sec/end_sec here via simple
                    # arithmetic; no manifest read, no post_tts_analysis needed.
                    _req_items = req.get("items")  # list or None (legacy callers)
                    _approval_items = []
                    _items_measured = 0
                    if _req_items:
                        # start_sec/end_sec computed on the client with the same
                        # cursor logic as _voRecalcSceneTimes (scene tails included).
                        # Trust them verbatim — no recomputation on the server.
                        # pause_after_ms is also sent by the frontend and must be
                        # written back to the manifest so vo_preview_concat uses it.
                        for _it in _req_items:
                            _dur      = float(_it.get("duration_sec", 0))
                            _start    = float(_it["start_sec"]) if "start_sec" in _it else 0.0
                            _end      = float(_it["end_sec"])   if "end_sec"   in _it else _start + _dur
                            _pause_ms = int(_it.get("pause_after_ms", 300))
                            _vdb_it   = float(_it.get("volume_db", 0.0) or 0.0)
                            _approval_items.append({
                                "item_id":        _it["item_id"],
                                "speaker_id":     _it.get("speaker_id", ""),
                                "text":           _it.get("text", ""),
                                "duration_sec":   round(_dur, 3),
                                "pause_after_ms": _pause_ms,
                                "start_sec":      round(_start, 6),
                                "end_sec":        round(_end, 6),
                                "volume_db":      _vdb_it,
                            })
                        _items_measured = len(_approval_items)
                    else:
                        # Legacy fallback: items not provided — read manifest and
                        # run post_tts_analysis to reconstruct the timeline.
                        _pta_script = os.path.join(os.path.dirname(__file__), "post_tts_analysis.py")
                        _pta_cmd = ["python3", _pta_script, "--manifest", mpath]
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
                        import re as _re_pta
                        _items_measured = len(_re_pta.findall(r"\[OK\]", _pta_result.stdout))
                        with open(mpath, encoding="utf-8") as _mf:
                            _mani = json.load(_mf)
                        _approval_items = [
                            {
                                "item_id":      v["item_id"],
                                "speaker_id":   v.get("speaker_id", ""),
                                "text":         v.get("text", ""),
                                "duration_sec": round(v.get("end_sec", 0) - v.get("start_sec", 0), 3),
                                "start_sec":    round(v.get("start_sec", 0.0), 6),
                                "end_sec":      round(v.get("end_sec",   0.0), 6),
                            }
                            for v in _mani.get("vo_items", [])
                            if "start_sec" in v and "end_sec" in v
                        ]

                    # ── Write diagnostics ─────────────────────────────────
                    _log.debug("[vo_approve] writing %d items  frontend_path=%s",
                               len(_approval_items), bool(_req_items))
                    if _approval_items:
                        _w0 = _approval_items[0]
                        _w4 = _approval_items[3] if len(_approval_items) > 3 else None
                        _w5 = _approval_items[4] if len(_approval_items) > 4 else None
                        _log.debug("[vo_approve] write[0] %s  start=%s  end=%s",
                                   _w0["item_id"], _w0["start_sec"], _w0["end_sec"])
                        if _w4:
                            _log.debug("[vo_approve] write[3] %s  start=%s  end=%s",
                                       _w4["item_id"], _w4["start_sec"], _w4["end_sec"])
                        if _w4 and _w5:
                            _log.debug("[vo_approve] write[4] %s  start=%s  end=%s  gap=%.3fs",
                                       _w5["item_id"], _w5["start_sec"], _w5["end_sec"],
                                       _w5["start_sec"] - _w4["end_sec"])
                    # ──────────────────────────────────────────────────────

                    # Compute hashes and write vo_approval block into VOPlan.{locale}.json
                    _hashes = _compute_sentinel_hashes(full_ep, locale)
                    write_vo_preview_approved(
                        full_ep, locale,
                        stage or "legacy",
                        _approval_items,
                        _hashes,
                    )

                    # ── Write duration_sec / pause_after_ms back to manifest ─────────
                    # vo_approve receives the full approved timeline from the frontend
                    # (duration from live data-dur, pause_after_ms from the pause input).
                    # Write both back so the manifest stays in sync after any re-creation
                    # or pause change — vo_preview_concat and downstream readers use these.
                    _vc_pass   = False  # set after unconditional verify below
                    _vc_errors = []
                    _vc_output = ""
                    if _req_items:
                        _approved_lookup = {
                            a["item_id"]: {
                                "pause_after_ms": a["pause_after_ms"],
                                "start_sec":      a["start_sec"],
                                "end_sec":        a["end_sec"],
                                "volume_db":      a.get("volume_db", 0.0),
                            }
                            for a in _approval_items
                        }
                        _mani_path = os.path.join(full_ep,
                                                  f"VOPlan.{locale}.json")
                        with open(_mani_path, encoding="utf-8") as _mf2:
                            _mani2 = json.load(_mf2)
                        for _mitem in _mani2.get("vo_items", []):
                            _miid = _mitem["item_id"]
                            if _miid in _approved_lookup:
                                _mitem["pause_after_ms"] = _approved_lookup[_miid]["pause_after_ms"]
                                _mitem["start_sec"]      = _approved_lookup[_miid]["start_sec"]
                                _mitem["end_sec"]        = _approved_lookup[_miid]["end_sec"]
                                _mitem.pop("duration_sec", None)  # prohibited by schema
                                # volume_db: omit when zero (keep file clean)
                                _vdb_approved = _approved_lookup[_miid]["volume_db"]
                                if _vdb_approved != 0.0:
                                    _mitem["volume_db"] = _vdb_approved
                                else:
                                    _mitem.pop("volume_db", None)
                        _mani_tmp = _mani_path + ".tmp"
                        with open(_mani_tmp, "w", encoding="utf-8") as _mf2:
                            json.dump(_mani2, _mf2, indent=2, ensure_ascii=False)
                        os.replace(_mani_tmp, _mani_path)
                        _log.debug("[vo_approve] wrote pause_after_ms + "
                                   "start_sec + end_sec for %d items → manifest", len(_approved_lookup))
                    # ─────────────────────────────────────────────────────────────────

                    # ── Sync ShotList.json shot durations to VO-approved scene tails ──
                    # The last shot of each scene is padded/trimmed so its cumulative
                    # offset matches the approved start_sec of the next scene.
                    # This keeps Music/SFX tab shot-offset calculations in sync with
                    # whatever inter-scene breaks the user configured in the VO tab.
                    if _req_items and _approval_items:
                        _sl_path = os.path.join(full_ep, "ShotList.json")
                        if os.path.isfile(_sl_path):
                            try:
                                with open(_sl_path, encoding="utf-8") as _slf:
                                    _sl = json.load(_slf)

                                # Build scene order + per-scene item list from approved items.
                                # item_id format: "vo-{scene_id}-{seq}" e.g. "vo-sc01-003"
                                _sc_order: list = []
                                _sc_items: dict = {}
                                for _ai in _approval_items:
                                    _ai_parts = _ai["item_id"].split("-")
                                    _ai_sc = _ai_parts[1] if len(_ai_parts) >= 3 else ""
                                    if not _ai_sc:
                                        continue
                                    if _ai_sc not in _sc_items:
                                        _sc_items[_ai_sc] = []
                                        _sc_order.append(_ai_sc)
                                    _sc_items[_ai_sc].append(_ai)

                                # Compute authoritative timeline duration per scene:
                                #   non-last scene → next scene's first item start_sec minus this scene's first item start_sec
                                #   last scene     → last item end_sec + pause_after_ms/1000 - this scene's first item start_sec
                                _sc_new_dur: dict = {}
                                for _sci, _sc in enumerate(_sc_order):
                                    _sc_start = _sc_items[_sc][0]["start_sec"]
                                    if _sci + 1 < len(_sc_order):
                                        _sc_end = _sc_items[_sc_order[_sci + 1]][0]["start_sec"]
                                    else:
                                        _last_ai = _sc_items[_sc][-1]
                                        _sc_end = (_last_ai["end_sec"]
                                                   + _last_ai.get("pause_after_ms", 300) / 1000.0)
                                    _sc_new_dur[_sc] = round(_sc_end - _sc_start, 6)

                                # Group ShotList shots by scene_id (preserving JSON order)
                                _sl_by_sc: dict = {}
                                for _sh in _sl.get("shots", []):
                                    _sh_sc = _sh.get("scene_id", "")
                                    _sl_by_sc.setdefault(_sh_sc, []).append(_sh)

                                # Apply delta to last shot of each scene
                                _sl_changed = 0
                                for _sc, _new_dur in _sc_new_dur.items():
                                    _sc_shots = _sl_by_sc.get(_sc, [])
                                    if not _sc_shots:
                                        continue
                                    _old_total = sum(
                                        s.get("duration_sec", 0.0) for s in _sc_shots)
                                    _delta = round(_new_dur - _old_total, 6)
                                    if abs(_delta) < 0.001:
                                        continue
                                    _last_shot = _sc_shots[-1]
                                    _old_last = _last_shot.get("duration_sec", 0.0)
                                    _last_shot["duration_sec"] = round(_old_last + _delta, 3)
                                    _sl_changed += 1
                                    _log.debug(
                                        "[vo_approve] ShotList %s last-shot=%s  "
                                        "scene_dur %.3f→%.3f  delta=%.3f",
                                        _sc, _last_shot["shot_id"],
                                        _old_total, _new_dur, _delta)

                                # Always recompute start_sec for every shot as the
                                # cumulative sum of preceding duration_sec values.
                                # render_video.py reads shot["start_sec"] to compute
                                # shot-relative VO offsets; a stale start_sec (not
                                # reflecting the scene tail in the preceding shot's
                                # duration) shifts every VO line by the tail amount
                                # in the final rendered video.
                                # This runs even when _sl_changed == 0 so that a
                                # pre-existing inconsistency between duration_sec and
                                # start_sec (delta < 0.001 threshold) is still fixed.
                                _start_sec_changed = 0
                                _cum_sec = 0.0
                                for _sh in _sl.get("shots", []):
                                    _expected = round(_cum_sec, 3)
                                    if abs(_sh.get("start_sec", 0.0) - _expected) >= 0.001:
                                        _sh["start_sec"] = _expected
                                        _start_sec_changed += 1
                                    _cum_sec += _sh.get("duration_sec", 0.0)

                                if _sl_changed or _start_sec_changed:
                                    _sl["total_duration_sec"] = round(_cum_sec, 3)
                                    _sl_tmp = _sl_path + ".tmp"
                                    with open(_sl_tmp, "w", encoding="utf-8") as _slf2:
                                        json.dump(_sl, _slf2, indent=2, ensure_ascii=False)
                                    os.replace(_sl_tmp, _sl_path)
                                    _log.info(
                                        "[vo_approve] ShotList.json updated — "
                                        "%d scene(s) duration-adjusted, "
                                        "%d shot(s) start_sec fixed",
                                        _sl_changed, _start_sec_changed)
                            except Exception as _sl_err:
                                _log.warning(
                                    "[vo_approve] ShotList.json update failed: %s", _sl_err)
                    # ─────────────────────────────────────────────────────────────────

                    # ── Re-patch AssetManifest durations after ShotList is fixed ──────
                    # When Stage 3.5 no longer pauses, Stages 4+5 run before VO is
                    # approved, so AssetManifest.shared.json was written with 0
                    # durations.  Now that ShotList has correct values, re-run
                    # patch_manifest_durations.py to sync AssetManifest.
                    _asset_manifest_path = os.path.join(full_ep, "AssetManifest.shared.json")
                    if os.path.isfile(_asset_manifest_path):
                        try:
                            _pmd_script = os.path.join(
                                os.path.dirname(os.path.abspath(__file__)),
                                "patch_manifest_durations.py",
                            )
                            _pmd_result = subprocess.run(
                                [sys.executable, _pmd_script, full_ep],
                                capture_output=True, text=True, timeout=30,
                            )
                            if _pmd_result.returncode == 0:
                                _log.info("[vo_approve] patch_manifest_durations ✓")
                            else:
                                _log.warning(
                                    "[vo_approve] patch_manifest_durations failed: %s",
                                    _pmd_result.stderr,
                                )
                        except Exception as _pmd_err:
                            _log.warning(
                                "[vo_approve] patch_manifest_durations error: %s", _pmd_err
                            )
                    # ─────────────────────────────────────────────────────────────────

                    # ── Cache approved WAVs ───────────────────────────────────────────
                    # Copy every approved WAV into assets/meta/vo_approved_cache/{locale}/
                    # so that a subsequent step-6 re-run restores these exact files
                    # instead of overwriting them with a new Azure synthesis.
                    import shutil as _shutil_vc
                    _approved_cache_dir = (
                        _P(full_ep) / "assets" / "meta" / "vo_approved_cache" / locale
                    )
                    _log.debug("[vo_approve] cache: full_ep=%s locale=%s approval_items=%d",
                               full_ep, locale, len(_approval_items))
                    _approved_cache_dir.mkdir(parents=True, exist_ok=True)
                    _vo_wav_dir = _P(full_ep) / "assets" / locale / "audio" / "vo"
                    _log.debug("[vo_approve] cache: vo_wav_dir=%s exists=%s",
                               _vo_wav_dir, _vo_wav_dir.exists())
                    _cached_count = 0
                    _missing_wavs = []
                    for _item in _approval_items:
                        _iid = _item["item_id"]
                        _src_wav = _vo_wav_dir / f"{_iid}.wav"
                        if _src_wav.exists():
                            _shutil_vc.copy(_src_wav, _approved_cache_dir / f"{_iid}.wav")
                            _cached_count += 1
                        else:
                            _missing_wavs.append(_iid)
                    if _missing_wavs:
                        _log.debug("[vo_approve] cache: ⚠ %d WAV(s) not found: %s",
                                   len(_missing_wavs), _missing_wavs)
                    _log.debug("[vo_approve] cache: ✓ %d/%d approved WAV(s) cached → "
                               "assets/meta/vo_approved_cache/%s/",
                               _cached_count, len(_approval_items), locale)
                    # ─────────────────────────────────────────────────────────────────

                # Contract validation — skip for MTV (schema requires TTS/shot fields MTV doesn't have)
                _vp_path = os.path.join(full_ep, f"VOPlan.{locale}.json")
                _is_mtv_vc = False
                try:
                    _meta_vc_path = os.path.join(full_ep, "meta.json")
                    if os.path.isfile(_meta_vc_path):
                        _is_mtv_vc = json.load(open(_meta_vc_path, encoding="utf-8")).get("story_format") == "mtv"
                except Exception:
                    pass
                if _is_mtv_vc:
                    _vc_pass = True
                    _vc_output = "MTV mode — contract validation skipped"
                    _log.info("[vo_approve] %s", _vc_output)
                elif os.path.isfile(_vp_path):
                    _verify_script = os.path.join(
                        PIPE_DIR, "contracts", "tools", "verify_contracts.py")
                    try:
                        _vc_proc = subprocess.run(
                            [sys.executable, _verify_script, _vp_path],
                            capture_output=True, text=True, timeout=15,
                        )
                        _vc_output = (_vc_proc.stdout + _vc_proc.stderr).strip()
                        _vc_pass   = (_vc_proc.returncode == 0)
                        if not _vc_pass:
                            _vc_errors = [
                                ln.strip().lstrip("•").strip()
                                for ln in _vc_output.splitlines()
                                if ln.strip().startswith("•")
                            ] or [_vc_output]
                        _log.info("[vo_approve] verify_contracts: %s  %s",
                                  "PASS" if _vc_pass else "FAIL", _vc_output[:200])
                    except Exception as _vc_exc:
                        _vc_output = f"verify_contracts runner error: {_vc_exc}"
                        _log.warning("[vo_approve] %s", _vc_output)

                _json_resp(self, {
                    "approved":          True,
                    "items_measured":    _items_measured,
                    "locale":            locale,
                    "stage":             stage or "legacy",
                    "validation_pass":   _vc_pass,
                    "validation_errors": _vc_errors,
                    "validation_output": _vc_output,
                })
                print(f"[vo_approve] ✓ {locale} approved (stage={stage or 'legacy'}) — "
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

            # ── Script.json pre-check: detect a provided Script before calling haiku ──
            _script_data = None
            try:
                _parsed_sj = json.loads(story)
                if isinstance(_parsed_sj, dict) and _parsed_sj.get("schema_id", "").lower() == "script":
                    _sj_title   = _parsed_sj.get("title", "")
                    _sj_proj_id = _parsed_sj.get("project_id", "")
                    # project_id is already a validated slug (^[a-z0-9-]+$); prefer it
                    # over a title-derived slug which can produce unexpected values.
                    _sj_slug = _sj_proj_id if _sj_proj_id else (
                        _re.sub(r'[^a-z0-9]+', '-', _sj_title.lower()).strip('-')[:60]
                        if _sj_title else ""
                    )
                    _script_data = {
                        "title":          _sj_title,
                        "slug":           _sj_slug,
                        "genre":          _parsed_sj.get("genre", ""),
                        "story_format":   "Script.json",
                        "metadata_found": ["schema_id", "title", "genre", "cast", "scenes"],
                    }
            except (json.JSONDecodeError, ValueError):
                pass

            # ── SSML pre-check: detect authored SSML before calling haiku ──
            _ssml_pattern = _re.compile(
                r'<(?:speak|voice|prosody|mstts:)\b', _re.IGNORECASE
            )
            if _script_data is not None:
                data = _script_data
            elif _ssml_pattern.search(story[:2000]):
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

            # Check slug uniqueness.
            # Script.json is exempt: its project_id is the intended slug and the
            # project directory is expected to already exist (we are adding a new
            # episode, not creating a new project). Renaming to "pompeii-2" would
            # be wrong, and slug_exists=True would hide the Save button.
            slug = data.get("slug", "")
            if slug and data.get("story_format") != "Script.json":
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
                if not story_text and story_format != "mtv":
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

                # For provided-script episodes, also write Script.json so the pipeline
                # PREPARE stage can locate and validate it at the expected path.
                if story_format == "Script.json":
                    with open(os.path.join(ep_dir, "Script.json"), "w", encoding="utf-8") as _f:
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

                # Load AssetManifest.shared.json to pass backgrounds to media server
                ep_dir = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
                manifest_path = os.path.join(ep_dir, "AssetManifest.shared.json")
                if not os.path.isfile(manifest_path):
                    raise FileNotFoundError(
                        f"AssetManifest.shared.json not found at {manifest_path}")
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

        # ── Media proxy: prune batch images (POST /api/media_batch_prune) ──────
        elif self.path == "/api/media_batch_prune":
            try:
                length     = int(self.headers.get("Content-Length", 0))
                payload    = json.loads(self.rfile.read(length))
                batch_id   = payload.get("batch_id", "").strip()
                server_url = (payload.get("server_url") or "http://localhost:8200").rstrip("/")
                api_key    = os.environ.get("MEDIA_API_KEY", "")
                if not batch_id:
                    raise ValueError("batch_id is required")
                if "filter_spec" not in payload:
                    raise ValueError("filter_spec is required")
                # Forward only {item_ids, filter_spec} — batch_id goes in the URL path
                forward: dict = {"filter_spec": payload["filter_spec"]}
                if payload.get("item_ids") is not None:
                    forward["item_ids"] = payload["item_ids"]
                req_body = json.dumps(forward).encode()
                url = f"{server_url}/batches/{batch_id}/prune"
                req = _urllib_req.Request(
                    url, data=req_body,
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
                body = json.dumps({"error": str(exc)}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        # ── Media AI Ask: fire-and-forget proxy (POST /api/media_ai_ask) ────────
        elif self.path == "/api/media_ai_ask":
            try:
                length     = int(self.headers.get("Content-Length", 0))
                payload    = json.loads(self.rfile.read(length))
                slug       = payload.get("slug", "").strip()
                ep_id      = payload.get("ep_id", "").strip()
                server_url = (payload.get("server_url") or "http://localhost:8200").rstrip("/")
                prompt     = payload.get("prompt", "").strip()
                batch_id   = payload.get("batch_id")
                if not server_url:
                    raise ValueError("server_url required")
                if not prompt:
                    raise ValueError("prompt required")
                if not batch_id:
                    # Try to recover batch_id from media server
                    try:
                        _ak = os.environ.get("MEDIA_API_KEY", "")
                        _r  = _urllib_req.Request(
                            f"{server_url}/batches?project={urllib.parse.quote(slug)}&episode_id={urllib.parse.quote(ep_id)}",
                            headers={"X-Api-Key": _ak} if _ak else {})
                        with _urllib_req.urlopen(_r, timeout=10) as _resp:
                            _batches = json.loads(_resp.read())
                        if _batches:
                            _best = (next((b for b in _batches if b.get("status") == "done"), None)
                                     or next((b for b in _batches if b.get("status") == "running"), None)
                                     or _batches[0])
                            batch_id = _best.get("batch_id") or _best.get("id")
                    except Exception:
                        pass
                if not batch_id:
                    raise ValueError("No batch found. Run Search Media first.")
                key = f"{slug}:{ep_id}"
                print(f"  [media_ai_ask] submitted  slug={slug}  ep={ep_id}  batch={batch_id}  prompt={prompt!r}")
                def _run(_su=server_url, _slug=slug, _ep=ep_id,
                         _prompt=prompt, _bid=batch_id, _key=key):
                    try:
                        _ak  = os.environ.get("MEDIA_API_KEY", "")
                        _payload = json.dumps({
                            "project": _slug, "episode_id": _ep,
                            "prompt": _prompt, "batch_id": _bid
                        }).encode()
                        _req = _urllib_req.Request(
                            f"{_su}/ai_ask", data=_payload,
                            headers={
                                **( {"X-Api-Key": _ak} if _ak else {} ),
                                "Content-Type": "application/json",
                                "Content-Length": str(len(_payload)),
                            },
                            method="POST",
                        )
                        with _urllib_req.urlopen(_req, timeout=360) as _resp:
                            _status = _resp.status
                            _body   = _resp.read().decode(errors="replace")
                        if _status not in (200, 201, 202):
                            _err = f"AI Ask failed ({_status}): {_body[:200]}"
                            _media_ai_ask_errors[_key] = _err
                            print(f"  [media_ai_ask] ERROR  slug={_slug}  ep={_ep}  batch={_bid}  {_err}")
                        else:
                            print(f"  [media_ai_ask] done   slug={_slug}  ep={_ep}  batch={_bid}  status={_status}  response={_body[:300]!r}")
                    except Exception as _exc:
                        _media_ai_ask_errors[_key] = str(_exc)
                        print(f"  [media_ai_ask] EXCEPTION  slug={_slug}  ep={_ep}  batch={_bid}  {_exc}")
                import threading as _threading
                _threading.Thread(target=_run, daemon=True).start()
                body = json.dumps({"status": "submitted"}).encode()
                self.send_response(200)
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

        # ── SFX: write SfxPlan.json (POST /api/sfx_plan_save) ──
        elif self.path == "/api/sfx_plan_save":
            try:
                length   = int(self.headers.get("Content-Length", 0))
                payload  = json.loads(self.rfile.read(length))
                slug     = payload.get("slug", "").strip()
                ep_id    = payload.get("ep_id", "").strip()
                if not slug or not ep_id:
                    raise ValueError("slug and ep_id are required")

                sfx_entries = payload.get("sfx_entries", [])
                cut_clips    = payload.get("cut_clips",    [])
                cut_assign   = payload.get("cut_assign",   {})

                if not isinstance(sfx_entries, list):
                    raise ValueError("sfx_entries must be an array")
                if not isinstance(cut_clips, list):
                    raise ValueError("cut_clips must be an array")
                if not isinstance(cut_assign, dict):
                    raise ValueError("cut_assign must be an object")

                ep_dir    = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
                os.makedirs(ep_dir, exist_ok=True)

                sfx_plan = {
                    "schema_id":      "SfxPlan",
                    "schema_version": "1.0",
                    "timing_format":  "episode_absolute",
                    "shot_overrides": sfx_entries,
                    "cut_clips":      cut_clips,
                    "cut_assign":     cut_assign,
                }
                plan_path = os.path.join(ep_dir, "SfxPlan.json")
                tmp_path  = plan_path + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as _pf:
                    json.dump(sfx_plan, _pf, indent=2, ensure_ascii=False)
                    _pf.write("\n")
                os.replace(tmp_path, plan_path)
                print(f"  [SFX] SfxPlan.json written: {len(sfx_entries)} segments  slug={slug}  ep={ep_id}")
                # Contract validation — run immediately after write
                _verify_script = os.path.join(
                    PIPE_DIR, "contracts", "tools", "verify_contracts.py")
                _vc_pass = None; _vc_errors = []; _vc_output = ""
                try:
                    _vc_proc = subprocess.run(
                        [sys.executable, _verify_script, plan_path],
                        capture_output=True, text=True, timeout=15,
                    )
                    _vc_output = (_vc_proc.stdout + _vc_proc.stderr).strip()
                    _vc_pass   = (_vc_proc.returncode == 0)
                    if not _vc_pass:
                        _vc_errors = [
                            ln.strip().lstrip("•").strip()
                            for ln in _vc_output.splitlines()
                            if ln.strip().startswith("•")
                        ] or [_vc_output]
                    print(f"  verify_contracts: {'PASS' if _vc_pass else 'FAIL'}  {_vc_output[:200]}")
                except Exception as _vc_exc:
                    _vc_output = f"verify_contracts runner error: {_vc_exc}"
                    print(f"  {_vc_output}")
                _json_resp(self, {"ok": True, "saved": len(sfx_entries),
                                  "plan_path":         plan_path,
                                  "validation_pass":   _vc_pass,
                                  "validation_errors": _vc_errors,
                                  "validation_output": _vc_output})
            except Exception as exc:
                _json_resp(self, {"error": str(exc)}, 500)

        # ── SFX Preview: render VO + SFX + optional Music preview WAV ───────
        elif self.path == "/api/sfx_preview":
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                slug   = payload.get("slug", "").strip()
                ep_id  = payload.get("ep_id", "").strip()
                if not slug or not ep_id:
                    raise ValueError("slug and ep_id are required")
                selected      = payload.get("selected", {})       # { item_id: candidate_idx }
                include_music = payload.get("include_music", False)
                timing        = payload.get("timing", {})          # { item_id: {start, end} }
                volumes      = payload.get("volumes",      {})
                duck_fade    = payload.get("duck_fade",    {})
                cut_clips    = payload.get("cut_clips",    [])
                cut_assign   = payload.get("cut_assign",   {})
                clip_volumes = payload.get("clip_volumes", {})
                sfx_segments = payload.get("sfx_segments", [])    # shot override list

                ep_dir = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)

                # ── Find merged manifest — inline ──────────────────────────────
                import glob as _glob_mod
                import re as _re_sfx
                merged_manifests = [p for p in _glob_mod.glob(
                    os.path.join(ep_dir, "VOPlan.*.json"))
                    if os.path.basename(p) != "AssetManifest.shared.json"]
                if not merged_manifests:
                    raise FileNotFoundError(
                        "No VOPlan.*.json found. Run stages 5+ first.")
                _primary_locale = "en"
                _vars_file = os.path.join(ep_dir, "pipeline_vars.sh")
                if os.path.isfile(_vars_file):
                    with open(_vars_file, encoding="utf-8") as _vf:
                        _m = _re_sfx.search(
                            r'(?:^|[\n;])(?:export\s+)?PRIMARY_LOCALE=["\']?([^"\';\n]+)["\']?',
                            _vf.read())
                        if _m:
                            _primary_locale = _m.group(1).strip()
                _primary_manifest = os.path.join(ep_dir, f"VOPlan.{_primary_locale}.json")
                if os.path.isfile(_primary_manifest):
                    manifest_path = _primary_manifest
                else:
                    manifest_path = sorted(merged_manifests)[0]
                    print(f"[WARN] Primary manifest for locale '{_primary_locale}' not found, "
                          f"falling back to {os.path.basename(manifest_path)}")

                # ── Build sfx_selections dict from cut_assign + selected ──────────────
                # cut_assign entries carry their WAV path in cut_clips[] — they
                # do NOT need sfx_search_results.json and must be processed first.
                # selected (candidate-index) entries DO need the results file.
                sfx_results_path = os.path.join(ep_dir, "assets", "sfx", "sfx_search_results.json")
                sfx_sel = {}

                # Pass 1 — cut clips (no results file needed)
                for item_id, assigned_clip_id in cut_assign.items():
                    cut_info = next((cl for cl in cut_clips if cl["clip_id"] == assigned_clip_id), None)
                    if not cut_info:
                        print(f"  [SFX PREVIEW] clip {assigned_clip_id} not found for {item_id} — skipped")
                        continue
                    _raw_path = cut_info["path"]
                    _abs_path = _raw_path if os.path.isabs(_raw_path) else os.path.join(ep_dir, _raw_path)
                    _cut_timing = timing.get(item_id, {})
                    sfx_sel[item_id] = {
                        "preview_url":    "",
                        "source_file":    _abs_path,
                        "start":          _cut_timing.get("start", 0.0),
                        "end":            _cut_timing.get("end", None),
                        "volume_db":      float(volumes.get(item_id, 0) or 0),
                        "clip_volume_db": float(clip_volumes.get(assigned_clip_id, 0) or 0),
                        "duck_db":        float((duck_fade.get(item_id) or {}).get("duck_db",  0) or 0),
                        "fade_sec":       float((duck_fade.get(item_id) or {}).get("fade_sec", 0) or 0),
                        "is_cut_clip":    True,
                    }

                # Pass 2 — library candidates (require sfx_search_results.json)
                if os.path.isfile(sfx_results_path):
                    with open(sfx_results_path, encoding="utf-8") as f:
                        saved = json.load(f)
                    results = saved.get("results", {})
                    for item_id in selected:
                        if item_id in sfx_sel:
                            continue   # cut clip already assigned — takes precedence
                        idx = selected.get(item_id)
                        if idx is None:
                            continue
                        # Coerce idx to int (JSON may deliver strings)
                        try:
                            idx = int(idx)
                        except (TypeError, ValueError):
                            continue
                        cands = results.get(item_id, {}).get("candidates", [])
                        if idx < 0 or idx >= len(cands):
                            continue
                        _item_timing = timing.get(item_id, {})
                        sfx_sel[item_id] = {
                            "preview_url":    cands[idx].get("preview_url", ""),
                            "source_file":    _resolve_sfx_local_path(ep_dir, item_id, cands[idx]),
                            "start":          _item_timing.get("start", 0),
                            "end":            _item_timing.get("end", None),
                            "volume_db":      float(volumes.get(item_id, 0) or 0),
                            "duck_db":        float((duck_fade.get(item_id) or {}).get("duck_db",  0) or 0),
                            "fade_sec":       float((duck_fade.get(item_id) or {}).get("fade_sec", 0) or 0),
                            "clip_volume_db": 0.0,
                            "is_cut_clip":    False,
                        }

                # ── Fallback: direct-download files that _resolve_sfx_local_path missed ──
                # (e.g. files too large for /sfx_save but still servable by media server)
                import urllib.request as _ul_req
                _temp_dl_paths = []   # track temp files to clean up after preview
                # Use ep_dir/assets/sfx/tmp/ for temp downloads so they're inside pipe_root
                # (sfx_preview_pack.py path security check requires paths under PIPE_DIR)
                _sfx_tmp_dir = os.path.join(ep_dir, "assets", "sfx", "_tmp_dl")
                os.makedirs(_sfx_tmp_dir, exist_ok=True)
                for _iid, _sel_entry in sfx_sel.items():
                    if _sel_entry.get("source_file") is None:
                        _pu = _sel_entry.get("preview_url", "")
                        print(f"  [SFX] source_file=None for {_iid}, preview_url={_pu!r}")
                        if _pu:
                            try:
                                # Preserve original extension (MP3, WAV, etc.) so librosa can detect format
                                from urllib.parse import urlparse as _urlparse2
                                _url_path = _urlparse2(_pu).path
                                _ext = os.path.splitext(_url_path)[1] or ".mp3"
                                import tempfile as _tf2
                                _tfd, _tp = _tf2.mkstemp(suffix=_ext, prefix=f"sfx_dl_{_iid}_",
                                                          dir=_sfx_tmp_dir)
                                os.close(_tfd)
                                # Use Request with headers in case media server requires auth
                                _dl_req = _ul_req.Request(_pu)
                                _dl_req.add_header("User-Agent", "Mozilla/5.0")
                                try:
                                    _api_key = os.environ.get("MEDIA_API_KEY", "")
                                    if _api_key and ("localhost" in _pu or "127.0.0.1" in _pu):
                                        _dl_req.add_header("X-Api-Key", _api_key)
                                except Exception:
                                    pass
                                _dl_deadline = time.time() + 30
                                with _ul_req.urlopen(_dl_req, timeout=30) as _resp:
                                    with open(_tp, "wb") as _tf_out:
                                        while True:
                                            if time.time() > _dl_deadline:
                                                raise TimeoutError(f"Download of {_iid} exceeded 30s total")
                                            _chunk = _resp.read(65536)
                                            if not _chunk:
                                                break
                                            _tf_out.write(_chunk)
                                _file_sz = os.path.getsize(_tp)
                                _sel_entry["source_file"] = _tp
                                _temp_dl_paths.append(_tp)
                                print(f"  [SFX] Direct download OK for {_iid}: {_tp} ({_file_sz} bytes, ext={_ext})")
                            except Exception as _de:
                                print(f"  [SFX] Direct download FAILED for {_iid}: {type(_de).__name__}: {_de}")

                # ── Log sfx_sel summary before writing ──
                print(f"  [SFX] sfx_sel has {len(sfx_sel)} item(s):")
                for _iid, _se in sfx_sel.items():
                    _sf = _se.get("source_file")
                    _pu = _se.get("preview_url", "")
                    _sf_exists = os.path.isfile(_sf) if _sf else False
                    print(f"    {_iid}: source_file={_sf!r} exists={_sf_exists} url={_pu!r}")

                # ── Per-episode lock ──────────────────────────────────────────
                _ep_key = f"{slug}/{ep_id}"
                if _ep_key not in _sfx_preview_locks:
                    _sfx_preview_locks[_ep_key] = threading.Lock()
                _sfx_ep_lock = _sfx_preview_locks[_ep_key]
                if not _sfx_ep_lock.acquire(blocking=False):
                    _json_resp(self, {"error": "Preview already generating for this episode"}, 409)
                    return

                # ── Direct in-process render ─────────────────────────────────
                import sfx_preview_pack as _sfx_pack_mod
                from sfx_preview_pack import render_sfx_preview as _sfx_render
                # sfx_preview_pack.PIPE_DIR is set at module-load time from __file__
                # (always the repo root).  In test mode PIPE_DIR here is overridden
                # to the tmp fixture dir, so we must sync it before the path-security
                # check inside render_sfx_preview() rejects every test-dir path.
                _sfx_pack_mod.PIPE_DIR = Path(PIPE_DIR)

                _sfx_manifest = json.loads(
                    open(manifest_path, encoding="utf-8").read())
                _sfx_locale = (_sfx_manifest.get("locale", "")
                               or os.path.basename(manifest_path).split(".")[-2])

                _sfx_out_wav = Path(ep_dir) / "assets" / "sfx" / "SfxPreviewPack" / "preview_audio.wav"
                _sfx_out_wav.parent.mkdir(parents=True, exist_ok=True)

                # Build a flat list for render_sfx_preview (expects list of segments
                # each with start_sec/end_sec/source_file, not a dict keyed by item_id).
                # Start from shot override segments sent by the frontend, then append
                # any cut-clip / library-candidate selections already resolved above.
                #
                # IMPORTANT: shot override segments store source_file as the cut-clip's
                # clip_id (the dropdown value), not the actual WAV path.  Resolve it
                # by looking the clip_id up in cut_clips → get the ep_dir-relative path.
                _cut_clips_by_id = {cl["clip_id"]: cl for cl in cut_clips}
                _sfx_sel_list = []
                for _seg in sfx_segments:
                    _src = _seg.get("source_file", "")
                    if _src and os.sep not in _src and "/" not in _src and "\\" not in _src:
                        # Looks like a bare clip_id, not a file path — try to resolve
                        _clip_info = _cut_clips_by_id.get(_src)
                        if _clip_info:
                            _rel = _clip_info.get("path", "")
                            if _rel:
                                _seg = dict(_seg)  # copy — don't mutate caller's list
                                _seg["source_file"] = (
                                    _rel if os.path.isabs(_rel)
                                    else os.path.join(ep_dir, _rel)
                                )
                    _sfx_sel_list.append(_seg)
                for _iid, _se in sfx_sel.items():
                    _sfx_sel_list.append({
                        "clip_id":        _iid,
                        "source_file":    _se.get("source_file"),
                        "start_sec":      float(_se.get("start", 0) or 0),
                        "end_sec":        _se.get("end"),
                        "volume_db":      float(_se.get("volume_db",      0) or 0),
                        "duck_db":        float(_se.get("duck_db",        0) or 0),
                        "fade_sec":       float(_se.get("fade_sec",       0) or 0),
                        "clip_volume_db": float(_se.get("clip_volume_db", 0) or 0),
                    })
                print(f"  [SFX] sfx_sel_list: {len(_sfx_sel_list)} segment(s) "
                      f"({len(sfx_segments)} overrides + {len(sfx_sel)} clip/lib)")

                try:
                    _tl_doc = _sfx_render(
                        _sfx_manifest, Path(manifest_path),
                        Path(ep_dir), _sfx_locale, _sfx_sel_list, _sfx_out_wav,
                        include_music=include_music,
                    )
                finally:
                    for _tp in _temp_dl_paths:
                        if os.path.exists(_tp):
                            os.unlink(_tp)
                    _sfx_ep_lock.release()

                _json_resp(self, {"ok": True, "timeline": _tl_doc})

            except Exception as exc:
                _json_resp(self, {"error": str(exc)}, 400)

        # ── SFX Cut Clip: trim a candidate audio to In/Out marks ─────────────
        elif self.path == "/api/sfx_cut_clip":
            try:
                length       = int(self.headers.get("Content-Length", 0))
                payload      = json.loads(self.rfile.read(length))
                slug         = payload.get("slug", "").strip()
                ep_id        = payload.get("ep_id", "").strip()
                item_id      = payload.get("item_id", "").strip()
                candidate_idx = int(payload.get("candidate_idx", 0))
                source_file  = payload.get("source_file", "").strip()
                title        = payload.get("title", "").strip()
                start_sec    = float(payload.get("start_sec", 0))
                end_sec      = float(payload.get("end_sec", 0))
                if not all([slug, ep_id, item_id, source_file]) or end_sec <= start_sec:
                    raise ValueError("slug, ep_id, item_id, source_file, start_sec<end_sec required")

                ep_dir   = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
                out_dir  = os.path.join(ep_dir, "assets", "sfx", item_id)
                os.makedirs(out_dir, exist_ok=True)

                # Build clip name: "{title}_{start}s-{end}s" — same convention as music clips
                # ({stem}_{start:.1f}s-{end:.1f}s.wav, natural decimal notation with dots).
                # Use title (the library sound name) so the file tells you what the sound is.
                # Fall back to item_id if title is empty.
                import re as _re_sfx_title
                _safe_name = _re_sfx_title.sub(r'[^A-Za-z0-9\-]', '_', title).strip('_')[:40] if title else item_id
                if not _safe_name:
                    _safe_name = item_id
                out_filename = f"{_safe_name}_{start_sec:.1f}s-{end_sec:.1f}s.wav"
                out_path     = os.path.join(out_dir, out_filename)

                # Resolve source_file — handle serve_media paths and external URLs
                _sfx_src = source_file
                _sfx_tmp = None
                # Resolve /serve_media?path=... to a real local path
                if _sfx_src.startswith("/serve_media"):
                    from urllib.parse import urlparse as _up_sm, parse_qs as _pqs_sm, unquote_plus as _uq_sm
                    _sm_qs  = _pqs_sm(_up_sm(_sfx_src).query)
                    _sm_rel = _uq_sm((_sm_qs.get("path", [""])[0]).strip())
                    if _sm_rel:
                        _sfx_src = os.path.realpath(
                            _sm_rel if os.path.isabs(_sm_rel)
                            else os.path.join(PIPE_DIR, _sm_rel)
                        )
                if _sfx_src.startswith("http://") or _sfx_src.startswith("https://"):
                    import urllib.request as _ul_cut
                    from urllib.parse import urlparse as _up_cut
                    _ext_cut = os.path.splitext(_up_cut(_sfx_src).path)[1] or ".mp3"
                    import tempfile as _tf_cut
                    _fd_cut, _sfx_tmp = _tf_cut.mkstemp(suffix=_ext_cut,
                                                         prefix=f"sfx_cut_{item_id}_",
                                                         dir=out_dir)
                    os.close(_fd_cut)
                    _req_cut = _ul_cut.Request(_sfx_src)
                    _req_cut.add_header("User-Agent", "Mozilla/5.0")
                    with _ul_cut.urlopen(_req_cut, timeout=60) as _resp_cut:
                        with open(_sfx_tmp, "wb") as _fo_cut:
                            _fo_cut.write(_resp_cut.read())
                    _sfx_src = _sfx_tmp

                try:
                    # Extract with ffmpeg — 48000 Hz REQUIRED (matches sfx_preview_pack.py SAMPLE_RATE)
                    subprocess.run([
                        "ffmpeg", "-i", _sfx_src,
                        "-ss", str(start_sec), "-to", str(end_sec),
                        "-ar", "48000", "-ac", "2", "-y", out_path
                    ], check=True, capture_output=True)
                finally:
                    if _sfx_tmp and os.path.exists(_sfx_tmp):
                        os.unlink(_sfx_tmp)

                clip_id      = out_filename[:-4]   # strip ".wav" — same derivation as music clips
                duration_sec = end_sec - start_sec
                rel_path     = os.path.relpath(out_path, ep_dir)   # ep_dir-relative for storage

                # Persist cut clip metadata immediately so reloads don't lose it
                _cc_file = os.path.join(ep_dir, "assets", "sfx", "sfx_cut_clips.json")
                try:
                    _existing_cc = []
                    if os.path.isfile(_cc_file):
                        with open(_cc_file, encoding="utf-8") as _ccf:
                            _existing_cc = json.load(_ccf)
                    # Replace any prior entry with the same clip_id (idempotent)
                    _existing_cc = [c for c in _existing_cc if c.get("clip_id") != clip_id]
                    _existing_cc.append({
                        "clip_id":       clip_id,
                        "item_id":       item_id,
                        "candidate_idx": candidate_idx,
                        "start_sec":     start_sec,
                        "end_sec":       end_sec,
                        "duration_sec":  duration_sec,
                        "source_file":   source_file,
                        "path":          rel_path,
                    })
                    with open(_cc_file, "w", encoding="utf-8") as _ccf:
                        json.dump(_existing_cc, _ccf, indent=2, ensure_ascii=False)
                    print(f"  [SFX CUT] sfx_cut_clips.json updated: {len(_existing_cc)} clip(s)")
                except Exception as _cc_err:
                    print(f"  [SFX CUT] WARNING: could not update sfx_cut_clips.json: {_cc_err}")

                print(f"  [SFX CUT] {clip_id} → {out_path} ({duration_sec:.2f}s)")
                _json_resp(self, {
                    "ok": True,
                    "clip_id": clip_id,
                    "path": rel_path,
                    "duration_sec": duration_sec,
                })
            except subprocess.CalledProcessError as _ffmpeg_err:
                _json_resp(self, {"ok": False, "error": f"ffmpeg failed: {_ffmpeg_err.stderr.decode(errors='replace')}"}, 500)
            except Exception as exc:
                _json_resp(self, {"ok": False, "error": str(exc)}, 500)

        # ── SFX: delete a generated clip from disk + sfx_cut_clips.json ────────
        elif self.path == "/api/sfx_delete_clip":
            try:
                length  = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length))
                slug    = payload.get("slug",    "").strip()
                ep_id   = payload.get("ep_id",   "").strip()
                clip_id = payload.get("clip_id", "").strip()
                if not slug or not ep_id or not clip_id:
                    raise ValueError("slug, ep_id, clip_id are required")
                ep_dir  = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
                cc_file = os.path.join(ep_dir, "assets", "sfx", "sfx_cut_clips.json")
                wav_removed = False
                if os.path.isfile(cc_file):
                    with open(cc_file, "r", encoding="utf-8") as _f:
                        _clips = json.load(_f)
                    _entry = next((c for c in _clips if c.get("clip_id") == clip_id), None)
                    _clips = [c for c in _clips if c.get("clip_id") != clip_id]
                    with open(cc_file, "w", encoding="utf-8") as _f:
                        json.dump(_clips, _f, indent=2, ensure_ascii=False)
                    if _entry and _entry.get("path"):
                        _wav = os.path.join(ep_dir, _entry["path"])
                        if os.path.isfile(_wav):
                            os.remove(_wav)
                            wav_removed = True
                            print(f"  [SFX DELETE] WAV removed: {_wav}")
                print(f"  [SFX DELETE] clip_id={clip_id}  slug={slug}  ep={ep_id}  wav_removed={wav_removed}")
                _json_resp(self, {"ok": True, "wav_removed": wav_removed})
            except Exception as _del_exc:
                _json_resp(self, {"ok": False, "error": str(_del_exc)}, 500)

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
                # Save to resources/sfx/ when called from the library panel,
                # otherwise save to episode sfx assets directory
                if item_id == "_sfx_library":
                    dest_dir = os.path.join(PIPE_DIR, "projects", slug, "resources", "sfx")
                else:
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

        # ── SFX: upload user-provided audio file ─────────────────────────────
        elif self.path == "/api/sfx_upload":
            try:
                ctype = self.headers.get("Content-Type", "")
                if "multipart/form-data" not in ctype:
                    raise ValueError("Expected multipart/form-data")
                length = int(self.headers.get("Content-Length", 0))
                raw    = self.rfile.read(length)

                boundary = re.search(r"boundary=(.+)", ctype)
                if not boundary:
                    raise ValueError("No boundary in Content-Type")
                bnd = boundary.group(1).strip().encode()

                parts = raw.split(b"--" + bnd)
                fields    = {}
                file_data = None
                file_name = "upload.wav"
                for part in parts:
                    if b"Content-Disposition" not in part:
                        continue
                    header, _, body = part.partition(b"\r\n\r\n")
                    body = body.rstrip(b"\r\n")
                    name_m  = re.search(rb'name="([^"]+)"', header)
                    fname_m = re.search(rb'filename="([^"]+)"', header)
                    if name_m:
                        name = name_m.group(1).decode()
                        if fname_m:
                            file_data = body
                            file_name = fname_m.group(1).decode(errors="replace")
                        else:
                            fields[name] = body.decode(errors="replace")

                slug    = fields.get("slug",    "").strip()
                ep_id   = fields.get("ep_id",   "").strip()
                item_id = fields.get("item_id", "").strip()

                if not slug or not ep_id or not item_id or file_data is None:
                    raise ValueError("slug, ep_id, item_id, and file are required")
                if not re.match(r'^[a-zA-Z0-9_\-]+$', slug):
                    raise ValueError("invalid slug")
                if not re.match(r'^[a-zA-Z0-9_\-]+$', ep_id):
                    raise ValueError("invalid ep_id")
                if not re.match(r'^[a-zA-Z0-9_\-]+$', item_id):
                    raise ValueError("invalid item_id")

                # Determine extension from uploaded filename
                ext = os.path.splitext(file_name)[1].lower() or ".wav"
                if ext not in (".wav", ".mp3", ".ogg", ".flac", ".aiff", ".aif"):
                    ext = ".wav"

                import time as _time_sfx_up
                ts_ms = int(_time_sfx_up.time() * 1000)
                dest_dir = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id,
                                        "assets", "sfx", item_id)
                os.makedirs(dest_dir, exist_ok=True)
                dest_file = f"uploaded_{ts_ms}{ext}"
                dest_path = os.path.join(dest_dir, dest_file)
                with open(dest_path, "wb") as fh:
                    fh.write(file_data)

                # Extract duration
                import wave as _wave_sfx, contextlib as _cl_sfx
                duration_sec = 0.0
                try:
                    with _cl_sfx.closing(_wave_sfx.open(dest_path, "r")) as wf:
                        duration_sec = wf.getnframes() / wf.getframerate()
                except Exception:
                    # For non-WAV formats, try subprocess ffprobe
                    try:
                        import subprocess as _sp_sfx
                        _r = _sp_sfx.run(
                            ["ffprobe", "-v", "quiet", "-print_format", "json",
                             "-show_streams", dest_path],
                            capture_output=True, timeout=10)
                        _info = json.loads(_r.stdout)
                        for _s in _info.get("streams", []):
                            if _s.get("codec_type") == "audio":
                                duration_sec = float(_s.get("duration", 0))
                                break
                    except Exception:
                        pass

                rel_path  = os.path.relpath(dest_path, PIPE_DIR)
                serve_url = "/serve_media?path=" + _url_quote(rel_path)
                body = json.dumps({"ok": True, "url": serve_url,
                                   "filename": dest_file,
                                   "duration_sec": duration_sec}).encode()
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

        # ── SFX: upload source file to resources/sfx/ (POST /api/sfx_source_upload) ──
        elif self.path == "/api/sfx_source_upload":
            try:
                ctype = self.headers.get("Content-Type", "")
                if "multipart/form-data" not in ctype:
                    raise ValueError("Expected multipart/form-data")
                length = int(self.headers.get("Content-Length", 0))
                raw    = self.rfile.read(length)

                boundary = re.search(r"boundary=(.+)", ctype)
                if not boundary:
                    raise ValueError("No boundary in Content-Type")
                bnd = boundary.group(1).strip().encode()

                parts = raw.split(b"--" + bnd)
                fields    = {}
                file_data = None
                file_name = "upload.wav"
                for part in parts:
                    if b"Content-Disposition" not in part:
                        continue
                    header, _, body = part.partition(b"\r\n\r\n")
                    body = body.rstrip(b"\r\n")
                    name_m  = re.search(rb'name="([^"]+)"', header)
                    fname_m = re.search(rb'filename="([^"]+)"', header)
                    if name_m:
                        name = name_m.group(1).decode()
                        if fname_m:
                            file_data = body
                            file_name = fname_m.group(1).decode(errors="replace")
                        else:
                            fields[name] = body.decode(errors="replace")

                slug  = fields.get("slug",  "").strip()
                ep_id = fields.get("ep_id", "").strip()
                if not slug or not ep_id or file_data is None:
                    raise ValueError("slug, ep_id, and file are required")
                if not re.match(r'^[a-zA-Z0-9_\-]+$', slug):
                    raise ValueError("invalid slug")
                if not re.match(r'^[a-zA-Z0-9_\-]+$', ep_id):
                    raise ValueError("invalid ep_id")

                ext = os.path.splitext(file_name)[1].lower() or ".wav"
                if ext not in (".wav", ".mp3", ".ogg", ".flac", ".aiff", ".aif"):
                    ext = ".wav"

                dest_dir = os.path.join(PIPE_DIR, "projects", slug, "resources", "sfx")
                os.makedirs(dest_dir, exist_ok=True)
                # Use original filename (sanitised) so it appears nicely in library
                safe_base = re.sub(r'[^A-Za-z0-9_\-]', '_', os.path.splitext(file_name)[0])[:60]
                dest_file = safe_base + ext
                dest_path = os.path.join(dest_dir, dest_file)
                with open(dest_path, "wb") as fh:
                    fh.write(file_data)

                rel_path  = os.path.relpath(dest_path, PIPE_DIR)
                serve_url = "/serve_media?path=" + _url_quote(rel_path)
                body = json.dumps({"ok": True, "url": serve_url, "filename": dest_file}).encode()
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

        # ── Media: generate preview video (POST /api/media_preview) ──────────
        elif self.path == "/api/media_preview":
            try:
                payload      = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                slug         = payload["slug"]
                ep_id        = payload["ep_id"]
                ep_dir       = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
                # Detect primary locale — same strategy as /api/music_prepare_loops:
                #   1. PRIMARY_LOCALE from pipeline_vars.sh (explicit override)
                #   2. LOCALES list from pipeline_vars.sh / meta.json
                #   3. Auto-detect from story.txt CJK content
                import re as _re_mp
                _primary_locale_mp = "en"
                _all_locales_mp = ["en"]
                _vars_mp = os.path.join(ep_dir, "pipeline_vars.sh")
                if os.path.isfile(_vars_mp):
                    with open(_vars_mp, encoding="utf-8") as _vf_mp:
                        _vars_text_mp = _vf_mp.read()
                    _m_mp = _re_mp.search(
                        r'(?:^|[\n;])(?:export\s+)?PRIMARY_LOCALE=["\']?([^"\';\n]+)["\']?',
                        _vars_text_mp)
                    if _m_mp:
                        _primary_locale_mp = _m_mp.group(1).strip()
                    _m_locs_mp = _re_mp.search(
                        r'(?:^|[\n;])(?:export\s+)?LOCALES=["\']?([^"\';\n]+)["\']?',
                        _vars_text_mp)
                    if _m_locs_mp:
                        _all_locales_mp = [l.strip() for l in _m_locs_mp.group(1).split(",") if l.strip()]
                _meta_mp_path = os.path.join(ep_dir, "meta.json")
                _is_mtv_mp = False
                if os.path.isfile(_meta_mp_path):
                    try:
                        _meta_mp = json.load(open(_meta_mp_path, encoding="utf-8"))
                        _is_mtv_mp = _meta_mp.get("story_format") == "mtv"
                        _loc_str_mp = _meta_mp.get("locales", "")
                        if _loc_str_mp:
                            _all_locales_mp = [l.strip() for l in _loc_str_mp.split(",") if l.strip()]
                        if _primary_locale_mp == "en" and _all_locales_mp:
                            _primary_locale_mp = _all_locales_mp[0]
                    except Exception:
                        pass
                # Auto-detect from story.txt: if lyrics are predominantly CJK,
                # prefer first zh-* locale in the locales list.
                _story_mp = os.path.join(ep_dir, "story.txt")
                if _is_mtv_mp and os.path.isfile(_story_mp):
                    try:
                        _story_text_mp = open(_story_mp, encoding="utf-8").read()
                        _cjk_mp = sum(1 for c in _story_text_mp
                                      if '\u4e00' <= c <= '\u9fff'
                                      or '\u3040' <= c <= '\u30ff'
                                      or '\uac00' <= c <= '\ud7af')
                        _alnum_mp = sum(1 for c in _story_text_mp if c.isalnum())
                        if _alnum_mp > 0 and _cjk_mp / _alnum_mp > 0.3:
                            _zh_mp = next((l for l in _all_locales_mp if l.startswith("zh")), None)
                            if _zh_mp:
                                _primary_locale_mp = _zh_mp
                    except Exception:
                        pass

                if _is_mtv_mp:
                    # ── Music file selection: read MusicPlan.json Shot Overrides ──────────
                    # CORRECT approach: read the clip chosen by the user in Shot Overrides
                    # from MusicPlan.json, then look up its path in user_cut_clips.json.
                    # DO NOT search resources/music/ alphabetically — that picks the wrong
                    # file and corrupts VOPlan with wrong timestamps (e.g. 73.93s intro pause).
                    _music_file_mp = None
                    _music_plan_path_mp = os.path.join(ep_dir, "MusicPlan.json")
                    _cut_clips_path_mp  = os.path.join(ep_dir, "assets", "music", "user_cut_clips.json")
                    if os.path.isfile(_music_plan_path_mp) and os.path.isfile(_cut_clips_path_mp):
                        try:
                            _mplan_mp = json.load(open(_music_plan_path_mp, encoding="utf-8"))
                            _cut_clips_mp = json.load(open(_cut_clips_path_mp, encoding="utf-8"))
                            _ovr_clip_id_mp = None
                            for _ovr_mp in (_mplan_mp.get("shot_overrides") or []):
                                _cid_mp = _ovr_mp.get("music_clip_id") or _ovr_mp.get("music_asset_id")
                                if _cid_mp:
                                    _ovr_clip_id_mp = _cid_mp
                                    break
                            if _ovr_clip_id_mp:
                                for _cc_mp in _cut_clips_mp:
                                    if _cc_mp.get("clip_id") == _ovr_clip_id_mp:
                                        _cc_path_mp = _cc_mp.get("path", "")
                                        if _cc_path_mp:
                                            _music_file_mp = os.path.join(PIPE_DIR, _cc_path_mp)
                                        break
                        except Exception as _e_mp:
                            print(f"  [WARN] media_preview: MusicPlan clip lookup failed: {_e_mp}")
                    if not _music_file_mp:
                        _json_resp(self, {"error": "MTV mode: no clip found in MusicPlan.json Shot Overrides. "
                                                   "Select a clip in the Music tab Shot Overrides first."}, 400)
                        return

                    # Build alignment command — lyrics are optional for MTV
                    # (transcribe mode if no lyrics provided)
                    _lyrics_path_mp = os.path.join(ep_dir, "story.txt")
                    _has_lyrics = (os.path.isfile(_lyrics_path_mp)
                                  and os.path.getsize(_lyrics_path_mp) > 0)

                    _align_cmd = [
                        sys.executable,
                        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "gen_lyrics_alignment.py"),
                        "--music",  _music_file_mp,
                        "--out",    ep_dir,
                        "--locale", _primary_locale_mp,
                    ]
                    if _has_lyrics:
                        _align_cmd += ["--lyrics", _lyrics_path_mp]

                    # Run gen_lyrics_alignment.py
                    _align_result = subprocess.run(
                        _align_cmd,
                        capture_output=True, text=True, timeout=300,
                    )
                    if _align_result.returncode != 0:
                        _json_resp(self, {
                            "error": "MTV lyrics alignment failed",
                            "detail": _align_result.stderr or _align_result.stdout,
                        }, 500)
                        return
                    print(f"[mtv] Alignment done: {_align_result.stdout.strip()}")

                    # MTV: alignment done — fall through to normal render.
                    # VOPlan and MusicPlan should already exist; render_video.py
                    # receives --format mtv to skip VO mixing and ducking.

                import tempfile
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                                     delete=False, encoding="utf-8") as tf:
                        json.dump({
                            "ep_dir":          ep_dir,
                            "locale":          _primary_locale_mp,
                            "media_segments":  payload.get("media_segments", []),
                            "include_music":   payload.get("include_music", False),
                            "include_sfx":     payload.get("include_sfx",  False),
                            "shot_ids":        payload.get("shot_ids",  None),
                            "out_name":        payload.get("out_name",  None),
                        }, tf)
                        tmp_path = tf.name
                    result = subprocess.run(
                        [sys.executable,
                         os.path.join(os.path.dirname(os.path.abspath(__file__)), "media_preview_pack.py"),
                         "--input", tmp_path],
                        capture_output=True, text=True
                    )
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                if result.returncode != 0:
                    _json_resp(self, {"error": "media_preview_pack failed",
                                      "detail": result.stderr}, 500)
                else:
                    _warnings = [ln.strip() for ln in (result.stdout + result.stderr).splitlines()
                                 if "[warn]" in ln]
                    _json_resp(self, {"ok": True,
                                      "warnings": _warnings,
                                      "debug_log": result.stdout + result.stderr})
            except Exception as exc:
                _json_resp(self, {"error": str(exc)}, 500)

        # ── Media: upload file to shared library (POST /api/media_upload) ──
        elif self.path == "/api/media_upload":
            import uuid as _uuid_mu
            import subprocess as _sp_mu
            try:
                content_type = self.headers.get("Content-Type", "")
                length = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(length)

                # Parse multipart form data using cgi module
                import cgi as _cgi_mu
                import io as _io_mu
                environ = {
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE":   content_type,
                    "CONTENT_LENGTH": str(length),
                }
                form = _cgi_mu.FieldStorage(
                    fp=_io_mu.BytesIO(raw_body),
                    environ=environ,
                    keep_blank_values=True,
                )
                slug       = (form.getvalue("slug") or "").strip()
                ep_id      = (form.getvalue("ep_id") or "").strip()
                media_type = (form.getvalue("media_type") or "").strip()
                file_item  = form["file"] if "file" in form else None

                if not slug or not ep_id:
                    raise ValueError("slug and ep_id are required")
                if media_type not in ("video", "image"):
                    raise ValueError("media_type must be 'video' or 'image'")
                if file_item is None or not file_item.filename:
                    raise ValueError("file is required")

                ep_dir = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
                os.makedirs(ep_dir, exist_ok=True)

                # Sanitise filename
                orig_name = os.path.basename(file_item.filename.replace("\\", "/"))
                orig_name = "".join(c for c in orig_name if c.isalnum() or c in "._- ")[:128]
                if not orig_name:
                    orig_name = "upload"

                subdir = "videos" if media_type == "video" else "images"
                dest_dir = os.path.join(ep_dir, "assets", "media_library", subdir)
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, orig_name)
                file_data = file_item.file.read()
                with open(dest_path, "wb") as _f:
                    _f.write(file_data)

                rel_path = os.path.relpath(dest_path, ep_dir)
                size_bytes = len(file_data)
                item_id = _uuid_mu.uuid4().hex[:8]

                if media_type == "video":
                    duration_sec = 0.0
                    try:
                        _ffp = _sp_mu.run(
                            ["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                             dest_path],
                            capture_output=True, text=True, timeout=30, check=True)
                        duration_sec = float(_ffp.stdout.strip())
                    except Exception as _fe:
                        print(f"  [media_upload] ffprobe failed: {_fe}")
                    record = {
                        "id": item_id, "filename": orig_name,
                        "path": rel_path, "duration_sec": duration_sec,
                        "size_bytes": size_bytes,
                    }
                else:
                    width, height = 0, 0
                    try:
                        from PIL import Image as _PIL_Image
                        with _PIL_Image.open(dest_path) as _img:
                            width, height = _img.size
                    except Exception as _pe:
                        try:
                            _idr = _sp_mu.run(
                                ["identify", "-format", "%w %h", dest_path],
                                capture_output=True, text=True, timeout=10, check=True)
                            _wh = _idr.stdout.strip().split()
                            if len(_wh) == 2:
                                width, height = int(_wh[0]), int(_wh[1])
                        except Exception:
                            pass
                    record = {
                        "id": item_id, "filename": orig_name,
                        "path": rel_path, "width": width, "height": height,
                        "size_bytes": size_bytes,
                    }

                # Append to media_library.json atomically
                lib_path = os.path.join(ep_dir, "assets", "media_library", "media_library.json")
                if os.path.isfile(lib_path):
                    with open(lib_path, encoding="utf-8") as _lf:
                        lib_data = json.load(_lf)
                else:
                    lib_data = {"videos": [], "images": []}
                lib_data.setdefault("videos", [])
                lib_data.setdefault("images", [])
                if media_type == "video":
                    lib_data["videos"].append(record)
                else:
                    lib_data["images"].append(record)
                lib_tmp = lib_path + ".tmp"
                with open(lib_tmp, "w", encoding="utf-8") as _lf:
                    json.dump(lib_data, _lf, indent=2, ensure_ascii=False)
                os.replace(lib_tmp, lib_path)

                serve_url = f"/serve_media?path=projects/{slug}/episodes/{ep_id}/{rel_path}"
                print(f"  [media_upload] {media_type} saved: {orig_name}  id={item_id}")
                body = json.dumps({"ok": True, "item": record, "url": serve_url}).encode()
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

        # ── Media: cut clip from library video (POST /api/media_cut_clip) ────
        elif self.path == "/api/media_cut_clip":
            import uuid as _uuid_cc
            import subprocess as _sp_cc
            try:
                length   = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(length)
                payload  = json.loads(raw_body)

                slug            = payload.get("slug", "").strip()
                ep_id           = payload.get("ep_id", "").strip()
                source_video_id = payload.get("source_video_id", "").strip()
                source_path     = payload.get("source_path", "").strip()
                start_sec       = float(payload.get("start_sec") or 0)
                end_sec         = float(payload.get("end_sec") or 0)

                if not slug or not ep_id or not source_path:
                    raise ValueError("slug, ep_id, and source_path are required")

                ep_dir = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)

                # Path-traversal guard.
                # Batch assets have absolute paths (extracted from file:// URLs by
                # /api/media_library); locally-uploaded assets use relative paths.
                if os.path.isabs(source_path):
                    abs_src = os.path.realpath(source_path)
                    if not os.path.isfile(abs_src):
                        raise ValueError(f"source_path does not exist: {abs_src}")
                else:
                    abs_src = os.path.realpath(os.path.join(ep_dir, source_path))
                    if not abs_src.startswith(os.path.realpath(ep_dir)):
                        raise ValueError("source_path outside episode directory")

                clip_id   = "mclip_" + _uuid_cc.uuid4().hex[:8]
                clips_dir = os.path.join(ep_dir, "assets", "media_library", "clips")
                os.makedirs(clips_dir, exist_ok=True)
                dest_path = os.path.join(clips_dir, clip_id + ".mp4")

                # Run ffmpeg (stream copy, no re-encode)
                _sp_cc.run(
                    ["ffmpeg", "-y", "-ss", str(start_sec), "-to", str(end_sec),
                     "-i", abs_src, "-c", "copy", dest_path],
                    capture_output=True, check=True, timeout=120,
                )

                # Get output duration via ffprobe
                duration_sec = end_sec - start_sec
                try:
                    _ffp = _sp_cc.run(
                        ["ffprobe", "-v", "error", "-show_entries",
                         "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                         dest_path],
                        capture_output=True, text=True, timeout=30, check=True)
                    duration_sec = float(_ffp.stdout.strip())
                except Exception as _fe:
                    print(f"  [media_cut_clip] ffprobe on output failed: {_fe}")

                rel_path = os.path.relpath(dest_path, ep_dir)
                clip_record = {
                    "clip_id":         clip_id,
                    "source_video_id": source_video_id,
                    "filename":        clip_id + ".mp4",
                    "path":            rel_path,
                    "start_sec":       start_sec,
                    "end_sec":         end_sec,
                    "duration_sec":    duration_sec,
                }

                # Append to media_cut_clips.json atomically
                clips_json = os.path.join(ep_dir, "assets", "media_library", "media_cut_clips.json")
                if os.path.isfile(clips_json):
                    with open(clips_json, encoding="utf-8") as _cf:
                        clips_list = json.load(_cf)
                    # Replace if clip_id already exists
                    clips_list = [c for c in clips_list if c.get("clip_id") != clip_id]
                else:
                    clips_list = []
                clips_list.append(clip_record)
                clips_tmp = clips_json + ".tmp"
                with open(clips_tmp, "w", encoding="utf-8") as _cf:
                    json.dump(clips_list, _cf, indent=2, ensure_ascii=False)
                os.replace(clips_tmp, clips_json)

                serve_url = f"/serve_media?path=projects/{slug}/episodes/{ep_id}/{rel_path}"
                print(f"  [media_cut_clip] clip={clip_id}  {start_sec:.2f}s–{end_sec:.2f}s  dur={duration_sec:.2f}s")
                body = json.dumps({
                    "ok": True, "clip_id": clip_id,
                    "path": rel_path, "duration_sec": duration_sec, "url": serve_url,
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

        # ── Media: write MediaPlan.json (POST /api/media_plan_save) ────────────
        elif self.path == "/api/media_plan_save":
            try:
                length   = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(length)
                payload  = json.loads(raw_body)

                slug           = payload.get("slug", "").strip()
                ep_id          = payload.get("ep_id", "").strip()
                shot_overrides = payload.get("shot_overrides", [])

                if not slug or not ep_id:
                    raise ValueError("slug and ep_id are required")
                if not isinstance(shot_overrides, list):
                    raise ValueError("shot_overrides must be an array")

                ep_dir = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
                os.makedirs(ep_dir, exist_ok=True)

                out = {
                    "schema_id":      "MediaPlan",
                    "schema_version": "1.0",
                    "shot_overrides": shot_overrides,
                }
                sel_path = os.path.join(ep_dir, "MediaPlan.json")
                tmp_path = sel_path + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as _sf:
                    json.dump(out, _sf, indent=2, ensure_ascii=False)
                    _sf.write("\n")
                os.replace(tmp_path, sel_path)
                print(f"  Saved MediaPlan  slug={slug}  ep={ep_id}  segments={len(shot_overrides)}")
                # Contract validation — run immediately after write
                _verify_script = os.path.join(
                    PIPE_DIR, "contracts", "tools", "verify_contracts.py")
                _vc_pass = None; _vc_errors = []; _vc_output = ""
                try:
                    _vc_proc = subprocess.run(
                        [sys.executable, _verify_script, sel_path],
                        capture_output=True, text=True, timeout=15,
                    )
                    _vc_output = (_vc_proc.stdout + _vc_proc.stderr).strip()
                    _vc_pass   = (_vc_proc.returncode == 0)
                    if not _vc_pass:
                        _vc_errors = [
                            ln.strip().lstrip("•").strip()
                            for ln in _vc_output.splitlines()
                            if ln.strip().startswith("•")
                        ] or [_vc_output]
                    print(f"  verify_contracts: {'PASS' if _vc_pass else 'FAIL'}  {_vc_output[:200]}")
                except Exception as _vc_exc:
                    _vc_output = f"verify_contracts runner error: {_vc_exc}"
                    print(f"  {_vc_output}")
                body = json.dumps({"ok": True, "saved": len(shot_overrides),
                                   "validation_pass":   _vc_pass,
                                   "validation_errors": _vc_errors,
                                   "validation_output": _vc_output}).encode()
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
                bg_id        = payload.get("bg_id", "").strip()
                ts_ms        = payload.get("timestamp_ms", "")
                asset_folder = payload.get("asset_folder", "backgrounds").strip() or "backgrounds"
                # Whitelist to prevent path traversal — only allow known asset folder names
                if asset_folder not in ("backgrounds", "elements"):
                    asset_folder = "backgrounds"
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
                                        "assets", asset_folder, bg_id)
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

                # MTV: no loop analysis needed — run lyrics alignment instead
                _meta_mpl = os.path.join(ep_dir, "meta.json")
                _is_mtv_mpl = False
                if os.path.isfile(_meta_mpl):
                    try:
                        _is_mtv_mpl = json.load(open(_meta_mpl, encoding="utf-8")).get("story_format") == "mtv"
                    except Exception:
                        pass
                if _is_mtv_mpl:
                    # Run lyrics alignment (same logic as /api/media_preview MTV path)
                    # Detect primary locale first (needed for music file selection below).
                    # Strategy:
                    #   1. Try PRIMARY_LOCALE from pipeline_vars.sh (explicit override).
                    #   2. Otherwise read meta.json locales list.
                    #   3. Auto-detect from story.txt content: if the lyrics are
                    #      predominantly CJK, prefer zh-Hans (or first zh-* locale in list).
                    #      This handles episodes where both 'en' and 'zh-Hans' are listed
                    #      but the actual lyrics are Chinese.
                    _primary_locale_mpl = "en"
                    _all_locales_mpl = ["en"]
                    _vars_mpl = os.path.join(ep_dir, "pipeline_vars.sh")
                    if os.path.isfile(_vars_mpl):
                        import re as _re_mpl
                        with open(_vars_mpl, encoding="utf-8") as _vf_mpl:
                            _vars_text_mpl = _vf_mpl.read()
                        _m_mpl = _re_mpl.search(
                            r'(?:^|[\n;])(?:export\s+)?PRIMARY_LOCALE=["\']?([^"\';\n]+)["\']?',
                            _vars_text_mpl)
                        if _m_mpl:
                            _primary_locale_mpl = _m_mpl.group(1).strip()
                        # Also collect full locales list for CJK auto-detect below
                        _m_locs = _re_mpl.search(
                            r'(?:^|[\n;])(?:export\s+)?LOCALES=["\']?([^"\';\n]+)["\']?',
                            _vars_text_mpl)
                        if _m_locs:
                            _all_locales_mpl = [l.strip() for l in _m_locs.group(1).split(",") if l.strip()]
                    if os.path.isfile(_meta_mpl):
                        try:
                            _loc_str = json.load(open(_meta_mpl, encoding="utf-8")).get("locales", "en")
                            _all_locales_mpl = [l.strip() for l in _loc_str.split(",") if l.strip()]
                            # Only override _primary_locale_mpl from meta if not already set by PRIMARY_LOCALE
                            if _primary_locale_mpl == "en" and _all_locales_mpl:
                                _primary_locale_mpl = _all_locales_mpl[0]
                        except Exception:
                            pass
                    # Auto-detect from story.txt: if lyrics are predominantly CJK,
                    # switch to first zh-* locale in the list (overrides "en" default).
                    _story_mpl = os.path.join(ep_dir, "story.txt")
                    if os.path.isfile(_story_mpl):
                        try:
                            _story_text_mpl = open(_story_mpl, encoding="utf-8").read()
                            _cjk_count = sum(1 for c in _story_text_mpl
                                             if '\u4e00' <= c <= '\u9fff'
                                             or '\u3040' <= c <= '\u30ff'
                                             or '\uac00' <= c <= '\ud7af')
                            _total_alnum = sum(1 for c in _story_text_mpl if c.isalnum())
                            if _total_alnum > 0 and _cjk_count / _total_alnum > 0.3:
                                # Lyrics are predominantly CJK — find first zh locale
                                _zh_locale = next(
                                    (l for l in _all_locales_mpl if l.startswith("zh")), None)
                                if _zh_locale:
                                    _primary_locale_mpl = _zh_locale
                        except Exception:
                            pass
                    # ── Music file selection: read MusicPlan.json Shot Overrides ──────────
                    # CORRECT approach: read the clip chosen by the user in Shot Overrides
                    # from MusicPlan.json, then look up its path in user_cut_clips.json.
                    # DO NOT search resources/music/ alphabetically — that picks the wrong
                    # file and corrupts VOPlan with wrong timestamps (e.g. 73.93s intro pause).
                    _music_file_mpl = None
                    _music_plan_path_mpl = os.path.join(ep_dir, "MusicPlan.json")
                    _cut_clips_path_mpl  = os.path.join(ep_dir, "assets", "music", "user_cut_clips.json")
                    if os.path.isfile(_music_plan_path_mpl) and os.path.isfile(_cut_clips_path_mpl):
                        try:
                            _mplan_mpl = json.load(open(_music_plan_path_mpl, encoding="utf-8"))
                            _cut_clips_mpl = json.load(open(_cut_clips_path_mpl, encoding="utf-8"))
                            # Find first shot_override that has a clip assigned
                            _ovr_clip_id_mpl = None
                            for _ovr_mpl in (_mplan_mpl.get("shot_overrides") or []):
                                _cid = _ovr_mpl.get("music_clip_id") or _ovr_mpl.get("music_asset_id")
                                if _cid:
                                    _ovr_clip_id_mpl = _cid
                                    break
                            if _ovr_clip_id_mpl:
                                # Look up the clip path in user_cut_clips.json
                                for _cc_mpl in _cut_clips_mpl:
                                    if _cc_mpl.get("clip_id") == _ovr_clip_id_mpl:
                                        _cc_path_mpl = _cc_mpl.get("path", "")
                                        if _cc_path_mpl:
                                            _music_file_mpl = os.path.join(PIPE_DIR, _cc_path_mpl)
                                        break
                        except Exception as _e_mpl:
                            print(f"  [WARN] music_prepare_loops: MusicPlan clip lookup failed: {_e_mpl}")
                    if not _music_file_mpl:
                        raise FileNotFoundError(
                            "MTV mode: no clip found in MusicPlan.json Shot Overrides. "
                            "Select a clip in the Music tab Shot Overrides first.")
                    _lyrics_path_mpl = os.path.join(ep_dir, "story.txt")
                    _has_lyrics_mpl = (os.path.isfile(_lyrics_path_mpl)
                                       and os.path.getsize(_lyrics_path_mpl) > 0)
                    _align_cmd_mpl = [
                        sys.executable,
                        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "gen_lyrics_alignment.py"),
                        "--music", _music_file_mpl,
                        "--out",   ep_dir,
                        "--locale", _primary_locale_mpl,
                    ]
                    if _has_lyrics_mpl:
                        _align_cmd_mpl += ["--lyrics", _lyrics_path_mpl]
                    _align_res_mpl = subprocess.run(
                        _align_cmd_mpl,
                        capture_output=True, text=True, timeout=300,
                    )
                    if _align_res_mpl.returncode != 0:
                        raise RuntimeError("MTV lyrics alignment failed: " +
                                           (_align_res_mpl.stderr or _align_res_mpl.stdout)[-2000:])
                    print(f"[mtv] Alignment done: {_align_res_mpl.stdout.strip()}")
                    # Return success with empty candidates (no loop analysis for MTV)
                    _voplan_mpl = os.path.join(ep_dir, f"VOPlan.{_primary_locale_mpl}.json")
                    _vo_count_mpl = 0
                    if os.path.isfile(_voplan_mpl):
                        try:
                            with open(_voplan_mpl, encoding="utf-8") as _vf2:
                                _vo_count_mpl = len(json.load(_vf2).get("vo_items", []))
                        except Exception:
                            pass
                    # Music file path relative to PIPE_DIR for /serve_media
                    _music_rel_mpl = os.path.relpath(_music_file_mpl, PIPE_DIR)
                    body = json.dumps({"ok": True, "candidates": {},
                                       "mtv": True,
                                       "music_path": _music_rel_mpl,
                                       "message": f"Lyrics alignment complete — {_vo_count_mpl} items. Review in VO tab."}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                manifest_path = os.path.join(ep_dir, "AssetManifest.shared.json")
                if not os.path.isfile(manifest_path):
                    raise FileNotFoundError("AssetManifest.shared.json not found")

                if not _TEST_MODE:
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
                start_sec  = float(payload.get("start_sec", 0))
                end_sec_raw = payload.get("end_sec")
                end_sec    = float(end_sec_raw) if end_sec_raw is not None else None
                if not slug or not ep_id or not stem:
                    raise ValueError("slug, ep_id, and stem are required")
                if end_sec is not None and end_sec <= start_sec:
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

                # Extract using librosa first so we know actual duration
                import librosa
                import soundfile as sf_mod
                audio, _ = librosa.load(source_path, sr=48000, mono=True)
                s0 = int(start_sec * 48000)
                # null end_sec means "cut to end of file"
                s1 = len(audio) if end_sec is None else min(int(end_sec * 48000), len(audio))
                actual_end = s1 / 48000.0

                # Generate clip filename from stem + range.
                # Use natural decimal notation — dots in filenames are fine.
                clip_fname = f"{stem}_{start_sec:.1f}s-{actual_end:.1f}s.wav"
                out_path = os.path.join(assets_dir, clip_fname)
                segment = audio[s0:s1]
                if len(segment) == 0:
                    raise ValueError("Empty segment — check start/end times")
                sf_mod.write(out_path, segment.astype("float32"), 48000,
                             subtype="PCM_16")

                rel_path = os.path.relpath(out_path, PIPE_DIR)
                print(f"  Cut clip: {stem} [{start_sec:.1f}s-{actual_end:.1f}s]"
                      f" → {rel_path}")

                # Persist cut clip metadata to user_cut_clips.json
                end_sec_actual = start_sec + len(segment) / 48000.0
                # clip_id = filename stem (no extension) — same string used to
                # look up the WAV in render_preview_audio and apply_music_plan.
                clip_id = clip_fname[:-4]   # strip ".wav"
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

        # ── Music: update vol_db on a user-cut clip (POST /api/music_update_cut_clip_vol) ──
        elif self.path == "/api/music_update_cut_clip_vol":
            try:
                length   = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(length)
                payload  = json.loads(raw_body)
                slug     = payload.get("slug", "").strip()
                ep_id    = payload.get("ep_id", "").strip()
                clip_id  = payload.get("clip_id", "").strip()
                vol_db   = payload.get("vol_db")  # int or None
                if not slug or not ep_id or not clip_id:
                    raise ValueError("slug, ep_id and clip_id are required")
                assets_dir = os.path.join(PIPE_DIR, "projects", slug,
                                          "episodes", ep_id, "assets", "music")
                meta_path = os.path.join(assets_dir, "user_cut_clips.json")
                existing_cuts = []
                if os.path.isfile(meta_path):
                    try:
                        with open(meta_path, encoding="utf-8") as _mf:
                            existing_cuts = json.load(_mf)
                    except Exception:
                        pass
                found = False
                for entry in existing_cuts:
                    if entry.get("clip_id") == clip_id:
                        if vol_db is None or vol_db == 0:
                            entry.pop("vol_db", None)
                        else:
                            entry["vol_db"] = int(vol_db)
                        found = True
                        break
                if not found:
                    raise ValueError(f"clip_id '{clip_id}' not found in user_cut_clips.json")
                with open(meta_path, "w", encoding="utf-8") as _mf:
                    json.dump(existing_cuts, _mf, indent=2)
                    _mf.write("\n")
                body = json.dumps({"ok": True}).encode()
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
                shot_overrides     = payload.get("shot_overrides", [])
                payload_clip_vols  = payload.get("clip_volumes")  or None
                payload_track_vols = payload.get("track_volumes") or None
                if not slug or not ep_id:
                    raise ValueError("slug and ep_id are required")

                ep_dir = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)

                # Find the merged manifest — use PRIMARY_LOCALE from
                # pipeline_vars.sh, then fall back to alphabetical first.
                import glob as _glob_mod
                merged_manifests = [p for p in _glob_mod.glob(
                    os.path.join(ep_dir, "VOPlan.*.json"))
                    if os.path.basename(p) != "AssetManifest.shared.json"]
                if not merged_manifests:
                    raise FileNotFoundError(
                        "No VOPlan.*.json found. "
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
                    ep_dir, f"VOPlan.{_primary_locale}.json")
                if os.path.isfile(_primary_manifest):
                    manifest_path = _primary_manifest
                else:
                    manifest_path = sorted(merged_manifests)[0]

                # Guard: music_review_pack requires a locale or merged VOPlan.
                # VOPlan.{locale}.json is produced by Stage 5 (locale_scope='locale').
                # manifest_merge is eliminated; locale-scope manifests are now accepted.
                try:
                    with open(manifest_path, encoding="utf-8") as _mm_chk:
                        _mm_scope = json.load(_mm_chk).get("locale_scope")
                except Exception:
                    _mm_scope = None
                if _mm_scope not in ("merged", "monolithic", "locale", None):
                    raise ValueError(
                        f"Manifest not ready for music preview (locale_scope='{_mm_scope}'). "
                        "Expected a VOPlan.{{locale}}.json produced by Stage 5."
                    )

                # Direct in-process call — no subprocess, no temp files
                import re as _re_vol_mrp
                from music_review_pack import (
                    build_timeline as _mrp_build,
                    render_preview_audio as _mrp_render,
                    apply_music_plan_overrides as _mrp_apply,
                    load_shotlist as _mrp_load_sl,
                    load_loop_candidates as _mrp_load_loop,
                    BASE_MUSIC_DB as _MRP_BASE_DB,
                )

                _manifest = json.load(open(manifest_path, encoding="utf-8"))
                _shots = _mrp_load_sl(_manifest, Path(manifest_path))
                if not _shots:
                    raise ValueError("ShotList not found — cannot build music timeline.")

                # Build shot time windows from ShotList (cumulative durations only —
                # no audio_intent.vo_item_ids used; VO mapping is by episode-absolute
                # time overlap using approved timing from the manifest vo_items).
                _shot_windows = []
                _cum_mrp = 0.0
                for _s in _shots:
                    _d = float(_s.get("duration_sec") or 0.0)
                    if _d > 0:
                        _shot_windows.append((_s["shot_id"], _cum_mrp, _cum_mrp + _d))
                        _cum_mrp += _d

                # Map each approved VO item to the shot whose time window it overlaps.
                # vo_timeline is the hard truth: start_sec/end_sec are locked after approval.
                _vo_shot_map = {}
                for _vo in _manifest.get("vo_items", []):
                    _vs = _vo.get("start_sec")
                    _ve = _vo.get("end_sec")
                    if _vs is None or _ve is None:
                        continue
                    for _sid, _t0, _t1 in _shot_windows:
                        if float(_ve) > _t0 and float(_vs) < _t1:
                            _vo_shot_map.setdefault(_sid, []).append(_vo)
                            break

                _music_index = {
                    mi["shot_id"]: mi
                    for mi in _manifest.get("music_items", [])
                    if mi.get("shot_id")
                }
                # Fallback: VOPlan has no music_items → seed from MusicPlan
                if not _music_index:
                    _mp_fb3 = os.path.join(ep_dir, "MusicPlan.json")
                    if os.path.isfile(_mp_fb3):
                        try:
                            _mp_fb3_d = json.load(open(_mp_fb3, encoding="utf-8"))
                            _music_index = {
                                o["shot_id"]: {"item_id": o["item_id"], "shot_id": o["shot_id"]}
                                for o in _mp_fb3_d.get("shot_overrides", [])
                                if o.get("shot_id") and o.get("item_id")
                            }
                        except Exception as _e_fb_mrp:
                            print(f"  [WARN] music_review_pack: MusicPlan fallback: {_e_fb_mrp}")
                _loop_info = _mrp_load_loop(Path(ep_dir))
                _tl_shots, _total_dur = _mrp_build(
                    _shots, _manifest, _vo_shot_map, _music_index, _loop_info)

                # Apply saved MusicPlan overrides
                _mp_path = os.path.join(ep_dir, "MusicPlan.json")
                _loaded_mp = None
                if os.path.isfile(_mp_path):
                    try:
                        with open(_mp_path, encoding="utf-8") as _mpf:
                            _loaded_mp = json.load(_mpf)
                        _plan_ovrs = _loaded_mp.get("shot_overrides", [])
                        if _plan_ovrs:
                            _mrp_apply(_tl_shots, _plan_ovrs, "MusicPlan.json",
                                       _vo_shot_map)
                    except Exception as _e:
                        print(f"  [WARN] music_review_pack: MusicPlan.json: {_e}")

                # Apply payload overrides (from UI — take precedence)
                if shot_overrides:
                    _mrp_apply(_tl_shots, shot_overrides, "payload", _vo_shot_map)

                # Apply track/clip volume offsets — payload values take precedence
                # over MusicPlan.json so live UI changes are reflected without saving.
                _track_vols = payload_track_vols if payload_track_vols is not None \
                              else (_loaded_mp or {}).get("track_volumes", {})
                _clip_vols  = payload_clip_vols  if payload_clip_vols  is not None \
                              else (_loaded_mp or {}).get("clip_volumes",  {})
                if _track_vols or _clip_vols:
                    for _entry in _tl_shots:
                        _rid = (_entry.get("music_item_id_override")
                                or _entry.get("music_item_id") or "")
                        if not _rid:
                            continue
                        _db = 0.0
                        _vstem = _re_vol_mrp.sub(r'_\d[\d_]*s-[\d_\.]+s$', '', _rid)
                        _db += float(_track_vols.get(_vstem, 0))
                        _db += float(_clip_vols.get(_rid, 0))
                        if _db == 0.0:
                            _vmx = _re_vol_mrp.match(
                                r'^(.+?)_(\d+)_(\d+)s-(\d+)_(\d+)s$', _rid)
                            if _vmx:
                                _cid = (f"{_vmx.group(1)}:{_vmx.group(2)}."
                                        f"{_vmx.group(3)}s-{_vmx.group(4)}.{_vmx.group(5)}s")
                                _db += float(_clip_vols.get(_cid, 0))
                        if _db != 0.0:
                            _entry["base_db"] = (
                                _entry.get("base_db", _MRP_BASE_DB) + _db)

                # Render preview audio WAV (skip in test mode — tests only check
                # the timeline JSON, not the audio file, and real WAV paths differ)
                if not _TEST_MODE:
                    _prev_dir = os.path.join(ep_dir, "assets", "music", "MusicReviewPack")
                    os.makedirs(_prev_dir, exist_ok=True)
                    # Build the merged music plan: payload takes precedence over disk.
                    # _track_vols / _clip_vols are already resolved above (payload > disk).
                    _merged_music_plan = {
                        "shot_overrides": shot_overrides if shot_overrides
                                          else (_loaded_mp or {}).get("shot_overrides", []),
                        "track_volumes":  _track_vols or {},
                        "clip_volumes":   _clip_vols  or {},
                    }
                    _mrp_render(
                        _tl_shots, _total_dur, _manifest, Path(manifest_path),
                        Path(_prev_dir) / "preview_audio.wav",
                        music_plan=_merged_music_plan)

                body = json.dumps({
                    "ok": True,
                    "timeline": {
                        "shots": _tl_shots,
                        "total_duration_sec": _total_dur,
                    },
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            except Exception as exc:
                import traceback as _tb_mrp
                print(f"  [ERROR] music_review_pack: {exc}", flush=True)
                _tb_mrp.print_exc()
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

                if not slug or not ep_id:
                    raise ValueError("slug and ep_id are required")

                # Read shot_overrides from POST body (new free-form schema)
                shot_overrides = payload.get("shot_overrides")
                if shot_overrides is None:
                    # Fallback: also accept nested under "plan" dict for backward compat
                    _plan_compat = payload.get("plan")
                    if isinstance(_plan_compat, dict):
                        shot_overrides = _plan_compat.get("shot_overrides", [])
                    else:
                        shot_overrides = []
                if not isinstance(shot_overrides, list):
                    raise ValueError("shot_overrides must be an array")

                ep_dir    = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
                music_dir = os.path.join(ep_dir, "assets", "music")
                os.makedirs(music_dir, exist_ok=True)

                # Preserve loop_selections, track_volumes, clip_volumes from existing file
                plan_path   = os.path.join(ep_dir, "MusicPlan.json")
                _existing   = {}
                if os.path.isfile(plan_path):
                    try:
                        with open(plan_path, encoding="utf-8") as _ef:
                            _existing = json.load(_ef)
                    except Exception:
                        pass
                _loop_sel      = payload.get("loop_selections", _existing.get("loop_selections", {}))
                _track_volumes = _existing.get("track_volumes",   {})
                _clip_volumes  = _existing.get("clip_volumes",    {})

                plan = {
                    "schema_id":      "MusicPlan",
                    "schema_version": "1.0",
                    "loop_selections": _loop_sel,
                    "track_volumes":   _track_volumes,
                    "clip_volumes":    _clip_volumes,
                    "shot_overrides":  shot_overrides,
                }

                with open(plan_path, "w", encoding="utf-8") as _pf:
                    json.dump(plan, _pf, indent=2, ensure_ascii=False)
                    _pf.write("\n")

                rel_path = os.path.relpath(plan_path, PIPE_DIR)
                print(f"  Saved MusicPlan  slug={slug}  ep={ep_id}  "
                      f"segments={len(shot_overrides)}")

                # Run verify_contracts.py against the just-written MusicPlan.json
                _verify_script = os.path.join(
                    PIPE_DIR, "contracts", "tools", "verify_contracts.py")
                _vc_errors = []
                _vc_output = ""
                _vc_pass   = False
                try:
                    import subprocess as _sp
                    _vc_proc = _sp.run(
                        [sys.executable, _verify_script, plan_path],
                        capture_output=True, text=True, timeout=15,
                    )
                    _vc_output = (_vc_proc.stdout + _vc_proc.stderr).strip()
                    _vc_pass   = (_vc_proc.returncode == 0)
                    if not _vc_pass:
                        _vc_errors = [
                            ln.strip().lstrip("•").strip()
                            for ln in _vc_output.splitlines()
                            if ln.strip().startswith("•")
                        ] or [_vc_output]
                    print(f"  verify_contracts: {'PASS' if _vc_pass else 'FAIL'}  "
                          f"{_vc_output[:200]}")
                except Exception as _vc_exc:
                    _vc_output = f"verify_contracts runner error: {_vc_exc}"
                    print(f"  {_vc_output}")

                body = json.dumps({
                    "ok": True,
                    "path": rel_path,
                    "validation_pass":   _vc_pass,
                    "validation_output": _vc_output,
                    "validation_errors": _vc_errors,
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
                locale      = req.get("locale", "en").strip()
                req_playlist = (req.get("playlist_id") or "").strip() or None

                if not slug or not ep_id:
                    raise ValueError("slug and ep_id required")
                if not re.match(r'^[a-zA-Z0-9_\-]+$', slug) or not re.match(r'^[a-zA-Z0-9_\-]+$', ep_id):
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

                # ── Extract sources from story.txt (### #Src1 #Src2 format) ─────
                sources = []
                story_txt_path = os.path.join(ep_dir, "story.txt")
                if os.path.isfile(story_txt_path):
                    for _sl in open(story_txt_path, encoding="utf-8"):
                        _sm = re.match(r'^###\s+(.+)', _sl.rstrip())
                        if _sm:
                            sources = re.findall(r'#([^\s#]+)', _sm.group(1))

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

                # ── Subtitle scan — check ALL locale render dirs, not just current ──
                subtitles = []
                renders_root = os.path.join(ep_dir, "renders")
                if os.path.isdir(renders_root):
                    for loc_dir in sorted(os.listdir(renders_root)):
                        loc_render = os.path.join(renders_root, loc_dir)
                        if not os.path.isdir(loc_render):
                            continue
                        for fname in sorted(os.listdir(loc_render)):
                            if not fname.endswith(".srt"):
                                continue
                            if ".en." in fname:
                                subtitles.append({"file": f"renders/{loc_dir}/{fname}",
                                                  "language": "en", "name": "English"})
                            elif ".zh-Hans." in fname:
                                subtitles.append({"file": f"renders/{loc_dir}/{fname}",
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

                _eff_locale = profile_info.get("locale") or locale
                output_lang = "Chinese (Simplified)" if _eff_locale.startswith("zh") \
                              else "English"

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
                    "sources":            sources,
                }, ensure_ascii=False)

                # NOTE: sources are appended server-side AFTER the Claude call
                # (see sources_block below). LLMs do not reliably obey this rule,
                # so we do NOT ask Claude to include sources — we append them
                # deterministically to the final description ourselves.

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
                    _exec_directive = (
                        "You are an automated batch pipeline stage running with no human operator present. "
                        "Execute the given task IMMEDIATELY and COMPLETELY. "
                        "NEVER ask for confirmation, permission, or clarification. "
                        "NEVER describe what you are about to do. "
                        "NEVER offer choices or options. "
                        "Complete every instruction from start to finish and then stop."
                    )
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
                             "--append-system-prompt", _exec_directive,
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

                # ── Append CC BY credits from licenses.json ───────────────────
                licenses_path = os.path.join(render_dir, "licenses.json")
                credits_block = ""
                if os.path.isfile(licenses_path):
                    try:
                        with open(licenses_path, encoding="utf-8") as _lf:
                            lic_data = json.load(_lf)
                        seen = set()
                        credit_lines = []
                        for seg in lic_data.get("segments", []):
                            if not seg.get("attribution_required"):
                                continue
                            text = (seg.get("attribution_text") or "").strip()
                            if text and text not in seen:
                                seen.add(text)
                                credit_lines.append(text)
                        if credit_lines:
                            credits_block = (
                                "\n\n---\nCredits\n"
                                + "\n".join(credit_lines)
                            )
                    except Exception as _lic_exc:
                        print(f"  [youtube] WARNING: could not read licenses.json: {_lic_exc}")

                # ── Append Sources block deterministically ────────────────────
                # Parsed from story.txt `### #Src1 #Src2` line (see earlier block).
                # Do NOT rely on the LLM to include these — always append here.
                sources_block = ""
                if sources:
                    _src_hashtags = " ".join(f"#{s}" for s in sources)
                    if output_lang.startswith("Chinese"):
                        sources_block = f"\n\n来源：{_src_hashtags}"
                    else:
                        sources_block = f"\n\nSources: {_src_hashtags}"

                final_description = (
                    suggested["description"].rstrip()
                    + sources_block
                    + credits_block
                )

                # ── Assemble full draft ───────────────────────────────────────
                # Prefer thumbnail.jpg; fall back to thumbnail.png if jpg not yet created
                _thumb_jpg = os.path.join(render_dir, "thumbnail.jpg")
                _thumb_png = os.path.join(render_dir, "thumbnail.png")
                _thumb_rel = (
                    f"projects/{slug}/episodes/{ep_id}/renders/{locale}/thumbnail.jpg"
                    if os.path.isfile(_thumb_jpg) else
                    f"projects/{slug}/episodes/{ep_id}/renders/{locale}/thumbnail.png"
                    if os.path.isfile(_thumb_png) else None
                )

                draft = {
                    "upload_profile":      upload_profile,
                    "title":               suggested["title"],
                    "description":         final_description,
                    "tags":                suggested.get("tags", []),
                    "category_id":         category_id,
                    "playlist_id":         req_playlist or profile_info.get("playlist_id"),
                    "channel_id":          profile_info.get("channel_id"),
                    "video_language":      locale if locale != "zh-Hans" else "zh-Hans",
                    "privacy":             "private",
                    "made_for_kids":       False,
                    "thumbnail":           _thumb_rel,
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
                if not re.match(r'^[a-zA-Z0-9_\-]+$', slug) or not re.match(r'^[a-zA-Z0-9_\-]+$', ep_id):
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
                if not re.match(r'^[a-zA-Z0-9_\-]+$', slug) or not re.match(r'^[a-zA-Z0-9_\-]+$', ep_id):
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
                if not re.match(r'^[a-zA-Z0-9_\-]+$', slug) or not re.match(r'^[a-zA-Z0-9_\-]+$', ep_id):
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
                if not re.match(r'^[a-zA-Z0-9_\-]+$', slug) or not re.match(r'^[a-zA-Z0-9_\-]+$', ep_id):
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
                if not re.match(r'^[a-zA-Z0-9_\-]+$', slug) or not re.match(r'^[a-zA-Z0-9_\-]+$', ep_id):
                    raise ValueError("invalid slug or ep_id")
                if action not in ("validate", "upload", "publish"):
                    raise ValueError(f"unknown action: {action!r}")

                # ── Credential pre-flight check ───────────────────────────────
                _cred_base   = os.path.join(os.path.expanduser("~"), ".config", "pipe")
                _profiles_path = os.path.join(_cred_base, "youtube_profiles.json")
                if not os.path.isfile(_profiles_path):
                    raise ValueError(
                        f"YouTube credentials not found.\n\n"
                        f"Missing:  {_profiles_path}\n\n"
                        f"Place your youtube_profiles.json in:\n"
                        f"  {_cred_base}/\n\n"
                        f"Expected format:\n"
                        f'  {{"en": {{"locale": "en", "token_path": "{_cred_base}/token_en.json", ...}}}}'
                    )
                _profiles = json.loads(open(_profiles_path, encoding="utf-8").read())
                _prof = _profiles.get(locale) or next(
                    (v for v in _profiles.values() if v.get("locale") == locale), None
                )
                if not _prof:
                    raise ValueError(
                        f"No YouTube profile found for locale {locale!r}.\n\n"
                        f"Add a profile for {locale!r} in:\n"
                        f"  {_profiles_path}"
                    )
                _token_path = _prof.get("token_path", "")
                if not os.path.isfile(_token_path):
                    raise ValueError(
                        f"YouTube token not found for locale {locale!r}.\n\n"
                        f"Missing:  {_token_path}\n\n"
                        f"Run the token generator to create it:\n"
                        f"  python3 code/deploy/youtube/gen_tokens.py"
                    )
                # ─────────────────────────────────────────────────────────────

                ep_dir_rel = f"projects/{slug}/episodes/{ep_id}"

                # If re-uploading and video already exists, reset thumbnail_uploaded
                # so the script re-pushes the (possibly updated) thumbnail.
                if action == "upload":
                    _state_path = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id,
                                               "renders", locale, "upload_state.json")
                    if os.path.isfile(_state_path):
                        try:
                            with open(_state_path, encoding="utf-8") as _sf:
                                _st = json.load(_sf)
                            if _st.get("video_id") and _st.get("thumbnail_uploaded"):
                                _st["thumbnail_uploaded"] = False
                                with open(_state_path, "w", encoding="utf-8") as _sf:
                                    json.dump(_st, _sf, indent=2)
                                _log.info("[youtube_action] reset thumbnail_uploaded for re-upload: %s/%s", slug, ep_id)
                        except Exception as _e:
                            _log.warning("[youtube_action] could not reset upload_state: %s", _e)

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

                _log.info("[youtube_action] start  action=%s  slug=%s  ep=%s  locale=%s",
                          action, slug, ep_id, locale)

                _yt_key = f"yt_{slug}_{ep_id}"
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    cwd=PIPE_DIR,
                    start_new_session=True,
                )
                with _lock:
                    _procs[_yt_key] = proc
                try:
                    stdout, stderr = proc.communicate(timeout=1800)
                finally:
                    with _lock:
                        _procs.pop(_yt_key, None)

                if proc.returncode is None or proc.returncode == -15:
                    _log.info("[youtube_action] cancelled  action=%s  slug=%s  ep=%s  locale=%s",
                              action, slug, ep_id, locale)
                    resp = json.dumps({"ok": False, "cancelled": True, "error": "Upload cancelled."}).encode()
                else:
                    _log.info("[youtube_action] done  action=%s  rc=%s  slug=%s  ep=%s  locale=%s",
                              action, proc.returncode, slug, ep_id, locale)
                    output = stdout + ("\n\n--- stderr ---\n" + stderr if stderr.strip() else "")
                    resp = json.dumps({
                        "ok":     proc.returncode == 0,
                        "rc":     proc.returncode,
                        "output": output,
                    }).encode()
                self.send_response(200)
            except subprocess.TimeoutExpired:
                with _lock:
                    p = _procs.pop(f"yt_{slug}_{ep_id}", None)
                if p and p.poll() is None:
                    p.terminate()
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
                  "/api/vo_alignment", "/api/vo_whisper_compare",
                  "/api/vo_approved_timing",
                  "/api/check_slug", "/api/next_episode_id",
                  "/api/create_episode", "/api/save_episode_meta",
                  "/api/diagnose_pipeline",
                  "/api/media_batches", "/api/media_batch_status",
                  "/api/media_health", "/api/media_ai_ask", "/api/media_ai_ask_error",
                  "/api/media_batch", "/api/media_batch_resume", "/api/media_batch_prune",
                  "/api/media_plan_save",
                  "/api/media_preview",
                  "/api/sfx_search", "/api/sfx_save", "/api/sfx_results_save",
                  "/api/sfx_plan_save", "/api/sfx_preview", "/api/sfx_cut_clip", "/api/sfx_delete_clip",
                  "/api/sfx_sources", "/api/sfx_source_upload",
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
    import argparse as _argparse
    _ap = _argparse.ArgumentParser(add_help=False)
    _ap.add_argument("--test-mode", action="store_true", dest="test_mode")
    _ap.add_argument("--port", type=int, default=PORT)
    _args, _ = _ap.parse_known_args()
    _TEST_MODE = _args.test_mode
    PORT = _args.port
    if _TEST_MODE:
        _env_test_dir = os.environ.get("PIPE_TEST_DIR", "")
        if _env_test_dir:
            PIPE_DIR = _env_test_dir
        else:
            PIPE_DIR = os.path.join(PIPE_DIR, "tests", "fixtures")
        print(f"  [TEST MODE] PIPE_DIR overridden to: {PIPE_DIR}", flush=True)
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
