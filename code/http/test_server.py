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
from urllib.parse import parse_qs, urlparse, unquote_plus
import urllib.request as _urllib_req

PORT     = 8000
PIPE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (pipe/)

# ── VC editor config (config.json) ────────────────────────────────────────────
_vc_config = {}
_vc_config_path = os.path.join(os.path.dirname(__file__), "config.json")
if os.path.isfile(_vc_config_path):
    with open(_vc_config_path, encoding="utf-8") as f:
        _vc_config = json.load(f)

# ── Running-process registry (so Stop button can kill it) ──────────────────────
_lock  = threading.Lock()
_procs = {}   # client_addr → subprocess.Popen

# ── Azure TTS preview — throttle state + lazy imports ─────────────────────────
_tts_lock             = threading.Lock()
_tts_last_call: float = 0.0               # monotonic timestamp
TTS_MIN_INTERVAL      = 3.5               # seconds (F0: 20 req/60 s → safe rate)
TTS_RETRY_BACKOFF     = [5, 10, 20]       # seconds; sleep before each 429 retry
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


def _tts_throttled_call(synth, ssml: str):
    """Call Azure TTS with F0-safe throttling (3.5 s min gap) and 429 retry."""
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
    --bg:      #0d0d10;
    --surface: #16161d;
    --border:  #2a2a38;
    --gold:    #c9a84c;
    --green:   #3ecf6e;
    --red:     #e05c5c;
    --blue:    #5b9cf6;
    --text:    #dde1ec;
    --dim:     #777;
    --mono:    "SFMono-Regular", Consolas, "Liberation Mono", monospace;
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
    background: #11111a;
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
    border: 1px solid var(--border); background: #ffffff08; color: var(--dim);
    transition: all .2s;
  }
  #status-badge.running { background:#3ecf6e18; border-color:#3ecf6e44; color:var(--green); }
  #status-badge.error   { background:#e05c5c18; border-color:#e05c5c44; color:var(--red);   }
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
    color: var(--blue); background: #5b9cf614;
    border: 1px solid #5b9cf630; border-radius: 4px;
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
  #story::placeholder { color: #4a4a5a; }

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
  #btn-run   { background: var(--gold); color: #0d0d10; }
  #btn-stop  { background: var(--red);  color: #fff; display: none; }
  #btn-clear { background: #ffffff12; color: var(--dim); border: 1px solid var(--border); }

  /* ── Stage progress indicator ── */
  #stage-progress {
    flex-shrink: 0;
    background: #c9a84c0a;
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
    word-break: break-word; color: #c8d0e0;
  }
  #output .sys  { color: var(--dim);   font-style: italic; }
  #output .err  { color: var(--red);   }
  #output .done { color: var(--green); font-style: italic; }
  #output .ts   { color: #4a7a9a; font-style: italic; font-size: 0.9em; }

  /* ── Spinner ── */
  @keyframes spin { to { transform: rotate(360deg); } }
  .spinner {
    display: inline-block; width: 10px; height: 10px;
    border: 2px solid #3ecf6e44; border-top-color: var(--green);
    border-radius: 50%; animation: spin .7s linear infinite; vertical-align: middle;
  }

  /* ── Scrollbar ── */
  #output::-webkit-scrollbar { width: 6px; }
  #output::-webkit-scrollbar-track { background: transparent; }
  #output::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  /* ── Confirm modal ── */
  #modal-overlay {
    display: none; position: fixed; inset: 0;
    background: #00000099; z-index: 100;
    align-items: center; justify-content: center;
  }
  #modal-overlay.visible { display: flex; }
  #modal-box {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 28px 32px; max-width: 430px; width: 90%;
    box-shadow: 0 24px 64px #000c;
  }
  #modal-box h2 { color: var(--gold); font-size: 1rem; margin-bottom: 10px; }
  #modal-box p  { color: var(--text); font-size: 0.85em; line-height: 1.65; margin-bottom: 14px; }
  #modal-path {
    font-family: var(--mono); color: var(--blue); font-size: 0.80em;
    background: #5b9cf614; border: 1px solid #5b9cf630; border-radius: 4px;
    padding: 4px 10px; display: inline-block; margin-bottom: 18px;
    word-break: break-all;
  }
  .modal-note { color: var(--dim) !important; font-size: 0.78em !important; margin-top: -8px; }
  .modal-btns { display: flex; gap: 10px; justify-content: flex-end; margin-top: 4px; }
  #btn-modal-yes { background: var(--red);   color: #fff; }
  #btn-modal-no  { background: #ffffff12; color: var(--dim);
                   border: 1px solid var(--border); }

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
    background: #ffffff08; border: 1px solid var(--border);
    border-radius: 8px; padding: 3px;
  }
  .tab {
    background: transparent; color: var(--dim);
    border: none; border-radius: 6px;
    font-size: 0.76em; font-weight: 700; letter-spacing: .04em;
    padding: 5px 14px; cursor: pointer;
    transition: background .15s, color .15s, box-shadow .15s;
  }
  .tab:hover  { color: var(--text); background: #ffffff0c; }
  .tab.active {
    background: #ffffff18; color: var(--text);
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
    background: var(--gold); color: #0d0d10; border: none;
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
  .media-sel-badge {
    position: absolute; top: 4px; right: 4px;
    background: var(--gold); color: #0d0d10; font-size: 0.66em; font-weight: 700;
    border-radius: 3px; padding: 2px 5px;
    pointer-events: none; display: none;
  }
  .media-thumb.selected .media-sel-badge { display: block; }
  .media-footer {
    flex-shrink: 0; display: flex; align-items: center; gap: 12px; padding-top: 2px;
  }
  #media-btn-confirm {
    background: var(--green); color: #0d0d10; border: none;
    border-radius: 6px; font-size: 0.85em; font-weight: 700;
    padding: 8px 20px; cursor: pointer; transition: opacity .15s;
  }
  #media-btn-confirm:hover    { opacity: 0.85; }
  #media-btn-confirm:disabled { opacity: 0.4; cursor: not-allowed; }
  #media-btn-reset {
    background: #ffffff10; color: var(--dim);
    border: 1px solid var(--border); border-radius: 6px;
    font-size: 0.82em; padding: 7px 14px; cursor: pointer;
    transition: background .15s, color .15s;
  }
  #media-btn-reset:hover { background: #ffffff1c; color: var(--text); }
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
  /* ── Multi-segment shot rows ── */
  .media-shot-bar {
    height: 6px; flex: 0 0 120px; border-radius: 3px;
    background: #ffffff12; overflow: hidden;
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
  .media-shot-row:hover { background: #ffffff08; }
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
  .music-cand-table td { padding: 5px 8px; border-bottom: 1px solid #ffffff08; }
  .music-cand-table tr:hover { background: #ffffff08; }
  .music-cand-selected { background: #5b9cf612 !important; }
  .music-override-table { width: 100%; border-collapse: collapse; font-size: 0.80em; }
  .music-override-table th {
    text-align: left; padding: 6px 8px; font-weight: 600;
    color: var(--dim); border-bottom: 1px solid var(--border);
    font-size: 0.78em; text-transform: uppercase; letter-spacing: 0.04em;
  }
  .music-override-table td { padding: 4px 8px; border-bottom: 1px solid #ffffff08; }
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
  .music-src-row { padding: 8px 0; border-bottom: 1px solid #ffffff08; }
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
    background: #ffffff10; color: var(--dim); border: 1px solid var(--border);
    border-radius: 4px;
  }
  .music-src-controls button:hover { background: #ffffff18; color: var(--text); }
  .music-src-controls button.active { background: var(--gold); color: #0d0d10; }
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
    padding: 5px 10px; background: #ffffff08;
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
    background: var(--gold); color: #0d0d10; border: none;
    border-radius: 6px; font-size: 0.82em; font-weight: 700;
    padding: 6px 16px; cursor: pointer; transition: opacity .15s;
  }
  .music-footer button:hover { opacity: 0.85; }
  .music-footer button:disabled { opacity: 0.4; cursor: not-allowed; }
  #music-confirm-msg { font-size: 0.80em; color: var(--dim); flex: 1; }
  .music-btn-secondary {
    background: #ffffff10 !important; color: var(--dim) !important;
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
    background: #ffffff10; color: var(--dim);
    border: 1px solid var(--border); border-radius: 6px;
    font-size: 0.76em; padding: 5px 12px; cursor: pointer;
    transition: background .15s, color .15s;
  }
  #btn-refresh:hover { background: #ffffff1c; color: var(--text); }
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
    background: #ffffff08; padding: 8px 14px;
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
    background: #5b9cf614; color: var(--blue); border-color: #5b9cf650;
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
    background: #ffffff10; color: var(--dim); border: 1px solid var(--border);
    border-radius: 4px; font-size: 1.1em; padding: 2px 10px; cursor: pointer;
    font-weight: 600; letter-spacing: 0; text-transform: none;
    transition: background .15s, color .15s;
  }
  .btn-review-tab.active {
    background: #5b9cf620; color: var(--blue); border-color: #5b9cf650;
  }
  .review-locale-tabs { display: flex; gap: 4px; margin-left: auto; }
  .btn-locale-tab {
    background: #ffffff10; color: var(--dim); border: 1px solid var(--border);
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
    background: #ffffff08; color: #6a7a9a;
    border: 1px solid var(--border); border-radius: 4px;
    font-size: 0.76em; padding: 2px 9px; cursor: pointer;
    font-family: var(--mono); transition: background .15s, color .15s;
  }
  .btn-substep:hover { background: #ffffff18; color: var(--text); }
  .vc-style-chip-UNUSED {   /* kept as placeholder; vc viewer removed from Pipeline tab */
    display: inline-block; background: #ffffff08;
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
    background: #ffffff08; color: var(--dim); border: 1px solid var(--border);
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
    background: #5b9cf614; color: var(--blue); border: 1px solid #5b9cf650;
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
    background: #ffffff08; padding: 7px 14px;
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
    background: #5b9cf614; color: var(--blue); border: 1px solid #5b9cf650;
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
  #btn-vc-continue { background: #5b9cf614; color: var(--blue); border-color: #5b9cf650; }

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
    <button class="tab"        data-tab="music"    onclick="switchTab('music')"   >🎵 Music</button>
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
placeholder="Story title  : The Pharaoh Who Defied Death
Project slug : the-pharaoh-who-defied-death
Episode num  : 01
Episode id   : s01e01
Locales      : en, zh-Hans
Genre        : Ancient Egyptian Epic / Mystery / Supernatural / Political Drama
Direction    : …"></textarea>
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

  <!-- hidden: stage range always 0–10; split into 0+1–N handled in runPrompt() -->
  <input type="hidden" id="prompt" value="0  10">
  <!-- ── Run options + buttons ── -->
  <div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
    <label id="label-no-music"
           style="display:flex; align-items:center; gap:6px; cursor:pointer;
                  font-size:0.82em; font-family:var(--mono); color:var(--dim);
                  user-select:none;"
           title="Skip background music — render VO + SFX only">
      <input type="checkbox" id="chk-no-music" onchange="toggleMusicMode()" checked
             style="width:14px; height:14px; cursor:pointer; accent-color:var(--gold);">
      🔇 No Music
    </label>
    <label style="display:flex; align-items:center; gap:6px; cursor:pointer;
                  font-size:0.82em; font-family:var(--mono); color:var(--dim);
                  user-select:none;"
           title="Delete cached WAVs, images, renders and manifests before each Stage 10 run — ensures a clean rebuild from scratch">
      <input type="checkbox" id="chk-purge-assets" onchange="togglePurgeMode()" checked
             style="width:14px; height:14px; cursor:pointer; accent-color:#e06c75;">
      🗑 Purge Cache
    </label>
    <button id="btn-prepare" onclick="runPrepare()" title="Analyse story and detect project name, format and genre">⚙ Prepare</button>
    <div class="btn-group" style="margin:0;">
      <button id="btn-run"   onclick="runPrompt()">▶ Run</button>
      <button id="btn-stop"  onclick="stopRun()">■ Stop</button>
      <button id="btn-clear" onclick="clearOutput()">✕ Clear</button>
    </div>
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
          <option value="episodic">Episodic (default)</option>
          <option value="continuous_narration">Continuous Narration</option>
          <option value="illustrated_narration">Illustrated Narration</option>
          <option value="documentary">Documentary / Explainer</option>
          <option value="monologue">Monologue / First-Person</option>
          <option value="ssml_narration">SSML Narration (authored)</option>
        </select>
        <span id="badge-format" class="info-badge" style="display:none"></span>
      </div>
      <div id="format-hint" class="format-hint"></div>
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
        <option value="episodic">Episodic (default)</option>
        <option value="continuous_narration">Continuous Narration</option>
        <option value="illustrated_narration">Illustrated Narration</option>
        <option value="documentary">Documentary / Explainer</option>
        <option value="monologue">Monologue / First-Person</option>
        <option value="ssml_narration">SSML Narration (authored)</option>
      </select>
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

<!-- ── Music panel ── -->
<div id="panel-music">
  <!-- toolbar row -->
  <div class="music-toolbar">
    <div class="section-label" style="margin-bottom:0">Music Review</div>
    <select id="music-ep-select" onchange="onMusicEpChange()">
      <option value="">— select episode —</option>
    </select>
    <button id="music-btn-review"  onclick="musicGenerateReview()" disabled
            style="background:#ffffff10;color:var(--dim);border:1px solid var(--border);border-radius:6px;font-size:0.80em;padding:5px 14px;cursor:pointer">🎵 Generate Music Review</button>
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

<!-- ── Pipeline panel ── -->
<div id="panel-pipeline">
  <!-- toolbar: episode selector -->
  <div class="pipe-toolbar">
    <div class="section-label" style="margin-bottom:0">Episode</div>
    <select id="pipe-ep-select" onchange="onPipeEpChange()">
      <option value="">— select episode —</option>
    </select>
    <button style="background:#ffffff10;color:var(--dim);border:1px solid var(--border);border-radius:6px;font-size:0.76em;padding:5px 12px;cursor:pointer" onclick="refreshPipeline()">↺ Refresh</button>
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
  let purgeAssets = true;   // true = purge cached WAVs/images/renders before each run (default ON)

  let _preparedMeta  = null;   // result from last /api/infer_story_meta call
  let _preparedEpId  = null;   // next episode ID fetched during Prepare (e.g. "s01e01")
  let _episodeCreated = false; // true after Create Episode completes
  let _runProjectList = [];    // cached project list for run-tab dropdowns
  let _selectedFormat  = 'episodic';
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
    if (pipeStepEs) { pipeStepEs.close(); pipeStepEs = null; }
    fetch('/stop', { method: 'POST' }).catch(() => {});
    appendLine('[ Stopped by user ]', 'sys');
    pipeRunning = null;
    hideStageProgress();
    setStatus('idle');
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
      appendLine(text, '');

      // ── Stage start  (run.sh banner:  "  STAGE N/10  —  label")
      const startM = text.match(/^\s{2}STAGE (\d+)\/\d+\s+[—\-]{1,2}\s+(.+)/);
      if (startM) {
        const n     = parseInt(startM[1]);
        const label = startM[2].trim();
        stageStartMs[n] = Date.now();
        appendLine(`  ⏱  started  ${fmtNow()}`, 'ts');
        updateStageProgress(n, label);
      }

      // ── Stage complete  (run.sh:  "✓ Stage N complete  →  log: …")
      const doneM = text.match(/^✓ Stage (\d+) complete/);
      if (doneM) {
        const n = parseInt(doneM[1]);
        const elapsed = stageStartMs[n] != null
          ? `  elapsed ${fmtElapsed(Date.now() - stageStartMs[n])}` : '';
        appendLine(`  ⏱  finished ${fmtNow()}${elapsed}`, 'ts');
        markStageDone(n);
        insertReviewButtons(n);
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
    noMusic = chk ? chk.checked : !noMusic;
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
    document.getElementById('panel-music').style.display    = name === 'music'    ? 'flex' : 'none';

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

  // ── Media tab ────────────────────────────────────────────────────────────────

  let _mediaSlug           = null;
  let _mediaEpId           = null;
  let _mediaBatchId        = null;
  let _mediaPollTimer      = null;
  let _mediaResults        = null;   // full items dict from last completed batch
  let _mediaItemIds        = [];     // ordered item IDs for confirm iteration
  let _mediaRecommendedSeq = null;   // recommended_sequence from batch response
  // selections: { item_id: { type:'image'|'video', url, path, score } }
  //   Per-shot:  { item_id: { per_shot: { shot_id: { media_type, url, path, score } } } }
  let _mediaSelections = {};
  let _mediaShotMap = null;  // { bg_id: [shot_id, ...] } or null if ShotList unavailable
  let _mediaShotDur = null;  // { shot_id: duration_sec } — flat lookup for shot durations
  let _mediaActiveShot = {}; // { itemId: shot_id } — which shot row is active (target) per card
  var _mediaPlayingVid = null; // currently playing video element (limit 1 active stream)

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
  function initMediaTab() {
    // Populate episode selector from list_projects (same as Pipeline tab)
    var needsSync = document.getElementById('media-ep-select').options.length <= 1;
    if (!needsSync) {
      // Already populated — but still sync from Run tab if media has no selection
      _mediaSyncFromRunTab();
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
      // After populating, sync from Run tab's current project/episode
      _mediaSyncFromRunTab();
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

  function onMediaEpChange() {
    const v = document.getElementById('media-ep-select').value;
    if (!v) { _mediaSlug = null; _mediaEpId = null; return; }
    [_mediaSlug, _mediaEpId] = v.split('|');
    document.getElementById('media-btn-search').disabled = false;
    _mediaSetStatus('Episode selected. Click Search Media to begin.');
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
  async function _mediaLoadShotMap() {
    if (!_mediaSlug || !_mediaEpId) { _mediaShotMap = null; _mediaShotDur = null; return; }
    try {
      const r = await fetch('/api/episode_file?slug=' + encodeURIComponent(_mediaSlug)
          + '&ep_id=' + encodeURIComponent(_mediaEpId)
          + '&file=ShotList.json');
      if (!r.ok) { _mediaShotMap = null; _mediaShotDur = null; return; }
      const d = await r.json();
      const shots = d.shots || [];
      const map = {};
      const durMap = {};
      shots.forEach(s => {
        const bg = s.background_id;
        if (!bg) return;
        if (!map[bg]) map[bg] = [];
        map[bg].push(s.shot_id);
        durMap[s.shot_id] = s.duration_sec || 0;
      });
      _mediaShotMap = map;
      _mediaShotDur = durMap;
    } catch (_) { _mediaShotMap = null; _mediaShotDur = null; }
  }

  // ── Start a new search batch ──
  async function mediaStartSearch() {
    if (!_mediaSlug || !_mediaEpId) return;
    const serverUrl = _mediaServerUrl();
    _mediaSetStatus('Submitting batch …', true);
    document.getElementById('media-btn-search').disabled = true;
    document.getElementById('media-footer').style.display = 'none';
    document.getElementById('media-body').innerHTML = '';
    _mediaSelections = {};
    _mediaBatchId    = null;
    _mediaResults    = null;

    try {
      const r = await fetch('/api/media_batch', {
        method:  'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ slug: _mediaSlug, ep_id: _mediaEpId,
                               server_url: serverUrl }),
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
      } else if (d.status === 'failed') {
        clearInterval(_mediaPollTimer); _mediaPollTimer = null;
        _mediaSetStatus('Batch failed: ' + (d.error || 'unknown'), false);
        document.getElementById('media-btn-search').disabled = false;
      } else {
        _mediaSetStatus((d.progress || d.status) + ' …', true);
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

  // ── Render results grid ──
  function _mediaRenderResults(items) {
    const body = document.getElementById('media-body');
    body.innerHTML = '';
    const d_topN = _mediaResults?._topN || 5;
    _mediaItemIds = Object.keys(items);
    _mediaSelections = {};

    _mediaItemIds.forEach(itemId => {
      const item   = items[itemId];
      const card   = document.createElement('div');
      card.className = 'media-item-card';
      card.id = 'media-card-' + itemId;

      // Card header: show shot info when _mediaShotMap is available
      const hdr = document.createElement('div');
      hdr.className = 'media-item-header';
      const shotIds = _mediaShotMap ? (_mediaShotMap[itemId] || []) : [];
      if (shotIds.length > 0) {
        hdr.textContent = itemId + ' \u2014 ' + shotIds.length + ' shot' + (shotIds.length > 1 ? 's' : '')
            + ': ' + shotIds.join(', ');
      } else {
        hdr.textContent = itemId;
      }
      card.appendChild(hdr);

      const prompt = document.createElement('div');
      prompt.className = 'media-item-prompt';
      prompt.textContent = item.search_prompt || '';
      card.appendChild(prompt);

      // Images section — lazy-scroll with pagination
      const imgs = item.images || [];
      if (imgs.length) {
        const lbl = document.createElement('div');
        lbl.className = 'media-section-label';
        lbl.textContent = 'Images (' + (item.total_images || imgs.length) + ' found)';
        card.appendChild(lbl);
        const row = document.createElement('div');
        row.className = 'media-thumb-row';
        _mediaAppendPage(row, imgs, itemId, 'image', 0, _MEDIA_PAGE, d_topN);
        card.appendChild(row);
        if (imgs.length > _MEDIA_PAGE) _mediaObserveSentinel(row, imgs, itemId, 'image', d_topN);
      }

      // Videos section — lazy-scroll with pagination
      const vids = item.videos || [];
      if (vids.length) {
        const lbl = document.createElement('div');
        lbl.className = 'media-section-label';
        lbl.textContent = 'Videos (' + (item.total_videos || vids.length) + ' found)';
        card.appendChild(lbl);
        const row = document.createElement('div');
        row.className = 'media-thumb-row';
        _mediaAppendPage(row, vids, itemId, 'video', 0, _MEDIA_PAGE, d_topN);
        card.appendChild(row);
        if (vids.length > _MEDIA_PAGE) _mediaObserveSentinel(row, vids, itemId, 'video', d_topN);
      }

      if (!imgs.length && !vids.length) {
        const empty = document.createElement('div');
        empty.className = 'media-empty';
        empty.textContent = item.error || 'No results found.';
        card.appendChild(empty);
      }

      // Per-shot assignment rows (only when ShotList is available and this bg has shots)
      if (_mediaShotMap && shotIds.length > 0) {
        const shotSection = document.createElement('div');
        shotSection.className = 'media-shot-section';
        shotSection.id = 'media-shots-' + itemId;

        // Auto-assign button (only when >1 shot for variety)
        if (shotIds.length > 1) {
          const autoBtn = document.createElement('button');
          autoBtn.className = 'media-btn-auto-assign';
          autoBtn.textContent = '\u26A1 Auto-assign';
          autoBtn.onclick = function() { mediaAutoAssign(itemId); };
          shotSection.appendChild(autoBtn);
        }

        shotIds.forEach(function(sid, idx) {
          const row = document.createElement('div');
          row.className = 'media-shot-row' + (idx === 0 ? ' media-shot-active' : '');
          row.id = 'media-shot-row-' + itemId + '-' + sid;
          const shotDur = (_mediaShotDur && _mediaShotDur[sid]) || 0;
          const durLabel = shotDur > 0 ? ' (' + shotDur.toFixed(1) + 's)' : '';
          row.innerHTML = '<span class="media-shot-label">' + sid + durLabel + ':</span>'
              + '<div class="media-shot-bar"><div class="media-shot-bar-fill" style="width:0%"></div></div>'
              + '<div class="media-shot-segments"></div>'
              + '<span class="media-shot-gap">\u2014 not assigned \u2014</span>';
          // Click row to make it the active target for thumbnail selections
          (function(iid, s) {
            row.addEventListener('click', function(e) {
              // Don't activate if clicking the remove button inside a segment
              if (e.target.closest('.media-seg-remove')) return;
              mediaSetActiveShot(iid, s);
            });
          })(itemId, sid);
          shotSection.appendChild(row);
        });
        // First shot is active by default
        if (shotIds.length > 0) _mediaActiveShot[itemId] = shotIds[0];
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

    // Store original URL and duration as data attributes
    wrap.dataset.url = entry.url || '';
    if (typeof entry.duration_sec === 'number') {
      wrap.dataset.durationSec = entry.duration_sec;
    }
    wrap.addEventListener('click', () => mediaSelect(itemId, type, entry, wrap));
    return wrap;
  }

  // ── Segment total duration helper ──
  function _mediaSegmentTotal(shotEntry) {
    if (!shotEntry || !shotEntry.segments) return 0;
    return shotEntry.segments.reduce(function(sum, seg) {
      return sum + (seg.hold_sec || seg.duration_sec || 0);
    }, 0);
  }

  // ── Rebalance flexible segments to share the remaining gap equally ──
  // "Fixed" = video with a duration_sec value.
  // "Flexible" = image (or video without duration).
  function _mediaRebalanceImages(shotId, segs) {
    var shotDur = (_mediaShotDur && _mediaShotDur[shotId]) || 0;
    if (shotDur <= 0 || segs.length === 0) return;
    var fixedTotal = 0;
    var flexIdxs = [];
    segs.forEach(function(seg, i) {
      if (seg.media_type === 'video' && seg.duration_sec) {
        fixedTotal += seg.duration_sec;
      } else {
        flexIdxs.push(i);
      }
    });
    var gap = Math.max(0, shotDur - fixedTotal);
    if (flexIdxs.length === 0) return;
    var per = gap / flexIdxs.length;
    flexIdxs.forEach(function(i) {
      if (segs[i].media_type === 'image') {
        segs[i].hold_sec = per;
      } else {
        segs[i].duration_sec = per;
      }
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
  function mediaSelect(itemId, type, entry, wrapEl) {
    const shotIds = _mediaShotMap ? (_mediaShotMap[itemId] || []) : [];

    if (_mediaShotMap && shotIds.length > 0) {
      // ── Multi-segment stacking mode ──
      if (!_mediaSelections[itemId]) _mediaSelections[itemId] = { per_shot: {} };
      const ps = _mediaSelections[itemId].per_shot;
      const clickUrl = entry.url || '';

      // ── Toggle-off: if this URL is already in any shot, remove it ──
      let removed = false;
      for (const sid of shotIds) {
        if (!ps[sid] || !ps[sid].segments) continue;
        const idx = ps[sid].segments.findIndex(function(s) { return s.url === clickUrl; });
        if (idx !== -1) {
          ps[sid].segments.splice(idx, 1);
          if (ps[sid].segments.length === 0) {
            delete ps[sid];
          } else {
            _mediaRebalanceImages(sid, ps[sid].segments);
          }
          _mediaRenderShotRow(itemId, sid);
          removed = true;
          break;
        }
      }
      if (removed) {
        wrapEl.classList.remove('selected');
        return;
      }

      // ── Target: the user-selected active shot row ──
      var targetShot = _mediaActiveShot[itemId] || shotIds[0];

      // Initialize segments array
      if (!ps[targetShot]) ps[targetShot] = { segments: [] };
      if (!ps[targetShot].segments) {
        // Upgrade v2 single-entry to v3 segments if needed
        const old = ps[targetShot];
        ps[targetShot] = { segments: old.url ? [old] : [] };
      }

      const segs = ps[targetShot].segments;
      const shotDur = (_mediaShotDur && _mediaShotDur[targetShot]) || 0;
      const filled = _mediaSegmentTotal({ segments: segs });
      const remaining = Math.max(0, shotDur - filled);

      // Video: use probed duration (capped to remaining gap)
      // Image: hold for remaining gap
      const segDur = type === 'video'
          ? Math.min(entry.duration_sec || remaining, remaining)
          : remaining;

      segs.push({
        media_type: type,
        url: entry.url || '',
        path: entry.path || '',
        score: entry.score,
        duration_sec: type === 'video' ? segDur : null,
        hold_sec: type === 'image' ? segDur : null,
      });

      // Rebalance images to share the remaining gap equally
      _mediaRebalanceImages(targetShot, segs);

      wrapEl.classList.add('selected');
      _mediaRenderShotRow(itemId, targetShot);
    } else {
      // ── Single-selection mode (no ShotList) ──
      const card = document.getElementById('media-card-' + itemId);
      if (card) {
        card.querySelectorAll('.media-thumb').forEach(t => t.classList.remove('selected'));
        card.querySelectorAll('.media-sel-badge').forEach(b => b.remove());
      }

      // Check if clicking the already-selected item → deselect
      if (_mediaSelections[itemId] && _mediaSelections[itemId].url === (entry.url || '')) {
        delete _mediaSelections[itemId];
        return;
      }

      _mediaSelections[itemId] = { media_type: type, url: entry.url || '',
                                    path: entry.path || '', score: entry.score };
      wrapEl.classList.add('selected');

      const selBadge = document.createElement('span');
      selBadge.className = 'media-sel-badge';
      selBadge.textContent = '\u2714 ' + type;
      wrapEl.appendChild(selBadge);
    }
  }

  // ── Render a per-shot row with segment list and fill bar ──
  function _mediaRenderShotRow(itemId, shotId) {
    const row = document.getElementById('media-shot-row-' + itemId + '-' + shotId);
    if (!row) return;
    const barFill = row.querySelector('.media-shot-bar-fill');
    const segContainer = row.querySelector('.media-shot-segments');
    const gapSpan = row.querySelector('.media-shot-gap');

    const ps = (_mediaSelections[itemId] && _mediaSelections[itemId].per_shot) || {};
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
    if (segContainer) {
      segContainer.innerHTML = '';
      segs.forEach(function(seg, idx) {
        const se = document.createElement('div');
        se.className = 'media-seg-entry';
        const fname = (seg.path || seg.url || '').split('/').pop() || 'media';
        const dur = seg.hold_sec || seg.duration_sec || 0;
        se.innerHTML = '<span class="seg-name">' + fname + '</span>'
            + '<span class="seg-dur">' + dur.toFixed(1) + 's</span>'
            + '<button class="media-seg-remove" onclick="mediaRemoveSegment(\''
            + itemId + '\',\'' + shotId + '\',' + idx + ')">\u2715</button>';
        segContainer.appendChild(se);
      });
    }

    // Gap indicator
    const gap = Math.max(0, shotDur - filled);
    if (gapSpan) {
      if (gap > 0.1) {
        gapSpan.textContent = 'needs ' + gap.toFixed(1) + 's more';
        row.classList.remove('media-shot-filled');
      } else {
        gapSpan.textContent = '\u2713 filled';
        row.classList.add('media-shot-filled');
      }
    }
  }

  // ── Remove a segment from a shot ──
  function mediaRemoveSegment(itemId, shotId, segIdx) {
    if (!_mediaSelections[itemId] || !_mediaSelections[itemId].per_shot) return;
    const shotEntry = _mediaSelections[itemId].per_shot[shotId];
    if (!shotEntry || !shotEntry.segments) return;
    shotEntry.segments.splice(segIdx, 1);
    if (shotEntry.segments.length === 0) {
      delete _mediaSelections[itemId].per_shot[shotId];
    } else {
      // Rebalance surviving images to absorb freed time
      _mediaRebalanceImages(shotId, shotEntry.segments);
    }
    _mediaRenderShotRow(itemId, shotId);
  }

  // ── Clear a single per-shot assignment (legacy + segments) ──
  function mediaClearShot(itemId, shotId) {
    if (!_mediaSelections[itemId] || !_mediaSelections[itemId].per_shot) return;
    delete _mediaSelections[itemId].per_shot[shotId];
    _mediaRenderShotRow(itemId, shotId);
  }

  // ── Auto-assign: stack segments to fill each shot's duration ──
  function mediaAutoAssign(itemId) {
    const shotIds = _mediaShotMap ? (_mediaShotMap[itemId] || []) : [];
    if (shotIds.length === 0) return;
    const item = _mediaResults[itemId];
    if (!item) return;

    // Build ranked candidate list: videos first (by score desc), then images
    const candidates = [];
    const vids = (item.videos || []).slice().sort((a, b) => (b.score || 0) - (a.score || 0));
    const imgs = (item.images || []).slice().sort((a, b) => (b.score || 0) - (a.score || 0));
    vids.forEach(v => candidates.push({ type: 'video', entry: v }));
    imgs.forEach(v => candidates.push({ type: 'image', entry: v }));
    if (candidates.length === 0) return;

    if (!_mediaSelections[itemId]) _mediaSelections[itemId] = { per_shot: {} };
    const ps = _mediaSelections[itemId].per_shot;
    const usedUrls = new Set();
    let candIdx = 0;

    shotIds.forEach(function(sid) {
      const shotDur = (_mediaShotDur && _mediaShotDur[sid]) || 0;
      ps[sid] = { segments: [] };
      let filled = 0;

      // Stack candidates until shot duration is filled
      while (filled < shotDur && candIdx < candidates.length * 2) {
        const idx = candIdx % candidates.length;
        const cand = candidates[idx];
        candIdx++;

        // Skip duplicates within the same shot
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
          // Image: fill remaining gap
          ps[sid].segments.push({
            media_type: 'image', url: cand.entry.url || '',
            path: cand.entry.path || '', score: cand.entry.score,
            duration_sec: null, hold_sec: remaining,
          });
          filled += remaining;
          usedUrls.add(cand.entry.url);
        }
      }

      // If no candidates were added (edge case), add best image for full duration
      if (ps[sid].segments.length === 0 && imgs.length > 0) {
        ps[sid].segments.push({
          media_type: 'image', url: imgs[0].url || '',
          path: imgs[0].path || '', score: imgs[0].score,
          duration_sec: null, hold_sec: shotDur,
        });
        usedUrls.add(imgs[0].url);
      }

      _mediaRenderShotRow(itemId, sid);
    });

    // Visual: mark used thumbs as selected in the grid
    const card = document.getElementById('media-card-' + itemId);
    if (card) {
      card.querySelectorAll('.media-thumb').forEach(t => {
        const url = t.dataset.url;
        if (usedUrls.has(url)) t.classList.add('selected');
        else t.classList.remove('selected');
      });
    }
  }

  // ── Reset all selections ──
  function mediaReset() {
    _mediaSelections = {};
    document.querySelectorAll('.media-thumb.selected').forEach(t => t.classList.remove('selected'));
    document.querySelectorAll('.media-sel-badge').forEach(b => b.remove());
    // Reset per-shot rows (segments mode)
    document.querySelectorAll('.media-shot-bar-fill').forEach(b => { b.style.width = '0%'; });
    document.querySelectorAll('.media-shot-segments').forEach(c => { c.innerHTML = ''; });
    document.querySelectorAll('.media-shot-gap').forEach(g => {
      g.textContent = '\u2014 not assigned \u2014';
    });
    document.querySelectorAll('.media-shot-row').forEach(r => r.classList.remove('media-shot-filled'));
    // Reset active shot to first shot per card
    document.querySelectorAll('.media-shot-row').forEach(r => r.classList.remove('media-shot-active'));
    for (var iid in _mediaActiveShot) {
      var sids = _mediaShotMap ? (_mediaShotMap[iid] || []) : [];
      if (sids.length > 0) {
        _mediaActiveShot[iid] = sids[0];
        var firstRow = document.getElementById('media-shot-row-' + iid + '-' + sids[0]);
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
    for (const [itemId, cand] of Object.entries(_mediaRecommendedSeq)) {
      if (!cand || !cand.url) continue;
      const shotIds = _mediaShotMap ? (_mediaShotMap[itemId] || []) : [];

      if (_mediaShotMap && shotIds.length > 0) {
        // Per-shot mode: distribute ranked candidates across shots
        const item = _mediaResults[itemId];
        if (!item) continue;
        // Build ranked candidate list from full results
        const candidates = [];
        const vids = (item.videos || []).slice().sort((a, b) => (b.score || 0) - (a.score || 0));
        const imgs = (item.images || []).slice().sort((a, b) => (b.score || 0) - (a.score || 0));
        vids.forEach(v => candidates.push({ type: 'video', entry: v }));
        imgs.forEach(v => candidates.push({ type: 'image', entry: v }));
        if (candidates.length === 0) continue;

        if (!_mediaSelections[itemId]) _mediaSelections[itemId] = { per_shot: {} };
        const ps = _mediaSelections[itemId].per_shot;
        const usedUrls = new Set();
        let ci = 0;

        shotIds.forEach(function(sid) {
          const shotDur = (_mediaShotDur && _mediaShotDur[sid]) || 0;
          ps[sid] = { segments: [] };
          let filled = 0;
          // Stack candidates until shot duration is filled
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
          _mediaRenderShotRow(itemId, sid);
        });

        // Mark used thumbs
        const card = document.getElementById('media-card-' + itemId);
        if (card) {
          card.querySelectorAll('.media-thumb').forEach(t => {
            if (usedUrls.has(t.dataset.url)) t.classList.add('selected');
          });
        }
        applied++;
      } else {
        // Single-selection mode (no ShotList)
        const ext = (cand.path || cand.url || '').split('.').pop().toLowerCase();
        const type = ['mp4', 'mov', 'webm', 'mkv'].includes(ext) ? 'video' : 'image';
        _mediaSelections[itemId] = {
          media_type: type,
          url:   cand.url  || '',
          path:  cand.path || '',
          score: cand.score || 0,
        };
        // Visually mark the matching thumb in the grid
        const card = document.getElementById('media-card-' + itemId);
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
    try {
      const r = await fetch(
        '/api/media_batches?slug=' + encodeURIComponent(_mediaSlug)
        + '&ep_id='      + encodeURIComponent(_mediaEpId)
        + '&server_url=' + encodeURIComponent(serverUrl));
      const d = await r.json();
      if (!r.ok || d.error) return;
      const done = (d.batches || []).find(b => b.status === 'done');
      if (!done) return;
      // Load full results for this batch
      _mediaBatchId = done.batch_id;
      const r2 = await fetch(
        '/api/media_batch_status?batch_id=' + encodeURIComponent(_mediaBatchId)
        + '&server_url=' + encodeURIComponent(serverUrl));
      const d2 = await r2.json();
      if (!r2.ok || d2.error || d2.status !== 'done') return;
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
    } catch (_) {}
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
      { n:0, key:'stage_0'  }, { n:1,  key:'stage_1'  }, { n:2,  key:'stage_2'  },
      { n:3, key:'stage_3'  }, { n:4,  key:'stage_4'  }, { n:5,  key:'stage_5'  },
      { n:6, key:'stage_6'  }, { n:7,  key:'stage_7'  }, { n:8,  key:'stage_8'  },
      { n:9, key:'stage_9'  }, { n:10, key:'stage_10' },
    ];
    const stageLabels = [
      'Cast voices & pipeline vars',  'Check story consistency',
      'Write episode direction',       'Write script & dialogue',
      'Break script into shots',       'List required assets',
      'Identify new story facts',      'Update story memory',
      'Translate & adapt locales',     'Finalize assets & render plan',
      'Merge assets & render video',
    ];
    // Sequential done propagation (mirrors renderPipelineStatus)
    let maxDone = -1;
    llmKeys.forEach(({ n, key }) => { if ((stagesMap[key] || {}).done) maxDone = n; });
    for (const { n, key } of llmKeys) {
      const done = n <= maxDone || ((stagesMap[key] || {}).done === true);
      if (!done) {
        return { state: 'action',
                 msg: '▶  Run ' + n + ' → 10  —  Stage ' + n + ': ' + stageLabels[n] };
      }
    }
    return { state: 'done', msg: '✅  All stages complete — episode is ready.' };
  }

  // Returns an alignment-based next-step if Stage 10 is done but VO is misaligned.
  function _alignmentNextStep() {
    if (!_lastAlignmentData || !_lastAlignmentData.locales) return null;
    // Only relevant when Stage 10 is already complete
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
      const r = await fetch('/api/preview_voice', {
        method:  'POST',
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
  let _musicOverrides = {};     // { item_id: {duck_db, fade_sec, ...} }
  let _musicLoopSel   = {};     // { track_stem: {start_sec, duration_sec, mode, crossfade_ms} }
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
        _musicLoopSel  = plan.loop_selections || {};
        _musicOverrides = {};
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
      await fetch('/api/music_plan_save', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ slug: _musicSlug, ep_id: _musicEpId, plan: plan }),
      });
    } catch (_) {}
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
          const currentClipId = ovr.music_clip_id || itemToClipId[origMid] || origMid;

          ovrHtml += '<div class="music-shot-block">'
            // header: shot id + episode time range
            + '<div class="music-shot-hdr" style="border-left:4px solid ' + col + '">'
            + '<span class="music-shot-hdr-id">' + s.shot_id.replace(/^s\d+e\d+_/, '') + '</span>'
            + '<span class="music-shot-hdr-ep">episode&nbsp;' + fmtEp(epStart) + ' – ' + fmtEp(epEnd)
            + '&nbsp;(' + shotDur.toFixed(1) + 's)</span>'
            + '</div>'
            // clip dropdown
            + '<div class="music-shot-clip">'
            + '<select style="width:100%" onchange="_musicSetClipOverride(\'' + origMid + '\',this.value)">';
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
        srcHtml += '<div class="music-src-row">'
          + '<div class="music-src-top">'
          + '<span class="music-src-stem">' + s.stem + '</span>'
          + '<span class="music-src-meta">' + dur + (bpm ? ' · ' + bpm : '') + '</span>'
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

    // ── Generated Clips (auto + user-cut) ──
    if (allClips.length > 0) {
      const clipCard = document.createElement('div');
      clipCard.className = 'music-card';
      let clipHtml = '<div class="music-card-hdr">Generated Clips</div>'
        + '<div class="music-card-sub">Auto-extracted and user-cut clips available for shot assignment.</div>'
        + '<table class="music-cand-table"><thead><tr>'
        + '<th>Clip</th><th>Duration</th><th>Score</th><th style="width:220px">Preview</th>'
        + '</tr></thead><tbody>';
      allClips.forEach(c => {
        const durTxt = c.duration_sec.toFixed(1) + 's';
        const scoreTxt = c.score != null ? c.score.toFixed(3) : (c.origin === 'user' ? 'cut' : '—');
        clipHtml += '<tr>'
          + '<td style="font-family:var(--mono);font-size:0.82em;color:'
          + (c.origin === 'user' ? 'var(--gold)' : 'var(--text)') + '">' + c.clip_id + '</td>'
          + '<td>' + durTxt + '</td>'
          + '<td>' + scoreTxt + '</td>'
          + '<td><audio controls preload="none" style="height:28px;width:210px"'
          + ' src="/serve_media?path=' + encodeURIComponent(c.wavPath) + '"></audio></td>'
          + '</tr>';
      });
      clipHtml += '</tbody></table>';
      clipCard.innerHTML = clipHtml;
      body.appendChild(clipCard);
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
    if (!_musicOverrides[itemId]) _musicOverrides[itemId] = { item_id: itemId };
    // Store full clip_id for UI dropdown matching
    _musicOverrides[itemId].music_clip_id = clipId;
    // Resolve to WAV filename stem via lookup (for backend)
    const info = _musicClipLookup[clipId];
    if (info) {
      _musicOverrides[itemId].music_asset_id = info.wavStem;
    } else {
      // Fallback: parse stem from clip_id
      const m = clipId.match(/^(.+?):(\d+\.?\d*)s-(\d+\.?\d*)s$/);
      _musicOverrides[itemId].music_asset_id = m ? m[1] : clipId;
    }
    // Also store clip timing for apply_music_plan
    const m = clipId.match(/^(.+?):(\d+\.?\d*)s-(\d+\.?\d*)s$/);
    if (m) {
      _musicOverrides[itemId].clip_start_sec = parseFloat(m[2]);
      _musicOverrides[itemId].clip_duration_sec = parseFloat(m[3]) - parseFloat(m[2]);
    }
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
      const r = await fetch('/api/music_plan_save', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ slug: _musicSlug, ep_id: _musicEpId, plan: plan }),
      });
      const d = await r.json();
      if (!r.ok || d.error) throw new Error(d.error || 'save failed');
      document.getElementById('music-confirm-msg').textContent =
        '✔ MusicPlan.json saved → ' + (d.path || 'assets/music/MusicPlan.json')
        + '. Resume pipeline with Stage 10 to apply.';
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
    0:  'claude -p --model haiku  prompts/p_0.txt\n→ reads meta.json + story.txt, casts voices per locale\n→ writes VoiceCast.json, overwrites pipeline_vars.sh (adds VOICE_CAST_FILE)',
    1:  'claude -p --model haiku  prompts/p_1.txt\n→ checks world & character consistency (no output file)',
    2:  'claude -p --model sonnet prompts/p_2.txt\n→ writes StoryPrompt.json',
    3:  'claude -p --model sonnet prompts/p_3.txt\n→ writes Script.json',
    4:  'claude -p --model sonnet prompts/p_4.txt\n→ writes ShotList.json',
    5:  'claude -p --model sonnet prompts/p_5.txt\n→ writes AssetManifest_draft.shared.json + AssetManifest_draft.en.json',
    6:  'claude -p --model haiku  prompts/p_6.txt\n→ writes canon_diff.json',
    7:  'claude -p --model haiku  prompts/p_7.txt\n→ updates canon.json',
    8:  'claude -p --model sonnet prompts/p_8.txt\n→ writes StoryPrompt.{locale}.json per non-en locale',
    9:  'claude -p --model haiku  prompts/p_9.txt\n→ writes AssetManifest_final.json, RenderPlan.json',
    10: '[1/7] gen_music_clip.py     --manifest AssetManifest_draft.shared.json\n' +
        '      (skips gracefully if no music resources found)\n' +
        '[2/7] manifest_merge.py     --shared ... --locale ...{locale}.json  (per locale)\n' +
        '[3/7] gen_tts_cloud.py      --manifest AssetManifest_merged.{locale}.json\n' +
        '[4/7] post_tts_analysis.py  --manifest AssetManifest_merged.{locale}.json\n' +
        '[5/7] resolve_assets.py     --manifest ... --out AssetManifest.media.{locale}.json\n' +
        '[6/7] gen_render_plan.py    --manifest ... --media ...\n' +
        '[7/7] render_video.py       --plan RenderPlan.{locale}.json  →  renders/{locale}/output.mp4',
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
      _selectedFormat = meta.story_format;
      document.getElementById('info-format-sel-existing').value = meta.story_format;
      updateFormatHint(meta.story_format, 'format-hint-existing');
    }
    if (meta.locales_str) {
      const locs = meta.locales_str.split(',').map(l => l.trim());
      document.getElementById('locale-en-ex').checked       = locs.includes('en');
      document.getElementById('locale-zh-Hans-ex').checked  = locs.includes('zh-Hans');
    }
    // Restore No Music state into the single top-bar checkbox
    if (meta.no_music !== undefined) {
      noMusic = !!meta.no_music;
      const chk = document.getElementById('chk-no-music');
      if (chk) chk.checked = noMusic;
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
    hdr.innerHTML = '⚡ Pipeline — <span style="font-family:var(--mono);color:var(--dim);font-weight:400">run.sh  stages 0 – 10</span>';
    section.appendChild(hdr);
    body.appendChild(section);

    const llmDefs = [
      { n:0,  label:'Stage 0  — Cast voices & write pipeline_vars.sh',         key:'stage_0'  },
      { n:1,  label:'Stage 1  — Check story & world consistency',             key:'stage_1'  },
      { n:2,  label:'Stage 2  — Write episode direction (StoryPrompt)',       key:'stage_2'  },
      { n:3,  label:'Stage 3  — Write script & character dialogue',           key:'stage_3'  },
      { n:4,  label:'Stage 4  — Break script into visual shots (ShotList)',   key:'stage_4'  },
      { n:5,  label:'Stage 5  — List required assets (images, voice, music)', key:'stage_5'  },
      { n:6,  label:'Stage 6  — Identify new story facts to record',          key:'stage_6'  },
      { n:7,  label:'Stage 7  — Update story memory (world canon)',           key:'stage_7'  },
      { n:8,  label:'Stage 8  — Translate & adapt for each language',         key:'stage_8'  },
      { n:9,  label:'Stage 9  — Finalize assets & build render plan',         key:'stage_9'  },
      { n:10, label:'Stage 10 — Merge assets & generate video (output.mp4)',  key:'stage_10' },
    ];

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

    llmDefs.forEach(({ n, label, key }) => {
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

      row.innerHTML =
        statusIcon(info.done) +
        '<span class="step-name">' + escHtml(label) + '</span>' +
        artHtml +
        '<span style="margin-left:auto;display:flex;gap:4px;flex-shrink:0">' +
          '<button class="btn-pipe-run" onclick="runLlmRange(' + n + ',' + n + ')">Run ' + n + '</button>' +
          (n < 10 ? '<button class="btn-pipe-run" onclick="runLlmRange(' + n + ',10)">Run ' + n + '→10</button>' : '') +
        '</span>';
      row.prepend(expandBtn);

      // Detail panel
      const detailEl = document.createElement('div');
      detailEl.className = 'pipe-detail';

      if (n === 10) {
        // Stage 10: numbered Run / Run→7 buttons matching the main stage button style
        const LOCALE_STEPS = [
          { num: 5, step: 'manifest_merge',  label: '5 — merge'    },
          { num: 6, step: 'gen_tts',         label: '6 — tts'      },
          { num: 7, step: 'post_tts',        label: '7 — post_tts' },
          { num: 8, step: 'resolve_assets',  label: '8 — resolve'  },
          { num: 9, step: 'gen_render_plan', label: '9 — plan'     },
          { num: 10, step: 'render_video',   label: '10 — render'  },
        ];
        const localeStepsMap = status.locale_steps || {};
        const sharedStepsMap = status.shared_steps || {};
        const locales        = status.locales || [];
        // stage10Running is kept for reference but no longer used to hard-clear ✓s.
        // Sub-step done state is sourced directly from the server (file-existence checks)
        // so completed steps stay checked even while Stage 10 is still running.
        const stage10Running = !!(pipeRunning && pipeRunning.from <= 10 && pipeRunning.to >= 10); // eslint-disable-line no-unused-vars

        function makeRunBtn(label, onclick) {
          const b = document.createElement('button');
          b.className = 'btn-pipe-run';
          b.style.cssText = 'font-size:0.72em;padding:2px 8px';
          b.textContent = label;
          b.onclick = onclick;
          return b;
        }

        // ── Steps 1–4: shared (no locale) ───────────────────────────────────
        [
          { num: 1, step: 'gen_music_clip',  label: '1 — gen_music_clip'  },
          { num: 2, step: 'gen_characters',  label: '2 — gen_characters'  },
          { num: 3, step: 'gen_backgrounds', label: '3 — gen_backgrounds' },
          { num: 4, step: 'gen_sfx',         label: '4 — gen_sfx'         },
        ].forEach(({ num, step, label }) => {
          const done = (sharedStepsMap[step] || {}).done || false;
          const row = document.createElement('div');
          row.className = 'pipe-substep-row';
          row.appendChild(Object.assign(document.createElement('span'), {
            innerHTML: statusIcon(done), style: 'flex-shrink:0'
          }));
          const nameSpan = document.createElement('span');
          nameSpan.className = 'pipe-substep-locale';
          nameSpan.style.cssText = 'min-width:0;flex:1';
          nameSpan.textContent = label;
          row.appendChild(nameSpan);
          const btnWrap = document.createElement('span');
          btnWrap.style.cssText = 'margin-left:auto;display:flex;gap:4px;flex-shrink:0';
          btnWrap.appendChild(makeRunBtn('Run ' + num, () =>
            startPipeStep({ type: 'post', step,
                            slug: pipeEpSlug, ep_id: pipeEpId, locale: '' })));
          btnWrap.appendChild(makeRunBtn('Run ' + num + '→10', () =>
            startPipeStep({ type: 'shared_chain', from_step: step,
                            slug: pipeEpSlug, ep_id: pipeEpId })));
          row.appendChild(btnWrap);
          detailEl.appendChild(row);
        });

        // ── Steps 5–10: per-locale ──────────────────────────────────────────
        if (locales.length === 0) {
          const hint = document.createElement('div');
          hint.style.cssText = 'color:var(--dim);font-size:0.78em;padding:4px 0';
          hint.textContent = 'No locales yet — run Stage 9 first.';
          detailEl.appendChild(hint);
        } else {
          locales.forEach(locale => {
            // Locale header
            const hdr = document.createElement('div');
            hdr.style.cssText = 'color:var(--dim);font-size:0.74em;padding:5px 0 2px;font-family:var(--mono)';
            hdr.textContent = locale + ':';
            detailEl.appendChild(hdr);

            const lsteps = localeStepsMap[locale] || {};
            LOCALE_STEPS.forEach(({ num, step, label }) => {
              const done = (lsteps[step] || {}).done || false;
              const row  = document.createElement('div');
              row.className = 'pipe-substep-row';
              // status icon
              row.appendChild(Object.assign(document.createElement('span'), {
                innerHTML: statusIcon(done), style: 'flex-shrink:0'
              }));
              // step label (flex:1 to push buttons right)
              row.appendChild(Object.assign(document.createElement('span'), {
                className: 'pipe-substep-locale',
                style: 'min-width:0;flex:1',
                textContent: label
              }));
              // Run N  [Run N→7]  — right-aligned
              const btnWrap = document.createElement('span');
              btnWrap.style.cssText = 'margin-left:auto;display:flex;gap:4px;flex-shrink:0';
              btnWrap.appendChild(makeRunBtn('Run ' + num, () =>
                startPipeStep({ type: 'post', step,
                                slug: pipeEpSlug, ep_id: pipeEpId, locale })));
              if (num < 10) {
                btnWrap.appendChild(makeRunBtn('Run ' + num + '→10', () =>
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
      _selectedFormat = d.story_format || 'episodic';
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
        no_music:     document.getElementById('chk-no-music')?.checked || false,
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
    if (!currentSlug || !currentEpId) return;
    const statusEl = document.getElementById('save-new-ep-status');
    const btnEl    = document.getElementById('btn-save-new-ep');
    const title    = document.getElementById('info-title').value.trim();
    const genre    = document.getElementById('info-genre').value.trim();
    const format   = document.getElementById('info-format-sel').value;
    const locs     = getSelectedLocales();
    const no_music = document.getElementById('chk-no-music')?.checked || false;
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
    // Format
    const fsel = document.getElementById('info-format-sel');
    fsel.value = _selectedFormat;
    updateFormatHint(_selectedFormat, 'format-hint');
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
  }
  function onFormatChangeExisting() {
    _selectedFormat = document.getElementById('info-format-sel-existing').value;
    updateFormatHint(_selectedFormat, 'format-hint-existing');
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
    const no_music = document.getElementById('chk-no-music')?.checked || false;
    btnEl.disabled = true;
    statusEl.textContent = 'Saving…';
    statusEl.style.color = 'var(--dim)';
    try {
      const resp = await fetch('/api/save_episode_meta', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          slug: currentSlug, ep_id: currentEpId,
          title, genre, story_format: format, locales: locs.join(','), no_music
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
    }

    pipeStepEs = new EventSource(url);
    pipeStepEs.addEventListener('line', e => {
      const text = e.data;
      appendLine(text, '');
      // Stage progress (same patterns as runPrompt)
      const startM = text.match(/^\s{2}STAGE (\d+)\/\d+\s+[—\-]{1,2}\s+(.+)/);
      if (startM) {
        stageStartMs[parseInt(startM[1])] = Date.now();
        updateStageProgress(parseInt(startM[1]), startM[2].trim());
      }
      const doneM = text.match(/^✓ Stage (\d+) complete/);
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
    --bg: #0d0d10; --surface: #16161d; --border: #2a2a38;
    --gold: #c9a84c; --text: #dde1ec; --dim: #777;
    --mono: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
    --c-key: #79b8ff; --c-str: #9ecbff; --c-num: #f8c555; --c-bool: #f97583;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    display: flex; flex-direction: column; height: 100vh; overflow: hidden;
  }
  header {
    background: #11111a; border-bottom: 1px solid var(--border);
    padding: 12px 20px; display: flex; align-items: center; gap: 14px;
    flex-shrink: 0;
  }
  .hdr-name { font-size: 0.92rem; font-weight: 700; color: var(--gold); font-family: var(--mono); }
  .hdr-path { font-size: 0.71em; color: var(--dim); font-family: var(--mono); margin-top: 3px; }
  #btn-copy {
    margin-left: auto; background: #ffffff12; color: var(--dim);
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
            "done": check(ep("AssetManifest_final.json"), ep("RenderPlan.json")),
            "artifacts": [ep_rel("AssetManifest_final.json"), ep_rel("RenderPlan.json")],
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

    # Stage 10 is done when every locale has an output.mp4
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

    # Shared (locale-free) post-processing steps — steps 1–4 in the Stage 10 panel
    def _any_files(d: str, ext: str) -> bool:
        return os.path.isdir(d) and any(f.endswith(ext) for f in os.listdir(d))

    _assets_dir = os.path.join(ep_dir, "assets")
    shared_steps = {
        "gen_music_clip":  {"done": _any_files(os.path.join(_assets_dir, "music"),  ".wav")},
        "gen_characters":  {"done": _any_files(os.path.join(proj_dir, "characters"), ".png")},
        "gen_backgrounds": {"done": _any_files(os.path.join(_assets_dir, "backgrounds"), ".png")},
        "gen_sfx":         {"done": _any_files(os.path.join(_assets_dir, "sfx"),    ".wav")},
    }

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
        "title":        meta_title,
        "genre":        meta_genre,
        "story_format": meta_format,
        "locales_str":  meta_locales_str,
        "no_music":     meta_no_music,
    }


def _step_is_done(step: str, slug: str, ep_id: str, locale: str) -> bool:
    """Return True if the step's output already exists (safe to skip)."""
    ep_dir = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)

    def check(*paths): return all(os.path.isfile(p) for p in paths)

    if step == "manifest_merge":
        return check(os.path.join(ep_dir, f"AssetManifest_merged.{locale}.json"))
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

        # Per-locale derived manifests
        for pat in (f"AssetManifest_merged.{loc}.json",
                    f"AssetManifest.media.{loc}.json",
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
                    no_music: bool = False) -> list | None:
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
        return [
            "python3", os.path.join(code_dir, "fetch_ai_assets.py"),
            "--manifest",    ep("AssetManifest_draft.shared.json"),
            "--asset_type",  "backgrounds",
        ]
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
        return [
            "python3", os.path.join(code_dir, "gen_tts_cloud.py"),
            "--manifest", ep(f"AssetManifest_merged.{locale}.json"),
        ]
    elif step == "post_tts":
        return [
            "python3", os.path.join(code_dir, "post_tts_analysis.py"),
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

            # Sanitise: digits only, 0–10
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
            if test_mode:
                run_env["MODEL"] = "haiku"   # cheapest model for all stages
            run_env["RENDER_PROFILE"] = render_profile   # preview_local or high
            if no_music:
                run_env["NO_MUSIC"] = "1"
            # Note: LOCALES and STORY_FORMAT are sourced from pipeline_vars.sh by run.sh

            client = self.client_address
            proc   = None
            try:
                proc = subprocess.Popen(
                    ["bash", "run.sh", ep_dir_param, from_stage, to_stage],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    env=run_env,
                    cwd=PIPE_DIR,
                )
                with _lock:
                    _procs[client] = proc

                for raw in proc.stdout:
                    self.wfile.write(sse("line", raw.rstrip("\n")))
                    self.wfile.flush()

                proc.wait()

                for raw in proc.stderr:
                    line = raw.rstrip("\n")
                    if line:
                        self.wfile.write(sse("error_line", line))
                        self.wfile.flush()

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
                body = (f"<html><body style='background:#0d0d10;color:#e05c5c;"
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
            import hashlib as _hl, tempfile as _tf, datetime as _dt
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

                    _prompt = f"""You are a pipeline diagnostics tool for a 10-stage AI video pipeline.

File dependency status for episode {slug}/{ep_id}
(FRESH = outputs newer than inputs; STALE = an input was modified after the output was created; MISSING = output file does not exist):

{_dep_summary}

Locales: {', '.join(_locales) if _locales else 'en only'}

Status report notes (last 30 lines):
{_sr_tail if _sr_tail else '(empty)'}

Rules:
- A STALE or MISSING stage means it and all downstream stages must re-run.
- Find the earliest such stage — that is from_stage; to_stage is always 10.
- If all FRESH: from_stage = null, to_stage = null.
- confidence: "high" if timestamps alone are conclusive; "medium" if notes affect the answer; "low" if ambiguous.

Reply with JSON only — no text outside the object:
{{"from_stage": <int or null>, "to_stage": <int or null>, "reason": "<one concise sentence>", "confidence": "high|medium|low"}}"""

                    # Call claude haiku via CLI
                    _result = {"error": "unknown"}
                    try:
                        with _tf.NamedTemporaryFile(
                                mode="w", suffix=".txt", delete=False,
                                encoding="utf-8") as _tmp:
                            _tmp.write(_prompt)
                            _tmp_path = _tmp.name
                        _proc = subprocess.run(
                            ["claude", "-p", _tmp_path,
                             "--model", "haiku",
                             "--dangerously-skip-permissions",
                             "--no-session-persistence"],
                            capture_output=True, timeout=45, cwd=PIPE_DIR,
                        )
                        os.unlink(_tmp_path)
                        _raw = _proc.stdout.decode("utf-8", errors="replace").strip()
                        _j0 = _raw.find("{"); _j1 = _raw.rfind("}") + 1
                        if _j0 >= 0 and _j1 > _j0:
                            _result = json.loads(_raw[_j0:_j1])
                        else:
                            _result = {"from_stage": None, "to_stage": None,
                                       "reason": _raw[:300], "confidence": "low"}
                    except subprocess.TimeoutExpired:
                        _result = {"error": "claude timed out (>45s)"}
                    except FileNotFoundError:
                        _result = {"error": "claude CLI not found — check PATH"}
                    except Exception as _exc:
                        _result = {"error": str(_exc)}

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
            profile  = unquote_plus(params.get("profile",  ["preview_local"])[0]).strip()
            no_music = params.get("no_music", ["0"])[0].strip() == "1"

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            cmd = _build_step_cmd(step, slug, ep_id, locale, profile, no_music)
            if not cmd:
                self.wfile.write(sse("error_line", f"Unknown step: {step!r}"))
                self.wfile.write(sse("done", "1"))
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
                            "resolve_assets", "gen_render_plan", "render_video"]
            # Honour optional from= param — start from a specific step
            from_idx = LOCALE_STEPS.index(from_step) if from_step in LOCALE_STEPS else 0
            # When user explicitly picks a start step ("Run N→7"), force-run all
            # steps in the range (delete stale outputs).  Without from= (full chain)
            # keep the skip-if-done behaviour to avoid re-running expensive TTS.
            force_run = bool(from_step)
            step_env = os.environ.copy()
            step_env.pop("CLAUDECODE", None)
            client = self.client_address

            try:
                # ── Purge stale assets before running ────────────────────────
                if purge:
                    self.wfile.write(sse("line",
                        f"\n🗑  Purging cached assets for [{locale}]…"))
                    self.wfile.flush()
                    removed = _purge_episode_assets(slug, ep_id, locale)
                    self.wfile.write(sse("line",
                        f"   Deleted {len(removed)} file(s)"))
                    self.wfile.flush()
                    force_run = True   # purge implies force-run all steps

                for step in LOCALE_STEPS[from_idx:]:
                    if force_run:
                        _delete_step_output(step, slug, ep_id, locale)
                    elif _step_is_done(step, slug, ep_id, locale):
                        self.wfile.write(sse("line",
                            f"  ✓ {step} — already done, skipping"))
                        self.wfile.flush()
                        continue

                    self.wfile.write(sse("line",
                        f"\n── {step} ──────────────────────────────────────────"))
                    self.wfile.flush()

                    cmd = _build_step_cmd(step, slug, ep_id, locale, profile, no_music)
                    if not cmd:
                        self.wfile.write(sse("error_line", f"Unknown step: {step!r}"))
                        self.wfile.write(sse("done", "1"))
                        self.wfile.flush()
                        return

                    proc = None
                    try:
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
                            self.wfile.write(sse("line", raw_line.rstrip("\n")))
                            self.wfile.flush()
                        proc.wait()
                    finally:
                        with _lock:
                            _procs.pop(client, None)

                    if proc.returncode != 0:
                        self.wfile.write(sse("error_line",
                            f"✗ {step} failed (exit {proc.returncode})"))
                        self.wfile.write(sse("done", str(proc.returncode)))
                        self.wfile.flush()
                        return

                    self.wfile.write(sse("line", f"✓ {step}"))
                    self.wfile.flush()

                self.wfile.write(sse("line",
                    f"\n✓ [{locale}] All post-processing steps complete"))
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
                if proc and proc.poll() is None:
                    proc.terminate()

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
                                  "resolve_assets", "gen_render_plan", "render_video"]
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
            client = self.client_address
            proc = None

            def _run_cmd_s10(cmd):
                nonlocal proc
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, env=step_env, cwd=PIPE_DIR,
                )
                with _lock:
                    _procs[client] = proc
                for raw_line in proc.stdout:
                    self.wfile.write(sse("line", raw_line.rstrip("\n")))
                    self.wfile.flush()
                proc.wait()
                with _lock:
                    _procs.pop(client, None)
                return proc.returncode

            try:
                # ── Purge stale assets (all locales) before running ───────────
                if purge:
                    self.wfile.write(sse("line",
                        "\n🗑  Purging cached assets for all locales…"))
                    self.wfile.flush()
                    _total_removed = _purge_episode_assets(slug, ep_id, "")
                    self.wfile.write(sse("line",
                        f"   Deleted {len(_total_removed)} file(s)"))
                    self.wfile.flush()

                # ── Shared steps (locale-free) ────────────────────────────────
                for _step in _SHARED_STEPS[from_idx:]:
                    self.wfile.write(sse("line",
                        f"\n── {_step} (shared) ────────────────────────────────"))
                    self.wfile.flush()
                    _cmd = _build_step_cmd(_step, slug, ep_id, "", profile, no_music)
                    if not _cmd:
                        self.wfile.write(sse("error_line", f"Unknown step: {_step!r}"))
                        self.wfile.write(sse("done", "1"))
                        self.wfile.flush()
                        return
                    _rc = _run_cmd_s10(_cmd)
                    if _rc != 0:
                        self.wfile.write(sse("error_line",
                            f"✗ {_step} failed (exit {_rc})"))
                        self.wfile.write(sse("done", str(_rc)))
                        self.wfile.flush()
                        return
                    self.wfile.write(sse("line", f"✓ {_step}"))
                    self.wfile.flush()

                # ── Per-locale steps ──────────────────────────────────────────
                if not _locales_s10:
                    self.wfile.write(sse("error_line",
                        "No locales found — run Stage 9 first to create manifests"))
                    self.wfile.write(sse("done", "1"))
                    self.wfile.flush()
                    return

                for _locale in _locales_s10:
                    self.wfile.write(sse("line",
                        f"\n── locale: {_locale} ────────────────────────────────"))
                    self.wfile.flush()
                    for _step in _LOCALE_STEPS_ALL:
                        self.wfile.write(sse("line",
                            f"\n── {_step} [{_locale}] ──────────────────────────"))
                        self.wfile.flush()
                        _cmd = _build_step_cmd(_step, slug, ep_id, _locale, profile, no_music)
                        if not _cmd:
                            self.wfile.write(sse("error_line", f"Unknown step: {_step!r}"))
                            self.wfile.write(sse("done", "1"))
                            self.wfile.flush()
                            return
                        _rc = _run_cmd_s10(_cmd)
                        if _rc != 0:
                            self.wfile.write(sse("error_line",
                                f"✗ {_step} [{_locale}] failed (exit {_rc})"))
                            self.wfile.write(sse("done", str(_rc)))
                            self.wfile.flush()
                            return
                        self.wfile.write(sse("line", f"✓ {_step} [{_locale}]"))
                        self.wfile.flush()
                    self.wfile.write(sse("line",
                        f"✓ [{_locale}] all locale steps complete"))
                    self.wfile.flush()

                self.wfile.write(sse("line", "\n✓ Stage 10 — all steps complete"))
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
                if proc and proc.poll() is None:
                    proc.terminate()

        # Range-request-capable media streaming (for HTML5 <video>)
        elif parsed.path == "/serve_media":
            params    = parse_qs(parsed.query)
            rel_path  = unquote_plus(params.get("path", [""])[0]).strip()
            safe_root = os.path.realpath(PIPE_DIR)

            if not rel_path:
                self.send_response(400); self.end_headers(); return

            full_path = os.path.realpath(os.path.join(PIPE_DIR, rel_path))
            if not full_path.startswith(safe_root + os.sep) and full_path != safe_root:
                self.send_response(403); self.end_headers(); return

            if not os.path.isfile(full_path):
                self.send_response(404); self.end_headers(); return

            ext  = os.path.splitext(full_path)[1].lower().lstrip(".")
            mime = {"mp4": "video/mp4", "webm": "video/webm", "ogg": "video/ogg",
                    "mp3": "audio/mpeg", "wav": "audio/wav",
                    "aac": "audio/aac", "m4a": "audio/mp4"}.get(ext, "application/octet-stream")

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
                                       "assets/music/MusicPlan.json",
                                       "assets/music/user_cut_clips.json",
                                       "assets/meta/gen_music_clip_results.json"}
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
                    if not any(p["hash"] == h for p in vp):
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
                    result = subprocess.run(
                        ["claude", "-p",
                         "--model", "haiku",
                         "--dangerously-skip-permissions",
                         "--no-session-persistence",
                         tmp_path],
                        capture_output=True, text=True, cwd=PIPE_DIR, timeout=30
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

                slug       = payload.get("slug", "").strip()
                ep_id      = payload.get("ep_id", "").strip()
                server_url = (payload.get("server_url") or "http://localhost:8200").rstrip("/")
                api_key    = os.environ.get("MEDIA_API_KEY", "")

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
                    "project":    slug,
                    "episode_id": ep_id,
                    "manifest":   manifest,
                    "top_n":      int(os.environ.get("MEDIA_TOP_N",
                                      _vc_config.get("media", {}).get("top_n", 5))),
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

                # Find the merged manifest — try primary locale first, then any
                import glob as _glob_mod
                merged_manifests = _glob_mod.glob(
                    os.path.join(ep_dir, "AssetManifest_merged.*.json"))
                if not merged_manifests:
                    raise FileNotFoundError(
                        "No AssetManifest_merged.*.json found. "
                        "Run stages 10[1]–10[4] first.")
                manifest_path = merged_manifests[0]

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
                existing_meta["story_title"]  = payload.get("title", "").strip()
                existing_meta["series_genre"] = payload.get("genre", "").strip()
                existing_meta["story_format"] = payload.get("story_format", "episodic").strip()
                existing_meta["locales"]      = payload.get("locales", "en").strip()
                existing_meta["no_music"]     = bool(payload.get("no_music", False))
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
                  "/api/media_batch", "/api/media_confirm",
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
