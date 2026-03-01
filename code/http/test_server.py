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

PORT     = 8000
PIPE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (pipe/)

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
    """Return True when hash h matches the pre-cached index entry for voice+style.

    Same hash means the user's params are identical to the index defaults,
    so no custom preset should be created.
    """
    if not style:
        return False
    idx = _load_index_cache()
    clip = idx.get(voice, {}).get("clips", {}).get(style)
    if not clip:
        return False
    # index stores full 64-char SHA256; h is the first 16 chars
    return clip.get("hash", "")[:16] == h


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
    """Parse prompts/azure_tts_styles.txt; return voice catalog grouped by story locale.

    Returns: { "en": [VoiceEntry, ...], "zh-Hans": [...], ... }
    VoiceEntry: { voice, azure_locale, gender, local_name, styles }
    Result is cached at module level (parsed once on first request).
    """
    global _voice_catalog_cache
    if _voice_catalog_cache is not None:
        return _voice_catalog_cache

    styles_path = os.path.join(PIPE_DIR, "prompts", "azure_tts_styles.txt")
    # BUG-1 fix: r'^(\S+Neural)' captures Dragon HD names with colon
    #   e.g. "zh-CN-Xiaoxiao2:DragonHDFlashLatestNeural" — broken if \w+ is used
    VOICE_RE  = re.compile(r'^(\S+Neural)')
    LOCALE_RE = re.compile(r'locale=(\S+)\s+gender=(\S+)\s+local_name=(.+)')
    STYLES_RE = re.compile(r"styles\(\d+\):\s*\[(.+)\]")

    entries: list[dict] = []
    cur: dict | None    = None

    if os.path.isfile(styles_path):
        with open(styles_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                vm = VOICE_RE.match(line)
                if vm:
                    if cur and "azure_locale" in cur:
                        entries.append(cur)
                    cur = {"voice": vm.group(1), "styles": []}
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
        if cur and "azure_locale" in cur:
            entries.append(cur)

    # Group by story locale: "en" → azure_locale.startswith("en-")
    catalog: dict[str, list] = {}
    for entry in entries:
        al = entry.get("azure_locale", "")
        if al.startswith("en-"):
            group = "en"
        elif al.startswith("zh-"):
            group = "zh-Hans"
        else:
            group = al.split("-")[0] if "-" in al else al
        catalog.setdefault(group, []).append(entry)

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
    spoken = escaped
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
      <div id="sr-content"><span style="color:var(--dim);font-style:italic">No episode selected.</span></div>
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
      <span class="info-value"><input id="info-title" type="text" placeholder="Story title"></span>
      <span id="badge-title" class="info-badge" style="display:none"></span>
    </div>
    <div class="info-row">
      <span class="info-label">🏷 Genre</span>
      <span class="info-value"><input id="info-genre" type="text" placeholder="Genre"></span>
      <span id="badge-genre" class="info-badge" style="display:none"></span>
    </div>
    <div class="info-row">
      <span class="info-label">📁 Slug</span>
      <span class="info-value"><input id="info-slug" type="text" placeholder="project-slug" oninput="onSlugInput()"></span>
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
    if (name === 'browse')   loadProjects();
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
    return bank[locale] ?? bank['en'];
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

      // No data for this locale yet — synthesise defaults so user can assign manually
      // Borrow pitch/break/rate from the 'en' locale if it exists, otherwise use role defaults
      if (!charLoc) {
        const enLoc = char['en'] || {};
        const roleDefaults = {
          narrator:    { pitch: '-5',  break_ms: 600, rate: '0'  },
          protagonist: { pitch: '0',   break_ms: 400, rate: '0'  },
          antagonist:  { pitch: '-5',  break_ms: 500, rate: '0'  },
          default:     { pitch: '0',   break_ms: 500, rate: '0'  },
        };
        const rd = roleDefaults[char.role] || roleDefaults.default;
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
      voiceList.forEach(v => {
        const opt = document.createElement('option');
        opt.value = v.voice;
        opt.textContent = v.voice + ' (' + v.gender + ')';
        if (v.voice === charLoc.azure_voice) opt.selected = true;
        voiceSel.appendChild(opt);
      });

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
      presetRow.appendChild(presetLbl);
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
      if (_matchedPreset) presetSel.value = _matchedPreset.hash;

      // ── Voice select onChange → rebuild style + preset dropdowns ──
      voiceSel.addEventListener('change', () => {
        const newV     = voiceList.find(v => v.voice === voiceSel.value);
        const curStyle = styleSel.value;
        styleSel.innerHTML = '';
        const blank2 = document.createElement('option');
        blank2.value = ''; blank2.textContent = '\u2014 no style \u2014';
        styleSel.appendChild(blank2);
        (newV?.styles || []).forEach(s => {
          const opt = document.createElement('option');
          opt.value = s; opt.textContent = s;
          if (s === curStyle) opt.selected = true;
          styleSel.appendChild(opt);
        });
        rebuildPresetSelect(card, voiceSel.value);
        card.querySelector('.vc-preset-select').selectedIndex = 0;
      });

      // ── Style select onChange → auto-apply preset or index params; refresh interp ──
      styleSel.addEventListener('change', () => {
        const voice   = voiceSel.value;
        const style   = styleSel.value;
        const presets = _vcPresets[voice] ?? [];
        // 1st priority: most-recent saved preset for this voice+style
        const match = style ? [...presets].reverse().find(p => p.style === style) : null;
        if (match) {
          card.querySelector('[data-field=degree]').value = match.style_degree;
          card.querySelector('[data-field=rate]').value   = String(match.rate  ?? '').replace('%', '');
          card.querySelector('[data-field=pitch]').value  = String(match.pitch ?? '').replace('%', '');
          card.querySelector('[data-field=break]').value  = match.break_ms;
          presetSel.value = match.hash;
        } else {
          // 2nd priority: params from the pre-cached index.json clip for this voice+style
          const clip = style ? (_voiceIndex?.[voice]?.clips?.[style]) : null;
          if (clip?.params) {
            const p = clip.params;
            card.querySelector('[data-field=degree]').value = p.style_degree ?? 1.0;
            card.querySelector('[data-field=rate]').value   = String(p.rate  ?? '0').replace('%', '');
            card.querySelector('[data-field=pitch]').value  = String(p.pitch ?? '' ).replace('%', '');
            card.querySelector('[data-field=break]').value  = p.break_ms ?? 0;
          }
          presetSel.selectedIndex = 0;
        }
        // Always fire input events so interpretation text reflects current values
        ['degree','rate','pitch','break'].forEach(f =>
          card.querySelector(`[data-field=${f}]`).dispatchEvent(new Event('input')));
      });

      // ── Preset select onChange → populate all param fields ──
      presetSel.addEventListener('change', () => {
        const presets = _vcPresets[voiceSel.value] ?? [];
        const preset  = presets.find(p => p.hash === presetSel.value);
        if (!preset) {
          // "— no preset —" selected: restore index defaults for the current style
          const defaults = _voiceIndex?.[voiceSel.value]?.clips?.[styleSel.value]?.params;
          card.querySelector('[data-field=degree]').value = defaults?.style_degree ?? 1.0;
          card.querySelector('[data-field=rate]').value   = String(defaults?.rate   ?? '0%').replace('%', '');
          card.querySelector('[data-field=pitch]').value  = String(defaults?.pitch  ?? '').replace('%', '');
          card.querySelector('[data-field=break]').value  = defaults?.break_ms ?? 0;
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

        if (data.preset) {
          (_vcPresets[params.azure_voice] ??= []).push(data.preset);
          rebuildPresetSelect(card, params.azure_voice);
          card.querySelector('.vc-preset-select').value = data.preset.hash;
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
  let pipeRunning   = null;   // { from, to } while an llm range is running; null otherwise
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
      running: pipeRunning   // client-side: forces re-render when run starts/stops
    });
    if (_fp === _lastStatusFingerprint) return;
    _lastStatusFingerprint = _fp;

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

    // If stages are actively running, clear their stale ✓ and artifact links
    // so Pipeline tab doesn't show completed state for a stage that is
    // currently re-running.
    if (pipeRunning) {
      llmDefs.forEach(({ n, key }) => {
        if (n >= pipeRunning.from && n <= pipeRunning.to) {
          if (stagesMap[key]) {
            stagesMap[key].done      = false;
            stagesMap[key].artifacts = [];   // hide stale links while re-running
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
        // True when any run that covers Stage 10 is active — clears stale sub-step ✓s
        const stage10Running = !!(pipeRunning && pipeRunning.from <= 10 && pipeRunning.to >= 10);

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
          const done = stage10Running ? false : ((sharedStepsMap[step] || {}).done || false);
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
              const done = stage10Running ? false : ((lsteps[step] || {}).done || false);
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
      // ── New Project ──────────────────────────────────────────────────────────
      epSel.disabled = true;
      epSel.innerHTML = '<option value="">—</option>';
      _usingExistingEp = false;
      document.getElementById('info-bar').classList.remove('visible');
      document.getElementById('existing-ep-bar').style.display = 'none';
      document.getElementById('btn-prepare').style.display = '';
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
      updateSlugBadge(d.exists, d.suggested);
    }, 400);
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
      // Keep progress-dot globals accurate for startPipeStep-driven runs
      // (vcContinue, runLlmRange from Pipeline tab, etc.)
      _runFromStage = params.from;
      _runToStage   = params.to;
    }
    refreshPipeline();   // immediately clear stale ✓ in Pipeline tab

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
        markStageDone(n);
        insertReviewButtons(n);
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
      setTimeout(() => refreshPipeline(), 600);
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

    # ── GET ───────────────────────────────────────────────────────────────────
    def do_GET(self):
        parsed = urlparse(self.path)

        # Serve UI
        if parsed.path == "/":
            body = HTML.encode()
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

            except BrokenPipeError:
                pass   # client disconnected
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

            except BrokenPipeError:
                pass
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

            except BrokenPipeError:
                pass
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

            except BrokenPipeError:
                pass
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

                # Derive azure_locale server-side (MIN 1)
                azure_locale = '-'.join(azure_voice.split('-')[:2])

                # Cache key (MIN 2 / BUG-B: pitch normalised to "" not "0%")
                key_dict = {
                    "v": azure_voice, "s": style or "",
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

                # Auto-save preset only when params differ from index defaults.
                # Same hash as the pre-cached clip = user hasn't customised anything
                # → skip, so the preset list stays clean.
                new_preset = None
                if style and not _is_default_clip(azure_voice, style, h):
                    pdata = load_presets()
                    vp    = pdata["presets"].setdefault(azure_voice, [])
                    if not any(p["hash"] == h for p in vp):
                        new_preset = {
                            "name":         f"custom{len(vp) + 1}",
                            "style":        style,
                            "style_degree": style_degree,
                            "rate":         rate,
                            "pitch":        pitch or "",
                            "break_ms":     break_ms,
                            "hash":         h,
                        }
                        vp.append(new_preset)
                        save_presets(pdata)

                resp = json.dumps({"url": url, "preset": new_preset}).encode()
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
                  "/api/preview_voice", "/api/save_voice_cast",
                  "/api/status_report", "/api/append_status_report",
                  "/api/check_slug", "/api/next_episode_id",
                  "/api/create_episode", "/api/save_episode_meta"}
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
        print("\nStopped.")
