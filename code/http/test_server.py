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
import json
import os
import re
import shutil
import socket
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse, unquote_plus

PORT     = 8000
PIPE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (pipe/)

# ── Running-process registry (so Stop button can kill it) ──────────────────────
_lock  = threading.Lock()
_procs = {}   # client_addr → subprocess.Popen


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
    overflow: hidden;
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
  .story-block { flex-shrink: 0; }
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

  /* ── Prompt area ── */
  .prompt-block { flex-shrink: 0; }
  .input-row { display: flex; gap: 10px; align-items: flex-end; }
  #prompt {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-family: var(--mono);
    font-size: 0.88em;
    padding: 12px 14px;
    resize: none;
    min-height: 54px;
    max-height: 140px;
    line-height: 1.5;
    outline: none;
    transition: border-color .15s;
  }
  #prompt:focus { border-color: var(--gold); }

  .btn-group { display: flex; flex-direction: column; gap: 6px; }
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
    min-height: 0;
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
</style>
</head>
<body>

<header>
  <h1>⚡ Claude Runner</h1>
  <span id="status-badge">IDLE</span>
  <span id="cost-badge"></span>
</header>

<main>

  <!-- ── Story input ── -->
  <div class="story-block">
    <div class="section-label">
      Story Input
      <span class="file-badge" id="file-badge">story_1.txt</span>
    </div>
    <textarea id="story" spellcheck="false"
placeholder="Story title  : The Pharaoh Who Defied Death
Project slug : the-pharaoh-who-defied-death
Episode num  : 01
Episode id   : s01e01
Locales      : en, zh-Hans
Genre        : Ancient Egyptian Epic / Mystery / Supernatural / Political Drama
Direction    : …"></textarea>
  </div>

  <!-- ── Stages + buttons ── -->
  <div class="prompt-block">
    <div class="section-label">Stages <span style="font-weight:400;letter-spacing:0;text-transform:none;color:#555">(from – to, e.g. "0 9" for full run, "2 4" to re-run steps 2–4)</span></div>
    <div class="input-row">
      <textarea id="prompt" rows="2" spellcheck="false"
        placeholder="0  9"></textarea>
      <div class="btn-group">
        <button id="btn-run"   onclick="runPrompt()">▶ Run</button>
        <button id="btn-stop"  onclick="stopRun()">■ Stop</button>
        <button id="btn-clear" onclick="clearOutput()">✕ Clear</button>
      </div>
    </div>
  </div>

  <div id="cmd-preview">$ <span id="cmd-text"></span></div>

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
    btnRun.disabled           = state === 'running';
    btnStop.style.display     = state === 'running' ? 'block' : 'none';
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

  function clearOutput() {
    outputEl.innerHTML = '<span class="sys">Cleared.\n</span>';
    lineCount = 0;
    lineCountEl.textContent = '0 lines';
    costEl.style.display    = 'none';
    cmdPreview.style.display = 'none';
  }

  function stopRun() {
    if (es) { es.close(); es = null; }
    fetch('/stop', { method: 'POST' }).catch(() => {});
    appendLine('[ Stopped by user ]', 'sys');
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
    return { from: Math.max(0, from), to: Math.min(9, to) };
  }

  // ── Init ────────────────────────────────────────────────────────────────────
  window.addEventListener('DOMContentLoaded', async () => {
    promptEl.value = '0  9';          // default: full pipeline run
    await refreshNextNum();
  });

  // ── Run ─────────────────────────────────────────────────────────────────────
  async function runPrompt() {
    const story        = storyEl.value.trim();
    const stagesRaw    = promptEl.value.trim() || '0 9';
    const { from, to } = parseStages(stagesRaw);

    if (!story) {
      storyEl.focus();
      storyEl.style.borderColor = 'var(--red)';
      setTimeout(() => (storyEl.style.borderColor = ''), 1200);
      return;
    }
    if (from > to) {
      promptEl.focus();
      promptEl.style.borderColor = 'var(--red)';
      setTimeout(() => (promptEl.style.borderColor = ''), 1200);
      return;
    }
    if (es) { es.close(); es = null; }

    // Reset output
    outputEl.innerHTML = '';
    lineCount = 0;
    lineCountEl.textContent = '0 lines';
    costEl.style.display     = 'none';
    cmdPreview.style.display = 'none';
    setStatus('running');

    // ── 1. Save story file ──────────────────────────────────────────────────
    let filename;
    try {
      appendLine('Saving story…', 'sys');
      const res = await fetch('/save_story', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ story }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      filename = data.filename;
      appendLine(`Saved as ${filename}`, 'sys');
      // Pre-update badge so the user sees the next slot immediately
      fileBadgeEl.textContent = `story_${data.num + 1}.txt`;
    } catch (err) {
      appendLine(`Failed to save story: ${err}`, 'err');
      setStatus('error');
      return;
    }

    // ── 1b. Check whether the episode output folder already exists ──────────
    try {
      const chk  = await fetch('/check_episode?story_file=' + encodeURIComponent(filename));
      const info = await chk.json();
      if (info.exists) {
        appendLine(`⚠  Episode folder exists: ${info.path}`, 'sys');
        const doDelete = await showConfirmModal(info.path);
        if (doDelete) {
          appendLine(`Deleting ${info.path} …`, 'sys');
          const del     = await fetch('/delete_episode_dir', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ path: info.path }),
          });
          const delData = await del.json();
          if (delData.deleted) {
            appendLine(`Deleted. Starting fresh.`, 'sys');
          } else {
            appendLine(`Delete failed: ${delData.error || 'unknown error'}`, 'err');
            setStatus('error');
            return;
          }
        } else {
          appendLine(`Keeping existing folder — continuing pipeline.`, 'sys');
        }
      }
    } catch (err) {
      // Non-fatal: if check fails just proceed
      appendLine(`(Episode-dir check skipped: ${err})`, 'sys');
    }

    // ── 2. Show command preview ─────────────────────────────────────────────
    cmdText.textContent      = `./run.sh ${filename} ${from} ${to}`;
    cmdPreview.style.display = 'block';

    // ── 3. Open SSE stream ──────────────────────────────────────────────────
    const url = `/stream?story_file=${encodeURIComponent(filename)}&from=${from}&to=${to}`;
    es = new EventSource(url);

    es.addEventListener('line',       e => appendLine(e.data, ''));
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

    es.addEventListener('done', e => {
      es.close(); es = null;
      const code = parseInt(e.data);
      if (code === 0) {
        appendLine('[ Done ]', 'done');
        setStatus('idle');
      } else {
        appendLine(`[ Exited with code ${code} ]`, 'err');
        setStatus('error');
      }
      refreshNextNum();   // update badge + prompt number for the next run
    });

    es.onerror = () => {
      if (es) { es.close(); es = null; }
      appendLine('[ Connection lost ]', 'err');
      setStatus('error');
    };
  }
</script>
</body>
</html>
"""


# ── SSE helpers ────────────────────────────────────────────────────────────────
def sse(event: str, data: str) -> bytes:
    return f"event: {event}\ndata: {data}\n\n".encode()


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

        # SSE stream  —  runs: bash run.sh <story_file> <from> <to>
        elif parsed.path == "/stream":
            params     = parse_qs(parsed.query)
            story_file = unquote_plus(params.get("story_file", [""])[0]).strip()
            from_stage = params.get("from", ["0"])[0].strip()
            to_stage   = params.get("to",   ["9"])[0].strip()

            # Sanitise: digits only, 0–9
            from_stage = str(max(0, min(9, int(from_stage)))) if from_stage.isdigit() else "0"
            to_stage   = str(max(0, min(9, int(to_stage))))   if to_stage.isdigit()   else "9"

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            if not story_file:
                self.wfile.write(sse("error_line", "No story_file provided."))
                self.wfile.write(sse("done", "1"))
                self.wfile.flush()
                return

            client = self.client_address
            proc   = None
            try:
                proc = subprocess.Popen(
                    ["bash", "run.sh", story_file, from_stage, to_stage],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    env=os.environ.copy(),
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

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        path = args[0].split()[1] if args else ""
        # Silence noisy but uninteresting routes
        if path not in ("/", "/stop", "/next_story_num", "/check_episode"):
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
