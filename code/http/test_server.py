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
  #review-pane-vc { max-height: 260px; overflow-y: auto; }
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
  /* ── Voice Cast viewer ────────────────────────────────────────────── */
  .vc-char-row {
    padding: 8px 14px; border-bottom: 1px solid #1a1a26;
  }
  .vc-char-row:last-child { border-bottom: none; }
  .vc-char-name { font-weight: 700; font-size: 0.83em; color: var(--text); }
  .vc-char-role { font-size: 0.75em; color: var(--dim); margin-left: 8px; }
  .vc-char-voice {
    font-family: var(--mono); font-size: 0.78em; color: var(--blue); margin-top: 3px;
  }
  .vc-char-params { font-size: 0.74em; color: var(--dim); margin-top: 2px; }
  .vc-style-chip {
    display: inline-block; background: #ffffff08;
    border: 1px solid var(--border); border-radius: 3px;
    padding: 1px 5px; font-size: 0.70em; color: #8a9ab8;
    margin: 2px 2px 0 0;
  }
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
  <div class="toggle-wrap" id="toggle-test"
       onclick="toggleTestMode()" tabindex="0"
       title="Test mode ON — cheapest model (haiku) for all stages">
    <span class="toggle-label toggle-left">🧪 Test</span>
    <div class="toggle-track"><div class="toggle-thumb"></div></div>
    <span class="toggle-label toggle-right">🎬 Prod</span>
  </div>
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
    <div class="section-label">Stages <span style="font-weight:400;letter-spacing:0;text-transform:none;color:#555">(from – to, e.g. "0 10" for full run, "2 4" to re-run steps 2–4)</span></div>
    <div class="input-row">
      <textarea id="prompt" rows="2" spellcheck="false"
        placeholder="0  10"></textarea>
      <div class="btn-group">
        <button id="btn-run"   onclick="runPrompt()">▶ Run</button>
        <button id="btn-stop"  onclick="stopRun()">■ Stop</button>
        <button id="btn-clear" onclick="clearOutput()">✕ Clear</button>
      </div>
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
    <div id="review-pane-vc" class="review-pane">
      <div id="voicecast-body"></div>
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
  let testMode   = true;     // default ON — use cheapest model
  let renderProd = false;    // false = preview_local (CRF 28), true = high (CRF 18)
  const stageStartMs = {};   // stage number → Date.now() at start
  let _runFromStage = 0;
  let _runToStage   = 10;

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
    return { from: Math.max(0, from), to: Math.min(10, to) };
  }

  // ── Init ────────────────────────────────────────────────────────────────────
  window.addEventListener('DOMContentLoaded', async () => {
    promptEl.value = '0  10';         // default: full pipeline run
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
      if (info.project_slug) currentSlug = info.project_slug;
      if (info.episode_id)   currentEpId = info.episode_id;
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
    const modeTag = testMode ? '  [MODEL=haiku 🧪]' : '  [per-stage models 🎬]';
    cmdText.textContent      = `./run.sh ${filename} ${from} ${to}${modeTag}`;
    cmdPreview.style.display = 'block';

    // ── 3. Open SSE stream ──────────────────────────────────────────────────
    const url = `/stream?story_file=${encodeURIComponent(filename)}&from=${from}&to=${to}&test=${testMode ? '1' : '0'}&profile=${renderProd ? 'high' : 'preview_local'}`;
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

    es.addEventListener('done', e => {
      es.close(); es = null;
      const code = parseInt(e.data);
      hideStageProgress();
      if (code === 0) {
        appendLine('[ ✓ Done — output.mp4 ready ]', 'done');
        setStatus('idle');
        switchTab('pipeline');   // jump to video player
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
  function toggleTestMode() {
    testMode = !testMode;
    const wrap = document.getElementById('toggle-test');
    wrap.classList.toggle('prod', !testMode);
    wrap.title = testMode
      ? 'Test mode ON — cheapest model (haiku) for all stages'
      : 'Production mode — quality models per stage (sonnet for creative, haiku for mechanical)';
  }
  function toggleRenderMode() {
    renderProd = !renderProd;
    const wrap = document.getElementById('toggle-render');
    wrap.classList.toggle('render-hd', renderProd);
    wrap.title = renderProd
      ? 'HD mode — high quality encode (CRF 18) for final upload'
      : 'Preview mode — fast encode (CRF 28) for review';
  }

  // Allow keyboard activation (Space / Enter)
  document.addEventListener('keydown', e => {
    if (e.target === document.getElementById('toggle-test') &&
        (e.key === ' ' || e.key === 'Enter')) {
      e.preventDefault(); toggleTestMode();
    }
    if (e.target === document.getElementById('toggle-render') &&
        (e.key === ' ' || e.key === 'Enter')) {
      e.preventDefault(); toggleRenderMode();
    }
  });

  // ── Tab switching ───────────────────────────────────────────────────────────
  function switchTab(name) {
    document.querySelectorAll('.tab').forEach(t =>
      t.classList.toggle('active', t.dataset.tab === name));
    document.getElementById('panel-run').style.display      = name === 'run'      ? 'flex' : 'none';
    document.getElementById('panel-browse').style.display   = name === 'browse'   ? 'flex' : 'none';
    document.getElementById('panel-pipeline').style.display = name === 'pipeline' ? 'flex' : 'none';
    if (name === 'browse')   loadProjects();
    if (name === 'pipeline') initPipelineTab();
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
        { label: 'pipeline_vars.sh',    path: 'pipeline_vars.sh' },
        { label: 'episode_direction.txt', path: 'episode_direction.txt' },
      ],
      2: [ep('StoryPrompt.json')],
      3: [ep('Script.json')],
      4: [ep('ShotList.json')],
      5: [ep('AssetManifest_draft.shared.json'), ep('AssetManifest_draft.en.json')],
      6: [ep('canon_diff.json')],
      7: [ep('canon.json')],
      9: [ep('AssetManifest_final.json'), ep('RenderPlan.json')],
    };
    // Stages > 0 need slug/ep_id to build the episode path
    if (n !== 0 && (!currentSlug || !currentEpId)) return [];
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
      // Auto-select if we have a current episode from Run tab, or restore prev selection
      if (!prev && currentSlug && currentEpId) {
        sel.value = currentSlug + '|' + currentEpId;
      } else if (prev) {
        sel.value = prev;
      }
      if (sel.value) onPipeEpChange();
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
    body.innerHTML = '<div style="color:var(--dim);font-style:italic;font-size:0.83em;padding:4px 0">Loading…</div>';
    try {
      const res = await fetch('/pipeline_status?slug=' + encodeURIComponent(pipeEpSlug) +
                              '&ep_id=' + encodeURIComponent(pipeEpId));
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
    0:  'claude -p --model haiku  prompts/p_0.txt\n→ writes pipeline_vars.sh, episode_direction.txt',
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

  function renderPipelineStatus(status) {
    // Store the auto-detected story file for use by runLlmRange
    pipeStoryFile = status.story_file || null;

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
      { n:0,  label:'Stage 0  — Extract story variables & set up project',    key:'stage_0'  },
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

    // If stages are actively running, clear their stale ✓ so Pipeline tab
    // doesn't show completed state for a stage that is currently re-running.
    if (pipeRunning) {
      llmDefs.forEach(({ n, key }) => {
        if (n >= pipeRunning.from && n <= pipeRunning.to) {
          if (stagesMap[key]) stagesMap[key].done = false;
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
          { num: 2, step: 'manifest_merge',  label: '2 — merge'    },
          { num: 3, step: 'gen_tts',         label: '3 — tts'      },
          { num: 4, step: 'post_tts',        label: '4 — post_tts' },
          { num: 5, step: 'resolve_assets',  label: '5 — resolve'  },
          { num: 6, step: 'gen_render_plan', label: '6 — plan'     },
          { num: 7, step: 'render_video',    label: '7 — render'   },
        ];
        const localeStepsMap = status.locale_steps || {};
        const locales        = status.locales || [];

        function makeRunBtn(label, onclick) {
          const b = document.createElement('button');
          b.className = 'btn-pipe-run';
          b.style.cssText = 'font-size:0.72em;padding:2px 8px';
          b.textContent = label;
          b.onclick = onclick;
          return b;
        }

        // ── Step 1: gen_music_clip (shared, no locale) ──────────────────────
        const sharedRow = document.createElement('div');
        sharedRow.className = 'pipe-substep-row';
        const sharedNameSpan = document.createElement('span');
        sharedNameSpan.className = 'pipe-substep-locale';
        sharedNameSpan.style.cssText = 'min-width:0;flex:1';
        sharedNameSpan.textContent = '1 — gen_music_clip';
        sharedRow.appendChild(sharedNameSpan);
        const sharedBtnWrap = document.createElement('span');
        sharedBtnWrap.style.cssText = 'margin-left:auto;display:flex;gap:4px;flex-shrink:0';
        sharedBtnWrap.appendChild(makeRunBtn('Run 1', () =>
          startPipeStep({ type: 'post', step: 'gen_music_clip',
                          slug: pipeEpSlug, ep_id: pipeEpId, locale: '' })));
        sharedRow.appendChild(sharedBtnWrap);
        detailEl.appendChild(sharedRow);

        // ── Steps 2–7: per-locale ───────────────────────────────────────────
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
              if (num < 7) {
                btnWrap.appendChild(makeRunBtn('Run ' + num + '→7', () =>
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
    const vcBody            = document.getElementById('voicecast-body');

    const readyVideos = status.ready_videos || [];
    const readyDubbed = status.ready_dubbed || [];
    const vc          = status.voice_cast;
    const vcChars     = (vc && vc.characters && vc.characters.length) ? vc.characters : null;
    const vcLocales   = vcChars
      ? Object.keys(vcChars[0]).filter(k => !['character_id','role','gender','personality'].includes(k))
      : [];

    const hasVideo = readyVideos.length > 0;
    const hasAudio = readyDubbed.length > 0;
    const hasVc    = !!vcChars;

    if (!hasVideo && !hasAudio && !hasVc) {
      reviewWrap.style.display = 'none';
    } else {
      reviewWrap.style.display = '';
      reviewContentTabs.innerHTML = '';
      reviewLocaleTabs.innerHTML  = '';

      // ── Voice cast renderer ─────────────────────────────────────────────────
      function renderVcLocale(locale) {
        vcBody.innerHTML = '';
        vcChars.forEach(char => {
          const loc = char[locale];
          if (!loc) return;
          const row = document.createElement('div');
          row.className = 'vc-char-row';
          const charHdr = document.createElement('div');
          charHdr.innerHTML =
            `<span class="vc-char-name">👤 ${char.character_id}</span>` +
            `<span class="vc-char-role">${char.role || ''}</span>`;
          row.appendChild(charHdr);
          if (loc.azure_voice) {
            const v = document.createElement('div');
            v.className = 'vc-char-voice';
            v.textContent = loc.azure_voice;
            row.appendChild(v);
          }
          const paramParts = [];
          if (loc.azure_pitch)            paramParts.push('pitch: ' + loc.azure_pitch);
          if (loc.azure_break_ms != null) paramParts.push('break: ' + loc.azure_break_ms + 'ms');
          if (loc.azure_style_degree)     paramParts.push('degree: ' + loc.azure_style_degree);
          if (paramParts.length) {
            const p = document.createElement('div');
            p.className = 'vc-char-params';
            p.textContent = paramParts.join('  ·  ');
            row.appendChild(p);
          }
          const styles = loc.available_styles || [];
          if (styles.length) {
            const chips = document.createElement('div');
            chips.style.marginTop = '4px';
            styles.forEach(s => {
              const chip = document.createElement('span');
              chip.className = 'vc-style-chip';
              chip.textContent = s;
              chips.appendChild(chip);
            });
            row.appendChild(chips);
          }
          vcBody.appendChild(row);
        });
      }

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
        } else if (pane === 'vc') {
          const firstVcLocale = vcLocales[0] || '';
          buildLocaleTabs(vcLocales, renderVcLocale, firstVcLocale);
          if (firstVcLocale) renderVcLocale(firstVcLocale);
        }
      }

      // ── Content tab buttons ─────────────────────────────────────────────────
      const tabDefs = [];
      if (hasVideo) tabDefs.push({ pane: 'video', label: 'Video' });
      if (hasAudio) tabDefs.push({ pane: 'audio', label: 'Soundtrack' });
      if (hasVc)    tabDefs.push({ pane: 'vc',    label: 'Voice Cast' });

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
                  '/renders/' + locale + '/youtube_dubbed.aac';
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
    if (!pipeStoryFile) {
      switchTab('run');
      appendLine('⚠  No story file detected for this episode. Run Stage 0 first.', 'err');
      return;
    }
    startPipeStep({ type: 'llm', from, to, story_file: pipeStoryFile });
  }

  function runPostStep(step, locale) {
    startPipeStep({ type: 'post', step, locale, slug: pipeEpSlug, ep_id: pipeEpId });
  }

  function runLocaleInPipeTerm(locale) {
    // Runs all 6 post-processing steps for this locale via /run_locale (skip-if-done)
    startPipeStep({ type: 'locale', locale, slug: pipeEpSlug, ep_id: pipeEpId });
  }

  function startPipeStep(params) {
    if (pipeStepEs) { pipeStepEs.close(); pipeStepEs = null; }
    fetch('/stop', { method: 'POST' }).catch(() => {});   // stop any running proc

    // Track which stage range is running so Pipeline tab doesn't show stale ✓
    pipeRunning = (params.type === 'llm') ? { from: params.from, to: params.to } : null;
    refreshPipeline();   // immediately clear stale ✓ in Pipeline tab

    // Route output to the Run tab — switch there and clear the output box
    switchTab('run');
    clearOutput();
    setStatus('running');

    const profile = renderProd ? 'high' : 'preview_local';
    let url;
    if (params.type === 'llm') {
      url = '/stream?story_file=' + encodeURIComponent(params.story_file) +
            '&from=' + params.from + '&to=' + params.to +
            '&test=' + (testMode ? '1' : '0') +
            '&profile=' + profile;
    } else if (params.type === 'locale') {
      url = '/run_locale?slug='   + encodeURIComponent(params.slug) +
            '&ep_id='  + encodeURIComponent(params.ep_id) +
            '&locale=' + encodeURIComponent(params.locale) +
            '&profile=' + profile +
            (params.from_step ? '&from=' + encodeURIComponent(params.from_step) : '');
    } else {
      url = '/run_step?step='   + encodeURIComponent(params.step) +
            '&slug='   + encodeURIComponent(params.slug) +
            '&ep_id='  + encodeURIComponent(params.ep_id) +
            '&locale=' + encodeURIComponent(params.locale) +
            '&profile=' + profile;
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

    # Detect locale variant files (StoryPrompt.{locale}.json)
    stage8_done = False
    if os.path.isdir(ep_dir):
        stage8_done = any(
            f.startswith("StoryPrompt.") and f.endswith(".json") and f != "StoryPrompt.json"
            for f in os.listdir(ep_dir)
        )

    # Stage 0 renames pipeline_vars.sh → pipeline_vars.{story}.sh, so check
    # for any matching per-story vars file as the stage_0 done indicator.
    import glob as _glob
    _vars_files = _glob.glob(os.path.join(PIPE_DIR, "pipeline_vars.*.sh"))
    _stage0_done = bool(_vars_files) and check(root("episode_direction.txt"))

    llm_stages = {
        "stage_0": {
            "done": _stage0_done,
            "artifacts": [root_rel("episode_direction.txt")],
        },
        "stage_1": {
            "done": check(ep("StoryPrompt.json")),   # proxy: stage 1 done if stage 2 output exists
            "artifacts": [],
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
            "done": check(ep("AssetManifest_draft.shared.json")) and check(ep("AssetManifest_draft.en.json")),
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
        if loc != "en" and check(os.path.join(ep_dir, "renders", loc, "youtube_dubbed.aac"))
    ]

    # Detect which story_N.txt produced this episode by scanning pipeline_vars.*.sh
    story_file_detected = ""
    for _vp in sorted(_glob.glob(os.path.join(PIPE_DIR, "pipeline_vars.*.sh")), reverse=True):
        try:
            _vc = open(_vp, encoding="utf-8").read()
            _slug_match = (f'PROJECT_SLUG="{slug}"' in _vc or f"PROJECT_SLUG={slug}\n" in _vc
                           or f"PROJECT_SLUG={slug}\r" in _vc)
            _ep_match   = (f'EPISODE_ID="{ep_id}"' in _vc or f"EPISODE_ID={ep_id}\n" in _vc
                           or f"EPISODE_ID={ep_id}\r" in _vc)
            if _slug_match and _ep_match:
                _m = re.match(r"pipeline_vars\.(.+)\.sh$", os.path.basename(_vp))
                if _m:
                    story_file_detected = f"{_m.group(1)}.txt"
                    break
        except Exception:
            pass

    # ── VoiceCast.json (project-level, written by Stage 0) ───────────────
    voice_cast = None
    vc_path = os.path.join(PIPE_DIR, "projects", slug, "VoiceCast.json")
    if os.path.isfile(vc_path):
        try:
            with open(vc_path, encoding="utf-8") as _f:
                voice_cast = json.load(_f)
        except Exception:
            pass

    return {
        "slug": slug, "ep_id": ep_id,
        "llm_stages": llm_stages,
        "locales": locales,
        "locale_steps": locale_steps,   # kept for /run_step recovery endpoint
        "ready_videos": ready_videos,
        "ready_dubbed": ready_dubbed,
        "story_file": story_file_detected,
        "voice_cast": voice_cast,
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


def _build_step_cmd(step: str, slug: str, ep_id: str, locale: str,
                    profile: str = "preview_local") -> list | None:
    """Build command list for a post-processing step."""
    ep_dir   = os.path.join(PIPE_DIR, "projects", slug, "episodes", ep_id)
    code_dir = os.path.join(PIPE_DIR, "code", "http")

    def ep(f): return os.path.join(ep_dir, f)

    if step == "gen_music_clip":
        return [
            "python3", os.path.join(code_dir, "gen_music_clip.py"),
            "--manifest", ep("AssetManifest_draft.shared.json"),
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
        return [
            "python3", os.path.join(code_dir, "gen_render_plan.py"),
            "--manifest", ep(f"AssetManifest_merged.{locale}.json"),
            "--media",    ep(f"AssetManifest.media.{locale}.json"),
            "--profile",  profile,
        ]
    elif step == "render_video":
        out_dir = os.path.join(ep_dir, "renders", locale)
        return [
            "python3", os.path.join(code_dir, "render_video.py"),
            "--plan",    ep(f"RenderPlan.{locale}.json"),
            "--locale",  locale,
            "--out",     out_dir,
            "--profile", profile,
        ]
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
            params        = parse_qs(parsed.query)
            story_file    = unquote_plus(params.get("story_file", [""])[0]).strip()
            from_stage    = params.get("from",    ["0"])[0].strip()
            to_stage      = params.get("to",      ["9"])[0].strip()
            test_mode     = params.get("test",    ["1"])[0].strip() == "1"
            render_profile = params.get("profile", ["preview_local"])[0].strip()
            if render_profile not in ("preview_local", "draft_720p", "high"):
                render_profile = "preview_local"

            # Sanitise: digits only, 0–9
            from_stage = str(max(0, min(10, int(from_stage)))) if from_stage.isdigit() else "0"
            to_stage   = str(max(0, min(10, int(to_stage))))   if to_stage.isdigit()   else "10"

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

            # Build subprocess environment
            run_env = os.environ.copy()
            run_env.pop("CLAUDECODE", None)   # prevent nested-session guard from firing
            if test_mode:
                run_env["MODEL"] = "haiku"   # cheapest model for all stages
            run_env["RENDER_PROFILE"] = render_profile   # preview_local or high

            client = self.client_address
            proc   = None
            try:
                proc = subprocess.Popen(
                    ["bash", "run.sh", story_file, from_stage, to_stage],
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
            self.end_headers()
            self.wfile.write(body)

        # SSE stream for a single post-processing step
        elif parsed.path == "/run_step":
            params  = parse_qs(parsed.query)
            step    = unquote_plus(params.get("step",    [""])[0]).strip()
            slug    = unquote_plus(params.get("slug",    [""])[0]).strip()
            ep_id   = unquote_plus(params.get("ep_id",   [""])[0]).strip()
            locale  = unquote_plus(params.get("locale",  [""])[0]).strip()
            profile = unquote_plus(params.get("profile", ["preview_local"])[0]).strip()

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            cmd = _build_step_cmd(step, slug, ep_id, locale, profile)
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

                    cmd = _build_step_cmd(step, slug, ep_id, locale, profile)
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
                    "aac": "audio/aac"}.get(ext, "application/octet-stream")

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
        silent = {"/", "/stop", "/next_story_num", "/check_episode",
                  "/list_projects", "/view_artifact", "/list_stories",
                  "/pipeline_status", "/serve_media", "/run_locale"}
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
