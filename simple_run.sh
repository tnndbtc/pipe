#!/usr/bin/env bash
# =============================================================================
# simple_run.sh — Simple Narration Pipeline Entry Point
# =============================================================================
#
# TWO OPERATING MODES:
#
# ── MODE 1: CLIPS (--input_folder) ───────────────────────────────────────────
#   Use when AI clip provider delivers video clips.
#   - Finds all .mp4 files in --input_folder, sorted by filename in natural order
#     (e.g. clip2.mp4 comes before clip12.mp4), excludes output.mp4
#   - Stitches them with 0.4s xfade dissolve (hardcoded, tuned default)
#   - Transcribes the final stitched output.mp4 with Whisper (whole-video).
#     This is the ONLY correct way to get subtitle timestamps — per-clip
#     Whisper + manual offset arithmetic is WRONG because xfade overlaps
#     accumulate into multi-second drift by the end of the video.
#     output.mp4 is always the ground truth. Never transcribe individual clips.
#   - Burns subtitles into video
#   No Azure TTS involved. Output: projects/<slug>/episodes/<id>/renders/<locale>/output.mp4
#
#   Usage:
#     ./simple_run.sh --input_folder /path/to/clips/dir [--story       /path/to/story.txt]
#
# ── MODE 2: TTS (--story, default) ───────────────────────────────────────────
#   Given a story.md, a background image/video, and a voice_config.json,
#   produces output.mp4 via Azure TTS with zero manual steps.
#
#   Usage:
#     ./simple_run.sh --story  /tmp/story.txt --image /tmp/cover.jpg --voice  /tmp/voice_config.json --out    /tmp/output.mp4
#
#   Optional flags:
#     --title  "My Title"     Override title (default: auto-extracted from story.md)
#     --locale en             BCP-47 locale (default: en)
#     --profile preview_local Render quality profile (default: preview_local)
#     --episode s01e01        Episode ID (default: s01e01)
#     --skip-sections "A,B"   Section headings to skip (merged with voice_config)
#     --no-default-skips      Disable built-in default skip list
#     --alt N                 Use alternatives[N] from voice_config (0-based)
#     --title-card            Burn title card for first 1s then fade out
#     --subtitles             Burn subtitles synced to TTS timing
#
# ── FUTURE: MODE 2 AUTO-CALIBRATION (not yet implemented) ───────────────────
#   Problem: Azure TTS duration is non-deterministic. To make TTS audio match
#   a target video duration (e.g. option_b_full_xfade.mp4 = 29.33s), the
#   azure_rate parameter must be tuned automatically.
#
#   Proposed approach (2-round loop inside simple_run.sh):
#     Round 1 — Run TTS at default azure_rate (or prior cached baseline).
#               Measure actual output duration D_actual.
#               Compute correction: new_rate = (D_actual/D_target - 1) * 100%
#               Update VOPlan azure_rate field.
#     Round 2 — Re-run TTS with corrected rate.
#               Verify abs(D_actual2 - D_target) / D_target < TOLERANCE (e.g. 2%).
#               If still off, do Round 3 with same formula.
#
#   Key constants to calibrate per voice/style (cache in projects/resources/):
#     baseline_cps: chars/sec at azure_rate=0% for a given voice+style
#     For zh-CN-XiaoxiaoNeural newscast: baseline_cps ≈ 5.22 c/s (measured 2026-04-20)
#
#   Trigger condition:
#     Add --target-dur <seconds> flag. When set, enable calibration loop.
#     Max rounds: 3. Tolerance: ±2%.
#
#   See: /home/tnnd/data/code/clips_stitch.txt for background on why this is needed.
#
# Requires:
#   - Python 3.10+ with pipe virtualenv (or system packages)
#   - ffmpeg (system)
#   - AZURE_SPEECH_KEY env var set (TTS mode only)
#   - AZURE_SPEECH_REGION env var set (default: eastus, TTS mode only)
#
# =============================================================================

set -euo pipefail

# ── Script directory (repo root) ─────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ──────────────────────────────────────────────────────────────────
CONFIG="code/http/simple_narration.config.json"
STORY=""
IMAGE=""
VOICE=""
TITLE=""
LOCALE=""
PROFILE=""
EPISODE="s01e01"
SKIP_SECTIONS=""
NO_DEFAULT_SKIPS=""
ALT=""
TITLE_CARD=""
SUBTITLES=""
VIDEO_EFFECT=""
# clips mode
INPUT_FOLDER=""
SPEED=""          # playback speed multiplier (e.g. 0.8 for 80% speed; clips mode only)
SUBTITLE_SHIFT_MS=""  # ms to shift subtitle timestamps; negative = earlier, positive = later
                       # default 0 (story or whisper-only); use a negative value (e.g. -300)
                       # only if subtitles appear noticeably after speech starts —
                       # negative shifts also advance t_out, which causes premature
                       # subtitle transitions (wrong subtitle mid-sentence) at slow speeds
# clips mode — TTS regen
TTS_REGEN=""
TTS_MARGIN="0.05"
TTS_ROUNDS="2"
TTS_CPS_MAP="${HOME}/.config/pipe/tts_cps_map.json"
TTS_FORCE=""
UNIFORM_RATE=""       # use one fixed TTS rate (host avg CPS); adjust clip video speed per-clip

# ── Help ──────────────────────────────────────────────────────────────────────
print_help() {
  cat <<'HELP'
USAGE
  ./simple_run.sh [OPTIONS]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 MODE 1 — CLIPS  (--input_folder)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Stitch AI-provided video clips with xfade dissolve, transcribe with Whisper,
  and burn subtitles. No Azure TTS involved.

  --input_folder  PATH   Directory containing .mp4 clips            (required)
  --story         PATH   Story text; passed to Whisper as prompt     (optional)
  --speed         N      Playback speed multiplier, e.g. 0.8        (optional)
  --subtitle_shift_ms N  Shift subtitle timestamps in ms (+/-)      (optional)

  TTS re-generation (clips mode only):
  --tts-regen            Re-synthesise TTS audio per clip to match clip durations
  --tts-margin    N      Acceptable timing error ratio   (default: 0.05)
  --tts-rounds    N      Max TTS calibration rounds      (default: 2)
  --tts-force            Force re-synthesis even if WAV is cached
  --uniform-rate         Measure host avg CPS from clips; generate all TTS at
                         that one fixed rate; adjust per-clip video speed to
                         match lip-movement window to TTS duration

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 MODE 2 — TTS  (--story, default)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Generate output.mp4 from a story file using Azure TTS over a background
  image or video. Background and voice are read from --config.

  --story    PATH    Story .txt file                                (required)
  --config   PATH    Config JSON (default: code/http/simple_narration.config.json)
                     Provides: background, narrator, locale, profile,
                     title_card, subtitles, skip_sections.
                     Output goes to projects/<slug>/episodes/<id>/renders/<locale>/output.mp4

  --title    "TEXT"  Override title (default: auto-extracted from story)
  --locale   LOCALE  BCP-47 locale             (default: en)
  --profile  NAME    Render quality profile    (default: preview_local)
  --episode  ID      Episode ID               (default: s01e01)
  --skip-sections "A,B"  Section headings to skip (merged with config)
  --no-default-skips     Disable built-in default skip list
  --alt      N       Use alternatives[N] from voice_config (0-based)
  --title-card       Burn title card for first 1 s then fade out
  --subtitles        Burn subtitles synced to TTS timing

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 COMMON FLAGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  -h, --help    Print this help message and exit

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

── EXAMPLE 1: Clips provided ──────────────────────────────────────────────────
  Stitch the .mp4 files in the clips folder, transcribe with Whisper (using the
  story as an initial prompt for better accuracy), and burn subtitles.

  ./simple_run.sh --input_folder /mnt/shared/story/my_episode/clips/ --story /mnt/shared/story/my_episode/story.txt

── EXAMPLE 2: Clips provided + re-generate TTS per clip ───────────────────────
  Same as Example 1, but also re-synthesises TTS audio for every clip so the
  spoken narration is tightly timed to each clip's duration.  Requires
  AZURE_SPEECH_KEY and AZURE_SPEECH_REGION env vars and a narrator entry in
  the config (or a --voice file).

  ./simple_run.sh --input_folder /mnt/shared/story/my_episode/clips/ --story        /mnt/shared/story/my_episode/story.txt --tts-regen

── EXAMPLE 3: Story + config → output.mp4 (background from config) ────────────
  Fully automated TTS pipeline: background image/video, voice, locale, profile,
  and subtitle settings are all read from simple_narration.config.json — only
  the story file needs to be specified on the command line.
  Output is written automatically to projects/<slug>/episodes/<id>/renders/<locale>/output.mp4

  ./simple_run.sh --story  /mnt/shared/story/my_episode/story.txt --config code/http/simple_narration.config.json

  Sample config (code/http/simple_narration.config.json):
    {
      "background": "/mnt/shared/story/news/newsroom_loop/option_b_full_xfade.mp4",
      "locale": "zh-Hans",
      "profile": "high",
      "title_card": false,
      "subtitles": true,
      "narrator": {
        "zh-Hans": {
          "azure_voice": "zh-CN-XiaoxiaoNeural",
          "azure_style": "newscast",
          "azure_rate":  "+45%"
        }
      },
      "skip_sections": ["Sources"]
    }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - Python 3.10+ with pipe virtualenv (or system packages)
  - ffmpeg (system)
  - AZURE_SPEECH_KEY    env var  (TTS / tts-regen modes only)
  - AZURE_SPEECH_REGION env var  (default: eastus)
HELP
}

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)           CONFIG="$2";         shift 2 ;;
    --story)            STORY="$2";          shift 2 ;;
    --image)            IMAGE="$2";          shift 2 ;;
    --voice)            VOICE="$2";          shift 2 ;;
    --title)            TITLE="$2";          shift 2 ;;
    --locale)           LOCALE="$2";         shift 2 ;;
    --profile)          PROFILE="$2";        shift 2 ;;
    --episode)          EPISODE="$2";        shift 2 ;;
    --skip-sections)    SKIP_SECTIONS="$2";  shift 2 ;;
    --no-default-skips) NO_DEFAULT_SKIPS="1"; shift ;;
    --alt)              ALT="$2";            shift 2 ;;
    --title-card)       TITLE_CARD="1";      shift ;;
    --subtitles)        SUBTITLES="1";       shift ;;
    --input_folder)     INPUT_FOLDER="$2";   shift 2 ;;
    --speed)            SPEED="$2";          shift 2 ;;
    --subtitle_shift_ms) SUBTITLE_SHIFT_MS="$2"; shift 2 ;;
    -h|--help)
      print_help
      exit 0
      ;;
    --tts-regen)        TTS_REGEN="1";        shift ;;
    --tts-margin)       TTS_MARGIN="$2";      shift 2 ;;
    --tts-rounds)       TTS_ROUNDS="$2";      shift 2 ;;
    --tts-cps-map)      TTS_CPS_MAP="$2";     shift 2 ;;
    --tts-force)        TTS_FORCE="1";        shift ;;
    --uniform-rate)     UNIFORM_RATE="1";     shift ;;
    *)
      echo "[ERROR] Unknown flag: $1" >&2
      exit 1
      ;;
  esac
done


# ── CLIPS MODE ────────────────────────────────────────────────────────────────
if [[ -n "$INPUT_FOLDER" ]]; then
  [[ ! -d "$INPUT_FOLDER" ]] && { echo "[ERROR] --input_folder not found: $INPUT_FOLDER" >&2; exit 1; }

  _story_json=""

  echo "════════════════════════════════════════════════════════════"
  echo "  simple_run.sh — clips mode"
  echo "  Folder    : $INPUT_FOLDER"
  echo "  Story     : ${STORY:-none}"
  echo "  Speed     : ${SPEED:-1.0}"
  echo "  TTS-regen : ${TTS_REGEN:-no}"
  echo "  Uniform-rate: ${UNIFORM_RATE:-no}"
  echo "════════════════════════════════════════════════════════════"
  echo ""

  # ── Derive LOCALE, EPISODE, slug, EP_DIR before anything writes to disk ──────
  if [[ -z "$LOCALE" && -n "$CONFIG" && -f "$CONFIG" ]]; then
    LOCALE="$(python3 -c "import json; d=json.load(open('$CONFIG',encoding='utf-8')); print(d.get('locale',''))" 2>/dev/null || true)"
  fi
  [[ -z "$LOCALE"  ]] && LOCALE="en"
  [[ -z "$EPISODE" ]] && EPISODE="s01e01"

  # Prefer the bare story filename stem as slug (stable, human-readable).
  # Fall back to content-derived slug (with random suffix) for non-slug filenames.
  _slug="$(python3 - "${STORY:-}" 2>/dev/null << 'CLIPS_SLUG_EOF'
import re, os, sys
fn = os.path.splitext(os.path.basename(sys.argv[1] if len(sys.argv) > 1 else ''))[0]
if fn and re.match(r'^[a-zA-Z0-9_\-]+$', fn):
    print(fn); sys.exit(0)
sys.exit(1)
CLIPS_SLUG_EOF
  )"

  if [[ -z "$_slug" ]]; then
    # Filename has non-slug chars — fall back to first heading/line of text
    _clips_title=""
    if [[ -n "$STORY" && -f "$STORY" ]]; then
      _clips_title="$(python3 -c "
import re, sys
text = open('$STORY', encoding='utf-8').read()
for line in text.splitlines():
    m = re.match(r'^#{1,6}\s+(.+)', line)
    if m: print(m.group(1).strip()); sys.exit(0)
for line in text.splitlines():
    s = line.strip()
    if s: print(s); sys.exit(0)
" 2>/dev/null)"
    fi
    [[ -z "$_clips_title" ]] && _clips_title="$(basename "$INPUT_FOLDER")"
    _slug="$(python3 -c "
import re, unicodedata, secrets
orig = '''$_clips_title'''
t = unicodedata.normalize('NFKD', orig).encode('ascii','ignore').decode('ascii')
t = t.lower().strip()
t = re.sub(r'[^\w\s-]', '', t)
t = re.sub(r'[\s_]+', '-', t)
t = re.sub(r'-{2,}', '-', t)
t = t.strip('-')[:32]
rand6 = secrets.token_hex(3)
print((t + '-' + rand6) if t else 'story-' + rand6)
" 2>/dev/null || echo "story-$(python3 -c 'import secrets; print(secrets.token_hex(3))')")"
  fi

  EP_DIR="projects/${_slug}/episodes/${EPISODE}"
  mkdir -p "$EP_DIR"
  _tts_dir="${EP_DIR}/assets/${LOCALE}/audio/vo"
  _renders_dir="${EP_DIR}/renders/${LOCALE}"
  mkdir -p "$_renders_dir"
  _out_video="${_renders_dir}/output.mp4"
  _out_srt="${_renders_dir}/output.${LOCALE}.srt"

  # ── Parse story → EP_DIR/.story_parsed.json ───────────────────────────────
  # Handles ## Title lines (not spoken), - separator lines (skipped),
  # and content lines (1:1 with clips).  Also derives display_text (no markup)
  # and ssml_text (pinyin <word|pinyin> → <phoneme> tags) for each line.
  if [[ -n "$STORY" && -f "$STORY" ]]; then
    _story_json="${EP_DIR}/.story_parsed.json"
    echo "  Parsing story: $(basename "$STORY")"
    python3 - "$STORY" "$_story_json" << 'STORY_PARSE_EOF'
import sys, json, re

story_path = sys.argv[1]
out_path   = sys.argv[2]

lines  = open(story_path, encoding='utf-8').read().splitlines()
text   = []    # raw content lines (may have <word|pinyin> markup)
titles = []    # [{title, clip_index, is_first}]

for line in lines:
    stripped = line.strip()
    if stripped.startswith('##'):
        title    = stripped[2:].strip()
        is_first = len(titles) == 0
        titles.append({'title': title, 'clip_index': len(text), 'is_first': is_first})
    elif stripped == '-' or not stripped:
        pass   # separator or blank — skip
    else:
        text.append(stripped)

def to_display(raw):
    """Strip <word|pinyin> markup to plain text for subtitles and char count."""
    return re.sub(r'<([^|>]+)\|[^>]+>', r'\1', raw)

def to_ssml(raw):
    """Convert <word|pinyin> to <phoneme alphabet="pinyin" ph="..."> for Azure TTS."""
    return re.sub(r'<([^|>]+)\|([^>]+)>',
                  r'<phoneme alphabet="pinyin" ph="\2">\1</phoneme>', raw)

result = {
    'text':         text,
    'display_text': [to_display(t) for t in text],
    'ssml_text':    [to_ssml(t)    for t in text],
    'chars':        [len(to_display(t)) for t in text],
    'titles':       titles,
}
json.dump(result, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(f"    {len(text)} content lines, {len(titles)} stories")
for t in titles:
    m = '[main]' if t['is_first'] else '[sub] '
    print(f"    {m} clip {t['clip_index']:3d}: {t['title']}")
STORY_PARSE_EOF
    echo ""
  fi

  # ── Step 0: Clip pre-flight + Whisper ordering ────────────────────────────
  # Phase A: MD5 dedup — auto-removes exact byte-copies, logs what was kept.
  # Phase B: faster-whisper transcription + greedy matching.  Handles N≠M:
  #   N < M → reports missing lines + writes status_report.txt, exits 3.
  #   N > M → renames older same-line clip to .mp4.bak, keeps newer, continues.
  #   N = M → orders and renames to clip01.mp4 … clipNN.mp4.
  _clip_order_json="${INPUT_FOLDER}/.clip_order.json"
  if [[ -n "$_story_json" && -f "$_story_json" ]]; then
    echo "  STEP 0 — Clip pre-flight + Whisper ordering"

    # ── Phase A: exact duplicate detection + auto-removal ──────────────────
    python3 - "$INPUT_FOLDER" << 'CLIP_DEDUP_EOF'
import sys, os, glob, re, hashlib

def natural_key(p):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', os.path.basename(p))]

def md5_file(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            buf = f.read(chunk)
            if not buf: break
            h.update(buf)
    return h.hexdigest()

folder = sys.argv[1]
clips  = sorted(
    [f for f in glob.glob(os.path.join(folder, '*.mp4'))
     if os.path.basename(f) != 'output.mp4'],
    key=natural_key)

if not clips:
    sys.exit(0)

print(f"  Phase A — dedup check ({len(clips)} clips)...")
by_hash = {}
for c in clips:
    by_hash.setdefault(md5_file(c), []).append(c)

removed = 0
for group in by_hash.values():
    if len(group) < 2:
        continue
    keep = group[0]
    for dupe in group[1:]:
        try:
            os.remove(dupe)
            print(f"  [info] Duplicate removed : {os.path.basename(dupe)}")
            print(f"         (identical to      {os.path.basename(keep)} — kept)")
            removed += 1
        except OSError as e:
            print(f"  [warn] Could not remove duplicate "
                  f"{os.path.basename(dupe)}: {e}")

if removed:
    print(f"  Phase A — removed {removed} exact duplicate(s), continuing")
else:
    print(f"  Phase A — no exact duplicates")
CLIP_DEDUP_EOF

    # ── Phase B: Whisper ordering + missing/extra resolution ───────────────
    python3 - "$INPUT_FOLDER" "$_story_json" "$_clip_order_json" \
              "$EP_DIR" "${PIPE_SERVER_URL:-http://localhost:8000}" \
              "${EPISODE:-s01e01}" << 'CLIP_ORDER_EOF'
import sys, os, glob, re, json, tempfile, subprocess, urllib.request
from collections import Counter
from datetime import datetime

folder       = sys.argv[1]
story_json_p = sys.argv[2]
order_out    = sys.argv[3]
ep_dir       = sys.argv[4]
server_url   = sys.argv[5]
ep_id        = sys.argv[6]

def natural_key(p):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', os.path.basename(p))]

def title_for_line(li, titles):
    """Return the story title covering line index li."""
    label = ''
    for t in titles:
        if t['clip_index'] <= li:
            label = t['title']
    return (label[:30] if label else f'line {li}')

story = json.load(open(story_json_p, encoding='utf-8'))
lines = story.get('display_text', story.get('text', []))
if not lines:
    print("  [skip] no story lines — keeping filename order"); sys.exit(0)

M = len(lines)

clips = sorted(
    [f for f in glob.glob(os.path.join(folder, '*.mp4'))
     if os.path.basename(f) != 'output.mp4'],
    key=natural_key)
N = len(clips)

if N != M:
    print(f"  [info] {N} clips vs {M} story lines — "
          f"will identify missing/extra via whisper")

# ── Whisper model options ──────────────────────────────────────────
# Library: faster-whisper (pip install faster-whisper)
# Uses CTranslate2 + int8 quantization on CPU — 4× faster than
# openai-whisper at the same accuracy.
#
# Model comparison (4-core CPU, 8 GB RAM, ~5s clips, 27 clips):
#   "medium"         ~400 MB RAM  first-run ~2 min  re-run ~4s   ← CURRENT
#   "large-v3-turbo" ~900 MB RAM  first-run ~5 min  re-run ~9s   ← upgrade if ordering mistakes
#   "large-v3"       ~1.5 GB RAM  first-run ~7 min  re-run ~13s  ← best quality, diminishing return
#
# To switch: change MODEL_SIZE below.  Cache auto-regenerates on next run.
# GPU: all options ~4× faster; prefer large-v3-turbo or large-v3 on GPU.
# ─────────────────────────────────────────────────────────────────
MODEL_SIZE = "medium"

cache_path = os.path.join(folder, '.whisper_order_cache.json')
cache = {}
try:
    cache = json.load(open(cache_path, encoding='utf-8'))
except Exception:
    pass

try:
    from faster_whisper import WhisperModel
    _model = WhisperModel(MODEL_SIZE, device='cpu', compute_type='int8')
    print(f"  faster-whisper model loaded ({MODEL_SIZE})")
except ImportError:
    print("  [ERROR] faster-whisper is not installed on this machine.")
    print("  This is required to transcribe and order AI-generated clips.")
    print("  Install it with:")
    print("    pip install faster-whisper")
    print("  Then re-run the same command.")
    sys.exit(1)

# Traditional → Simplified converter (opencc-python-reimplemented).
# Whisper defaults to Traditional Chinese for Mandarin audio; story text
# is Simplified.  Without conversion, 國/国, 們/们 etc. score as mismatches
# and clip-to-line assignment degrades.  We convert both sides to Simplified
# before scoring so the character overlap is computed on the same script.
# The initial_prompt below also biases Whisper toward Simplified output,
# but opencc is the authoritative fix regardless of what Whisper emits.
try:
    import opencc as _opencc_mod
    _cc = _opencc_mod.OpenCC('t2s')   # Traditional → Simplified
    def _to_simplified(s): return _cc.convert(s)
    print("  opencc loaded — Trad→Simplified normalisation enabled")
except ImportError:
    print("  [warn] opencc-python-reimplemented not installed — "
          "Traditional/Simplified mismatch may reduce match accuracy.")
    print("  Install with: pip install opencc-python-reimplemented")
    def _to_simplified(s): return s   # no-op fallback

def transcribe_clip(clip_path):
    """Return (text, seg_list) where seg_list is [{start, end, text}, ...].
    Segments are stored in the order cache so TTS-B can reuse them for
    S_start/S_end timing without a second transcription pass."""
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, 'clip.wav')
        r = subprocess.run(
            ['ffmpeg', '-y', '-i', clip_path, '-ar', '16000', '-ac', '1',
             '-c:a', 'pcm_s16le', wav, '-loglevel', 'error'],
            capture_output=True)
        if r.returncode != 0:
            return '', []
        # initial_prompt biases the decoder toward Simplified Chinese output;
        # opencc in norm() is the authoritative normalisation backstop.
        segs, _ = _model.transcribe(wav, language='zh',
                                    initial_prompt='以下是简体中文内容。')
        seg_list = [{'start': s.start, 'end': s.end, 'text': s.text}
                    for s in segs]
        text = ''.join(s['text'] for s in seg_list).strip()
        return text, seg_list

def norm(s):
    return re.sub(r'\s+', '', _to_simplified(s).lower())

def overlap_score(a, b):
    na, nb = norm(a), norm(b)
    ca, cb = Counter(na), Counter(nb)
    return sum((ca & cb).values()) / max(len(na), len(nb), 1)

# Cache invalidation: re-transcribe if stale, wrong library, or missing text
need_transcribe = [
    c for c in clips
    if abs(cache.get(os.path.basename(c), {}).get('mtime', 0)
           - os.path.getmtime(c)) >= 1.0
    or cache.get(os.path.basename(c), {}).get('library') != 'faster-whisper'
    or 'text' not in cache.get(os.path.basename(c), {})]

if need_transcribe:
    print(f"  Transcribing {len(need_transcribe)}/{N} clips "
          f"(cache: {N - len(need_transcribe)} hits)...")
    for clip in need_transcribe:
        cn         = os.path.basename(clip)
        text, segs = transcribe_clip(clip)
        cache[cn]  = {'mtime': os.path.getmtime(clip), 'text': text,
                      'library': 'faster-whisper', 'segments': segs}
        print(f"    {cn}: {text[:60]!r}")
    json.dump(cache, open(cache_path, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
else:
    print(f"  Whisper cache valid ({N} clips, faster-whisper/{MODEL_SIZE})")

clip_texts = [cache.get(os.path.basename(c), {}).get('text', '') for c in clips]

# Build N×M score matrix
scores = [[overlap_score(clip_texts[i], lines[j])
           for j in range(M)] for i in range(N)]

# Greedy best-match assignment
assigned   = {}
used_lines = set()
for sc, ci, li in sorted(
        [(scores[i][j], i, j) for i in range(N) for j in range(M)],
        reverse=True):
    if ci not in assigned and li not in used_lines:
        assigned[ci] = li
        used_lines.add(li)
    if len(assigned) == min(N, M):
        break

extra_clips = sorted(set(range(N)) - set(assigned.keys()))

# ── B.4: Extra clip resolution (N > M) ───────────────────────────
resolved_extras    = set()
replaced_clips_log = []

for ci in extra_clips:
    best_li = max(range(M), key=lambda j: scores[ci][j])
    best_sc = scores[ci][best_li]

    if best_sc > 0.15:
        rival_ci = [k for k, v in assigned.items() if v == best_li][0]
        clip_new, clip_old = sorted(
            [clips[ci], clips[rival_ci]],
            key=lambda p: os.path.getmtime(p), reverse=True)
        bak_path = clip_old + '.bak'
        try:
            os.rename(clip_old, bak_path)
        except OSError as e:
            print(f"  [warn] Could not rename {os.path.basename(clip_old)} "
                  f"to .bak: {e} — keeping both, line {best_li} may be wrong")
            continue
        del assigned[rival_ci]
        new_ci = clips.index(clip_new)
        assigned[new_ci] = best_li
        resolved_extras.add(ci)
        replaced_clips_log.append({
            'bak':  os.path.basename(bak_path),
            'kept': os.path.basename(clip_new),
            'line': best_li,
        })
        print(f"  [info] Line {best_li}: kept newer  {os.path.basename(clip_new)}")
        print(f"         renamed older {os.path.basename(clip_old)} → .mp4.bak")
    else:
        print(f"  [WARN] Unknown extra clip (no story match): "
              f"{os.path.basename(clips[ci])} — will append at end")

unmatched_extras = [ci for ci in extra_clips if ci not in resolved_extras]

# Recompute after resolution
used_lines    = set(assigned.values())
missing_lines = sorted(set(range(M)) - used_lines)

# ── B.5: Report missing clips (hard error) ───────────────────────
if missing_lines:
    ts             = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    story_basename = os.path.basename(story_json_p).replace('.story_parsed.json', '')

    lines_block = ''
    for li in missing_lines:
        label = title_for_line(li, story['titles'])
        lines_block += f"  Line {li:3d}  [{label}]\n"
        lines_block += f"           {lines[li][:80]}\n"

    report = (
        f"[CLIP VALIDATION] Missing clips detected — {ts}\n"
        f"Story  : {story_basename}\n"
        f"Folder : {folder}\n"
        f"Clips  : {N} found, {M} expected\n\n"
        f"Missing clips ({len(missing_lines)}):\n"
        f"{lines_block}\n"
        f"Action: regenerate the above clips in Grok, add to folder, re-run.\n"
        f"Whisper cache is warm — only new clips will be re-transcribed.\n"
    )

    print(f"\n  [ERROR] Missing clips — no video for these story lines:")
    for li in missing_lines:
        label = title_for_line(li, story['titles'])
        print(f"    Line {li:3d}  [{label}]  {lines[li][:70]}")
    print(f"  ACTION: Regenerate the listed clips in Grok and re-run.")

    # Write status_report.txt (primary path)
    os.makedirs(ep_dir, exist_ok=True)
    sr_path = os.path.join(ep_dir, 'status_report.txt')
    try:
        with open(sr_path, 'a', encoding='utf-8') as fh:
            sep = '=' * 60
            fh.write(f"\n{sep}\n{ts}\n{sep}\n{report}\n")
        print(f"  STATUS REPORT written → {sr_path}")
    except OSError as e:
        print(f"  [warn] Could not write status_report.txt: {e}")
        print(f"  (missing clip details printed to terminal above)")

    # Attempt POST to server (secondary, optional)
    try:
        slug = os.path.basename(os.path.dirname(os.path.dirname(ep_dir)))
        body = json.dumps(
            {'slug': slug, 'ep_id': ep_id, 'text': report}).encode()
        req = urllib.request.Request(
            f'{server_url}/api/append_status_report',
            data=body, headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        pass  # server not running — file write above is the primary path

    sys.exit(3)

# ── B.6: Rename to clip01.mp4 … clipNN.mp4 (clean run only) ──────
ordered_idx = sorted(assigned.keys(), key=lambda ci: assigned[ci])
for rank, ci in enumerate(ordered_idx):
    old_path = clips[ci]
    new_name = f"clip{rank+1:02d}.mp4"
    new_path = os.path.join(folder, new_name)
    if os.path.basename(old_path) != new_name:
        try:
            os.rename(old_path, new_path)
            clips[ci] = new_path
            print(f"  [rename] {os.path.basename(old_path):<48} → {new_name}")
        except OSError as e:
            print(f"  [warn] rename failed: {os.path.basename(old_path)} "
                  f"→ {new_name}: {e} (using original name)")

# ── B.6b: Rekey whisper cache to post-rename basenames ───────────
# After B.6 renames grok_xxxx.mp4 → clip01.mp4 etc., the cache keys
# are stale (old basenames).  Rewrite the cache now so the next run
# gets cache hits instead of re-transcribing every clip.
new_cache = {}
for ci, entry_path in enumerate(clips):
    new_bn = os.path.basename(entry_path)          # post-rename name
    # find the matching entry under either the new or any old key
    entry = cache.get(new_bn)
    if entry is None:
        # fallback: scan cache for an entry whose mtime matches this file
        mt = os.path.getmtime(entry_path)
        for k, v in cache.items():
            if abs(v.get('mtime', -1) - mt) < 1.0:
                entry = v
                break
    if entry:
        new_cache[new_bn] = entry
json.dump(new_cache, open(cache_path, 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)

# ── B.7: Write .clip_order.json ───────────────────────────────────
ordered_clips = [os.path.basename(clips[i]) for i in ordered_idx]
match_scores  = [round(scores[i][assigned[i]], 3) for i in ordered_idx]
story_indices = [assigned[i] for i in ordered_idx]

json.dump({
    'ordered_clips':  ordered_clips,
    'match_scores':   match_scores,
    'story_indices':  story_indices,
    'missing_lines':  missing_lines,
    'missing_texts':  [lines[li] for li in missing_lines],
    'replaced_clips': replaced_clips_log,
    'extra_clips':    [os.path.basename(clips[ci]) for ci in unmatched_extras],
}, open(order_out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

print(f"\n  Clip order ({len(ordered_clips)} clips):")
for rank, name in enumerate(ordered_clips):
    si = story_indices[rank]; sc = match_scores[rank]
    print(f"    [{rank+1:2d}] {name:<20} → line {si:3d}  "
          f"score={sc:.2f}  {lines[si][:35]!r}")
CLIP_ORDER_EOF
    _clip_order_exit=$?
    echo ""
    if [[ $_clip_order_exit -ne 0 ]]; then
      echo "════════════════════════════════════════════════════════════"
      echo "  CLIP VALIDATION FAILED — see error above for details."
      echo "  Fix the issue, then re-run the same command."
      echo "════════════════════════════════════════════════════════════"
      exit $_clip_order_exit
    fi
  fi

  # ── TTS Regen ───────────────────────────────────────────────────────────────
  if [[ -n "$TTS_REGEN" ]]; then
    [[ -z "$CONFIG" || ! -f "$CONFIG" ]] && {
      echo "[ERROR] --tts-regen requires --config <path>" >&2; exit 1; }
    [[ -z "${AZURE_SPEECH_KEY:-}" ]] && {
      echo "[ERROR] AZURE_SPEECH_KEY env var not set" >&2; exit 1; }
    [[ -z "$_story_json" || ! -f "$_story_json" ]] && {
      echo "[ERROR] --tts-regen requires --story <path>" >&2; exit 1; }
    AZURE_SPEECH_REGION="${AZURE_SPEECH_REGION:-eastus}"
    mkdir -p "$_tts_dir"

    echo "  TTS REGEN — Azure TTS per clip"
    python3 - "$INPUT_FOLDER" "$_story_json" "$CONFIG" \
              "$TTS_CPS_MAP" "$TTS_MARGIN" "$TTS_ROUNDS" \
              "${TTS_FORCE:-0}" "$AZURE_SPEECH_REGION" "$EP_DIR" \
              "${UNIFORM_RATE:-0}" << 'TTS_REGEN_EOF'
import sys, os, json, glob, re, subprocess, tempfile, shutil

folder       = sys.argv[1]
story_json_p = sys.argv[2]
config_path  = sys.argv[3]
cps_map_path = sys.argv[4]
tts_margin   = float(sys.argv[5])
tts_rounds   = int(sys.argv[6])
tts_force    = sys.argv[7] == '1'
azure_region = sys.argv[8]
ep_dir       = sys.argv[9]   # projects/{slug}/episodes/{ep_id}
uniform_rate = sys.argv[10] == '1' if len(sys.argv) > 10 else False
azure_key    = os.environ['AZURE_SPEECH_KEY']

TTS_TOL = 0.05

# Hardcoded CPS anchors for known voices (seed data on first run)
KNOWN_ANCHORS = {
    'zh-CN-XiaoxiaoNeural:newscast': [{'rate': '0%', 'cps': 5.22}],
}

def natural_key(p):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', os.path.basename(p))]

def probe_dur(path):
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1', path],
        capture_output=True, text=True, check=True)
    return float(r.stdout.strip().split('=')[1])

def parse_rate(s): return float(s.replace('%', ''))
def fmt_rate(f):   return f"{(f - 1.0) * 100:+.0f}%"
def clamp_rate(f):
    pct = max(-50.0, min(200.0, (f - 1.0) * 100.0))
    return 1.0 + pct / 100.0

def predict_rate(needed_cps, vk, cps_map):
    """Interpolate/extrapolate azure_rate factor for needed_cps from CPS map."""
    pts = [(1.0 + parse_rate(a['rate']) / 100.0, a['cps'])
           for a in KNOWN_ANCHORS.get(vk, [])]
    pts += [(1.0 + parse_rate(m['rate']) / 100.0, m['cps'])
            for m in cps_map.get(vk, {}).get('measurements', [])[-20:]]
    pts.sort()
    if not pts: return 1.0
    if len(pts) == 1:
        f0, c0 = pts[0]; return clamp_rate(f0 * needed_cps / c0)
    cpss = [p[1] for p in pts]
    if needed_cps <= cpss[0]:
        f1, c1 = pts[0]; f2, c2 = pts[1]
        sl = (f2 - f1) / (c2 - c1) if c2 != c1 else 0
        r = clamp_rate(f1 + sl * (needed_cps - c1))
        if r < pts[0][0] - 0.20:
            print("  [WARN] CPS map extrapolating below range; R2 will correct")
        return r
    if needed_cps >= cpss[-1]:
        f1, c1 = pts[-2]; f2, c2 = pts[-1]
        sl = (f2 - f1) / (c2 - c1) if c2 != c1 else 0
        r = clamp_rate(f2 + sl * (needed_cps - c2))
        if r > pts[-1][0] + 0.20:
            print("  [WARN] CPS map extrapolating above range; R2 will correct")
        return r
    for i in range(len(pts) - 1):
        f1, c1 = pts[i]; f2, c2 = pts[i + 1]
        if c1 <= needed_cps <= c2:
            t = (needed_cps - c1) / (c2 - c1) if c2 != c1 else 0.5
            return clamp_rate(f1 + t * (f2 - f1))
    return 1.0

def upd_map(cps_map, vk, rate, cps, chars, dur):
    e = cps_map.setdefault(vk, {'measurements': []})
    e['measurements'].append({'rate': rate, 'cps': round(cps, 4),
                               'chars': chars, 'dur_s': round(dur, 3)})
    e['measurements'] = e['measurements'][-20:]

def save_map(cps_map, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json.dump(cps_map, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

def build_ssml(text, voice, style, sdeg, rate, pitch):
    return (f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' "
            f"xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='zh-CN'>"
            f"<voice name='{voice}'>"
            f"<mstts:express-as style='{style}' styledegree='{sdeg}'>"
            f"<prosody rate='{rate}' pitch='{pitch}'>{text}</prosody>"
            f"</mstts:express-as></voice></speak>")

def synthesize(ssml, out_path):
    import azure.cognitiveservices.speech as sdk
    cfg = sdk.SpeechConfig(subscription=azure_key, region=azure_region)
    cfg.set_speech_synthesis_output_format(
        sdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm)
    audio_cfg = sdk.audio.AudioOutputConfig(filename=out_path)
    synth = sdk.SpeechSynthesizer(speech_config=cfg, audio_config=audio_cfg)
    res = synth.speak_ssml_async(ssml).get()
    if res.reason != sdk.ResultReason.SynthesizingAudioCompleted:
        d = res.cancellation_details
        raise RuntimeError(f"TTS: {d.reason} - {d.error_details}")

# Load story + config
story   = json.load(open(story_json_p, encoding='utf-8'))
ssml_t  = story['ssml_text']
chars   = story['chars']

config  = json.load(open(config_path, encoding='utf-8'))
locale  = config.get('locale', 'zh-Hans')
narr_raw = config.get('narrator', {}).get(locale, {})
# Handle new multi-voice format: narrator[locale][voice_name] = {enabled, gender, notes, azure_*}
_voice_entries = {k: v for k, v in narr_raw.items() if isinstance(v, dict)}
if _voice_entries:
    _enabled = {n: c for n, c in _voice_entries.items() if c.get('enabled', False)}
    if len(_enabled) != 1:
        print(f"[ERROR] tts-regen: locale '{locale}' must have exactly 1 enabled voice, "
              f"found {len(_enabled)} "
              f"({'none' if not _enabled else ', '.join(_enabled)})", file=sys.stderr)
        sys.exit(1)
    _vname, narr = next(iter(_enabled.items()))
    print(f"  tts-regen voice: {_vname!r}")
else:
    narr = narr_raw  # old flat format
voice   = narr.get('azure_voice', 'zh-CN-XiaoxiaoNeural')
style   = narr.get('azure_style', 'newscast')
sdeg    = narr.get('azure_style_degree', 1.0)
pitch   = narr.get('azure_pitch', '0%')
vk      = f"{voice}:{style}"

# Collect original clips
tts_dir = os.path.join(ep_dir, 'assets', locale, 'audio', 'vo')
os.makedirs(tts_dir, exist_ok=True)

# Respect Whisper-determined clip order if available
_order_path = os.path.join(folder, '.clip_order.json')
if os.path.exists(_order_path):
    _order_data = json.load(open(_order_path))
    _all_mp4    = {os.path.basename(f)
                   for f in glob.glob(os.path.join(folder, '*.mp4'))
                   if os.path.basename(f) != 'output.mp4'}
    _ordered    = [n for n in _order_data['ordered_clips'] if n in _all_mp4]
    _unmatched  = sorted(_all_mp4 - set(_ordered), key=natural_key)
    clips       = [os.path.join(folder, n) for n in _ordered + _unmatched]
else:
    clips = sorted(
        [f for f in glob.glob(os.path.join(folder, '*.mp4'))
         if os.path.basename(f) != 'output.mp4'],
        key=natural_key)
N = len(clips)

if len(chars) != N:
    print(f"[ERROR] story has {len(chars)} lines but folder has {N} clips", file=sys.stderr)
    sys.exit(1)

clip_durs = [probe_dur(c) for c in clips]

# Load caches
meta_path  = os.path.join(tts_dir, '.metadata.json')
wo_path    = os.path.join(tts_dir, '.whisper_orig.json')
cps_map = {}; metadata = {}; wo_cache = {}
if not tts_force:
    for p, d in [(cps_map_path, cps_map), (meta_path, metadata), (wo_path, wo_cache)]:
        try: d.update(json.load(open(p, encoding='utf-8')))
        except: pass

# TTS-B: Whisper on original clips to get S[i] (lip-sync window)
need_w = [i for i, c in enumerate(clips)
          if tts_force
          or abs(wo_cache.get(os.path.basename(c), {}).get('mtime', 0)
                 - os.path.getmtime(c)) >= 1.0
          or 'S' not in wo_cache.get(os.path.basename(c), {})
    or 'S_start' not in wo_cache.get(os.path.basename(c), {})]

if need_w:
    # ── Reuse STEP 0 faster-whisper segments where available ─────────
    # STEP 0 already transcribed every clip with faster-whisper and stored
    # per-segment timestamps in .whisper_order_cache.json.  Read those now
    # so TTS-B doesn't have to run a second transcription pass.
    order_cache_path = os.path.join(folder, '.whisper_order_cache.json')
    order_cache = {}
    try:
        order_cache = json.load(open(order_cache_path, encoding='utf-8'))
    except Exception:
        pass

    still_need_w = []
    for i in need_w:
        cn = os.path.basename(clips[i])
        oc = order_cache.get(cn, {})
        oc_segs = oc.get('segments', [])
        if oc_segs and abs(oc.get('mtime', 0) - os.path.getmtime(clips[i])) < 1.0:
            S    = max((s['end']   for s in oc_segs), default=clip_durs[i] * 0.9)
            S_st = min((s['start'] for s in oc_segs), default=0.0)
            wo_cache[cn] = {'mtime': oc['mtime'], 'S': float(S),
                            'S_start': float(S_st), 'segments': oc_segs}
            print(f"    [{i+1}/{N}] {cn}: S={S_st:.2f}s–{S:.2f}s (order cache)")
        else:
            still_need_w.append(i)
    need_w = still_need_w

    if need_w:
        json.dump(wo_cache, open(wo_path, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)

    print(f"  TTS-B — Whisper on {len(need_w)}/{N} clips (model=small)"
          f"{' — all from order cache' if not need_w else ''}")
    if need_w:
        try:
            import whisper as _wm
            _wmod = _wm.load_model('small'); _api = True
        except ImportError:
            _api = False
            print("  [warn] whisper API unavailable — using S=clip_dur*0.9 fallback", file=sys.stderr)
        with tempfile.TemporaryDirectory() as tmp:
            for i in need_w:
                c = clips[i]; cn = os.path.basename(c); mt = os.path.getmtime(c)
                wav = os.path.join(tmp, f'c{i:04d}.wav')
                r = subprocess.run(['ffmpeg', '-y', '-i', c, '-ar', '16000', '-ac', '1',
                                    '-c:a', 'pcm_s16le', wav, '-loglevel', 'error'],
                                   capture_output=True)
                if r.returncode != 0:
                    wo_cache[cn] = {'mtime': mt, 'S': clip_durs[i] * 0.9,
                                    'S_start': 0.0, 'segments': []}
                    continue
                if _api:
                    res  = _wmod.transcribe(wav, task='transcribe', verbose=None, fp16=False)
                    segs = res.get('segments', [])
                    S    = max((s['end']   for s in segs), default=clip_durs[i] * 0.9)
                    S_st = min((s['start'] for s in segs), default=0.0)
                    sd   = [{'start': s['start'], 'end': s['end'],
                              'text': s['text'].strip()} for s in segs]
                else:
                    S = clip_durs[i] * 0.9; S_st = 0.0; sd = []
                wo_cache[cn] = {'mtime': mt, 'S': float(S), 'S_start': float(S_st),
                                'segments': sd}
                print(f"    [{i+1}/{N}] {cn}: S={S_st:.2f}s–{S:.2f}s")
        json.dump(wo_cache, open(wo_path, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)
else:
    print(f"  TTS-B — whisper_orig cache valid ({N} clips)")

S_vals       = [wo_cache.get(os.path.basename(c), {}).get('S',       clip_durs[i] * 0.9)
                for i, c in enumerate(clips)]
S_start_vals = [wo_cache.get(os.path.basename(c), {}).get('S_start', 0.0)
                for i, c in enumerate(clips)]
# T is the TTS target duration = speech window (S_end - S_start), minus margin.
# S_start offset is added back as silence at mux time so voice aligns to lip movement.
T_vals = [min((S_vals[i] - S_start_vals[i]) * (1 - tts_margin),
              (clip_durs[i] - S_start_vals[i]) * 0.95)
          for i in range(N)]

# ── Phase 1 (uniform-rate): measure host avg CPS from Whisper transcripts ────
# We compute a weighted-average chars/sec over every clip that has Whisper data,
# then convert it to a single azure_rate that all TTS calls will share.
fixed_rate_str = None  # set only when uniform_rate=True
if uniform_rate:
    total_chars_w = 0; total_speech_s = 0.0
    for i, c in enumerate(clips):
        cn   = os.path.basename(c)
        oc   = wo_cache.get(cn, {})
        segs = oc.get('segments', [])
        if not segs:
            continue
        text = ''.join(s.get('text', '') for s in segs).strip()
        ch_w = len(text)
        dur  = oc.get('S', clip_durs[i] * 0.9) - oc.get('S_start', 0.0)
        if dur > 0.1 and ch_w > 0:
            total_chars_w += ch_w
            total_speech_s += dur
    if total_speech_s > 0:
        avg_cps      = total_chars_w / total_speech_s
        fixed_rate_f = predict_rate(avg_cps, vk, cps_map)
        fixed_rate_str = fmt_rate(fixed_rate_f)
        print(f"  [uniform-rate] host avg CPS: {avg_cps:.2f} "
              f"({total_chars_w} chars / {total_speech_s:.1f}s across {N} clips)")
        print(f"  [uniform-rate] fixed TTS rate: {fixed_rate_str}")
    else:
        print("  [WARN] uniform-rate: no Whisper transcript data; "
              "falling back to per-clip rate", file=sys.stderr)
        uniform_rate = False

# TTS-C..H: Sequential per-clip TTS generation
try:
    import azure.cognitiveservices.speech  # noqa: just validate import
except ImportError:
    print("[ERROR] azure-cognitiveservices-speech not installed.\n"
          "  pip install azure-cognitiveservices-speech", file=sys.stderr)
    sys.exit(1)

print(f"  TTS-C..H — {N} clips | {vk}")

for i, clip in enumerate(clips):
    cn        = os.path.basename(clip)
    cs        = os.path.splitext(cn)[0]
    wav_p     = os.path.join(tts_dir, f'{cs}.wav')
    mp4_p     = os.path.join(tts_dir, cn)
    S_start_i = S_start_vals[i]   # speech-start offset for lip-sync alignment

    # Cache check
    meta = metadata.get(cn, {})
    # Uniform-rate: invalidate cache if WAV was not generated with the same fixed rate
    _cache_stale = uniform_rate and meta.get('rate_fixed') != fixed_rate_str
    if not tts_force and not _cache_stale and meta.get('ok') and os.path.exists(wav_p):
        try:
            if probe_dur(wav_p) > 0:
                if chars[i] > 0:
                    upd_map(cps_map, vk, meta.get('rate_final', '+0%'),
                            chars[i] / meta['dur_actual'], chars[i], meta['dur_actual'])
                print(f"  [{i+1}/{N}] {cn}: cached ({meta['dur_actual']:.2f}s)")
                if os.path.exists(mp4_p):
                    continue
                # wav cached but mp4 missing — fall through to rebuild mp4 only
                shutil.copy2(wav_p, wav_p)  # no-op to reuse wav_p below
                fw = wav_p; fd = meta['dur_actual']; fr = meta.get('rate_final', '+0%')
                A1 = fd; r1_s = fr
                goto_rebuild = True
            else:
                goto_rebuild = False
        except:
            goto_rebuild = False
    else:
        goto_rebuild = False

    if not goto_rebuild:
        T_i = T_vals[i]; S_i = S_vals[i]; ch = chars[i]
        if ch == 0:
            print(f"  [{i+1}/{N}] {cn}: empty text, skipping"); continue

        if uniform_rate:
            # ── Uniform-rate path: one fixed TTS rate, no calibration loop ──
            r1_s   = fixed_rate_str
            wav_r1 = os.path.join(tts_dir, f'{cs}_r1.wav')
            print(f"  [{i+1}/{N}] {cn}: rate={r1_s} (fixed) chars={ch}")
            try:
                synthesize(build_ssml(ssml_t[i], voice, style, sdeg, r1_s, pitch), wav_r1)
                A1 = probe_dur(wav_r1)
            except Exception as e:
                print(f"  [ERROR] R1: {e}", file=sys.stderr); continue
            upd_map(cps_map, vk, r1_s, ch / A1, ch, A1)
            save_map(cps_map, cps_map_path)
            fw, fd, fr = wav_r1, A1, r1_s
            print(f"    tts_dur={A1:.2f}s")

            shutil.copy2(fw, wav_p)
            metadata[cn] = {'rate_fixed': fr, 'dur_actual': round(fd, 3),
                             'S_start': round(S_start_i, 3),
                             'chars': ch, 'ok': True}
            json.dump(metadata, open(meta_path, 'w', encoding='utf-8'),
                      ensure_ascii=False, indent=2)

        else:
            # ── Per-clip rate calibration (original path) ─────────────────
            r1_f  = predict_rate(ch / T_i, vk, cps_map)
            r1_s  = fmt_rate(r1_f)
            wav_r1 = os.path.join(tts_dir, f'{cs}_r1.wav')
            print(f"  [{i+1}/{N}] {cn}: rate={r1_s} T={T_i:.2f}s chars={ch}")
            try:
                synthesize(build_ssml(ssml_t[i], voice, style, sdeg, r1_s, pitch), wav_r1)
                A1 = probe_dur(wav_r1)
            except Exception as e:
                print(f"  [ERROR] R1: {e}", file=sys.stderr); continue

            upd_map(cps_map, vk, r1_s, ch / A1, ch, A1)
            save_map(cps_map, cps_map_path)
            e1 = (A1 - T_i) / T_i

            if abs(e1) < TTS_TOL or tts_rounds <= 1:
                fw, fd, fr = wav_r1, A1, r1_s
                print(f"    R1: {A1:.2f}s err={e1:+.1%} ok")
            else:
                r2_f  = clamp_rate((1.0 + parse_rate(r1_s) / 100.0) * (A1 / T_i))
                r2_s  = fmt_rate(r2_f)
                print(f"    R1: {A1:.2f}s err={e1:+.1%} -> R2 rate={r2_s}")
                wav_r2 = os.path.join(tts_dir, f'{cs}_r2.wav')
                try:
                    synthesize(build_ssml(ssml_t[i], voice, style, sdeg, r2_s, pitch), wav_r2)
                    A2 = probe_dur(wav_r2)
                except Exception as e:
                    print(f"  [ERROR] R2: {e}", file=sys.stderr)
                    fw, fd, fr = wav_r1, A1, r1_s
                else:
                    upd_map(cps_map, vk, r2_s, ch / A2, ch, A2)
                    save_map(cps_map, cps_map_path)
                    e2 = (A2 - T_i) / T_i
                    print(f"    R2: {A2:.2f}s err={e2:+.1%} ok")
                    fw, fd, fr = wav_r2, A2, r2_s

            speech_window = S_vals[i] - S_start_vals[i]
            if fd > speech_window:
                print(f"  [WARN] clip {i}: TTS {fd:.2f}s > speech window "
                      f"{speech_window:.2f}s ({S_start_vals[i]:.2f}s–{S_vals[i]:.2f}s)")

            shutil.copy2(fw, wav_p)
            metadata[cn] = {'rate_r1': r1_s, 'dur_r1': round(A1, 3), 'rate_final': fr,
                             'dur_actual': round(fd, 3), 'dur_target': round(T_i, 3),
                             'S_start': round(S_start_i, 3),
                             'chars': ch, 'ok': True}
            json.dump(metadata, open(meta_path, 'w', encoding='utf-8'),
                      ensure_ascii=False, indent=2)

    # Rebuild clip with TTS audio.
    if uniform_rate:
        # ── Uniform-rate mux: time-scale video so lip-movement window ≈ TTS ──
        tts_dur_i       = probe_dur(wav_p)
        speech_window_i = S_vals[i] - S_start_vals[i]
        SPEED_CLAMP_LO, SPEED_CLAMP_HI = 0.60, 1.50
        if tts_dur_i > 0 and speech_window_i > 0:
            clip_speed_i = speech_window_i / tts_dur_i
            if clip_speed_i < SPEED_CLAMP_LO:
                print(f"  [WARN] {cn}: clip_speed {clip_speed_i:.3f} clamped to {SPEED_CLAMP_LO}")
                clip_speed_i = SPEED_CLAMP_LO
            elif clip_speed_i > SPEED_CLAMP_HI:
                print(f"  [WARN] {cn}: clip_speed {clip_speed_i:.3f} clamped to {SPEED_CLAMP_HI}")
                clip_speed_i = SPEED_CLAMP_HI
        else:
            clip_speed_i = 1.0
        new_dur_i  = clip_durs[i] / clip_speed_i
        # TTS audio starts at S_start_i mapped into the sped clip's time base
        delay_ms   = int((S_start_i / clip_speed_i) * 1000)
        pts_factor = 1.0 / clip_speed_i
        print(f"    clip_speed={clip_speed_i:.3f}x  tts={tts_dur_i:.2f}s  "
              f"speech_win={speech_window_i:.2f}s  new_dur={new_dur_i:.2f}s")
        metadata[cn]['clip_speed'] = round(clip_speed_i, 4)
        json.dump(metadata, open(meta_path, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)
        if delay_ms > 10:
            r = subprocess.run([
                'ffmpeg', '-y', '-i', clip, '-i', wav_p,
                '-filter_complex',
                    f'[0:v]setpts={pts_factor:.6f}*PTS[vout];'
                    f'[1:a]adelay={delay_ms}:all=1[aout]',
                '-map', '[vout]', '-map', '[aout]',
                '-c:v', 'libx264', '-crf', '23', '-preset', 'fast',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-b:a', '192k', '-t', f'{new_dur_i:.3f}',
                mp4_p,
            ], capture_output=True, text=True)
        else:
            r = subprocess.run([
                'ffmpeg', '-y', '-i', clip, '-i', wav_p,
                '-filter_complex',
                    f'[0:v]setpts={pts_factor:.6f}*PTS[vout]',
                '-map', '[vout]', '-map', '1:a:0',
                '-c:v', 'libx264', '-crf', '23', '-preset', 'fast',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-b:a', '192k', '-t', f'{new_dur_i:.3f}',
                mp4_p,
            ], capture_output=True, text=True)
    else:
        # ── Per-clip mux: time-scale video to match TTS audio duration ───────
        # setpts speeds up (or slows down) the video so the lip-movement window
        # matches the TTS duration exactly.  Without this, each clip's video
        # plays at original speed while the TTS is shorter → audio ends early
        # per clip → accumulated ~7s of silent lip movement at end of video.
        # If the original speaker's speech starts after t=0 (e.g. 1.2s into clip),
        # adelay inserts that many ms of silence before TTS for lip-sync alignment.
        tts_dur_i  = probe_dur(wav_p)
        delay_ms   = int(S_start_i * 1000)
        total_dur_i = tts_dur_i + delay_ms / 1000.0   # video must cover delay + TTS
        pts_factor  = total_dur_i / clip_durs[i]        # setpts = new_dur/old_dur
        print(f"    per-clip mux: speed={clip_durs[i]/total_dur_i:.3f}x  "
              f"old={clip_durs[i]:.3f}s → new={total_dur_i:.3f}s")
        if delay_ms > 10:
            r = subprocess.run([
                'ffmpeg', '-y', '-i', clip, '-i', wav_p,
                '-filter_complex',
                    f'[0:v]setpts={pts_factor:.6f}*PTS[vout];'
                    f'[1:a]adelay={delay_ms}:all=1[aout]',
                '-map', '[vout]', '-map', '[aout]',
                '-c:v', 'libx264', '-crf', '23', '-preset', 'fast',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-b:a', '192k', '-t', f'{total_dur_i:.3f}',
                mp4_p,
            ], capture_output=True, text=True)
        else:
            r = subprocess.run([
                'ffmpeg', '-y', '-i', clip, '-i', wav_p,
                '-filter_complex', f'[0:v]setpts={pts_factor:.6f}*PTS[vout]',
                '-map', '[vout]', '-map', '1:a:0',
                '-c:v', 'libx264', '-crf', '23', '-preset', 'fast',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-b:a', '192k', '-t', f'{total_dur_i:.3f}',
                mp4_p,
            ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [ERROR] ffmpeg rebuild {cn}: {r.stderr[-200:]}", file=sys.stderr)

# Fall back to original clip for any VO mp4 still missing
for i, clip in enumerate(clips):
    mp4_p = os.path.join(tts_dir, os.path.basename(clip))
    if not os.path.exists(mp4_p):
        print(f"  [WARN] {os.path.basename(clip)}: TTS failed — using original clip")
        shutil.copy2(clip, mp4_p)

print(f"  TTS complete -> {tts_dir}/")
TTS_REGEN_EOF

    echo ""
    _stitch_src="$_tts_dir"
    _wts_cache="${_tts_dir}/.whisper_tts.json"
  else
    _stitch_src="$INPUT_FOLDER"
    _wts_cache=""
  fi

  # ── Step 1: xfade stitch ────────────────────────────────────────────────────
  # Saves .clip_info.json to INPUT_FOLDER for use by later steps.
  echo "  STEP 1 — xfade stitch (parallel batch)"
  python3 - "$_stitch_src" "$_out_video" "$INPUT_FOLDER" << 'PYEOF'
import subprocess, sys, os, glob, re, json, tempfile, concurrent.futures

folder, out, base_folder = sys.argv[1], sys.argv[2], sys.argv[3]
XFADE      = 0.4
BATCH_SIZE = 5
N_WORKERS  = os.cpu_count() or 4

def natural_key(path):
    name = os.path.basename(path)
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r'(\d+)', name)]

def probe_dur(path):
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1', path],
        capture_output=True, text=True, check=True)
    return float(r.stdout.strip().split('=')[1])

def stitch(clip_list, out_path, preset='ultrafast', crf=23):
    import shutil
    if len(clip_list) == 1:
        shutil.copy2(clip_list[0], out_path); return
    n    = len(clip_list)
    durs = [probe_dur(c) for c in clip_list]
    fp   = []
    for i in range(n):
        pad_s = XFADE if i > 0     else 0.0
        pad_e = XFADE if i < n - 1 else 0.0
        if pad_s and pad_e:
            tpad = (f"tpad=start_mode=clone:start_duration={pad_s:.3f}"
                    f":stop_mode=clone:stop_duration={pad_e:.3f}")
        elif pad_s: tpad = f"tpad=start_mode=clone:start_duration={pad_s:.3f}"
        elif pad_e: tpad = f"tpad=stop_mode=clone:stop_duration={pad_e:.3f}"
        else:       tpad = "copy"
        fp.append(f"[{i}:v]{tpad}[pv{i}]")
    cum = durs[0]; vprev = "[pv0]"
    for i in range(1, n):
        vl = "vout" if i == n - 1 else f"v{i}"
        fp.append(f"{vprev}[pv{i}]xfade=transition=fade"
                  f":duration={XFADE:.3f}:offset={cum:.3f}[{vl}]")
        vprev = f"[{vl}]"; cum += XFADE + durs[i]
    delay_ms = 0.0
    for i in range(n):
        d = int(delay_ms)
        fp.append(f"[{i}:a]adelay={d}|{d}[da{i}]")
        delay_ms += (durs[i] + XFADE) * 1000
    total_dur = sum(durs) + (n - 1) * XFADE
    amix_in   = "".join(f"[da{i}]" for i in range(n))
    fp.append(f"{amix_in}amix=inputs={n}:normalize=0:dropout_transition=0"
              f",apad=pad_dur={total_dur:.3f}[aout]")
    cmd = ['ffmpeg', '-y', '-threads', '1']
    for c in clip_list: cmd += ['-i', c]
    cmd += ['-filter_complex', ';'.join(fp),
            '-map', '[vout]', '-map', '[aout]',
            '-c:v', 'libx264', '-crf', str(crf), '-preset', preset,
            '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k',
            '-t', f'{total_dur:.3f}', out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-600:], file=sys.stderr)
        raise RuntimeError(f"ffmpeg failed: {os.path.basename(out_path)}")

# Respect Whisper-determined clip order if available
_order_path = os.path.join(base_folder, '.clip_order.json')
if os.path.exists(_order_path):
    _order_data = json.load(open(_order_path))
    _all_mp4    = {os.path.basename(f)
                   for f in glob.glob(os.path.join(folder, '*.mp4'))
                   if os.path.basename(f) != 'output.mp4'}
    _ordered    = [n for n in _order_data['ordered_clips'] if n in _all_mp4]
    _unmatched  = sorted(_all_mp4 - set(_ordered), key=natural_key)
    clips       = [os.path.join(folder, n) for n in _ordered + _unmatched]
    print(f"  Using Whisper clip order ({len(_ordered)} ordered, {len(_unmatched)} unmatched)")
else:
    clips = sorted(
        [f for f in glob.glob(os.path.join(folder, '*.mp4'))
         if os.path.basename(f) != 'output.mp4'],
        key=natural_key)
N = len(clips)
if N < 2:
    print(f"[ERROR] Need >=2 clips, found {N}", file=sys.stderr); sys.exit(1)

durs_all = [probe_dur(c) for c in clips]
total    = sum(durs_all) + XFADE * (N - 1)
print(f"  Clips: {N}   Total: {total:.2f}s")
for c in clips:
    print(f"    {os.path.basename(c)}")

# Save clip durations for title overlay timing in Step 4+5+6
json.dump({'clip_durs': durs_all, 'N': N, 'xfade': XFADE},
          open(os.path.join(base_folder, '.clip_info.json'), 'w'), indent=2)

if N <= BATCH_SIZE:
    print(f"  Strategy: single-pass (<={BATCH_SIZE} clips)")
    stitch(clips, out, preset='medium', crf=18)
    sys.exit(0)

batches    = [clips[i:i + BATCH_SIZE] for i in range(0, N, BATCH_SIZE)]
print(f"  Strategy: {len(batches)} batches x <={BATCH_SIZE} clips "
      f"({N_WORKERS} workers / {os.cpu_count()} cores)")

with tempfile.TemporaryDirectory() as tmp:
    batch_outs = [os.path.join(tmp, f'batch_{i:04d}.mp4') for i in range(len(batches))]
    def _sb(args):
        idx, b, bo = args
        print(f"  [batch {idx+1}/{len(batches)}] {len(b)} clips...")
        stitch(b, bo, preset='ultrafast', crf=23)
        print(f"  [batch {idx+1}/{len(batches)}] done")
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        for f in concurrent.futures.as_completed(
                [ex.submit(_sb, (i, b, bo))
                 for i, (b, bo) in enumerate(zip(batches, batch_outs))]):
            f.result()
    print(f"  Joining {len(batch_outs)} batches (medium CRF 18)...")
    stitch(batch_outs, out, preset='medium', crf=18)

print(f"  Done -> {os.path.basename(out)}")
PYEOF
  echo ""

  # ── Step 2+3: Transcribe output.mp4 → output.srt ────────────────────────────
  # IMPORTANT: subtitle timestamps MUST come from transcribing the final
  # stitched output.mp4.  Per-clip Whisper + manual offset arithmetic is WRONG
  # because xfade overlaps (0.4s each) accumulate multi-second drift by clip 11.
  # output.mp4 is the only ground truth.  Do not change this to per-clip.
  echo "  STEP 2+3 — Whisper transcription of output.mp4 → SRT"
  _whisper_model="small"
  [[ -z "$STORY" ]] && _whisper_model="medium"
  python3 - "$_out_video" "$_out_srt" "$_whisper_model" \
             "${_story_json:-}" << 'WHISPER_CLIP_EOF'
import sys, os, subprocess, json

out_video_p  = sys.argv[1]
srt_out      = sys.argv[2]
model_name   = sys.argv[3]
story_json_p = sys.argv[4] if len(sys.argv) > 4 else ''

def fmt_ts(ms):
    ms = max(0, int(ms))
    h, ms = divmod(ms, 3_600_000); m, ms = divmod(ms, 60_000); s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

if not os.path.exists(out_video_p):
    print(f"[ERROR] output.mp4 not found: {out_video_p}", file=sys.stderr)
    sys.exit(1)

# Build initial_prompt from story display_text to improve Whisper accuracy
initial_prompt = None
if story_json_p and os.path.exists(story_json_p):
    try:
        story = json.load(open(story_json_p, encoding='utf-8'))
        lines = story.get('display_text') or story.get('text') or []
        initial_prompt = ' '.join(lines[:20])   # first 20 lines as hint
    except Exception:
        pass

# Extract audio from output.mp4
tmp_wav = out_video_p + '.tmp_w.wav'
subprocess.run(['ffmpeg', '-y', '-i', out_video_p, '-ar', '16000', '-ac', '1',
                '-c:a', 'pcm_s16le', tmp_wav, '-loglevel', 'error'], check=True)

from faster_whisper import WhisperModel
print(f"  Loading faster-whisper model '{model_name}'...")
wmodel = WhisperModel(model_name, compute_type='int8')

kwargs = dict(language='zh', beam_size=5)
if initial_prompt:
    kwargs['initial_prompt'] = initial_prompt

segs_iter, _ = wmodel.transcribe(tmp_wav, **kwargs)
out_lines = []; seq = 1
for seg in segs_iter:
    t_in  = int(seg.start * 1000)
    t_out = int(seg.end   * 1000)
    text  = seg.text.strip()
    if not text: continue
    out_lines += [str(seq), f"{fmt_ts(t_in)} --> {fmt_ts(t_out)}", text, '']
    seq += 1

try: os.remove(tmp_wav)
except: pass

open(srt_out, 'w', encoding='utf-8').write('\n'.join(out_lines))
print(f"  -> {seq-1} segments from output.mp4 -> {srt_out}")
WHISPER_CLIP_EOF

  if [[ ! -f "$_out_srt" ]]; then
    echo "[WARN] Per-clip Whisper SRT not generated — skipping subtitle burn" >&2
    _out_srt=""
  else
    echo "    SRT (combined): $_out_srt"
  fi

  # ── Step 3b: Build final SRT ─────────────────────────────────────────────────
  # Mode A (story3 JSON): use display_text[] from parsed story — handles ## and -
  # Mode A (legacy):      re-parse story.txt stripping ## lines and bare - lines
  # Mode B (no story):    split Whisper's own transcribed text into <=25-char cards
  if [[ -n "$_out_srt" && -f "$_out_srt" ]]; then
    _shift_ms="${SUBTITLE_SHIFT_MS:-0}"
    if [[ -n "$_story_json" && -f "$_story_json" ]]; then
      echo "  STEP 3b — Map story3.txt display_text onto Whisper timestamps"
    elif [[ -n "$STORY" && -f "$STORY" ]]; then
      echo "  STEP 3b — Map story.txt text onto Whisper timestamps (shift ${_shift_ms}ms)"
    else
      echo "  STEP 3b — Split Whisper lines to <=25 chars (shift ${_shift_ms}ms)"
    fi
    python3 - "$_out_srt" "${STORY:-}" "$_shift_ms" "${_story_json:-}" << 'SPLITEOF'
import sys, re, unicodedata, json, os

srt_path     = sys.argv[1]
story_path   = sys.argv[2] if len(sys.argv) > 2 else ''
shift_ms     = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else 0
story_json_p = sys.argv[4] if len(sys.argv) > 4 else ''

STRONG_BREAKS = set('。！？…')
WEAK_BREAKS   = set('，；、,;!?')
ALL_BREAKS    = STRONG_BREAKS | WEAK_BREAKS
MAX_CARD_VW   = 36   # 18 CJK chars max → 18×60px=1080px < 1198px usable on 1264×720
MIN_CARD_VW   = 10

def vw(s):
    return sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in s)

def _hard_split(text):
    """Split text into <=MAX_CARD_VW chunks by character count when no punctuation found."""
    chunks = []; buf = ''; w = 0
    for c in text:
        cw = 2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1
        if w + cw > MAX_CARD_VW and buf:
            chunks.append(buf); buf = c; w = cw
        else:
            buf += c; w += cw
    if buf: chunks.append(buf)
    return chunks

def natural_chunks(text):
    text = text.strip()
    if not text: return []
    parts = re.split(r'(?<=[。！？…])', text)
    cards = []
    for part in parts:
        part = part.strip()
        if not part: continue
        if vw(part) <= MAX_CARD_VW:
            cards.append(part)
        else:
            # Split on weak punctuation AND em-dash (——)
            subs = re.split(r'(?<=[，；、,;!?])|(?<=——)', part)
            buf = ''
            for s in subs:
                s = s.strip()
                if not s: continue
                if not buf: buf = s
                elif vw(buf) + vw(s) <= MAX_CARD_VW: buf += s
                else:
                    if buf: cards.extend(_hard_split(buf))
                    buf = s
            if buf: cards.extend(_hard_split(buf))
    merged = []
    for card in cards:
        if (merged and vw(card) < MIN_CARD_VW
                and vw(merged[-1]) + vw(card) <= MAX_CARD_VW):
            merged[-1] += card
        else:
            merged.append(card)
    return merged or [text]

def snap_to_break(text, pos, window=25):
    n = len(text)
    for delta in range(0, window + 1):
        if pos - delta >= 0 and text[pos - delta] in ALL_BREAKS: return pos - delta + 1
        if pos + delta < n and text[pos + delta] in ALL_BREAKS:  return pos + delta + 1
    return pos

def srt_ts(ms):
    h, ms = divmod(ms, 3_600_000); m, ms = divmod(ms, 60_000); s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def parse_ts(t):
    h, m, rest = t.split(':'); s, ms = rest.split(',')
    return int(h)*3_600_000 + int(m)*60_000 + int(s)*1_000 + int(ms)

def emit(out_lines, seq, t_in, t_out, text):
    chunks = natural_chunks(text); total = sum(len(c) for c in chunks) or 1
    dur, cur = t_out - t_in, t_in
    for i, chunk in enumerate(chunks):
        share = round(dur * len(chunk) / total); end = cur + max(share, 200)
        if i == len(chunks) - 1: end = t_out
        out_lines += [str(seq), f"{srt_ts(cur)} --> {srt_ts(end)}", chunk, '']
        seq += 1; cur = end
    return seq

def map_story_to_segs(raw, segments, out_lines, label):
    total_chars = len(raw); total_wc = sum(len(wt) for _, _, wt in segments) or 1
    story_cursor = 0; seq = 1
    for seg_i, (t_in, t_out, wt) in enumerate(segments):
        n_chars = round(len(wt) / total_wc * total_chars)
        target  = story_cursor + n_chars
        end_pos = (total_chars if seg_i == len(segments) - 1
                   else max(snap_to_break(raw, min(target, total_chars - 1)),
                             story_cursor + 1))
        text = raw[story_cursor:end_pos].strip(); story_cursor = end_pos
        if not text: continue
        cards = natural_chunks(text)
        if not cards: continue
        seg_chars = sum(len(c) for c in cards) or 1
        dur, cur  = t_out - t_in, t_in
        for i, card in enumerate(cards):
            share = round(dur * len(card) / seg_chars); end = cur + max(share, 200)
            if i == len(cards) - 1: end = t_out
            out_lines += [str(seq), f"{srt_ts(cur)} --> {srt_ts(end)}", card, '']
            seq += 1; cur = end
    print(f"    {seq-1} cards ({label}, {total_chars} chars)")

def cleanup_cards(lines):
    """Post-process SRT out_lines: strip card text, merge cards < 800 ms into neighbors.
    Tries backward merge first, then forward merge; allows up to MAX_CARD_VW+10 when merging."""
    MIN_CARD_MS  = 800
    MERGE_MAX_VW = MAX_CARD_VW + 10   # slightly relaxed limit only for merging short cards
    cards = []
    i = 0
    while i < len(lines):
        if i < len(lines) and lines[i] and lines[i].isdigit():
            ts  = lines[i+1] if i+1 < len(lines) else ''
            txt = lines[i+2].strip() if i+2 < len(lines) else ''
            m2  = re.match(r'(\S+)\s*-->\s*(\S+)', ts)
            if m2:
                cards.append([parse_ts(m2.group(1)), parse_ts(m2.group(2)), txt])
            i += 4
        else:
            i += 1
    # Pass 1: backward merge
    merged = []
    for card in cards:
        t_in, t_out, text = card
        dur = t_out - t_in
        if merged and dur < MIN_CARD_MS:
            prev = merged[-1]
            combined = prev[2] + text
            if vw(combined) <= MERGE_MAX_VW:
                prev[2] = combined; prev[1] = t_out; continue
        merged.append(card)
    # Pass 2: forward merge any remaining short cards
    result = []
    i = 0
    while i < len(merged):
        t_in, t_out, text = merged[i]
        dur = t_out - t_in
        if dur < MIN_CARD_MS and i + 1 < len(merged):
            nxt = merged[i + 1]
            combined = text + nxt[2]
            if vw(combined) <= MERGE_MAX_VW:
                result.append([t_in, nxt[1], combined])
                i += 2; continue
        result.append([t_in, t_out, text])
        i += 1
    out = []
    for seq2, (t_in, t_out, text) in enumerate(result, 1):
        out += [str(seq2), f"{srt_ts(t_in)} --> {srt_ts(t_out)}", text, '']
    return out

# Parse Whisper SRT
blocks   = re.split(r'\n{2,}', open(srt_path, encoding='utf-8').read().strip())
raw_segs = []
for block in blocks:
    lines = block.strip().splitlines()
    if len(lines) < 3: continue
    m = re.match(r'(\S+)\s*-->\s*(\S+)', lines[1])
    if not m: continue
    raw_segs.append((parse_ts(m.group(1)), parse_ts(m.group(2)),
                     ' '.join(lines[2:]).strip()))
if not raw_segs:
    print("  [warn] no segments found", file=sys.stderr); sys.exit(0)

segments = []
for t_in, t_out, text in raw_segs:
    t_in_s  = max(0, t_in  + shift_ms)
    t_out_s = max(t_in_s + 50, t_out + shift_ms)
    segments.append((t_in_s, t_out_s, text))

out_lines = []

# Mode A (story3 JSON)
if story_json_p and os.path.exists(story_json_p):
    story = json.load(open(story_json_p, encoding='utf-8'))
    display_texts = story.get('display_text', [])
    raw = ' '.join(display_texts)
    raw = re.sub(r'(?<=[一-鿿])\s+(?=[一-鿿])', '', raw)
    raw = re.sub(r'(?<=[，。！？；：、—])\s+', '', raw)  # drop space after CJK punct
    raw = re.sub(r'\s+', ' ', raw).strip()
    map_story_to_segs(raw, segments, out_lines, 'story3 mode')

# Mode A (legacy story.txt)
elif story_path:
    raw = open(story_path, encoding='utf-8').read()
    raw = re.sub(r'^#{1,6}\s+.*$', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'^-\s*$',        '', raw, flags=re.MULTILINE)
    raw = re.sub(r'^[-*]\s+',      '', raw, flags=re.MULTILINE)
    raw = re.sub(r'(?<=[一-鿿])\s+(?=[一-鿿])', '', raw)
    raw = re.sub(r'(?<=[，。！？；：、—])\s+', '', raw)  # drop space after CJK punct
    raw = re.sub(r'\s+', ' ', raw).strip()
    map_story_to_segs(raw, segments, out_lines, 'story.txt mode')

# Mode B
else:
    seq = 1
    for t_in, t_out, text in segments:
        if not text: continue
        seq = emit(out_lines, seq, t_in, t_out, text)
    print(f"    {seq-1} cards (Whisper mode)")

# Apply to all modes: merge short cards, strip stray spaces
out_lines = cleanup_cards(out_lines)
open(srt_path, 'w', encoding='utf-8').write('\n'.join(out_lines))
SPLITEOF
  fi
  echo ""

  # ── Step 4+5+6: Speed + title overlay + thumbnail ────────────────────────────
  # NOTE: subtitle burn is intentionally NOT done here for simple_narration.
  # Subtitles are burned by Stage 9 Step 11 (render_video.py) which transcribes
  # the final output.mp4 for ground-truth timing.  Burning here would produce
  # double-burned subtitles when Step 11 runs.
  # Title overlays (drawbox+drawtext) are added BEFORE setpts so their enable=
  # between(t,...) timestamps are in pre-speed space, consistent with SRT timing.
  _has_spd=0; [[ -n "$SPEED" && "$SPEED" != "1" && "$SPEED" != "1.0" ]] && _has_spd=1

  echo "  STEP 4+5+6 — speed / title overlay / thumbnail  (subtitles deferred to Step 11)"
  python3 - "$_out_video" "$INPUT_FOLDER" \
             "" "${SPEED:-1.0}" \
             "${_story_json:-}" "$_stitch_src" << 'FINAL_EOF'
import sys, os, json, re, glob, subprocess

out_video    = sys.argv[1]
base_folder  = sys.argv[2]
srt_path     = sys.argv[3]
speed        = float(sys.argv[4]) if sys.argv[4] else 1.0
story_json_p = sys.argv[5]
stitch_src   = sys.argv[6]

XFADE       = 0.4
OVERLAY_DUR = 2.0
NOTO_FONT   = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD_FONT   = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

has_srt = bool(srt_path) and os.path.exists(srt_path)
has_spd = abs(speed - 1.0) > 0.001

def natural_key(p):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', os.path.basename(p))]

def probe_dur(path):
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1', path],
        capture_output=True, text=True, check=True)
    return float(r.stdout.strip().split('=')[1])

# Build video filter chain
vf_parts = []
titles   = []

# 1. Title badge overlays (pre-setpts)
if story_json_p and os.path.exists(story_json_p):
    story      = json.load(open(story_json_p, encoding='utf-8'))
    titles     = story.get('titles', [])
    sub_titles = [t for t in titles if not t['is_first']]
    if sub_titles:
        ci_path = os.path.join(base_folder, '.clip_info.json')
        if os.path.exists(ci_path):
            clip_durs = json.load(open(ci_path))['clip_durs']
        else:
            clips = sorted([f for f in glob.glob(os.path.join(stitch_src, '*.mp4'))
                           if os.path.basename(f) != 'output.mp4'], key=natural_key)
            clip_durs = [probe_dur(c) for c in clips]
        offsets = []; cur = 0.0
        for d in clip_durs:
            offsets.append(cur); cur += d + XFADE
        total_stories = len(titles)
        def esc(t):
            return (t.replace("\\", "\\\\").replace("'", "'")
                     .replace(":", "\\:").replace("[", "\\[").replace("]", "\\]"))
        font_arg = f":fontfile={NOTO_FONT}" if os.path.exists(NOTO_FONT) else ""
        for t in sub_titles:
            ci = t['clip_index']
            if ci >= len(offsets): continue
            t0 = offsets[ci]; t1 = t0 + OVERLAY_DUR
            sn = titles.index(t) + 1
            badge = f"\\[{sn:02d}/{total_stories:02d}\\] {esc(t['title'])}"
            en = f"between(t,{t0:.3f},{t1:.3f})"
            vf_parts.append(
                f"drawbox=x=16:y=14:w=iw*0.62:h=68:color=black@0.65:t=fill:enable='{en}'")
            vf_parts.append(
                f"drawtext=text='{badge}'{font_arg}:fontsize=28:fontcolor=white"
                f":borderw=1:bordercolor=black@0.4:x=28:y=32:enable='{en}'")
        print(f"    Title overlays: {len(sub_titles)} badges")

# 2. Subtitles (pre-setpts)
if has_srt:
    srt_esc = srt_path.replace('\\', '/').replace(':', '\\:')
    vf_parts.append(
        f"subtitles='{srt_esc}':force_style='"
        f"FontName=Noto Sans CJK SC,FontSize=24,"
        f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Bold=1,Outline=2'")

# 3. Speed (setpts — must be last video filter)
if has_spd:
    vf_parts.append(f"setpts={1.0 / speed:.6f}*PTS")

# Thumbnail raw frame — grabbed BEFORE subtitle burn so subtitle text is not baked in
thumb_path = os.path.join(base_folder, 'thumbnail.png')
thumb_raw  = os.path.join(base_folder, 'thumbnail_raw.png')
total_dur  = probe_dur(out_video)
ss = f"{min(5.0, total_dur * 0.1):.2f}"
tr = subprocess.run([
    'ffmpeg', '-y', '-ss', ss, '-i', out_video, '-frames:v', '1',
    '-vf', 'scale=1280:720:force_original_aspect_ratio=decrease'
           ',pad=1280:720:(ow-iw)/2:(oh-ih)/2',
    '-q:v', '2', thumb_raw
], capture_output=True, text=True)

# Run ffmpeg
tmp_out = out_video + '.tmp.mp4'

if not vf_parts:
    print("    No video filters — output unchanged.")
else:
    if has_spd:
        rem = speed; af = []
        while rem < 0.5: af.append('atempo=0.5'); rem /= 0.5
        while rem > 2.0: af.append('atempo=2.0'); rem /= 2.0
        af.append(f'atempo={rem:.6f}')
        audio_args = ['-af', ','.join(af), '-c:a', 'aac', '-b:a', '128k']
    else:
        audio_args = ['-c:a', 'copy']
    vf_str = ','.join(vf_parts)
    cmd = (['ffmpeg', '-y', '-i', out_video, '-vf', vf_str]
           + audio_args
           + ['-c:v', 'libx264', '-crf', '18', '-preset', 'medium',
              '-pix_fmt', 'yuv420p', tmp_out, '-loglevel', 'error'])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[ERROR] ffmpeg Step 4+5+6:\n{r.stderr[-400:]}", file=sys.stderr)
        sys.exit(1)
    os.replace(tmp_out, out_video)
    print("    Video done.")
if tr.returncode == 0:
    if titles:
        try:
            from PIL import Image, ImageDraw, ImageFont
            main_title = titles[0]['title']
            W, H = 1280, 720; FS = 72; LP = 16; BP = 24
            tf = BOLD_FONT if os.path.exists(BOLD_FONT) else NOTO_FONT
            try:   pf = ImageFont.truetype(tf, FS)
            except: pf = ImageFont.load_default()
            def split_t(title):
                title = title.strip()
                if len(title) <= 14: return [title]
                mid = len(title) // 2; best = -1
                for d in range(len(title)):
                    for idx in [mid - d, mid + d]:
                        if 0 < idx < len(title) and title[idx] in '，,。！？ ：:':
                            best = idx; break
                    if best != -1: break
                cut = (best + 1) if best != -1 else mid
                return [title[:cut].strip(), title[cut:].strip()]
            lns  = split_t(main_title)
            img  = Image.open(thumb_raw).convert('RGB').resize((W, H), Image.LANCZOS)
            draw = ImageDraw.Draw(img, 'RGBA')
            dims = []
            for ln in lns:
                bb = draw.textbbox((0, 0), ln, font=pf)
                dims.append((bb[2] - bb[0], bb[3] - bb[1]))
            lh = max(d[1] for d in dims) + LP; bh = lh * len(lns) + BP * 2
            by0 = (H - bh) // 2; by1 = by0 + bh
            draw.rectangle([(0, by0), (W, by1)], fill=(0, 0, 0, 178))
            for li, (ln, (tw, th)) in enumerate(zip(lns, dims)):
                tx = (W - tw) // 2; ty = by0 + BP + li * lh
                for dx, dy in [(-2,0),(2,0),(0,-2),(0,2),(-2,-2),(2,2)]:
                    draw.text((tx+dx, ty+dy), ln, font=pf, fill=(0, 0, 0, 200))
                draw.text((tx, ty), ln, font=pf, fill=(255, 255, 255, 255))
            img.save(thumb_path, 'PNG')
            if os.path.exists(thumb_raw): os.unlink(thumb_raw)
            print(f"    Thumbnail: {thumb_path}")
        except Exception as e:
            if os.path.exists(thumb_raw):
                try: os.rename(thumb_raw, thumb_path)
                except: pass
            print(f"  [warn] thumbnail text overlay failed: {e}", file=sys.stderr)
    else:
        # No ## headings — use raw frame as-is
        os.rename(thumb_raw, thumb_path)
        print(f"    Thumbnail: {thumb_path}")
else:
    print(f"  [warn] thumbnail frame grab failed: {tr.stderr[-200:]}", file=sys.stderr)
FINAL_EOF

  # ── Create project structure ──────────────────────────────────────────────
  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "  STEP — Create project & pre-generate YouTube metadata"
  echo "════════════════════════════════════════════════════════════"

  if [[ -z "$LOCALE" && -n "$CONFIG" && -f "$CONFIG" ]]; then
    LOCALE="$(python3 -c "import json; d=json.load(open('$CONFIG',encoding='utf-8')); print(d.get('locale',''))" 2>/dev/null || true)"
  fi
  [[ -z "$LOCALE"  ]] && LOCALE="en"
  [[ -z "$EPISODE" ]] && EPISODE="s01e01"

  # Slug + EP_DIR already derived early (before STEP 0); skip re-derivation.
  # Only re-derive if somehow not set (e.g. no-story mode edge case).
  if [[ -z "$_slug" ]]; then
    _clips_title=""
    if [[ -n "$STORY" && -f "$STORY" ]]; then
      _clips_title="$(python3 -c "
import re, sys
text = open('$STORY', encoding='utf-8').read()
for line in text.splitlines():
    m = re.match(r'^#{1,6}\s+(.+)', line)
    if m: print(m.group(1).strip()); sys.exit(0)
for line in text.splitlines():
    s = line.strip()
    if s: print(s); sys.exit(0)
" 2>/dev/null)"
    fi
    [[ -z "$_clips_title" ]] && _clips_title="$(basename "$INPUT_FOLDER")"

    _slug="$(python3 -c "
import re, unicodedata, secrets
orig = '''$_clips_title'''
t = unicodedata.normalize('NFKD', orig).encode('ascii','ignore').decode('ascii')
t = t.lower().strip()
t = re.sub(r'[^\w\s-]', '', t)
t = re.sub(r'[\s_]+', '-', t)
t = re.sub(r'-{2,}', '-', t)
t = t.strip('-')[:32]
rand6 = secrets.token_hex(3)
print((t + '-' + rand6) if t else 'story-' + rand6)
" 2>/dev/null || echo "story-$(python3 -c 'import secrets; print(secrets.token_hex(3))')")"

    EP_DIR="projects/${_slug}/episodes/${EPISODE}"
  fi
  RENDERS_DIR="${EP_DIR}/renders/${LOCALE}"
  mkdir -p "$RENDERS_DIR"

  echo "  Slug      : $_slug"
  echo "  Episode   : $EPISODE   Locale: $LOCALE"
  echo "  Project   : $EP_DIR"
  echo ""

  echo "  ✓ output.mp4"

  if [[ -f "$_out_srt" ]]; then
    echo "  ✓ output.${LOCALE}.srt"
  fi

  # Convert thumbnail PNG → JPEG (youtube.json references .jpg)
  if [[ -f "${INPUT_FOLDER}/thumbnail.png" ]]; then
    ffmpeg -y -i "${INPUT_FOLDER}/thumbnail.png" \
      "${RENDERS_DIR}/thumbnail.jpg" -loglevel error 2>/dev/null \
      && echo "  ✓ thumbnail.jpg"
  fi

  # Copy story as story.txt so server has context for YouTube metadata generation
  if [[ -n "$STORY" && -f "$STORY" ]]; then
    cp "$STORY" "${EP_DIR}/story.txt"
    echo "  ✓ story.txt"
  fi

  # ── Create contract files: VOPlan, meta.json, VoiceCast.json ─────────────
  # These are required for Stage 9 Step 11 (render_video) and the VO tab.
  # Without VOPlan.{locale}.json, the server cannot detect the locale and
  # Step 11 is invisible in the pipeline UI.
  if [[ -n "$_story_json" && -f "$_story_json" ]]; then
    echo "  Creating contract files (VOPlan, meta.json, VoiceCast.json)..."
    python3 - "$EP_DIR" "$LOCALE" "$_story_json" "$CONFIG" "$_slug" \
               "${EPISODE:-s01e01}" << 'CONTRACTS_EOF'
import sys, json, os, glob, re, subprocess

ep_dir       = sys.argv[1]
locale       = sys.argv[2]
story_json_p = sys.argv[3]
config_path  = sys.argv[4]
slug         = sys.argv[5]
episode_id   = sys.argv[6]

XFADE = 0.4

def natural_key(p):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', os.path.basename(p))]

def probe_dur(path):
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1', path],
        capture_output=True, text=True)
    try: return float(r.stdout.strip().split('=')[1])
    except: return 5.0

# 1. Find WAV files (speed-adjusted finals, not _r1/_r2 variants)
vo_dir = os.path.join(ep_dir, 'assets', locale, 'audio', 'vo')
wavs   = sorted(
    [f for f in glob.glob(os.path.join(vo_dir, 'clip*.wav'))
     if not re.search(r'_r\d+\.wav$', f)],
    key=natural_key)
if not wavs:
    print(f"  [contracts] No WAVs in {vo_dir} — skipping", flush=True)
    sys.exit(0)

clip_ids  = [os.path.splitext(os.path.basename(w))[0] for w in wavs]
clip_durs = [probe_dur(w) for w in wavs]

# 2. Story text and titles
story  = json.load(open(story_json_p, encoding='utf-8'))
texts  = story.get('display_text') or story.get('text') or []
titles = story.get('titles', [])
story_title = titles[0]['title'] if titles else slug

# 3. Voice settings from config
config      = json.load(open(config_path, encoding='utf-8'))
locale_narr = config.get('narrator', {}).get(locale, {})
voice_entries = {k: v for k, v in locale_narr.items() if isinstance(v, dict)}
if voice_entries:
    enabled = {n: c for n, c in voice_entries.items() if c.get('enabled', False)}
    narr = next(iter(enabled.values())) if enabled else {}
else:
    narr = locale_narr

# 4. Build vo_items with xfade-aware timing
vo_items = []
cur = 0.0
for i, clip_id in enumerate(clip_ids):
    dur  = clip_durs[i]
    text = texts[i] if i < len(texts) else ''
    vo_items.append({
        "item_id":    clip_id,
        "speaker_id": "narrator",
        "text":       text,
        "start_sec":  round(cur, 3),
        "end_sec":    round(cur + dur, 3),
        "tts_prompt": {
            "locale":             locale,
            "azure_voice":        narr.get("azure_voice", ""),
            "azure_rate":         narr.get("azure_rate",  "0%"),
            "azure_pitch":        narr.get("azure_pitch", "0%"),
            "azure_style":        narr.get("azure_style", ""),
            "azure_style_degree": float(narr.get("azure_style_degree", 1.0)),
            "azure_break_ms":     int(narr.get("azure_break_ms", 500)),
        }
    })
    if i < len(clip_ids) - 1:
        cur += dur - XFADE

# 5. story_segments for badge overlays
n_stories = len(titles)
story_segments = []
for idx, t in enumerate(titles):
    ci  = t['clip_index']
    fid = clip_ids[ci] if ci < len(clip_ids) else (clip_ids[0] if clip_ids else 'clip01')
    story_segments.append({
        "story_index":   idx + 1,
        "story_count":   n_stories,
        "title":         t['title'],
        "first_item_id": fid,
    })

# 6. Write VOPlan.{locale}.json
voplan = {
    "schema_id":       "VOPlan",
    "schema_version":  "1.0",
    "locale":          locale,
    "locale_scope":    "merged",
    "story_format":    "simple_narration",
    "vo_items":        vo_items,
    "story_segments":  story_segments,
    "scene_heads":     {},
    "resolved_assets": [],
}
json.dump(voplan, open(os.path.join(ep_dir, f"VOPlan.{locale}.json"), 'w',
          encoding='utf-8'), ensure_ascii=False, indent=2)
print(f"  ✓ VOPlan.{locale}.json  ({len(vo_items)} clips)", flush=True)

# 7. Write meta.json
json.dump({
    "story_title":  story_title,
    "story_format": "simple_narration",
    "locales":      locale,
    "episode_id":   episode_id,
}, open(os.path.join(ep_dir, "meta.json"), 'w', encoding='utf-8'),
   ensure_ascii=False, indent=2)
print("  ✓ meta.json", flush=True)

# 8. Write VoiceCast.json at project level (projects/{slug}/)
project_dir = os.path.dirname(os.path.dirname(ep_dir))
json.dump({
    "narrator": {
        locale: {
            "azure_voice":        narr.get("azure_voice", ""),
            "azure_style":        narr.get("azure_style", ""),
            "azure_style_degree": float(narr.get("azure_style_degree", 1.0)),
            "azure_rate":         narr.get("azure_rate",  "0%"),
            "azure_pitch":        narr.get("azure_pitch", "0%"),
            "azure_break_ms":     int(narr.get("azure_break_ms", 500)),
        }
    }
}, open(os.path.join(project_dir, "VoiceCast.json"), 'w', encoding='utf-8'),
   ensure_ascii=False, indent=2)
print("  ✓ VoiceCast.json", flush=True)
CONTRACTS_EOF
  fi

  # ── Generate youtube.json directly (no server required) ───────────────────
  echo ""
  _yt_generated=""
  _story_bn="$(basename "${STORY:-}")"
  echo "  Generating YouTube metadata (takes ~20s) …"
  _yt_result="$(python3 "${SCRIPT_DIR}/code/http/gen_youtube_json.py" \
    --slug        "${_slug}" \
    --ep_id       "${EPISODE}" \
    --locale      "${LOCALE}" \
    --story_basename "${_story_bn}" \
    2>/tmp/simple_run_yt_err.txt)"
  if [[ "$_yt_result" == "ok" ]]; then
    echo "  ✓ youtube.json generated → open the YouTube tab to review & edit"
    _yt_generated="${RENDERS_DIR}/youtube.json"
  else
    _yt_err="$(cat /tmp/simple_run_yt_err.txt 2>/dev/null)"
    echo "  ✗ Generation failed: ${_yt_err:-unknown error}"
    echo "     Open the YouTube tab and click ✨ Generate youtube.json manually"
  fi
  echo ""

  echo "════════════════════════════════════════════════════════════"
  echo "  ✓ Done"
  echo "  Video   : $_out_video"
  echo "  Project : $EP_DIR"
  [[ -n "$_out_srt" && -f "$_out_srt"    ]] && echo "  SRT     : $_out_srt"
  [[ -f "${INPUT_FOLDER}/thumbnail.png"   ]] && echo "  Thumb   : ${INPUT_FOLDER}/thumbnail.png"
  [[ -n "$_yt_generated"                  ]] && echo "  YouTube : $_yt_generated"
  echo "════════════════════════════════════════════════════════════"
  exit 0
fi


# ── Load config file (sets defaults; CLI flags above override) ────────────────
_VOICE_TMPFILE=""
if [[ -n "$CONFIG" ]]; then
  [[ ! -f "$CONFIG" ]] && { echo "[ERROR] config not found: $CONFIG" >&2; exit 1; }
  echo "  Config  : $CONFIG"
  _cfg_get() { python3 -c "
import json, sys
d = json.load(open('$CONFIG', encoding='utf-8'))
v = d.get('$1', '')
if isinstance(v, bool): print('1' if v else '')
else: print(v)
" 2>/dev/null; }
  [[ -z "$IMAGE"        ]] && IMAGE="$(_cfg_get background)"
  [[ -z "$LOCALE"       ]] && LOCALE="$(_cfg_get locale)"
  if [[ -z "$PROFILE" ]]; then
    PROFILE="$(python3 -c "
import json
d = json.load(open('$CONFIG', encoding='utf-8'))
p = d.get('profile', '')
if isinstance(p, dict):
    print(p.get('mode', ''))
elif isinstance(p, str):
    print(p)
" 2>/dev/null)"
  fi
  [[ -z "$TITLE_CARD"   ]] && TITLE_CARD="$(_cfg_get title_card)"
  [[ -z "$SUBTITLES"    ]] && SUBTITLES="$(_cfg_get subtitles)"
  if [[ -z "$VIDEO_EFFECT" ]]; then
    VIDEO_EFFECT="$(python3 -c "
import json
d = json.load(open('$CONFIG', encoding='utf-8'))
ve = d.get('video_effect', '')
if isinstance(ve, dict):
    print(ve.get('mode', ''))
elif isinstance(ve, str):
    print(ve)
" 2>/dev/null)"
  fi

  # If narrator is embedded in config and --voice not given, extract to a temp file.
  # Narrator format: narrator[locale][voice_name] = {enabled, gender, notes, azure_*}
  # Exactly one voice per locale must have "enabled": true; 0 or 2+ is a fatal error.
  if [[ -z "$VOICE" ]]; then
    _has_narrator="$(python3 -c "
import json
d = json.load(open('$CONFIG', encoding='utf-8'))
print('1' if 'narrator' in d else '')
" 2>/dev/null || true)"
    if [[ -n "$_has_narrator" ]]; then
      _VOICE_TMPFILE="$(mktemp /tmp/simple_narration_voice_XXXXXX.json)"
      python3 - "$CONFIG" "$_VOICE_TMPFILE" << 'VOICE_EXTRACT_EOF'
import json, sys

config_path = sys.argv[1]
out_path    = sys.argv[2]

d            = json.load(open(config_path, encoding='utf-8'))
narrator_raw = d.get('narrator', {})
narrator_out = {}
errors       = []

for locale, locale_block in narrator_raw.items():
    # Detect new multi-voice format: locale block values are dicts (named voice blocks).
    # Old flat format: locale block values are scalars (azure_voice, azure_rate, etc.).
    voice_entries = {k: v for k, v in locale_block.items() if isinstance(v, dict)}
    if not voice_entries:
        # Old flat format — pass through unchanged
        narrator_out[locale] = locale_block
        continue

    enabled = {name: cfg for name, cfg in voice_entries.items()
               if cfg.get('enabled', False)}
    count = len(enabled)
    if count == 0:
        errors.append(
            f"  locale '{locale}': no voice is enabled "
            f"(set exactly one to \"enabled\": true)")
    elif count >= 2:
        names = ', '.join(f'"{n}"' for n in enabled)
        errors.append(
            f"  locale '{locale}': {count} voices are enabled ({names}) "
            f"— enable exactly one")
    else:
        voice_name, cfg = next(iter(enabled.items()))
        # Strip config-management fields; keep TTS params (including gender for VoiceCast)
        tts_params = {k: v for k, v in cfg.items() if k not in ('enabled', 'notes')}
        narrator_out[locale] = tts_params
        print(f"  Voice selected for '{locale}': {voice_name!r}")

if errors:
    print("[ERROR] Voice selection in config is invalid:", file=sys.stderr)
    for e in errors:
        print(e, file=sys.stderr)
    sys.exit(1)

voice = {
    'narrator':      narrator_out,
    'skip_sections': d.get('skip_sections', []),
}
json.dump(voice, open(out_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
VOICE_EXTRACT_EOF
      VOICE="$_VOICE_TMPFILE"
    fi
  fi
fi

# ── Cleanup temp voice file on exit ──────────────────────────────────────────
trap '[[ -n "$_VOICE_TMPFILE" && -f "$_VOICE_TMPFILE" ]] && rm -f "$_VOICE_TMPFILE"' EXIT

# ── Apply built-in defaults for anything still unset ─────────────────────────
[[ -z "$LOCALE"  ]] && LOCALE="en"
[[ -z "$PROFILE" ]] && PROFILE="preview_local"

# ── Validate required args ────────────────────────────────────────────────────
missing=()
[[ -z "$STORY" ]] && missing+=("--story")
[[ -z "$IMAGE" ]] && missing+=("--image")
[[ -z "$VOICE" ]] && missing+=("--voice  (or embed 'narrator' in --config)")
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "[ERROR] Missing required arguments: ${missing[*]}" >&2
  echo "  Run: $0 --help" >&2
  exit 1
fi

[[ ! -f "$STORY" ]] && { echo "[ERROR] story not found: $STORY" >&2; exit 1; }
[[ ! -f "$IMAGE" ]] && { echo "[ERROR] image not found: $IMAGE" >&2; exit 1; }
[[ ! -f "$VOICE" ]] && { echo "[ERROR] voice config not found: $VOICE" >&2; exit 1; }

# ── Azure credentials check ───────────────────────────────────────────────────
if [[ -z "${AZURE_SPEECH_KEY:-}" ]]; then
  echo "[ERROR] AZURE_SPEECH_KEY env var is not set." >&2
  echo "  Export it before running:  export AZURE_SPEECH_KEY=<your-key>" >&2
  exit 1
fi

echo "════════════════════════════════════════════════════════════"
echo "  simple_run.sh — simple_narration pipeline"
echo "  Story   : $(basename "$STORY")"
echo "  Image   : $(basename "$IMAGE")"
echo "  Voice   : $(basename "$VOICE")"
echo "  Locale  : $LOCALE"
echo "  Profile : $PROFILE"
echo "  Episode : $EPISODE"
echo "════════════════════════════════════════════════════════════"
echo ""

# ── Step 1: Extract title (Python one-liner; setup.py will do the full parse) ─
if [[ -n "$TITLE" ]]; then
  _title="$TITLE"
else
  _title="$(python3 -c "
import re, sys
text = open('$STORY', encoding='utf-8').read()
lines = text.splitlines()
for line in lines:
    m = re.match(r'^#{1,6}\s+(.+)', line)
    if m:
        print(m.group(1).strip())
        sys.exit(0)
for line in lines:
    s = line.strip()
    if s:
        print(s)
        sys.exit(0)
print('narration')
" 2>/dev/null || echo "narration")"
fi

# ── Step 2: Slug = story filename (without extension); fallback to title-based ─
_slug="$(python3 - "${STORY:-}" 2>/dev/null << 'SLUG_EOF'
import re, os, sys
fn = os.path.splitext(os.path.basename(sys.argv[1] if len(sys.argv) > 1 else ''))[0]
if fn and re.match(r'^[a-zA-Z0-9_\-]+$', fn):
    print(fn); sys.exit(0)
sys.exit(1)
SLUG_EOF
)"
# Fall back to title-derived slug if filename contains non-slug characters
if [[ -z "$_slug" ]]; then
  _slug="$(python3 -c "
import re, unicodedata, secrets
orig = '''$_title'''
t = unicodedata.normalize('NFKD', orig).encode('ascii','ignore').decode('ascii')
t = t.lower().strip()
t = re.sub(r'[^\w\s-]', '', t)
t = re.sub(r'[\s_]+', '-', t)
t = re.sub(r'-{2,}', '-', t)
t = t.strip('-')[:32]
rand6 = secrets.token_hex(3)
print((t + '-' + rand6) if t else 'story-' + rand6)
" 2>/dev/null || echo "story-$(python3 -c 'import secrets; print(secrets.token_hex(3))')")"
fi

echo "  Title   : $_title"
echo "  Slug    : $_slug"
echo ""

# ── Step 3: Create episode directory structure ────────────────────────────────
EP_DIR="projects/${_slug}/episodes/${EPISODE}"
ASSETS_DIR="${EP_DIR}/assets"
RENDERS_DIR="${EP_DIR}/renders/${LOCALE}"

mkdir -p "$ASSETS_DIR" "$RENDERS_DIR"
echo "  Episode dir : $EP_DIR"

# ── Step 4: Copy image to assets/bg-provided.<ext> ───────────────────────────
_img_ext="${IMAGE##*.}"
_img_ext_lower="$(echo "$_img_ext" | tr '[:upper:]' '[:lower:]')"
COPIED_IMAGE="${ASSETS_DIR}/bg-provided.${_img_ext_lower}"
# Remove any stale bg-provided.* files from previous runs (different extension)
find "$ASSETS_DIR" -maxdepth 1 -name "bg-provided.*" ! -name "bg-provided.${_img_ext_lower}" -delete 2>/dev/null || true
cp "$IMAGE" "$COPIED_IMAGE"
echo "  Background  : $COPIED_IMAGE"
echo ""

# ── Step 5: Run simple_narration_setup.py ────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  STEP 5 — Generate contracts"
echo "════════════════════════════════════════════════════════════"

_setup_args=(
  "$EP_DIR"
  --story   "$STORY"
  --voice   "$VOICE"
  --image   "$COPIED_IMAGE"
  --title   "$_title"
  --slug    "$_slug"
  --locale  "$LOCALE"
  --profile "$PROFILE"
  --episode "$EPISODE"
)
[[ -n "$SKIP_SECTIONS"    ]] && _setup_args+=(--skip-sections "$SKIP_SECTIONS")
[[ -n "$NO_DEFAULT_SKIPS" ]] && _setup_args+=(--no-default-skips)
[[ -n "$ALT"              ]] && _setup_args+=(--alt "$ALT")
[[ -n "$SUBTITLES"        ]] && _setup_args+=(--subtitles)

python3 code/http/simple_narration_setup.py "${_setup_args[@]}"
echo ""

# ── Step 6: Run Stage 9 render (TTS + resolve + render) ──────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  STEP 6 — Stage 9 render"
echo "════════════════════════════════════════════════════════════"

# render_video.py reads these env vars when run.sh doesn't pass --title-card /
# --subtitles as CLI flags.  Export "1" when the flag is active, "" otherwise.
[[ -n "$TITLE_CARD"   ]] && export SIMPLE_NARRATION_TITLE_CARD=1    || export SIMPLE_NARRATION_TITLE_CARD=""
[[ -n "$SUBTITLES"    ]] && export SIMPLE_NARRATION_SUBTITLES=1     || export SIMPLE_NARRATION_SUBTITLES=""
# VIDEO_EFFECT passes the effect name (e.g. "light_streak") directly to render_video.py.
export SIMPLE_NARRATION_VIDEO_EFFECT="${VIDEO_EFFECT:-}"

NO_MUSIC=1 ./run.sh "$EP_DIR" 10 10
echo ""

# ── Step 7: Verify output and optionally copy ────────────────────────────────
_rendered="${EP_DIR}/renders/${LOCALE}/output.mp4"
if [[ ! -f "$_rendered" ]]; then
  echo "[ERROR] Expected output not found: $_rendered" >&2
  echo "  Check run.sh log above for render errors." >&2
  exit 1
fi

# ── Write VO approval sentinel ────────────────────────────────────────────────
# Marks TTS as reviewed so subsequent web-UI re-renders skip re-synthesis,
# and the VO tab shows clean durations (no strikethrough "timing stale" text).
python3 -c "
import sys
sys.path.insert(0, sys.argv[1])
from vo_utils import compute_sentinel_hashes, write_sentinel
ep_dir, locale = sys.argv[2], sys.argv[3]
hashes = compute_sentinel_hashes(ep_dir, locale)
write_sentinel(ep_dir, locale, hashes)
print(f'  ✓ VO sentinel approved for {locale}')
" "$SCRIPT_DIR/code/http" "$EP_DIR" "$LOCALE" || true

_final="$_rendered"

_render_dir="$(cd "$(dirname "$_final")" && pwd)"
_abs_final="$_render_dir/$(basename "$_final")"

# ── Convert thumbnail PNG → JPEG (youtube.json references .jpg) ──────────────
if [[ -f "${RENDERS_DIR}/thumbnail.png" && ! -f "${RENDERS_DIR}/thumbnail.jpg" ]]; then
  ffmpeg -y -i "${RENDERS_DIR}/thumbnail.png" \
    "${RENDERS_DIR}/thumbnail.jpg" -loglevel error 2>/dev/null \
    && echo "  ✓ thumbnail.jpg" || echo "  ⚠ thumbnail.jpg conversion failed"
fi

# ── Step 8: Pre-generate youtube.json (direct — no server required) ───────────
# Calls gen_youtube_json.py directly so youtube.json is generated even when
# the HTTP server is not running (e.g. during unattended crontab execution).
echo "════════════════════════════════════════════════════════════"
echo "  STEP 8 — Pre-generate YouTube metadata"
echo "════════════════════════════════════════════════════════════"
_yt_json_path="${RENDERS_DIR}/youtube.json"
_yt_generated=""
_story_bn="$(basename "${STORY:-}")"
echo "  Slug      : ${_slug}   Episode: ${EPISODE}   Locale: ${LOCALE}"
echo "  Generating YouTube metadata (takes ~20s) …"
_yt_result="$(python3 "${SCRIPT_DIR}/code/http/gen_youtube_json.py" \
  --slug        "${_slug}" \
  --ep_id       "${EPISODE}" \
  --locale      "${LOCALE}" \
  --story_basename "${_story_bn}" \
  2>/tmp/simple_run_yt_err.txt)"
if [[ "$_yt_result" == "ok" ]]; then
  echo "  ✓ youtube.json generated → open the YouTube tab to review & edit"
  _yt_generated="$_yt_json_path"
else
  _yt_err="$(cat /tmp/simple_run_yt_err.txt 2>/dev/null)"
  echo "  ✗ Generation failed: ${_yt_err:-unknown error}"
  echo "     Open the YouTube tab and click ✨ Generate youtube.json manually"
fi
echo ""

echo "════════════════════════════════════════════════════════════"
echo "  ✓ Done"
echo "  Video     : $_abs_final"
[[ -f "$_render_dir/chapters.txt"   ]] && echo "  Chapters  : $_render_dir/chapters.txt"
[[ -f "$_render_dir/thumbnail.png"  ]] && echo "  Thumbnail : $_render_dir/thumbnail.png"
[[ -f "$_render_dir/output.${LOCALE}.srt" ]] && echo "  SRT       : $_render_dir/output.${LOCALE}.srt"
[[ -n "$_yt_generated"              ]] && echo "  YouTube   : $_yt_generated"
echo "════════════════════════════════════════════════════════════"
