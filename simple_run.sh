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
#   - Runs Whisper on stitched audio to generate subtitles
#     (if --story is provided, its text is passed as Whisper initial_prompt
#      to improve transcription accuracy)
#   - Burns subtitles into video
#   - Saves output.mp4 and output.srt to --input_folder
#   No Azure TTS involved. Output: <input_folder>/output.mp4
#
#   Usage:
#     ./simple_run.sh \
#       --input_folder /path/to/clips/dir \
#       [--story       /path/to/story.txt]
#
# ── MODE 2: TTS (--story, default) ───────────────────────────────────────────
#   Given a story.md, a background image/video, and a voice_config.json,
#   produces output.mp4 via Azure TTS with zero manual steps.
#
#   Usage:
#     ./simple_run.sh \
#       --story  /tmp/story.md \
#       --image  /tmp/cover.jpg \
#       --voice  /tmp/voice_config.json \
#       --out    /tmp/output.mp4
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
CONFIG=""
STORY=""
IMAGE=""
VOICE=""
OUT=""
TITLE=""
LOCALE=""
PROFILE=""
EPISODE="s01e01"
SKIP_SECTIONS=""
NO_DEFAULT_SKIPS=""
ALT=""
TITLE_CARD=""
SUBTITLES=""
# clips mode
INPUT_FOLDER=""
SPEED=""          # playback speed multiplier (e.g. 0.8 for 80% speed; clips mode only)
SUBTITLE_SHIFT_MS=""  # ms to shift subtitle timestamps; negative = earlier, positive = later
                       # default 0 (story or whisper-only); use a negative value (e.g. -300)
                       # only if subtitles appear noticeably after speech starts —
                       # negative shifts also advance t_out, which causes premature
                       # subtitle transitions (wrong subtitle mid-sentence) at slow speeds

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)           CONFIG="$2";         shift 2 ;;
    --story)            STORY="$2";          shift 2 ;;
    --image)            IMAGE="$2";          shift 2 ;;
    --voice)            VOICE="$2";          shift 2 ;;
    --out)              OUT="$2";            shift 2 ;;
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
      sed -n '/^# Usage/,/^# Requires/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown flag: $1" >&2
      exit 1
      ;;
  esac
done

# ── CLIPS MODE: xfade stitch + Whisper subtitles (early exit) ────────────────
if [[ -n "$INPUT_FOLDER" ]]; then
  [[ ! -d "$INPUT_FOLDER" ]] && { echo "[ERROR] --input_folder not found: $INPUT_FOLDER" >&2; exit 1; }

  _out_video="${INPUT_FOLDER}/output.mp4"
  _out_srt="${INPUT_FOLDER}/output.srt"
  # _out_audio removed — per-clip Whisper uses temp files (no full-audio WAV)

  echo "════════════════════════════════════════════════════════════"
  echo "  simple_run.sh — clips mode (xfade stitch + Whisper SRT)"
  echo "  Folder  : $INPUT_FOLDER"
  echo "  Story   : ${STORY:-none (no initial_prompt)}"
  echo "  Out     : $_out_video"
  echo "════════════════════════════════════════════════════════════"
  echo ""

  # ── Step 1: xfade stitch → output.mp4 ──────────────────────────────────────
  # Strategy: parallel batch stitching.
  #   Problem: a single filter_complex with N inputs keeps N decoders open and
  #   evaluates the full N-stage chain per output frame → O(N) slowdown (50 min
  #   for 21 clips instead of ~2 min).
  #   Fix: split clips into BATCH_SIZE groups, stitch each group in parallel
  #   (one process per CPU core) using ultrafast preset for intermediates, then
  #   do a final quality stitch of the batch results.
  echo "  STEP 1 — xfade stitch (parallel batch)"
  python3 - "$INPUT_FOLDER" "$_out_video" << 'PYEOF'
import subprocess, sys, os, glob, re, tempfile, concurrent.futures

folder, out = sys.argv[1], sys.argv[2]
XFADE      = 0.4   # dissolve duration (seconds)
BATCH_SIZE = 5     # clips per batch; tune down if RAM is tight
N_WORKERS  = os.cpu_count() or 4   # auto-detect cores; fallback 4

def natural_key(path: str):
    name = os.path.basename(path)
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r'(\d+)', name)]

def probe_dur(path: str) -> float:
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1', path],
        capture_output=True, text=True, check=True)
    return float(r.stdout.strip().split('=')[1])

def stitch(clip_list: list, out_path: str, preset: str = 'ultrafast', crf: int = 23) -> None:
    """Stitch clip_list with:
      VIDEO: tpad (freeze end/start frames) + xfade → no real content consumed
      AUDIO: adelay each clip to its video-start time + amix → no speech blending
    During the XFADE transition window the audio has a brief natural silence
    (~0.4s) which is imperceptible on narration/news content.
    """
    import shutil
    if len(clip_list) == 1:
        shutil.copy2(clip_list[0], out_path)
        return

    n    = len(clip_list)
    durs = [probe_dur(c) for c in clip_list]
    fp   = []   # filter_complex parts

    # ── 1. tpad each video: freeze start (incoming) and/or end (outgoing) ─────
    for i in range(n):
        pad_s = XFADE if i > 0     else 0.0
        pad_e = XFADE if i < n - 1 else 0.0
        if pad_s and pad_e:
            tpad = (f"tpad=start_mode=clone:start_duration={pad_s:.3f}"
                    f":stop_mode=clone:stop_duration={pad_e:.3f}")
        elif pad_s:
            tpad = f"tpad=start_mode=clone:start_duration={pad_s:.3f}"
        elif pad_e:
            tpad = f"tpad=stop_mode=clone:stop_duration={pad_e:.3f}"
        else:
            tpad = "copy"
        fp.append(f"[{i}:v]{tpad}[pv{i}]")

    # ── 2. xfade chain on padded video ────────────────────────────────────────
    # offset = when clip[i-1]'s real content ends in the output timeline
    cum   = durs[0]
    vprev = "[pv0]"
    for i in range(1, n):
        vl = "vout" if i == n - 1 else f"v{i}"
        fp.append(
            f"{vprev}[pv{i}]xfade=transition=fade"
            f":duration={XFADE:.3f}:offset={cum:.3f}[{vl}]"
        )
        vprev  = f"[{vl}]"
        cum   += XFADE + durs[i]

    # ── 3. Audio: delay each clip to align with its video start time ──────────
    # clip[i] video starts at: sum(D[0..i-1]) + i*XFADE
    # → brief natural silence during each transition window
    delay_ms = 0.0
    for i in range(n):
        d = int(delay_ms)
        fp.append(f"[{i}:a]adelay={d}|{d}[da{i}]")
        delay_ms += (durs[i] + XFADE) * 1000

    total_dur = sum(durs) + (n - 1) * XFADE
    amix_in   = "".join(f"[da{i}]" for i in range(n))
    fp.append(
        f"{amix_in}amix=inputs={n}:normalize=0:dropout_transition=0"
        f",apad=pad_dur={total_dur:.3f}[aout]"
    )

    cmd = ['ffmpeg', '-y', '-threads', '1']
    for c in clip_list:
        cmd += ['-i', c]
    cmd += [
        '-filter_complex', ';'.join(fp),
        '-map', '[vout]', '-map', '[aout]',
        '-c:v', 'libx264', '-crf', str(crf), '-preset', preset,
        '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k',
        '-t', f'{total_dur:.3f}',
        out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-600:], file=sys.stderr)
        raise RuntimeError(f"ffmpeg failed on {os.path.basename(out_path)}")

# ── Collect clips ─────────────────────────────────────────────────────────────
clips = sorted(
    (f for f in glob.glob(os.path.join(folder, '*.mp4'))
     if os.path.basename(f) != 'output.mp4'),
    key=natural_key,
)
N = len(clips)
if N < 2:
    print(f"[ERROR] Need ≥2 clips in {folder}, found {N}", file=sys.stderr)
    sys.exit(1)

print(f"  Clips      : {N}")
for c in clips:
    print(f"    {os.path.basename(c)}")

durs_all = [probe_dur(c) for c in clips]
total    = sum(durs_all) + XFADE * (N - 1)   # pad approach: adds XFADE per boundary
print(f"  Durations  : {[f'{d:.2f}s' for d in durs_all]}")
print(f"  Xfade      : {XFADE}s × {N-1} transitions  →  total ≈ {total:.2f}s")

# ── Fast path: ≤ BATCH_SIZE clips → single stitch ────────────────────────────
if N <= BATCH_SIZE:
    print(f"  Strategy   : single-pass (≤{BATCH_SIZE} clips)")
    stitch(clips, out, preset='medium', crf=18)
    sys.exit(0)

# ── Batch path: parallel batch stitches → final quality join ─────────────────
batches = [clips[i:i + BATCH_SIZE] for i in range(0, N, BATCH_SIZE)]
print(f"  Strategy   : {len(batches)} batches × ≤{BATCH_SIZE} clips  "
      f"({N_WORKERS} workers / {os.cpu_count()} cores detected)  →  quality final join")

with tempfile.TemporaryDirectory() as tmp:
    batch_outs = [os.path.join(tmp, f'batch_{i:04d}.mp4') for i in range(len(batches))]

    # Round 1 — parallel batch stitches (ultrafast, quality not critical)
    def _stitch_batch(args):
        idx, b, bo = args
        print(f"  [batch {idx+1}/{len(batches)}] stitching {len(b)} clips…")
        stitch(b, bo, preset='ultrafast', crf=23)
        print(f"  [batch {idx+1}/{len(batches)}] done")

    with concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(_stitch_batch, (i, b, bo))
                for i, (b, bo) in enumerate(zip(batches, batch_outs))]
        for f in concurrent.futures.as_completed(futs):
            f.result()   # re-raises any exception

    # Round 2 — final quality stitch of batch results
    print(f"  Joining {len(batch_outs)} batches (medium CRF 18)…")
    stitch(batch_outs, out, preset='medium', crf=18)

print(f"  Done → {os.path.basename(out)}")
PYEOF
  echo ""

  # ── Step 2+3: Per-clip Whisper → combined output.srt ────────────────────────
  # ROOT CAUSE FIX: running Whisper on the full N-clip stitched audio (old approach)
  # caused accumulated timestamp drift in the "small" model on long audio (5+ min).
  # By clip 14 the drift reached ~1-2s, which after --speed 0.8 (×1.25 amplifier)
  # produced a mid-sentence subtitle jump at 2:58.
  #
  # Fix: load Whisper model once; transcribe each clip independently; offset each
  # clip's timestamps by its known start position in the stitched video timeline.
  # Per-clip accuracy confirmed correct by the user's 2-clip test.
  #
  # Model choice: 'small' when --story is provided (timing only; text comes from
  # story.txt). 'medium' when no story (need transcription accuracy).
  echo "  STEP 2+3 — Per-clip Whisper (accurate timing per clip → combined SRT)"
  _whisper_model="small"
  [[ -z "$STORY" ]] && _whisper_model="medium"   # no story.txt → need accuracy
  python3 - "$INPUT_FOLDER" "$_out_srt" "$_whisper_model" << 'WHISPER_CLIP_EOF'
import sys, os, glob, re, subprocess, tempfile

folder     = sys.argv[1]   # input folder with clips
srt_out    = sys.argv[2]   # combined SRT output path
model_name = sys.argv[3]   # 'small' or 'medium'
XFADE      = 0.4           # must match Step 1 stitch XFADE constant

def natural_key(p):
    name = os.path.basename(p)
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', name)]

def probe_dur(p):
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1', p],
        capture_output=True, text=True, check=True)
    return float(r.stdout.strip().split('=')[1])

def fmt_ts(ms):
    ms = max(0, int(ms))
    h,  ms = divmod(ms, 3_600_000)
    m,  ms = divmod(ms,    60_000)
    s,  ms = divmod(ms,     1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

# ── Collect clips ──────────────────────────────────────────────────────────
clips = sorted(
    (f for f in glob.glob(os.path.join(folder, '*.mp4'))
     if os.path.basename(f) != 'output.mp4'),
    key=natural_key
)
N = len(clips)
print(f"  Per-clip Whisper: {N} clips, model={model_name}")

clip_durs = [probe_dur(c) for c in clips]

# offset[i] = position in the stitched video where clip[i]'s content starts
# = sum(durs[0..i-1]) + i * XFADE   (same formula used by the stitch filter)
offsets_sec = []
cur = 0.0
for d in clip_durs:
    offsets_sec.append(cur)
    cur += d + XFADE

# ── Load Whisper model once for all clips ─────────────────────────────────
try:
    import whisper as _wm
    print(f"  Loading Whisper model '{model_name}'…")
    _model = _wm.load_model(model_name)
    _use_api = True
    print(f"  Model loaded.")
except ImportError:
    _use_api = False
    print("  [warn] whisper Python API not available — using CLI (slower)",
          file=sys.stderr)

# ── Transcribe each clip; accumulate globally-offset segments ─────────────
all_segs = []   # (abs_t_in_ms, abs_t_out_ms, text)

with tempfile.TemporaryDirectory() as tmp:
    for i, (clip, off_sec) in enumerate(zip(clips, offsets_sec)):
        clip_name = os.path.basename(clip)
        dur_sec   = clip_durs[i]
        off_ms    = int(off_sec * 1000)
        dur_ms    = int(dur_sec * 1000)

        print(f"  [{i+1}/{N}] {clip_name}  offset={off_sec:.3f}s")

        # Extract 16 kHz mono WAV from this clip only
        wav = os.path.join(tmp, f'c{i:04d}.wav')
        r = subprocess.run(
            ['ffmpeg', '-y', '-i', clip,
             '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le',
             wav, '-loglevel', 'error'],
            capture_output=True
        )
        if r.returncode != 0:
            print(f"  [warn] audio extract failed for clip {i}", file=sys.stderr)
            continue

        if _use_api:
            # Python API: model already loaded — no repeated model-load overhead
            result   = _model.transcribe(wav, task='transcribe', verbose=None)
            segs_raw = [
                (int(s['start'] * 1000),
                 int(s['end']   * 1000),
                 s['text'].strip())
                for s in result.get('segments', [])
                if s['text'].strip()
            ]
        else:
            # CLI fallback (slower: one model load per clip)
            wdir = os.path.join(tmp, f'w{i:04d}')
            os.makedirs(wdir, exist_ok=True)
            subprocess.run(
                ['whisper', wav, '--model', model_name,
                 '--output_format', 'srt', '--output_dir', wdir,
                 '--task', 'transcribe'],
                capture_output=True
            )
            clip_srt = os.path.join(wdir, f'c{i:04d}.srt')
            segs_raw = []
            if os.path.exists(clip_srt):
                def _ts(t):
                    h, mn, rest = t.split(':')
                    s2, ms = rest.split(',')
                    return (int(h)*3_600_000 + int(mn)*60_000
                            + int(s2)*1_000 + int(ms))
                for blk in re.split(
                        r'\n{2,}',
                        open(clip_srt, encoding='utf-8').read().strip()):
                    ls = blk.strip().splitlines()
                    if len(ls) < 3:
                        continue
                    m2 = re.match(r'(\S+)\s*-->\s*(\S+)', ls[1])
                    if not m2:
                        continue
                    segs_raw.append((_ts(m2.group(1)), _ts(m2.group(2)),
                                     ' '.join(ls[2:]).strip()))

        for t_in_ms, t_out_ms, text in segs_raw:
            # Clamp t_out to this clip's boundary (prevents bleed into next gap)
            abs_in  = t_in_ms  + off_ms
            abs_out = min(t_out_ms, dur_ms) + off_ms
            if abs_out <= abs_in:
                abs_out = abs_in + 50
            all_segs.append((abs_in, abs_out, text))

all_segs.sort(key=lambda x: x[0])

# ── Write combined SRT ─────────────────────────────────────────────────────
out_lines = []
for seq, (t_in, t_out, text) in enumerate(all_segs, 1):
    out_lines += [str(seq), f"{fmt_ts(t_in)} --> {fmt_ts(t_out)}", text, '']

open(srt_out, 'w', encoding='utf-8').write('\n'.join(out_lines))
print(f"  → {len(all_segs)} segments from {N} clips → {srt_out}")
WHISPER_CLIP_EOF

  if [[ ! -f "$_out_srt" ]]; then
    echo "[WARN] Per-clip Whisper SRT not generated — skipping subtitle burn" >&2
    _out_srt=""
  else
    echo "    SRT (combined): $_out_srt"
  fi

  # ── Step 3b: Build final SRT ─────────────────────────────────────────────────
  # Mode A (--story provided): use Whisper timestamps only; pull text from
  #   story.txt in reading order → correct characters, proper nouns, punctuation.
  # Mode B (no --story): split Whisper's transcribed text into ≤25-char lines.
  if [[ -n "$_out_srt" && -f "$_out_srt" ]]; then
    if [[ -n "$SUBTITLE_SHIFT_MS" ]]; then
      _shift_ms="$SUBTITLE_SHIFT_MS"
    elif [[ -n "$STORY" && -f "$STORY" ]]; then
      _shift_ms="0"
    else
      _shift_ms="0"
    fi
    if [[ -n "$STORY" && -f "$STORY" ]]; then
      echo "  STEP 3b — Map story.txt text onto Whisper timestamps (shift ${_shift_ms} ms)"
    else
      echo "  STEP 3b — Split Whisper lines to ≤25 chars (shift ${_shift_ms} ms)"
    fi
    python3 - "$_out_srt" "${STORY:-}" "$_shift_ms" << 'SPLITEOF'
import sys, re, unicodedata

srt_path   = sys.argv[1]
story_path = sys.argv[2] if len(sys.argv) > 2 else ''
shift_ms   = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != '' else 0

STRONG_BREAKS = set('。！？…')          # sentence-end — preferred card boundary
WEAK_BREAKS   = set('，；、,;!?')        # clause-end   — fallback boundary
ALL_BREAKS    = STRONG_BREAKS | WEAK_BREAKS
MAX_CARD_VW   = 60    # max visual width per card ≈ 30 CJK chars
MIN_CARD_VW   = 14    # merge cards shorter than this into the previous card

def vw(s):
    return sum(2 if unicodedata.east_asian_width(c) in ('W','F') else 1 for c in s)

def natural_chunks(text):
    """Split text into subtitle cards at natural sentence/clause boundaries.
    Never breaks mid-word; merges short trailing fragments into the previous card."""
    text = text.strip()
    if not text:
        return []
    # Split at sentence-ending punctuation first (preferred card boundaries)
    parts = re.split(r'(?<=[。！？…])', text)
    cards = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if vw(part) <= MAX_CARD_VW:
            cards.append(part)
        else:
            # Sentence too wide: split at clause boundaries, grouping greedily
            subs = re.split(r'(?<=[，；、,;!?])', part)
            buf = ''
            for s in subs:
                s = s.strip()
                if not s:
                    continue
                if not buf:
                    buf = s
                elif vw(buf) + vw(s) <= MAX_CARD_VW:
                    buf += s
                else:
                    if buf:
                        cards.append(buf)
                    buf = s
            if buf:
                cards.append(buf)
    # Merge short trailing cards into the previous card
    merged = []
    for card in cards:
        if (merged
                and vw(card) < MIN_CARD_VW
                and vw(merged[-1]) + vw(card) <= MAX_CARD_VW):
            merged[-1] += card
        else:
            merged.append(card)
    return merged or [text]

def snap_to_break(text, pos, window=25):
    """Snap pos to the nearest ALL_BREAKS position within ±window chars.
    BACKWARD is checked first: never pull the next sentence into this segment.
    Forward is a fallback only when no backward break is found within window."""
    n = len(text)
    for delta in range(0, window + 1):
        if pos - delta >= 0 and text[pos - delta] in ALL_BREAKS:
            return pos - delta + 1   # prefer backward: don't overshoot
        if pos + delta < n and text[pos + delta] in ALL_BREAKS:
            return pos + delta + 1   # fallback: forward only if no backward found
    return pos   # no break found within window; use exact position

def srt_ts(ms):
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms,    60_000)
    s, ms = divmod(ms,     1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def parse_ts(t):
    h, m, rest = t.split(':'); s, ms = rest.split(',')
    return int(h)*3_600_000 + int(m)*60_000 + int(s)*1_000 + int(ms)

def emit(out_lines, seq, t_in, t_out, text):
    """Mode B only: split Whisper text into cards using natural_chunks."""
    chunks      = natural_chunks(text)
    total_chars = sum(len(c) for c in chunks) or 1
    dur, cur    = t_out - t_in, t_in
    for i, chunk in enumerate(chunks):
        share = round(dur * len(chunk) / total_chars)
        end   = cur + max(share, 200)
        if i == len(chunks) - 1: end = t_out
        out_lines += [str(seq), f"{srt_ts(cur)} --> {srt_ts(end)}", chunk, '']
        seq += 1; cur = end
    return seq

# ── Parse Whisper SRT for timing ──────────────────────────────────────────────
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

# Apply timestamp shift.  Default shift_ms=0 uses Whisper's native boundaries.
# Negative shift (e.g. -300) corrects Whisper's late-start bias but ALSO advances
# t_out, which can make transitions appear before the sentence ends — especially at
# slow playback speeds where each 1ms original-timeline error × (1/speed) output ms.
# Clamp: t_in ≥ 0 ms; t_out ≥ t_in + 50 ms.
segments = []
for t_in, t_out, text in raw_segs:
    t_in_s  = max(0, t_in  + shift_ms)
    t_out_s = max(t_in_s + 50, t_out + shift_ms)
    segments.append((t_in_s, t_out_s, text))
if shift_ms != 0:
    print(f"    Timestamp shift: {shift_ms:+d} ms applied to {len(segments)} segment(s)")

out_lines, seq = [], 1

# ── Mode A: story.txt text mapped onto Whisper timing ─────────────────────────
if story_path:
    raw = open(story_path, encoding='utf-8').read()
    raw = re.sub(r'^#{1,6}\s+', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'^[-*]\s+',   '', raw, flags=re.MULTILINE)
    raw = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', raw)
    raw = re.sub(r'\s+', ' ', raw).strip()

    total_chars         = len(raw)
    # Proportion story chars by Whisper's transcribed char count per segment,
    # NOT by segment duration. Whisper char count is immune to trailing silence
    # (silence adds time but not characters), so this avoids the drift caused by
    # xfade transition gaps inflating each segment's duration.
    total_whisper_chars = sum(len(wt) for _, _, wt in segments) or 1
    story_cursor        = 0

    for seg_i, (t_in, t_out, whisper_text) in enumerate(segments):
        n_chars = round(len(whisper_text) / total_whisper_chars * total_chars)
        target  = story_cursor + n_chars
        # Last segment: consume all remaining text
        if seg_i == len(segments) - 1:
            end_pos = total_chars
        else:
            # Snap to nearest sentence/clause break to avoid mid-word cuts
            end_pos = snap_to_break(raw, min(target, total_chars - 1))
            end_pos = max(end_pos, story_cursor + 1)   # always advance

        text = raw[story_cursor:end_pos].strip()
        story_cursor = end_pos
        if not text:
            continue

        # Split text into natural cards (sentence/clause boundaries, no mid-word cuts)
        cards = natural_chunks(text)
        if not cards:
            continue

        # Time cards proportionally within this segment's window
        seg_chars = sum(len(c) for c in cards) or 1
        dur, cur  = t_out - t_in, t_in
        for i, card in enumerate(cards):
            share = round(dur * len(card) / seg_chars)
            end   = cur + max(share, 200)
            if i == len(cards) - 1:
                end = t_out
            out_lines += [str(seq), f"{srt_ts(cur)} --> {srt_ts(end)}", card, '']
            seq += 1
            cur  = end

    print(f"    {seq-1} cards from story.txt "
          f"({total_chars} story chars / {total_whisper_chars} whisper chars, "
          f"natural boundaries)")

# ── Mode B: split Whisper's own transcribed text ──────────────────────────────
else:
    for t_in, t_out, text in segments:
        if not text: continue
        seq = emit(out_lines, seq, t_in, t_out, text)
    print(f"    {seq-1} cards from Whisper transcription")

open(srt_path, 'w', encoding='utf-8').write('\n'.join(out_lines))
SPLITEOF
  fi
  # (per-clip temp WAVs are cleaned up automatically by Python tempfile.TemporaryDirectory)
  echo ""

  # ── Steps 4+5: Subtitle burn + optional speed — single ffmpeg pass ──────────
  # Merged to avoid a redundant full re-encode.  Four cases:
  #   subtitles + speed  → -vf "subtitles,setpts"  -af "atempo"
  #   subtitles only     → -vf "subtitles"          -c:a copy
  #   speed only         → -vf "setpts"             -af "atempo"
  #   neither            → stream-copy (instant)
  _has_srt=0;  [[ -n "$_out_srt" && -f "$_out_srt" ]] && _has_srt=1
  _has_spd=0;  [[ -n "$SPEED" && "$SPEED" != "1" && "$SPEED" != "1.0" ]] && _has_spd=1

  if [[ $_has_srt -eq 0 && $_has_spd -eq 0 ]]; then
    echo "  STEP 4+5 — No subtitles, no speed change (skipped)"
  else
    _out_final="${INPUT_FOLDER}/_output_final.mp4"

    # Build video filter chain
    _vf_parts=()
    if [[ $_has_srt -eq 1 ]]; then
      _srt_esc="$(python3 -c "
s = '$_out_srt'
s = s.replace('\\\\', '\\\\\\\\').replace(\"'\", \"\\\\'\").replace(':', '\\\\:')
print(s)
")"
      _vf_parts+=("subtitles='${_srt_esc}':force_style='FontName=Noto Sans CJK SC,FontSize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Bold=1,Outline=2'")
    fi
    if [[ $_has_spd -eq 1 ]]; then
      _speed_factors="$(python3 -c "
s = float('$SPEED')
pts = 1.0 / s
filters = []
rem = s
while rem < 0.5:
    filters.append('atempo=0.5'); rem /= 0.5
while rem > 2.0:
    filters.append('atempo=2.0'); rem /= 2.0
filters.append(f'atempo={rem:.6f}')
print(f'{pts:.6f}', ','.join(filters))
")"
      _setpts_factor="$(echo "$_speed_factors" | awk '{print $1}')"
      _atempo_filter="$(echo "$_speed_factors" | awk '{print $2}')"
      _vf_parts+=("setpts=${_setpts_factor}*PTS")
    fi
    _vf_str="$(IFS=,; echo "${_vf_parts[*]}")"

    # Build audio args
    if [[ $_has_spd -eq 1 ]]; then
      _audio_args=(-af "$_atempo_filter" -c:a aac -b:a 128k)
    else
      _audio_args=(-c:a copy)
    fi

    _label="subtitles$([ $_has_spd -eq 1 ] && echo ' + speed')"
    echo "  STEP 4+5 — ${_label} (single pass)"
    ffmpeg -y -i "$_out_video" \
      -vf "$_vf_str" \
      "${_audio_args[@]}" \
      -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p \
      "$_out_final" -loglevel error
    mv "$_out_final" "$_out_video"
    echo "    Done."
  fi
  echo ""

  echo "════════════════════════════════════════════════════════════"
  echo "  ✓ Done"
  echo "  Video : $_out_video"
  [[ -n "$_out_srt" && -f "$_out_srt" ]] && echo "  SRT   : $_out_srt"
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
  [[ -z "$IMAGE"      ]] && IMAGE="$(_cfg_get background)"
  [[ -z "$LOCALE"     ]] && LOCALE="$(_cfg_get locale)"
  [[ -z "$PROFILE"    ]] && PROFILE="$(_cfg_get profile)"
  [[ -z "$TITLE_CARD" ]] && TITLE_CARD="$(_cfg_get title_card)"
  [[ -z "$SUBTITLES"  ]] && SUBTITLES="$(_cfg_get subtitles)"

  # If narrator is embedded in config and --voice not given, extract to a temp file
  if [[ -z "$VOICE" ]]; then
    _has_narrator="$(python3 -c "
import json
d = json.load(open('$CONFIG', encoding='utf-8'))
print('1' if 'narrator' in d else '')
" 2>/dev/null)"
    if [[ -n "$_has_narrator" ]]; then
      _VOICE_TMPFILE="$(mktemp /tmp/simple_narration_voice_XXXXXX.json)"
      python3 -c "
import json
d = json.load(open('$CONFIG', encoding='utf-8'))
voice = {
    'narrator':      d['narrator'],
    'skip_sections': d.get('skip_sections', []),
}
json.dump(voice, open('$_VOICE_TMPFILE', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
" 2>/dev/null
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

# ── Step 2: Slugify title → always a fresh random suffix ─────────────────────
_slug="$(python3 -c "
import re, sys, unicodedata, secrets
orig = '''$_title'''
t = unicodedata.normalize('NFKD', orig).encode('ascii','ignore').decode('ascii')
t = t.lower().strip()
t = re.sub(r'[^\w\s-]', '', t)
t = re.sub(r'[\s_]+', '-', t)
t = re.sub(r'-{2,}', '-', t)
t = t.strip('-')[:32]  # cap at 32 chars to keep slug readable
rand6 = secrets.token_hex(3)
print((t + '-' + rand6) if t else 'story-' + rand6)
" 2>/dev/null || echo "story-$(python3 -c 'import secrets; print(secrets.token_hex(3))')")"

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

python3 code/http/simple_narration_setup.py "${_setup_args[@]}"
echo ""

# ── Step 6: Run Stage 9 render (TTS + resolve + render) ──────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  STEP 6 — Stage 9 render"
echo "════════════════════════════════════════════════════════════"

# render_video.py reads these env vars when run.sh doesn't pass --title-card /
# --subtitles as CLI flags.  Export "1" when the flag is active, "" otherwise.
[[ -n "$TITLE_CARD" ]] && export SIMPLE_NARRATION_TITLE_CARD=1 || export SIMPLE_NARRATION_TITLE_CARD=""
[[ -n "$SUBTITLES"  ]] && export SIMPLE_NARRATION_SUBTITLES=1  || export SIMPLE_NARRATION_SUBTITLES=""

NO_MUSIC=1 ./run.sh "$EP_DIR" 10 10
echo ""

# ── Step 7: Verify output and optionally copy ────────────────────────────────
_rendered="${EP_DIR}/renders/${LOCALE}/output.mp4"
if [[ ! -f "$_rendered" ]]; then
  echo "[ERROR] Expected output not found: $_rendered" >&2
  echo "  Check run.sh log above for render errors." >&2
  exit 1
fi

if [[ -n "$OUT" ]]; then
  _out_dir="$(dirname "$OUT")"
  [[ -n "$_out_dir" && "$_out_dir" != "." ]] && mkdir -p "$_out_dir"
  cp "$_rendered" "$OUT"
  _final="$OUT"
else
  _final="$_rendered"
fi

_render_dir="$(cd "$(dirname "$_final")" && pwd)"
_abs_final="$_render_dir/$(basename "$_final")"
echo "════════════════════════════════════════════════════════════"
echo "  ✓ Done"
echo "  Video     : $_abs_final"
[[ -f "$_render_dir/chapters.txt"   ]] && echo "  Chapters  : $_render_dir/chapters.txt"
[[ -f "$_render_dir/thumbnail.png"  ]] && echo "  Thumbnail : $_render_dir/thumbnail.png"
[[ -f "$_render_dir/output.${LOCALE}.srt" ]] && echo "  SRT       : $_render_dir/output.${LOCALE}.srt"
echo "════════════════════════════════════════════════════════════"
