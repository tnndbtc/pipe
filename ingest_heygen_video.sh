#!/usr/bin/env bash
# =============================================================================
# ingest_heygen_video.sh — Drop a HeyGen-generated MP4 into the latest AI
# story project, generate youtube.json, and upload to YouTube (private).
#
# Usage (explicit video path):
#   ./ingest_heygen_video.sh <video.mp4> --ep_dir projects/<slug>/episodes/<id>
#
# Usage (auto-detect — copy any *.mp4 into the episode dir first):
#   ./ingest_heygen_video.sh --ep_dir projects/<slug>/episodes/<id>
#
# If --ep_dir is not given, reads the latest project path from:
#   ../story_engine/.last_ai_ep_dir
#
# The script expects the project skeleton to already exist (created by
#   run_generate.sh --deep-story --story-only --profile run2_ai).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VIDEO_PATH=""
EP_DIR_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ep_dir) EP_DIR_OVERRIDE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [video.mp4] --ep_dir projects/<slug>/episodes/<id>"
            echo "       (video.mp4 optional — auto-detected from episode dir if omitted)"
            exit 0 ;;
        -*)
            echo "[ERROR] Unknown flag: $1" >&2; exit 1 ;;
        *)
            VIDEO_PATH="$1"; shift ;;
    esac
done

# ── Resolve EP_DIR (relative to pipe/) ────────────────────────────────────────
if [[ -n "$EP_DIR_OVERRIDE" ]]; then
    EP_DIR="$EP_DIR_OVERRIDE"
else
    LAST_EP_FILE="../story_engine/.last_ai_ep_dir"
    if [[ ! -f "$LAST_EP_FILE" ]]; then
        echo "[ERROR] No .last_ai_ep_dir found." >&2
        echo "        Run story generation first or pass --ep_dir explicitly." >&2
        exit 1
    fi
    EP_DIR="$(cat "$LAST_EP_FILE")"
fi

if [[ ! -d "$EP_DIR" ]]; then
    echo "[ERROR] EP_DIR not found: $EP_DIR" >&2
    exit 1
fi

# ── Resolve VIDEO_PATH (auto-detect from episode dir if not given) ─────────────
if [[ -z "$VIDEO_PATH" ]]; then
    mapfile -t _mp4s < <(find "$EP_DIR" -maxdepth 1 -name "*.mp4" 2>/dev/null)
    if [[ ${#_mp4s[@]} -eq 0 ]]; then
        echo "[ERROR] No .mp4 found in $EP_DIR" >&2
        echo "        Copy your video there or pass it as the first argument." >&2
        exit 1
    fi
    if [[ ${#_mp4s[@]} -gt 1 ]]; then
        echo "[ERROR] Multiple .mp4 files found in $EP_DIR — ambiguous:" >&2
        for f in "${_mp4s[@]}"; do echo "        $f" >&2; done
        echo "        Remove all but one, or pass the path explicitly." >&2
        exit 1
    fi
    VIDEO_PATH="${_mp4s[0]}"
    echo "  Auto-detected video: $VIDEO_PATH"
fi

if [[ ! -f "$VIDEO_PATH" ]]; then
    echo "[ERROR] Video file not found: $VIDEO_PATH" >&2
    exit 1
fi

# ── Detect locale from renders/ subdir ────────────────────────────────────────
LOCALE="$(ls "${EP_DIR}/renders/" 2>/dev/null | head -1)"
[[ -z "$LOCALE" ]] && LOCALE="zh-Hans"
RENDERS_DIR="${EP_DIR}/renders/${LOCALE}"
mkdir -p "$RENDERS_DIR"

# ── Derive slug and ep_id from path ───────────────────────────────────────────
# EP_DIR pattern: projects/<slug>/episodes/<ep_id>
_SLUG="$(basename "$(dirname "$(dirname "$EP_DIR")")")"
_EP_ID="$(basename "$EP_DIR")"

echo "════════════════════════════════════════════════════════════"
echo "  ingest_heygen_video.sh"
echo "  Video     : $VIDEO_PATH"
echo "  Project   : $EP_DIR"
echo "  Locale    : $LOCALE"
echo "════════════════════════════════════════════════════════════"
echo ""

# ── Load env (AZURE keys not needed here, but YouTube creds may be) ───────────
if [[ -f "../story_engine/.env" ]]; then
    set -a; source "../story_engine/.env"; set +a
fi
source /home/tnnd/.virtualenvs/pipe/bin/activate

# ── STEP 1: Copy video → renders/<locale>/output.mp4 ─────────────────────────
echo "  STEP 1 — Copy video → renders/${LOCALE}/output.mp4"
OUT_VIDEO="${RENDERS_DIR}/output.mp4"
cp "$VIDEO_PATH" "$OUT_VIDEO"
echo "  ✓ $(du -sh "$OUT_VIDEO" | cut -f1)  $OUT_VIDEO"
echo ""

# ── STEP 2: Generate thumbnail (frame grab + title text burn) ─────────────────
echo "  STEP 2 — Generate thumbnail"
THUMB_RAW="${RENDERS_DIR}/thumbnail_raw.png"
THUMB="${RENDERS_DIR}/thumbnail.jpg"

_dur="$(ffprobe -v error -show_entries format=duration \
        -of default=noprint_wrappers=1 "$OUT_VIDEO" 2>/dev/null \
        | cut -d= -f2 || echo '10')"
_ss="$(python3 -c "print(f'{min(5.0, float(\"$_dur\") * 0.1):.2f}')" 2>/dev/null || echo '1')"

if ffmpeg -y -ss "$_ss" -i "$OUT_VIDEO" -frames:v 1 \
    -vf 'scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2' \
    -q:v 2 "$THUMB_RAW" -loglevel error 2>/dev/null; then

    # Extract title from story.txt (first ## heading)
    _TITLE=""
    if [[ -f "${EP_DIR}/story.txt" ]]; then
        _TITLE="$(grep -m1 '^## ' "${EP_DIR}/story.txt" | sed 's/^## //')"
    fi

    if [[ -n "$_TITLE" ]]; then
        THUMB_RAW="$THUMB_RAW" THUMB="$THUMB" STORY_TITLE="$_TITLE" \
        python3 << 'PYEOF'
import os, sys
try:
    from PIL import Image, ImageDraw, ImageFont
    thumb_raw = os.environ['THUMB_RAW']
    thumb_out = os.environ['THUMB']
    title     = os.environ['STORY_TITLE']
    NOTO_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    BOLD_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    W, H = 1280, 720; FS = 72; LP = 16; BP = 24
    tf = BOLD_FONT if os.path.exists(BOLD_FONT) else NOTO_FONT
    try:
        pf = ImageFont.truetype(tf, FS)
    except Exception:
        pf = ImageFont.load_default()
    def split_t(t):
        t = t.strip()
        if len(t) <= 14: return [t]
        mid = len(t) // 2; best = -1
        for d in range(len(t)):
            for idx in [mid - d, mid + d]:
                if 0 < idx < len(t) and t[idx] in '，,。！？ ：:':
                    best = idx; break
            if best != -1: break
        cut = (best + 1) if best != -1 else mid
        return [t[:cut].strip(), t[cut:].strip()]
    lns  = split_t(title)
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
    img.save(thumb_out, 'JPEG', quality=92)
    if os.path.exists(thumb_raw):
        os.unlink(thumb_raw)
    print("  ✓ thumbnail.jpg (title burned)")
except Exception as e:
    print(f"  [warn] title burn failed: {e} — falling back to raw frame", file=sys.stderr)
    import subprocess
    subprocess.run(
        ['ffmpeg', '-y', '-i', os.environ['THUMB_RAW'], os.environ['THUMB'], '-loglevel', 'error'],
        capture_output=True
    )
    raw = os.environ['THUMB_RAW']
    if os.path.exists(raw):
        os.unlink(raw)
    print("  ✓ thumbnail.jpg (raw frame fallback)")
PYEOF
    else
        # No title — convert raw PNG to JPG
        if ffmpeg -y -i "$THUMB_RAW" -q:v 2 "$THUMB" -loglevel error 2>/dev/null; then
            rm -f "$THUMB_RAW"
            echo "  ✓ thumbnail.jpg (raw frame, no title)"
        else
            mv "$THUMB_RAW" "$THUMB" 2>/dev/null || true
            echo "  [warn] thumbnail jpg conversion failed (non-fatal)"
        fi
    fi
else
    echo "  [warn] thumbnail generation failed (non-fatal)"
fi
echo ""

# ── STEP 3: Generate youtube.json ─────────────────────────────────────────────
echo "  STEP 3 — Generate YouTube metadata"
_story_bn=""
if [[ -f "${EP_DIR}/story_source_basename" ]]; then
    _story_bn="$(cat "${EP_DIR}/story_source_basename")"
fi

_yt_result="$(python3 "${SCRIPT_DIR}/code/http/gen_youtube_json.py" \
    --slug           "${_SLUG}" \
    --ep_id          "${_EP_ID}" \
    --locale         "${LOCALE}" \
    ${_story_bn:+--story_basename "$_story_bn"} \
    2>/tmp/ingest_heygen_yt_err.txt)" || true

if [[ "$_yt_result" == "ok" ]]; then
    echo "  ✓ youtube.json generated"
else
    _yt_err="$(cat /tmp/ingest_heygen_yt_err.txt 2>/dev/null)"
    echo "  ✗ youtube.json generation failed: ${_yt_err:-unknown error}"
    echo "     Open the YouTube tab and click ✨ Generate youtube.json manually"
fi
echo ""

# ── STEP 4: Upload to YouTube (private, background) ───────────────────────────
echo "  STEP 4 — Upload to YouTube (private)"
_upload_state="${RENDERS_DIR}/upload_state.json"
_upload_log="${RENDERS_DIR}/upload_private.log"

if [[ ! -f "${RENDERS_DIR}/youtube.json" ]]; then
    echo "  [SKIP] youtube.json missing — upload skipped."
    echo "         Fix generation above, then re-run:"
    echo "         ./ingest_heygen_video.sh $VIDEO_PATH --ep_dir $EP_DIR"
else
    _already_uploaded="$(python3 -c "
import json, sys
try:
    s = json.load(open('${_upload_state}'))
    print('true' if s.get('video_uploaded') and s.get('video_id') else 'false')
except Exception:
    print('false')
" 2>/dev/null || echo 'false')"

    if [[ "$_already_uploaded" == "true" ]]; then
        _vid_id="$(python3 -c "
import json; print(json.load(open('${_upload_state}')).get('video_id',''))
" 2>/dev/null || true)"
        echo "  ✓ Already uploaded: https://www.youtube.com/watch?v=${_vid_id}"
        # Ensure production_method is tagged (idempotent — no-op if already set)
        if [[ -n "$_vid_id" ]]; then
            python3 -c "
import sqlite3
conn = sqlite3.connect('/home/tnnd/data/code/story_engine/db.sqlite3')
conn.execute(\"UPDATE youtube_publish_log SET production_method='heygen' WHERE video_id=? AND (production_method IS NULL OR production_method != 'heygen')\", ('$_vid_id',))
conn.commit()
rows = conn.execute('SELECT changes()').fetchone()[0]
conn.close()
if rows:
    print('[heygen-tagger] Tagged existing upload: video_id=$_vid_id')
" 2>/dev/null || true
        fi
    else
        nohup python3 "${SCRIPT_DIR}/code/deploy/youtube/upload_private.py" \
            "$EP_DIR" \
            --locale "$LOCALE" \
            > "$_upload_log" 2>&1 &
        _upload_pid=$!
        echo "  ▶ Upload started (PID=${_upload_pid})"
        echo "  Log : ${_upload_log}"

        # ── Background tagger: once upload_state.json has video_id, mark as heygen
        _DB_PATH="/home/tnnd/data/code/story_engine/db.sqlite3"
        _state_file="${_upload_state}"
        _tag_log="${_upload_log}"
        (
            _tagged=false
            for _i in $(seq 1 60); do
                sleep 10
                _vid="$(python3 -c "
import json, sys
try:
    s = json.load(open('${_state_file}'))
    v = s.get('video_id', '')
    if s.get('video_uploaded') and v:
        print(v)
except Exception:
    pass
" 2>/dev/null || true)"
                if [[ -n "$_vid" ]]; then
                    python3 -c "
import sqlite3
conn = sqlite3.connect('${_DB_PATH}')
conn.execute(\"UPDATE youtube_publish_log SET production_method='heygen' WHERE video_id=?\", ('$_vid',))
conn.commit()
rows = conn.execute('SELECT changes()').fetchone()[0]
conn.close()
print(f'[heygen-tagger] SET production_method=heygen for video_id=$_vid ({rows} row updated)')
" >> "${_tag_log}" 2>&1
                    _tagged=true
                    break
                fi
            done
            if [[ "$_tagged" != "true" ]]; then
                echo "[heygen-tagger] WARNING: video_id not found after 10 min — skipping DB tag" >> "${_tag_log}"
            fi
        ) &
        echo "  ▶ HeyGen tagger started (will mark DB once upload completes)"
    fi
fi
echo ""

echo "════════════════════════════════════════════════════════════"
echo "  ✓ Done"
echo "  Video   : $OUT_VIDEO"
echo "  Project : $EP_DIR"
echo "════════════════════════════════════════════════════════════"
