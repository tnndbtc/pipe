#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# run.sh — AI Narrative Pipeline Runner
# ══════════════════════════════════════════════════════════════════════
#
# Usage:
#   ./run.sh <ep_dir> [from_stage] [to_stage]
#
# Arguments:
#   ep_dir      — path to episode directory
#                 e.g.  projects/my-project/episodes/s01e01
#   from_stage  — first stage to run  (default: 0)
#   to_stage    — last  stage to run  (default: 10)
#
# Examples:
#   ./run.sh projects/my-project/episodes/s01e01          # stages 0–10
#   ./run.sh projects/my-project/episodes/s01e01 2 4      # re-run stages 2–4
#   ./run.sh projects/my-project/episodes/s01e01 9 9      # re-run stage 9 only
#
# NOTE: The old ./run.sh story_N.txt interface is gone. New interface only.
#
# Prerequisites (handled by Prepare before this script runs):
#   • ep_dir/story.txt              — the episode story file
#   • ep_dir/pipeline_vars.sh      — stub (or full) vars file
#
# Stage 0 overwrites ep_dir/pipeline_vars.sh adding VOICE_CAST_FILE and
# other derived vars; run.sh re-sources it after Stage 0 completes.
#
# Each stage prompt is filled (placeholders substituted) → temp file →
# claude -p <temp>  and output is tee'd to both stdout (web UI) and
# stage_logs/<slug>.<ep_id>.stage_N.log for debugging.
# ══════════════════════════════════════════════════════════════════════

set -euo pipefail

EP_DIR="${1:-}"
FROM_STAGE="${2:-0}"
TO_STAGE="${3:-10}"

# ── Validate inputs ────────────────────────────────────────────────────
if [[ -z "$EP_DIR" ]]; then
  echo "✗ ERROR: ep_dir argument is required" >&2
  echo "  Usage: ./run.sh <ep_dir> [from_stage] [to_stage]" >&2
  exit 1
fi
if [[ ! -d "$EP_DIR" ]]; then
  echo "✗ ERROR: ep_dir is not a directory: $EP_DIR" >&2
  exit 1
fi

STORY_FILE="${EP_DIR}/story.txt"
VARS_FILE="${EP_DIR}/pipeline_vars.sh"

if [[ ! -f "$STORY_FILE" ]]; then
  echo "✗ ERROR: story file not found: $STORY_FILE" >&2
  exit 1
fi
if [[ "$FROM_STAGE" -gt "$TO_STAGE" ]]; then
  echo "✗ ERROR: FROM_STAGE ($FROM_STAGE) > TO_STAGE ($TO_STAGE)" >&2
  exit 1
fi
if [[ ! -f "$VARS_FILE" ]]; then
  echo "✗ ERROR: $VARS_FILE not found." >&2
  echo "  Use the ✦ Create Episode button in the web UI to create the episode first." >&2
  exit 1
fi

export STORY_FILE

# ── Model selection ────────────────────────────────────────────────────
#
# Defaults (tuned for speed vs quality):
#   haiku  — fast/cheap, used for mechanical extraction & JSON assembly
#   sonnet — balanced,   used for creative writing & complex derivation
#
# Override all stages:  MODEL=opus ./run.sh ep_dir
# Override one stage:   STAGE_MODEL_3=opus ./run.sh ep_dir
#
get_stage_model() {
  local n="$1"
  # Global override
  if [[ -n "${MODEL:-}" ]]; then echo "$MODEL"; return; fi
  # Per-stage override  (e.g.  STAGE_MODEL_3=opus)
  local var="STAGE_MODEL_${n}"
  local val="${!var:-}"
  if [[ -n "$val" ]]; then echo "$val"; return; fi
  # Defaults
  case "$n" in
    2|3|4|5|8) echo "sonnet" ;;   # creative writing / complex JSON derivation
    *)          echo "haiku"  ;;   # variable extraction, diffs, assembly
  esac
}

# ── Stage human-readable labels ────────────────────────────────────────
stage_label() {
  case "$1" in
    0) echo "Extract story variables & set up project" ;;
    1) echo "Check story & world consistency" ;;
    2) echo "Write episode direction (StoryPrompt)" ;;
    3) echo "Write script & character dialogue" ;;
    4) echo "Break script into visual shots (ShotList)" ;;
    5) echo "List required assets — images, voice, music" ;;
    6) echo "Identify new story facts to record" ;;
    7) echo "Update story memory (world canon)" ;;
    8) echo "Translate & adapt for each language" ;;
    9) echo "Finalize assets & build render plan" ;;
   10) echo "Merge assets, render video & export dubbed audio" ;;
    *) echo "Stage $1" ;;
  esac
}

# ── Stage 10: asset merge + render (no LLM) ────────────────────────────
run_stage_10() {
  local label
  label=$(stage_label "10")
  echo ""
  echo "══════════════════════════════════════════════════════════════"
  echo "  STAGE 10/10  —  ${label}"
  echo "  locales: ${LOCALES:-en}"
  echo "══════════════════════════════════════════════════════════════"

  local code_dir
  code_dir="$(cd "$(dirname "$0")" && pwd)/code/http"

  # ── [1/8] Music clips — locale-free, runs once, skips if no resources ─
  echo "  [1/8] Generating music clips (skips gracefully if no resources)…"
  python3 "${code_dir}/gen_music_clip.py" \
    --manifest "${EP_DIR}/AssetManifest_draft.shared.json"

  # ── Per-locale steps ──────────────────────────────────────────────────
  # Parse comma-separated locales (e.g. "en, zh-Hans")
  IFS=',' read -ra _locale_arr <<< "${LOCALES:-en}"
  for _raw in "${_locale_arr[@]}"; do
    local locale
    locale="$(echo "$_raw" | tr -d ' ')"
    [[ -z "$locale" ]] && continue

    echo ""
    echo "  ── Locale: ${locale} ───────────────────────────────────────────"

    echo "  [2/8] Merging shared + locale manifests…"
    python3 "${code_dir}/manifest_merge.py" \
      --shared "${EP_DIR}/AssetManifest_draft.shared.json" \
      --locale "${EP_DIR}/AssetManifest_draft.${locale}.json" \
      --out    "${EP_DIR}/AssetManifest_merged.${locale}.json"

    echo "  [3/8] Generating voice-over audio…"
    python3 "${code_dir}/gen_tts_cloud.py" \
      --manifest "${EP_DIR}/AssetManifest_merged.${locale}.json"

    echo "  [4/8] Analysing voice timing…"
    python3 "${code_dir}/post_tts_analysis.py" \
      --manifest "${EP_DIR}/AssetManifest_merged.${locale}.json"

    echo "  [5/8] Resolving asset file paths…"
    python3 "${code_dir}/resolve_assets.py" \
      --manifest "${EP_DIR}/AssetManifest_merged.${locale}.json" \
      --out      "${EP_DIR}/AssetManifest.media.${locale}.json"

    echo "  [6/8] Building per-shot render plan…"
    python3 "${code_dir}/gen_render_plan.py" \
      --manifest      "${EP_DIR}/AssetManifest_merged.${locale}.json" \
      --media         "${EP_DIR}/AssetManifest.media.${locale}.json" \
      --story-format  "${STORY_FORMAT:-episodic}"

    echo "  [7/8] Rendering video…"
    python3 "${code_dir}/render_video.py" \
      --plan    "${EP_DIR}/RenderPlan.${locale}.json" \
      --locale  "${locale}" \
      --out     "${EP_DIR}/renders/${locale}" \
      --profile "${RENDER_PROFILE:-preview_local}" \
      ${NO_MUSIC:+--no-music}

    echo "  ✓ ${locale}  →  ${EP_DIR}/renders/${locale}/output.mp4"

    echo "  [8/8] Exporting YouTube dubbed audio…"
    if [[ "$locale" != "en" ]]; then
      python3 "${code_dir}/export_youtube_dubbed.py" \
        "${EP_DIR}" \
        "${locale}"
      echo "  ✓ ${locale}  →  ${EP_DIR}/renders/${locale}/youtube_dubbed.m4a"
    else
      echo "  ↷ ${locale}  English is the primary upload — dubbed audio export skipped"
    fi
  done

  echo ""
  echo "✓ Stage 10 complete"
}

# ── File inlining ──────────────────────────────────────────────────────
#
# Pre-embeds referenced input files into the filled prompt so Claude
# never needs to call the Read tool for them.  Each eliminated Read
# call saves ~10-20 s of API round-trip time.
#
# Detects lines of the form:
#   "  N. some/path.json  (optional comment)"
# and appends matching files that exist on disk.
#
inline_files_into_prompt() {
  local src="$1"
  local dst="$2"

  local files_tmp
  files_tmp=$(mktemp /tmp/pipe_files_XXXXXX.txt)
  local found=0

  while IFS= read -r line; do
    if [[ "$line" =~ ^[[:space:]]+[0-9]+\.[[:space:]]+([^[:space:]]+\.(json|txt|sh))[[:space:]]* ]]; then
      local fp="${BASH_REMATCH[1]}"
      if [[ -f "$fp" ]]; then
        found=1
        printf '### `%s`\n```\n' "$fp" >> "$files_tmp"
        cat "$fp"                       >> "$files_tmp"
        printf '\n```\n\n'             >> "$files_tmp"
      fi
    fi
  done < "$src"

  if [[ "$found" -eq 0 ]]; then
    cp "$src" "$dst"
    rm -f "$files_tmp"
    return
  fi

  {
    printf 'NOTE: All input files listed under "Read these files before writing anything"\n'
    printf 'are pre-embedded at the end of this prompt (## Pre-loaded file contents).\n'
    printf 'Do NOT call the Read tool for those paths — use the embedded content directly.\n\n'
    cat "$src"
    printf '\n---\n\n## Pre-loaded file contents\n\n'
    printf 'These files are already embedded here. Do NOT use the Read tool for them.\n\n'
    cat "$files_tmp"
  } > "$dst"

  rm -f "$files_tmp"
}

# ── Helpers ────────────────────────────────────────────────────────────
mkdir -p stage_logs

fill_and_run() {
  local N="$1"
  local prompt_src="prompts/p_${N}.txt"
  # Per-project log so concurrent / sequential runs on different projects don't
  # overwrite each other.  Falls back to generic name before Stage 0 sets PROJECT_SLUG.
  local _log_slug="${PROJECT_SLUG:-unknown}"
  local _log_ep="${EPISODE_ID:-unknown}"
  local log_file="stage_logs/${_log_slug}.${_log_ep}.stage_${N}.log"
  local model
  model=$(get_stage_model "$N")
  local tmp tmp2
  tmp=$(mktemp  /tmp/pipe_stage_${N}_XXXXXX.txt)
  tmp2=$(mktemp /tmp/pipe_stage_${N}_inlined_XXXXXX.txt)

  local label
  label=$(stage_label "$N")
  echo ""
  echo "══════════════════════════════════════════════════════════════"
  echo "  STAGE ${N}/10  —  ${label}"
  echo "  model: ${model}"
  echo "══════════════════════════════════════════════════════════════"

  # 1. Substitute all {{PLACEHOLDER}} tokens
  sed \
    -e "s|{{EPISODE_DIR}}|${EP_DIR}|g" \
    -e "s|{{STORY_FILE}}|${STORY_FILE}|g" \
    -e "s|{{PROJECT_SLUG}}|${PROJECT_SLUG:-}|g" \
    -e "s|{{EPISODE_ID}}|${EPISODE_ID:-}|g" \
    -e "s|{{EPISODE_NUMBER}}|${EPISODE_NUMBER:-}|g" \
    -e "s|{{STORY_TITLE}}|${STORY_TITLE:-}|g" \
    -e "s|{{SERIES_GENRE}}|${SERIES_GENRE:-}|g" \
    -e "s|{{GENERATION_SEED}}|${GENERATION_SEED:-}|g" \
    -e "s|{{RENDER_PROFILE}}|${RENDER_PROFILE:-preview_local}|g" \
    -e "s|{{LOCALES}}|${LOCALES:-en}|g" \
    -e "s|{{STORY_FORMAT}}|${STORY_FORMAT:-episodic}|g" \
    "$prompt_src" > "$tmp"

  # 2. Pre-embed referenced input files → eliminate Read tool calls
  inline_files_into_prompt "$tmp" "$tmp2"
  rm -f "$tmp"

  # 3. Run claude with speed-optimised flags
  #    --append-system-prompt forces immediate execution with no confirmation prompts
  local exec_directive="You are an automated batch pipeline stage running with no human operator present. Execute the given task IMMEDIATELY and COMPLETELY. NEVER ask for confirmation, permission, or clarification. NEVER describe what you are about to do. NEVER offer choices or options. Complete every instruction from start to finish and then stop."
  if claude -p \
       --model                        "$model" \
       --dangerously-skip-permissions         \
       --no-session-persistence               \
       --append-system-prompt         "$exec_directive" \
       "$tmp2" | tee "$log_file"; then
    echo ""
    echo "✓ Stage ${N} complete  →  log: ${log_file}"
  else
    local exit_code=${PIPESTATUS[0]}
    echo ""
    echo "✗ Stage ${N} FAILED (exit code ${exit_code})" >&2
    echo "  Full output: ${log_file}" >&2
    rm -f "$tmp2"
    exit "$exit_code"
  fi

  rm -f "$tmp2"
}

# ── Print pipeline plan ────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  PIPELINE PLAN  —  10 stages  (step 9 is conditional)"
echo "══════════════════════════════════════════════════════════════"
echo "  [0]  Extract story variables & set up project"
echo "  [1]  Check story & world consistency"
echo "  [2]  Write episode direction (StoryPrompt)"
echo "  [3]  Write script & character dialogue"
echo "  [4]  Break script into visual shots (ShotList)"
echo "  [5]  List required assets — images, voice, music"
echo "  [6]  Identify new story facts to record"
echo "  [7]  Update story memory (world canon)"
echo "  [8]  Translate & adapt for each language"
echo "  [9]  Finalize assets & build render plan  ← conditional"
echo "  [10] Merge assets, render video & export dubbed audio"
echo "══════════════════════════════════════════════════════════════"
echo "  Episode dir : $EP_DIR"
echo "  Story file  : $STORY_FILE"
echo "  Running stages $FROM_STAGE → $TO_STAGE"
echo "══════════════════════════════════════════════════════════════"

# ── Source vars from pipeline_vars.sh (stub written by Prepare, full after Stage 0) ──
# shellcheck disable=SC1091
source "$VARS_FILE"
echo "  Loaded vars from $VARS_FILE"
echo "  PROJECT_SLUG=$PROJECT_SLUG  EPISODE_ID=$EPISODE_ID"

# ── Stage 0: read story, write pipeline_vars.sh ────────────────────────
if [[ "$FROM_STAGE" -le 0 && "$TO_STAGE" -ge 0 ]]; then
  fill_and_run 0

  if [[ ! -f "$VARS_FILE" ]]; then
    echo "✗ ERROR: Stage 0 did not produce ${VARS_FILE}" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$VARS_FILE"
  echo "  Loaded vars: PROJECT_SLUG=$PROJECT_SLUG  EPISODE_ID=$EPISODE_ID"
  echo "  Vars file  : $VARS_FILE"
fi

# ── Stages 1–9 (LLM) ─────────────────────────────────────────────────
for N in 1 2 3 4 5 6 7 8 9; do
  if [[ "$N" -ge "$FROM_STAGE" && "$N" -le "$TO_STAGE" ]]; then
    fill_and_run "$N"
  fi
done

# ── Stage 10: merge assets & render video ─────────────────────────────
if [[ "$FROM_STAGE" -le 10 && "$TO_STAGE" -ge 10 ]]; then
  run_stage_10
fi

# ── Final summary ──────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  PIPELINE COMPLETE  (stages $FROM_STAGE → $TO_STAGE)"
echo "  Episode dir: ${EP_DIR}/"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "Stage logs:"
for N in $(seq "$FROM_STAGE" "$TO_STAGE"); do
  log="stage_logs/${PROJECT_SLUG:-unknown}.${EPISODE_ID:-unknown}.stage_${N}.log"
  if [[ -f "$log" ]]; then
    size=$(wc -c < "$log")
    echo "  $(basename "$log")  →  ${size} bytes"
  fi
done
