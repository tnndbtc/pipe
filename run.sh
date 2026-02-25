#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# run.sh — AI Narrative Pipeline Runner
# ══════════════════════════════════════════════════════════════════════
#
# Usage:
#   ./run.sh [STORY_FILE] [FROM_STAGE] [TO_STAGE]
#
# Examples:
#   ./run.sh                        # story_2.txt, stages 0–9
#   ./run.sh story_2.txt            # all stages
#   ./run.sh story_2.txt 2 4        # re-run stages 2–4 only
#   ./run.sh story_2.txt 9 9        # re-run stage 9 only (after media-agent)
#
# Each stage prompt is filled (placeholders substituted) → temp file →
# claude -p <temp>  and output is tee'd to both stdout (web UI) and
# stage_logs/stage_N.log for debugging.
# ══════════════════════════════════════════════════════════════════════

set -euo pipefail

STORY_FILE="${1:-story_2.txt}"
FROM_STAGE="${2:-0}"
TO_STAGE="${3:-9}"

# ── Model selection ────────────────────────────────────────────────────
#
# Defaults (tuned for speed vs quality):
#   haiku  — fast/cheap, used for mechanical extraction & JSON assembly
#   sonnet — balanced,   used for creative writing & complex derivation
#
# Override all stages:  MODEL=opus ./run.sh story.txt
# Override one stage:   STAGE_MODEL_3=opus ./run.sh story.txt
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
    2|3|4|5) echo "sonnet" ;;   # creative writing / complex JSON derivation
    *)       echo "haiku"  ;;   # variable extraction, diffs, assembly
  esac
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

# ── Validate inputs ────────────────────────────────────────────────────
if [[ ! -f "$STORY_FILE" ]]; then
  echo "✗ ERROR: story file not found: $STORY_FILE" >&2
  exit 1
fi
if [[ "$FROM_STAGE" -gt "$TO_STAGE" ]]; then
  echo "✗ ERROR: FROM_STAGE ($FROM_STAGE) > TO_STAGE ($TO_STAGE)" >&2
  exit 1
fi

# ── Helpers ────────────────────────────────────────────────────────────
mkdir -p stage_logs

fill_and_run() {
  local N="$1"
  local prompt_src="prompts/p_${N}.txt"
  local log_file="stage_logs/stage_${N}.log"
  local model
  model=$(get_stage_model "$N")
  local tmp tmp2
  tmp=$(mktemp  /tmp/pipe_stage_${N}_XXXXXX.txt)
  tmp2=$(mktemp /tmp/pipe_stage_${N}_inlined_XXXXXX.txt)

  echo ""
  echo "══════════════════════════════════════════════════════════════"
  echo "  STAGE ${N}  →  ${prompt_src}  [model: ${model}]"
  echo "══════════════════════════════════════════════════════════════"

  # 1. Substitute all {{PLACEHOLDER}} tokens
  sed \
    -e "s|{{STORY_FILE}}|${STORY_FILE}|g" \
    -e "s|{{PROJECT_SLUG}}|${PROJECT_SLUG:-}|g" \
    -e "s|{{EPISODE_ID}}|${EPISODE_ID:-}|g" \
    -e "s|{{EPISODE_NUMBER}}|${EPISODE_NUMBER:-}|g" \
    -e "s|{{STORY_TITLE}}|${STORY_TITLE:-}|g" \
    -e "s|{{SERIES_GENRE}}|${SERIES_GENRE:-}|g" \
    -e "s|{{GENERATION_SEED}}|${GENERATION_SEED:-}|g" \
    -e "s|{{RENDER_PROFILE}}|${RENDER_PROFILE:-preview_local}|g" \
    -e "s|{{LOCALES}}|${LOCALES:-en}|g" \
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
echo "  [0]  Read story file & populate variables"
echo "  [1]  Canon check"
echo "  [2]  Produce StoryPrompt.json"
echo "  [3]  Produce Script.json"
echo "  [4]  Produce ShotList.json"
echo "  [5]  Produce AssetManifest_draft.json"
echo "  [6]  Write canon_diff.json"
echo "  [7]  Update canon.json"
echo "  [8]  Produce locale variants  (one pass per non-en locale)"
echo "  [9]  AssetManifest_final.json + RenderPlan.json  ← conditional"
echo "══════════════════════════════════════════════════════════════"
echo "  Input story  : $STORY_FILE"
echo "  Running stages $FROM_STAGE → $TO_STAGE"
echo "══════════════════════════════════════════════════════════════"

# ── Stage 0: read story, write pipeline_vars.sh ────────────────────────
if [[ "$FROM_STAGE" -le 0 && "$TO_STAGE" -ge 0 ]]; then
  export STORY_FILE="$STORY_FILE"
  fill_and_run 0

  if [[ ! -f pipeline_vars.sh ]]; then
    echo "✗ ERROR: Stage 0 did not produce pipeline_vars.sh" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source pipeline_vars.sh
  echo "  Loaded vars: PROJECT_SLUG=$PROJECT_SLUG  EPISODE_ID=$EPISODE_ID"
fi

# ── If starting from stage > 0, load vars from prior run ──────────────
if [[ "$FROM_STAGE" -gt 0 ]]; then
  if [[ ! -f pipeline_vars.sh ]]; then
    echo "✗ ERROR: pipeline_vars.sh not found." >&2
    echo "  Run stage 0 first:  ./run.sh $STORY_FILE 0 0" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source pipeline_vars.sh
  echo "  Loaded vars from pipeline_vars.sh"
  echo "  PROJECT_SLUG=$PROJECT_SLUG  EPISODE_ID=$EPISODE_ID"
fi

# ── Stages 1–9 ────────────────────────────────────────────────────────
for N in 1 2 3 4 5 6 7 8 9; do
  if [[ "$N" -ge "$FROM_STAGE" && "$N" -le "$TO_STAGE" ]]; then
    fill_and_run "$N"
  fi
done

# ── Final summary ──────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  PIPELINE COMPLETE  (stages $FROM_STAGE → $TO_STAGE)"
echo "  Episode dir: projects/${PROJECT_SLUG:-?}/episodes/${EPISODE_ID:-?}/"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "Stage logs:"
for N in $(seq "$FROM_STAGE" "$TO_STAGE"); do
  log="stage_logs/stage_${N}.log"
  if [[ -f "$log" ]]; then
    size=$(wc -c < "$log")
    echo "  stage_${N}.log  →  ${size} bytes"
  fi
done
