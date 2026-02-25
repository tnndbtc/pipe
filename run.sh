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
  local tmp
  tmp=$(mktemp /tmp/pipe_stage_${N}_XXXXXX.txt)

  echo ""
  echo "══════════════════════════════════════════════════════════════"
  echo "  STAGE ${N}  →  ${prompt_src}"
  echo "══════════════════════════════════════════════════════════════"

  # Substitute all {{PLACEHOLDER}} tokens with current env vars
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

  # Run claude and tee output to both stdout and log
  if claude -p "$tmp" | tee "$log_file"; then
    echo ""
    echo "✓ Stage ${N} complete  →  log: ${log_file}"
  else
    local exit_code=${PIPESTATUS[0]}
    echo ""
    echo "✗ Stage ${N} FAILED (exit code ${exit_code})" >&2
    echo "  Full output: ${log_file}" >&2
    rm -f "$tmp"
    exit "$exit_code"
  fi

  rm -f "$tmp"
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
