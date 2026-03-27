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
#   to_stage    — last  stage to run  (default: 10, which maps to Stage 9 render)
#
# Examples:
#   ./run.sh projects/my-project/episodes/s01e01          # stages 0–9 (full run)
#   ./run.sh projects/my-project/episodes/s01e01 2 4      # re-run stages 2–4
#   ./run.sh projects/my-project/episodes/s01e01 10 10    # re-run Stage 9 render only
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

# Strip _c / _a / _b suffix for integer comparisons and seq (e.g. "5_c" → 5)
FROM_BASE="${FROM_STAGE%%_*}"
TO_BASE="${TO_STAGE%%_*}"
FROM_SUFFIX="${FROM_STAGE#"$FROM_BASE"}"   # "_c" or ""

# Support "3.5" as FROM_STAGE (Stage 3.5 = VO preview, before Stage 4)
# Treat it as FROM_BASE=4 for the integer stage loop; Stage 3.5 gate in N=4 block handles it.
_RUN_35_EXPLICIT=0
if [[ "$FROM_STAGE" == "3.5" ]]; then
  _RUN_35_EXPLICIT=1
  FROM_BASE=4
  FROM_STAGE=4
  FROM_SUFFIX=""
fi

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

_fmt_pre=$(python3 -c \
  "import json,sys; d=json.load(open('${EP_DIR}/meta.json')); \
   print(d.get('story_format',''))" 2>/dev/null || echo "")
if [[ "$_fmt_pre" != "Script.json" && ! -f "$STORY_FILE" ]]; then
  echo "✗ ERROR: story file not found: $STORY_FILE" >&2
  exit 1
fi
if [[ "$FROM_BASE" -gt "$TO_BASE" ]]; then
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
  local n_base="${n%%_*}"   # strip _c suffix: "5_c" → "5", "4" → "4"
  # Global override
  if [[ -n "${MODEL:-}" ]]; then echo "$MODEL"; return; fi
  # Per-stage override  (e.g.  STAGE_MODEL_3=opus)
  local var="STAGE_MODEL_${n}"
  local val="${!var:-}"
  if [[ -n "$val" ]]; then echo "$val"; return; fi
  # Defaults (match on base stage number so "5_c" → sonnet, "6_c" → haiku)
  case "$n_base" in
    2|3|4|5|8) echo "sonnet" ;;   # creative writing / complex JSON derivation
    *)          echo "haiku"  ;;   # variable extraction, diffs, assembly
  esac
}

# ── Stage human-readable labels ────────────────────────────────────────
stage_label() {
  local n_base="${1%%_*}"    # strip _c suffix: "5_c" → "5"
  local n_suffix="${1#"$n_base"}"  # "" or "_c"
  local _label
  case "$n_base" in
    0)   _label="Extract story variables & set up project" ;;
    1)   _label="Check story & world consistency" ;;
    2)   _label="Write episode direction (StoryPrompt)" ;;
    3)   _label="Write script & character dialogue" ;;
    3.5) _label="VO preview — synthesise & review before ShotList" ;;
    4)   _label="Break script into visual shots (ShotList)" ;;
    5)   _label="List required assets — images, voice, music" ;;
    6)   _label="Identify new story facts to record" ;;
    7)   _label="Update story memory (world canon)" ;;
    8)   _label="Translate & adapt for each language" ;;
    8.5) _label="Non-primary locale VO review" ;;
    9)   _label="Merge assets, render video & export dubbed audio" ;;
    *)   _label="Stage $n_base" ;;
  esac
  if [[ "$n_suffix" == "_c" ]]; then
    _label="${_label} (creative fill)"
  fi
  echo "$_label"
}

# ── Stage 9: asset merge + render (no LLM) ─────────────────────────────
run_stage_10() {
  local label
  label=$(stage_label "9")
  echo ""
  echo "══════════════════════════════════════════════════════════════"
  echo "  STAGE 9  —  ${label}"
  echo "  locales: ${LOCALES:-en}"
  echo "══════════════════════════════════════════════════════════════"

  local code_dir
  code_dir="$(cd "$(dirname "$0")" && pwd)/code/http"

  # ── [1] Music clips — skip if music already approved ─────────────────
  _music_approved="${EP_DIR}/MusicPlan.json"
  if [[ -f "$_music_approved" ]]; then
    echo "  [1] Music already approved — skipping gen_music_clip."
    echo "      (Delete MusicPlan.json to force re-run)"
  else
    echo "  [1] Generating music clips (skips gracefully if no resources)…"
    python3 "${code_dir}/gen_music_clip.py" \
      --manifest "${EP_DIR}/AssetManifest.shared.json" || true
  fi

  # ── [1b] Music loop candidates — skip if music already approved ───────
  if [[ -f "$_music_approved" ]]; then
    echo "  [1b] Music already approved — skipping music_prepare_loops."
  else
    echo "  [1b] Analysing music loop candidates…"
    python3 "${code_dir}/music_prepare_loops.py" \
      --manifest "${EP_DIR}/AssetManifest.shared.json" || true
  fi

  # ── Per-locale steps ──────────────────────────────────────────────────
  # Parse comma-separated locales (e.g. "en, zh-Hans").
  # Reorder so PRIMARY_LOCALE is processed first — its RenderPlan becomes
  # the timing reference for all other locales.
  local _primary="${PRIMARY_LOCALE:-en}"
  IFS=',' read -ra _locale_raw <<< "${LOCALES:-en}"
  _locale_arr=()
  # First pass: add primary locale
  for _raw in "${_locale_raw[@]}"; do
    local _l; _l="$(echo "$_raw" | tr -d ' ')"
    [[ "$_l" == "$_primary" ]] && _locale_arr+=("$_l")
  done
  # Second pass: add remaining locales
  for _raw in "${_locale_raw[@]}"; do
    local _l; _l="$(echo "$_raw" | tr -d ' ')"
    [[ "$_l" != "$_primary" ]] && _locale_arr+=("$_l")
  done
  # Fallback: if primary wasn't in LOCALES, prepend it
  [[ ${#_locale_arr[@]} -eq 0 ]] && _locale_arr=("$_primary")

  for _raw in "${_locale_arr[@]}"; do
    local locale
    locale="$(echo "$_raw" | tr -d ' ')"
    [[ -z "$locale" ]] && continue

    echo ""
    echo "  ── Locale: ${locale} ───────────────────────────────────────────"

    # [5] manifest_merge.py eliminated — gen_tts_cloud.py writes VOPlan.{locale}.json
    #     directly from Script.json + VoiceCast.json. No merge step required.

    echo "  [5.5] Validating tts_prompt fields match VoiceCast.json…"
    python3 "${code_dir}/validate_tts_prompts.py" \
      "${EP_DIR}" --locale "${locale}" --warn-only || true

    # ── [6] TTS synthesis — skip if VO already approved ─────────────────
    # If VOPlan.{locale}.json has a vo_approval block the audio is final.
    if python3 "${code_dir}/check_vo_approved.py" "${EP_DIR}" "${locale}"; then
      echo "  [6] VO already approved — skipping TTS synthesis."
      echo "      (Remove vo_approval from VOPlan.${locale}.json to force re-run)"
    else
      echo "  [6] Generating voice-over audio…"
      if [[ "${STORY_FORMAT:-}" == "ssml_narration" && "$locale" == "$_primary" ]]; then
        # PRIMARY_LOCALE uses ssml_narration: wrapper-rebuild + inner passthrough
        python3 "${code_dir}/gen_tts_cloud.py" \
          --manifest "${EP_DIR}/VOPlan.${locale}.json" \
          --ssml-narration \
          --ssml-inner "${EP_DIR}/ssml_inner.xml" \
          --voice-cast "projects/${PROJECT_SLUG}/VoiceCast.json"
      else
        # Other locales (or non-ssml_narration): regular per-item TTS
        python3 "${code_dir}/gen_tts_cloud.py" \
          --manifest "${EP_DIR}/VOPlan.${locale}.json"
      fi
    fi

    # ── [6b] Phase 1 — convergence loop (non-primary locales only) ───────
    # Skip if VO already approved (same guard as [6] — no point re-aligning
    # if TTS was not re-synthesised).
    # Read alignment thresholds from their single source of truth.
    local vo_thresh vo_thresh_high
    vo_thresh=$(python3 -c "import sys; sys.path.insert(0,'${code_dir}'); from polish_locale_vo import THRESHOLD; print(f'{THRESHOLD:.2f}')" 2>/dev/null || echo "0.90")
    vo_thresh_high=$(python3 -c "import sys; sys.path.insert(0,'${code_dir}'); from polish_locale_vo import THRESHOLD_HIGH; print(f'{THRESHOLD_HIGH:.2f}')" 2>/dev/null || echo "1.10")
    # Measures locale/primary WAV duration ratios; rewrites lines outside
    # [${vo_thresh}, ${vo_thresh_high}] via Claude sonnet; re-synthesizes;
    # repeats up to 3 times.
    # Writes calibration data to prompts/tts_calibration.{locale}.json.
    if [[ "$locale" != "$_primary" ]]; then
      if python3 "${code_dir}/check_vo_approved.py" "${EP_DIR}" "${locale}"; then
        echo "  [6b] VO already approved — skipping alignment loop."
      else
        echo "  [6b] Polishing locale VO duration alignment…"
        python3 "${code_dir}/polish_locale_vo.py" \
          --manifest        "${EP_DIR}/VOPlan.${locale}.json" \
          --locale          "${locale}" \
          --ep-dir          "${EP_DIR}" \
          --primary-locale  "${_primary}" || true
      fi
    fi

    echo "  [7] Analysing voice timing…"
    python3 "${code_dir}/post_tts_analysis.py" \
      --manifest "${EP_DIR}/VOPlan.${locale}.json"

    # ── Stage 8.5: Non-primary locale VO approval gate ──────────────────
    # After TTS + convergence loop + timing analysis, pause for user VO review
    # of each non-primary locale before generating the render plan.
    if [[ "$locale" != "$_primary" ]]; then
      if ! python3 "${code_dir}/check_vo_approved.py" "${EP_DIR}" "${locale}"; then
        echo ""
        echo "══════════════════════════════════════════════════════════════"
        echo "  ⏸  PAUSED — Stage 8.5: VO review required for locale: ${locale}"
        echo ""
        echo "  Translated voice-over for ${locale} is ready:"
        echo "    1. Switch to the 🎙 VO tab → select locale: ${locale}"
        echo "    2. Review translated lines and adjust timing as needed"
        echo "    3. Click ✓ Approve VO to write vo_approval into:"
        echo "       VOPlan.${locale}.json"
        echo ""
        echo "  Then resume Stage 9 for all remaining locales:"
        echo "    ./run.sh ${EP_DIR} 10"
        echo "══════════════════════════════════════════════════════════════"
        exit 0
      else
        echo "  ✓ Stage 8.5 approved for locale: ${locale}"
      fi
    fi
    # ── end Stage 8.5 gate ───────────────────────────────────────────────

    # ── [8] Music review checkpoint (only when Music is enabled) ──────────
    _music_plan="${EP_DIR}/MusicPlan.json"
    if [[ -z "${NO_MUSIC:-}" ]]; then
      if [[ ! -f "$_music_plan" ]]; then
        echo ""
        echo "══════════════════════════════════════════════════════════════"
        echo "  ⏸  PAUSED — Music review required"
        echo ""
        echo "  Open the VC editor to generate and review the music preview:"
        echo "    1. Go to the 🎵 Music tab"
        echo "    2. Click 'Generate Preview' to build the preview audio"
        echo "    3. Review the timeline and adjust loop selections / shot overrides"
        echo "    4. Click ✔ Confirm to save MusicPlan.json"
        echo ""
        echo "  Then resume the pipeline:"
        echo "    ./run.sh ${EP_DIR} 9"
        echo "══════════════════════════════════════════════════════════════"
        exit 0
      elif [[ -f "$_music_approved" ]]; then
        echo "  [8] Music already approved — skipping apply_music_plan."
        echo "      (render_video reads MusicPlan.json directly)"
      else
        echo "  [8] Applying music plan overrides…"
        python3 "${code_dir}/apply_music_plan.py" \
          --plan "$_music_plan" \
          --manifest "${EP_DIR}/VOPlan.${locale}.json"
      fi
    else
      echo "  [8] Music disabled — skipping music review checkpoint"
    fi

    echo "  [9] Resolving asset file paths…"
    # Pass --selections when the VC editor has saved stock media choices.
    # resolve_assets.py also auto-detects this file, but being explicit here
    # makes the intent clear in the log output.
    _sel_arg=""
    if [[ -f "${EP_DIR}/MediaPlan.json" ]]; then
      _sel_arg="--selections ${EP_DIR}/MediaPlan.json"
      echo "         (using stock media selections from VC editor)"
    fi
    python3 "${code_dir}/resolve_assets.py" \
      --manifest "${EP_DIR}/VOPlan.${locale}.json" \
      ${_sel_arg}

    # [10] gen_render_plan.py eliminated — render_video.py absorbs duration_ms
    #      logic and reads VOPlan + ShotList + MusicPlan + SfxPlan + MediaPlan directly.

    echo "  [11] Rendering video…"
    # Phase 2 — Timeline Lock: floor locale shot durations to the primary locale's
    # approved VO timing.  The primary locale is rendered first (loop reorder above),
    # so its VOPlan.{primary}.json is the authority passed via --reference-approval.
    local _ref_approval_arg=""
    local _ref_vp="${EP_DIR}/VOPlan.${_primary}.json"
    if [[ "$locale" != "$_primary" && -f "$_ref_vp" ]]; then
      _ref_approval_arg="--reference-approval ${_ref_vp}"
    fi
    # shellcheck disable=SC2086
    python3 "${code_dir}/render_video.py" \
      --plan    "${EP_DIR}/VOPlan.${locale}.json" \
      --locale  "${locale}" \
      --out     "${EP_DIR}/renders/${locale}" \
      --profile "${RENDER_PROFILE:-preview_local}" \
      ${NO_MUSIC:+--no-music} \
      ${_ref_approval_arg}

    echo "  ✓ ${locale}  →  ${EP_DIR}/renders/${locale}/output.mp4"

    echo "  [11b] Exporting YouTube dubbed audio…"
    if [[ "$locale" != "$_primary" ]]; then
      python3 "${code_dir}/export_youtube_dubbed.py" \
        "${EP_DIR}" \
        "${locale}"
      echo "  ✓ ${locale}  →  ${EP_DIR}/renders/${locale}/youtube_dubbed.m4a"
    else
      echo "  ↷ ${locale}  ${_primary} is the primary upload — dubbed audio export skipped"
    fi
  done

  # ── [11c] Cross-language SRT files (runs once after all locales) ──────
  echo "  [11c] Generating cross-language SRT files…"
  python3 "${code_dir}/gen_cross_srt.py" \
    --ep-dir  "${EP_DIR}" \
    --locales "${LOCALES:-en}" \
    --primary "${PRIMARY_LOCALE:-en}" || true

  echo ""
  echo "✓ Stage 9 complete"
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

# ── auto_refresh_assets ──────────────────────────────────────────────────
# Re-resolves assets for every locale whose VOPlan exists (i.e. Stage 3.5 ran).
# Called after Stages 4-7 so resolved_assets[] in VOPlan stays current after
# any upstream LLM stage re-runs. manifest_merge.py and gen_render_plan.py
# are fully eliminated — render_video.py reads VOPlan + ShotList directly.
auto_refresh_render_plan() {
  local _primary="${PRIMARY_LOCALE:-en}"
  IFS=',' read -ra _arfp_locales <<< "${LOCALES:-${_primary}}"
  local _did_refresh=0
  for _arfp_raw in "${_arfp_locales[@]}"; do
    local _loc
    _loc="$(echo "$_arfp_raw" | tr -d ' ')"
    [[ -z "$_loc" ]] && continue
    local _unified="${EP_DIR}/VOPlan.${_loc}.json"
    if [[ -f "$_unified" ]]; then
      echo "  [auto] Re-resolving assets into VOPlan.${_loc}.json…"
      python3 "${code_dir}/resolve_assets.py" \
        --manifest "$_unified" || true
      _did_refresh=1
    fi
  done
  if [[ "$_did_refresh" -eq 0 ]]; then
    echo "  [auto] No VOPlan found — prerequisites not yet available (run Stage 3.5 first)"
  fi
}

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
  echo "  STAGE ${N}/9  —  ${label}"
  echo "  model: ${model}"
  echo "══════════════════════════════════════════════════════════════"

  # 1. Substitute all {{PLACEHOLDER}} tokens
  # Derive human-readable name for PRIMARY_LOCALE
  local _primary_locale_name
  case "${PRIMARY_LOCALE:-en}" in
    en)       _primary_locale_name="English" ;;
    zh-Hans)  _primary_locale_name="Chinese (Simplified)" ;;
    zh-Hant)  _primary_locale_name="Chinese (Traditional)" ;;
    ja)       _primary_locale_name="Japanese" ;;
    ko)       _primary_locale_name="Korean" ;;
    es)       _primary_locale_name="Spanish" ;;
    fr)       _primary_locale_name="French" ;;
    de)       _primary_locale_name="German" ;;
    pt)       _primary_locale_name="Portuguese" ;;
    ar)       _primary_locale_name="Arabic" ;;
    hi)       _primary_locale_name="Hindi" ;;
    *)        _primary_locale_name="${PRIMARY_LOCALE:-en}" ;;
  esac

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
    -e "s|{{PRIMARY_LOCALE}}|${PRIMARY_LOCALE:-en}|g" \
    -e "s|{{PRIMARY_LOCALE_NAME}}|${_primary_locale_name}|g" \
    "$prompt_src" > "$tmp"

  # 2. Pre-embed referenced input files → eliminate Read tool calls
  inline_files_into_prompt "$tmp" "$tmp2"
  rm -f "$tmp"

  # 3. Run claude with speed-optimised flags
  #    --append-system-prompt forces immediate execution with no confirmation prompts
  local exec_directive="You are an automated batch pipeline stage running with no human operator present. Execute the given task IMMEDIATELY and COMPLETELY. NEVER ask for confirmation, permission, or clarification. NEVER describe what you are about to do. NEVER offer choices or options. Complete every instruction from start to finish and then stop."
  if CLAUDE_CODE_MAX_OUTPUT_TOKENS=100000 claude -p \
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
echo "  PIPELINE PLAN  —  9 stages  (LLM: 0–8, Render: 9)"
echo "══════════════════════════════════════════════════════════════"
echo "  [0]   Extract story variables & set up project"
echo "  [1]   Check story & world consistency"
echo "  [2]   Write episode direction (StoryPrompt)"
echo "  [3]   Write script & character dialogue"
echo "  [3.5] VO preview — synthesise & review before ShotList  ⏸ pause"
echo "  [4]   Break script into visual shots (ShotList)"
echo "  [5]   List required assets — images, voice, music"
echo "  [6]   Identify new story facts to record"
echo "  [7]   Update story memory (world canon)"
echo "  [8]   Translate & adapt for each language"
echo "  [9]   Merge assets, render video & export dubbed audio"
echo "        Sub-steps: [1] music  [5] merge  [6] tts  [6b] align  [7] post-tts"
echo "                   [8.5] locale VO review ⏸  [8] music-plan  [9] resolve"
echo "                   [10] plan  [11] render"
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

# ── PREPARE: Script.json format — validate contract & create story.txt ──
if [[ "${STORY_FORMAT:-}" == "Script.json" ]]; then
  echo ""
  echo "══════════════════════════════════════════════════════════════"
  echo "  PREPARE  —  Validate Script.json contract"
  echo "══════════════════════════════════════════════════════════════"
  # Recovery: episodes created before the create_episode fix only have
  # story.txt (which contains the Script.json content).  Restore Script.json
  # from story.txt so validation and all downstream stages can find it.
  if [[ ! -f "$EP_DIR/Script.json" && -f "$STORY_FILE" ]]; then
    cp "$STORY_FILE" "$EP_DIR/Script.json"
    echo "  ℹ  Script.json restored from story.txt"
  fi
  python3 "contracts/tools/verify_contracts.py" \
      "$EP_DIR/Script.json"
  _rc=$?
  if [[ $_rc -ne 0 ]]; then
    echo "✗ Script.json failed contract validation — pipeline aborted."
    echo "  Fix the Script.json provided by the external agent and re-run."
    exit 1
  fi
  echo "✓ Script.json contract valid"
  # Mirror Script.json → story.txt for consistency with all other formats.
  # story.txt is for human reference; every stage reads Script.json directly.
  cp "$EP_DIR/Script.json" "$STORY_FILE"
  echo "✓ story.txt created (copy of Script.json)"
fi

# NOTE: VO_DURATION_TABLE removed. ShotList duration_sec is now patched
# deterministically by patch_shotlist_durations.py after Stage 4 completes.

# ── Stage 0: read story, write pipeline_vars.sh ────────────────────────
if [[ "$FROM_BASE" -le 0 && "$TO_BASE" -ge 0 ]]; then
  _s0_code_dir="$(cd "$(dirname "$0")" && pwd)/code/http"
  _s0_fmt="${STORY_FORMAT:-}"
  if [[ "$_s0_fmt" == "ssml_narration" ]]; then
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  STAGE 0/9  —  Extract story variables & set up project"
    echo "  mode: ssml_preprocess.py (ssml_narration format)"
    echo "══════════════════════════════════════════════════════════════"
    python3 "${_s0_code_dir}/ssml_preprocess.py" "$EP_DIR"
    echo ""
    echo "✓ Stage 0 complete (ssml_preprocess.py)"
  elif [[ "$_s0_fmt" == "continuous_narration" || \
          "$_s0_fmt" == "illustrated_narration" || \
          "$_s0_fmt" == "documentary" || \
          "$_s0_fmt" == "monologue" ]]; then
    # P4: narrator-only formats — deterministic vars + voice selection
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  STAGE 0/9  —  Extract story variables & set up project"
    echo "  mode: gen_pipeline_vars.py + voice_cast_narrator.py (${_s0_fmt})"
    echo "══════════════════════════════════════════════════════════════"
    python3 "${_s0_code_dir}/gen_pipeline_vars.py" "$EP_DIR"
    python3 "${_s0_code_dir}/voice_cast_narrator.py" "$EP_DIR"
    echo ""
    echo "✓ Stage 0 complete (gen_pipeline_vars.py + voice_cast_narrator.py)"
  else
    # episodic: LLM (character identification + voice matching)
    fill_and_run 0
  fi

  if [[ ! -f "$VARS_FILE" ]]; then
    echo "✗ ERROR: Stage 0 did not produce ${VARS_FILE}" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$VARS_FILE"
  echo "  Loaded vars: PROJECT_SLUG=$PROJECT_SLUG  EPISODE_ID=$EPISODE_ID"
  echo "  Vars file  : $VARS_FILE"
fi

# ── Helper: path to deterministic Python scripts ──────────────────────
code_dir="$(cd "$(dirname "$0")" && pwd)/code/http"

# ── Stages 1–9 (LLM) ─────────────────────────────────────────────────
for N in 1 2 3 4 5 6 7 8 9; do
  if [[ "$N" -ge "$FROM_BASE" && "$N" -le "$TO_BASE" ]]; then

    # Skip stages 1,2,3,9 for ssml_narration
    # Stage 8 still runs for ssml_narration: it translates secondary-locale manifests.
    if [[ "${STORY_FORMAT:-}" == "ssml_narration" && ( "$N" -eq 1 || "$N" -eq 2 || "$N" -eq 3 || "$N" -eq 9 ) ]]; then
      echo ""
      echo "  ⏭  Stage $N skipped (ssml_narration)"
      continue
    fi

    # Skip Stage 2 for Script.json (no StoryPrompt needed — Script.json is the input)
    if [[ "${STORY_FORMAT:-}" == "Script.json" && "$N" -eq 2 ]]; then
      echo ""
      echo "  ⏭  Stage 2 skipped (Script.json — no StoryPrompt needed)"
      continue
    fi

    # ── P1+FIX3: Stage 9 skip (render_video.py reads VOPlan directly) ──────
    # Also preserves the pre-render checkpoint pause (FIX3).
    if [[ "$N" -eq 9 ]]; then
      echo ""
      echo "  ⏭  Stage 9 (p_9.txt) skipped — render_video.py reads VOPlan.{locale}.json directly"
      if [[ "$TO_STAGE" -gt 9 ]]; then
        _sel_file="${EP_DIR}/MediaPlan.json"
        echo ""
        echo "══════════════════════════════════════════════════════════════"
        echo "  ⏸  PAUSED after Stage 9 — ready to render"
        echo ""
        echo "  All LLM stages complete.  Before rendering, verify:"
        if [[ ! -f "$_sel_file" ]]; then
          echo "    ⚠  No media selections found — Stage 9 will lack stock media"
        else
          echo "    ✓  Media selections: $_sel_file"
        fi
        echo "    •  VOPlan.*         — asset manifest"
        echo "    •  VOPlan.*            — render plan"
        echo ""
        echo "  To render:"
        echo "    ./run.sh ${EP_DIR} 9"
        echo "══════════════════════════════════════════════════════════════"
        exit 0
      fi
      continue
    fi

    # ── P2: Stage 1 → canon_check.py (deterministic) ────────────────────
    if [[ "$N" -eq 1 ]]; then
      echo ""
      echo "══════════════════════════════════════════════════════════════"
      echo "  STAGE 1/9  —  Check story & world consistency"
      echo "  mode: canon_check.py (deterministic)"
      echo "══════════════════════════════════════════════════════════════"
      _log1="stage_logs/${PROJECT_SLUG:-unknown}.${EPISODE_ID:-unknown}.stage_1.log"
      python3 "${code_dir}/canon_check.py" "$EP_DIR" 2>&1 | tee "$_log1"
      echo ""
      echo "✓ Stage 1 complete  →  log: ${_log1}"
      continue
    fi

    # ── P7: Stage 3 → gen_script_narration.py for narration formats ─────
    if [[ "$N" -eq 3 ]]; then
      _fmt3="${STORY_FORMAT:-episodic}"
      if [[ "$_fmt3" == "Script.json" ]]; then
        echo ""
        echo "  ⏭  Stage 3 skipped (Script.json provided by external agent)"
        continue
      fi
      if [[ "$_fmt3" == "continuous_narration" || \
            "$_fmt3" == "illustrated_narration" || \
            "$_fmt3" == "documentary" ]]; then
        echo ""
        echo "══════════════════════════════════════════════════════════════"
        echo "  STAGE 3/9  —  Write script & character dialogue"
        echo "  mode: gen_script_narration.py (deterministic, ${_fmt3})"
        echo "══════════════════════════════════════════════════════════════"
        _log3="stage_logs/${PROJECT_SLUG:-unknown}.${EPISODE_ID:-unknown}.stage_3.log"
        python3 "${code_dir}/gen_script_narration.py" "$EP_DIR" 2>&1 | tee "$_log3"
        echo ""
        echo "✓ Stage 3 complete (gen_script_narration.py)  →  log: ${_log3}"
        continue
      fi
    fi

    # ── P3: Stage 7 → canon_merge.py (deterministic) ────────────────────
    if [[ "$N" -eq 7 ]]; then
      echo ""
      echo "══════════════════════════════════════════════════════════════"
      echo "  STAGE 7/9  —  Update story memory (world canon)"
      echo "  mode: canon_merge.py (deterministic)"
      echo "══════════════════════════════════════════════════════════════"
      _log7="stage_logs/${PROJECT_SLUG:-unknown}.${EPISODE_ID:-unknown}.stage_7.log"
      python3 "${code_dir}/canon_merge.py" "$EP_DIR" 2>&1 | tee "$_log7"
      echo ""
      echo "✓ Stage 7 complete (canon_merge.py)  →  log: ${_log7}"
      continue
    fi

    # ── P9: Stage 6 → canon_diff_chars.py + LLM narrative fill ─────────
    if [[ "$N" -eq 6 ]]; then
      python3 "${code_dir}/canon_diff_chars.py" "$EP_DIR"
      fill_and_run "6"
      # FIX2: validate scaffold fidelity (no residual __FILL__, no drift)
      python3 "${code_dir}/validate_scaffold.py" \
        --scaffold "${EP_DIR}/canon_diff_partial.json" \
        --output   "${EP_DIR}/canon_diff.json"
      continue
    fi

    # ── P8: Stage 4 → Stage 3.5 pre-gate + scaffold + creative fill ─────
    if [[ "$N" -eq 4 ]]; then

      # ── Stage 3.5 pre-gate: ensure VO preview is approved before ShotList ──
      _s35_primary="${PRIMARY_LOCALE:-en}"
      _s35_script="${EP_DIR}/Script.json"

      if ! python3 "${code_dir}/check_vo_approved.py" "${EP_DIR}" "${_s35_primary}" && \
         [[ -f "$_s35_script" && "${STORY_FORMAT:-}" != "ssml_narration" ]]; then
        echo ""
        echo "══════════════════════════════════════════════════════════════"
        echo "  STAGE 3.5  —  VO Preview (primary locale: ${_s35_primary})"
        echo "══════════════════════════════════════════════════════════════"

        _s35_log="stage_logs/${PROJECT_SLUG:-unknown}.${EPISODE_ID:-unknown}.stage_3.5.log"

        echo "  [1] Synthesizing VO from Script.json…"
        python3 "${code_dir}/gen_tts_cloud.py" \
          --script    "${_s35_script}" \
          --voicecast "projects/${PROJECT_SLUG}/VoiceCast.json" \
          --locale    "${_s35_primary}" \
          --stage     "3.5" 2>&1 | tee "$_s35_log"
        _s35_tts_exit=${PIPESTATUS[0]}
        if [[ "$_s35_tts_exit" -ne 0 ]]; then
          echo ""
          echo "✗ Stage 3.5 TTS failed (exit ${_s35_tts_exit})" >&2
          exit "$_s35_tts_exit"
        fi

        echo "  [2] Analysing voice timing…"
        python3 "${code_dir}/post_tts_analysis.py" \
          --manifest "${EP_DIR}/VOPlan.${_s35_primary}.json" 2>&1 | tee -a "$_s35_log"
        _s35_pta_exit=${PIPESTATUS[0]}
        if [[ "$_s35_pta_exit" -ne 0 ]]; then
          echo ""
          echo "✗ Stage 3.5 post_tts_analysis failed (exit ${_s35_pta_exit})" >&2
          exit "$_s35_pta_exit"
        fi

        echo "  [3.5c] Checking TTS accuracy with Whisper…"
        python3 "${code_dir}/whisper_compare.py" "${EP_DIR}" "${_s35_primary}" 2>&1 | tee -a "$_s35_log" || true
        # || true — Whisper failure must never block the pipeline
        # Contract validation — warn only (Stage 3.5 is a preview pass)
        python3 "contracts/tools/verify_contracts.py" \
          "${EP_DIR}/VOPlan.${_s35_primary}.json" || true

        echo ""
        echo "══════════════════════════════════════════════════════════════"
        echo "  ✔  Stage 3.5 complete — VO synthesis done"
        echo ""
        echo "  Voice-over is ready for review.  No pause here — pipeline"
        echo "  continues to Stage 5.  Review and approve VO during the"
        echo "  media selection pause at the end of Stage 5:"
        echo "    1. Switch to the 🎙 VO tab"
        echo "    2. Review every line — trim and adjust timing as needed"
        echo "    3. Click ✓ Approve VO"
        echo "══════════════════════════════════════════════════════════════"
      elif python3 "${code_dir}/check_vo_approved.py" "${EP_DIR}" "${_s35_primary}"; then
        echo "  ✓ Stage 3.5 approved — patch_shotlist_durations.py will set duration_sec after Stage 4"
      elif [[ "${STORY_FORMAT:-}" == "ssml_narration" ]]; then
        echo "  ↷  Stage 3.5 skipped (ssml_narration format)"
      fi
      # ── end Stage 3.5 pre-gate ────────────────────────────────────────────

      _fmt4="${STORY_FORMAT:-episodic}"
      if [[ "$_fmt4" == "continuous_narration" || \
            "$_fmt4" == "illustrated_narration" || \
            "$_fmt4" == "documentary"           || \
            "$_fmt4" == "ssml_narration" ]]; then
        # Skip scaffold generation when resuming at the _c sub-stage directly
        if [[ "$N" -ne "$FROM_BASE" || -z "$FROM_SUFFIX" ]]; then
          python3 "${code_dir}/gen_shotlist_scaffold.py" "$EP_DIR"
        else
          echo "  [4] Skipping scaffold generation (resuming at 4_c)"
        fi
        fill_and_run "4_c"
        # Restore top-level pre-filled scalar fields the LLM may have dropped
        # (e.g. script_ref, schema_id, shotlist_id).  Shots array is untouched.
        python3 "${code_dir}/patch_scaffold_toplevel.py" \
          "${EP_DIR}/ShotList_scaffold.json" \
          "${EP_DIR}/ShotList.json"
        # FIX2: validate scaffold fidelity (no residual __FILL__, no drift).
        # --fix restores any drifted pre-filled values (e.g. vo_text sentences
        # the LLM silently dropped or rewrote) from the scaffold back into the
        # output JSON in-place, so downstream stages get the correct data.
        python3 "${code_dir}/validate_scaffold.py" \
          --scaffold "${EP_DIR}/ShotList_scaffold.json" \
          --output   "${EP_DIR}/ShotList.json" \
          --fix
        # Contract validation — hard stop (downstream stages are meaningless with a broken ShotList)
        python3 "contracts/tools/verify_contracts.py" "${EP_DIR}/ShotList.json" || {
          echo "✗ ShotList.json failed contract validation — aborting"; exit 1; }
        # Patch duration_sec from approved VO timings (replaces LLM placeholder 0s)
        _s4_primary="${PRIMARY_LOCALE:-en}"
        if python3 "${code_dir}/check_vo_approved.py" "${EP_DIR}" "${_s4_primary}"; then
          echo "  [4] Patching ShotList.json duration_sec from VOPlan.${_s4_primary}.json…"
          python3 "${code_dir}/patch_shotlist_durations.py" \
            "${EP_DIR}" --locale "${_s4_primary}" || true
        fi
        auto_refresh_render_plan
        continue
      fi
      # episodic / monologue: fall through to fill_and_run below
    fi

    # ── P6: Stage 5 → scaffold + creative fill (narration) or LLM (episodic)
    # Handles the media selection checkpoint for both paths.
    if [[ "$N" -eq 5 ]]; then
      _fmt5="${STORY_FORMAT:-episodic}"
      if [[ "$_fmt5" == "continuous_narration" || \
            "$_fmt5" == "illustrated_narration" || \
            "$_fmt5" == "documentary"           || \
            "$_fmt5" == "ssml_narration" ]]; then
        # Narration: deterministic scaffold → creative fill → locale VO manifests
        # Skip scaffold generation when resuming at the _c sub-stage directly
        if [[ "$N" -ne "$FROM_BASE" || -z "$FROM_SUFFIX" ]]; then
          python3 "${code_dir}/gen_manifest_structure.py" "$EP_DIR"
        else
          echo "  [5] Skipping scaffold generation (resuming at 5_c)"
        fi
        fill_and_run "5_c"
        # FIX2: check for residual __FILL__ tokens in output (same-file pattern)
        python3 "${code_dir}/validate_scaffold.py" \
          --scaffold "${EP_DIR}/AssetManifest.shared.json" \
          --output   "${EP_DIR}/AssetManifest.shared.json" \
          --warn-only
        # Generate per-locale VO manifests deterministically
        IFS=',' read -ra _s5_locales <<< "${LOCALES:-en}"
        for _s5_raw in "${_s5_locales[@]}"; do
          _s5_locale="$(echo "$_s5_raw" | tr -d ' ')"
          [[ -z "$_s5_locale" ]] && continue
          echo "  [5] Generating VO manifest for locale: ${_s5_locale}…"
          python3 "${code_dir}/gen_vo_manifest.py" \
            --script    "${EP_DIR}/Script.json" \
            --shotlist  "${EP_DIR}/ShotList.json" \
            --voice-cast "projects/${PROJECT_SLUG}/VoiceCast.json" \
            --locale    "$_s5_locale" \
            --out       "${EP_DIR}/VOPlan.${_s5_locale}.json"
          # Patch vo_items start_sec/end_sec/duration_sec from approved VO timing.
          # gen_vo_manifest.py only writes estimated_duration_sec; the approved file
          # is the hard truth and must be applied before resolve_assets runs.
          # Skips gracefully if approval not yet done.
          if python3 "${code_dir}/check_vo_approved.py" "${EP_DIR}" "${_s5_locale}"; then
            echo "  [5] Patching VO manifest timing from VOPlan.${_s5_locale}.json…"
            python3 "${code_dir}/patch_vo_draft_timings.py" \
              "${EP_DIR}" --locale "${_s5_locale}" || true
          fi
          # Contract validation — hard stop (catches writer bugs before TTS/render stages)
          python3 "contracts/tools/verify_contracts.py" \
            "${EP_DIR}/VOPlan.${_s5_locale}.json" || {
            echo "✗ VOPlan.${_s5_locale}.json failed contract validation — aborting"; exit 1; }
        done
        # Patch sfx/music/video-search duration fields from authoritative ShotList
        echo "  [5] Patching AssetManifest.shared.json duration fields from ShotList.json…"
        python3 "${code_dir}/patch_manifest_durations.py" "${EP_DIR}" || true
      else
        # Episodic / monologue: full LLM (shared + locale manifests)
        fill_and_run "5"
        # Patch vo_items start_sec/end_sec from approved VO timing (episodic path).
        # The LLM does not read VOPlan vo_approval, so timing fields are absent.
        IFS=',' read -ra _s5ep_locales <<< "${LOCALES:-en}"
        for _s5ep_raw in "${_s5ep_locales[@]}"; do
          _s5ep_locale="$(echo "$_s5ep_raw" | tr -d ' ')"
          [[ -z "$_s5ep_locale" ]] && continue
          if python3 "${code_dir}/check_vo_approved.py" "${EP_DIR}" "${_s5ep_locale}" && \
             [[ -f "${EP_DIR}/VOPlan.${_s5ep_locale}.json" ]]; then
            echo "  [5] Patching VO manifest timing from VOPlan.${_s5ep_locale}.json…"
            python3 "${code_dir}/patch_vo_draft_timings.py" \
              "${EP_DIR}" --locale "${_s5ep_locale}" || true
          fi
        done
      fi
      # Both paths: run the media selection checkpoint
      if [[ "$TO_STAGE" -gt 5 ]]; then
        _sel_file="${EP_DIR}/MediaPlan.json"
        if [[ ! -f "$_sel_file" ]]; then
          echo ""
          echo "══════════════════════════════════════════════════════════════"
          echo "  ⏸  PAUSED after Stage 5 — VO review + media selections required"
          echo ""
          echo "  Complete both tasks before resuming:"
          echo "    1. 🎙 Switch to the VO tab — review every line, trim and"
          echo "       adjust timing as needed, then click ✓ Approve VO"
          echo "    2. 🖼 Go to the Media tab — select this episode and click"
          echo "       Search Media, choose images/videos for each background,"
          echo "       then click ✔ Confirm Selections"
          echo ""
          echo "  Then resume the pipeline:"
          echo "    ./run.sh ${EP_DIR} 6"
          echo "══════════════════════════════════════════════════════════════"
          exit 0
        else
          echo "  ✓ Media selections found: $_sel_file"
        fi
      fi
      auto_refresh_render_plan
      continue
    fi

    # ── Stage 8 VO review gate (INVARIANT I) ────────────────────────────
    # Stage 8 translation calibrates VO text length against primary-locale
    # WAV durations.  Without an approved VO sentinel those durations are
    # absent or stale, so Stage 8 should not run.
    if [[ "$N" -eq 8 ]]; then
      _s8_primary="${PRIMARY_LOCALE:-en}"
      if ! python3 "${code_dir}/check_vo_approved.py" "${EP_DIR}" "${_s8_primary}"; then
        echo ""
        echo "══════════════════════════════════════════════════════════════"
        echo "  ⛔  Stage 8 gate — VO review not yet approved"
        echo ""
        echo "  VOPlan.${_s8_primary}.json has no vo_approval block."
        echo ""
        echo "  Before running Stage 8 (translation):"
        echo "    1. Run stages 0–4 first (Stage 3.5 pauses for VO review)"
        echo "    2. Switch to the 🎙 VO tab and review every line"
        echo "    3. Click ✓ Approve VO to write vo_approval into:"
        echo "       VOPlan.${_s8_primary}.json"
        echo "    4. Then re-run Stage 8"
        echo "══════════════════════════════════════════════════════════════"
        exit 1
      fi
      echo "  ✓ VO approval found — Stage 8 proceeding with approved timings"
    fi

    # ── Pre-Stage 8 hook: compute locale character-count hints ──────────
    # Uses primary-locale WAV durations from the previous Stage 9 run to give
    # Stage 8 (the translation LLM) calibrated target_chars per VO line.
    # Skips gracefully on the first run when no EN WAVs exist yet.
    if [[ "$N" -eq 8 && "${STORY_FORMAT:-}" != "ssml_narration" ]]; then
      _hint_code_dir="$(cd "$(dirname "$0")" && pwd)/code/http"
      _hint_primary="${PRIMARY_LOCALE:-en}"
      IFS=',' read -ra _hint_locales <<< "${LOCALES:-en}"
      for _raw in "${_hint_locales[@]}"; do
        _loc="$(echo "$_raw" | tr -d ' ')"
        [[ -z "$_loc" || "$_loc" == "$_hint_primary" ]] && continue
        echo "  [pre-8] Computing locale hints for ${_loc}…"
        python3 "${_hint_code_dir}/prep_locale_hints.py" \
          --manifest       "${EP_DIR}/VOPlan.${_hint_primary}.json" \
          --locale         "${_loc}" \
          --primary-locale "${_hint_primary}" || true
      done
    fi

    fill_and_run "$N"

    # ── Patch VOPlan duration fields after Stage 5 (episodic / monologue) ──
    # Narration formats do this inside the narration block above (before continue).
    # Episodic / monologue fall through to fill_and_run, so we patch here.
    if [[ "$N" -eq 5 ]]; then
      if [[ -f "${EP_DIR}/AssetManifest.shared.json" ]]; then
        echo "  [5] Patching AssetManifest.shared.json duration fields from ShotList.json…"
        python3 "${code_dir}/patch_manifest_durations.py" "${EP_DIR}" || true
      fi
    fi

    # ── Patch ShotList duration_sec after Stage 4 (episodic / monologue) ──
    # Narration formats do this inside the narration block above (before continue).
    # Episodic / monologue fall through to fill_and_run, so we patch here.
    if [[ "$N" -eq 4 ]]; then
      _s4ep_primary="${PRIMARY_LOCALE:-en}"
      if python3 "${code_dir}/check_vo_approved.py" "${EP_DIR}" "${_s4ep_primary}"; then
        echo "  [4] Patching ShotList.json duration_sec from VOPlan.${_s4ep_primary}.json…"
        python3 "${code_dir}/patch_shotlist_durations.py" \
          "${EP_DIR}" --locale "${_s4ep_primary}" || true
      fi
      auto_refresh_render_plan
    fi

    # ── Auto-refresh RenderPlan after Stages 5, 6, 7 ─────────────────────────
    # Stages 4 and 5 (narration) call auto_refresh_render_plan before their
    # `continue`.  Episodic Stage 4 calls it above.  Stages 5, 6, 7 (episodic /
    # fallthrough) reach here after fill_and_run, so refresh once for all of them.
    if [[ "$N" -eq 5 || "$N" -eq 6 || "$N" -eq 7 ]]; then
      auto_refresh_render_plan
    fi

    # ── Hard stop after Stage 5: media selection checkpoint ────────────
    # Stage 5 produces VOPlan.  Stage 9 needs
    # MediaPlan.json (written by the VC editor Media tab).
    # If the user asked to continue past stage 5 but hasn't selected
    # media yet, pause here so they can do that first.
    if [[ "$N" -eq 5 && "$TO_STAGE" -gt 5 ]]; then
      _sel_file="${EP_DIR}/MediaPlan.json"
      if [[ ! -f "$_sel_file" ]]; then
        echo ""
        echo "══════════════════════════════════════════════════════════════"
        echo "  ⏸  PAUSED after Stage 5 — VO review + media selections required"
        echo ""
        echo "  Complete both tasks before resuming:"
        echo "    1. 🎙 Switch to the VO tab — review every line, trim and"
        echo "       adjust timing as needed, then click ✓ Approve VO"
        echo "    2. 🖼 Go to the Media tab — select this episode and click"
        echo "       Search Media, choose images/videos for each background,"
        echo "       then click ✔ Confirm Selections"
        echo ""
        echo "  Then resume the pipeline:"
        echo "    ./run.sh ${EP_DIR} 6"
        echo "══════════════════════════════════════════════════════════════"
        exit 0
      else
        echo "  ✓ Media selections found: $_sel_file"
      fi
    fi

    # ── Hard stop after Stage 9: pre-render checkpoint ─────────────────
    # All LLM stages are done.  Pause before the (potentially long)
    # Stage 9 render so the user can review artefacts first.
    if [[ "$N" -eq 9 && "$TO_STAGE" -gt 9 ]]; then
      _sel_file="${EP_DIR}/MediaPlan.json"
      echo ""
      echo "══════════════════════════════════════════════════════════════"
      echo "  ⏸  PAUSED after Stage 9 — ready to render"
      echo ""
      echo "  All LLM stages complete.  Before rendering, verify:"
      if [[ ! -f "$_sel_file" ]]; then
        echo "    ⚠  No media selections found — Stage 9 will lack stock media"
      else
        echo "    ✓  Media selections: $_sel_file"
      fi
      echo "    •  VOPlan.*         — asset manifest"
      echo "    •  VOPlan.*            — render plan"
      echo ""
      echo "  To render:"
      echo "    ./run.sh ${EP_DIR} 9"
      echo "══════════════════════════════════════════════════════════════"
      exit 0
    fi

  fi
done

# ── Stage 9: merge assets & render video ──────────────────────────────
if [[ "$FROM_BASE" -le 10 && "$TO_BASE" -ge 10 ]]; then
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
for N in $(seq "$FROM_BASE" "$TO_BASE"); do
  log="stage_logs/${PROJECT_SLUG:-unknown}.${EPISODE_ID:-unknown}.stage_${N}.log"
  if [[ -f "$log" ]]; then
    size=$(wc -c < "$log")
    echo "  $(basename "$log")  →  ${size} bytes"
  fi
done
