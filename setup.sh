#!/usr/bin/env bash
# =============================================================================
# setup.sh — Development utilities for the narrative pipeline
# =============================================================================
set -euo pipefail

PIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

print_header() {
    echo
    echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════╗${RESET}"
    echo -e "${BOLD}${CYAN}║   Narrative Pipeline — Setup & Utilities ║${RESET}"
    echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════╝${RESET}"
    echo
}

# ── Install dependencies ──────────────────────────────────────────────────────
install_deps() {
    echo -e "${BOLD}Installing dependencies…${RESET}"
    echo

    local exit_code=0

    # ── 1. Main Linux Python requirements ────────────────────────────────────
    echo -e "${CYAN}[1/4] Python — requirements_linux.txt${RESET}"
    pip install -r "$PIPE_DIR/requirements_linux.txt" || exit_code=$?

    # ── 2. Media server Python requirements ──────────────────────────────────
    echo
    echo -e "${CYAN}[2/4] Python — code/media/http/requirements.txt${RESET}"
    pip install -r "$PIPE_DIR/code/media/http/requirements.txt" || exit_code=$?

    # ── 3. AI HTTP server Python requirements ────────────────────────────────
    echo
    echo -e "${CYAN}[3/4] Python — code/ai/http/requirements.txt${RESET}"
    pip install -r "$PIPE_DIR/code/ai/http/requirements.txt" || exit_code=$?

    # ── 4. Node / Playwright ─────────────────────────────────────────────────
    echo
    echo -e "${CYAN}[4/4] Node — tests/ npm install + playwright browsers${RESET}"
    (cd "$PIPE_DIR/tests" && npm install && npx playwright install) || exit_code=$?

    # ── Summary ──────────────────────────────────────────────────────────────
    echo
    if [[ $exit_code -eq 0 ]]; then
        echo -e "${GREEN}${BOLD}✓ All dependencies installed${RESET}"
    else
        echo -e "${RED}${BOLD}✗ One or more steps failed (exit code $exit_code)${RESET}"
    fi
    echo

    return $exit_code
}

# ── Test runner ───────────────────────────────────────────────────────────────
run_tests() {
    echo -e "${BOLD}Running tests…${RESET}"
    echo

    local exit_code=0

    # ── 1. Contract / schema validation ──────────────────────────────────────
    echo -e "${CYAN}[1/3] Contract validation${RESET}  contracts/tools/verify_contracts.py"
    python3 "$PIPE_DIR/contracts/tools/verify_contracts.py" || exit_code=$?

    # ── 2. Media source-limit unit tests ─────────────────────────────────────
    echo -e "${CYAN}[2/3] Media source limits${RESET}  code/media/http/tests/test_source_limits.py"
    python3 -m pytest "$PIPE_DIR/code/media/http/tests/test_source_limits.py" -v || exit_code=$?

    # ── 3. Playwright integration tests ──────────────────────────────────────
    echo -e "${CYAN}[3/3] Playwright integration${RESET}  tests/"
    (cd "$PIPE_DIR/tests" && npm test) || exit_code=$?

    # ── Summary ──────────────────────────────────────────────────────────────
    echo
    if [[ $exit_code -eq 0 ]]; then
        echo -e "${GREEN}${BOLD}✓ All tests passed${RESET}"
    else
        echo -e "${RED}${BOLD}✗ Tests failed (exit code $exit_code)${RESET}"
    fi
    echo

    return $exit_code
}

# ── Project status ────────────────────────────────────────────────────────────
show_project_status() {
    local projects_dir="$PIPE_DIR/projects"

    echo
    echo -e "  ${BOLD}Simple Narration Projects — Status Report${RESET}"
    echo

    if [ ! -d "$projects_dir" ]; then
        echo -e "  ${RED}Projects directory not found: $projects_dir${RESET}"
        echo
        return
    fi

    python3 - "$projects_dir" <<'PYEOF'
import sys, os, json

projects_dir = sys.argv[1]

RESET = "\033[0m"
BOLD  = "\033[1m"
CYAN  = "\033[0;36m"
DIM   = "\033[2m"

ready_to_review        = []
uploaded_not_published = []

def find_file(base, name):
    for root, _dirs, files in os.walk(base):
        if name in files:
            return os.path.join(root, name)
    return None

def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

for proj_name in sorted(os.listdir(projects_dir)):
    proj_path = os.path.join(projects_dir, proj_name)
    if not os.path.isdir(proj_path):
        continue

    meta_path = find_file(proj_path, "meta.json")
    if not meta_path:
        continue
    meta = load_json(meta_path)
    if not meta or meta.get("story_format") != "simple_narration":
        continue

    render_path  = find_file(proj_path, "render_output.json")
    upload_path  = find_file(proj_path, "upload_state.json")
    youtube_path = find_file(proj_path, "youtube.json")

    has_render = render_path is not None
    has_upload = upload_path is not None

    if has_render and not has_upload:
        ready_to_review.append(proj_name)
    elif has_render and has_upload:
        upload_data = load_json(upload_path)
        yt_data     = load_json(youtube_path) if youtube_path else None
        video_up    = upload_data and upload_data.get("video_uploaded") is True
        privacy     = yt_data.get("privacy") if yt_data else None
        if video_up and privacy == "private":
            uploaded_not_published.append(proj_name)

sep = f"  {CYAN}{'─'*55}{RESET}"

print(sep)
print(f"  {BOLD}Ready to review{RESET}  ({len(ready_to_review)})")
print(sep)
if ready_to_review:
    for name in ready_to_review:
        print(f"    {name}")
else:
    print(f"    {DIM}(none){RESET}")

print()
print(sep)
print(f"  {BOLD}Uploaded — not yet published{RESET}  ({len(uploaded_not_published)})")
print(sep)
if uploaded_not_published:
    for name in uploaded_not_published:
        print(f"    {name}")
else:
    print(f"    {DIM}(none){RESET}")
PYEOF

    echo
}

# ── Menu ──────────────────────────────────────────────────────────────────────
main_menu() {
    while true; do
        print_header
        echo -e "  ${BOLD}1)${RESET}  Run tests"
        echo -e "  ${BOLD}2)${RESET}  Install dependencies"
        echo -e "  ${BOLD}3)${RESET}  Project status  (ready to review | uploaded, not published)"
        echo -e "  ${BOLD}0)${RESET}  Exit"
        echo
        read -rp "  Select an option [0-3]: " choice
        echo

        case "$choice" in
            1)
                run_tests
                read -rp "  Press Enter to return to menu…" _
                ;;
            2)
                install_deps
                read -rp "  Press Enter to return to menu…" _
                ;;
            3)
                show_project_status
                read -rp "  Press Enter to return to menu…" _
                ;;
            0)
                echo -e "${YELLOW}Bye.${RESET}"
                echo
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid option — enter 0, 1, 2, or 3.${RESET}"
                sleep 1
                ;;
        esac
    done
}

main_menu
