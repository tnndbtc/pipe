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

# ── Menu ──────────────────────────────────────────────────────────────────────
main_menu() {
    while true; do
        print_header
        echo -e "  ${BOLD}1)${RESET}  Run tests"
        echo -e "  ${BOLD}2)${RESET}  Install dependencies"
        echo -e "  ${BOLD}0)${RESET}  Exit"
        echo
        read -rp "  Select an option [0-2]: " choice
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
            0)
                echo -e "${YELLOW}Bye.${RESET}"
                echo
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid option — enter 0, 1, or 2.${RESET}"
                sleep 1
                ;;
        esac
    done
}

main_menu
