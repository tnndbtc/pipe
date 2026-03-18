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
        echo -e "  ${BOLD}0)${RESET}  Exit"
        echo
        read -rp "  Select an option [0-1]: " choice
        echo

        case "$choice" in
            1)
                run_tests
                read -rp "  Press Enter to return to menu…" _
                ;;
            0)
                echo -e "${YELLOW}Bye.${RESET}"
                echo
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid option — enter 0 or 1.${RESET}"
                sleep 1
                ;;
        esac
    done
}

main_menu
