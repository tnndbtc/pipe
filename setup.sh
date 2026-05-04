#!/usr/bin/env bash
# =============================================================================
# setup.sh — Development utilities for the narrative pipeline
# =============================================================================
set -euo pipefail

PIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'

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

# ── Delete published project ──────────────────────────────────────────────────
delete_published_project() {
    local projects_dir="$PIPE_DIR/projects"

    echo
    echo -e "  ${BOLD}Published Projects — Select one to delete${RESET}"
    echo

    if [ ! -d "$projects_dir" ]; then
        echo -e "  ${RED}Projects directory not found: $projects_dir${RESET}"
        echo
        return
    fi

    local tmp_list
    tmp_list=$(mktemp)

    python3 - "$projects_dir" "$tmp_list" <<'PYEOF'
import sys, os, json

projects_dir = sys.argv[1]
tmp_list     = sys.argv[2]

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[0;36m"
DIM    = "\033[2m"

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

published = []  # list of (proj_name, video_id)

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

    upload_path  = find_file(proj_path, "upload_state.json")
    youtube_path = find_file(proj_path, "youtube.json")
    if not upload_path or not youtube_path:
        continue

    upload_data = load_json(upload_path)
    yt_data     = load_json(youtube_path)
    if not upload_data or not yt_data:
        continue

    video_uploaded = upload_data.get("video_uploaded") is True
    privacy        = yt_data.get("privacy", "")
    video_id       = upload_data.get("video_id", "")

    if video_uploaded and privacy == "public":
        published.append((proj_name, video_id))

sep = f"  {CYAN}{'─'*65}{RESET}"
print(sep)
if published:
    for i, (name, vid) in enumerate(published, 1):
        vid_str = f"  {DIM}youtu.be/{vid}{RESET}" if vid else ""
        print(f"  {BOLD}{i:>2}){RESET}  {name}{vid_str}")
else:
    print(f"    {DIM}(no published projects found){RESET}")
print(sep)

with open(tmp_list, "w") as f:
    for name, _ in published:
        f.write(name + "\n")
PYEOF

    local count
    count=$(cat "$tmp_list" | wc -l)

    if [[ "$count" -eq 0 ]]; then
        echo
        rm -f "$tmp_list"
        return
    fi

    echo
    read -rp "  Select number or range to delete (e.g. 3 or 1-8), or 0 to cancel: " sel
    echo

    if [[ -z "$sel" || "$sel" == "0" ]]; then
        echo -e "  ${YELLOW}Cancelled.${RESET}"
        rm -f "$tmp_list"
        echo
        return
    fi

    local lo hi
    if [[ "$sel" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        # Range input: N-M
        lo="${BASH_REMATCH[1]}"
        hi="${BASH_REMATCH[2]}"
    elif [[ "$sel" =~ ^[0-9]+$ ]]; then
        # Single number
        lo="$sel"
        hi="$sel"
    else
        echo -e "  ${RED}Invalid input — enter a number (e.g. 3) or a range (e.g. 1-8).${RESET}"
        rm -f "$tmp_list"
        echo
        return
    fi

    if [[ "$lo" -lt 1 || "$hi" -gt "$count" || "$lo" -gt "$hi" ]]; then
        echo -e "  ${RED}Invalid range — values must be between 1 and ${count}, low ≤ high.${RESET}"
        rm -f "$tmp_list"
        echo
        return
    fi

    # Collect the selected project names
    local selected=()
    local i
    for (( i = lo; i <= hi; i++ )); do
        selected+=("$(sed -n "${i}p" "$tmp_list")")
    done
    rm -f "$tmp_list"

    if [[ "${#selected[@]}" -eq 1 ]]; then
        # ── Single project: require typing the project name ────────────────────
        local proj_name="${selected[0]}"
        echo -e "  ${BOLD}Project to delete:${RESET}  $proj_name"
        echo -e "  ${RED}This will permanently remove the entire project folder and all its files.${RESET}"
        echo
        read -rp "  Type the project name to confirm: " confirm
        echo

        if [[ "$confirm" == "$proj_name" ]]; then
            rm -rf "$projects_dir/$proj_name"
            echo -e "  ${GREEN}${BOLD}✓ Deleted:${RESET}  $proj_name"
        else
            echo -e "  ${YELLOW}Cancelled — name did not match.${RESET}"
        fi
    else
        # ── Range: list all affected projects, require typing YES ──────────────
        echo -e "  ${BOLD}Projects to delete (${#selected[@]}):${RESET}"
        local p
        for p in "${selected[@]}"; do
            echo -e "    ${RED}•${RESET}  $p"
        done
        echo
        echo -e "  ${RED}This will permanently remove all ${#selected[@]} project folders and their files.${RESET}"
        echo
        read -rp "  Type YES to confirm bulk deletion: " confirm
        echo

        if [[ "$confirm" == "YES" ]]; then
            for p in "${selected[@]}"; do
                rm -rf "$projects_dir/$p"
                echo -e "  ${GREEN}${BOLD}✓ Deleted:${RESET}  $p"
            done
        else
            echo -e "  ${YELLOW}Cancelled — 'YES' not entered.${RESET}"
        fi
    fi
    echo
}

# ── Start / restart pipe server ───────────────────────────────────────────────
start_pipe_server() {
    local server_script="code/http/server.py"
    local log_file="$PIPE_DIR/code/http/logs/server.log"

    # Kill any existing instance
    local existing_pid
    existing_pid=$(pgrep -f "$server_script" 2>/dev/null || true)
    if [[ -n "$existing_pid" ]]; then
        echo -e "  ${YELLOW}Found running server (PID $existing_pid) — stopping…${RESET}"
        kill "$existing_pid" 2>/dev/null || true
        # Wait up to 5 s for it to die
        local i
        for (( i = 0; i < 10; i++ )); do
            sleep 0.5
            pgrep -f "$server_script" &>/dev/null || break
        done
        if pgrep -f "$server_script" &>/dev/null; then
            echo -e "  ${RED}Process did not exit — sending SIGKILL…${RESET}"
            pkill -9 -f "$server_script" 2>/dev/null || true
            sleep 1
        fi
        echo -e "  ${GREEN}✓ Stopped${RESET}"
    else
        echo -e "  ${CYAN}No running server found.${RESET}"
    fi

    # Start fresh
    mkdir -p "$(dirname "$log_file")"
    echo -e "  ${BOLD}Starting pipe server…${RESET}"
    echo -e "  ${DIM}Log: $log_file${RESET}"
    nohup python3 "$PIPE_DIR/$server_script" \
        > "$log_file" 2>&1 &
    local new_pid=$!
    sleep 1

    if pgrep -f "$server_script" &>/dev/null; then
        echo -e "  ${GREEN}${BOLD}✓ Server started (PID $new_pid)${RESET}"
        echo -e "  ${CYAN}http://localhost:8000${RESET}"
    else
        echo -e "  ${RED}${BOLD}✗ Server failed to start — check log:${RESET}"
        echo -e "  ${DIM}tail -20 $log_file${RESET}"
    fi
    echo
}

# ── Menu ──────────────────────────────────────────────────────────────────────
main_menu() {
    while true; do
        print_header
        echo -e "  ${BOLD}1)${RESET}  Start / restart pipe server  (code/http/server.py)"
        echo -e "  ${BOLD}2)${RESET}  Run tests"
        echo -e "  ${BOLD}3)${RESET}  Install dependencies"
        echo -e "  ${BOLD}4)${RESET}  Project status  (ready to review | uploaded, not published)"
        echo -e "  ${BOLD}5)${RESET}  Delete a published project"
        echo -e "  ${BOLD}0)${RESET}  Exit"
        echo
        read -rp "  Select an option [0-5]: " choice
        echo

        case "$choice" in
            1)
                start_pipe_server
                read -rp "  Press Enter to return to menu…" _
                ;;
            2)
                run_tests
                read -rp "  Press Enter to return to menu…" _
                ;;
            3)
                install_deps
                read -rp "  Press Enter to return to menu…" _
                ;;
            4)
                show_project_status
                read -rp "  Press Enter to return to menu…" _
                ;;
            5)
                delete_published_project
                read -rp "  Press Enter to return to menu…" _
                ;;
            0)
                echo -e "${YELLOW}Bye.${RESET}"
                echo
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid option — enter 0, 1, 2, 3, 4, or 5.${RESET}"
                sleep 1
                ;;
        esac
    done
}

main_menu
