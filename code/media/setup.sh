#!/usr/bin/env bash
# -------------------------------------------------------------------
# setup.sh — Launcher menu for Media Search & Rating Service
#
# Usage:  bash setup.sh
#         ./setup.sh          (if chmod +x)
# -------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HTTP_DIR="$SCRIPT_DIR/http"

# -------------------------------------------------------------------
# Colors (ANSI-C quoting so they work inside heredocs)
# -------------------------------------------------------------------
BOLD=$'\033[1m'
DIM=$'\033[2m'
CYAN=$'\033[36m'
GREEN=$'\033[32m'
YELLOW=$'\033[33m'
RESET=$'\033[0m'

# -------------------------------------------------------------------
# Show usage / help
# -------------------------------------------------------------------
show_usage() {
    cat <<EOF

${BOLD}=== Media Search & Rating Service ===${RESET}

${BOLD}${CYAN}1. Start the main server${RESET}
${DIM}   (orchestrates batches, serves API, distributes jobs to workers)${RESET}

   cd ${HTTP_DIR}
   python3 server.py [OPTIONS]

   ${BOLD}Options:${RESET}
     --host HOST       Bind address            ${DIM}(default: 0.0.0.0)${RESET}
     --port PORT       Bind port               ${DIM}(default: 8200)${RESET}

   ${BOLD}Required env vars:${RESET}
     MEDIA_API_KEY       Server authentication key (X-Api-Key header)
     PEXELS_API_KEY      Pexels search API key
     PIXABAY_API_KEY     Pixabay search API key

   ${BOLD}Example:${RESET}
     ${GREEN}export MEDIA_API_KEY=xxx PEXELS_API_KEY=xxx PIXABAY_API_KEY=xxx
     python3 server.py --host 0.0.0.0 --port 8200${RESET}

${BOLD}${CYAN}2. Start a scoring worker${RESET}
${DIM}   (polls server for video scoring jobs, runs ffmpeg + CLIP locally)${RESET}

   cd ${HTTP_DIR}
   python3 worker.py [OPTIONS]

   ${BOLD}Options:${RESET}
     --server URL       Main server URL         ${DIM}(required)${RESET}
                        e.g. http://192.168.86.33:8200
     --name NAME        Worker name             ${DIM}(default: hostname)${RESET}
     --nfs-root PATH    Local NFS mount path    ${DIM}(default: /mnt/shared)${RESET}
     --clip-model NAME  CLIP model override     ${DIM}(default: from server)${RESET}
     --clip-pretrained W Pretrained weights      ${DIM}(default: from server)${RESET}

   ${BOLD}Examples:${RESET}

     ${YELLOW}# Remote worker (NFS client machine):${RESET}
     ${GREEN}python3 worker.py --server http://192.168.86.33:8200 \\
         --name alma-41 --nfs-root /mnt/shared${RESET}

     ${YELLOW}# Worker on the media server itself (no NFS needed):${RESET}
     ${GREEN}python3 worker.py --server http://localhost:8200 \\
         --name media-local --nfs-root /data/shared${RESET}

     ${YELLOW}# Worker with different NFS mount point:${RESET}
     ${GREEN}python3 worker.py --server http://192.168.86.33:8200 \\
         --name ubuntu-42 --nfs-root /nfs/media${RESET}

${BOLD}${CYAN}3. Prerequisites (all nodes)${RESET}

   - Python 3.11+
   - ffmpeg
   - pip install -r ${HTTP_DIR}/requirements.txt
   - NFS mount (or local path on media server)
   - CLIP model auto-downloads on first run (~350 MB)

${BOLD}${CYAN}4. Monitoring${RESET}

   ${GREEN}curl http://192.168.86.33:8200/health${RESET}     ${DIM}# server status${RESET}
   ${GREEN}curl http://192.168.86.33:8200/workers${RESET}    ${DIM}# registered workers + queue${RESET}

${BOLD}${CYAN}5. config.json (workers section)${RESET}

   ${DIM}# ${HTTP_DIR}/config.json${RESET}
   "workers": {
     "enabled": true,              ${DIM}# false = local-only scoring${RESET}
     "timeout_seconds": 120,       ${DIM}# re-queue stale jobs after this${RESET}
     "fallback_grace_seconds": 10, ${DIM}# wait for workers before fallback${RESET}
     "server_nfs_root": "/data/shared"
   }

EOF
}

# -------------------------------------------------------------------
# Menu loop
# -------------------------------------------------------------------
while true; do
    echo ""
    echo "${BOLD}=== Media Service Setup ===${RESET}"
    echo "  1) Show usage (server & worker options)"
    echo "  2) Install dependencies"
    echo "  0) Exit"
    echo ""
    read -rp "Choose [0-2]: " choice

    case "$choice" in
        1) show_usage ;;
        2)
            echo ""
            echo "${BOLD}Install dependencies${RESET}"
            echo "${DIM}This will:${RESET}"
            echo "${DIM}  - Install system packages: ffmpeg, ffprobe${RESET}"
            echo "${DIM}  - Install PyTorch CPU-only wheel${RESET}"
            echo "${DIM}  - Run: pip install -r ${HTTP_DIR}/requirements.txt${RESET}"
            echo ""
            bash "$SCRIPT_DIR/install_deps.sh" --pip --torch-cpu
            ;;
        0) echo "Bye."; exit 0 ;;
        *) echo "Invalid option: $choice" ;;
    esac
done
