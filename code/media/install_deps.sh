#!/usr/bin/env bash
# =============================================================================
# install_deps.sh — Install system dependencies for the media server & workers
#
# Installs everything NOT covered by requirements.txt:
#   - ffmpeg + ffprobe  (video frame extraction + duration probing)
#
# Supports:
#   - Ubuntu 20.04 / 22.04 / 24.04  (apt)
#   - AlmaLinux 8 / 9               (dnf + EPEL + RPM Fusion)
#
# Usage:
#   bash install_deps.sh            # installs system deps only
#   bash install_deps.sh --pip      # also installs pip packages from requirements.txt
#   bash install_deps.sh --pip --torch-cpu   # also installs PyTorch CPU-only wheel
#
# Sudo password will be prompted when needed.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="$SCRIPT_DIR/http/requirements.txt"

# -----------------------------------------------------------------------------
# Flags
# -----------------------------------------------------------------------------
DO_PIP=false
DO_TORCH=false

for arg in "$@"; do
    case "$arg" in
        --pip)       DO_PIP=true ;;
        --torch-cpu) DO_TORCH=true ;;
        --help|-h)
            grep '^#' "$0" | head -20 | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            echo "Usage: $0 [--pip] [--torch-cpu]" >&2
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Colors
# -----------------------------------------------------------------------------
BOLD=$'\033[1m'
GREEN=$'\033[32m'
YELLOW=$'\033[33m'
RED=$'\033[31m'
DIM=$'\033[2m'
RESET=$'\033[0m'

info()    { echo "${GREEN}[+]${RESET} $*"; }
warn()    { echo "${YELLOW}[!]${RESET} $*"; }
section() { echo ""; echo "${BOLD}--- $* ---${RESET}"; }
die()     { echo "${RED}[✗]${RESET} $*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# OS detection
# -----------------------------------------------------------------------------
section "Detecting OS"

OS_ID=""
OS_VERSION=""

if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    OS_ID="${ID:-}"
    OS_VERSION="${VERSION_ID:-}"
fi

case "$OS_ID" in
    ubuntu)
        info "Detected Ubuntu ${OS_VERSION}"
        ;;
    almalinux|alma)
        info "Detected AlmaLinux ${OS_VERSION}"
        ;;
    *)
        die "Unsupported OS: '${OS_ID}'. This script supports Ubuntu and AlmaLinux only."
        ;;
esac

# Prompt for sudo upfront so it's cached for the rest of the script
info "Requesting sudo access (you may be prompted for your password)"
sudo -v

# Keep sudo alive in background for long installs
while true; do sudo -n true; sleep 50; kill -0 "$$" 2>/dev/null || exit; done 2>/dev/null &
_SUDO_KEEPALIVE_PID=$!
trap 'kill $_SUDO_KEEPALIVE_PID 2>/dev/null || true' EXIT

# =============================================================================
# Ubuntu
# =============================================================================
install_ubuntu() {
    section "Ubuntu: Updating package index"
    sudo apt-get update -qq

    section "Ubuntu: Installing ffmpeg (includes ffprobe)"
    sudo apt-get install -y ffmpeg
    info "ffmpeg installed: $(ffmpeg -version 2>&1 | head -1)"
    info "ffprobe installed: $(ffprobe -version 2>&1 | head -1)"

}

# =============================================================================
# AlmaLinux
# =============================================================================
install_almalinux() {
    section "AlmaLinux: Installing EPEL release"
    if rpm -q epel-release &>/dev/null; then
        info "epel-release already installed"
    else
        sudo dnf install -y epel-release
        info "epel-release installed"
    fi

    section "AlmaLinux: Installing RPM Fusion (free) for ffmpeg"
    ALMA_MAJ="${OS_VERSION%%.*}"
    RPMFUSION_FREE="https://mirrors.rpmfusion.org/free/el/rpmfusion-free-release-${ALMA_MAJ}.noarch.rpm"
    if rpm -q rpmfusion-free-release &>/dev/null; then
        info "rpmfusion-free-release already installed"
    else
        sudo dnf install -y "$RPMFUSION_FREE"
        info "rpmfusion-free-release installed"
    fi

    section "AlmaLinux: Installing ffmpeg (includes ffprobe)"
    # --allowerasing needed because AlmaLinux ships ffmpeg-free stub that conflicts
    sudo dnf install -y --allowerasing ffmpeg ffmpeg-libs
    info "ffmpeg installed: $(ffmpeg -version 2>&1 | head -1)"
    info "ffprobe installed: $(ffprobe -version 2>&1 | head -1)"

}

# =============================================================================
# Dispatch to OS-specific installer
# =============================================================================
case "$OS_ID" in
    ubuntu)        install_ubuntu ;;
    almalinux|alma) install_almalinux ;;
esac

# =============================================================================
# Optional: PyTorch CPU-only (must come before requirements.txt)
# =============================================================================
if $DO_TORCH; then
    section "Installing PyTorch (CPU-only wheel)"
    PY_CMD="$(command -v python3)"
    PIP_CMD="$PY_CMD -m pip"
    info "Using Python: $PY_CMD"
    $PIP_CMD install --upgrade pip
    $PIP_CMD install torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu
    info "PyTorch installed: $(python3 -c 'import torch; print(torch.__version__)')"
fi

# =============================================================================
# Optional: pip install -r requirements.txt
# =============================================================================
if $DO_PIP; then
    section "Installing Python packages from requirements.txt"
    if [[ ! -f "$REQ_FILE" ]]; then
        die "requirements.txt not found at: $REQ_FILE"
    fi
    PY_CMD="$(command -v python3)"
    PIP_CMD="$PY_CMD -m pip"
    info "Using: $PIP_CMD"
    info "File : $REQ_FILE"
    $PIP_CMD install --upgrade pip
    $PIP_CMD install -r "$REQ_FILE"
    info "pip packages installed"
fi

# =============================================================================
# Verification summary
# =============================================================================
section "Verification"

check_cmd() {
    local cmd="$1" label="$2"
    if command -v "$cmd" &>/dev/null; then
        info "${label}: $(command -v "$cmd")  $("$cmd" --version 2>&1 | head -1)"
    else
        warn "${label}: NOT FOUND"
    fi
}

check_cmd ffmpeg  "ffmpeg "
check_cmd ffprobe "ffprobe"

if $DO_PIP || $DO_TORCH; then
    echo ""
    echo "${DIM}Checking key Python packages:${RESET}"
    $PY_CMD -c "import torch; print('  torch          ' + torch.__version__)"  2>/dev/null \
        || echo "${YELLOW}  torch          not installed${RESET}"
    $PY_CMD -c "import open_clip; print('  open_clip      ok')"                2>/dev/null \
        || echo "${YELLOW}  open_clip      not installed${RESET}"
    $PY_CMD -c "import cv2; print('  opencv         ' + cv2.__version__)"      2>/dev/null \
        || echo "${YELLOW}  opencv         not installed${RESET}"
    $PY_CMD -c "import imagehash; print('  imagehash      ok')"                2>/dev/null \
        || echo "${YELLOW}  imagehash      not installed${RESET}"
    $PY_CMD -c "import requests; print('  requests       ' + requests.__version__)" 2>/dev/null \
        || echo "${YELLOW}  requests       not installed${RESET}"
fi

echo ""
info "Done. System dependencies installed successfully."
if ! $DO_PIP; then
    echo "${DIM}Tip: run with --pip to also install Python packages from requirements.txt${RESET}"
    echo "${DIM}     run with --pip --torch-cpu to include the PyTorch CPU wheel${RESET}"
fi
echo ""
