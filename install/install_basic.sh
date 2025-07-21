#!/bin/bash
set -e
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/_utils.sh"

try_install "Installing flash-attn..." \
    "pip install flash-attn==2.7.0.post2 --no-build-isolation" \
    "flash-attn installation failed."

try_install "Installing requirements.txt..." \
    "pip install -r requirements.txt --use-deprecated=legacy-resolver --default-timeout=60" \
    "requirements installation failed."

try_install "Installing kolors..." \
    "pip install kolors@git+https://github.com/HochCC/Kolors.git" \
    "kolors installation failed."

try_install "Installing kaolin..." \
    "pip install kaolin@git+https://github.com/NVIDIAGameWorks/kaolin.git@v0.16.0" \
    "kaolin installation failed."

log_info "Installing diff-gaussian-rasterization..."
TMP_DIR="/tmp/mip-splatting"
rm -rf "$TMP_DIR"
git clone --recursive https://github.com/autonomousvision/mip-splatting.git "$TMP_DIR"
pip install "$TMP_DIR/submodules/diff-gaussian-rasterization"
rm -rf "$TMP_DIR"

try_install "Installing gsplat..." \
    "pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3" \
    "gsplat installation failed."

try_install "Installing EmbodiedGen..." \
    "pip install triton==2.1.0 --no-deps && pip install -e ." \
    "EmbodiedGen installation failed."
