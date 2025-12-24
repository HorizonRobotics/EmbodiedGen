#!/bin/bash
set -e
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/_utils.sh"

PIP_INSTALL_PACKAGES=(
    "pip==22.3.1"
    "torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118"
    "xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118"
    "-r requirements.txt --use-deprecated=legacy-resolver"
    # "flash-attn==2.7.0.post2"
    "utils3d@git+https://github.com/EasternJournalist/utils3d.git@9a4eb15"
    "clip@git+https://github.com/openai/CLIP.git"
    "segment-anything@git+https://github.com/facebookresearch/segment-anything.git@dca509f"
    "nvdiffrast@git+https://github.com/NVlabs/nvdiffrast.git@729261d"
    "kolors@git+https://github.com/HochCC/Kolors.git"
    "kaolin@git+https://github.com/NVIDIAGameWorks/kaolin.git@v0.16.0"
    "git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3"
    "git+https://github.com/facebookresearch/pytorch3d.git@stable"
    "MoGe@git+https://github.com/microsoft/MoGe.git@a8c3734"
)

for pkg in "${PIP_INSTALL_PACKAGES[@]}"; do
    try_install "Installing $pkg..." \
        "pip install $pkg" \
        "$pkg installation failed."
done

log_info "Installing diff-gaussian-rasterization..."
TMP_DIR="/tmp/mip-splatting"
rm -rf "$TMP_DIR"
git clone --recursive https://github.com/autonomousvision/mip-splatting.git "$TMP_DIR"
pip install "$TMP_DIR/submodules/diff-gaussian-rasterization"
rm -rf "$TMP_DIR"

try_install "Installing EmbodiedGen..." \
    "pip install -e .[dev]" \
    "EmbodiedGen installation failed."
