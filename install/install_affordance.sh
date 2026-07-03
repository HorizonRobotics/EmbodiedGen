#!/bin/bash
set -e
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
source "$SCRIPT_DIR/_utils.sh"

PIP_INSTALL_PACKAGES=(
    "fpsample==1.0.2"
    "viser==1.0.30"
    "pickle5==0.0.11"
    "webdataset==1.0.2"
    "yourdfpy==0.0.56"
    "sharedarray==3.2.4"
    "meshcat==0.3.2"
)

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    if [[ -f "$CONDA_PREFIX/etc/conda/activate.d/cuda126.sh" ]]; then
        source "$CONDA_PREFIX/etc/conda/activate.d/cuda126.sh"
    fi
fi

export MAX_JOBS="${MAX_JOBS:-8}"
log_info "Using TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-unset} for affordance CUDA extensions."

try_install "Installing torch-scatter==2.1.2 from source..." \
    "FORCE_CUDA=1 pip install --no-cache-dir --no-build-isolation --no-binary torch-scatter --force-reinstall --no-deps torch-scatter==2.1.2" \
    "torch-scatter source installation failed."

try_install "Installing flash-attn==2.8.3 from source..." \
    "FLASH_ATTENTION_FORCE_BUILD=TRUE pip install --no-cache-dir --no-build-isolation --no-binary flash-attn --force-reinstall --no-deps flash-attn==2.8.3" \
    "flash-attn source installation failed."

for pkg in "${PIP_INSTALL_PACKAGES[@]}"; do
    try_install "Installing $pkg..." \
        "pip install ${pkg}" \
        "$pkg installation failed."
done

POINTNET2_OPS_DIR="$PROJECT_ROOT/thirdparty/GraspGen/pointnet2_ops"

if [[ ! -d "$POINTNET2_OPS_DIR" ]]; then
    log_error "GraspGen pointnet2_ops directory not found: $POINTNET2_OPS_DIR"
    exit 1
fi

try_install "Installing GraspGen pointnet2_ops..." \
    "pip install --no-build-isolation $POINTNET2_OPS_DIR" \
    "GraspGen pointnet2_ops installation failed."

rm -r "$POINTNET2_OPS_DIR/build" 2>/dev/null || true
