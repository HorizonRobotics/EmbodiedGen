#!/bin/bash
set -e
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
source "$SCRIPT_DIR/_utils.sh"
cd "$PROJECT_ROOT"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
    cat > "$CONDA_PREFIX/etc/conda/activate.d/opencv_openexr.sh" <<'HOOK'
export OPENCV_IO_ENABLE_OPENEXR="${OPENCV_IO_ENABLE_OPENEXR:-1}"
HOOK
    log_info "Wrote OpenEXR activation hook into the conda environment."
fi

CUDA_VARIANT=$(detect_cuda_variant)
TORCH_INDEX_URL="${EMBODIEDGEN_TORCH_INDEX_URL:-https://download.pytorch.org/whl/$CUDA_VARIANT}"
log_info "Installing PyTorch dependencies for CUDA variant: $CUDA_VARIANT"

PIP_INSTALL_PACKAGES=(
    "pip==22.3.1"
    "setuptools==80.10.2 wheel packaging 'Cython>=0.29.37'"
    "torch==2.8.0 torchvision==0.23.0 --index-url $TORCH_INDEX_URL"
    "xformers==0.0.32.post2 --index-url $TORCH_INDEX_URL"
    "-r requirements.txt --use-deprecated=legacy-resolver"
    "utils3d@git+https://github.com/EasternJournalist/utils3d.git@9a4eb15"
    "clip@git+https://github.com/openai/CLIP.git"
    "segment-anything@git+https://github.com/facebookresearch/segment-anything.git@dca509f"
    "nvdiffrast@git+https://github.com/NVlabs/nvdiffrast.git@729261d"
    "kolors@git+https://github.com/HochCC/Kolors.git"
    "--no-build-isolation kaolin@git+https://github.com/NVIDIAGameWorks/kaolin.git@v0.18.0"
    "--no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3"
    "--no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git@stable"
    "MoGe@git+https://github.com/microsoft/MoGe.git@a8c3734"
)

for pkg in "${PIP_INSTALL_PACKAGES[@]}"; do
    try_install "Installing $pkg..." \
        "pip install ${pkg}" \
        "$pkg installation failed."
done

log_info "Installing diff-gaussian-rasterization..."
pip install --no-build-isolation diff-gaussian-rasterization@git+https://github.com/autonomousvision/mip-splatting.git#subdirectory=submodules/diff-gaussian-rasterization

try_install "Installing EmbodiedGen..." \
    "pip install -e '.[dev]'" \
    "EmbodiedGen installation failed."

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    pre-commit install
else
    log_info "Skipping pre-commit hook installation outside a Git worktree."
fi
