#!/bin/bash
set -e
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/_utils.sh"

CONDA_CMD="${CONDA_EXE:-}"

if [[ -n "$CONDA_CMD" && ! -x "$CONDA_CMD" ]]; then
    CONDA_CMD=""
fi

if [[ -z "$CONDA_CMD" ]]; then
    CONDA_CMD=$(command -v conda || true)
fi

if [[ -z "$CONDA_CMD" ]]; then
    log_error "conda is required to install CUDA 12.6 into the active environment."
    exit 1
fi

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    log_error "No active conda environment detected. Please run 'conda activate <env>' first."
    exit 1
fi

log_info "Installing CUDA 12.6 toolkit into conda environment: $CONDA_PREFIX"
log_info "Using conda executable: $CONDA_CMD"
"$CONDA_CMD" install \
    -p "$CONDA_PREFIX" \
    --override-channels \
    -c nvidia \
    -c conda-forge \
    cuda-toolkit=12.6 \
    cuda-nvcc=12.6 \
    -y

log_info "Writing CUDA 12.6 activation hook into the conda environment..."
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/cuda126.sh" <<'HOOK'
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export CUDA_TARGET_LIB="$CONDA_PREFIX/targets/x86_64-linux/lib"
_cuda_conda_lib="$CONDA_PREFIX/lib"
_cuda_conda_lib64="$CONDA_PREFIX/lib64"
_cuda_ld_path=":${LD_LIBRARY_PATH:-}:"
_cuda_ld_path="${_cuda_ld_path//:$_cuda_conda_lib:/:}"
_cuda_ld_path="${_cuda_ld_path//:$_cuda_conda_lib64:/:}"
_cuda_ld_path="${_cuda_ld_path#:}"
_cuda_ld_path="${_cuda_ld_path%:}"
export LD_LIBRARY_PATH="$CUDA_TARGET_LIB${_cuda_ld_path:+:$_cuda_ld_path}"
export LIBRARY_PATH="$CUDA_TARGET_LIB:${LIBRARY_PATH:-}"
unset _cuda_ld_path _cuda_conda_lib _cuda_conda_lib64
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:${CPATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
HOOK

log_info "Verifying CUDA 12.6 compiler from the active conda environment..."
source "$CONDA_PREFIX/etc/conda/activate.d/cuda126.sh"

which nvcc
nvcc --version

log_info "CUDA 12.6 toolkit installation finished."
log_info "Future install.sh stages will load CUDA 12.6 variables automatically."
ENV_NAME="${CONDA_PREFIX##*/}"
log_info "For interactive nvcc in this terminal, run: conda activate $ENV_NAME"
