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
CONDA_CHANNEL_ARGS=()
if [[ -n "${EMBODIEDGEN_CUDA_CHANNEL:-}" ]]; then
    CONDA_CHANNEL_ARGS=(
        --override-channels
        -c "$EMBODIEDGEN_CUDA_CHANNEL"
    )
    log_info "Using CUDA conda channel: $EMBODIEDGEN_CUDA_CHANNEL"
fi

"$CONDA_CMD" install \
    -p "$CONDA_PREFIX" \
    "${CONDA_CHANNEL_ARGS[@]}" \
    cuda-toolkit=12.6 \
    cuda-nvcc=12.6 \
    -y

log_info "Writing CUDA 12.6 activation hook into the conda environment..."
prepare_cxx_runtime_libs
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
rm -f "$CONDA_PREFIX/etc/conda/activate.d/cuda128.sh"
cat > "$CONDA_PREFIX/etc/conda/activate.d/cuda126.sh" <<'HOOK'
export EMBODIEDGEN_CUDA_VARIANT="cu126"
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export CUDA_TARGET_LIB="$CONDA_PREFIX/targets/x86_64-linux/lib"
export EMBODIEDGEN_RUNTIME_LIB="$CONDA_PREFIX/lib/embodiedgen-runtime"
_cuda_conda_lib="$CONDA_PREFIX/lib"
_cuda_conda_lib64="$CONDA_PREFIX/lib64"
_cuda_target_include="$CONDA_PREFIX/targets/x86_64-linux/include"
_cuda_ld_path=":${LD_LIBRARY_PATH:-}:"
_cuda_ld_path="${_cuda_ld_path//:$EMBODIEDGEN_RUNTIME_LIB:/:}"
_cuda_ld_path="${_cuda_ld_path//:$CUDA_TARGET_LIB:/:}"
_cuda_ld_path="${_cuda_ld_path//:$_cuda_conda_lib:/:}"
_cuda_ld_path="${_cuda_ld_path//:$_cuda_conda_lib64:/:}"
_cuda_ld_path="${_cuda_ld_path#:}"
_cuda_ld_path="${_cuda_ld_path%:}"
export LD_LIBRARY_PATH="$EMBODIEDGEN_RUNTIME_LIB:$CUDA_TARGET_LIB${_cuda_ld_path:+:$_cuda_ld_path}"
_cuda_library_path=":${LIBRARY_PATH:-}:"
_cuda_library_path="${_cuda_library_path//:$CUDA_TARGET_LIB:/:}"
_cuda_library_path="${_cuda_library_path#:}"
_cuda_library_path="${_cuda_library_path%:}"
export LIBRARY_PATH="$CUDA_TARGET_LIB${_cuda_library_path:+:$_cuda_library_path}"
_cuda_cpath=":${CPATH:-}:"
_cuda_cpath="${_cuda_cpath//:$_cuda_target_include:/:}"
_cuda_cpath="${_cuda_cpath#:}"
_cuda_cpath="${_cuda_cpath%:}"
export CPATH="$_cuda_target_include${_cuda_cpath:+:$_cuda_cpath}"
unset _cuda_conda_lib _cuda_conda_lib64 _cuda_target_include
unset _cuda_ld_path _cuda_library_path _cuda_cpath
export TORCH_CUDA_ARCH_LIST="${EMBODIEDGEN_TORCH_CUDA_ARCH_LIST:-8.9}"
HOOK
write_cuda_deactivation_hook "cu126"

log_info "Verifying CUDA 12.6 compiler from the active conda environment..."
source "$CONDA_PREFIX/etc/conda/activate.d/cuda126.sh"

which nvcc
nvcc --version

log_info "CUDA 12.6 toolkit installation finished."
log_info "Future install.sh stages will load CUDA 12.6 variables automatically."
ENV_NAME="${CONDA_PREFIX##*/}"
log_info "To load CUDA 12.6 in the current shell, run: conda deactivate && conda activate $ENV_NAME"
