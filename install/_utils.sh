#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO] $1${NC}"
}

log_error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
}

try_install() {
    log_info "$1"
    eval "$2" || {
        log_error "$3"
        exit 1
    }
}

prepare_cxx_runtime_libs() {
    local runtime_dir="$CONDA_PREFIX/lib/embodiedgen-runtime"
    local runtime_lib

    mkdir -p "$runtime_dir"
    for runtime_lib in libstdc++.so.6 libgcc_s.so.1; do
        if [[ ! -e "$CONDA_PREFIX/lib/$runtime_lib" ]]; then
            log_error "Missing required C++ runtime library: $CONDA_PREFIX/lib/$runtime_lib"
            return 1
        fi
        ln -sfn "../$runtime_lib" "$runtime_dir/$runtime_lib"
    done
}

write_cuda_deactivation_hook() {
    local cuda_variant="$1"
    local deactivate_dir="$CONDA_PREFIX/etc/conda/deactivate.d"
    local deactivate_hook="$deactivate_dir/cuda${cuda_variant#cu}.sh"

    mkdir -p "$deactivate_dir"
    rm -f "$deactivate_dir/cuda126.sh" "$deactivate_dir/cuda128.sh"
    cat > "$deactivate_hook" <<'HOOK'
_embodiedgen_remove_path() {
    local variable_name="$1"
    local path_to_remove="$2"
    local current_value

    if [[ -v "$variable_name" ]]; then
        current_value=":${!variable_name}:"
    else
        current_value=":"
    fi
    current_value="${current_value//:$path_to_remove:/:}"
    current_value="${current_value#:}"
    current_value="${current_value%:}"
    if [[ -n "$current_value" ]]; then
        export "$variable_name=$current_value"
    else
        unset "$variable_name"
    fi
}

_embodiedgen_remove_path LD_LIBRARY_PATH "$CONDA_PREFIX/lib/embodiedgen-runtime"
_embodiedgen_remove_path LD_LIBRARY_PATH "$CONDA_PREFIX/targets/x86_64-linux/lib"
_embodiedgen_remove_path LIBRARY_PATH "$CONDA_PREFIX/targets/x86_64-linux/lib"
_embodiedgen_remove_path CPATH "$CONDA_PREFIX/targets/x86_64-linux/include"
unset -f _embodiedgen_remove_path
unset EMBODIEDGEN_CUDA_VARIANT EMBODIEDGEN_RUNTIME_LIB
unset CUDA_HOME CUDA_PATH CUDA_TARGET_LIB
unset TORCH_CUDA_ARCH_LIST TCNN_CUDA_ARCHITECTURES
HOOK
}

detect_cuda_variant() {
    local cuda_variant="cu126"

    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        if [[ -f "$CONDA_PREFIX/etc/conda/activate.d/cuda128.sh" ]]; then
            cuda_variant="cu128"
        elif [[ -f "$CONDA_PREFIX/etc/conda/activate.d/cuda126.sh" ]]; then
            cuda_variant="cu126"
        fi
    fi

    printf '%s\n' "$cuda_variant"
}

source_cuda_activation() {
    local cuda_variant
    local cuda_hook

    cuda_variant=$(detect_cuda_variant) || return 1
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        return 0
    fi

    cuda_hook="$CONDA_PREFIX/etc/conda/activate.d/cuda${cuda_variant#cu}.sh"
    if [[ -f "$cuda_hook" ]]; then
        source "$cuda_hook"
    fi
}
