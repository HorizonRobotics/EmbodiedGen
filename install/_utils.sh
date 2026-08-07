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

    cuda_hook="$CONDA_PREFIX/etc/conda/activate.d/${cuda_variant}.sh"
    if [[ -f "$cuda_hook" ]]; then
        source "$cuda_hook"
    fi
}
