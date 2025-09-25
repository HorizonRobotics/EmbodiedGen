#!/bin/bash
set -e

STAGE=$1 # "basic" | "extra" | "all"
STAGE=${STAGE:-all}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

source "$SCRIPT_DIR/install/_utils.sh"
git config --global http.postBuffer 524288000

log_info "===== Starting installation stage: $STAGE ====="

if [[ "$STAGE" == "basic" || "$STAGE" == "all" ]]; then
    bash "$SCRIPT_DIR/install/install_basic.sh"
fi

if [[ "$STAGE" == "extra" || "$STAGE" == "all" ]]; then
    # Patch submodule .gitignore to ignore __pycache__, if submodule exists
    PANO2ROOM_PATH="$SCRIPT_DIR/thirdparty/pano2room"
    if [ -d "$PANO2ROOM_PATH" ]; then
        echo "__pycache__/" > "$PANO2ROOM_PATH/.gitignore"
        log_info "Added .gitignore to ignore __pycache__ in $PANO2ROOM_PATH"
    fi

    bash "$SCRIPT_DIR/install/install_extra.sh"
fi

pip install triton==2.1.0 numpy==1.26.4

log_info "===== Installation completed successfully. ====="
