#!/bin/bash
set -e

STAGE=$1 # "basic" | "scene3d" | "room" | "affordance" | "cu126" | "all"
STAGE=${STAGE:-basic}

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$REPO_ROOT/install/_utils.sh"
git config --global http.postBuffer 524288000

log_info "===== Starting installation stage: $STAGE ====="

if [[ "$STAGE" == "basic" || "$STAGE" == "all" ]]; then
    bash "$REPO_ROOT/install/install_basic.sh"
fi

if [[ "$STAGE" == "scene3d" || "$STAGE" == "all" ]]; then
    PANO2ROOM_PATH="$REPO_ROOT/thirdparty/pano2room"
    if [ -d "$PANO2ROOM_PATH" ]; then
        echo "__pycache__/" > "$PANO2ROOM_PATH/.gitignore"
        log_info "Added .gitignore to ignore __pycache__ in $PANO2ROOM_PATH"
    fi
    bash "$REPO_ROOT/install/install_scene3d.sh"
fi

if [[ "$STAGE" == "room" || "$STAGE" == "all" ]]; then
    bash "$REPO_ROOT/install/install_room.sh"
fi

if [[ "$STAGE" == "affordance" || "$STAGE" == "all" ]]; then
    bash "$REPO_ROOT/install/install_affordance.sh"
fi

if [[ "$STAGE" == "cu126" ]]; then
    bash "$REPO_ROOT/install/install_cu126.sh"
fi

# Global constraints for all stages
pip install numpy==1.26.4
log_info "===== Installation completed successfully. ====="
