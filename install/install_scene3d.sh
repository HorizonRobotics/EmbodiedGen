#!/bin/bash
set -e
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/_utils.sh"

PYTHON_PACKAGES_NODEPS=(
    "txt2panoimg@git+https://github.com/HochCC/SD-T2I-360PanoImage"
)

PYTHON_PACKAGES=(
    "fused-ssim@git+https://github.com/rahul-goel/fused-ssim#egg=328dc98 --no-build-isolation"
    "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    "kornia"
    "h5py"
    "albumentations==0.5.2"
    "webdataset"
    "icecream"
    "pyequilib"
)

for pkg in "${PYTHON_PACKAGES_NODEPS[@]}"; do
    try_install "Installing $pkg without dependencies..." \
        "pip install --no-deps $pkg" \
        "$pkg installation failed."
done

for pkg in "${PYTHON_PACKAGES[@]}"; do
    try_install "pip install $pkg..." \
        "pip install $pkg" \
        "$pkg installation failed."
done
