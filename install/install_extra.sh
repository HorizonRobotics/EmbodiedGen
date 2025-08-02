#!/bin/bash
set -e
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/_utils.sh"

PYTHON_PACKAGES_NODEPS=(
    "timm"
    "txt2panoimg@git+https://github.com/HochCC/SD-T2I-360PanoImage"
)

PYTHON_PACKAGES=(
    "fused-ssim@git+https://github.com/rahul-goel/fused-ssim#egg=328dc98"
    "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    "git+https://github.com/facebookresearch/pytorch3d.git@stable"
    "kornia"
    "h5py"
    "albumentations==0.5.2"
    "webdataset"
    "icecream"
    "open3d"
    "pyequilib"
    "numpy==1.26.4"
    "triton==2.1.0"
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
