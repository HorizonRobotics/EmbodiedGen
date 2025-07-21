#!/bin/bash
set -e
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/_utils.sh"

# try_install "Installing txt2panoimg..." \
#     "pip install txt2panoimg@git+https://github.com/HochCC/SD-T2I-360PanoImage --no-deps" \
#     "txt2panoimg installation failed."

# try_install "Installing fused-ssim..." \
#     "pip install fused-ssim@git+https://github.com/rahul-goel/fused-ssim#egg=328dc98" \
#     "fused-ssim installation failed."

# try_install "Installing tiny-cuda-nn..." \
#     "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch" \
#     "tiny-cuda-nn installation failed."

# try_install "Installing pytorch3d" \
#     "pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7" \
#     "pytorch3d installation failed."


PYTHON_PACKAGES_NODEPS=(
    timm
    txt2panoimg@git+https://github.com/HochCC/SD-T2I-360PanoImage
    kornia
    kornia_rs
)

PYTHON_PACKAGES=(
    fused-ssim@git+https://github.com/rahul-goel/fused-ssim#egg=328dc98
    git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
    git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7
    h5py
    albumentations==0.5.2
    webdataset
    icecream
    open3d
    pyequilib
    numpy==1.26.4
    triton==2.1.0
)

for pkg in "${PYTHON_PACKAGES_NODEPS[@]}"; do
    try_install "Installing $pkg without dependencies..." \
        "pip install --no-deps $pkg" \
        "$pkg installation failed."
done

try_install "Installing other Python dependencies..." \
    "pip install ${PYTHON_PACKAGES[*]}" \
    "Python dependencies installation failed."
