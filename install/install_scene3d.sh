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

log_info "Preparing tiny-cuda-nn runtime compilation headers..."
python - <<'PY'
import os
import shutil
import sys
from pathlib import Path

import tinycudann
import torch

site_packages = Path(tinycudann.__file__).resolve().parents[1]
roots = [
    Path(value)
    for variable in ("CUDA_HOME", "CUDA_PATH")
    if (value := os.environ.get(variable))
]
if nvcc := shutil.which("nvcc"):
    roots.append(Path(nvcc).resolve().parent.parent)
roots.extend(
    [
        Path(sys.prefix),
        site_packages / "nvidia" / "cuda_runtime",
        Path("/usr/local/cuda"),
    ]
)
source_dir = next(
    (
        candidate
        for root in roots
        for candidate in (
            root / "include",
            root / "targets" / "x86_64-linux" / "include",
        )
        if (candidate / "cuda_fp16.h").is_file()
    ),
    None,
)
blackwell_visible = torch.cuda.is_available() and any(
    torch.cuda.get_device_capability(index) >= (12, 0)
    for index in range(torch.cuda.device_count())
)

if source_dir is None:
    message = "Could not locate CUDA headers for tiny-cuda-nn JIT support."
    if blackwell_visible:
        raise RuntimeError(message)
    print(f"[WARNING] {message}")
    raise SystemExit(0)

target_dir = Path(tinycudann.__file__).resolve().parent / "rtc" / "include"
target_dir.mkdir(parents=True, exist_ok=True)
for pattern in ("cuda_fp16*", "vector*"):
    for header in source_dir.glob(pattern):
        shutil.copy2(header, target_dir / header.name)

print(f"Prepared tiny-cuda-nn RTC headers from {source_dir}")
PY
