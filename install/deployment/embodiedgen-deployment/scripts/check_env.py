#!/usr/bin/env python3
"""Run the minimum EmbodiedGen environment checks."""

import os
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

if sys.version_info < (3, 10):
    print("[FATAL] EmbodiedGen requires Python >= 3.10", file=sys.stderr)
    raise SystemExit(2)

warnings.filterwarnings("ignore", category=FutureWarning)

LEVELS = {"OK": 0, "WARN": 1, "FATAL": 2}
status = 0


def report(level: str, message: str) -> None:
    """Print a check result and retain the highest severity."""
    global status
    status = max(status, LEVELS[level])
    print(f"[{level}] {message}")


def load_module(name: str):
    """Import a required module without aborting later checks."""
    try:
        return __import__(name)
    except Exception as error:
        report("FATAL", f"{name} import failed: {error}")
        return None


conda_prefix = os.getenv("CONDA_PREFIX")
if conda_prefix:
    report("OK", f"Conda env: {os.getenv('CONDA_DEFAULT_ENV', 'unknown')}")
else:
    report("WARN", "No active conda environment")

user_paths = [path for path in sys.path if "/.local/" in path]
if user_paths:
    report("WARN", "User site is enabled; set PYTHONNOUSERSITE=1")

numpy = load_module("numpy")
torch = load_module("torch")
for name, module in (("numpy", numpy), ("torch", torch)):
    if module is None:
        continue
    module_path = str(getattr(module, "__file__", "unknown"))
    level = "FATAL" if "/.local/" in module_path else "OK"
    report(level, f"{name}={module.__version__}: {module_path}")

if numpy is not None and numpy.__version__ != "1.26.4":
    level = "FATAL" if numpy.__version__.startswith("2.") else "WARN"
    report(level, f"Expected numpy=1.26.4, found {numpy.__version__}")

blackwell_visible = False
if torch is None:
    report("FATAL", "GPU check skipped because torch is unavailable")
elif not torch.cuda.is_available():
    report("FATAL", "torch.cuda.is_available() is False")
else:
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    blackwell_visible = (properties.major, properties.minor) >= (12, 0)
    report(
        "OK",
        f"GPU {device}: {properties.name} (sm_{properties.major}{properties.minor}), "
        f"torch CUDA {torch.version.cuda}",
    )
    try:
        tensor = torch.randn(1024, 1024, device=f"cuda:{device}")
        torch.matmul(tensor, tensor).sum().item()
        report("OK", "GPU matmul succeeded")
    except RuntimeError as error:
        report("FATAL", str(error).splitlines()[0])

nvcc_path = shutil.which("nvcc")
cuda_home = os.getenv("CUDA_HOME")
if nvcc_path is None and cuda_home:
    candidate = Path(cuda_home) / "bin" / "nvcc"
    if candidate.is_file():
        nvcc_path = str(candidate)

nvcc_version = None
if nvcc_path:
    output = subprocess.run(
        [nvcc_path, "--version"], capture_output=True, text=True
    ).stdout
    match = re.search(r"release ([0-9.]+)", output)
    if match:
        nvcc_version = tuple(int(part) for part in match.group(1).split("."))
        report("OK", f"nvcc={match.group(1)}: {nvcc_path}")
    else:
        report("WARN", f"Could not determine nvcc version: {nvcc_path}")
else:
    report("WARN", "nvcc not found")

if blackwell_visible and (nvcc_version is None or nvcc_version < (12, 8)):
    report("FATAL", "Blackwell requires nvcc >= 12.8")

result = ("All checks passed", "Warnings found", "Blocking failures found")
level = ("OK", "WARN", "FATAL")[status]
print(f"[{level}] {result[status]}")
raise SystemExit(status)
