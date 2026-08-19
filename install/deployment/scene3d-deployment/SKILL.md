---
name: scene3d-deployment
description: "Deploy and troubleshoot the EmbodiedGen scene3d pipeline. Use when bash install.sh scene3d, tiny-cuda-nn, fused-ssim, txt2panoimg, or scene3d-cli fails on RTX 4090/5090."
---

# Scene3D Deployment

Use the repository installer first. It initializes `thirdparty/pano2room`,
installs `txt2panoimg` without dependencies, and builds fused-ssim and
tiny-cuda-nn.

```bash
bash install.sh scene3d
scene3d-cli --help
```

## tiny-cuda-nn

tiny-cuda-nn reads `TCNN_CUDA_ARCHITECTURES`, not
`TORCH_CUDA_ARCH_LIST`.

- RTX 4090: `TCNN_CUDA_ARCHITECTURES=89`.
- RTX 5090: `TCNN_CUDA_ARCHITECTURES=120` with nvcc 12.8 or newer.

If a VCS install reports missing `fmt`, CUTLASS, or CMakeRC files, clone the
source recursively and install its torch binding:

```bash
git clone --recursive https://github.com/NVlabs/tiny-cuda-nn /tmp/tiny-cuda-nn
python -m pip install --no-build-isolation \
  /tmp/tiny-cuda-nn/bindings/torch
```

Reduce `MAX_JOBS` if compilation runs out of memory.

## basicsr and txt2panoimg

Keep torch 2.8 with torchvision 0.23. The installer intentionally uses
`--no-deps` for `txt2panoimg`, and `gen_scene3d.py` applies the compatibility
patch before importing basicsr. Do not downgrade torchvision to fix
`torchvision.transforms.functional_tensor`.

## Blackwell double backward

On RTX 5090, tiny-cuda-nn's eager HashGrid double backward can fail with
`CUDA error: an illegal memory access was encountered` during Pano2Mesh depth
optimization. The scene3d installer prepares tiny-cuda-nn's runtime CUDA
headers, and EmbodiedGen enables its exact JIT gradient path only on Blackwell.
Rerun `bash install.sh scene3d` after updating an existing checkout.

## Quick Reference

| Error | Root Cause | Fix |
|---|---|---|
| `Could NOT find fmt` or CUTLASS | tiny-cuda-nn submodules missing | Use a recursive clone |
| `functional_tensor` missing | txt2panoimg pulled incompatible deps | Reinstall through `install.sh scene3d` |
| `compute_120` unsupported | nvcc older than 12.8 | Load the cu128 hook |
| HashGrid backward illegal memory access | Eager double backward on Blackwell | Reinstall scene3d to enable exact JIT gradients |
| Build OOM | Too many compiler jobs | Set `MAX_JOBS=4` |
