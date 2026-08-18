---
name: affordance-deployment
description: "Deploy and troubleshoot the EmbodiedGen affordance pipeline. Use when bash install.sh affordance, pointnet2_ops, torch-scatter, or flash-attn fails on RTX 4090/5090, or when affordance-cli cannot import."
---

# Affordance Deployment

Use the repository installer first. It initializes both submodules, builds
`torch-scatter`, `flash-attn==2.8.2`, and GraspGen's `pointnet2_ops`, then
restores the supported OpenCV versions.

```bash
bash install.sh affordance
affordance-cli --help
img3d-cli --help
```

## CUDA Architecture

Load the CUDA hook before rebuilding extensions and verify the selected target:

```bash
echo "$CUDA_HOME"
echo "$TORCH_CUDA_ARCH_LIST"
nvcc --version
```

- RTX 4090: `TORCH_CUDA_ARCH_LIST=8.9`.
- RTX 5090: `TORCH_CUDA_ARCH_LIST=12.0` and nvcc 12.8 or newer.

If `pointnet2_ops` was built for the wrong architecture, rebuild the package
directly instead of running `thirdparty/GraspGen/install_pointnet.sh`, which
hardcodes an older target:

```bash
python -m pip install --no-build-isolation --force-reinstall --no-deps \
  thirdparty/GraspGen/pointnet2_ops
```

## Flash Attention

Keep `flash-attn` at 2.8.2 because the installed diffusers version rejects
newer releases. Rebuild from source with the GPU-specific target when needed:

```bash
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTN_CUDA_ARCHS=89   # use 120 on RTX 5090 with nvcc >= 12.8
python -m pip install --no-cache-dir --no-build-isolation \
  --no-binary flash-attn --force-reinstall --no-deps flash-attn==2.8.2
```

If the source checkout lacks CUTLASS or composable-kernel files, initialize its
submodules before retrying. Reduce `MAX_JOBS` when compilation runs out of RAM.

## Quick Reference

| Error | Root Cause | Fix |
|---|---|---|
| `Unsupported gpu architecture 'compute_120'` | nvcc is older than 12.8 | Use the cu128 hook or target the actual GPU |
| `no kernel image` in pointnet2 | Extension built for another GPU | Rebuild `pointnet2_ops` directly |
| diffusers rejects flash-attn | Version is newer than 2.8.2 | Reinstall 2.8.2 |
| Build OOM | Too many compiler jobs | Set `MAX_JOBS=4` |
