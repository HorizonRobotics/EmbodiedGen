# EmbodiedGen on AMD ROCm (MI300X / MI308X)

Run the EmbodiedGen **image-to-3D generation** stack on AMD GPUs (gfx942) with
ROCm 6.4.3 + PyTorch 2.6, by swapping the CUDA-only libraries for verified ROCm
equivalents. Verified on **AMD Instinct MI308X**: the SAM3D and TRELLIS pipelines
import and initialize (spconv backend + flash-attn; SAM3D attention auto-selects
`sdpa`).

> Library swaps follow the `rocm-lib-compat` reference
> ([ZJLi2013/rocm3d](https://github.com/ZJLi2013/rocm3d)). The same TRELLIS-v1
> stack is independently verified there via `VAST-AI/AniGen`.

## TL;DR

```bash
# from repo root, with submodules checked out:
git submodule update --init --recursive
docker build -f docker/amd_rocm/Dockerfile -t embodiedgen:rocm6.4.3 .

# import+init smoke (no download / no GPT):
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  --shm-size 32g -e FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
  embodiedgen:rocm6.4.3

# full GPT-free image->3D (downloads facebook/sam-3d-objects, ~15GB; saves splat.ply):
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
  --shm-size 32g -e FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
  -v $PWD:/workspace/EmbodiedGen embodiedgen:rocm6.4.3 \
  python -m embodied_gen.models.sam3d
```

To run the swaps directly in a base container instead of building the image:

```bash
docker run -it --device=/dev/kfd --device=/dev/dri --group-add video --shm-size 32g \
  -v $PWD:/workspace/EmbodiedGen -w /workspace/EmbodiedGen \
  rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0 \
  bash docker/amd_rocm/install_rocm.sh
```

## CUDA -> ROCm dependency map

| Upstream (CUDA) | ROCm replacement | Status on MI308X |
|---|---|---|
| `spconv-cu120` | [`ZJLi2013/spconv_rocm`](https://github.com/ZJLi2013/spconv_rocm) (source) | ✅ import OK |
| `nvdiffrast` | [`ZJLi2013/nvdiffrast@rocm`](https://github.com/ZJLi2013/nvdiffrast) | ✅ import OK |
| `gsplat` | `amd_gsplat` (`pypi.amd.com/rocm-6.4.3`), import name `gsplat` | ✅ default GS backend |
| `pytorch3d` | ROCm 6.4 / py3.12 prebuilt wheel | ✅ import OK |
| `flash-attn` | FA2-Triton (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` at install **and** runtime) | ✅ import OK |
| `xformers` | not needed — SAM3D attention auto-selects `sdpa` | ✅ skipped |
| `numpy` (base ships 2.x) | pin `numpy<2` (diffusers/transformers requirement) | ✅ |
| `kaolin` (no ROCm wheel; setup.py requires `nvcc`) | `sitecustomize` stub (`docker/amd_rocm/kaolin_stub.py`) | ⚠️ texture-stage only |
| `diff-gaussian-rasterization` | optional 'inria' GS backend (gsplat is default) | ⏸ optional |

## The kaolin stub (`docker/amd_rocm/kaolin_stub.py`)

`kaolin` is CUDA-only and is imported at the top of `embodied_gen/data/utils.py`,
but is only **used** inside the texture-backprojection / mesh-IO stage
(`kal.io.*.import_mesh`, `kal.render.materials.PBRMaterial`,
`kaolin.render.camera.Camera`) and as type references in `thirdparty/sam3d`.
None of it is on the core geometry+gaussian generation path. The stub (installed
as `sitecustomize.py`) fabricates any `kaolin.*` module so every `import kaolin`
resolves; the texture stage raises a clear error if actually invoked. This mirrors
the proven `ZJLi2013/RealWonder` bypass (~85% pipeline usable on ROCm).

The upstream-friendly long-term fix is to make the kaolin imports in
`data/utils.py` lazy/optional so the stub is unnecessary.

## Known gaps

- **Texture backprojection** (`backproject_v3` / `differentiable_render`) calls
  real kaolin mesh-IO and is not available under the stub. Core image-to-3D
  (segmentation -> SAM3D geometry + gaussian + mesh export) runs without it.
- **GPT quality-checkers / URDF semantics** (`img3d-cli`) need a GPT key; the
  `python -m embodied_gen.models.sam3d` path skips them entirely.
- **`diff-gaussian-rasterization`** (mip-splatting / antialiasing fork) needs
  `__trap`->`abort` and a `cooperative_groups/reduce.h` shim to build on ROCm;
  it is optional because `gsplat` is the default gaussian backend.
