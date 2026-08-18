# RTX 5090 Docker Image

This image builds the EmbodiedGen `basic` stage for RTX 5090 / Blackwell
(`sm_120`) with PyTorch 2.8 and CUDA 12.8. Model weights are downloaded at
runtime and are not included in the image.

## Prerequisites

- Linux with an RTX 5090 and a CUDA 12.8-compatible NVIDIA driver.
- Docker with NVIDIA Container Toolkit.
- Access to PyPI and GitHub while building, and Hugging Face at runtime.

## Build

`.git` is excluded from the build context, so initialize the basic-stage
submodules on the host first:

```bash
git clone <repository-url> EmbodiedGen
cd EmbodiedGen
git submodule update --init --recursive --progress \
    thirdparty/TRELLIS thirdparty/sam3d
docker build -t embodiedgen:cu128-sm120 -f docker/5090/Dockerfile .
```

Lower `MAX_JOBS` in the Dockerfile if CUDA extension builds exhaust memory.

## Verify

```bash
docker run --rm embodiedgen:cu128-sm120 \
    conda run -n embodiedgen python -c "import embodied_gen, gsplat, kaolin, pytorch3d; print('imports OK')"
docker run --rm embodiedgen:cu128-sm120 \
    conda run -n embodiedgen img3d-cli --help
docker run --rm --gpus all embodiedgen:cu128-sm120 \
    conda run -n embodiedgen python -c "import torch; x=torch.randn(1024,1024,device='cuda:0'); print('matmul', (x@x).sum().item())"
```

The last command must print a numeric result. A `no kernel image` error usually
means a cu126 PyTorch wheel replaced the cu128 wheel.

## Run

```bash
docker run --gpus all --shm-size=64g -it --rm \
    -v embodiedgen_hf_cache:/hf_cache \
    -v "$PWD":/workspace \
    embodiedgen:cu128-sm120 bash
```

Inside the container:

```bash
conda activate embodiedgen
img3d-cli --image_path <image> --output_root /workspace/outputs
```

## Notes

- `EMBODIEDGEN_TORCH_INDEX_URL` keeps newer installers on the cu128 index.
- For RTX 4090, A100, or H100, use `docker/Dockerfile` instead.
