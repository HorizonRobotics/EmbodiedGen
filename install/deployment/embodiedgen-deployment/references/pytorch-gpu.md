# PyTorch and GPU Compatibility

## 1. 5090 / sm_120: PyTorch no kernel

> ⚠️ **RTX 5090/Blackwell (sm_120) only**. 4090 (sm_89) does NOT trigger this.

**Symptom**: `torch.cuda.is_available()=True` but matmul fails with `CUDA error: no kernel image is available for execution on the device`.

**Root cause**: A cu126 wheel or extension compiled without `sm_120` replaced
the expected cu128 build.

**Fix**: install the repository's cu128 environment, reactivate it, then run
the basic installer:

```bash
bash install.sh cu128
conda deactivate && conda activate <env-name>
bash install.sh basic
```

For source-built extensions, verify `TORCH_CUDA_ARCH_LIST=12.0`,
`TCNN_CUDA_ARCHITECTURES=120`, and nvcc 12.8 or newer.

---

## 2. 5090 / sm_120: xformers flash-attn crash

> ⚠️ **RTX 5090/Blackwell (sm_120) only**.

**Symptom**: `invalid argument` + GIL segfault.

**Root cause**: xformers FlashAttention 3 can select an incompatible path on
Blackwell.

Current `master` disables xformers FlashAttention 3 on Blackwell and keeps the
xformers FlashAttention 2 fallback. Update the checkout before adding manual
environment overrides. As a diagnostic fallback only:

```bash
export XFORMERS_DISABLED=1
```

Verify with `img3d-cli --help` and a small GPU inference.
