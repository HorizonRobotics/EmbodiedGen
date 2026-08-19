---
name: embodiedgen-deployment
description: "Diagnose EmbodiedGen installation failures on shared machines and RTX 4090/5090 systems. Use for broken conda environments, user-site package leakage, missing build tools, CUDA/PyTorch architecture mismatches, or xformers failures."
---

# EmbodiedGen Deployment

Run the health check before changing the environment:

```bash
conda activate <env-name>
python install/deployment/embodiedgen-deployment/scripts/check_env.py
```

Exit codes: `0` is healthy, `1` means warnings, and `2` means a blocking
failure.

## Error Lookup

| Error / Symptom | Reference |
|---|---|
| Broken conda shebang, `~/.local` leakage, missing build command | [env-conda.md](references/env-conda.md) |
| `no kernel image`, CUDA target, xformers failure | [pytorch-gpu.md](references/pytorch-gpu.md) |
