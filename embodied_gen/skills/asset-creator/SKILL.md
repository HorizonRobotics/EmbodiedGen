---
name: asset-creator
description: Create 3D assets with EmbodiedGen using img3d-cli, text3d-cli, and texture-cli. Use this skill whenever users ask to generate assets from images/text, texture existing meshes, run retry/seed controlled generation, or choose the proper asset-generation CLI from mixed requirements.
---

# Assets Creator

Unified entry for three EmbodiedGen asset-generation CLIs: `img3d-cli`, `text3d-cli`, and `texture-cli`.

## When To Use

Use this skill when users want to:
- Generate a 3D asset from one or more input images.
- Generate 3D assets in batch from text prompts.
- Generate or edit textures for existing meshes.
- Get help choosing the correct CLI from mixed asset-generation requirements.

## Routing Rules (Core)

1. `img3d-cli`: input is image paths (`--image_path` or `--image_root`).
2. `text3d-cli`: input is text prompts (`--prompts`) and target is direct asset output.
3. `texture-cli`: input is existing mesh path(s) (`--mesh_path`) plus texture prompt(s) (`--prompt`).

## Pre-checks

1. Run commands from the repository root.
2. Confirm the active environment is `embodiedgen`.
3. If CLI commands are unavailable, run `pip install -e .` to register entrypoints.

## Standard Command Templates

### 1) Image to 3D: `img3d-cli`

```bash
img3d-cli --image_path .../sample.jpg --n_retry 1 --output_root outputs/imageto3d
```

Common parameters:
- `--image_path` / `--image_root`
- `--output_root`
- `--n_retry`
- `--seed`
- `--skip_exists`

---

### 2) Text to 3D: `text3d-cli`

```bash
text3d-cli \
  --prompts "small bronze figurine of a lion" "A globe with wooden base" \
  --n_image_retry 1 --n_asset_retry 1 --n_pipe_retry 1 \
  --seed_img 0 \
  --output_root outputs/textto3d
```

Common parameters:
- `--prompts`
- `--output_root`
- `--asset_names`
- `--n_image_retry --n_asset_retry --n_pipe_retry`
- `--seed_img --seed_3d`

---

### 3) Mesh Texture Generation: `texture-cli`

```bash
texture-cli \
  --mesh_path ".../horse.obj" \
  --prompt "A gray horse head with flying mane and brown eyes" \
  --output_root "outputs/texture_gen" \
  --seed 0
```

Common parameters:
- `--mesh_path` (supports multiple inputs)
- `--prompt` (must align 1:1 with mesh inputs)
- `--output_root`
- `--seed`
- `--texture_size`
- `--ip_adapt_scale --ip_img_path` (optional reference-image control)

---

## Output Conventions

- `img3d-cli`: each sample is typically under `<output_root>/<sample>/result/`.
- `text3d-cli`: `<output_root>/asset3d/<asset_name>/result/`.
- `texture-cli`: `<output_root>/<mesh_stem>/texture_mesh/`.

## Failure Handling and Retry

1. OOM or GPU pressure: reduce batch size and concurrency.
2. Unstable quality: increase `--n_retry` or `--n_*_retry`.
3. Missing outputs: verify output-root permissions and path spelling; prefer absolute paths.
