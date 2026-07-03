---
name: background-creator
description: Generate background 3D scenes with EmbodiedGen using scene3d-cli. Use this skill whenever users ask to create room/indoor background scenes from text prompts, pre-generate backgrounds for layout-cli, or control scene3d generation quality/runtime with retry, seed, and gs3d settings.
---

# Background Creator

Unified entry for EmbodiedGen background scene generation via `scene3d-cli`.

## When To Use

Use this skill when users want to:
- Generate indoor/background 3D scenes from text prompts.
- Pre-generate scene assets for `layout-cli`.
- Control `scene3d-cli` runtime/quality via seed, retry, and `gs3d` settings.

## Routing Rule (Core)

Use `scene3d-cli` when input is scene-level text prompts and output target is a background scene (mesh + 3DGS), not single foreground assets.

## Pre-checks

1. Run commands from the repository root.
2. Confirm the active environment is `embodiedgen`.
3. Install scene3d dependencies first if needed:
   `bash install.sh scene3d`
4. If CLI commands are unavailable, run `pip install -e .` to register entrypoints.

## Standard Command Template

```bash
scene3d-cli --prompts "Art studio with easel and canvas" \
  --output_dir outputs/bg_scenes \
  --seed 0 \
  --gs3d.max_steps 4000 \
  --disable_pano_check
```

## Common Parameters

- `--prompts`: one or more scene text prompts.
- `--output_dir`: output root directory for generated scenes.
- `--seed`: random seed for reproducibility.
- `--n_retry`: panorama generation retries.
- `--real_height`: force target real-world room height in meters.
- `--pano_image_only`: generate only panorama image (debug/fast validation).
- `--disable_pano_check`: skip panorama quality check.
- `--keep_middle_result`: keep intermediate training artifacts.
- `--gs3d.max_steps`: training steps for 3DGS optimization.

## Output Conventions

Each prompt is saved under `<output_dir>/scene_xxxx/`, typically including:
- `gs_model.ply`
- `mesh_model.ply`
- `pano_image.png`
- `prompt.txt`
- `video.mp4`
- `gsplat_cfg.yml`

## Runtime Expectations

- Typical full generation time is around 30 minutes per scene.
- Use `--pano_image_only` for quick prompt validation before full generation.

## Failure Handling and Retry

1. OOM or GPU pressure: reduce concurrency and lower `--gs3d.max_steps`.
2. Unstable scene quality: increase `--n_retry` or adjust prompt specificity.
3. Missing outputs: verify `--output_dir` permissions and use absolute paths.
