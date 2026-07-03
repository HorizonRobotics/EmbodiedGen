---
name: layout-creator
description: Generate interactive 3D layouts from task descriptions with EmbodiedGen using layout-cli. Use this skill whenever users ask to build task-driven 3D scenes, batch-generate layouts from task files, tune layout generation retries/seeds, or produce simulator-ready layout outputs from background scene lists.
---

# Layout Creator

Unified entry for EmbodiedGen interactive layout generation via `layout-cli`.

## When To Use

Use this skill when users want to:
- Generate interactive 3D scenes from task descriptions.
- Batch-generate layouts from a task list file.
- Build simulator-ready layout outputs (`layout.json`, renders) with optional robot insertion.
- Tune generation quality and stability via retry and seed settings.

## Routing Rule (Core)

Use `layout-cli` when the user input is task-level natural language descriptions (e.g., "put the pen in the mug") and the target output is an interactive layout scene, not standalone assets or standalone background scenes.

## Pre-checks

1. Run commands from the repository root.
2. Confirm the active environment is `embodiedgen`.
3. Confirm background scene list file exists and is readable (via `--bg_list`).
4. If CLI commands are unavailable, run `pip install -e .` to register entrypoints.

## Standard Command Templates

### 1) Generate layouts from inline task descriptions

```bash
layout-cli \
  --task_descs "Place the pen in the mug on the desk" "Put the fruit on the table on the plate" \
  --bg_list "outputs/example_gen_scenes/scene_part_list.txt" \
  --output_root "outputs/layouts_gen" \
  --insert_robot
```

### 2) Batch generation from task list file (background run)

```bash
layout-cli \
  --task_descs "apps/assets/example_layout/task_list.txt" \
  --bg_list "outputs/example_gen_scenes/scene_part_list.txt" \
  --n_image_retry 4 --n_asset_retry 3 --n_pipe_retry 3 \
  --output_root "outputs/layouts_gens" \
  --insert_robot > layouts_gens.log 2>&1 &
```

## Common Parameters

- `--task_descs`: task descriptions or a task-list text file path.
- `--output_root`: root output directory.
- `--bg_list`: background scene list file (scene retrieval pool).
- `--insert_robot`: include robot pose in layout generation/simulation output.
- `--output_iscene`: export composed scene mesh (`Iscene.glb`).
- `--n_image_retry --n_asset_retry --n_pipe_retry`: retry controls for text-to-3D subpipeline.
- `--seed_img --seed_3d --seed_layout`: reproducibility controls.
- `--n_img_sample --text_guidance_scale --img_denoise_step`: text-to-image / asset-generation controls.
- `--keep_intermediate`: keep intermediate files from generation substeps.

## Output Conventions

Outputs are organized by task index:
- `<output_root>/task_0000/layout.json`
- `<output_root>/task_0000/scene_tree.jpg`
- `<output_root>/task_0000/background/`
- `<output_root>/task_0000/asset3d/`
- Optional: `<output_root>/task_0000/Iscene.glb` (when `--output_iscene` is enabled)

## Runtime Expectations

- Typical generation time is around 30 minutes per task (depends on retries/GPU/background matching).
- Batch jobs should use background execution (`nohup`) with log redirection.

## Failure Handling and Retry

1. Missing background candidate: verify `--bg_list` path and referenced scene directories.
2. OOM or GPU pressure: reduce concurrency and lower retry/sample settings.
3. Poor asset/layout quality: increase retry counts or refine task text.
4. Missing outputs: verify output permissions and use absolute paths.
