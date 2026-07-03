---
name: sim-runner
description: Run SAPIEN-based simulation rendering from EmbodiedGen layout outputs using sim-cli. Use this skill whenever users ask to load a generated layout.json into simulation, render interactive scene videos, control camera/render settings, or enable robot grasp trajectory rendering.
---

# Sim Runner

Unified entry for EmbodiedGen simulation rendering via `sim-cli`.

## When To Use

Use this skill when users want to:
- Load a generated `layout.json` into simulation.
- Render interactive scene videos (foreground + 3DGS background composition).
- Adjust camera, rendering, or simulation-step parameters.
- Include robot grasp trajectory rendering with `--insert_robot`.

## Routing Rule (Core)

Use `sim-cli` when the input is an existing layout result (especially `layout.json`) and the target output is simulation visualization (e.g., `Iscene.mp4`), not generation of new assets/backgrounds/layouts.

## Pre-checks

1. Run commands from the repository root.
2. Confirm the active environment is `embodiedgen`.
3. Confirm input `--layout_path` exists and points to a valid layout output.
4. Ensure referenced background and asset files in the layout directory are present.
5. If CLI commands are unavailable, run `pip install -e .` to register entrypoints.

## Standard Command Template

```bash
sim-cli \
  --layout_path "outputs/layouts_gen/task_0000/layout.json" \
  --output_dir "outputs/layouts_gen/task_0000/sapien_render" \
  --insert_robot
```

## Common Parameters

- `--layout_path`: input layout file path.
- `--output_dir`: output directory for rendered video.
- `--insert_robot`: render robot grasp actions for manipulated objects.
- `--sim_freq --control_freq --sim_step`: simulation/control timing settings.
- `--render_interval`: render every N simulation steps.
- `--num_cameras --camera_radius --camera_height --fovy_deg`: camera configuration.
- `--image_hw`: output frame size.
- `--render_keys`: render channels (requires `Foreground` for final compositing).
- `--ray_tracing`: enable/disable ray tracing backend.
- `--device`: rendering device (e.g., `cuda`).

## Output Conventions

Primary output:
- `<output_dir>/Iscene.mp4`

Typical input dependencies resolved from layout directory:
- `layout.json`
- background `gs_model.ply`
- per-object assets referenced by layout

## Runtime Expectations

- Runtime depends on `sim_step`, `render_interval`, camera count, and ray-tracing mode.
- Enabling `--insert_robot` increases render time due to grasp-action rollout.

## Failure Handling and Retry

1. Missing file errors: verify layout-relative asset/background paths exist.
2. GPU memory pressure: reduce `--num_cameras`, `--image_hw`, or disable heavy settings.
3. Empty/invalid video output: ensure `Foreground` is included in `--render_keys`.
4. Slow runtime: reduce `--sim_step` or increase `--render_interval`.