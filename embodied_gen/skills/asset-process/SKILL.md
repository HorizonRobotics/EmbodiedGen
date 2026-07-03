---
name: asset-process
description: Scale and rotate complete URDF-based 3D assets, including OBJ, GLB, collision meshes, Gaussian splats, and height metadata. Use this skill whenever a user asks to resize, rotate, reorient, or otherwise transform an EmbodiedGen asset while keeping its related files synchronized.
---

# Asset Processing

## Overview

Use this skill to scale or rotate an asset and all of its supported geometry
files consistently. Rotation is provided as XYZ Euler angles in degrees and is
applied around the asset's local origin.

The input URDF may use either supported asset layout:

```text
<asset_dir>/result/<asset_name>.urdf
<asset_dir>/<asset_name>.urdf
```

The mesh directory must be beside the URDF as `mesh/`.

Normal mode copies the complete asset directly to `<output_dir>`. Pass the
full target asset directory, including the desired asset directory name.
Inplace mode modifies the source asset.

After processing, the tool calculates the transformed mesh's Y-axis extent and
writes that value to `min_height`, `max_height`, and `real_height` in the URDF.
It uses the main OBJ first, then the GLB, then the collision OBJ.

## CLI Examples

### Scale Only

```bash
python -m embodied_gen.skills.asset-process.asset_process \
  --urdf-path outputs/assets/red_box/result/red_box.urdf \
  --scale-factor 0.8 \
  --output-dir outputs/processed
```

### Rotate Only

```bash
python -m embodied_gen.skills.asset-process.asset_process \
  --urdf-path outputs/assets/red_box/result/red_box.urdf \
  --rot-xyz 90 0 45 \
  --output-dir outputs/processed
```

### Scale and Rotate

```bash
python -m embodied_gen.skills.asset-process.asset_process \
  --urdf-path outputs/test_wuwen/facial_cleanser_001/facial_cleanser_001.urdf \
  --scale-factor 1.0 \
  --rot-xyz 0 90 0 \
  --keep-urdf-raw-rot \
  --output-dir outputs/test_wuwen/facial_cleanser_001_processed
```

`--keep-urdf-raw-rot` is optional. When enabled, the inverse baked rotation is
composed into visual and collision `origin rpy` values so their effective
rotation remains unchanged. It is disabled by default.

### Process Inplace

```bash
python -m embodied_gen.skills.asset-process.asset_process \
  --urdf-path outputs/assets/red_box/result/red_box.urdf \
  --scale-factor 0.8 \
  --rot-xyz 0 0 90 \
  --inplace
```

Inplace mode modifies the original asset files.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `urdf_path` | Required | Input URDF path |
| `scale_factor` | `1.0` | Positive uniform scaling factor |
| `rot_xyz` | `0 0 0` | XYZ roll, pitch, yaw in degrees |
| `keep_urdf_raw_rot` | `False` | Preserve the original effective URDF rotation |
| `inplace` | `False` | Modify source files directly |
| `output_dir` | `None` | Complete target asset directory; required outside inplace mode |

## Processed Files

- `mesh/<name>.obj`
- `mesh/<name>.glb`
- `mesh/<name>_collision.obj`
- `mesh/<name>_gs.ply` (optional; skipped when absent)
- `<name>.urdf`

Rotation is always baked into the asset files. URDF visual and collision
`origin rpy` values are updated only when `keep_urdf_raw_rot` is enabled.

For Python API details, read
`embodied_gen/skills/asset-process/asset_process.py`.
