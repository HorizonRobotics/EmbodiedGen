---
name: asset-converter
description: Convert EmbodiedGen URDF assets to simulator-specific formats (USD/MJCF/URDF) using embodied_gen.data.asset_converter APIs. Use this skill whenever users ask to export assets for IsaacSim, MuJoCo, Genesis, IsaacGym, PyBullet, or SAPIEN, batch-convert URDF assets, or choose the correct converter/source_type per simulator.
---

# Asset Converter

Unified entry for simulator-targeted asset conversion using `embodied_gen.data.asset_converter`.

## When To Use

Use this skill when users want to:
- Convert EmbodiedGen assets for IsaacSim (`USD`) or MuJoCo/Genesis (`MJCF`).
- Batch-convert multiple URDF assets into simulator-ready outputs.
- Map simulator names to the correct target format and conversion strategy.
- Decide when conversion is unnecessary (URDF can be used directly).

## Routing Rules (Core)

1. **IsaacSim** -> convert to `USD`.
2. **MuJoCo / Genesis** -> convert to `MJCF` (`.xml`).
3. **SAPIEN / IsaacGym / PyBullet** -> use EmbodiedGen `.urdf` directly (no conversion required).

## Pre-checks

1. Run from repository root with `embodiedgen` environment active.
2. Confirm input URDF path(s) exist.
3. For USD conversion, ensure IsaacLab/IsaacSim conversion dependencies are available.
4. Prefer list inputs for `urdf_files` and `target_dirs` (same length, aligned by index).

## Standard Python API Template

```python
from embodied_gen.data.asset_converter import cvt_embodiedgen_asset_to_anysim
from embodied_gen.utils.enum import AssetType, SimAssetMapper

simulator_name = "mujoco"  # isaacsim / mujoco / genesis / sapien / isaacgym / pybullet

asset_paths = cvt_embodiedgen_asset_to_anysim(
    urdf_files=[
        "outputs/demo_assets/remote_control/result/remote_control.urdf",
    ],
    target_dirs=[
        "outputs/demo_assets/remote_control/mjcf",
    ],
    target_type=SimAssetMapper[simulator_name],
    source_type=AssetType.URDF,
    overwrite=True,
)
print(asset_paths)
```

## Source Type Guidance

- For `MJCF` target: prefer `source_type=AssetType.URDF`.
- For `USD` target: use `source_type=AssetType.MESH` by default; `AssetType.URDF` path is also supported when needed.
- For direct-URDF simulators (`sapien`, `isaacgym`, `pybullet`): skip conversion.

## Direct Converter Template (Advanced)

```python
from embodied_gen.data.asset_converter import AssetConverterFactory
from embodied_gen.utils.enum import AssetType

converter = AssetConverterFactory.create(
    target_type=AssetType.USD,
    source_type=AssetType.MESH,
)

with converter:
    converter.convert(
        "outputs/demo_assets/remote_control/result/remote_control.urdf",
        "outputs/demo_assets/remote_control/usd/remote_control.usd",
    )
```

## Output Conventions

- `MJCF`: `<target_dir>/<asset_name>.xml`
- `USD`: `<target_dir>/<asset_name>.usd`
- API return: `{<input_urdf_path>: <converted_output_path>}`

## Failure Handling and Retry

1. Unsupported conversion pair: verify `target_type` + `source_type` mapping.
2. Missing dependencies (USD path): install/activate IsaacLab + required USD stack.
3. Missing output file: verify parent output directory permissions and path correctness.
4. Batch mismatch: ensure `len(urdf_files) == len(target_dirs)`.
