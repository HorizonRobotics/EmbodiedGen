# Floorplan Skill — API Reference

This document provides API details, configuration items, errors, and dependencies for reference beyond the usage instructions in [SKILL.md](SKILL.md).

## Contents

- [Floorplan Skill — API Reference](#floorplan-skill--api-reference)
  - [Contents](#contents)
  - [LLM Environment Configuration](#llm-environment-configuration)
  - [FloorplanManager](#floorplanmanager)
    - [Constructor](#constructor)
    - [Methods](#methods)
  - [Convenience Functions](#convenience-functions)
  - [CLI Features](#cli-features)
    - [Command Line Parameters](#command-line-parameters)
  - [Configuration and Ignore Items](#configuration-and-ignore-items)
  - [Smart File Naming Strategy](#smart-file-naming-strategy)
  - [USD and Blender](#usd-and-blender)
  - [Errors and Return Values](#errors-and-return-values)
  - [Dependencies](#dependencies)
  - [Usage Recommendations](#usage-recommendations)

---

## LLM Environment Configuration

Before using semantic matching (`resolve_*` methods), configure the LLM API:

```bash
# Use the project-provided env (Azure + proxy), if outputs/env.sh exists:
source outputs/env.sh
```

If access to the LLM interface is unavailable, prompt the user.

---

## FloorplanManager

### Constructor

```python
from importlib import import_module

FloorplanManager = import_module(
    "embodied_gen.skills.spatial-computing.api"
).FloorplanManager

manager = FloorplanManager(
    urdf_path="scene.urdf",      # Required
    usd_path=None,               # Optional; USD write after insert/delete if provided
    mesh_sample_num=50000,
    ignore_items=None,           # Default ["ceiling", "light", "exterior"]
    output_strategy="suffix",    # "suffix" (default) / "timestamp" / "overwrite"
)
```

### Methods

| Method | Description |
|--------|-------------|
| `visualize(output_path)` | Generate floorplan and save as image |
| `insert_object(asset_path, instance_key, in_room=..., on_instance=..., beside_instance=..., place_strategy=..., n_max_attempt=2000, rotation_rpy=...)` | Place object, automatically write back to URDF/USD on success, return `[x,y,z]` or `None` |
| `delete_object(instance_key, in_room=..., urdf_output_path=..., usd_output_path=...)` | Delete instance from scene, return `True`/`False`. Supports room constraint via `in_room` |
| `query_instance_center(instance_key)` | Query instance center coordinates, return `[x,y,z]` or `None` |
| `update_scene(urdf_output_path=..., usd_output_path=...)` | Manually write back currently placed instances; generally not needed (called inside `insert_object`) |
| `get_room_names()` | List of room names |
| `get_instance_names()` | List of instance names (excluding walls/floor) |
| `get_instance_names_in_room(in_room)` | List of instance names within a specific room |
| `resolve_on_instance(on_instance, gpt_client=None)` | Resolve user description to exact instance name |
| `resolve_in_room(in_room, gpt_client=None)` | Resolve user description to exact room name |
| `resolve_beside_instance(beside_instance, gpt_client=None, in_room=None)` | Resolve user description to exact instance name for beside placement |
| `resolve_delete_instance(delete_instance, gpt_client=None, in_room=None)` | Resolve user description to exact instance name for deletion |
| `resolve_and_query_instance(query_instance, gpt_client=None)` | Resolve and query instance center in one call, return `(resolved_name, [x,y,z])` or `(None, None)` |
| `get_occupied_area()` | Occupied area Shapely geometry |
| `get_floor_union()` | Floor area union geometry |

**Key parameters**:
- `on_instance` / `beside_instance` / `delete_instance`: Exact instance name or semantic description (with `gpt_client`)
- `in_room`: Room constraint for placement/deletion/query
- `place_strategy`: `"random"` (default) or `"top"` (select highest surface)
- `beside_distance`: Max distance in meters for beside placement (default 0.5)

---

## Convenience Functions

| Function | Description |
|----------|-------------|
| `visualize_floorplan(urdf_path, output_path, ...)` | Generate floorplan only |
| `insert_object_to_scene(urdf_path, asset_path, instance_key, output_path, ...)` | Insert object and generate floorplan, return `[x,y,z]` or `None` |
| `delete_object_from_scene(urdf_path, instance_key, in_room=..., output_path=...)` | Delete instance and optionally generate floorplan, return `True`/`False` |
| `query_instance_position(urdf_path, instance_key)` | Quick query instance center coordinates, return `[x,y,z]` or `None` |
| `resolve_instance_with_llm(gpt_client, instance_names, user_spec, ...)` | Use LLM to match user description to exact instance name |

---

## CLI Features

### Command Line Parameters

| Parameter | Description |
|-----------|-------------|
| `--urdf_path` | Input URDF scene file path (required) |
| `--usd_path` | Optional USD scene file path, update USD simultaneously if specified |
| `--asset_path` | Object mesh file path (.obj) for insertion |
| `--instance_key` | Unique identifier for the new instance, default `inserted_object` |
| `--in_room` | Limit placement to specified room, supports semantic description |
| `--on_instance` | Place on top of specified instance, supports semantic description |
| `--beside_instance` | Place beside specified instance on floor, supports semantic description |
| `--beside_distance` | Max distance (meters) from target instance, default 0.5 |
| `--place_strategy` | Placement strategy: `"random"` (default) or `"top"` |
| `--rotation_rpy` | Initial rotation angle (roll, pitch, yaw radians) |
| `--output_path` | Floorplan output path |
| `--output_strategy` | File naming strategy: `"suffix"` (default) / `"timestamp"` / `"overwrite"` |
| `--list_instances` | List instance names and room names, then exit |
| `--delete_instance` | Instance name to delete (supports semantic description) |
| `--delete_in_room` | Room constraint for deletion |
| `--query_instance` | Instance name to query position (supports semantic description) |
| `--max_placement_attempts` | Maximum placement attempts, default 2000 |

### CLI Usage Examples

**View scene info**:
```bash
python -m embodied_gen.skills.spatial-computing.cli.main \
  --urdf_path .../scene.urdf --list_instances
```

**Insert object with semantic matching**:
```bash
source outputs/env.sh
python -m embodied_gen.skills.spatial-computing.cli.main \
  --urdf_path .../scene.urdf --asset_path .../lamp.obj --instance_key lamp_1 \
  --on_instance 书柜
```

**Delete object with room constraint**:
```bash
python -m embodied_gen.skills.spatial-computing.cli.main \
  --urdf_path .../scene.urdf --delete_instance 沙发 --delete_in_room 客厅
```

**Query instance position**:
```bash
python -m embodied_gen.skills.spatial-computing.cli.main \
  --urdf_path .../scene.urdf --query_instance 床
```

**Update both URDF and USD (room-cli)**:
```bash
room-cli -m embodied_gen.skills.spatial-computing.cli.main \
  --urdf_path .../scene.urdf --usd_path .../scene.usdc \
  --delete_instance 沙发
```

---

## Configuration and Ignore Items

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mesh_sample_num` | 50000 | Number of mesh sampling points |
| `ignore_items` | `["ceiling", "light", "exterior"]` | Link name patterns to skip during URDF parsing |
| `output_strategy` | `"suffix"` | File naming strategy for output files |

---

## Smart File Naming Strategy

Default `output_strategy="suffix"` provides intelligent continuous operation support:

| Operation | Input File | Output File | Behavior |
|-----------|-----------|-------------|----------|
| First insert | `scene.urdf` | `scene_updated.urdf` | Creates new file |
| Second insert | `scene_updated.urdf` | `scene_updated.urdf` | **Overwrites** (continuous) |
| Delete | `scene_updated.urdf` | `scene_updated.urdf` | **Overwrites** (continuous) |

**Key features**:
- ✅ No `*_updated_updated.urdf` accumulation
- ✅ Original `scene.urdf` never modified
- ✅ Seamless insert/delete workflow

**Alternative strategies**:
- `"timestamp"`: Unique versioning (`scene_20260311_180235.urdf`)
- `"overwrite"`: Direct overwrite (use with caution)

---

## USD and Blender

- Writing USD requires **Blender (bpy)**. Use **room-cli** environment for USD operations.
- Without `usd_path`, only URDF is updated (no bpy needed).
- Assets in `.usd`/`.usdc`/`.usda` format are directly referenced; only `.obj` files are converted via bpy.
- If `*_collision.obj` exists alongside visual mesh, it will be used for URDF collision.

---

## Errors and Return Values

**Exceptions**

- **ValueError**: Room/instance not found; `update_scene()` called before insertion; `instance_key` already exists; attempting to delete protected instances (`walls`, `*floor*`).

**Return Values**

- `insert_object` / `insert_object_to_scene`: `[x, y, z]` on success, `None` on failure.
- `delete_object` / `delete_object_from_scene`: `True` on success, `False` on failure.
- `query_instance_center` / `query_instance_position`: `[x, y, z]` or `None`.

**Exit Codes (CLI)**

- `0`: Success
- `1`: Instance/room not found, deletion failed, or placement failed

---

## Dependencies

| Type | Package | Description |
|------|---------|-------------|
| Core | trimesh, shapely, matplotlib, numpy | Parsing and visualization |
| USD Writing | pxr, bpy | Required only when using `usd_path`; bpy requires Blender |
| LLM Semantic Matching | openai, project gpt_config | `resolve_*` methods require `GPTclient` instance |
| CLI | tyro | Required only for CLI entry point |

---

## Usage Recommendations

- **Upright objects**: Default orientation applies; for special orientations, pass `(roll, pitch, yaw)` radians.
- **Placing on furniture**: Use `resolve_on_instance()` to get exact name, then `insert_object(..., on_instance=resolved, place_strategy="top")`.
- **Placing beside furniture**: Use `insert_object(..., beside_instance=resolved, beside_distance=0.5)` for floor placement near target.
- **Deleting objects**: Use `resolve_delete_instance()` for semantic matching, then `delete_object(..., in_room=room)` for room-specific deletion.
- **Protected instances**: Cannot delete `walls` or instances containing `floor` in their names.
- **Continuous editing**: Use `scene_updated.urdf` as input for subsequent operations to maintain changes.
