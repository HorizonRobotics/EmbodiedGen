---
name: spatial-computing
description: Visualizes floorplans from URDF scene files and inserts/removes 3D assets with collision-aware placement on surfaces. Supports semantic instance matching via LLM (e.g., "put lamp on bookshelf", "delete sofa in living room"). Use when working with URDF/USD indoor scenes, floorplan visualization, object placement/deletion, or room-level scene editing.
---

# Floorplan & Object Placement/Deletion

## Overview

Parse indoor scenes from URDF, generate 2D floorplans, or place/remove 3D objects in scenes and write back to URDF/USD. After successful insertion/deletion, the corresponding file is automatically updated based on whether `urdf_path`/`usd_path` is provided.

**When to use**: Use this skill when you need to generate floorplans from URDF, place/delete objects on specified rooms/furniture surfaces, or batch update URDF/USD files.

> ⚠️ **USD updates require `room-cli`**: To update USD files, you **must** use `room-cli` instead of `python -m`, and specify the USD file via `--usd_path`. `room-cli` runs on Blender Python which includes the `bpy` module for OBJ→USD conversion; using `python -m` with `--usd_path` will fail with `ModuleNotFoundError: No module named 'bpy'`.
>
> ```bash
> # ✅ Correct: use room-cli to update both URDF and USD
> room-cli -m embodied_gen.skills.spatial-computing.cli.main \
>   --urdf_path .../scene.urdf --usd_path .../scene.usdc ...


**Smart File Naming Strategy**:
- **Default behavior**: First operation creates `scene_updated.urdf`, subsequent operations automatically overwrite it
- **No file bloat**: Prevents `*_updated_updated.urdf` files from accumulating
- **Safe**: Original `scene.urdf` is never modified unless explicitly requested
- **Works for both insert and delete**: Seamless continuous scene editing

---

## Best Practices & Constraints

### 1. Workflow for Continuous Scene Editing

**Recommended workflow** for multiple insert/delete operations:

```bash
# Step 1: View current scene
python -m embodied_gen.skills.spatial-computing.cli.main \
  --urdf_path .../scene.urdf --list_instances

# Step 2: First insert → creates scene_updated.urdf
python -m embodied_gen.skills.spatial-computing.cli.main \
  --urdf_path .../scene.urdf \
  --asset_path .../apple.obj --instance_key apple_1

# Step 3: Second insert → overwrites scene_updated.urdf
python -m embodied_gen.skills.spatial-computing.cli.main \
  --urdf_path .../scene_updated.urdf \
  --asset_path .../lamp.obj --instance_key lamp_1

# Step 4: Delete operation → overwrites scene_updated.urdf
python -m embodied_gen.skills.spatial-computing.cli.main \
  --urdf_path .../scene_updated.urdf \
  --delete_instance apple_1
```

**Key benefits**:
- ✅ No multiple `*_updated_updated.urdf` files
- ✅ Original file `scene.urdf` always preserved
- ✅ Continuous insert/delete operations are seamless

**Result**: Clean workflow with only two files:
- `scene.urdf` (original, untouched)
- `scene_updated.urdf` (final state)

### 2. When to Use Different Strategies

| Strategy | Use Case | Example |
|----------|----------|---------|
| **suffix** (default) | Standard workflow, continuous editing | Most scenarios |
| **timestamp** | Version tracking, backup before risky changes | `scene_20260311_180235.urdf` |
| **overwrite** | Confident single operation, no backup needed | Automated pipelines |

### 3. Performance Optimization: Batch Insert

**Problem**: CLI commands re-parse URDF and process all meshes on every call, leading to slow performance when inserting multiple objects.

**Solution**: Use `--batch_insert_config` with JSON config for 3-4x speedup:

**Step 1**: Create JSON config file (`batch_chairs.json`):

```json
[
    {
        "asset_path": "path/to/chair1.obj",
        "instance_key": "chair_1",
        "beside_instance": "table_dining_7178300",
        "in_room": "dining_room_0_floor"
    },
    {
        "asset_path": "path/to/chair2.obj",
        "instance_key": "chair_2",
        "beside_instance": "table_dining_7178300",
        "in_room": "dining_room_0_floor"
    },
    {
        "asset_path": "path/to/chair3.obj",
        "instance_key": "chair_3",
        "beside_instance": "table_dining_7178300",
        "in_room": "dining_room_0_floor"
    }
]
```

**Step 2**: Run batch insertion:

```bash
# Update URDF only
room-cli -m embodied_gen.skills.spatial-computing.cli.main \
  --urdf_path .../scene.urdf \
  --batch_insert_config batch_chairs.json

# Update both URDF and USD
room-cli -m embodied_gen.skills.spatial-computing.cli.main \
  --urdf_path .../scene.urdf \
  --usd_path .../scene.usdc \
  --batch_insert_config batch_chairs.json
```

**JSON Config Fields**:
- `asset_path` (required): Path to asset mesh file (.obj)
- `instance_key` (required): Unique instance identifier
- `beside_instance`: Place beside target instance (on floor). **Must be exact name**.
- `on_instance`: Place on top of target instance. **Must be exact name**.
- `in_room`: Limit placement to specified room. **Must be exact name**.
- `beside_distance`: Max distance from target (default: 0.5m)
- `place_strategy`: "random" or "top" (default: "random")

> **⚠️ Batch insert does NOT support fuzzy/semantic matching.**
> `beside_instance`, `on_instance`, and `in_room` require exact names.
> Use `--list_instances` to get the exact instance / room names first:
> ```bash
> python -m embodied_gen.skills.spatial-computing.cli.main \
>   --urdf_path .../scene.urdf --list_instances
> ```

**When to Use**:
- ✅ Inserting 2+ objects at once
- ✅ Performance-critical workflows
- ✅ Automated scene generation pipelines

⚠️ **Batch config file cleanup**: The JSON config file for `--batch_insert_config` is a **temporary file** and **must not** be left in the project root directory. Always:
1. Create the JSON config in the **same directory as the target scene** (e.g., `.../House_seed5/batch_fruits.json`).
2. **Delete the JSON config file immediately after the batch command finishes**, regardless of success or failure.

### 3. Important Constraints

**USD prim hierarchy**: When updating a USD file, inserted assets must be
authored under the stage's `defaultPrim` (for existing room exports this is
usually `/World`), for example `/World/<instance_key>`. Do not write inserted
objects as pseudo-root children like `/<instance_key>` because USD references in
IsaacSim load the `defaultPrim` only; root-level siblings outside `defaultPrim`
will be omitted.

❌ **Wrong**: Using `scene.urdf` for all operations (ignores previous changes)
```bash
# This will NOT see apple_1 from previous operation
python -m ... --urdf_path scene.urdf --asset_path lamp.obj
```

✅ **Right**: Chain operations using `scene_updated.urdf`
```bash
# This WILL see apple_1 and add lamp_1
python -m ... --urdf_path scene_updated.urdf --asset_path lamp.obj
```

---

## LLM Environment (Required for Semantic Matching)

Before using `resolve_instance_with_llm` for semantic matching in **Python**, configure the LLM API and ensure access to the interface. Prompt the user if access is unavailable.

```bash
# Use the project-provided env (Azure + proxy, etc.), if outputs/env.sh exists:
source outputs/env.sh
```

---

## Core Convention: Placement/Deletion/Query Requests Must Use This Skill's Interface

When users request "put A somewhere", "delete A", "find A", or "visualize urdf", you **must** implement it using this skill's interface:

| User Request Example | Corresponding Parameter & Usage |
|---------------------|---------------------------------|
| **Put A on B** (e.g., "put lamp on bookshelf") | `on_instance` (instance name, obtained from `--list_instances`) |
| **Put A beside B** (e.g., "put chair beside table") | `beside_instance` (instance name, obtained from `--list_instances`); placed on floor near target |
| **Put A in a room** (e.g., "put table in living room") | `in_room` (room name, obtained from `--list_instances`) |
| **Put A beside B in a room** (e.g., "put chair beside table in kitchen") | `beside_instance` + `in_room` |
| **Put A on B in a room** (e.g., "put apple on table in living room") | Decomposed into "apple" and "living room" as `in_room` and `on_instance` |
| **Delete A** (e.g., "delete lamp") | `delete_instance` (instance name or semantic description, supports fuzzy matching with LLM) |
| **Delete A in a room** (e.g., "delete sofa in living room") | `delete_instance` + `delete_in_room` (only deletes if instance is in specified room) |
| **Find A** (e.g., "find lamp", "where is the bed") | `query_instance` (returns center coordinates [x, y, z], supports fuzzy matching with LLM) |

| `output_strategy` | `"suffix"` / `"timestamp"` / `"overwrite"` | File naming strategy for output files. Default is "suffix" (non-destructive). |
| **Visualize scene.urdf** | `cli.main --urdf_path .../scene.urdf --output_path .../floorplan.png`; output_path defaults to same directory as urdf |

- When no match is found, prompt "The object/room does not exist, please re-enter" and provide the current scene object or room list.
- Instance names should not use the `<link name="...">` from URDF. **Recommended**: Run `--list_instances` before placement/deletion/query to view current instance name list, and select the closest semantic match.

---

## CLI Examples

> **Tip**: The URDF file is typically located at `<room_folder>/urdf/export_scene/scene.urdf` (e.g., `outputs/rooms/Kitchen_seed0/urdf/export_scene/scene.urdf`).

### Example 1: View Instance Names and Room Names in Current Scene

```bash
# View instance names and room names in current scene (to fill in --on_instance / --in_room)
python -m embodied_gen.skills.spatial-computing.cli.main \
  --urdf_path .../scene.urdf --list_instances
```

### Example 2: Visualize Floorplan Only

```bash
python -m embodied_gen.skills.spatial-computing.cli.main \
  --urdf_path .../scene.urdf --output_path .../floorplan.png
```

### Example 3: Put Lamp on Bookshelf (Place on an Object)

`--on_instance` can be filled with the instance name returned by `--list_instances` or a semantic description.

```bash
python -m embodied_gen.skills.spatial-computing.cli.main \
  --urdf_path .../scene.urdf --output_path .../floorplan.png \
  --asset_path .../lamp.obj --instance_key lamp_on_bookcase --on_instance 书柜
```

---

### Example 4: Put Table in Living Room (Place in a Room)

```bash
python -m embodied_gen.skills.spatial-computing.cli.main \
  --urdf_path .../scene.urdf --output_path .../floorplan.png \
  --asset_path .../table.obj --instance_key table_1 \
  --in_room living_room
```

---

### Example 5: Put Apple on Table in Living Room (Room + on Object)

```bash
python -m embodied_gen.skills.spatial-computing.cli.main \
  --urdf_path .../scene.urdf --output_path .../floorplan.png \
  --asset_path .../apple.obj --instance_key apple_1 \
  --in_room living_room --on_instance table --place_strategy top
```

---

### Example 7: Delete an Object (Exact Name)

```bash
python -m embodied_gen.skills.spatial-computing.cli.main \
  --urdf_path .../scene.urdf --output_path .../floorplan.png \
  --delete_instance bed_192207
```

---

### Example 8: Delete Object with Fuzzy Matching (Semantic Description)

Requires LLM environment (see "LLM Environment" section).

```bash
python -m embodied_gen.skills.spatial-computing.cli.main \
  --urdf_path .../scene.urdf --output_path .../floorplan.png \
  --delete_instance "沙发"
```

---

### Example 9: Delete Object in Specific Room

Only deletes the instance if it's located in the specified room.

```bash
python -m embodied_gen.skills.spatial-computing.cli.main \
  --urdf_path .../scene.urdf --output_path .../floorplan.png \
  --delete_instance "沙发" --delete_in_room "客厅"
```

**Update both URDF and USD using room-cli:**
```bash
room-cli -m embodied_gen.skills.spatial-computing.cli.main \
  --urdf_path .../scene.urdf --usd_path .../scene.usdc \
  --output_path .../floorplan.png \
  --delete_instance "沙发" --delete_in_room "客厅"
```

---

### Example 10: Query Instance Position (Exact Name)

```bash
python -m embodied_gen.skills.spatial-computing.cli.main \
  --urdf_path .../scene.urdf \
  --query_instance bed_192207
```

**Expected output**:
```
📍 Instance 'bed_192207' center: (-0.9250, -6.5830, 0.5000)
```

---

### Example 11: Query Instance Position with Fuzzy Matching

Requires LLM environment (see "LLM Environment" section).

```bash
python -m embodied_gen.skills.spatial-computing.cli.main \
  --urdf_path .../scene.urdf \
  --query_instance "床"
```

---

#### **Alternative Strategies**

**Timestamp** - Unique versioning for each operation:
```bash
# Output: scene_20260311_180235.urdf
python -m embodied_gen.skills.spatial-computing.cli.main \
  --urdf_path .../scene.urdf \
  --asset_path .../apple.obj --instance_key apple_1 \
  --output_strategy timestamp
```

**Overwrite** - Directly overwrite original (use with caution):
```bash
# Overwrites: scene.urdf
python -m embodied_gen.skills.spatial-computing.cli.main \
  --urdf_path .../scene.urdf \
  --asset_path .../apple.obj --instance_key apple_1 \
  --output_strategy overwrite
```

---

### Query Instance Position

Query the center coordinates of an instance in the scene. Supports fuzzy matching with LLM.

**CLI Interface**:
```bash
# Exact instance name
python -m embodied_gen.skills.spatial-computing.cli.main \
  --urdf_path scene.urdf \
  --query_instance bed_192207

# Fuzzy matching (requires GPT)
source outputs/env.sh
python -m embodied_gen.skills.spatial-computing.cli.main \
  --urdf_path scene.urdf \
  --query_instance "床"
```

### 6. Common Parameters

| Parameter | Meaning |
|-----------|---------|
| `in_room` | Limit placement to specified room |
| `on_instance` | Place on top of specified instance; must be **exact instance name** (obtained via `resolve_instance_with_llm`) |
| `beside_instance` | Place beside specified instance on the floor; must be **exact instance name** (obtained via `resolve_instance_with_llm`). Mutually exclusive with `on_instance` |
| `beside_distance` | Max distance (meters) from target instance for beside placement. Default `0.5`. Increase if placement fails |
| `place_strategy` | `"random"` random placement (default, e.g., bookshelf with 3 layers will randomly select one), `"top"` select highest surface |
| `rotation_rpy` | Not required by default; pass (roll, pitch, yaw) radians for special orientations |
| `delete_instance` | Instance name or semantic description to delete (supports fuzzy matching with LLM). Cannot delete protected items (walls, floors) |
| `delete_in_room` | Optional room constraint for deletion - only delete if instance is in this room |
| `query_instance` | Instance name or semantic description to query center coordinates (supports fuzzy matching with LLM). Returns [x, y, z] position |

---

## Roaming Trajectory Generation (Optional)

Generate a smooth, collision-free robot **roaming trajectory** on a floorplan
and overlay it on the floorplan image. **Opt-in and off by default**: the
normal floorplan visualization never draws a trajectory unless you run this
CLI. Doors are treated as open passages (excluded from obstacles).

**CLI Interface**:
```bash
python -m embodied_gen.scripts.room_gen.gen_trajectory \
  --urdf_path .../scene.urdf \
  --output_dir .../trajectory \
  --clearance 0.4 --num_waypoints 8
```

Outputs `<stem>_trajectory.json` (equidistant `{x, y, rot, t}` waypoints at a
constant speed) and `.png` (red-path overlay). `rot` in degrees: `0°` = `+Y`
(12 o'clock), counter-clockwise positive, tangent to the path (forward
heading); `t` is the timestamp in seconds.

---

## Next Steps

- For complete API, configuration, errors, and dependencies, see [REFERENCE.md](REFERENCE.md).
