## Using with IDE Agent via Natural Language

The Agent will automatically load this skill based on its **description** when you mention URDF, floorplan, indoor scene, object placement, etc. You only need to specify in natural language **what to do** and provide **key information like paths/room names**.

### LLM Environment Configuration (When Using Semantic Matching)

If you want to use natural language descriptions (e.g., "put lamp on bookshelf") instead of exact instance/room names, you need to configure the LLM environment first:

```bash
# If outputs/env.sh exists, source it first
source outputs/env.sh
```

If access to the LLM interface is unavailable, please provide exact instance names (you can check them via `--list_instances`).

### URDF Visualization Only (Generate Floorplan)

**You can say:**
- "Help me visualize `path_to/scene.urdf` or `path_to/folder_contain/scene.urdf`"

**Agent will:** Use `visualize_floorplan(urdf_path=..., output_path=...)` or the corresponding CLI to generate the floorplan only, without modifying URDF/USD.

### Insert Object and Update Scene (URDF, or URDF+USD)

**You can say:**
- "Put `chair.obj` into scene.urdf's kitchen room"
- "Put `bottle.obj` into the URDF at `outputs/rooms/Kitchen_seed3773`, instance name bottle_1, update scene and generate floorplan"
- "Put a cup on the table in the living room" → Agent will use `on_instance="table"`, `place_strategy="top"`, etc.

**If you also want to update USD:**
- "Put a chair in the kitchen, update both URDF and USD, USD path is `xxx/usd/export_scene.usdc`"
- Note that you need to use **room-cli** to execute (this skill will prompt the Agent), because writing USD requires bpy.

**Agent will:** Use `FloorplanManager` + `insert_object` (or `insert_object_to_scene`), execute according to the paths and room names you provided; when USD is needed, use room-cli to run the CLI.

### View Instances and Rooms in the Scene

Before placing objects, you can first view what instances and rooms are in the scene:

**You can say:**
- "Help me list all instances and room names in `.../scene.urdf`"

**Agent will:** Execute `--list_instances` to display the instance names and room names in the current scene.

### URDF/USD Output Notes

- **URDF Output**: The updated URDF is written to `*_updated.urdf` by default (e.g., `scene.urdf` → `scene_updated.urdf`), and **will not overwrite** the original `scene.urdf`
- **USD Output**: If `usd_path` is specified, the USD file will be written to `*_updated.usdc` following the same rule
- **Only Update USD**: Requires using **room-cli** to execute, because writing USD needs Blender (bpy)

### What Information to Provide

| Goal | Suggested Information to Provide in Conversation |
|------|-----------------------------------------------|
| Visualization only | URDF path, floorplan save path (optional, Agent can default to floorplan.png in same directory) |
| View instances/rooms | URDF path, let Agent list instance names and room names in current scene |
| Placement + update | URDF path, object mesh path (.obj), instance name (e.g., chair_1), room name (e.g., kitchen); if placing on table, say "place on table"; if updating USD, also provide USD path and use room-cli |

Example in one go: "Use spatial-computing skill, generate floorplan for `.../scene.urdf` and save to floorplan.png in same directory, then put `path/to/bottle.obj` into kitchen, instance name bottle_1, update URDF only."
