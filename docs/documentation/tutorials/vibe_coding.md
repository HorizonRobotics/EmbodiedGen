# 💬 3D Vibe Coding — Build 3D Worlds Through Dialogue

Build and edit **sim-ready 3D worlds through natural-language dialogue**. EmbodiedGen ships a set of reusable agent skills and a **Claude Code plugin** whose slash commands wrap them — each instruction is a bounded, physics-validated skill call that preserves a deployable, simulator-ready world state.

<img src="../assets/vibe_coding.gif" alt="3D Vibe Coding: editing a sim-ready world through dialogue" style="width: 800px; max-width: 100%; border-radius: 12px; display: block; margin: 16px auto;">

---

## 🔌 Install the Claude Code Plugin

```bash
bash install/install_agent_plugin.sh
```

This registers the local marketplace `embodiedgen-local` and installs the `embodiedgen` plugin into [Claude Code](https://claude.com/claude-code). The plugin package lives in `embodied_gen/skills/claude_plugin/plugins/embodiedgen/`.

## 🧰 Available Commands

| Command | What it does |
|---------|--------------|
| `/embodiedgen:gen_assets` | Generate 3D assets from images or text (`img3d-cli`, `text3d-cli`, `texture-cli`) |
| `/embodiedgen:gen_indoor` | Generate rooms or multi-room houses ([Room Generation](room_gen.md)) |
| `/embodiedgen:gen_bg` | Generate 3DGS background scenes (`scene3d-cli`) |
| `/embodiedgen:gen_layout` | Compose task-driven interactive worlds (`layout-cli`) |
| `/embodiedgen:vibe3d` | Insert / remove / place objects in a scene via natural language (spatial computing) |
| `/embodiedgen:sim` | Render layouts in SAPIEN simulation (`sim-cli`) |
| `/embodiedgen:convert` | Export assets to USD / MJCF / URDF ([Any Simulators](any_simulators.md)) |
| `/embodiedgen:process` | Scale or rotate existing URDF-based assets |

Example session:

```text
> /embodiedgen:gen_indoor Create a simple kitchen, seed 42
> /embodiedgen:vibe3d Put a lamp on the bookshelf in the living room
> /embodiedgen:sim Render outputs/layouts_gen/task_0000/layout.json
```

## 🧠 Skill Sources

The slash commands are thin adapters over generic, runtime-agnostic skills under `embodied_gen/skills/`:

- `asset-creator`, `asset-retrieval`, `asset-converter`, `asset-process`
- `background-creator`, `room-creator`, `layout-creator`, `sim-runner`
- `spatial-computing` — floorplan visualization and collision-aware object placement/deletion with semantic instance matching (e.g. *"put lamp on bookshelf"*, *"delete sofa in living room"*). See `embodied_gen/skills/spatial-computing/SKILL.md` for details.

!!! note "USD updates require Blender Python"
    Scene edits that update USD files must run under `room-cli` (Blender Python provides `bpy`); the skills prompt the agent to choose the correct invocation automatically.

---

!!! tip "Next Steps"
    - Load the edited world into simulation — see [Robot Learning](robot_learning.md).
    - Export to your simulator of choice — see [Any Simulators](any_simulators.md).
