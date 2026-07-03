---
description: Run the EmbodiedGen simulator asset conversion workflow for USD, MJCF, or direct URDF usage
argument-hint: "[target simulator or conversion request]"
---

# Convert Skill Command

Route the user's request to the EmbodiedGen asset conversion workflow.

## Workflow

### Step 1: Interpret the request

Use `$ARGUMENTS` if provided. If it is empty, ask for the target simulator and input URDF path.

### Step 2: Load the skill

Use `skill: "embodiedgen:asset-converter"`.

### Step 3: Execute the workflow

Follow the skill and choose:
- `USD` for IsaacSim
- `MJCF` for MuJoCo or Genesis
- direct `URDF` for SAPIEN, IsaacGym, or PyBullet

### Step 4: Deliver

Return:
1. The exact Python API or command used
2. The converted output path
3. Any dependency or simulator-specific notes
