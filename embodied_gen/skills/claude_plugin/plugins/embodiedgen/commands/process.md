---
description: Scale or rotate EmbodiedGen URDF-based assets and related files
argument-hint: "[asset scaling or rotation request]"
---

# Process Skill Command

Route the user's request to the EmbodiedGen asset processing workflow.

## Workflow

### Step 1: Interpret the request

Use `$ARGUMENTS` if provided. If it is empty, ask for the URDF path and
desired scale factor and/or XYZ rotation in degrees.

### Step 2: Load the skill

Use `skill: "embodiedgen:asset-process"`.

### Step 3: Execute the workflow

Follow the skill and build the correct
`python -m embodied_gen.skills.asset-process.asset_process` command.

### Step 4: Deliver

Return:
1. The exact command used
2. The output path
3. Whether the operation is normal mode or inplace mode
4. The applied scale factor and XYZ rotation
