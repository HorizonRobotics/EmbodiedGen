---
description: Run the EmbodiedGen background scene generation workflow with scene3d-cli
argument-hint: "[scene prompt or generation request]"
---

# Gen Bg Skill Command

Route the user's request to the EmbodiedGen background generation workflow.

## Workflow

### Step 1: Interpret the request

Use `$ARGUMENTS` if provided. If it is empty, ask for the target room or background scene description.

### Step 2: Load the skill

Use `skill: "embodiedgen:background-creator"`.

### Step 3: Execute the workflow

Follow the skill and build the correct `scene3d-cli` command.

### Step 4: Deliver

Return:
1. The exact command used
2. The output directory
3. Expected runtime and any caveats
