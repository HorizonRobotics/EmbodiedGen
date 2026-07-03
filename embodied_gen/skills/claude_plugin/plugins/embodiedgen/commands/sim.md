---
description: Run the EmbodiedGen simulation rendering workflow with sim-cli
argument-hint: "[layout path or simulation request]"
---

# Sim Skill Command

Route the user's request to the EmbodiedGen simulation rendering workflow.

## Workflow

### Step 1: Interpret the request

Use `$ARGUMENTS` if provided. If it is empty, ask for the `layout.json` path or target simulation request.

### Step 2: Load the skill

Use `skill: "embodiedgen:sim-runner"`.

### Step 3: Execute the workflow

Follow the skill and build the correct `sim-cli` command.

### Step 4: Deliver

Return:
1. The exact command used
2. The output video path
3. Any camera, performance, or rendering notes
