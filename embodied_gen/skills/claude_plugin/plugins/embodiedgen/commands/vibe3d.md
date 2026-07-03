---
description: Run the EmbodiedGen spatial computing workflow for floorplans and object placement or deletion in scenes
argument-hint: "[scene editing request]"
---

# Vibe3D Skill Command

Route the user's request to the EmbodiedGen spatial computing workflow.

## Workflow

### Step 1: Interpret the request

Use `$ARGUMENTS` if provided. If it is empty, ask for the URDF path and target placement, deletion, or query request.

### Step 2: Load the skill

Use `skill: "embodiedgen:spatial-computing"`.

### Step 3: Execute the workflow

Follow the skill and choose the correct `python -m embodied_gen.skills.spatial-computing.cli.main` or `room-cli -m ...` invocation.

### Step 4: Deliver

Return:
1. The exact command used
2. The updated output file path
3. Any constraints about USD updates or exact instance matching
