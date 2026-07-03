---
description: Run the EmbodiedGen room generation workflow with room-creator for room or house generation and export
argument-hint: "[room generation request]"
---

# Gen Indoor Skill Command

Route the user's request to the EmbodiedGen room creation workflow.

## Workflow

### Step 1: Interpret the request

Use `$ARGUMENTS` if provided. If it is empty, ask for room type, output root, and whether export is needed.

### Step 2: Load the skill

Use `skill: "embodiedgen:room-creator"`.

### Step 3: Execute the workflow

Follow the skill and build the correct `python -m embodied_gen.scripts.room_gen.gen_room` or `room-cli` command.

### Step 4: Deliver

Return:
1. The exact command used
2. The output directory
3. Runtime and export-stage notes
