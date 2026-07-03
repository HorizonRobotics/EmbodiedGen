---
description: Run the EmbodiedGen asset generation workflow for image-to-3D, text-to-3D, or texture generation
argument-hint: "[request or command requirements]"
---

# Gen Assets Skill Command

Route the user's request to the EmbodiedGen asset generation workflow.

## Workflow

### Step 1: Interpret the request

Use `$ARGUMENTS` if provided. If it is empty, ask what the user wants to generate or texture.

### Step 2: Load the skill

Use `skill: "embodiedgen:asset-creator"`.

### Step 3: Execute the correct route

Follow the skill to choose one of:
- `img3d-cli`
- `text3d-cli`
- `texture-cli`

### Step 4: Deliver

Return:
1. The exact command used
2. The output directory
3. Any important runtime notes or dependency issues
