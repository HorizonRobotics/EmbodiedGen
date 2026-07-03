---
description: Run the EmbodiedGen layout generation workflow with layout-cli
argument-hint: "[task description or layout generation request]"
---

# Gen Layout Skill Command

Route the user's request to the EmbodiedGen interactive layout workflow.

## Workflow

### Step 1: Interpret the request

Use `$ARGUMENTS` if provided. If it is empty, ask for the task description or task file path.

### Step 2: Load the skill

Use `skill: "embodiedgen:layout-creator"`.

### Step 3: Execute the workflow

Follow the skill and build the correct `layout-cli` command, including `--bg_list` and output settings.

### Step 4: Deliver

Return:
1. The exact command used
2. The output root
3. Expected runtime and any dependency warnings
