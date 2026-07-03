# EmbodiedGen Skills

This directory is the canonical home for EmbodiedGen reusable skills.

The root of `embodied_gen/skills` only contains generic skill source.
Runtime-specific packaging should live in adapter subdirectories such as
`embodied_gen/skills/claude_plugin/`.

## Included generic skills

- `asset-creator`
- `asset-retrieval`
- `background-creator`
- `layout-creator`
- `sim-runner`
- `asset-converter`
- `asset-process`
- `room-creator`
- `spatial-computing`

## Claude plugin package

Claude-compatible slash commands and plugin manifest are under:

```text
embodied_gen/skills/claude_plugin/
```

The local marketplace manifest is:

```text
embodied_gen/skills/claude_plugin/.claude-plugin/marketplace.json
```

The actual Claude plugin package is:

```text
embodied_gen/skills/claude_plugin/plugins/embodiedgen/
```

Current commands include:

- `/embodiedgen:gen_assets`
- `/embodiedgen:gen_bg`
- `/embodiedgen:gen_layout`
- `/embodiedgen:sim`
- `/embodiedgen:convert`
- `/embodiedgen:process`
- `/embodiedgen:gen_indoor`
- `/embodiedgen:vibe3d`

## Local install for Claude

```bash
bash install/install_agent_plugin.sh
```

## Notes

- Generic skills stay in their original directories under `embodied_gen/skills/`.
- Claude-specific files live only under `embodied_gen/skills/claude_plugin/`.
- This keeps the skill source portable for Codex, Copilot, and other runtimes.
