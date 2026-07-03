---
name: room-creator
description: Generate indoor rooms (single room or house) and export URDF/USD by wrapping embodied_gen/scripts/room_gen/gen_room.py. Use when users ask to create rooms with seed control, choose room type and complexity, run generation/export stages, or run reproducible room generation jobs (batch runs can be done by wrapping this command in an outer loop/script).
---

# Room Creator

Generate room scenes with `python -m embodied_gen.scripts.room_gen.gen_room` from infinigen(https://github.com/princeton-vl/infinigen) and optionally export URDF/USD.

## Use This Workflow

1. Confirm output root and target room profile.
2. Choose generation scope:
- `--gen --urdf --usd` for full pipeline.
- `--gen --no-urdf --no-usd` for generation only.
- `--no-gen --urdf --usd` for export from existing blender output.
3. Run the command from repository root.
4. Verify output folder: `<output_root>/<RoomType>_seed<seed>/` (if `--seed` is omitted, check the generated seed from logs first).

## Parameters

- `--output-root` (required): base output directory.
- `--prompt`: natural-language task/scene description; a GPT router infers room type and complexity from it, overriding `--room-type`/`--complexity`. Requires a configured GPT agent.
- `--room-type`: `Bedroom | LivingRoom | Kitchen | Bathroom | DiningRoom | Office | House`.
- `--seed`: random seed. For reproducible runs, pass this explicitly; if omitted, a random seed is generated.
- `--complexity`: `minimalist | simple | medium | detail`.
- `--custom-params`: gin file copied to Infinigen `custom_solve.gin`.
- `--large-scene`: only for `House`; enables more rooms.
- `--gen/--no-gen`, `--urdf/--no-urdf`, `--usd/--no-usd`: pipeline switches.

## Complexity Guidance

- `minimalist`: fastest, sparse furniture.
- `simple`: default, balanced quality/time.
- `medium`: richer layout, slower.
- `detail`: highest detail, longest runtime.

## Command Templates

```bash
# Full pipeline for one kitchen
python -m embodied_gen.scripts.room_gen.gen_room \
  --output-root outputs/rooms \
  --room-type Kitchen \
  --seed 42 \
  --complexity simple
```

```bash
# Generation only (no export)
python -m embodied_gen.scripts.room_gen.gen_room \
  --output-root outputs/rooms \
  --room-type LivingRoom \
  --seed 100 \
  --complexity medium \
  --no-urdf --no-usd
```

```bash
# Export only from existing blender result
python -m embodied_gen.scripts.room_gen.gen_room \
  --output-root outputs/rooms \
  --room-type Kitchen \
  --seed 42 \
  --no-gen --urdf --usd
```

```bash
# House generation (use --large-scene for more rooms)
python -m embodied_gen.scripts.room_gen.gen_room \
  --output-root outputs/rooms \
  --room-type House \
  --seed 7 \
  --complexity simple \
  --large-scene
```

## Runtime Requirements

- Run from repo root so relative paths resolve.
- Ensure Blender Python exists at `$BLENDER_PYTHON`. If not set, the command will try:
  `thirdparty/infinigen/blender/4.2/python/bin/python3.11` (must exist on disk).
- `--no-gen` requires existing blender output at:
  `<output_root>/<RoomType>_seed<seed>/blender`.
