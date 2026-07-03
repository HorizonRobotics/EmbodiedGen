---
name: asset-retrieval
description: Retrieve existing EmbodiedGen assets from a configurable dataset index CSV by natural-language descriptions and return matching `.urdf` paths. Use when users describe an asset they want, ask to find one or several existing URDF assets, or need a fast lookup over the local asset index instead of scanning folders manually.
---

# Asset Retrieval

Look up simulation-ready assets from `dataset_index.csv` and return `.urdf`
paths. The CSV index is the single source of truth.

## Workflow

**Preferred — agent reads CSV directly:**

1. Read `dataset_index.csv` into context.
2. Semantically match the user's description (open-ended, fuzzy, or in any
   language) against `category`, `secondary_category`, `primary_category`,
   and `description` columns.
3. Return the best-matching absolute `.urdf` path; return multiple candidates
   when the request is broad or explicitly asks for several.
4. Briefly explain why the returned asset matches.

This path handles open-ended queries like "a tall chair suitable for a
coffee shop" or "能放在客厅角落的落地灯" that pure keyword matching cannot
resolve.

**Fallback — CLI script (no network / no LLM):**

When the agent is unavailable, use the helper script which performs offline
keyword-based ranking:

```bash
python embodied_gen/skills/asset-retrieval/scripts/retrieve_asset.py \
  "modern dining chair curved backrest"
```

For the CLI path, rewrite open-ended or Chinese descriptions into compact
English keywords first (e.g. `能放在客厅角落的落地灯` → `floor lamp`).

## Index Resolution

Checked in order — first match wins:

1. `--index-file` CLI argument
2. `$EMBODIEDGEN_DATASET_INDEX` environment variable
3. `$EMBODIEDGEN_DATASET_ROOT/dataset_index.csv`
4. `<repo-root>/outputs/EmbodiedGenData/dataset/dataset_index.csv`

Dataset root follows a parallel order (`--dataset-root` →
`$EMBODIEDGEN_DATASET_ROOT` → repo default).

### Required CSV Columns

`uuid`, `primary_category`, `secondary_category`, `category`, `description`,
`generate_time`, `urdf_path`

## Query Guidelines

- Use explicit object words: `chair`, `bar stool`, `remote control`.
- Keep discriminating modifiers: `wooden`, `orange`, `modern`, `round`.
- Open-ended or Chinese descriptions are fine for the agent path; rewrite
  to English keywords only when using the CLI script.

## Script Usage

```bash
# Single best match (absolute path on stdout)
python embodied_gen/skills/asset-retrieval/scripts/retrieve_asset.py \
  "modern dining chair curved backrest"

# Multiple candidates with scores
python embodied_gen/skills/asset-retrieval/scripts/retrieve_asset.py \
  "orange cushioned bar stool" \
  --top-k 5 --format json

# Custom dataset location
python embodied_gen/skills/asset-retrieval/scripts/retrieve_asset.py \
  "black remote control" \
  --dataset-root /path/to/dataset \
  --index-file /path/to/dataset/dataset_index.csv

# Relative paths instead of absolute
python embodied_gen/skills/asset-retrieval/scripts/retrieve_asset.py \
  "wooden bar stool" --relative-paths
```

Exit code 1 with `"No matching assets found."` on stderr when nothing matches.
