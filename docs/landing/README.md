# EmbodiedGen V2 — Project Landing Page

Standalone static site (`docs/landing/`). No build step, no framework, no dependencies —
open `index.html` over any static server.

On GitHub Pages this is published at the site **root** (`https://horizonrobotics.github.io/EmbodiedGen/`);
the MkDocs documentation is published under **`/docs/`**. See `.github/workflows/deploy_docs.yml`.

## Local preview
```bash
cd docs/landing
python3 -m http.server 8123
# open http://localhost:8123
```
Fonts load from Google Fonts (needs internet); a system fallback stack is configured.
The repo `.gitignore` ignores `*.html` / `*.mp4` / `*.json` globally, but re-includes them under
`docs/landing/**` (see the negation block at the end of `.gitignore`), so these site files are tracked normally.

## Files
| File | Responsibility |
|------|----------------|
| `index.html` | Structure + all section containers |
| `style.css`  | Design system (CSS vars, type, grid, components, responsive) — single source of truth |
| `script.js`  | Scroll-spy, reveal-on-scroll, lazy video play/pause, asset 3D viewer, vibe sessions, count-up, BibTeX copy |
| `tools/cut_videos_from_master.sh` | Section videos cut from the master demo video |
| `tools/build_assets.sh` | Figures (JPG), brand logo, sim-ready models + `assets.json`, vibe clips, affordance |
| `tools/build_affordance.py` | Part-seg GLB + baked grippers + `affordance.json` |
| `favicon.svg` + `favicon-{32,180}.png` | Browser tab icon — brand-blue 3D cube (SVG primary, PNG fallback / apple-touch). Static brand asset. |

## Assets — how they're produced (idempotent)
Two scripts own disjoint sets of outputs under `assets/`:

- **`tools/cut_videos_from_master.sh`** → the five section videos + posters
  (`hero_bg`, `scenes_gen`, `worlds`, `multi_sim`, `closed_loop`), cut as timestamped
  segments from the master demo video. The master `.mov` is **not** kept in the repo.
- **`tools/build_assets.sh`** → everything else: figure JPGs (≤1 MB), `logo.png`,
  the GLB models + `assets.json`, the six `scene_*` vibe clips, and affordance data.
  Reads READ-ONLY sources from `outputs/embodiedgenv2/{paper/figure,video}` and the asset buckets.

## Performance notes (production)
- Section videos use `preload="none"` + `data-autoplay`: fetched and played only when
  scrolled into view, paused when offscreen (see `script.js`).
- The `model-viewer` library (~1 MB) and the first GLB load lazily — warmed right after
  page load (idle) or when the Assets section nears, whichever comes first.
- Hero streams progressively (`+faststart` + poster). Large figures are JPG inside
  collapsed `<details>` with `loading="lazy"`, so they don't load until expanded.
- Server side (deploy): enable brotli/gzip, long `Cache-Control` for hashed assets, a CDN,
  and consider self-hosting `model-viewer` instead of the unpkg CDN.

## Pre-release TODO
- [ ] Replace placeholder authors with the final list.
- [ ] Point the Dataset button to the final HF dataset URL.
- [ ] Confirm the arXiv number for EmbodiedGen V2 (currently `2506.10600`).
- [ ] Add an OG/share image.
- [ ] Capture acceptance screenshots at 1440 / 1024 / 390.
