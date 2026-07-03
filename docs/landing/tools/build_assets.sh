#!/usr/bin/env bash
# Idempotent asset build for the EmbodiedGen V2 webpage.
# Produces: figures (JPG), brand logo (PNG), sim-ready models + assets.json,
# vibe-coding scene clips, and affordance data.
# NOTE: the five master-derived section videos (hero / scenes_gen / worlds /
# multi_sim / closed_loop) are produced by tools/cut_videos_from_master.sh,
# NOT here. Sources are READ-ONLY. Outputs -> docs/landing/assets/{img,video,models}
set -euo pipefail

# repo root (this script lives at docs/landing/tools/)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SRC_VID="$ROOT/outputs/embodiedgenv2/video"
SRC_FIG="$ROOT/outputs/embodiedgenv2/paper/figure"
IMG="$ROOT/docs/landing/assets/img"
VID="$ROOT/docs/landing/assets/video"
MOD="$ROOT/docs/landing/assets/models"
mkdir -p "$IMG" "$VID" "$MOD"

# Render a source (PDF first page, or raster image) to a web JPG (<=1MB, <=1920w wide).
fig2jpg() {  # <src> <dst.jpg>
  python3 - "$1" "$2" <<'PY'
import sys, os, subprocess, tempfile
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]
if src.lower().endswith(".pdf"):
    base = tempfile.mktemp(prefix="_fig")
    subprocess.run(["pdftoppm", "-png", "-r", "150", "-singlefile", src, base], check=True)
    src = base + ".png"
im = Image.open(src).convert("RGB")
w, h = im.size
if w > 1920:
    im = im.resize((1920, round(h * 1920 / w)), Image.LANCZOS)
for q in (90, 88, 85, 82, 78, 74):
    im.save(dst, "JPEG", quality=q, optimize=True, progressive=True)
    if os.path.getsize(dst) <= 1024 * 1024:
        break
print(f"  -> {os.path.basename(dst)} {im.size} q={q} {os.path.getsize(dst)//1024}K")
PY
}

echo "[1/4] Figures -> JPG (<=1MB)"
fig2jpg "$SRC_FIG/sim_ready_asset_gen.pdf"        "$IMG/sim_pipeline.jpg"
fig2jpg "$SRC_FIG/overview.pdf"                   "$IMG/overview.jpg"
fig2jpg "$SRC_FIG/layout_pipe.pdf"                "$IMG/layout_pipe.jpg"
fig2jpg "$SRC_FIG/large_scale_scene_gen.pdf"      "$IMG/large_scale.jpg"
fig2jpg "$SRC_FIG/heatmap_grid_2x6.png"           "$IMG/deformable_heatmap.jpg"
fig2jpg "$SRC_FIG/policy_learning_and_deploy.png" "$IMG/policy_deploy.jpg"

echo "[1b] Brand logo (keep PNG transparency, trim margins)"
python3 - "$ROOT/docs/documentation/assets/logo.png" "$IMG/logo.png" <<'PY'
import sys; from PIL import Image
im = Image.open(sys.argv[1]).convert("RGBA"); bbox = im.getbbox()
if bbox: im = im.crop(bbox)
im.save(sys.argv[2]); print("  -> logo.png", im.size)
PY

echo "[2/3] Sim-ready asset viewer models (visual + collision + affordance + json)"
# All three viewer variants, assets.json and affordance.json for every complete
# asset in the project_example_v2 export. Requires the horizon-bucket mount.
python3 "$ROOT/docs/landing/tools/build_models_v2.py" "$MOD"

echo "[3/3] Vibe-coding scene clips (kitchen/living, S0-2) -> 1080w/crf26"
for n in kitchen_S0 kitchen_S1 kitchen_S2 living_room_S0 living_room_S1 living_room_S2; do
  ffmpeg -y -i "$SRC_VID/$n.mp4" -vf "scale='min(1080,iw)':-2" -c:v libx264 -crf 26 -preset slow -an \
    -movflags +faststart "$VID/scene_$n.mp4" -hide_banner -loglevel error
done

echo "Done. (Section videos come from tools/cut_videos_from_master.sh.)"
du -h "$IMG"/*.jpg "$VID"/scene_*.mp4 "$MOD"/*.glb 2>/dev/null | sort -h | tail -24
