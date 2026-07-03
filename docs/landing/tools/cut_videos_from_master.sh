#!/usr/bin/env bash
# Re-cut landing section videos from the latest master video.
# Source (READ-ONLY): docs/landing/assets/EmbodiedGenV2_0627_web.mov (kept out of the repo)
# Outputs (idempotent, overwritten): docs/landing/assets/video/<slot>.mp4 + assets/img/<slot>_poster.jpg
# Timestamps locked in outputs/embodiedgenv2/webpage_polish/WORKLOG.md (T0.1).
# Does NOT touch the 6 scene_*_S* vibe-coding clips.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # docs/landing/
MASTER="$ROOT/assets/EmbodiedGenV2_0627_web.mov"
VID="$ROOT/assets/video"
IMG="$ROOT/assets/img"
mkdir -p "$VID" "$IMG"
[ -f "$MASTER" ] || { echo "ERROR: master not found: $MASTER" >&2; exit 1; }

# slot | in(s) | dur(s) | width | crf | poster_offset(s into clip) | poster_basename
# NOTE: index.html references the hero poster as `hero_poster.jpg` (not hero_bg_poster).
# NOTE: multi_sim has ~8px black pillarbars in the master; after this step it was
#   cropped with: ffmpeg -i multi_sim.mp4 -vf "crop=1904:1080:8:0" -crf 16 ... (full height kept).
SLOTS=(
  "hero_bg|0|35|1600|32|17|hero"
  "scenes_gen|80|20|1200|27|10|scenes_gen"
  "worlds|51.83|15.6|1200|27|5|worlds"
  "multi_sim|68|3|1920|18|1.5|multi_sim"
  "closed_loop|127|15|1200|27|12|closed_loop"
)

cut_one() {
  local name=$1 ss=$2 dur=$3 w=$4 crf=$5 poff=$6 pname=$7
  local out="$VID/$name.mp4" poster="$IMG/${pname}_poster.jpg"
  echo "[cut] $name  in=${ss}s dur=${dur}s ${w}w crf${crf}"
  ffmpeg -y -hide_banner -loglevel error \
    -ss "$ss" -t "$dur" -i "$MASTER" \
    -map 0:v:0 -an \
    -vf "scale='min(${w},iw)':-2:flags=lanczos" \
    -c:v libx264 -crf "$crf" -preset slow -pix_fmt yuv420p \
    -movflags +faststart "$out"
  ffmpeg -y -hide_banner -loglevel error \
    -ss "$poff" -i "$out" -vframes 1 -q:v 4 "$poster"
}

for row in "${SLOTS[@]}"; do
  IFS='|' read -r name ss dur w crf poff pname <<<"$row"
  cut_one "$name" "$ss" "$dur" "$w" "$crf" "$poff" "$pname"
done

echo "Done. New section videos + posters:"
for row in "${SLOTS[@]}"; do
  IFS='|' read -r name _ _ _ _ _ <<<"$row"
  d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VID/$name.mp4")
  wh=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$VID/$name.mp4")
  sz=$(du -h "$VID/$name.mp4" | cut -f1)
  printf "  %-14s %ss  %s  %s\n" "$name.mp4" "$d" "$wh" "$sz"
done
