#!/usr/bin/env bash
# Download the Mixkit sample clips used in the gallery and cut them into
# 97-frame (8n+1) 24 fps plates. Mixkit free license: https://mixkit.co/license/
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .runtime/footage .runtime/plates
# id:start_seconds
for spec in 46762:3 1781:2 4915:2 49806:2 4852:0 4170:1 1218:3 50847:4 52427:0.5 50111:2; do
  id=${spec%%:*}; ss=${spec##*:}
  src=.runtime/footage/$id.mp4
  if [[ ! -s $src ]]; then
    curl -sfL -A "Mozilla/5.0" -o "$src" "https://assets.mixkit.co/videos/$id/$id-1080.mp4" \
      || curl -sfL -A "Mozilla/5.0" -o "$src" "https://assets.mixkit.co/videos/$id/$id-720.mp4"
  fi
  ffmpeg -v error -y -ss "$ss" -i "$src" -vf "fps=24,scale=-2:'min(1080,ih)':flags=lanczos" \
    -frames:v 97 -an -c:v libx264 -crf 12 -pix_fmt yuv420p ".runtime/plates/$id.mp4"
  echo "plate $id ready"
done
