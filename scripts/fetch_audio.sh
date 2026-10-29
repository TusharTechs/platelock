#!/usr/bin/env bash
# Music + sound effects for the demo videos, from Mixkit (free license: https://mixkit.co/license/).
set -euo pipefail
cd "$(dirname "$0")/.." && mkdir -p .runtime/audio && cd .runtime/audio
get() { [[ -s $2 ]] || curl -sfL -A "Mozilla/5.0" -o "$2" "$1"; }
get https://assets.mixkit.co/music/134/134.mp3 music_main.mp3                              # "Deep Techno Ambience"
for spec in 1143:impact_deep 1490:whoosh_fast 1492:whoosh_cine_fast 2350:sparkle 2900:logo_impact 2853:safe_lock; do
  id=${spec%%:*}; get "https://assets.mixkit.co/active_storage/sfx/$id/$id-preview.mp3" "sfx_${spec##*:}.mp3"
done
ls -1
