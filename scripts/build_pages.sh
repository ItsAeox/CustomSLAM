#!/usr/bin/env bash
# Assemble the static web app for GitHub Pages into docs/ (Pages source: branch slam-rework, folder /docs).
#   - pages + JS from web/, the browser build from web/build/, the cactus model
#   - the vocabulary gzipped (48.7 MB -> 33 MB), decompressed in the browser (slam.worker.js)
#   - coi.js / coi-sw.js: service worker that adds the COOP/COEP headers GitHub Pages cannot send (WASM threads)
# Usage: scripts/build_pages.sh [path/to/ORBvoc.bin]   (default: ../vocab/ORBvoc.bin next to the repo)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VOCAB="${1:-$REPO/../vocab/ORBvoc.bin}"
OUT="$REPO/docs"
W="$REPO/web"
rm -rf "$OUT"
mkdir -p "$OUT/build" "$OUT/assets" "$OUT/vocab"
cp "$W/ar.html" "$W/ar.js" "$W/slamclient.js" "$W/options.js" "$W/plane.js" "$W/slam.worker.js" "$W/coi.js" "$W/coi-sw.js" \
   "$W/selftest.html" "$OUT/"
cp "$W/build/orbslam3.js" "$W/build/orbslam3.wasm" "$W/build/orbslam3.worker.mjs" "$OUT/build/"
cp "$W/assets/cactus.glb" "$OUT/assets/"
gzip -6 -n -c "$VOCAB" > "$OUT/vocab/ORBvoc.bin.gz"
touch "$OUT/.nojekyll"  # serve files as-is (no Jekyll processing)
cat > "$OUT/index.html" <<'EOF'
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Web SLAM</title>
<script src="./coi.js"></script>
<style>
  body { margin: 0; padding: 16px; background: #0f1115; color: #e6e6e6; font: 16px/1.5 system-ui, sans-serif; }
  a { color: #5bd18b; } li { margin: 10px 0; } code { color: #9aa0a6; }
</style>
</head>
<body>
<h1>Markerless SLAM in the browser</h1>
<p>ORB-SLAM3 (monocular) compiled to WebAssembly, with an ORB+KLT hybrid tracker. Everything runs on this device.</p>
<ul>
  <li><a href="./ar.html"><b>AR: place a cactus on the floor</b></a> (rear camera, hybrid ORB+KLT tracker)</li>
  <li><a href="./ar.html?tracker=stock">AR with stock ORB-SLAM3 tracking</a> (ORB on every frame, for comparison)</li>
  <li><a href="./selftest.html">Self-test</a>: threads, SIMD, vocabulary download (33 MB) and SLAM start-up</li>
</ul>
<p><code>The first visit installs a small service worker and reloads once (WebAssembly threads need it on static hosting).</code></p>
</body>
</html>
EOF
du -sh "$OUT"
