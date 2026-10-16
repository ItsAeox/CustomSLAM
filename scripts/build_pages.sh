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
mkdir -p "$OUT/build" "$OUT/assets" "$OUT/vocab" "$OUT/models"
cp "$W/ar.html" "$W/ar.js" "$W/calib.html" "$W/calib.js" "$W/slamclient.js" "$W/options.js" "$W/plane.js" "$W/slam.worker.js" "$W/coi.js" "$W/coi-sw.js" \
   "$W/selftest.html" "$W/xfeat.html" "$OUT/"
cp "$W/build/orbslam3.js" "$W/build/orbslam3.wasm" "$W/build/orbslam3.worker.mjs" "$OUT/build/"
if [ -f "$W/build-prof/orbslam3.wasm" ]; then  # profiling build (REGISTER_TIMES), loaded by ar.html?prof=1
  mkdir -p "$OUT/build-prof" && cp "$W/build-prof/orbslam3.js" "$W/build-prof/orbslam3.wasm" "$W/build-prof/orbslam3.worker.mjs" "$OUT/build-prof/"
fi
cp "$W/assets/tree.glb" "$W/assets/cactus.glb" "$OUT/assets/"  # tree.glb: the user's free_tree.blend, base trimmed
cp "$W/models/xfeat_640x480.onnx" "$W/models/xfeat_480x640.onnx" "$OUT/models/"  # scripts/xfeat_export.py
gzip -6 -n -c "$VOCAB" > "$OUT/vocab/ORBvoc.bin.gz"
TEBLID_VOCAB="$(dirname "$VOCAB")/TEBLIDvoc_own_k10L6.bin"  # optional: descriptor option 'TEBLID' (vocab_train)
if [ -f "$TEBLID_VOCAB" ]; then gzip -6 -n -c "$TEBLID_VOCAB" > "$OUT/vocab/TEBLIDvoc_own_k10L6.bin.gz"; fi
touch "$OUT/.nojekyll"  # serve files as-is (no Jekyll processing)
cat > "$OUT/index.html" <<'EOF'
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WebVIO</title>
<script src="./coi.js"></script>
<style>
  body { margin: 0; padding: 16px; background: #0f1115; color: #e6e6e6; font: 16px/1.5 system-ui, sans-serif; }
  a { color: #5bd18b; } li { margin: 10px 0; } code { color: #9aa0a6; }
</style>
</head>
<body>
<h1>WebVIO</h1>
<p>Markerless SLAM in the web browser: camera tracking, mapping and AR placement, running on this device.</p>
<ul>
  <li><a href="./ar.html"><b>AR: place a tree on the floor</b></a> (rear camera; resize the tree with the slider). Choose a
      <b>Preset</b> on the start screen (baseline, hybrid ORB + KLT, hybrid fastest, light ORB, IMU); the
      <b>Algorithm options</b> below it change single settings.</li>
  <li><a href="./calib.html">Camera calibration</a>: film a chessboard (shown on a laptop / monitor with the same page) so the
      AR page uses this phone's real focal length and lens distortion instead of an assumed 69° field of view. Needed for
      IMU mode (a 10 % focal error breaks the IMU's scale).</li>
</ul>
<p>In the AR page, <b>Log</b> shows the session log (tracking losses, relocalisation, map events). Copy or download it
   to report a test.</p>
<h2>Diagnostics</h2>
<ul>
  <li><a href="./selftest.html">Self-test</a>: checks this browser can run the SLAM (threads, SIMD, the 33 MB vocabulary
      download, start-up on synthetic frames). Use it when the AR page does not start.</li>
  <li><a href="./xfeat.html">XFeat benchmark</a>: research only. Times a learned feature detector (XFeat, CVPR 2024) on
      this device, GPU (WebGPU) vs CPU (WebAssembly). Not used by the AR page.</li>
</ul>
<p><code>The first visit installs a small service worker and reloads once (WebAssembly threads need it on static hosting).</code></p>
<p><code>WebVIO builds on parts of ORB-SLAM3 (GPL-3.0); credits and licence in the repository.</code></p>
</body>
</html>
EOF
du -sh "$OUT"
