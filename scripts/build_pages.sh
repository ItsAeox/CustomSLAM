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
cp "$W/ar.html" "$W/ar.js" "$W/calib.html" "$W/calib.js" "$W/slamclient.js" "$W/options.js" "$W/plane.js" "$W/imucalib.js" "$W/imuscale.js" "$W/minimap.js" "$W/slam.worker.js" "$W/coi.js" "$W/coi-sw.js" \
   "$W/selftest.html" "$W/xfeat.html" "$W/gpubench.html" "$W/gpubench.js" "$W/gpubench.worker.js" "$W/viewer.html" "$W/viewer.js" "$W/align.js" "$OUT/"
cp "$W/build/orbslam3.js" "$W/build/orbslam3.wasm" "$W/build/orbslam3.worker.mjs" "$OUT/build/"
if [ -f "$W/build-prof/orbslam3.wasm" ]; then  # profiling build (REGISTER_TIMES), loaded by ar.html?prof=1
  mkdir -p "$OUT/build-prof" && cp "$W/build-prof/orbslam3.js" "$W/build-prof/orbslam3.wasm" "$W/build-prof/orbslam3.worker.mjs" "$OUT/build-prof/"
fi
cp "$W/assets/tree.glb" "$W/assets/cactus.glb" "$OUT/assets/"  # tree.glb: the user's free_tree.blend, base trimmed
cp "$W/models/xfeat_640x480.onnx" "$W/models/xfeat_480x640.onnx" "$OUT/models/"  # scripts/xfeat_export.py
gzip -6 -n -c "$VOCAB" > "$OUT/vocab/ORBvoc.bin.gz"
TEBLID_VOCAB="$(dirname "$VOCAB")/TEBLIDvoc_own_k10L6.bin"  # optional: descriptor option 'TEBLID' (vocab_train)
if [ -f "$TEBLID_VOCAB" ]; then gzip -6 -n -c "$TEBLID_VOCAB" > "$OUT/vocab/TEBLIDvoc_own_k10L6.bin.gz"; fi
SMALL_VOCAB="$(dirname "$VOCAB")/ORBvoc_own_k10L5.bin"  # optional: vocabulary option 'small' (5 MB, vocab_eval 2026-10-17)
if [ -f "$SMALL_VOCAB" ]; then gzip -6 -n -c "$SMALL_VOCAB" > "$OUT/vocab/ORBvoc_own_k10L5.bin.gz"; fi
touch "$OUT/.nojekyll"  # serve files as-is (no Jekyll processing)
cp "$W/index.html" "$OUT/index.html"  # web/index.html adapts: on Pages only what works without the dev server
du -sh "$OUT"
