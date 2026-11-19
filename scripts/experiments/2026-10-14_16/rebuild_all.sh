#!/usr/bin/env bash
# Rebuild native + WASM (driver, numerics, browser module) after source changes; copy the browser module to web/build.
set -uo pipefail
echo "=== native $(date +%T)"
cmake --build /root/build/slam-native -j3 2>&1 | grep -E "error|FAILED|Linking" || true
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
echo "=== wasm $(date +%T)"
cmake --build /root/build/slam-wasm -j3 2>&1 | grep -E "error|FAILED|Linking" || true
cp /root/build/slam-wasm/orbslam3.js /root/build/slam-wasm/orbslam3.wasm /root/build/slam-wasm/orbslam3.worker.mjs \
   /mnt/h/Research/CustomSLAM/web/build/
ls -la /mnt/h/Research/CustomSLAM/web/build/
echo "=== done $(date +%T)"
