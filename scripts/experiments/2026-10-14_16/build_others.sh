#!/usr/bin/env bash
# Rebuild every build dir except slam-native (in use by the gate runs).
set -uo pipefail
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
for b in slam-native-prof slam-wasm slam-wasm-prof; do
  echo "=== $b $(date +%T)"
  cmake --build /root/build/$b -j2 2>&1 | grep -E "error|FAILED" || true
done
cp /root/build/slam-wasm/orbslam3.js /root/build/slam-wasm/orbslam3.wasm /root/build/slam-wasm/orbslam3.worker.mjs \
   /mnt/h/Research/CustomSLAM/web/build/
echo "=== done $(date +%T)"
