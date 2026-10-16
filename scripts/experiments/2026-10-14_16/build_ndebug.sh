#!/usr/bin/env bash
# Rebuild with -DNDEBUG (CMakeLists change regenerates; every object recompiles). One dir at a time, -j2, nothing else
# running (WSL 7.8 GB). Order: browser module + WASM runner first (the user's phone test), then native, then profiling.
set -uo pipefail
LOG=/mnt/h/Research/logs/build_ndebug.log
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
chk() {  # -DNDEBUG in the generated build + the assert string gone from the binary
  echo "  $1: NDEBUG in build.ninja: $(grep -c 'DNDEBUG' $1/build.ninja); assert string in $2: $(grep -c 'mit->second>=3' $2)"
}
{
  echo "=== start $(date +%T)"
  nice -n 10 cmake --build /root/build/slam-wasm-ui -j2 --target orbslam3_web_js orbslam3_run 2>&1 | grep -E "error|FAILED"
  chk /root/build/slam-wasm-ui /root/build/slam-wasm-ui/orbslam3.wasm
  chk /root/build/slam-wasm-ui /root/build/slam-wasm-ui/orbslam3_run.wasm
  cp /root/build/slam-wasm-ui/orbslam3.js /root/build/slam-wasm-ui/orbslam3.wasm /root/build/slam-wasm-ui/orbslam3.worker.mjs /mnt/h/Research/CustomSLAM/web/build/
  echo "  web/build: assert string $(grep -c 'mit->second>=3' /mnt/h/Research/CustomSLAM/web/build/orbslam3.wasm)"
  echo "=== wasm-ui done $(date +%T)"
  nice -n 10 cmake --build /root/build/slam-native-teblid -j2 --target orbslam3_run vocab_train 2>&1 | grep -E "error|FAILED"
  chk /root/build/slam-native-teblid /root/build/slam-native-teblid/orbslam3_run
  echo "=== native done $(date +%T)"
  nice -n 10 cmake --build /root/build/slam-wasm-prof -j2 --target orbslam3_web_js 2>&1 | grep -E "error|FAILED"
  chk /root/build/slam-wasm-prof /root/build/slam-wasm-prof/orbslam3.wasm
  mkdir -p /mnt/h/Research/CustomSLAM/web/build-prof
  cp /root/build/slam-wasm-prof/orbslam3.js /root/build/slam-wasm-prof/orbslam3.wasm /root/build/slam-wasm-prof/orbslam3.worker.mjs /mnt/h/Research/CustomSLAM/web/build-prof/
  echo "=== all done $(date +%T)"
} >> $LOG 2>&1
