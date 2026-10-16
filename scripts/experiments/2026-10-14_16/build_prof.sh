#!/usr/bin/env bash
# Profiling builds (REGISTER_TIMES): native, WASM Node harness, WASM browser module.
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
echo "=== native prof $(date +%T)"
cmake -S slam -B /root/build/slam-native-prof -G Ninja -DCMAKE_BUILD_TYPE=Release -DWEB_PROFILE=ON \
  -DOpenCV_DIR=/opt/slam/opencv-4.10.0-native/lib/cmake/opencv4 > /dev/null
cmake --build /root/build/slam-native-prof --target orbslam3_run -j 3 2>&1 | grep -E "error|FAILED" || true
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
echo "=== wasm prof $(date +%T)"
emcmake cmake -S slam -B /root/build/slam-wasm-prof -G Ninja -DBUILD_WASM=ON -DWEB_PROFILE=ON -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/opt/slam/opencv-4.10.0-wasm-mt/lib/cmake/opencv4 > /dev/null
cmake --build /root/build/slam-wasm-prof --target orbslam3_run orbslam3_web_js -j 3 2>&1 | grep -E "error:|FAILED" || true
ls -la /root/build/slam-native-prof/orbslam3_run /root/build/slam-wasm-prof/orbslam3_run.js /root/build/slam-wasm-prof/orbslam3.wasm
echo "=== done $(date +%T)"
