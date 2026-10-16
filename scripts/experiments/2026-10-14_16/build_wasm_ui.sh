#!/usr/bin/env bash
# Browser module with the latest source (reloc timeout, KLT gates, TEBLID option), separate build dir, lowest priority.
set -uo pipefail
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
B=/root/build/slam-wasm-ui
if [ ! -f $B/build.ninja ]; then
  emcmake cmake -S /mnt/h/Research/CustomSLAM/slam -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_WASM=ON     -DWEB_SIMD=ON -DOpenCV_DIR=/opt/slam/opencv-4.10.0-wasm-mt/lib/cmake/opencv4 > /dev/null
fi
nice -n 19 cmake --build $B -j4 --target orbslam3_web_js 2>&1 | grep -E "error|FAILED" || true
ls -la $B/orbslam3.js $B/orbslam3.wasm $B/orbslam3.worker.mjs
echo "built $(date +%T)"
