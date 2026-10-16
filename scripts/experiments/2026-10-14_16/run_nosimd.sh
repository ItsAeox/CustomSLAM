#!/usr/bin/env bash
# After the sync sweep: WASM build without -msimd128 for the port code, then a real-time sweep.
set -uo pipefail
until grep -q "=== done" /mnt/h/Research/logs/seed_sweep_sync.log 2>/dev/null; do sleep 10; done
cd /mnt/h/Research/CustomSLAM
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
echo "=== wasm no-simd build $(date +%T)"
emcmake cmake -S slam -B /root/build/slam-wasm-nosimd -G Ninja -DBUILD_WASM=ON -DWEB_SIMD=OFF -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/opt/slam/opencv-4.10.0-wasm-mt/lib/cmake/opencv4 > /dev/null
cmake --build /root/build/slam-wasm-nosimd --target orbslam3_run -j 3 2>&1 | grep -E "error:|FAILED" || true
ls -la /root/build/slam-wasm-nosimd/orbslam3_run.js
R=/mnt/h/Research/results/scratch
echo "=== sweep no-simd $(date +%T)"
FLAVOURS="wasm" WASM_JS=/root/build/slam-wasm-nosimd/orbslam3_run.js EXTRA="--realtime" \
  bash scripts/seed_sweep.sh "$R/seed_sweep_nosimd" "1 2 3 4 5" \
  "mocap-desk2:$R/settings/tumvie_mocap-desk2.yaml:$R/mocap-desk2/gt_tum.txt"
/opt/slam/venv/bin/python eval/summarize_sweep.py "$R/seed_sweep_nosimd"
echo "=== done $(date +%T)"
