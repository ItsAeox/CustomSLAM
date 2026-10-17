#!/usr/bin/env bash
# Re-configure (new source files are found by file(GLOB) only at configure time), then build native and WASM at -j4
# (-j8 ran out of memory on Tracking.cc); browser module -> web/build
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
echo "=== start $(date +%T)"
for d in /root/build/slam-native-teblid /root/build/slam-wasm-ui; do cmake $d > /dev/null 2>&1 || echo "configure failed: $d"; done
nice -n 10 cmake --build /root/build/slam-native-teblid -j4 --target orbslam3_run vocab_train 2>&1 | grep -E "error|FAILED" | head -30
echo "=== native $(date +%T)"
nice -n 10 cmake --build /root/build/slam-wasm-ui -j4 --target orbslam3_run orbslam3_web_js 2>&1 | grep -E "error|FAILED" | head -30
cp /root/build/slam-wasm-ui/orbslam3.js /root/build/slam-wasm-ui/orbslam3.wasm /root/build/slam-wasm-ui/orbslam3.worker.mjs /mnt/h/Research/CustomSLAM/web/build/
ls -la --time-style=+%T /root/build/slam-native-teblid/orbslam3_run /root/build/slam-wasm-ui/orbslam3_run.wasm /mnt/h/Research/CustomSLAM/web/build/orbslam3.wasm
echo "=== done $(date +%T)"
