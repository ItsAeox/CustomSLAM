#!/usr/bin/env bash
# Native then WASM build of the working tree (idle PC: -j8); browser module -> web/build
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
echo "=== start $(date +%T)"
nice -n 10 cmake --build /root/build/slam-native-teblid -j8 --target orbslam3_run 2>&1 | grep -E "error|FAILED" | head -20
echo "=== native $(date +%T)"
nice -n 10 cmake --build /root/build/slam-wasm-ui -j8 --target orbslam3_run orbslam3_web_js 2>&1 | grep -E "error|FAILED" | head -20
cp /root/build/slam-wasm-ui/orbslam3.js /root/build/slam-wasm-ui/orbslam3.wasm /root/build/slam-wasm-ui/orbslam3.worker.mjs /mnt/h/Research/CustomSLAM/web/build/
ls -la --time-style=+%T /root/build/slam-native-teblid/orbslam3_run /root/build/slam-wasm-ui/orbslam3_run.wasm /mnt/h/Research/CustomSLAM/web/build/orbslam3.wasm
echo "=== done $(date +%T)"
