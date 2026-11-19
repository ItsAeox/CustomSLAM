#!/usr/bin/env bash
# WASM build of the working tree (Node runner + browser module into web/build), lowest priority (screens running)
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
echo "=== start $(date +%T)"
nice -n 19 cmake --build /root/build/slam-wasm-ui -j1 --target orbslam3_run orbslam3_web_js 2>&1 | grep -E "error|FAILED" | head -20
ls -la --time-style=+%T /root/build/slam-wasm-ui/orbslam3_run.wasm /mnt/h/Research/CustomSLAM/web/build/orbslam3.wasm
echo "=== done $(date +%T)"
