#!/usr/bin/env bash
# WASM dataset runner + browser module with the time-scaled motion model and --live pacing (slam-wasm-ui, -j1:
# WSL has 7 GB). Copies the browser module to web/build.
set -uo pipefail
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
B=/root/build/slam-wasm-ui
nice -n 19 cmake --build $B -j1 --target orbslam3_run orbslam3_web_js 2>&1 | grep -E "error|FAILED" || true
ls -la --time-style=+%T $B/orbslam3_run.js $B/orbslam3.wasm
cp $B/orbslam3.js $B/orbslam3.wasm $B/orbslam3.worker.mjs /mnt/h/Research/CustomSLAM/web/build/
echo "built $(date +%T)"
