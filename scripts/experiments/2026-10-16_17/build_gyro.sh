#!/usr/bin/env bash
# Build the gyro prior (WebConfig.h + Tracking.h changed: most of ORB-SLAM3 recompiles) + H3 runner: native, then
# WASM (runner + browser module). -j1, nice: a native sweep (imu_sens, pinned snapshot) is running alongside.
set -uo pipefail
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
echo "=== start $(date +%T)"
nice -n 10 cmake --build /root/build/slam-native-teblid -j1 --target orbslam3_run 2>&1 | grep -E "error|FAILED|warning: unused|Linking"
echo "=== native done $(date +%T)"
nice -n 10 cmake --build /root/build/slam-wasm-ui -j1 --target orbslam3_run orbslam3_web_js 2>&1 | grep -E "error|FAILED|Linking"
echo "=== wasm-ui done $(date +%T)"
