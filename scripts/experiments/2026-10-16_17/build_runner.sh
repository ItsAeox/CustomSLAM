#!/usr/bin/env bash
# Rebuild only the dataset runner (orbslam3_run.cc changes; the ORB-SLAM3 library is unchanged), native + WASM, -j1.
set -uo pipefail
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
echo "=== start $(date +%T)"
nice -n 10 cmake --build /root/build/slam-native-teblid -j1 --target orbslam3_run 2>&1 | grep -E "error|FAILED|Linking|Building"
nice -n 10 cmake --build /root/build/slam-wasm-ui -j1 --target orbslam3_run 2>&1 | grep -E "error|FAILED|Linking|Building"
echo "=== done $(date +%T)"
