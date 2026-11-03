#!/usr/bin/env bash
# Profiling browser module (AR ?prof=1, WEB_PROFILE=ON) for the paper phone session, part A (claude/PHONE.md).
# Same SLAM code as snap-84c1f17 / web/build (no SLAM source change since 84c1f17, only baselines and scripts); the
# slam-wasm-prof cache differs from slam-wasm-ui only in WEB_PROFILE. Nothing else running: -j6.
set -uo pipefail
LOG=/mnt/h/Research/logs/build_prof_1103.log
W=/mnt/h/Research/CustomSLAM/web/build-prof
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
{
  echo "=== start $(date +%T) HEAD $(git -C /mnt/h/Research/CustomSLAM rev-parse --short HEAD)"
  nice -n 10 cmake --build /root/build/slam-wasm-prof -j6 --target orbslam3_web_js 2>&1 | grep -E "error|FAILED|Linking"
  ls -la --time-style=full-iso /root/build/slam-wasm-prof/orbslam3.wasm
  mkdir -p $W
  cp /root/build/slam-wasm-prof/orbslam3.js /root/build/slam-wasm-prof/orbslam3.wasm /root/build/slam-wasm-prof/orbslam3.worker.mjs $W/
  md5sum $W/orbslam3.wasm
  echo "=== done $(date +%T)"
} >> $LOG 2>&1
