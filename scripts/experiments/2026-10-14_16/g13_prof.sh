#!/usr/bin/env bash
# Profiling browser module (?prof=1) with the G13 fix, after the reproduction runs (-j1, nice: inertial sweep running).
set -uo pipefail
until grep -q "=== done" /mnt/h/Research/logs/g13_fix2.log 2>/dev/null; do sleep 30; done
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
echo "=== start $(date +%T)"
nice -n 10 cmake --build /root/build/slam-wasm-prof -j1 --target orbslam3_web_js 2>&1 | grep -E "error|FAILED|Linking"
mkdir -p /mnt/h/Research/CustomSLAM/web/build-prof
cp /root/build/slam-wasm-prof/orbslam3.js /root/build/slam-wasm-prof/orbslam3.wasm /root/build/slam-wasm-prof/orbslam3.worker.mjs /mnt/h/Research/CustomSLAM/web/build-prof/
cd /mnt/h/Research/CustomSLAM && bash scripts/build_pages.sh > /dev/null && git status --short docs
echo "=== done $(date +%T)"
