#!/usr/bin/env bash
# ISSUES G40 / G41: the merge record (LoopClosing mnWebLastMerge, map stats [6]-[8], runner WEBMERGE lines) in the browser
# module (web/build) and the WASM runner. slam-wasm-ui builds both (snap-84c1f17 came from it). Nothing else running: -j6.
set -uo pipefail
LOG=/mnt/h/Research/logs/build_g40_1104.log
# 5 Nov: on 4 Nov 21:19 this copied the module into web/build and replaced the frozen paper module there (restored by
# 2026-11-05/build_paper_web_1105.sh); the copy now goes to web/build-g40, which the AR page does not load.
B=/root/build/slam-wasm-ui W=/mnt/h/Research/CustomSLAM/web/build-g40
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
{
  echo "=== start $(date +%T) HEAD $(git -C /mnt/h/Research/CustomSLAM rev-parse --short HEAD) (+ working tree)"
  nice -n 10 cmake --build $B -j6 --target orbslam3_run orbslam3_web_js 2>&1 | grep -E "error|FAILED|Linking"
  ls -la --time-style=full-iso $B/orbslam3.wasm $B/orbslam3_run.wasm
  cp $B/orbslam3.js $B/orbslam3.wasm $B/orbslam3.worker.mjs $W/
  md5sum $W/orbslam3.wasm $B/orbslam3_run.wasm
  echo "=== done $(date +%T)"
} >> $LOG 2>&1
