#!/usr/bin/env bash
# Runners with honest real-time pacing (--rt-wall / camera --live; ISSUES H1). -j2 with nothing else running, else -j1 (WSL has 7.8 GB; see H2).
#  native: slam-native-teblid (newest source; TEBLID code present, default ORB) -- consistent rebuild of every object
#          that depends on a header edited since its last build;
#  wasm:   slam-wasm-ui orbslam3_run.js (libraries already current).
set -uo pipefail
LOG=/mnt/h/Research/logs/build_rt.log
echo "=== build_rt $(date +%T)" >> $LOG
nice -n 19 cmake --build /root/build/slam-native-teblid -j2 --target orbslam3_run 2>&1 | grep -E "error|FAILED" >> $LOG
ls -la --time-style=+%T /root/build/slam-native-teblid/orbslam3_run >> $LOG
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
nice -n 19 cmake --build /root/build/slam-wasm-ui -j2 --target orbslam3_run 2>&1 | grep -E "error|FAILED" >> $LOG
ls -la --time-style=+%T /root/build/slam-wasm-ui/orbslam3_run.js /root/build/slam-wasm-ui/orbslam3_run.wasm >> $LOG
echo "=== build_rt done $(date +%T)" >> $LOG
