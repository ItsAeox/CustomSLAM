#!/usr/bin/env bash
# Queue 10 (after queue 9): focal-from-gyro check (native), WASM speed bench (snap with the new WASM build)
set -uo pipefail
E=/mnt/h/Research/CustomSLAM/scripts/experiments/2026-10-17; L=/mnt/h/Research/logs
until grep -q "=== queue9 done" $L/queue9.log 2>/dev/null; do sleep 60; done
bash $E/focal_check.sh 97b9685 > $L/focal_check.log 2>&1
[ -f /root/build/snap-97b9685w/wasm/orbslam3_run.js ] && bash $E/speed_bench.sh 97b9685w wasm > $L/speed_bench_wasm.log 2>&1
echo "=== queue10 done $(date +%T)"
