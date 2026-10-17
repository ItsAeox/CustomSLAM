#!/usr/bin/env bash
# Queue 9 (after queue 8): round-2 screens on snap-97b9685 (native), then the clean speed bench.
#   screen A: preset-fast | fast-speed | fast-coast2 | fast-norm | fast-init
#   screen B: preset-fast | preset-fast-gyro | fast-gyro-coast2 | fast-gyro-ginit | fast-kf080 | fast-nofb
set -uo pipefail
E=/mnt/h/Research/CustomSLAM/scripts/experiments/2026-10-17; L=/mnt/h/Research/logs
until grep -q "=== queue8 done" $L/queue8.log 2>/dev/null; do sleep 60; done
bash $E/screen_a.sh 97b9685 screen_a preset-fast fast-speed fast-coast2 fast-norm fast-init > $L/screen_a.log 2>&1
bash $E/screen_a.sh 97b9685 screen_b preset-fast preset-fast-gyro fast-gyro-coast2 fast-gyro-ginit fast-kf080 fast-nofb > $L/screen_b.log 2>&1
bash $E/speed_bench.sh 97b9685 native > $L/speed_bench_native.log 2>&1
echo "=== queue9 done $(date +%T)"
