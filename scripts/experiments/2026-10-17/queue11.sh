#!/usr/bin/env bash
# Queue 11 (after queue 9; replaces queue 10): on snap-a75dc17 (own ORB thread pool, SSC, refined gyro init)
#   focal-from-gyro check, gyro init with phone-like gyro, speed bench native + WASM
set -uo pipefail
E=/mnt/h/Research/CustomSLAM/scripts/experiments/2026-10-17; L=/mnt/h/Research/logs
until grep -q "=== queue9 done" $L/queue9.log 2>/dev/null; do sleep 60; done
bash $E/focal_check.sh a75dc17 > $L/focal_check.log 2>&1
bash $E/ginit_phone.sh a75dc17 > $L/ginit_phone.log 2>&1
mv /mnt/h/Research/results/speed_bench /mnt/h/Research/results/speed_bench_97b9685 2>/dev/null
bash $E/speed_bench.sh a75dc17 native > $L/speed_bench_native_a75.log 2>&1
bash $E/speed_bench.sh a75dc17 wasm > $L/speed_bench_wasm.log 2>&1
echo "=== queue11 done $(date +%T)"
