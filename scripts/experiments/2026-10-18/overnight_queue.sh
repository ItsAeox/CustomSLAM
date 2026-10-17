#!/usr/bin/env bash
# Overnight queue 2026-10-18, one job at a time (7.8 GB RAM; timing benches need an idle machine). Each script skips
# runs that already finished, so the queue can be restarted after an interruption.
#   1. WASM bench: fast-speed and fast-speed2 again, alone (their first runs shared the CPU with scale_check)
#   2. map_reuse   room map save -> reload (native, rt-wall)
#   3. vocab_hbrb  HBRB + 2x-data vocabularies: training (alone), then the vocab_eval2 protocol
#   4. rot_screen  pure-rotation prototype vs preset-fast (live@0.5)
#   5. imu_calib   calibrate-then-run mono-inertial (rt-wall)
# Usage: overnight_queue.sh <snapshot tag>
set -uo pipefail
E=/mnt/h/Research/CustomSLAM/scripts/experiments/2026-10-18; L=/mnt/h/Research/logs
echo "=== queue start $(date +%T) snap-$1"
ARMS="fast-speed fast-speed2" bash $E/speed_bench_klt.sh f6aa2cf wasm > $L/speed_bench_klt_wasm_rerun.log 2>&1; echo "  bench rerun done $(date +%T)"
bash $E/map_reuse.sh $1 > $L/map_reuse.log 2>&1; echo "  map_reuse done $(date +%T)"
bash $E/vocab_hbrb.sh > $L/vocab_hbrb.log 2>&1; echo "  vocab_hbrb done $(date +%T)"
bash $E/rot_screen.sh $1 rot_screen preset-fast fast-rot > $L/rot_screen.log 2>&1; echo "  rot_screen done $(date +%T)"
bash $E/imu_calib.sh $1 > $L/imu_calib.log 2>&1; echo "  imu_calib done $(date +%T)"
echo "=== queue done $(date +%T)"
