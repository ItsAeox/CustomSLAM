#!/usr/bin/env bash
# Queue 13 (after the pause for the user, 2026-10-17 evening): finish screen B, then what queues 9/11/12 had left:
# focal check, gyro init with phone gyro, speed benches (snap-a75dc17 native + WASM), the page presets in WASM.
set -uo pipefail
E=/mnt/h/Research/CustomSLAM/scripts/experiments/2026-10-17; L=/mnt/h/Research/logs
bash $E/resume_screen.sh 97b9685 screen_b preset-fast preset-fast-gyro fast-gyro-coast2 fast-gyro-ginit fast-kf080 fast-nofb >> $L/screen_b.log 2>&1
/opt/slam/venv/bin/python /mnt/h/Research/CustomSLAM/eval/screen_table.py /mnt/h/Research/results/screen_b >> $L/screen_b.log 2>&1
bash $E/focal_check.sh a75dc17 > $L/focal_check.log 2>&1
bash $E/ginit_phone.sh a75dc17 > $L/ginit_phone.log 2>&1
bash $E/speed_bench.sh a75dc17 native > $L/speed_bench_native_a75.log 2>&1
bash $E/speed_bench.sh a75dc17 wasm > $L/speed_bench_wasm.log 2>&1
bash $E/page_wasm.sh a75dc17 page_wasm_a fast fastq fastqp small lean > $L/page_wasm_a.log 2>&1
bash $E/page_wasm.sh a75dc17 page_wasm_b fast gstart grecover expo next > $L/page_wasm_b.log 2>&1
echo "=== queue13 done $(date +%T)"
