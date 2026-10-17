#!/usr/bin/env bash
# Queue 15 (after the pool-exit fix): speed benches (native + WASM), gyro init A/B, the page presets in WASM; snap-$1
set -uo pipefail
E=/mnt/h/Research/CustomSLAM/scripts/experiments/2026-10-17; L=/mnt/h/Research/logs
rm -rf /mnt/h/Research/results/speed_bench
bash $E/speed_bench.sh $1 native > $L/speed_bench_native.log 2>&1
bash $E/speed_bench.sh $1 wasm > $L/speed_bench_wasm.log 2>&1
bash $E/ginit_ab.sh $1 > $L/ginit_ab.log 2>&1
bash $E/page_wasm.sh $1 page_wasm_a fast fastq fastqp small lean > $L/page_wasm_a.log 2>&1
bash $E/page_wasm.sh $1 page_wasm_b fast gstart grecover expo next > $L/page_wasm_b.log 2>&1
echo "=== queue15 done $(date +%T)"
