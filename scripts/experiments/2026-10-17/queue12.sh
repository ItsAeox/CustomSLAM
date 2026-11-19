#!/usr/bin/env bash
# Queue 12 (after queue 11): the AR page's round-2 presets as built by the page, WASM, two batches of 5 arms
set -uo pipefail
E=/mnt/h/Research/CustomSLAM/scripts/experiments/2026-10-17; L=/mnt/h/Research/logs
until grep -q "=== queue11 done" $L/queue11.log 2>/dev/null; do sleep 60; done
bash $E/page_wasm.sh $1 page_wasm_a fast fastq fastqp small lean > $L/page_wasm_a.log 2>&1
bash $E/page_wasm.sh $1 page_wasm_b fast gstart grecover expo next > $L/page_wasm_b.log 2>&1
echo "=== queue12 done $(date +%T)"
