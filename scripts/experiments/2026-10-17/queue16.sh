#!/usr/bin/env bash
# Queue 16 (after queue 15): the "maximum speed" preset in WASM against "Hybrid, fastest" (page settings)
set -uo pipefail
E=/mnt/h/Research/CustomSLAM/scripts/experiments/2026-10-17; L=/mnt/h/Research/logs
until grep -q "=== queue15 done" $L/queue15.log 2>/dev/null; do sleep 60; done
bash $E/page_wasm.sh $1 page_wasm_c fast fastmax > $L/page_wasm_c.log 2>&1
echo "=== queue16 done $(date +%T)"
