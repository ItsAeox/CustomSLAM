#!/usr/bin/env bash
set -uo pipefail
E=/mnt/h/Research/CustomSLAM/scripts/experiments/2026-10-17; L=/mnt/h/Research/logs
until grep -q "=== queue13 done" $L/queue13.log 2>/dev/null; do sleep 60; done
bash $E/ginit_ab.sh > $L/ginit_ab.log 2>&1
echo "=== queue14 done $(date +%T)"
