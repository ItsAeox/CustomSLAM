#!/usr/bin/env bash
# Wait for the M3 variant sweep to finish, then run the hybrid-vs-stock sweep (needs the CPU for itself).
until grep -q "=== done" /mnt/h/Research/logs/m3_variants.log 2>/dev/null; do sleep 60; done
echo "m3 finished $(date +%T); starting KLT sweep"
bash /mnt/c/Users/Aeox/AppData/Local/Temp/claude/H--Research/6624e451-5f38-4671-9a62-226588a85597/scratchpad/run_klt_sweep.sh > /mnt/h/Research/logs/klt_sweep.log 2>&1
echo "klt sweep finished $(date +%T)"
