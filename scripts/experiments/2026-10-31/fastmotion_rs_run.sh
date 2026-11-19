#!/usr/bin/env bash
# 31 Oct (ISSUES G31): re-run of the rolling-shutter arms after the grid fix: the 60 synthetic sequences (S1-S10 v0-v5,
# phone-like IMU), beta + scale filter (control) vs rolling-shutter correction +20 / -20 ms (sign check) / + gyro warp
# (arms_rs.txt), WASM snap-735b90a, live@1.0, seeds 1-2, trace; eval/synth_results.py. Usage: fastmotion_rs_run.sh
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets/synth O=/mnt/h/Research/results/${OUT:-fastmotion_rs_1031} X=scripts/experiments/2026-10-31
export PH=$D TRACE=1 SEEDS="${SEEDS:-1 2}"
mkdir -p $O
RECS="${RECS:-$(cd $D && ls -d */ | tr -d / | grep -E '^(bedside|rapid|longar)[0-9]$'; cd $D && ls -d */ | tr -d / | grep -E '^(ar|gesture|orbit|rotate|shake|slide|walk)[0-9]$')}" \
  bash scripts/experiments/2026-10-18/replay_arms.sh 735b90a $O $X/arms_rs.txt > $O/run.log 2>&1
/opt/slam/venv/bin/python eval/synth_results.py $O --data $D --gt /mnt/h/Research/results/gt --jobs ${JOBS:-8} > $O/results.log 2>&1
echo "=== all done $(date +%T)"
