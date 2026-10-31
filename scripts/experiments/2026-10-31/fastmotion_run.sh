#!/usr/bin/env bash
# 31 Oct (ISSUES G31): the 60 synthetic sequences (S1-S10 v0-v5, phone-like IMU): WebVIO beta + scale filter vs + gyro-
# warped KLT / rolling-shutter correction / both (arms_fastmotion.txt), WASM snap-923f0dc, live@1.0, seeds 1-2, trace;
# eval/synth_results.py. Usage: fastmotion_run.sh
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets/synth O=/mnt/h/Research/results/${OUT:-fastmotion_1031} X=scripts/experiments/2026-10-31
export PH=$D TRACE=1 SEEDS="${SEEDS:-1 2}"
mkdir -p $O
RECS="${RECS:-$(cd $D && ls -d */ | tr -d / | grep -E '^(bedside|rapid|longar)[0-9]$'; cd $D && ls -d */ | tr -d / | grep -E '^(ar|gesture|orbit|rotate|shake|slide|walk)[0-9]$')}" \
  bash scripts/experiments/2026-10-18/replay_arms.sh 923f0dc $O $X/arms_fastmotion.txt > $O/run.log 2>&1
/opt/slam/venv/bin/python eval/synth_results.py $O --data $D --gt /mnt/h/Research/results/gt --jobs ${JOBS:-8} > $O/results.log 2>&1
echo "=== all done $(date +%T)"
