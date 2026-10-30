#!/usr/bin/env bash
# 31 Oct: the synthetic set with the three longer scenarios (bedside / rapid / longar, synth/traj.py S8-S10; phone-like
# IMU) and the original 49 (phone-like IMU): WebVIO beta + scale filter vs + KLT 21 px / 3 levels (arms_klt.txt), WASM
# snap-858bac8, live@1.0, seeds 1-2, trace; eval/synth_results.py. Usage: synth2_run.sh
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets/synth O=/mnt/h/Research/results/${OUT:-synth2_1031} X=scripts/experiments/2026-10-31
export PH=$D TRACE=1 SEEDS="${SEEDS:-1 2}"
mkdir -p $O
RECS="${RECS:-$(cd $D && ls -d */ | tr -d / | grep -E '^(bedside|rapid|longar)[0-9]$'; cd $D && ls -d */ | tr -d / | grep -E '^(ar|gesture|orbit|rotate|shake|slide|walk)[0-9]$')}" \
  bash scripts/experiments/2026-10-18/replay_arms.sh 858bac8 $O $X/arms_klt.txt > $O/run.log 2>&1
/opt/slam/venv/bin/python eval/synth_results.py $O --data $D --gt /mnt/h/Research/results/gt --jobs ${JOBS:-8} > $O/results.log 2>&1
echo "=== all done $(date +%T)"
