#!/usr/bin/env bash
# The C++ scale filter (Web.Scale.Method 1) live in WebVIO beta on the whole synthetic set: the 49 sequences (phone-like IMU,
# -88 ms) and the 42 ideal-IMU variants; WASM live@1.0, seeds 1-2, trace; eval/synth_results.py (online columns = the filter)
# Usage: scaleF_run.sh <snapshot tag>
cd /mnt/h/Research/CustomSLAM
TAG=$1 D=/mnt/h/Research/datasets/synth O=/mnt/h/Research/results/${OUT:-scaleF_1030} X=scripts/experiments/2026-10-30
export PH=$D TRACE=1 SEEDS="${SEEDS:-1 2}"
mkdir -p $O
seqs() { (cd $D && ls -d */ | tr -d / | grep -E "$1"); }
RECS="${RECS_A:-$(seqs '^[a-z]+[0-9](_clean)?$')}" bash scripts/experiments/2026-10-18/replay_arms.sh $TAG $O $X/arms_scaleF_phone.txt > $O/phone.log 2>&1
RECS="${RECS_B:-$(seqs '__imuideal$')}" bash scripts/experiments/2026-10-18/replay_arms.sh $TAG $O $X/arms_scaleF_ideal.txt > $O/ideal.log 2>&1
/opt/slam/venv/bin/python eval/synth_results.py $O --data $D --gt /mnt/h/Research/results/gt --jobs ${JOBS:-8} > $O/results.log 2>&1
echo "=== all done $(date +%T)"
