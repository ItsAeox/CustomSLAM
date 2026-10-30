#!/usr/bin/env bash
# Screen of the tolerant IMU start-up options (ISSUES G28, arms_imuA_*.txt) on one synthetic sequence per scenario
# (variant 2), phone-like IMU and ideal-IMU variant, WASM live@1.0, seeds 1-2, trace; then eval/synth_results.py
# Usage: imuA_screen.sh <snapshot tag>
cd /mnt/h/Research/CustomSLAM
TAG=$1 D=/mnt/h/Research/datasets/synth O=/mnt/h/Research/results/${OUT:-imuA_screen_1030} X=scripts/experiments/2026-10-30
export PH=$D TRACE=1 SEEDS="${SEEDS:-1 2}"
mkdir -p $O
P="${RECS_P:-slide3 gesture2 shake2 rotate2 orbit2 walk2 ar2}"
RECS="$P" bash scripts/experiments/2026-10-18/replay_arms.sh $TAG $O $X/arms_imuA_phone.txt > $O/phone.log 2>&1
RECS="$(for r in $P; do echo ${r}__imuideal; done)" bash scripts/experiments/2026-10-18/replay_arms.sh $TAG $O $X/arms_imuA_ideal.txt > $O/ideal.log 2>&1
/opt/slam/venv/bin/python eval/synth_results.py $O --data $D --gt /mnt/h/Research/results/gt --jobs ${JOBS:-8} > $O/results.log 2>&1
echo "=== all done $(date +%T)"
