#!/usr/bin/env bash
# Synthetic set, full run (claude/PLAN.md 4d, EXPERIMENTS synth_run): WASM snap-99436ab, live@1.0, seeds 1-2, trace.
#   A: the 49 sequences (phone-like IMU): WebVIO beta + online scale (offset -88 ms), stock mono-inertial (--imu-dt -88)
#   B: the 42 ideal-IMU variants (<seq>__imuideal): the same two arms, offset 0, IMU.Frequency 100
# then eval/synth_results.py: tracking, per-map Sim3 / SE3 against the ground truth, online scale against the ground
# truth, offline estimators (alignment, windows) with the phone-like and the ideal IMU on the same trajectories
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets/synth O=/mnt/h/Research/results/${OUT:-synth_run_1030} X=scripts/experiments/2026-10-30
export PH=$D TRACE=1 SEEDS="${SEEDS:-1 2}"
mkdir -p $O
seqs() { (cd $D && ls -d */ | tr -d / | grep -E "$1"); }
echo "=== batch A $(date +%T)"
RECS="${RECS_A:-$(seqs '^[a-z]+[0-9](_clean)?$')}" bash scripts/experiments/2026-10-18/replay_arms.sh 99436ab $O $X/arms_synth_a.txt > $O/batch_a.log 2>&1
echo "=== batch B $(date +%T)"
RECS="${RECS_B:-$(seqs '__imuideal$')}" bash scripts/experiments/2026-10-18/replay_arms.sh 99436ab $O $X/arms_synth_b.txt > $O/batch_b.log 2>&1
echo "=== scoring $(date +%T)"
/opt/slam/venv/bin/python eval/synth_results.py $O --data $D --gt /mnt/h/Research/results/gt --jobs ${JOBS:-8} > $O/results.log 2>&1
echo "=== all done $(date +%T)"
