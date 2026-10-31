#!/usr/bin/env bash
# 1 Nov (ISSUES G35): the gyro rotation in the pose solve and the ablation (arms_gyropose.txt) on the synthetic orbit / ar /
# longar / walk v0-v5 (phone IMU), WASM snap-7957aac, live@1.0 (4 at a time), seeds 1-2, trace; then eval/scale_collapse.py
# (COLLAPSE.md), eval/anchor_return.py (ar / longar; keep map is not on here: stock anchor logic), synth_results.py.
# Usage: gyropose_run.sh
cd /mnt/h/Research/CustomSLAM
while [ -e /tmp/webvio_build.lock ]; do sleep 15; done
D=/mnt/h/Research/datasets/synth O=/mnt/h/Research/results/${OUT:-gyropose_1101} X=scripts/experiments/2026-10-31
export PH=$D TRACE=1 SEEDS="${SEEDS:-1 2}"
mkdir -p $O
RECS="${RECS:-$(cd $D && ls -d */ | tr -d / | grep -E '^(orbit|ar|longar|walk)[0-9]$')}" \
  bash scripts/experiments/2026-10-18/replay_arms.sh 7957aac $O $X/arms_gyropose.txt > $O/run.log 2>&1
for r in $O/ar*/*/seed* $O/longar*/*/seed*; do
  q=$(basename $(dirname $(dirname $r)))
  [ -s $r/anchor.txt ] || /opt/slam/venv/bin/python eval/anchor_return.py $r $D/$q --gt /mnt/h/Research/results/gt/synth_$q.txt > $r/anchor.txt 2>&1
done
/opt/slam/venv/bin/python eval/anchor_summary.py $O > $O/ANCHOR.md 2>&1
/opt/slam/venv/bin/python eval/scale_collapse.py $O --data $D --gt /mnt/h/Research/results/gt > $O/COLLAPSE.md 2>&1
/opt/slam/venv/bin/python eval/synth_results.py $O --data $D --gt /mnt/h/Research/results/gt --jobs ${JOBS:-8} > $O/results.log 2>&1
echo "=== all done $(date +%T)"
