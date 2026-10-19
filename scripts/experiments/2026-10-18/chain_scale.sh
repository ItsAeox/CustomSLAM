#!/usr/bin/env bash
# PLAN 4d b2, one after the other (no overlap): datasets with the gated estimator, then the phone recordings (WASM,
# live@1.0) with WebVIO beta + online scale, then the per-recording scale stability (no ground truth there).
# Usage: chain_scale.sh <snapshot tag>
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
T=$1 R=/mnt/h/Research/results PH=/mnt/h/Research/datasets/phone
echo "=== datasets $(date +%T)"
bash scripts/experiments/2026-10-18/scale_online.sh /root/build/snap-$T/native/orbslam3_run $R/scale_online_1019b > $R/scale_online_1019b.txt 2>&1
echo "=== recordings $(date +%T)"
TRACE=1 bash scripts/experiments/2026-10-18/replay_arms.sh $T $R/replays_1019_scale scripts/experiments/2026-10-18/arms_scale.txt > $R/replays_1019_scale.log 2>&1
echo "=== stability $(date +%T)"
for r in $(awk '!/^#/ && NF {print $1}' $PH/labels.txt); do
  for s in 1 2; do
    d=$R/replays_1019_scale/$r/beta_scale/seed$s
    [ -s $d/trace.txt ] || continue
    echo "$(awk -v r=$r '$1 == r {print $2}' $PH/labels.txt) $(/opt/slam/venv/bin/python eval/scale_online_check.py $d $PH/$r 2>&1 | sed "s|$R/replays_1019_scale/||") | $(grep -h SCALE-STATS $d/run.log | cut -c1-140)"
  done
done > $R/replays_1019_scale_stability.txt
echo "=== done $(date +%T)"
