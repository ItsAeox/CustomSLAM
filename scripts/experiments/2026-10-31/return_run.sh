#!/usr/bin/env bash
# 31 Oct (ISSUES G22, the user's goal): place the object, leave, come back - is it found and where does it appear?
# Synthetic ar / longar v0-v5 (phone IMU): stock (new map after the relocalisation timeout, merge restores) vs keep map
# (the page default once placed) / + turn bridge (tv, gyro) / + map-growth option cov2 (arms_return_<scenario>.txt),
# WASM snap-aa3763f (trace with map id), seeds 1-2, trace; live@1.0 (4 at a time) and phone pace (rate-scale 0.25 = a 4x
# slower device, 2 at a time); eval/anchor_return.py per run -> anchor_summary.py, eval/synth_results.py.
# Usage: return_run.sh
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets/synth X=scripts/experiments/2026-10-31
export PH=$D TRACE=1 SEEDS="${SEEDS:-1 2}"
for pace in live phone; do
  O=/mnt/h/Research/results/return_1031; [ $pace = phone ] && O=/mnt/h/Research/results/return_1031_phone
  if [ $pace = phone ]; then export MODE="--live --rate-scale 0.25" PAR=2; else unset MODE; export PAR=4; fi
  mkdir -p $O
  for sc in ar longar; do
    RECS="$(cd $D && ls -d */ | tr -d / | grep -E "^${sc}[0-9]$")" bash scripts/experiments/2026-10-18/replay_arms.sh aa3763f $O $X/arms_return_$sc.txt > $O/run_$sc.log 2>&1
  done
  for r in $O/*/*/seed*; do
    q=$(basename $(dirname $(dirname $r)))
    [ -s $r/anchor.txt ] || /opt/slam/venv/bin/python eval/anchor_return.py $r $D/$q --gt /mnt/h/Research/results/gt/synth_$q.txt > $r/anchor.txt 2>&1
  done
  /opt/slam/venv/bin/python eval/anchor_summary.py $O > $O/ANCHOR.md 2>&1
  /opt/slam/venv/bin/python eval/synth_results.py $O --data $D --gt /mnt/h/Research/results/gt --jobs ${JOBS:-8} > $O/results.log 2>&1
  echo "=== $pace done $(date +%T)"
done
