#!/usr/bin/env bash
# 1 Nov (ISSUES G22 / G35): the flat return test with the logged placement (--tap-at) and the gyro attitude prior (S11 flat0-5, rendered by
# synth/make_dataset.py --only flat0,...,flat5): the same arms as return_run.sh (arms_flat2.txt, keep map from the
# tap at 5.8 s), WASM snap-0a5443f, seeds 1-2, trace; live@1.0 (4 at a time) and phone pace (rate-scale 0.25, 2 at a time);
# eval/anchor_return.py -> anchor_summary.py, eval/synth_results.py. Usage: flat_return_run.sh
cd /mnt/h/Research/CustomSLAM
while [ -e /tmp/webvio_build.lock ]; do sleep 15; done  # a build in progress would disturb live@1.0 timing
D=/mnt/h/Research/datasets/synth X=scripts/experiments/2026-10-31
export PH=$D TRACE=1 SEEDS="${SEEDS:-1 2}"
for pace in live phone; do
  O=/mnt/h/Research/results/flat2_1101; [ $pace = phone ] && O=/mnt/h/Research/results/flat2_1101_phone
  if [ $pace = phone ]; then export MODE="--live --rate-scale 0.25" PAR=2; else unset MODE; export PAR=4; fi
  mkdir -p $O
  RECS="$(cd $D && ls -d */ | tr -d / | grep -E '^flat[0-9]$')" bash scripts/experiments/2026-10-18/replay_arms.sh 0a5443f $O $X/arms_flat2.txt > $O/run.log 2>&1
  for r in $O/*/*/seed*; do
    q=$(basename $(dirname $(dirname $r)))
    [ -s $r/anchor.txt ] || /opt/slam/venv/bin/python eval/anchor_return.py $r $D/$q --gt /mnt/h/Research/results/gt/synth_$q.txt > $r/anchor.txt 2>&1
  done
  /opt/slam/venv/bin/python eval/anchor_summary.py $O > $O/ANCHOR.md 2>&1
  /opt/slam/venv/bin/python eval/synth_results.py $O --data $D --gt /mnt/h/Research/results/gt --jobs ${JOBS:-8} > $O/results.log 2>&1
  echo "=== $pace done $(date +%T)"
done
