#!/usr/bin/env bash
# 1 Nov night (ISSUES G22, the user's goal: recognise the place and show the object again when coming back by another
# door / angle): the flat (S11 flat0-5) and the flat with the bedroom mapped first after placing (S12 flatmap0-5, rendered
# by synth/make_dataset.py --only flatmap0,...,flatmap5), WASM snap-9cba366 (Web.Gyro.MinInliers, Web.Reloc.Global), trace.
# live@1.0: arms_mapfirst.txt, seeds 1-3, 4 at a time; phone pace (rate-scale 0.25): arms_mapfirst_phone.txt, seeds 1-2,
# 4 at a time (a phone-pace run uses ~1/4 of the CPU of a live one; flat2 / flatkeep used 2 at a time).
# Then eval/anchor_return.py -> anchor_summary.py, scale_collapse.py, eval/synth_results.py.
# Usage: mapfirst_run.sh   (finished runs are skipped: re-run to resume)
cd /mnt/h/Research/CustomSLAM
while [ -e /tmp/webvio_build.lock ]; do sleep 15; done  # a build in progress would disturb live@1.0 timing
D=/mnt/h/Research/datasets/synth X=scripts/experiments/2026-10-31
export PH=$D TRACE=1 PAR=4
RECS_ALL="flat0 flat1 flat2 flat3 flat4 flat5 flatmap0 flatmap1 flatmap2 flatmap3 flatmap4 flatmap5"
for pace in live phone; do
  O=/mnt/h/Research/results/mapfirst_1101; A=$X/arms_mapfirst.txt; export SEEDS="1 2 3"
  if [ $pace = phone ]; then O=/mnt/h/Research/results/mapfirst_1101_phone; A=$X/arms_mapfirst_phone.txt; export SEEDS="1 2" MODE="--live --rate-scale 0.25"; else unset MODE; fi
  mkdir -p $O
  RECS="$RECS_ALL" bash scripts/experiments/2026-10-18/replay_arms.sh 9cba366 $O $A >> $O/run.log 2>&1
  for r in $O/*/*/seed*; do
    q=$(basename $(dirname $(dirname $r)))
    [ -s $r/anchor.txt ] || /opt/slam/venv/bin/python eval/anchor_return.py $r $D/$q --gt /mnt/h/Research/results/gt/synth_$q.txt > $r/anchor.txt 2>&1
  done
  /opt/slam/venv/bin/python eval/anchor_summary.py $O > $O/ANCHOR.md 2>&1
  /opt/slam/venv/bin/python eval/scale_collapse.py $O --data $D --gt /mnt/h/Research/results/gt > $O/COLLAPSE.md 2>&1
  /opt/slam/venv/bin/python eval/synth_results.py $O --data $D --gt /mnt/h/Research/results/gt --jobs ${JOBS:-8} > $O/results.log 2>&1
  echo "=== $pace done $(date +%T)"
done
