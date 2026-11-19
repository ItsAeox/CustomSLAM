#!/usr/bin/env bash
# 2 Nov (ISSUES G22): the whole-map view diagnostic (trace columns of the runner after df7b5b2) on the flat and
# flatmap, baseline arm only. live@1.0 seeds 1-3, phone pace seeds 1-2. Then local_match.py on Windows.
# Usage: wholemap_run.sh <snapshot tag>   (finished runs are skipped: re-run to resume)
TAG=$1
cd /mnt/h/Research/CustomSLAM
while [ -e /tmp/webvio_build.lock ]; do sleep 15; done
D=/mnt/h/Research/datasets/synth X=scripts/experiments/2026-10-31 PY=/opt/slam/venv/bin/python
export PH=$D TRACE=1 PAR=4
RECS_ALL="flat0 flat1 flat2 flat3 flat4 flat5 flatmap0 flatmap1 flatmap2 flatmap3 flatmap4 flatmap5"
for pace in live phone; do
  O=/mnt/h/Research/results/wholemap_1102; A=$X/arms_wholemap.txt; export SEEDS="1 2 3"
  if [ $pace = phone ]; then O=/mnt/h/Research/results/wholemap_1102_phone; A=$X/arms_wholemap.txt; export SEEDS="1 2" MODE="--live --rate-scale 0.25"; else unset MODE; fi
  mkdir -p $O
  RECS="$RECS_ALL" bash scripts/experiments/2026-10-18/replay_arms.sh $TAG $O $A >> $O/run.log 2>&1
  ls -d $O/*/*/seed* | xargs -P 8 -I{} bash -c '
    r={}; q=$(basename $(dirname $(dirname $r))); g="--gt /mnt/h/Research/results/gt/synth_$q.txt"
    [ -s $r/anchor.txt ] || '$PY' eval/anchor_return.py $r '$D'/$q $g > $r/anchor.txt 2>&1'
  $PY eval/anchor_summary.py $O > $O/ANCHOR.md 2>&1
  { $PY $X/ret_table.py $O; echo; $PY $X/track_phase.py $O flat; echo; $PY $X/track_phase.py $O flatmap; } > $O/RETURN.md 2>&1
  echo "=== $pace done $(date +%T)"
done
