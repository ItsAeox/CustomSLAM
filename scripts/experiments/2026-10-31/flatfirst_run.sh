#!/usr/bin/env bash
# 2 Nov early (ISSUES G22, the user's own way, 1 Nov: "If I map the entire room before placing the object ... then I can
# just walk around everywhere"): the flat with the bedroom mapped BEFORE placing (S13 flatfirst0-5: the S12 loop right
# after the start-up, then place, look, the flat routes; rendered by synth/make_dataset.py --only flatfirst0,...).
# WASM snap-<tag> (orbslam3_run --dump-local), trace; the tap is at 32.79 s after the first frame (touches.csv), keep
# from there. live@1.0: arms_flatfirst.txt, seeds 1-4, 4 at a time; phone pace (rate-scale 0.25): seeds 1-2. Then per run
# eval/anchor_return.py four ways (centre / floor / floor + refine / best fixed) -> PLACE.md, ANCHOR.md, RETURN.md.
# Usage: flatfirst_run.sh <snapshot tag>   (finished runs are skipped: re-run to resume)
TAG=$1
cd /mnt/h/Research/CustomSLAM
while [ -e /tmp/webvio_build.lock ]; do sleep 15; done
D=/mnt/h/Research/datasets/synth X=scripts/experiments/2026-10-31 PY=/opt/slam/venv/bin/python
export PH=$D TRACE=1 PAR=4
RECS_ALL="flatfirst0 flatfirst1 flatfirst2 flatfirst3 flatfirst4 flatfirst5"
for pace in live phone; do
  O=/mnt/h/Research/results/flatfirst_1102; export SEEDS="1 2 3 4"
  if [ $pace = phone ]; then O=/mnt/h/Research/results/flatfirst_1102_phone; export SEEDS="1 2" MODE="--live --rate-scale 0.25"; else unset MODE; fi
  mkdir -p $O
  RECS="$RECS_ALL" bash scripts/experiments/2026-10-18/replay_arms.sh $TAG $O $X/arms_flatfirst.txt >> $O/run.log 2>&1
  ls -d $O/*/*/seed* | xargs -P 8 -I{} bash -c '
    r={}; q=$(basename $(dirname $(dirname $r))); g="--gt /mnt/h/Research/results/gt/synth_$q.txt"
    [ -s $r/anchor.txt ] || '$PY' eval/anchor_return.py $r '$D'/$q $g > $r/anchor.txt 2>&1
    [ -s $r/anchor_floor.txt ] || '$PY' eval/anchor_return.py $r '$D'/$q $g --place floor > $r/anchor_floor.txt 2>&1
    [ -s $r/anchor_refine.txt ] || '$PY' eval/anchor_return.py $r '$D'/$q $g --place floor --refine > $r/anchor_refine.txt 2>&1
    [ -s $r/anchor_oracle.txt ] || '$PY' eval/anchor_return.py $r '$D'/$q $g --anchor-oracle > $r/anchor_oracle.txt 2>&1'
  $PY $X/place_table.py $O > $O/PLACE.md 2>&1
  $PY eval/anchor_summary.py $O > $O/ANCHOR.md 2>&1
  { $PY $X/ret_table.py $O; echo; $PY $X/track_phase.py $O flatfirst; } > $O/RETURN.md 2>&1
  echo "=== $pace done $(date +%T)"
done
