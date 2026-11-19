#!/usr/bin/env bash
# Re-score the floor / floor+refine placements of a batch after the anchor_return.py fix (the floor code shadowed the
# trace's tracking state: every floor score crashed). One run at a time, nice 19 (a timed pass may be running).
# Usage: rescore_floor.sh <batch dir>
O=$1; D=/mnt/h/Research/datasets/synth PY=/opt/slam/venv/bin/python
cd /mnt/h/Research/CustomSLAM
for r in $(ls -d $O/*/*/seed*); do
  q=$(basename $(dirname $(dirname $r))); g="--gt /mnt/h/Research/results/gt/synth_$q.txt"
  grep -q Traceback $r/anchor_floor.txt 2>/dev/null && nice -n 19 $PY eval/anchor_return.py $r $D/$q $g --place floor > $r/anchor_floor.txt 2>&1
  grep -q Traceback $r/anchor_refine.txt 2>/dev/null && nice -n 19 $PY eval/anchor_return.py $r $D/$q $g --place floor --refine > $r/anchor_refine.txt 2>&1
done
nice -n 19 $PY scripts/experiments/2026-10-31/place_table.py $O > $O/PLACE.md 2>&1
echo "=== rescored $O $(date +%T)"
