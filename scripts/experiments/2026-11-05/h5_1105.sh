#!/usr/bin/env bash
# 5 Nov (ISSUES H5, the user's OK "Yes, rerun them"): stock and beta_newmap of return_1104 again on the same frozen
# build (snap-84c1f17), to separate the build from run-to-run spread. g40_1104 (snap-a30b538) gave different counts
# for the same arms, seeds and settings (on screen again live: stock flatmap 6 -> 14, beta_newmap flatfirst 14 -> 6).
# Same scenarios, paces, seeds and parallelism as return_1104 / g40_1104 (flat0-5, flatmap0-5 tap 5.8 s; flatfirst0-5
# tap 32.79 s; live@1.0 seeds 1-3, phone pace seeds 1-2, 4 at a time). Scored with the page's floor placement as
# return_1104 (eval/anchor_return.py default, per-frame dump); the frozen build has no merge record, so no --page-fix.
# Then h5_table_1105.py: return_1104 vs this batch (same build) vs g40_1104 (later build).
# Usage: h5_1105.sh   (finished runs are skipped: re-run to resume)
cd /mnt/h/Research/CustomSLAM
while [ -e /tmp/webvio_build.lock ]; do sleep 15; done
D=/mnt/h/Research/datasets/synth X=scripts/experiments/2026-11-05 X4=scripts/experiments/2026-11-04 PY=/opt/slam/venv/bin/python
R=/mnt/h/Research
export PH=$D TRACE=1 PAR=4
(while sleep 60; do echo "$(date '+%F %T') $(cut -d' ' -f1 /proc/loadavg)"; done >> $R/logs/h5_1105_load.log) &
LOADPID=$!
score() {
  local O=$1
  ls -d $O/*/*/seed* | xargs -P 8 -I{} bash -c '
    r={}; q=$(basename $(dirname $(dirname $r))); g="--gt '$R'/results/gt/synth_$q.txt --place floor"
    [ -s $r/anchor_floor.txt ] || '$PY' eval/anchor_return.py $r '$D'/$q $g --dump $r/anchor_dump_place_floor.txt > $r/anchor_floor.txt 2>&1'
  $PY eval/scale_collapse.py $O --data $D --gt $R/results/gt > $O/COLLAPSE.md 2>&1
}
for pace in live phone; do
  O=$R/results/return_h5_1105; export SEEDS="1 2 3"; unset MODE
  [ $pace = phone ] && { O=$R/results/return_h5_1105_phone; export SEEDS="1 2" MODE="--live --rate-scale 0.25"; }
  mkdir -p $O
  for st in "5.8:flat0 flat1 flat2 flat3 flat4 flat5 flatmap0 flatmap1 flatmap2 flatmap3 flatmap4 flatmap5" \
            "32.79:flatfirst0 flatfirst1 flatfirst2 flatfirst3 flatfirst4 flatfirst5"; do
    tap=${st%%:*}
    grep -E '^(stock|beta_newmap)\|' $X4/arms_return.txt | sed "s/TAP/$tap/g" > $O/arms_$tap.txt
    echo "=== $pace tap $tap start $(date '+%F %T')" >> $O/run.log
    RECS="${st#*:}" bash scripts/experiments/2026-10-18/replay_arms.sh 84c1f17 $O $O/arms_$tap.txt >> $O/run.log 2>&1
  done
  score $O
  echo "=== $pace done $(date '+%F %T')"
done
$PY $X/h5_table_1105.py --root $R > $R/results/return_h5_1105/H5.md 2>&1
kill $LOADPID
