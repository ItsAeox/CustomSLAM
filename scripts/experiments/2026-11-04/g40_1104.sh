#!/usr/bin/env bash
# 4 Nov evening (ISSUES G40 / G41, the user's OK "Yes, build them"; not for the paper): the leave-and-return test again for
# the two arms that start new maps after placing (stock, beta_newmap; arms_return.txt), on a build with the merge record
# (runner WEBMERGE lines: which map each merge absorbed). Same scenarios and paces as return_1104 (flat0-5, flatmap0-5
# tap 5.8 s; flatfirst0-5 tap 32.79 s; live@1.0 seeds 1-3, phone pace seeds 1-2, 4 at a time). Each run scored with the
# page's floor placement twice: as the page was (eval/anchor_return.py default + stash_drop_s for the G41 rule) and with
# the fixes (--page-fix: option stash=fixed), each with its per-frame dump. Then g40_table_1104.py.
# Usage: g40_1104.sh <snapshot tag>   (finished runs are skipped: re-run to resume)
cd /mnt/h/Research/CustomSLAM
TAG=$1
while [ -e /tmp/webvio_build.lock ]; do sleep 15; done
D=/mnt/h/Research/datasets/synth X=scripts/experiments/2026-11-04 PY=/opt/slam/venv/bin/python R=/mnt/h/Research
export PH=$D TRACE=1 PAR=4
(while sleep 60; do echo "$(date '+%F %T') $(cut -d' ' -f1 /proc/loadavg)"; done >> $R/logs/g40_1104_load.log) &
LOADPID=$!
score() {
  local O=$1
  ls -d $O/*/*/seed* | xargs -P 8 -I{} bash -c '
    r={}; q=$(basename $(dirname $(dirname $r))); g="--gt '$R'/results/gt/synth_$q.txt --place floor"
    [ -s $r/anchor_floor.txt ] || '$PY' eval/anchor_return.py $r '$D'/$q $g --dump $r/anchor_dump_place_floor.txt > $r/anchor_floor.txt 2>&1
    [ -s $r/anchor_floor_fix.txt ] || '$PY' eval/anchor_return.py $r '$D'/$q $g --page-fix --dump $r/anchor_dump_place_floor_fix.txt > $r/anchor_floor_fix.txt 2>&1'
  $PY eval/scale_collapse.py $O --data $D --gt $R/results/gt > $O/COLLAPSE.md 2>&1
}
for pace in live phone; do
  O=$R/results/return_g40_1104; export SEEDS="1 2 3"; unset MODE
  [ $pace = phone ] && { O=$R/results/return_g40_1104_phone; export SEEDS="1 2" MODE="--live --rate-scale 0.25"; }
  mkdir -p $O
  for st in "5.8:flat0 flat1 flat2 flat3 flat4 flat5 flatmap0 flatmap1 flatmap2 flatmap3 flatmap4 flatmap5" \
            "32.79:flatfirst0 flatfirst1 flatfirst2 flatfirst3 flatfirst4 flatfirst5"; do
    tap=${st%%:*}
    grep -E '^(stock|beta_newmap)\|' $X/arms_return.txt | sed "s/TAP/$tap/g" > $O/arms_$tap.txt
    echo "=== $pace tap $tap start $(date '+%F %T')" >> $O/run.log
    RECS="${st#*:}" bash scripts/experiments/2026-10-18/replay_arms.sh $TAG $O $O/arms_$tap.txt >> $O/run.log 2>&1
  done
  score $O
  echo "=== $pace done $(date '+%F %T')"
done
$PY $X/g40_table_1104.py --root $R > $R/results/return_g40_1104/G40.md 2>&1
kill $LOADPID
