#!/usr/bin/env bash
# 4 Nov (paper VII, the AR section; the user's OK 4 Nov): the leave-and-return AR test with the presets in the paper's scope
# (before the scale work: stock / fast / next / beta, arms_return.txt), frozen build snap-84c1f17, trace. Scenarios:
# S11 flat0-5 (place on a 5.8 s old map, leave through the flat, come back), S12 flatmap0-5 (place, walk around the
# bedroom, leave, come back), S13 flatfirst0-5 (map the bedroom first, place at 32.79 s, leave, come back by the other
# door). Two passes so the beta arms finish first (beta, beta_newmap; then next, fast, stock); live@1.0 seeds 1-3 and
# phone pace (rate-scale 0.25) seeds 1-2, 4 at a time. Then per run eval/anchor_return.py three ways: centre (the scorer's
# centre depth, as the earlier return batches), floor (the pre-cut page's floor-plane placement; no refine: that option
# came after the cut), best fixed anchor (map error without the placement) -> PLACE.md, ANCHOR.md, RETURN.md,
# COLLAPSE.md, results.log (synth_results.py). 1-min load average every 60 s: logs/return_1104_load.log.
# Usage: return_1104.sh   (finished runs are skipped: re-run to resume)
cd /mnt/h/Research/CustomSLAM
while [ -e /tmp/webvio_build.lock ]; do sleep 15; done  # a build in progress would disturb live@1.0 timing
D=/mnt/h/Research/datasets/synth X=scripts/experiments/2026-11-04 X0=scripts/experiments/2026-10-31 PY=/opt/slam/venv/bin/python
R=/mnt/h/Research
export PH=$D TRACE=1 PAR=4
(while sleep 60; do echo "$(date '+%F %T') $(cut -d' ' -f1 /proc/loadavg)"; done >> $R/logs/return_1104_load.log) &
LOADPID=$!
score() {  # score() <batch dir>: every run three ways (skips files already there), then the tables
  local O=$1
  ls -d $O/*/*/seed* | xargs -P 8 -I{} bash -c '
    r={}; q=$(basename $(dirname $(dirname $r))); g="--gt '$R'/results/gt/synth_$q.txt"
    [ -s $r/anchor.txt ] || '$PY' eval/anchor_return.py $r '$D'/$q $g > $r/anchor.txt 2>&1
    [ -s $r/anchor_floor.txt ] || '$PY' eval/anchor_return.py $r '$D'/$q $g --place floor > $r/anchor_floor.txt 2>&1
    [ -s $r/anchor_oracle.txt ] || '$PY' eval/anchor_return.py $r '$D'/$q $g --anchor-oracle > $r/anchor_oracle.txt 2>&1'
  $PY $X0/place_table.py $O > $O/PLACE.md 2>&1
  $PY eval/anchor_summary.py $O > $O/ANCHOR.md 2>&1
  $PY $X0/ret_table.py $O > $O/RETURN.md 2>&1
  $PY eval/scale_collapse.py $O --data $D --gt $R/results/gt > $O/COLLAPSE.md 2>&1
  $PY eval/synth_results.py $O --data $D --gt $R/results/gt --jobs ${JOBS:-8} > $O/results.log 2>&1
}
for pass in beta rest; do
  [ $pass = beta ] && ARMRE='^(beta|beta_newmap)\|' || ARMRE='^(next|fast|stock)\|'
  for pace in live phone; do
    O=$R/results/return_1104; export SEEDS="1 2 3"; unset MODE
    [ $pace = phone ] && { O=$R/results/return_1104_phone; export SEEDS="1 2" MODE="--live --rate-scale 0.25"; }
    mkdir -p $O
    for st in "5.8:flat0 flat1 flat2 flat3 flat4 flat5 flatmap0 flatmap1 flatmap2 flatmap3 flatmap4 flatmap5" \
              "32.79:flatfirst0 flatfirst1 flatfirst2 flatfirst3 flatfirst4 flatfirst5"; do
      tap=${st%%:*}
      grep -E "$ARMRE" $X/arms_return.txt | sed "s/TAP/$tap/g" > $O/arms_${pass}_$tap.txt
      echo "=== $pass $pace tap $tap start $(date '+%F %T')" >> $O/run.log
      RECS="${st#*:}" bash scripts/experiments/2026-10-18/replay_arms.sh 84c1f17 $O $O/arms_${pass}_$tap.txt >> $O/run.log 2>&1
    done
    score $O
    echo "=== $pass $pace done $(date '+%F %T')"
  done
done
kill $LOADPID
