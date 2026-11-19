#!/usr/bin/env bash
# 2 Nov (PLAN 0: map start at 9-12 fps for ~3 s on the phone): frames before the first map extract 5x nFeatures (~4,800;
# 53 ms per frame vs 21 ms for an ORB frame in WASM on the desktop, combo_1102). Web.Init.Features 3 / 2 vs stock 5 on
# the whole synthetic set at phone pace (rate-scale 0.25), seeds 1-3, trace: arms_initf_obj.txt (ar / flat / flatmap,
# object at 5.8 s), arms_initf_first.txt (flatfirst, 32.79 s), arms_initf_rest.txt (the others, no keep). Then
# startup_table.py (time to the first map, frame time and frame rate while it starts, map points at the start) and
# eval/synth_results.py; the object (anchor_return.py --place floor --refine) offline.
# Usage: initf_run.sh <snapshot tag (with Web.Init.Features)>   (finished runs are skipped: re-run to resume)
TAG=$1
cd /mnt/h/Research/CustomSLAM
while [ -e /tmp/webvio_build.lock ]; do sleep 15; done
D=/mnt/h/Research/datasets/synth X=scripts/experiments/2026-10-31 O=/mnt/h/Research/results/initf_1102_phone
export PH=$D TRACE=1 PAR=4 SEEDS="1 2 3" MODE="--live --rate-scale 0.25"
recs() { (cd $D && ls -d */ | tr -d / | grep -E "^($1)[0-9]$" | tr '\n' ' '); }
mkdir -p $O
RECS="$(recs 'ar|flat|flatmap')" bash $X/../2026-10-18/replay_arms.sh $TAG $O $X/arms_initf_obj.txt >> $O/run.log 2>&1
RECS="$(recs flatfirst)" bash $X/../2026-10-18/replay_arms.sh $TAG $O $X/arms_initf_first.txt >> $O/run.log 2>&1
RECS="$(recs 'longar|bedside|rapid|gesture|orbit|rotate|shake|slide|walk')" bash $X/../2026-10-18/replay_arms.sh $TAG $O $X/arms_initf_rest.txt >> $O/run.log 2>&1
/opt/slam/venv/bin/python $X/startup_table.py $O > $O/STARTUP.md 2>&1
/opt/slam/venv/bin/python eval/synth_results.py $O --data $D --gt /mnt/h/Research/results/gt --jobs 8 > $O/results.log 2>&1
echo "=== initf done $(date +%T)" >> $O/run.log
