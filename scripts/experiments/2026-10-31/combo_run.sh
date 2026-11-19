#!/usr/bin/env bash
# 2 Nov (user: "go ahead"): the map-first phone-test candidate on the whole synthetic set, each lever alone and together
# against the same batch's baseline (route-by-route pairing; baselines vary between batches). WASM snap-<tag>, trace.
#   flat / flatmap / flatfirst (S11-S13, the object): arms_combo_flat.txt (tap 5.8 s) / arms_combo_flatfirst.txt
#     (tap 32.79 s): keep + g005t5, + cov2, + globall, + both; live@1.0 seeds 1-3, phone pace seeds 1-2
#   ar / longar (an object): arms_combo_ar.txt / arms_combo_longar.txt: baseline vs both
#   bedside rapid gesture orbit rotate shake slide walk (no object, no keep): arms_combo_rest.txt: baseline vs both
#     (regression check: tracked %, frame time, drops)
# Scoring is offline afterwards (anchor_return.py --place floor --refine, ret_snap.py, track_phase.py, synth_results.py).
# Usage: combo_run.sh <snapshot tag>   (finished runs are skipped: re-run to resume)
TAG=$1
cd /mnt/h/Research/CustomSLAM
while [ -e /tmp/webvio_build.lock ]; do sleep 15; done
D=/mnt/h/Research/datasets/synth X=scripts/experiments/2026-10-31
export PH=$D TRACE=1 PAR=4
recs() { (cd $D && ls -d */ | tr -d / | grep -E "^($1)[0-9]$" | tr '\n' ' '); }
for pace in live phone; do
  O=/mnt/h/Research/results/combo_1102
  if [ $pace = phone ]; then O=/mnt/h/Research/results/combo_1102_phone; export MODE="--live --rate-scale 0.25"; else unset MODE; fi
  mkdir -p $O
  S3="1 2 3"; [ $pace = phone ] && S3="1 2"
  SEEDS="$S3" RECS="$(recs 'flat|flatmap')" bash $X/../2026-10-18/replay_arms.sh $TAG $O $X/arms_combo_flat.txt >> $O/run.log 2>&1
  SEEDS="$S3" RECS="$(recs flatfirst)" bash $X/../2026-10-18/replay_arms.sh $TAG $O $X/arms_combo_flatfirst.txt >> $O/run.log 2>&1
  SEEDS="1 2" RECS="$(recs ar)" bash $X/../2026-10-18/replay_arms.sh $TAG $O $X/arms_combo_ar.txt >> $O/run.log 2>&1
  SEEDS="1 2" RECS="$(recs longar)" bash $X/../2026-10-18/replay_arms.sh $TAG $O $X/arms_combo_longar.txt >> $O/run.log 2>&1
  SEEDS="1 2" RECS="$(recs 'bedside|rapid|gesture|orbit|rotate|shake|slide|walk')" \
    bash $X/../2026-10-18/replay_arms.sh $TAG $O $X/arms_combo_rest.txt >> $O/run.log 2>&1
  echo "=== $pace runs done $(date +%T)"
done
