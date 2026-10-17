#!/usr/bin/env bash
# Replay one phone recording (record.html -> datasets/phone/<name>) through AR presets, side by side: WASM runner
# (Node), phone emulation (--live --rate-scale 0.5), seeds, then eval/recording_metrics.py. Settings per preset come
# from scripts/page_presets.mjs --base <recording settings.yaml> (run on Windows first; see replay_recording.ps1).
# Usage: replay_recording.sh <snapshot tag> <recording> "<seeds>" <preset...>
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
TAG=$1 REC=$2 SEEDS=$3; shift 3
SEQ=/mnt/h/Research/datasets/phone/$REC; SET=/mnt/h/Research/results/replays/$REC/settings; O=/mnt/h/Research/results/replays/$REC
NODE=/opt/slam/emsdk/node/24.19.0_64bit/bin/node; JS=/root/build/snap-$TAG/wasm/orbslam3_run.js
echo "=== replay $REC $(date +%T) binary $JS presets $*"
for p in "$@"; do
  (
    st=$SET/${REC}_page-$p.yaml
    voc=/mnt/h/Research/vocab/ORBvoc.bin; grep -q "Web.BoW.LevelsUp: 3" $st && voc=/mnt/h/Research/vocab/ORBvoc_own_k10L5.bin
    g=""; grep -q "Web.Gyro.Prior: 1" $st && g="--gyro"
    for s in $SEEDS; do
      o=$O/$p/seed$s; mkdir -p $o
      timeout 1800 $NODE $JS --dataset euroc --mode mono --seq $SEQ --vocab $voc --settings $st --seed $s --live --rate-scale 0.5 $g \
        --out $o > $o/run.log 2>&1 || echo "$p seed $s: exit $?"
    done
  ) &
done
wait
/opt/slam/venv/bin/python eval/recording_metrics.py $O
echo "=== done $(date +%T)"
