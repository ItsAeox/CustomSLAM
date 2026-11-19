#!/usr/bin/env bash
# Turning losses (bed -> PC): per-frame trace (orbslam3_run --trace, extended 18 Oct) of the WASM runner, live@1.0, on
# recordings 3 (medium), 7-8 (turning) and 10 (revisit B), "Hybrid, fastest" and gyro recovery (-88 ms), seeds 1-2,
# then eval/loss_anatomy.py per run: what the 2 s before each loss looked like.
# Usage: turn_trace.sh <snapshot tag>
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
NODE=/opt/slam/emsdk/node/24.19.0_64bit/bin/node; JS=/root/build/snap-$1/wasm/orbslam3_run.js
PH=/mnt/h/Research/datasets/phone; S=/mnt/h/Research/results/replays_1018_arms; O=/mnt/h/Research/results/turn_trace
VOC=/mnt/h/Research/vocab/ORBvoc.bin; P=/opt/slam/venv/bin/python
RECS="rec_20261018_110106 rec_20261018_110500 rec_20261018_110548 rec_20261018_110751"
for r in $RECS; do for a in fast grecover_dt; do for s in 1 2; do
  o=$O/$r/$a/seed$s; mkdir -p $o
  st=$S/$r/settings/${r}_arm-$a.yaml; g=""; grep -q "Web.Gyro.Prior: 1" $st && g="--gyro"
  [ -s $o/trace.txt ] || $NODE $JS --dataset euroc --mode mono --seq $PH/$r --vocab $VOC --settings $st --seed $s --live --rate-scale 1.0 $g \
    --trace --out $o > $o/run.log 2>&1 &
done; done; wait; done
for r in $RECS; do for a in fast grecover_dt; do for s in 1 2; do
  $P eval/loss_anatomy.py $O/$r/$a/seed$s $PH/$r
done; done; done
