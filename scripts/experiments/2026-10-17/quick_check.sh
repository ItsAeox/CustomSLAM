#!/usr/bin/env bash
# Functional check (working-tree native build): MH01 + room1, live@0.5, seed 1. Usage: quick_check.sh <out> <arm...>
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; V=/mnt/h/Research/results/settings_variants; T=/mnt/h/Research/results/scratch/$1; shift
B=/root/build/slam-native-teblid/orbslam3_run; VOC=/mnt/h/Research/vocab/ORBvoc.bin
for v in "$@"; do
  g=""; [[ $v == *gyro* ]] && g="--gyro"
  mkdir -p $T/$v/mh01 $T/$v/room1
  $B --dataset euroc --mode mono --seq $D/euroc/MH_01_easy --vocab $VOC --settings $V/EuRoC_mono_$v.yaml --seed 1 --live --rate-scale 0.5 $g --out $T/$v/mh01 > $T/$v/mh01/run.log 2>&1 &
  $B --dataset tumvi --mode mono --seq $D/tumvi/dataset-room1_512_16 --vocab $VOC --settings $V/TUM-VI_mono_$v.yaml --clahe --seed 1 --live --rate-scale 0.5 $g --out $T/$v/room1 > $T/$v/room1/run.log 2>&1 &
done
wait
for v in "$@"; do for s in mh01 room1; do
  f=$T/$v/$s
  echo "$v $s: exit-log '$(tail -1 $f/run.log | cut -c1-60)' frames-before-map $(awk '$3==1{n++} END{print n+0}' $f/track_times.txt) ok $(awk '{n++; c[$3]++} END{printf "%.2f", c[2]/n}' $f/track_times.txt) maps $(( $(grep -c 'Creation of new map' $f/run.log) / 2 )) $(grep -h 'Gyro two-view' $f/run.log | head -2 | tr '\n' ' ')"
done; done
