#!/usr/bin/env bash
# Functional check of Web.Track.Coast (working-tree native build): V2_03 + room2, live@0.5, preset-fast vs + coast
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; T=/mnt/h/Research/results/scratch/coast_try; B=/root/build/slam-native-teblid/orbslam3_run
for v in fast fast-coast; do
  mkdir -p $T/$v/v203 $T/$v/room2
  $B --dataset euroc --mode mono --seq $D/euroc/V2_03_difficult --vocab /mnt/h/Research/vocab/ORBvoc.bin --settings $T/EuRoC_$v.yaml --seed 1 --live --rate-scale 0.5 --out $T/$v/v203 > $T/$v/v203/run.log 2>&1 &
  $B --dataset tumvi --mode mono --seq $D/tumvi/dataset-room2_512_16 --vocab /mnt/h/Research/vocab/ORBvoc.bin --settings $T/TUM-VI_$v.yaml --clahe --seed 1 --live --rate-scale 0.5 --out $T/$v/room2 > $T/$v/room2/run.log 2>&1 &
done
wait
for v in fast fast-coast; do for s in v203 room2; do
  echo "$v $s: $(grep -E 'RECOVERY-STATS' $T/$v/$s/run.log) $(awk '{n++; c[$3]++} END{printf "ok %.2f rlost %.2f", c[2]/n, c[3]/n}' $T/$v/$s/track_times.txt) maps $(grep -c 'Creation of new map' $T/$v/$s/run.log)"
done; done
