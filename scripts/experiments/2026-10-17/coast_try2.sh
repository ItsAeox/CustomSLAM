#!/usr/bin/env bash
# Diagnostics (working-tree native build): coast without / with gyro, pose solver compare mode; V2_03 + room2, live@0.5
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; V=/mnt/h/Research/results/settings_variants; T=/mnt/h/Research/results/scratch/coast_try2
B=/root/build/slam-native-teblid/orbslam3_run; VOC=/mnt/h/Research/vocab/ORBvoc.bin
run() {  # variant gyroflag
  local v=$1 g=$2
  mkdir -p $T/$v/v203 $T/$v/room2
  $B --dataset euroc --mode mono --seq $D/euroc/V2_03_difficult --vocab $VOC --settings $V/EuRoC_mono_$v.yaml --seed 1 --live --rate-scale 0.5 $g --out $T/$v/v203 > $T/$v/v203/run.log 2>&1 &
  $B --dataset tumvi --mode mono --seq $D/tumvi/dataset-room2_512_16 --vocab $VOC --settings $V/TUM-VI_mono_$v.yaml --clahe --seed 1 --live --rate-scale 0.5 $g --out $T/$v/room2 > $T/$v/room2/run.log 2>&1 &
}
run fast-coast ""; run fast-gyro-coast --gyro; run fast-posecheck ""
wait
for v in fast-coast fast-gyro-coast fast-posecheck; do for s in v203 room2; do
  echo "$v $s: $(grep -E 'RECOVERY-STATS' $T/$v/$s/run.log | cut -d' ' -f2-) $(awk '{n++; c[$3]++} END{printf "ok %.2f rlost %.2f", c[2]/n, c[3]/n}' $T/$v/$s/track_times.txt) maps $(grep -c 'Creation of new map' $T/$v/$s/run.log)"
  grep FASTPOSE-STATS $T/$v/$s/run.log
done; done
grep -h "coast try" $T/fast-gyro-coast/*/run.log | head -30
