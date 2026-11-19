#!/usr/bin/env bash
# Diagnostics 3: coast2 (+ local-map PnP), speed (parallel ORB + warm fast pose), warm compare; V2_03 + room2, live@0.5
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; V=/mnt/h/Research/results/settings_variants; T=/mnt/h/Research/results/scratch/coast_try3
B=/root/build/slam-native-teblid/orbslam3_run; VOC=/mnt/h/Research/vocab/ORBvoc.bin
run() {  # variant gyroflag
  local v=$1 g=$2
  mkdir -p $T/$v/v203 $T/$v/room2
  $B --dataset euroc --mode mono --seq $D/euroc/V2_03_difficult --vocab $VOC --settings $V/EuRoC_mono_$v.yaml --seed 1 --live --rate-scale 0.5 $g --out $T/$v/v203 > $T/$v/v203/run.log 2>&1 &
  $B --dataset tumvi --mode mono --seq $D/tumvi/dataset-room2_512_16 --vocab $VOC --settings $V/TUM-VI_mono_$v.yaml --clahe --seed 1 --live --rate-scale 0.5 $g --out $T/$v/room2 > $T/$v/room2/run.log 2>&1 &
}
run fast-coast2 ""; run fast-gyro-coast2 --gyro; run fast-speed ""; run fast-posecheckw ""; run fast ""
wait
for v in fast fast-coast2 fast-gyro-coast2 fast-speed fast-posecheckw; do for s in v203 room2; do
  echo "$v $s: $(grep -E 'RECOVERY-STATS' $T/$v/$s/run.log | cut -d' ' -f2-) $(awk '{n++; c[$3]++} END{printf "ok %.2f rlost %.2f", c[2]/n, c[3]/n}' $T/$v/$s/track_times.txt) maps $(grep -c 'Creation of new map' $T/$v/$s/run.log)"
  grep FASTPOSE-STATS $T/$v/$s/run.log
done; done
grep -h "local-map PnP" $T/fast-*coast2/*/run.log | head -12; for v in fast fast-speed; do for s in v203 room2; do echo "$v $s track ms mean $(awk "{s+=$2;n++} END{print s/n}" $T/$v/$s/track_times.txt)"; grep -h "KLT-STATS" $T/$v/$s/run.log | cut -c1-200; done; done
