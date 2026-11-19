#!/usr/bin/env bash
# G18 fix check: 24 short mono-inertial runs (rooms 1/3/5 x seeds 1-8, 600 frames, rt-wall, 8 at a time), build dir binary
cd /mnt/h/Research/CustomSLAM
B=/root/build/slam-native-teblid/orbslam3_run; D=/mnt/h/Research/datasets/tumvi; VOC=/mnt/h/Research/vocab/ORBvoc.bin
SET=baselines/orbslam3/settings/TUM-VI_mono_inertial.yaml; O=/mnt/h/Research/results/scratch/verify_g18; rm -rf $O; mkdir -p $O
for r in 1 3 5; do
  for s in 1 2 3 4 5 6 7 8; do n=r${r}s$s; mkdir -p $O/$n
    $B --dataset tumvi --mode mono_inertial --seq $D/dataset-room${r}_512_16 --vocab $VOC --settings $SET --clahe --seed $s \
      --rt-wall --max-frames 600 --out $O/$n > $O/$n/run.log 2>&1; echo "$n exit $?" >> $O/exits.txt &
  done; wait
done
echo "exits: $(awk '{print $3}' $O/exits.txt | sort | uniq -c | tr '\n' ' ')"
echo "runs with 'scale too small': $(grep -l 'scale too small' $O/*/run.log | wc -l)"
