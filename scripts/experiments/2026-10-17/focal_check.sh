#!/usr/bin/env bash
# Focal length from the gyro (estimate_time_offset.py --focal): visual-only runs with the focal deliberately 10 % off
# (fast-fx090 / fast-fx110) vs right (preset-fast); room1 + V1_02, live@0.5, seeds 1-2, then the rate-ratio estimate.
# Expected: ratio ~ f_true / f_assumed = 1/0.9 = 1.11 and 1/1.1 = 0.91.
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; V=/mnt/h/Research/results/settings_variants; P=/opt/slam/venv/bin/python
B=/root/build/snap-$1/native/orbslam3_run; VOC=/mnt/h/Research/vocab/ORBvoc.bin; O=/mnt/h/Research/results/focal_check
for v in preset-fast fast-fx090 fast-fx110; do for s in 1 2; do
  mkdir -p $O/$v/room1_s$s $O/$v/v102_s$s
  $B --dataset tumvi --mode mono --seq $D/tumvi/dataset-room1_512_16 --vocab $VOC --settings $V/TUM-VI_mono_$v.yaml --clahe --seed $s --live --rate-scale 0.5 --out $O/$v/room1_s$s > $O/$v/room1_s$s/run.log 2>&1 &
  $B --dataset euroc --mode mono --seq $D/euroc/V1_02_medium --vocab $VOC --settings $V/EuRoC_mono_$v.yaml --seed $s --live --rate-scale 0.5 --out $O/$v/v102_s$s > $O/$v/v102_s$s/run.log 2>&1 &
done; wait; done
for v in preset-fast fast-fx090 fast-fx110; do for s in 1 2; do
  $P eval/estimate_time_offset.py --traj $O/$v/room1_s$s/traj_tum.txt --imu $D/tumvi/dataset-room1_512_16/mav0/imu0/data.csv --tbc $V/TUM-VI_mono_preset-fast-gyro.yaml --focal | tail -1 | sed "s#^#$v room1 s$s #"
  $P eval/estimate_time_offset.py --traj $O/$v/v102_s$s/traj_tum.txt --imu $D/euroc/V1_02_medium/mav0/imu0/data.csv --tbc $V/EuRoC_mono_preset-fast-gyro.yaml --focal | tail -1 | sed "s#^#$v v102 s$s #"
done; done
