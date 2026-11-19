#!/usr/bin/env bash
# Body-frame (IMU) GT for mono-inertial evaluation: EuRoC 11 + TUM-VI rooms 6.
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; P=/opt/slam/venv/bin/python
for s in $D/euroc/*; do n=$(basename $s); nice -n 19 $P eval/gt_asl.py --dataset euroc --seq $s --out $G/euroc_${n}_body.txt --frame body | tail -1; done
for r in room1 room2 room3 room4 room5 room6; do n=dataset-${r}_512_16
  nice -n 19 $P eval/gt_asl.py --dataset tumvi --seq $D/tumvi/$n --out $G/tumvi_${n}_body.txt --frame body | tail -1; done
