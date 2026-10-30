#!/usr/bin/env bash
# The final filter (WebScaleFilter.h 858bac8: prune 100, 'stable over 3 s' 1-sigma) replayed over every live run of
# scaleF_1030 (same trajectories; eval/scale_filter_replay.cpp = the live code); outputs for replay_score.py
cd /mnt/h/Research/CustomSLAM
g++ -O2 -std=c++17 -I/usr/include/eigen3 -Islam/orbslam3/include eval/scale_filter_replay.cpp -o /root/build/scale_filter_replay 2>&1 | grep error
O=/mnt/h/Research/results/scaleF_1030_replay858; mkdir -p $O
for r in /mnt/h/Research/results/scaleF_1030/*/beta_scaleF*/seed*; do
  q=$(basename $(dirname $(dirname $r))); a=$(basename $(dirname $r)); s=$(basename $r); d=/mnt/h/Research/datasets/synth/$q
  off=-88; [[ $a == *ideal ]] && off=0
  /root/build/scale_filter_replay $r $d $d/mav0/imu0/data.csv $off $O/${q}__${a}__$s.txt
done
ls $O | wc -l
