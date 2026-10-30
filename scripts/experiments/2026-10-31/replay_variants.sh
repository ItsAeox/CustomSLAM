#!/usr/bin/env bash
# 31 Oct (ISSUES G32): the scale filter (WebScaleFilter.h via eval/scale_filter_replay.cpp) replayed over the live runs of a
# results dir with the bank lifetime (RESTART s; a new bank every RESTART / 2) and the 'stable' look-back (STABLEBACK s)
# varied; outputs per variant for replay_summary.py (converged runs, error, wrong claims > 10 %).
#   replay_variants.sh <results dir> <arm> "<restart:stableback ...>"
cd /mnt/h/Research/CustomSLAM
g++ -O2 -std=c++17 -I/usr/include/eigen3 -Islam/orbslam3/include eval/scale_filter_replay.cpp -o /root/build/scale_filter_replay 2>&1 | grep error
L=$1 ARM=$2
for v in $3; do
  rs=${v%%:*} sb=${v##*:}; O=${L}_replay_r${rs}_b${sb}; mkdir -p $O
  for r in $L/*/$ARM/seed*; do
    q=$(basename $(dirname $(dirname $r))); s=$(basename $r); d=/mnt/h/Research/datasets/synth/$q
    [ -s $O/${q}__${ARM}__$s.txt ] || RESTART=$rs STABLEBACK=$sb /root/build/scale_filter_replay $r $d $d/mav0/imu0/data.csv -88 $O/${q}__${ARM}__$s.txt &
    while [ $(jobs -r | wc -l) -ge 12 ]; do wait -n; done
  done
  wait
  echo "restart $rs stableback $sb: $(ls $O | wc -l) runs"
done
