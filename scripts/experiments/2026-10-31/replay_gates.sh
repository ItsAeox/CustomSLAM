#!/usr/bin/env bash
# 31 Oct (ISSUES G32): the scale filter replayed with frames left out after fast rotation (GATE_W rad/s within GATE_WIN s)
# or soon after a loss (GATE_LOST s); eval/scale_filter_replay.cpp test gates. Outputs <results>_gate_<name>/ for
# replay_summary.py.   replay_gates.sh <results dir> <arm> "<name:VAR=v,VAR=v ...>"
cd /mnt/h/Research/CustomSLAM
g++ -O2 -std=c++17 -I/usr/include/eigen3 -Islam/orbslam3/include eval/scale_filter_replay.cpp -o /root/build/scale_filter_replay 2>&1 | grep error
L=$1 ARM=$2
for v in $3; do
  name=${v%%:*} envs=${v#*:}; O=${L}_gate_$name; mkdir -p $O
  for r in $L/*/$ARM/seed*; do
    q=$(basename $(dirname $(dirname $r))); s=$(basename $r); d=/mnt/h/Research/datasets/synth/$q
    [ -s $O/${q}__${ARM}__$s.txt ] || env $(echo $envs | tr ',' ' ') /root/build/scale_filter_replay $r $d $d/mav0/imu0/data.csv -88 $O/${q}__${ARM}__$s.txt &
    while [ $(jobs -r | wc -l) -ge 12 ]; do wait -n; done
  done
  wait
  echo "$name ($envs): $(ls $O | wc -l) runs"
done
