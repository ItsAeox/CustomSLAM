#!/usr/bin/env bash
# OpenVINS (native, baselines/openvins) on synthetic sequences: phone-like IMU (config synth_phone, -88 ms) on <seq>,
# ideal IMU (synth_ideal) on <seq>__imuideal; 4 at a time; evaluate.py per run (SE3 / Sim3 ATE vs GT); offline, not real time.
# Usage: openvins_synth.sh <out dir> "<seq ...>"   (default: all 42 base sequences)
O=$1; SEQS=${2:-$(cd /mnt/h/Research/datasets/synth && ls -d */ | tr -d / | grep -E '^[a-z]+[0-9]$')}
C=/mnt/h/Research/CustomSLAM/baselines/openvins/config D=/mnt/h/Research/datasets/synth G=/mnt/h/Research/results/gt
mkdir -p $O
one() {
  local seq=$1 cfg=$2 o=$O/$1
  [ -s $o/run_info.txt ] && return
  mkdir -p $o
  LD_LIBRARY_PATH=/root/build/openvins timeout 1200 /root/build/ov_run/ov_run $C/$cfg/estimator_config.yaml $D/$seq $o > $o/run.log 2>&1
  /opt/slam/venv/bin/python /mnt/h/Research/CustomSLAM/eval/evaluate.py --gt $G/synth_$seq.txt --est $o/traj_tum.txt --out $o/metrics.json > $o/eval.txt 2>&1
  echo "$seq: $(grep -E 'init_frame|wall_s|dt_cam' $o/run_info.txt | tr '\n' ' ') | $(tail -1 $o/eval.txt | sed 's/.*matched/matched/')"
}
export -f one; export O C D G
for s in $SEQS; do echo "$s synth_phone"; echo "${s}__imuideal synth_ideal"; done | xargs -P 4 -L 1 bash -c 'one $0 $1'
