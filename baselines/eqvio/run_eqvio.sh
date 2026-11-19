#!/usr/bin/env bash
# EqVIO (native) on synthetic sequences: a shadow folder per sequence on the WSL disk (/root/eqvio_data, H:\WSL) links the
# data and adds mav0/cam0/sensor.yaml (EqVIO's ASL reader needs it; camera_synth.yaml), so the dataset stays unchanged.
# Phone-like IMU: <seq> with config_synth_phone.yaml (camera lag 0.088 s); ideal: <seq>__imuideal, config_synth_ideal.yaml.
# Writes <out>/<seq>/ (EqVIO output + run.log); evaluate.py on its trajectory (eqvio_to_tum.py).
# Usage: run_eqvio.sh <out dir> "<seq ...>"
set -uo pipefail
O=$1; SEQS=$2; B=/mnt/h/Research/CustomSLAM/baselines/eqvio
one() {
  local s=$1 cfg=$2 d=/root/eqvio_data/$1/mav0 o=$O/$1
  mkdir -p $d/cam0 $o
  ln -sfn /mnt/h/Research/datasets/synth/$s/mav0/cam0/data $d/cam0/data
  ln -sfn /mnt/h/Research/datasets/synth/$s/mav0/cam0/data.csv $d/cam0/data.csv
  ln -sfn /mnt/h/Research/datasets/synth/$s/mav0/imu0 $d/imu0
  ln -sfn /mnt/h/Research/datasets/synth/$s/mav0/state_groundtruth_estimate0 $d/state_groundtruth_estimate0
  cp $B/camera_synth.yaml $d/cam0/sensor.yaml
  ( cd $o && timeout 1200 /root/build/eqvio/eqvio_opt /root/eqvio_data/$s/ $B/$cfg -o $o/ --mode asl > $o/run.log 2>&1 )
  echo "$s: exit $? $(ls $o | tr '\n' ' ')"
}
export -f one; export O B
for s in $SEQS; do echo "$s config_synth_phone.yaml"; echo "${s}__imuideal config_synth_ideal.yaml"; done | xargs -P 2 -L 1 bash -c 'one $0 $1'
# scoring: camera poses (eval/eqvio_to_tum.py), phone runs shifted back by the camera lag to the camera clock, evaluate.py
for o in $O/*/; do
  s=$(basename $o); [ -f $o/IMUState.csv ] || continue; [ -s $o/eval.txt ] && continue
  /opt/slam/venv/bin/python /mnt/h/Research/CustomSLAM/eval/eqvio_to_tum.py $o $o/traj_tum.txt > /dev/null
  if [[ $s == *__imuideal ]]; then cp $o/traj_tum.txt $o/traj_camclock.txt
  else awk 'NR==1{print; next} {printf "%.9f", $1+0.088; for(i=2;i<=8;i++) printf " %s", $i; print ""}' $o/traj_tum.txt > $o/traj_camclock.txt; fi
  /opt/slam/venv/bin/python /mnt/h/Research/CustomSLAM/eval/evaluate.py --gt /mnt/h/Research/results/gt/synth_$s.txt --est $o/traj_camclock.txt --out $o/metrics.json 2>&1 | tail -1 | sed "s|.*matched|$s: matched|" > $o/eval.txt
  cat $o/eval.txt
done
