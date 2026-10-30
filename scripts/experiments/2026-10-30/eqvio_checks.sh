#!/usr/bin/env bash
# EqVIO (native, /root/build/eqvio/eqvio_opt, baselines/eqvio configs) checks before a browser port (claude/PLAN §0):
#   synth   all six variants of every synthetic scenario, phone (<seq>) / ideal (<seq>__imuideal) IMU, plus the clean
#           (no blur / rolling shutter) variant 0 with the phone IMU
#   real    the Pixel 9 recordings (same intrinsics / T_b_c as the synthetic set; phone config, camera lag 0.088 s)
#   drop    every 2nd / 3rd frame (15 / 10 fps) of variant 2 (slide3) per scenario and of the real 1 m slides
# Shadow folders on the WSL disk (/root/eqvio_data) add cam0/sensor.yaml and the thinned data.csv; datasets unchanged.
# --timing writes timing.csv (per-section seconds). Scoring: score_eqvio_checks.sh.
#   eqvio_checks.sh [parallel jobs, default 2]
set -uo pipefail
P=${1:-2}
B=/mnt/h/Research/CustomSLAM/baselines/eqvio; S=/mnt/h/Research/datasets/synth; R=/mnt/h/Research/datasets/phone
export O=/mnt/h/Research/results/eqvio_checks_1030 B
mkdir -p $O
one() {  # <run name> <source dir> <config> <keep every k-th frame>
  local n=$1 src=$2 cfg=$3 k=$4 d=/root/eqvio_data/$1/mav0 o=$O/$1
  [ -s $o/IMUState.csv ] && return
  mkdir -p $d/cam0 $o
  ln -sfn $src/mav0/cam0/data $d/cam0/data
  # run_eqvio.sh's shadow folders link data.csv to the dataset: remove the link (never write through it; on 30 Oct this
  # emptied synth/ar2{,__imuideal}/mav0/cam0/data.csv, restored exactly from make_dataset.py's stamp generation)
  rm -f $d/cam0/data.csv $d/cam0/sensor.yaml
  awk -v k=$k 'NR == 1 || (NR - 2) % k == 0' $src/mav0/cam0/data.csv > $d/cam0/data.csv
  ln -sfn $src/mav0/imu0 $d/imu0
  if [ -d $src/mav0/state_groundtruth_estimate0 ]; then
    ln -sfn $src/mav0/state_groundtruth_estimate0 $d/state_groundtruth_estimate0
  else  # phone recordings: eqvio_opt always reads a ground truth (for its viewer; a missing file crashes it) -> placeholder
    rm -rf $d/state_groundtruth_estimate0; mkdir -p $d/state_groundtruth_estimate0
    local t1=$(awk -F, 'NR == 2 {print $1}' $src/mav0/imu0/data.csv)
    { echo "#timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z,v_x,v_y,v_z,bw_x,bw_y,bw_z,ba_x,ba_y,ba_z"
      echo "$t1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0"; echo "$((t1 + 1000000)),0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0"; } > $d/state_groundtruth_estimate0/data.csv
  fi
  cp $B/camera_synth.yaml $d/cam0/sensor.yaml
  local t0=$(date +%s)
  ( cd $o && timeout 1800 /root/build/eqvio/eqvio_opt /root/eqvio_data/$n/ $B/$cfg -o $o/ --mode asl --timing > $o/run.log 2>&1 )
  echo "$n: exit $? in $(( $(date +%s) - t0 )) s"
}
export -f one
{
  for sc in ar gesture orbit rotate shake slide walk; do
    for v in 0 1 2 3 4 5; do
      echo "$sc$v $S/$sc$v config_synth_phone.yaml 1"
      echo "$sc${v}__imuideal $S/$sc${v}__imuideal config_synth_ideal.yaml 1"
    done
    echo "${sc}0_clean $S/${sc}0_clean config_synth_phone.yaml 1"
  done
  for r in rec_20261029_202938 rec_20261021_070221 rec_20261018_105829 rec_20261018_105951 rec_20261018_110106 \
           rec_20261018_110210 rec_20261018_110310 rec_20261018_110407 rec_20261018_110500 rec_20261018_110548 \
           rec_20261018_110632 rec_20261018_110751 rec_20261018_184922 rec_20261018_185037 rec_20261018_185134 \
           rec_20261018_220132 rec_20261018_220403; do
    echo "$r $R/$r config_synth_phone.yaml 1"
  done
  for k in 2 3; do
    for s in ar2 gesture2 orbit2 rotate2 shake2 slide3 walk2; do echo "${s}_k$k $S/$s config_synth_phone.yaml $k"; done
    echo "rec_20261029_202938_k$k $R/rec_20261029_202938 config_synth_phone.yaml $k"
  done
} | xargs -P $P -L 1 bash -c 'one $0 $1 $2 $3'
echo "=== all done $(date +%T)"
