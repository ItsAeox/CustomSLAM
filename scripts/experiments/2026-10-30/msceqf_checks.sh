#!/usr/bin/env bash
# MSCEqF (native, /root/build/msceqf/msceqf_run, baselines/msceqf configs) on the same runs as eqvio_checks.sh:
#   synth   all six variants of every synthetic scenario, phone (<seq>, camera lag 0.088 s) / ideal (<seq>__imuideal, 0)
#           IMU, plus the clean variant 0 with the phone IMU
#   real    the Pixel 9 recordings (phone config, camera lag 0.088 s)
#   drop    every 2nd / 3rd frame of variant 2 (slide3) per scenario and of the real 1 m slides
# The dropped-frame runs read a folder of their own on the WSL disk (/root/msceqf_data/<run>: links to the frames and the
# IMU, a new thinned cam0/data.csv); every other run reads the dataset directly (read only).
# Scoring: eqvio_checks_score.py <results dir> (traj_camclock.txt, timing.csv).
#   msceqf_checks.sh [parallel jobs, default 2]
set -uo pipefail
P=${1:-2}
B=/mnt/h/Research/CustomSLAM/baselines/msceqf; S=/mnt/h/Research/datasets/synth; R=/mnt/h/Research/datasets/phone
export O=/mnt/h/Research/results/msceqf_checks_1030 B
mkdir -p $O
one() {  # <run name> <source dir> <config> <camera lag s> <keep every k-th frame>
  local n=$1 src=$2 cfg=$3 lag=$4 k=$5 o=$O/$1
  [ -s $o/timing.csv ] && return
  mkdir -p $o
  if [ $k != 1 ]; then
    local d=/root/msceqf_data/$n
    rm -rf $d; mkdir -p $d/mav0/cam0
    ln -s $src/mav0/cam0/data $d/mav0/cam0/data
    ln -s $src/mav0/imu0 $d/mav0/imu0
    awk -v k=$k 'NR == 1 || (NR - 2) % k == 0' $src/mav0/cam0/data.csv > $d/mav0/cam0/data.csv
    src=$d
  fi
  local t0=$(date +%s)
  timeout 1800 /root/build/msceqf/msceqf_run $B/$cfg $src $lag $o > $o/run.log 2>&1
  echo "$n: exit $? in $(( $(date +%s) - t0 )) s"
}
export -f one
{
  for sc in ar gesture orbit rotate shake slide walk; do
    for v in 0 1 2 3 4 5; do
      echo "$sc$v $S/$sc$v config_synth_phone.yaml 0.088 1"
      echo "$sc${v}__imuideal $S/$sc${v}__imuideal config_synth_ideal.yaml 0 1"
    done
    echo "${sc}0_clean $S/${sc}0_clean config_synth_phone.yaml 0.088 1"
  done
  for r in rec_20261029_202938 rec_20261021_070221 rec_20261018_105829 rec_20261018_105951 rec_20261018_110106 \
           rec_20261018_110210 rec_20261018_110310 rec_20261018_110407 rec_20261018_110500 rec_20261018_110548 \
           rec_20261018_110632 rec_20261018_110751 rec_20261018_184922 rec_20261018_185037 rec_20261018_185134 \
           rec_20261018_220132 rec_20261018_220403; do
    echo "$r $R/$r config_synth_phone.yaml 0.088 1"
  done
  for k in 2 3; do
    for s in ar2 gesture2 orbit2 rotate2 shake2 slide3 walk2; do echo "${s}_k$k $S/$s config_synth_phone.yaml 0.088 $k"; done
    echo "rec_20261029_202938_k$k $R/rec_20261029_202938 config_synth_phone.yaml 0.088 $k"
  done
} | xargs -P $P -L 1 bash -c 'one $0 $1 $2 $3 $4'
echo "=== all done $(date +%T)"
