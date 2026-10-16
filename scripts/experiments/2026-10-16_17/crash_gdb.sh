#!/usr/bin/env bash
# Backtrace of a native mono-inertial crash (imu_sens base, room3 seed 2: segfault after "scale too small"), up to 3 tries
# (thread timing makes it nondeterministic). Usage: crash_gdb.sh <room> <seed>
set -uo pipefail
n=dataset-$1_512_16
O=/mnt/h/Research/results/imu_sens_gdb/$1_seed$2; mkdir -p $O
for try in 1 2 3; do
  timeout 900 gdb -batch -ex "set pagination off" -ex run -ex "bt 30" -ex "info threads" \
    --args /root/build/snap-cea2e3b/native/orbslam3_run --dataset tumvi --mode mono_inertial \
    --seq /mnt/h/Research/datasets/tumvi/$n --vocab /mnt/h/Research/vocab/ORBvoc.bin \
    --settings /mnt/h/Research/CustomSLAM/baselines/orbslam3/settings/TUM-VI_mono_inertial.yaml --out $O --seed $2 --clahe > $O/gdb_try$try.log 2>&1
  if grep -q "received signal" $O/gdb_try$try.log; then
    echo "try $try: crashed"; grep -A30 "received signal" $O/gdb_try$try.log | grep "^#" | cut -c1-200 | head -25; exit 0
  fi
  echo "try $try: no crash"
done
