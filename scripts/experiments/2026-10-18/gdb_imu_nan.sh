#!/usr/bin/env bash
# IMU NaN abort (replays_1018_imu: "SO3::exp failed! omega: nan" in mono-inertial on phone recordings): reproduce
# natively under gdb (stop on the abort, backtrace of the failing thread + all threads). Usage: gdb_imu_nan.sh <out> <rec> <seed> [rate]
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
O=$1 R=$2 S=$3 RATE=${4:-1.0}
B=${B:-/root/build/snap-bbb071c/native/orbslam3_run} PH=/mnt/h/Research/datasets/phone
ST=/mnt/h/Research/results/replays_1018_imu/$R/settings/${R}_arm-imu_dt88.yaml
mkdir -p $O
gdb -q -batch -ex "set pagination off" -ex "handle SIGPIPE nostop noprint" -ex run -ex "bt 30" -ex "thread apply all bt 8" \
  --args $B --dataset euroc --mode mono_inertial --seq $PH/$R --vocab /mnt/h/Research/vocab/ORBvoc.bin --settings $ST \
  --seed $S ${PACE:---live --rate-scale $RATE} --imu-dt -88 --out $O > $O/gdb.log 2>&1
echo "exit $?; $(grep -c 'SO3::exp failed' $O/gdb.log) NaN aborts; $(grep -m1 -n '^#0' $O/gdb.log)"
