#!/usr/bin/env bash
# Backtrace of the NaN abort (Sophus SO3::exp) with the patched native build: r40_j5_d5 seed 3 (aborts ~frame 1650).
set -uo pipefail
which gdb > /dev/null || { echo "no gdb"; exit 1; }
S=/mnt/h/Research/datasets_synth/euroc
R=/mnt/h/Research/results/g13_repro
O=$R/r40_j5_d5/gdb_seed3; mkdir -p $O
timeout 900 gdb -batch -ex "set pagination off" -ex run -ex "thread apply all bt 25" \
  --args /root/build/slam-native-teblid/orbslam3_run --dataset euroc --mode mono_inertial --seq $S/MH_01_easy_r40_j5_d5 \
  --vocab /mnt/h/Research/vocab/ORBvoc.bin --settings $R/settings/EuRoC_mono_inertial_r40_j5_d5.yaml --out $O --seed 3 > $O/gdb.log 2>&1
echo "exit $?" >> $O/gdb.log
grep -n "SIGABRT\|^Thread\|^#" $O/gdb.log | grep -v "libc\|pthread_kill\|raise\|abort ()" | head -80
