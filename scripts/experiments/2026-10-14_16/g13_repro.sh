#!/usr/bin/env bash
# G13 reproduction: ORB-SLAM3 mono-inertial with phone-like IMU (EuRoC MH01, IMU resampled to 40 / 30 Hz + jitter).
# A hang shows as exit 124 (timeout) with "Active map reset, waiting..." as the last reset line.
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
P=/opt/slam/venv/bin/python
D=/mnt/h/Research/datasets/euroc/MH_01_easy
S=/mnt/h/Research/datasets_synth/euroc
R=/mnt/h/Research/results/g13_repro
NATIVE=/root/build/slam-native-teblid/orbslam3_run
VOCAB=/mnt/h/Research/vocab/ORBvoc.bin
SET=baselines/orbslam3/settings/EuRoC_mono_inertial.yaml
mkdir -p $R/settings
echo "=== start $(date +%T)"
for v in "r40_j5_d5|40|5|0.05" "r30_j5|30|5|0"; do
  IFS='|' read -r name rate jit drop <<< "$v"
  $P scripts/degrade_imu.py $D $S/MH_01_easy_$name --rate $rate --jitter-ms $jit --drop $drop --seed 1
  sed "s/^IMU.Frequency:.*/IMU.Frequency: ${rate}.0/" $SET > $R/settings/EuRoC_mono_inertial_$name.yaml
done
run() {  # name seed
  local O=$R/$1/native_seed$2; mkdir -p $O
  timeout 600 $NATIVE --dataset euroc --mode mono_inertial --seq $S/MH_01_easy_$1 --vocab $VOCAB \
    --settings $R/settings/EuRoC_mono_inertial_$1.yaml --out $O --seed $2 > $O/run.log 2>&1
  echo "exit $?" >> $O/run.log
}
for name in r40_j5_d5 r30_j5; do
  run $name 1 & run $name 2 & wait
  run $name 3
done
for f in $R/*/native_seed*/run.log; do
  echo "--- $f"
  echo "  $(tail -1 $f) | empty-IMU frames $(grep -c 'Empty IMU measurements' $f) | IMU resets $(grep -c 'IMU is not or recently' $f)" \
       "| reset waits $(grep -c 'Active map reset, waiting' $f) done $(grep -c 'Active map reset, Done' $f)" \
       "| IMU init $(grep -c 'end VIBA 1' $f) VIBA2 $(grep -c 'end VIBA 2' $f)"
  grep -v "^frame\|GROWABLE" $f | tail -4 | cut -c1-150 | sed 's/^/    /'
done
echo "=== done $(date +%T)"
