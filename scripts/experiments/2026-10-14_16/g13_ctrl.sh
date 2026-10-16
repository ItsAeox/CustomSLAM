#!/usr/bin/env bash
# G13 protocol control: is the poor accuracy of the g13 runs the IMU rate, or the unpaced protocol / the script?
#   orig200:     original MH01 (200 Hz IMU), same run() as the g13 runs (patched native, unpaced)
#   r200_near:   degrade_imu.py --rate 200 round trip (should equal orig200 if the script is faithful)
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
P=/opt/slam/venv/bin/python
D=/mnt/h/Research/datasets/euroc/MH_01_easy
S=/mnt/h/Research/datasets_synth/euroc
R=/mnt/h/Research/results/g13_repro
NATIVE=/root/build/slam-native-teblid/orbslam3_run
VOCAB=/mnt/h/Research/vocab/ORBvoc.bin
SET=baselines/orbslam3/settings/EuRoC_mono_inertial.yaml
echo "=== start $(date +%T)"
$P scripts/degrade_imu.py $D $S/MH_01_easy_r200_near --rate 200 --seed 1
cp $SET $R/settings/EuRoC_mono_inertial_r200_near.yaml
run() {  # name seqdir seed
  local O=$R/$1/native_fix_seed$3; mkdir -p $O
  timeout 900 $NATIVE --dataset euroc --mode mono_inertial --seq $2 --vocab $VOCAB \
    --settings $SET --out $O --seed $3 > $O/run.log 2>&1
  echo "exit $?" >> $O/run.log
  $P eval/evaluate.py --gt /mnt/h/Research/results/gt/euroc_MH_01_easy_body.txt \
    --est $O/traj_tum.txt --out $O/metrics.json --label "g13 $1 native seed$3" > /dev/null 2>&1 || true
}
run orig200 $D 1 & run orig200 $D 2 & wait
run orig200 $D 3 & run r200_near $S/MH_01_easy_r200_near 1 & wait
run r200_near $S/MH_01_easy_r200_near 2
for d in $R/orig200/native_fix_seed* $R/r200_near/native_fix_seed*; do
  f=$d/run.log; m=$(tr -d '\n ' < $d/metrics.json 2>/dev/null)
  echo "--- ${d#$R/}: $(tail -1 $f) | IMU resets $(grep -c 'IMU is not or recently' $f) | VIBA2 $(grep -c 'end VIBA 2' $f) | NaN $(grep -c 'SO3::exp failed' $f)" \
       "| se3 $(echo $m | grep -oE '"ate_se3":\{"rmse":[0-9.e-]+' | grep -oE '[0-9.e-]+$') scale $(echo $m | grep -oE '"sim3_scale":[0-9.]+' | cut -d: -f2) cov $(echo $m | grep -oE '"coverage":[0-9.]+' | cut -d: -f2)"
done
echo "=== done $(date +%T)"
