#!/usr/bin/env bash
# G13 follow-up (patched builds): is the later NaN divergence (Sophus SO3::exp abort ~frame 2600) caused by the
# synthetic IMU (nearest-sample aliasing of the drone's rotor vibration) or by the zero-order hold?
#   r40_avg_j5_d5: 40 Hz averaged (anti-aliased, like a phone sensor), jitter 5 ms, 5% dropped  (hold used)
#   r100_near:     100 Hz nearest sample, no jitter                                            (hold never used)
set -uo pipefail
until grep -q "=== done" /mnt/h/Research/logs/g13_fix.log 2>/dev/null; do sleep 20; done
cd /mnt/h/Research/CustomSLAM
P=/opt/slam/venv/bin/python
D=/mnt/h/Research/datasets/euroc/MH_01_easy
S=/mnt/h/Research/datasets_synth/euroc
R=/mnt/h/Research/results/g13_repro
NATIVE=/root/build/slam-native-teblid/orbslam3_run
NODE=/opt/slam/emsdk/node/24.19.0_64bit/bin/node
WASM_JS=/root/build/slam-wasm-ui/orbslam3_run.js
VOCAB=/mnt/h/Research/vocab/ORBvoc.bin
SET=baselines/orbslam3/settings/EuRoC_mono_inertial.yaml
echo "=== start $(date +%T)"
# reproducibility check: the regenerated r40_j5_d5 must equal the one used above
$P scripts/degrade_imu.py $D /tmp/g13_regen --rate 40 --jitter-ms 5 --drop 0.05 --seed 1 > /dev/null
cmp /tmp/g13_regen/mav0/imu0/data.csv $S/MH_01_easy_r40_j5_d5/mav0/imu0/data.csv && echo "r40_j5_d5 regenerates identically"
rm -rf /tmp/g13_regen
$P scripts/degrade_imu.py $D $S/MH_01_easy_r40_avg_j5_d5 --rate 40 --jitter-ms 5 --drop 0.05 --seed 1 --average
$P scripts/degrade_imu.py $D $S/MH_01_easy_r100_near --rate 100 --seed 1
for v in r40_avg_j5_d5:40 r100_near:100; do
  sed "s/^IMU.Frequency:.*/IMU.Frequency: ${v#*:}.0/" $SET > $R/settings/EuRoC_mono_inertial_${v%:*}.yaml
done
run() {  # name flavour seed
  local O=$R/$1/$2_fix_seed$3; mkdir -p $O
  local CMD=($NATIVE); [ $2 = wasm ] && CMD=($NODE $WASM_JS)
  timeout 900 "${CMD[@]}" --dataset euroc --mode mono_inertial --seq $S/MH_01_easy_$1 --vocab $VOCAB \
    --settings $R/settings/EuRoC_mono_inertial_$1.yaml --out $O --seed $3 > $O/run.log 2>&1
  echo "exit $?" >> $O/run.log
  $P eval/evaluate.py --gt /mnt/h/Research/results/gt/euroc_MH_01_easy_body.txt \
    --est $O/traj_tum.txt --out $O/metrics.json --label "g13 $1 $2 seed$3" > /dev/null 2>&1 || true
}
run r40_avg_j5_d5 native 1 & run r40_avg_j5_d5 native 2 & wait
run r40_avg_j5_d5 native 3 & run r40_avg_j5_d5 wasm 1 & wait
run r100_near native 1 & run r100_near native 2 & wait
run r100_near native 3 & run r40_avg_j5_d5 wasm 2 & wait
for f in $R/r40_avg_j5_d5/*_fix_seed*/run.log $R/r100_near/*_fix_seed*/run.log; do
  d=$(dirname $f)
  m=$(tr -d '\n ' < $d/metrics.json 2>/dev/null)
  echo "--- ${d#$R/}: $(tail -1 $f) | empty-IMU $(grep -c 'Empty IMU' $f) | not-preint $(grep -c 'Not preintegrated' $f)" \
       "| IMU resets $(grep -c 'IMU is not or recently' $f) | VIBA1 $(grep -c 'end VIBA 1' $f) VIBA2 $(grep -c 'end VIBA 2' $f) | NaN abort $(grep -c 'SO3::exp failed' $f)" \
       "| se3 $(echo $m | grep -oE '"ate_se3":\{"rmse":[0-9.e-]+' | grep -oE '[0-9.e-]+$') scale $(echo $m | grep -oE '"sim3_scale":[0-9.]+' | cut -d: -f2) cov $(echo $m | grep -oE '"coverage":[0-9.]+' | cut -d: -f2)"
done
echo "=== done $(date +%T)"
