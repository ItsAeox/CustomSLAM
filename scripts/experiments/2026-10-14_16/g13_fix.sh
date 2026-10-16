#!/usr/bin/env bash
# G13 fix: rebuild (Tracking.cc PreintegrateIMU zero-order hold, Optimizer.cc guard) and re-run the phone-like IMU
# reproduction with the patched builds. -j1 + nice: the inertial_rt2 sweep is running (WSL 7.8 GB).
set -uo pipefail
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
S=/mnt/h/Research/datasets_synth/euroc
R=/mnt/h/Research/results/g13_repro
NATIVE=/root/build/slam-native-teblid/orbslam3_run
NODE=/opt/slam/emsdk/node/24.19.0_64bit/bin/node
WASM_JS=/root/build/slam-wasm-ui/orbslam3_run.js
VOCAB=/mnt/h/Research/vocab/ORBvoc.bin
echo "=== build start $(date +%T)"
nice -n 10 cmake --build /root/build/slam-native-teblid -j1 --target orbslam3_run 2>&1 | grep -E "error|FAILED|Linking"
echo "=== native built $(date +%T)"
nice -n 10 cmake --build /root/build/slam-wasm-ui -j1 --target orbslam3_web_js orbslam3_run 2>&1 | grep -E "error|FAILED|Linking"
cp /root/build/slam-wasm-ui/orbslam3.js /root/build/slam-wasm-ui/orbslam3.wasm /root/build/slam-wasm-ui/orbslam3.worker.mjs /mnt/h/Research/CustomSLAM/web/build/
echo "=== wasm-ui built + copied to web/build $(date +%T)"
run() {  # name flavour seed
  local O=$R/$1/$2_fix_seed$3; mkdir -p $O
  local CMD=($NATIVE); [ $2 = wasm ] && CMD=($NODE $WASM_JS)
  timeout 900 "${CMD[@]}" --dataset euroc --mode mono_inertial --seq $S/MH_01_easy_$1 --vocab $VOCAB \
    --settings $R/settings/EuRoC_mono_inertial_$1.yaml --out $O --seed $3 > $O/run.log 2>&1
  echo "exit $?" >> $O/run.log
  /opt/slam/venv/bin/python /mnt/h/Research/CustomSLAM/eval/evaluate.py --gt /mnt/h/Research/results/gt/euroc_MH_01_easy_body.txt \
    --est $O/traj_tum.txt --out $O/metrics.json > /dev/null 2>&1 || true
}
for name in r40_j5_d5 r30_j5; do
  run $name native 1 & run $name native 2 & wait
  run $name native 3 & run $name wasm 1 & wait
  run $name wasm 2
done
for f in $R/*/*_fix_seed*/run.log; do
  d=$(dirname $f)
  m=$(tr -d '\n ' < $d/metrics.json 2>/dev/null)
  echo "--- ${d#$R/}: $(tail -1 $f) | empty-IMU frames $(grep -c 'Empty IMU' $f) | not-preintegrated $(grep -c 'Not preintegrated' $f)" \
       "| IMU resets $(grep -c 'IMU is not or recently' $f) | VIBA1 $(grep -c 'end VIBA 1' $f) VIBA2 $(grep -c 'end VIBA 2' $f)" \
       "| se3 $(echo $m | grep -oE '"ate_se3":\{"rmse":[0-9.e-]+' | grep -oE '[0-9.e-]+$') scale $(echo $m | grep -oE '"sim3_scale":[0-9.]+' | cut -d: -f2) cov $(echo $m | grep -oE '"coverage":[0-9.]+' | cut -d: -f2)"
done
echo "=== done $(date +%T)"
