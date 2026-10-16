#!/usr/bin/env bash
# G13: the same phone-like IMU sequence through the WASM runner (null pointer -> no trap in WASM: hang?)
set -uo pipefail
S=/mnt/h/Research/datasets_synth/euroc
R=/mnt/h/Research/results/g13_repro
NODE=/opt/slam/emsdk/node/24.19.0_64bit/bin/node
WASM_JS=/root/build/slam-wasm-ui/orbslam3_run.js
VOCAB=/mnt/h/Research/vocab/ORBvoc.bin
run() {  # name seed
  local O=$R/$1/wasm_seed$2; mkdir -p $O
  timeout 900 $NODE $WASM_JS --dataset euroc --mode mono_inertial --seq $S/MH_01_easy_$1 --vocab $VOCAB \
    --settings $R/settings/EuRoC_mono_inertial_$1.yaml --out $O --seed $2 > $O/run.log 2>&1
  echo "exit $?" >> $O/run.log
}
echo "=== start $(date +%T)"
run r40_j5_d5 2 & run r40_j5_d5 1 & wait
for f in $R/r40_j5_d5/wasm_seed*/run.log; do
  echo "--- $f: $(tail -1 $f) | empty-IMU $(grep -c 'Empty IMU' $f) | not-preintegrated $(grep -c 'Not preintegrated' $f)" \
       "| reset waits $(grep -c 'Active map reset, waiting' $f) done $(grep -c 'Active map reset, Done' $f) | last frame line: $(grep '^frame' $f | tail -1 | cut -c1-80)"
  grep -v "^frame\|GROWABLE\|Empty IMU" $f | tail -5 | cut -c1-150 | sed 's/^/    /'
done
echo "=== done $(date +%T)"
