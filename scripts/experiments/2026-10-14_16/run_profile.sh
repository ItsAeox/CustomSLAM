#!/usr/bin/env bash
# M1 (runs one at a time for clean timings): per-stage profile (ORB-SLAM3 REGISTER_TIMES) native vs WASM (node), EuRoC MH01 + TUM-VI room1, seed 1, real-time.
set -uo pipefail
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
echo "=== rebuild profiling builds $(date +%T)"
cmake --build /root/build/slam-native-prof -j3 --target orbslam3_run 2>&1 | grep -E "error|FAILED" || true
cmake --build /root/build/slam-wasm-prof -j3 --target orbslam3_run orbslam3_web_js 2>&1 | grep -E "error|FAILED" || true
D=/mnt/h/Research/datasets; S=/mnt/h/Research/CustomSLAM/baselines/orbslam3/settings; V=/mnt/h/Research/vocab/ORBvoc.bin
O=/mnt/h/Research/results/m1_profile
NODE=/opt/slam/emsdk/node/24.19.0_64bit/bin/node
run() {  # name flavour args...
  local name=$1 fl=$2; shift 2
  mkdir -p $O/$name/$fl
  if [ $fl = native ]; then /root/build/slam-native-prof/orbslam3_run "$@" --out $O/$name/$fl > $O/$name/$fl/run.log 2>&1
  else $NODE /root/build/slam-wasm-prof/orbslam3_run.js "$@" --out $O/$name/$fl > $O/$name/$fl/run.log 2>&1; fi
  echo "$name $fl: $(ls $O/$name/$fl | tr '\n' ' ')"
}
echo "=== runs $(date +%T)"
for fl in native wasm; do
  run euroc_MH_01_easy $fl --dataset euroc --mode mono --seq $D/euroc/MH_01_easy --vocab $V --settings $S/EuRoC_mono.yaml --seed 1 --realtime
  run tumvi_room1 $fl --dataset tumvi --mode mono --seq $D/tumvi/dataset-room1_512_16 --vocab $V --settings $S/TUM-VI_mono.yaml --clahe --seed 1 --realtime
done

echo "=== done $(date +%T)"
