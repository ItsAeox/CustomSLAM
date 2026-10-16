#!/usr/bin/env bash
# Rebuild native + wasm ports, convert vocabulary, rerun parity (log: /mnt/h/Research/logs/m0_rebuild_parity.log)
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
echo "=== native build $(date +%T)"
cmake -S slam -B /root/build/slam-native -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/opt/slam/opencv-4.10.0-native/lib/cmake/opencv4 > /dev/null
cmake --build /root/build/slam-native -j 3 2>&1 | grep -E "error|FAILED" || true
echo "=== vocab convert $(date +%T)"
mkdir -p /mnt/h/Research/vocab
/root/build/slam-native/vocab_convert /mnt/h/Research/third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt /mnt/h/Research/vocab/ORBvoc.bin
echo "vocab_convert exit=$?"
ls -la /mnt/h/Research/vocab/
echo "=== wasm build $(date +%T)"
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
cmake --build /root/build/slam-wasm -j 3 2>&1 | grep -E "error:|FAILED" || true
echo "=== parity $(date +%T)"
R=/mnt/h/Research/results/scratch
rm -rf "$R/parity2_mocap-desk2"
VOCAB=/mnt/h/Research/vocab/ORBvoc.bin bash scripts/parity_runs.sh tumvie /mnt/h/Research/datasets/mocap-desk2 \
  "$R/settings/tumvie_mocap-desk2.yaml" "$R/mocap-desk2/gt_tum.txt" "$R/parity2_mocap-desk2" 5 mono --clahe 2>&1 | grep -E "matched|failed"
for r in native_1 wasm_1; do echo "$r: $(grep -m1 'New Map created' $R/parity2_mocap-desk2/$r/run.log)"; done
echo "=== done $(date +%T)"
