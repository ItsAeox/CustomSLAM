#!/usr/bin/env bash
# Differential diagnosis native(clang+libc++) vs WASM: identical keypoints? first diverging frame (sync mapping)?
# Then the no-SIMD WASM sweep.
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
R=/mnt/h/Research/results/scratch
D=/mnt/h/Research/datasets/mocap-desk2
S=$R/settings/tumvie_mocap-desk2.yaml
V=/mnt/h/Research/vocab/ORBvoc.bin
NODE=/opt/slam/emsdk/node/24.19.0_64bit/bin/node
echo "=== libc++ native build $(date +%T)"
cmake --build /root/build/slam-native-libcxx --target orbslam3_run orb_dump -j 3 2>&1 | grep -E "error|FAILED" || true
echo "=== wasm build $(date +%T)"
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
cmake --build /root/build/slam-wasm --target orbslam3_run orb_dump -j 3 2>&1 | grep -E "error:|FAILED" || true
mkdir -p $R/diff
echo "=== keypoints libc++ vs wasm $(date +%T)"
for i in 00100 00250 00400; do
  /root/build/slam-native-libcxx/orb_dump $D/left_images/$i.jpg 1 1500 $R/diff/kp_libcxx_$i.txt > /dev/null
  (cd /root/build/slam-wasm && $NODE orb_dump.js $D/left_images/$i.jpg 1 1500 $R/diff/kp_wasm_$i.txt > /dev/null)
done
echo "=== traces (sync mapping, seed 1) $(date +%T)"
/root/build/slam-native-libcxx/orbslam3_run --dataset tumvie --mode mono --seq $D --vocab $V --settings $S \
  --out $R/diff/libcxx_sync --clahe --seed 1 --sync-mapping --trace > $R/diff/libcxx_sync.log 2>&1
(cd /root/build/slam-wasm && $NODE orbslam3_run.js --dataset tumvie --mode mono --seq $D --vocab $V --settings $S \
  --out $R/diff/wasm_sync --clahe --seed 1 --sync-mapping --trace > $R/diff/wasm_sync.log 2>&1)
for f in libcxx_sync wasm_sync; do
  /opt/slam/venv/bin/python eval/evaluate.py --gt $R/mocap-desk2/gt_tum.txt --est $R/diff/$f/traj_tum.txt \
    --out $R/diff/$f/metrics.json --label $f | tail -1
done
echo "=== wasm no-simd build $(date +%T)"
emcmake cmake -S slam -B /root/build/slam-wasm-nosimd -G Ninja -DBUILD_WASM=ON -DWEB_SIMD=OFF -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/opt/slam/opencv-4.10.0-wasm-mt/lib/cmake/opencv4 > /dev/null
cmake --build /root/build/slam-wasm-nosimd --target orbslam3_run -j 3 2>&1 | grep -E "error:|FAILED" || true
echo "=== sweep no-simd $(date +%T)"
FLAVOURS="wasm" WASM_JS=/root/build/slam-wasm-nosimd/orbslam3_run.js EXTRA="--realtime" \
  bash scripts/seed_sweep.sh "$R/seed_sweep_nosimd" "1 2 3 4 5" "mocap-desk2:$S:$R/mocap-desk2/gt_tum.txt"
/opt/slam/venv/bin/python eval/summarize_sweep.py "$R/seed_sweep_nosimd"
echo "=== done $(date +%T)"
