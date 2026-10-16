#!/usr/bin/env bash
# Bisect OpenCV-wasm SIMD: build OpenCV-wasm without intrinsics, rebuild the port against it, compare keypoints
# with native libc++ and run a 5-seed real-time sweep on mocap-desk2.
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
R=/mnt/h/Research/results/scratch
D=/mnt/h/Research/datasets/mocap-desk2
S=$R/settings/tumvie_mocap-desk2.yaml
NODE=/opt/slam/emsdk/node/24.19.0_64bit/bin/node
echo "=== opencv scalar $(date +%T)"
JOBS=6 bash scripts/setup_env.sh opencv_wasm_mt_scalar > /mnt/h/Research/logs/setup_opencv_wasm_scalar.log 2>&1
grep "=== " /mnt/h/Research/logs/setup_opencv_wasm_scalar.log | tail -1
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
echo "=== slam wasm (scalar opencv) build $(date +%T)"
emcmake cmake -S slam -B /root/build/slam-wasm-cvscalar -G Ninja -DBUILD_WASM=ON -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/opt/slam/opencv-4.10.0-wasm-mt-scalar/lib/cmake/opencv4 > /dev/null
cmake --build /root/build/slam-wasm-cvscalar --target orbslam3_run orb_dump -j 4 2>&1 | grep -E "error:|FAILED" || true
echo "=== keypoints: scalar-wasm vs native libc++ $(date +%T)"
mkdir -p $R/diff
for i in 00100 00250 00400; do
  (cd /root/build/slam-wasm-cvscalar && $NODE orb_dump.js $D/left_images/$i.jpg 1 1500 $R/diff/kp_wasmscalar_$i.txt > /dev/null)
  if cmp -s $R/diff/kp_libcxx_$i.txt $R/diff/kp_wasmscalar_$i.txt; then echo "img $i: IDENTICAL to native libc++";
  else echo "img $i: differs ($(diff <(cut -d' ' -f1-3 $R/diff/kp_libcxx_$i.txt) <(cut -d' ' -f1-3 $R/diff/kp_wasmscalar_$i.txt) | grep -c '^[<>]') lines)"; fi
done
echo "=== sweep (scalar opencv) $(date +%T)"
FLAVOURS="wasm" WASM_JS=/root/build/slam-wasm-cvscalar/orbslam3_run.js EXTRA="--realtime" \
  bash scripts/seed_sweep.sh "$R/seed_sweep_cvscalar" "1 2 3 4 5" "mocap-desk2:$S:$R/mocap-desk2/gt_tum.txt"
/opt/slam/venv/bin/python eval/summarize_sweep.py "$R/seed_sweep_cvscalar"
echo "=== done $(date +%T)"
