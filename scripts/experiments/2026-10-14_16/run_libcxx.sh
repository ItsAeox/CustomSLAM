#!/usr/bin/env bash
# After the half-speed sweep: build OpenCV + the slam port natively with clang + libc++, then sweep it.
set -uo pipefail
until grep -q "=== done" /mnt/h/Research/logs/seed_sweep_half.log 2>/dev/null; do sleep 10; done
cd /mnt/h/Research/CustomSLAM
echo "=== opencv libc++ $(date +%T)"
JOBS=6 bash scripts/setup_env.sh opencv_native_libcxx > /mnt/h/Research/logs/setup_opencv_libcxx.log 2>&1
grep "=== " /mnt/h/Research/logs/setup_opencv_libcxx.log | tail -1
echo "=== slam libc++ build $(date +%T)"
cmake -S slam -B /root/build/slam-native-libcxx -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang-14 -DCMAKE_CXX_COMPILER=clang++-14 \
  -DCMAKE_CXX_FLAGS="-stdlib=libc++" -DCMAKE_EXE_LINKER_FLAGS="-stdlib=libc++ -lc++abi" \
  -DOpenCV_DIR=/opt/slam/opencv-4.10.0-native-libcxx/lib/cmake/opencv4 > /dev/null
cmake --build /root/build/slam-native-libcxx --target orbslam3_run -j 4 2>&1 | grep -E "error|FAILED" || true
ls -la /root/build/slam-native-libcxx/orbslam3_run
R=/mnt/h/Research/results/scratch
echo "=== libc++ sweep $(date +%T)"
FLAVOURS="native" NATIVE=/root/build/slam-native-libcxx/orbslam3_run EXTRA="--realtime" \
  bash scripts/seed_sweep.sh "$R/seed_sweep_libcxx" "1 2 3 4 5" \
  "mocap-desk2:$R/settings/tumvie_mocap-desk2.yaml:$R/mocap-desk2/gt_tum.txt"
/opt/slam/venv/bin/python eval/summarize_sweep.py "$R/seed_sweep_libcxx"
echo "=== done $(date +%T)"
