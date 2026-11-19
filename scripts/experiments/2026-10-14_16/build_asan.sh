#!/usr/bin/env bash
# Native clang-14 + AddressSanitizer (incl. use-after-scope) + UBSan build of the port, to hunt UB like the
# EdgeSE3ProjectXYZ dangling temporary (invisible to Valgrind).
set -euo pipefail
B=/root/build/slam-native-asan
SAN="-fsanitize=address,undefined -fno-omit-frame-pointer -fsanitize-address-use-after-scope"
cmake -S /mnt/h/Research/CustomSLAM/slam -B $B -G Ninja -DCMAKE_BUILD_TYPE=None \
  -DCMAKE_C_COMPILER=clang-14 -DCMAKE_CXX_COMPILER=clang++-14 \
  -DCMAKE_C_FLAGS="-O1 -g $SAN" -DCMAKE_CXX_FLAGS="-O1 -g $SAN" \
  -DCMAKE_EXE_LINKER_FLAGS="$SAN" \
  -DOpenCV_DIR=/opt/slam/opencv-4.10.0-native/lib/cmake/opencv4 > /dev/null
cmake --build $B -j3 --target orbslam3_run numerics_check 2>&1 | grep -E "error|FAILED|Linking" || true
ls -la $B/orbslam3_run $B/numerics_check
