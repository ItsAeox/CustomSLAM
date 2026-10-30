#!/usr/bin/env bash
# Native OpenVINS reference (claude/PLAN.md 4d, approach D). Sources on H: (/mnt/h/Research/ext: open_vins shallow clone of
# rpng/open_vins master 6948812, ceres-solver 2.2.0); builds in /root/build (WSL disk, H:\WSL). -j4 (7 GB VM).
#   Ceres 2.2.0: miniglog, no gflags / SuiteSparse / CUDA (apt's libceres-dev needs libunwind-dev, which would remove
#   libc++-14-dev used by the libc++ parity build) -> /opt/ceres
#   OpenVINS ov_msckf without ROS and without ArUco -> /root/build/openvins/libov_msckf_lib.so
#   ov_run (this folder) -> /root/build/ov_run/ov_run
set -euo pipefail
E=/mnt/h/Research/ext
if [ ! -f /opt/ceres/lib/cmake/Ceres/CeresConfig.cmake ]; then
  cmake -S $E/ceres-solver -B /root/build/ceres -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/ceres -DMINIGLOG=ON \
    -DGFLAGS=OFF -DSUITESPARSE=OFF -DUSE_CUDA=OFF -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF \
    -DBUILD_DOCUMENTATION=OFF > /root/build/ceres_cmake.log 2>&1 || { tail -20 /root/build/ceres_cmake.log; exit 1; }
  nice -n 10 cmake --build /root/build/ceres -j4 2>&1 | { grep -E "error|Error" || true; } | head -20
  cmake --install /root/build/ceres > /dev/null
fi
echo "=== ceres $(date +%T)"
cmake -S $E/open_vins/ov_msckf -B /root/build/openvins -DCMAKE_BUILD_TYPE=Release -DENABLE_ROS=OFF -DENABLE_ARUCO_TAGS=OFF \
  -DCeres_DIR=/opt/ceres/lib/cmake/Ceres > /root/build/openvins_cmake.log 2>&1 || { tail -20 /root/build/openvins_cmake.log; exit 1; }
nice -n 10 cmake --build /root/build/openvins -j4 --target ov_msckf_lib 2>&1 | { grep -E "error|Error" || true; } | head -20
echo "=== openvins $(date +%T)"
cmake -S /mnt/h/Research/CustomSLAM/baselines/openvins -B /root/build/ov_run -DCMAKE_BUILD_TYPE=Release \
  -DCeres_DIR=/opt/ceres/lib/cmake/Ceres -DOPENVINS_SRC=$E/open_vins -DOPENVINS_LIB=/root/build/openvins/libov_msckf_lib.so \
  > /root/build/ov_run_cmake.log 2>&1 || { tail -20 /root/build/ov_run_cmake.log; exit 1; }
cmake --build /root/build/ov_run -j4 2>&1 | { grep -E "error|Error" || true; } | head -20
ls -la --time-style=+%T /root/build/openvins/libov_msckf_lib.so /root/build/ov_run/ov_run
echo "=== done $(date +%T)"
