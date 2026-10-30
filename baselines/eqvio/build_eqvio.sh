#!/usr/bin/env bash
# Native EqVIO reference (equivariant filter VIO, van Goor & Mahony, T-RO 2023; github.com/pvangoor/eqvio 3bcc01a, GPL-3.0;
# submodules GIFT a4ed387, LiePP 574f543, argparse eba16b3 checked out over HTTPS). Source on H: (/mnt/h/Research/ext/eqvio),
# build in /root/build/eqvio (WSL disk on H:). No tests, no visualisation, no rosbag. -j4.
set -euo pipefail
cmake -S /mnt/h/Research/ext/eqvio -B /root/build/eqvio -DCMAKE_BUILD_TYPE=Release -DEQVIO_BUILD_TESTS=OFF \
  -DEQVIO_BUILD_VISUALISATION=ON -DEQVIO_BUILD_ROSBAG=OFF -DEXTRA_WARNINGS=OFF > /root/build/eqvio_cmake.log 2>&1 \
  || { tail -20 /root/build/eqvio_cmake.log; exit 1; }
nice -n 10 cmake --build /root/build/eqvio -j4 --target eqvio_opt 2>&1 | { grep -E "error|Error" || true; } | head -20
ls -la --time-style=+%T /root/build/eqvio/eqvio_opt
echo "=== done $(date +%T)"
