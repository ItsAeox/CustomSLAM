#!/usr/bin/env bash
# slam/eqvio_web: native and WebAssembly builds (eqvio_run: replay; eqvio.js / eqvio.wasm: browser module -> web/build).
#   build_eqvio_web.sh [jobs, default 3]
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
J=${1:-3}
S=/mnt/h/Research/CustomSLAM/slam/eqvio_web
echo "=== start $(date +%T)"
if [ ! -f /root/build/eqvio-native/CMakeCache.txt ]; then
  cmake -S $S -B /root/build/eqvio-native -DOpenCV_DIR=/opt/slam/opencv-4.10.0-native/lib/cmake/opencv4 \
    > /root/build/eqvio-native.cmake.log 2>&1 || { tail -20 /root/build/eqvio-native.cmake.log; exit 1; }
fi
nice -n 10 cmake --build /root/build/eqvio-native -j$J --target eqvio_run 2>&1 | { grep -E "error|Error" || true; } | head -30
echo "=== native $(date +%T)"
if [ ! -f /root/build/eqvio-wasm/CMakeCache.txt ]; then
  emcmake cmake -S $S -B /root/build/eqvio-wasm -DBUILD_WASM=ON -DOpenCV_DIR=/opt/slam/opencv-4.10.0-wasm-mt/lib/cmake/opencv4 \
    > /root/build/eqvio-wasm.cmake.log 2>&1 || { tail -20 /root/build/eqvio-wasm.cmake.log; exit 1; }
fi
nice -n 10 cmake --build /root/build/eqvio-wasm -j$J --target eqvio_run eqvio_web_js 2>&1 | { grep -E "error|Error" || true; } | head -30
mkdir -p /mnt/h/Research/CustomSLAM/web/build
cp /root/build/eqvio-wasm/eqvio.js /root/build/eqvio-wasm/eqvio.wasm /root/build/eqvio-wasm/eqvio.worker.mjs /mnt/h/Research/CustomSLAM/web/build/ 2>/dev/null
ls -la --time-style=+%T /root/build/eqvio-native/eqvio_run /root/build/eqvio-wasm/eqvio_run.wasm /root/build/eqvio-wasm/eqvio.wasm 2>&1
echo "=== done $(date +%T)"
