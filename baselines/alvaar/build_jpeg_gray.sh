#!/usr/bin/env bash
# Builds baselines/alvaar/jpeg_gray.mjs (single file, ES module, Node) from jpeg_gray.cc with the WebAssembly OpenCV
# 4.10.0 build WebVIO links (/opt/slam/opencv-4.10.0-wasm-mt; -pthread to match it, no thread is started).
set -euo pipefail
H="$(cd "$(dirname "$0")" && pwd)"; CV=/opt/slam/opencv-4.10.0-wasm-mt
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
em++ -O2 -std=c++17 -pthread -I$CV/include/opencv4 "$H/jpeg_gray.cc" -o "$H/jpeg_gray.mjs" \
  $CV/lib/libopencv_imgcodecs.a $CV/lib/libopencv_imgproc.a $CV/lib/libopencv_core.a \
  $CV/lib/opencv4/3rdparty/liblibjpeg-turbo.a $CV/lib/opencv4/3rdparty/liblibpng.a $CV/lib/opencv4/3rdparty/libzlib.a \
  -sMODULARIZE -sEXPORT_ES6 -sSINGLE_FILE -sENVIRONMENT=node,worker -sALLOW_MEMORY_GROWTH \
  -sEXPORTED_FUNCTIONS=_decode_gray,_malloc,_free -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAP32
ls -la "$H/jpeg_gray.mjs"
