#!/usr/bin/env bash
# 5 Nov: put the frozen paper build back into web/build (the phone's page module). build_g40_1104.sh copied the
# a30b538 module there on 4 Nov 21:19, while claude/PHONE.md ("Paper session") says web/build = snap-84c1f17, and the
# 3 Nov module was not kept anywhere (web/build is gitignored; snap-<tag> holds the runners only). Builds the SLAM code
# of 84c1f17 from a clean export (git archive, so the working tree and slam-wasm-ui stay as they are) with
# slam-wasm-ui's cache options, both targets. Check: the runner built here against snap-84c1f17's runner (sha256;
# source paths differ, so a byte difference alone does not mean other code). The a30b538 module moves to
# web/build-g40/ (the AR page loads web/build/ only). Nothing else running: -j6.
set -uo pipefail
LOG=/mnt/h/Research/logs/build_paper_web_1105.log
S=/root/src/paper-84c1f17 B=/root/build/slam-wasm-paper W=/mnt/h/Research/CustomSLAM/web
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
{
  echo "=== start $(date +%T)"
  rm -rf $S && mkdir -p $S && git -C /mnt/h/Research/CustomSLAM archive 84c1f17 | tar -x -C $S
  cmake -S $S/slam -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=/opt/slam/emsdk/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake \
    -DBUILD_WASM=ON -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3 -DBOOST_HEADERS_DIR=/opt/slam/boost-headers \
    -DOpenCV_DIR=/opt/slam/opencv-4.10.0-wasm-mt/lib/cmake/opencv4 -DWEB_CHECKS=OFF -DWEB_PROFILE=OFF -DWEB_SIMD=ON \
    2>&1 | tail -3
  nice -n 10 cmake --build $B -j6 --target orbslam3_run orbslam3_web_js 2>&1 | grep -E "error|FAILED|Linking"
  ls -la --time-style=full-iso $B/orbslam3.wasm $B/orbslam3_run.wasm
  sha256sum $B/orbslam3_run.wasm /root/build/snap-84c1f17/wasm/orbslam3_run.wasm
  if [ -s $B/orbslam3.wasm ] && [ $B/orbslam3.wasm -nt $S/slam ]; then
    mkdir -p $W/build-g40 && cp $W/build/orbslam3.js $W/build/orbslam3.wasm $W/build/orbslam3.worker.mjs $W/build-g40/
    cp $B/orbslam3.js $B/orbslam3.wasm $B/orbslam3.worker.mjs $W/build/
    md5sum $W/build/orbslam3.wasm $W/build-g40/orbslam3.wasm $B/orbslam3.wasm
  else
    echo "BUILD FAILED: web/build left as it was"
  fi
  echo "=== done $(date +%T)"
} >> $LOG 2>&1
