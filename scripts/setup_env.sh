#!/usr/bin/env bash
# Reproducible environment setup for CustomSLAM research (Ubuntu 22.04, native or WSL2).
#
# Usage:  bash scripts/setup_env.sh [step ...]
#   steps: apt pangolin orbslam3 python emsdk opencv_native opencv_wasm   (default: all, in that order)
#
# Layout (override with env vars):
#   RESEARCH_DIR  /mnt/h/Research          sources of third-party repos live in $RESEARCH_DIR/third_party
#   BUILD_ROOT    $HOME/build              out-of-source build trees (keep on a native Linux FS for speed)
#   OPT_ROOT      /opt/slam                install prefixes (OpenCV native/wasm, emsdk)
#   JOBS          6                        parallel jobs (ORB-SLAM3 OOMs with high -j on ~8 GB RAM)
#
# Pinned versions: Pangolin v0.8, ORB-SLAM3 4452a3c (v1.0), OpenCV 4.10.0, emsdk 3.1.64.
set -euo pipefail

RESEARCH_DIR="${RESEARCH_DIR:-/mnt/h/Research}"
TP="$RESEARCH_DIR/third_party"
BUILD_ROOT="${BUILD_ROOT:-$HOME/build}"
OPT_ROOT="${OPT_ROOT:-/opt/slam}"
JOBS="${JOBS:-6}"

PANGOLIN_TAG="v0.8"
ORBSLAM3_COMMIT="4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4"
OPENCV_TAG="4.10.0"
EMSDK_VERSION="3.1.64"

OPENCV_NATIVE_PREFIX="$OPT_ROOT/opencv-$OPENCV_TAG-native"
OPENCV_WASM_PREFIX="$OPT_ROOT/opencv-$OPENCV_TAG-wasm"
EMSDK_DIR="$OPT_ROOT/emsdk"
VENV_DIR="$OPT_ROOT/venv"

log() { printf '\n=== [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"

clone_pinned() {  # url dir ref
  local url="$1" dir="$2" ref="$3"
  if [ ! -d "$dir/.git" ]; then
    git -c core.autocrlf=false -c advice.detachedHead=false clone --quiet "$url" "$dir"
  fi
  git -C "$dir" -c advice.detachedHead=false checkout --quiet "$ref"
}

step_apt() {
  log "apt packages"
  export DEBIAN_FRONTEND=noninteractive
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq \
    build-essential cmake git pkg-config wget curl unzip xz-utils ninja-build \
    python3 python3-pip python3-venv python3-dev \
    libeigen3-dev libopencv-dev libboost-serialization-dev libboost-filesystem-dev libssl-dev \
    libglew-dev libgl1-mesa-dev libepoxy-dev libwayland-dev libxkbcommon-dev wayland-protocols \
    libjpeg-dev libpng-dev libtiff-dev
}

step_pangolin() {
  log "Pangolin $PANGOLIN_TAG"
  clone_pinned https://github.com/stevenlovegrove/Pangolin.git "$TP/Pangolin" "$PANGOLIN_TAG"
  cmake -S "$TP/Pangolin" -B "$BUILD_ROOT/pangolin" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_TOOLS=OFF -DBUILD_PANGOLIN_PYTHON=OFF
  cmake --build "$BUILD_ROOT/pangolin" -j "$JOBS"
  $SUDO cmake --install "$BUILD_ROOT/pangolin"
  $SUDO ldconfig
}

step_orbslam3() {
  log "ORB-SLAM3 $ORBSLAM3_COMMIT"
  local src="$TP/ORB_SLAM3"
  clone_pinned https://github.com/UZ-SLAMLab/ORB_SLAM3.git "$src" "$ORBSLAM3_COMMIT"

  # Ubuntu 22.04 / Pangolin 0.8 compatibility: Pangolin headers need C++17.
  # Keep -DCOMPILEDWITHC11 (ORB-SLAM3 uses it to pick std::chrono::steady_clock).
  sed -i 's/-std=c++11")/-std=c++17")/' "$src/CMakeLists.txt"
  # C++17 forbids ++ on bool. Under C++11, `mnFullBAIdx++` (a bool) simply sets it to true, so keep exactly
  # that behaviour (the baseline must behave like the official C++11 build).
  sed -i 's/mnFullBAIdx++;/mnFullBAIdx = true;  \/\/ was mnFullBAIdx++ (bool); identical C++11 semantics/' \
      "$src/src/LoopClosing.cc"

  # Thirdparty libs (DBoW2, g2o). Their CMake writes libs into $src/Thirdparty/*/lib.
  # Sophus is header-only for ORB-SLAM3; its own build is tests only (and fails with -Werror on GCC 11) -> skipped.
  for lib in DBoW2 g2o; do
    cmake -S "$src/Thirdparty/$lib" -B "$BUILD_ROOT/orbslam3-$lib" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD_ROOT/orbslam3-$lib" -j "$JOBS"
  done

  if [ ! -f "$src/Vocabulary/ORBvoc.txt" ]; then
    tar -xf "$src/Vocabulary/ORBvoc.txt.tar.gz" -C "$src/Vocabulary"
  fi

  cmake -S "$src" -B "$BUILD_ROOT/orbslam3" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$BUILD_ROOT/orbslam3" -j "$JOBS"
}

step_python() {
  log "Python venv (evo, numpy, matplotlib, scipy, pandas, pyyaml) at $VENV_DIR"
  $SUDO mkdir -p "$OPT_ROOT" && $SUDO chown "$(id -u):$(id -g)" "$OPT_ROOT"
  python3 -m venv "$VENV_DIR"
  local PIP=("$VENV_DIR/bin/pip" --timeout 60 --retries 10)
  "${PIP[@]}" install --progress-bar off --upgrade pip
  "${PIP[@]}" install --progress-bar off evo numpy scipy matplotlib pandas pyyaml
}

step_emsdk() {
  log "emsdk $EMSDK_VERSION"
  $SUDO mkdir -p "$OPT_ROOT" && $SUDO chown "$(id -u):$(id -g)" "$OPT_ROOT"
  [ -d "$EMSDK_DIR/.git" ] || git clone --quiet https://github.com/emscripten-core/emsdk.git "$EMSDK_DIR"
  "$EMSDK_DIR/emsdk" install "$EMSDK_VERSION"
  "$EMSDK_DIR/emsdk" activate "$EMSDK_VERSION"
}

# Modules needed by webar-vio: core imgproc video features2d calib3d flann (+ imgcodecs natively, for the runner).
OPENCV_COMMON_FLAGS=(
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF -DBUILD_opencv_apps=OFF
  -DBUILD_JAVA=OFF -DBUILD_opencv_python3=OFF -DBUILD_opencv_python2=OFF
  -DWITH_IPP=OFF -DWITH_TBB=OFF -DWITH_OPENMP=OFF -DWITH_OPENCL=OFF -DWITH_LAPACK=OFF -DWITH_EIGEN=OFF
  -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_GTK=OFF -DWITH_QT=OFF -DWITH_V4L=OFF -DWITH_1394=OFF
  -DWITH_OPENEXR=OFF -DWITH_WEBP=OFF -DWITH_JASPER=OFF -DWITH_OPENJPEG=OFF -DWITH_TIFF=OFF
  -DWITH_ITT=OFF -DWITH_ADE=OFF -DWITH_QUIRC=OFF -DWITH_PROTOBUF=OFF -DWITH_VTK=OFF -DCV_TRACE=OFF
  -DBUILD_ZLIB=ON
)

step_opencv_native() {
  log "OpenCV $OPENCV_TAG native (static) -> $OPENCV_NATIVE_PREFIX"
  clone_pinned https://github.com/opencv/opencv.git "$TP/opencv" "$OPENCV_TAG"
  $SUDO mkdir -p "$OPT_ROOT" && $SUDO chown "$(id -u):$(id -g)" "$OPT_ROOT"
  cmake -S "$TP/opencv" -B "$BUILD_ROOT/opencv-native" -G Ninja "${OPENCV_COMMON_FLAGS[@]}" \
        -DBUILD_LIST=core,imgproc,video,features2d,calib3d,flann,imgcodecs \
        -DWITH_JPEG=ON -DBUILD_JPEG=ON -DWITH_PNG=ON -DBUILD_PNG=ON \
        -DCMAKE_INSTALL_PREFIX="$OPENCV_NATIVE_PREFIX"
  cmake --build "$BUILD_ROOT/opencv-native" -j "$JOBS"
  cmake --install "$BUILD_ROOT/opencv-native"
}

step_opencv_wasm() {
  log "OpenCV $OPENCV_TAG wasm (static, SIMD128) -> $OPENCV_WASM_PREFIX"
  clone_pinned https://github.com/opencv/opencv.git "$TP/opencv" "$OPENCV_TAG"
  # shellcheck disable=SC1091
  source "$EMSDK_DIR/emsdk_env.sh" >/dev/null
  emcmake cmake -S "$TP/opencv" -B "$BUILD_ROOT/opencv-wasm" -G Ninja "${OPENCV_COMMON_FLAGS[@]}" \
        -DBUILD_LIST=core,imgproc,video,features2d,calib3d,flann \
        -DWITH_JPEG=OFF -DWITH_PNG=OFF -DBUILD_opencv_imgcodecs=OFF -DBUILD_opencv_highgui=OFF -DBUILD_opencv_videoio=OFF \
        -DCPU_BASELINE='' -DCPU_DISPATCH='' -DCV_ENABLE_INTRINSICS=ON -DWITH_PTHREADS_PF=OFF -DENABLE_PIC=FALSE \
        -DCMAKE_C_FLAGS="-msimd128" -DCMAKE_CXX_FLAGS="-msimd128" \
        -DCMAKE_INSTALL_PREFIX="$OPENCV_WASM_PREFIX"
  cmake --build "$BUILD_ROOT/opencv-wasm" -j "$JOBS"
  cmake --install "$BUILD_ROOT/opencv-wasm"
}

ALL_STEPS=(apt pangolin orbslam3 python emsdk opencv_native opencv_wasm)
STEPS=("$@"); [ ${#STEPS[@]} -eq 0 ] && STEPS=("${ALL_STEPS[@]}")
mkdir -p "$BUILD_ROOT"
for s in "${STEPS[@]}"; do "step_$s"; done
log "done: ${STEPS[*]}"
