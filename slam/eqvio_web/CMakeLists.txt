cmake_minimum_required(VERSION 3.16)
project(eqvio_web CXX)

# WebVIO's EqVIO mode (claude/PLAN.md 4d): EqVIO (pvangoor/eqvio) with its GIFT feature tracker and yaml-cpp, compiled from
# their source trees (EQVIO_SRC, YAML_SRC: the H:\Research\ext clones, commits in baselines/eqvio/build_eqvio.sh and
# baselines/msceqf/build_msceqf.sh) around EqvioWeb.cc. Separate from the ORB-SLAM3 build (slam/CMakeLists.txt), same
# OpenCV builds:
#   native (parity reference):
#     cmake -S slam/eqvio_web -B build/eqvio-native -DOpenCV_DIR=/opt/slam/opencv-4.10.0-native/lib/cmake/opencv4
#   WebAssembly (eqvio_run under Node + the browser module eqvio.js):
#     emcmake cmake -S slam/eqvio_web -B build/eqvio-wasm -DBUILD_WASM=ON -DOpenCV_DIR=/opt/slam/opencv-4.10.0-wasm-mt/lib/cmake/opencv4
option(BUILD_WASM "Build for WebAssembly" OFF)
set(EQVIO_SRC "/mnt/h/Research/ext/eqvio" CACHE PATH "EqVIO source tree")
set(YAML_SRC "/mnt/h/Research/ext/yaml-cpp" CACHE PATH "yaml-cpp source tree")
set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3" CACHE PATH "Eigen 3.4 headers")

set(CMAKE_CXX_STANDARD 20)  # EqVIO uses C++20 concepts
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

if (BUILD_WASM)
  # as slam/CMakeLists.txt: SIMD128 and threads (the prebuilt OpenCV uses shared memory); C++ exceptions abort
  add_compile_options(-pthread -msimd128)
  add_link_options(-pthread -msimd128)
  if (OpenCV_DIR)
    list(APPEND CMAKE_FIND_ROOT_PATH "${OpenCV_DIR}/../../..")
  endif()
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc video features2d calib3d imgcodecs)
message(STATUS "OpenCV ${OpenCV_VERSION} from ${OpenCV_DIR}")

set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(${YAML_SRC} yaml-cpp EXCLUDE_FROM_ALL)

# ---- GIFT (eqvio external/GIFT, double precision as eqvio_opt; without its visualisation)
set(G ${EQVIO_SRC}/external/GIFT/GIFT)
add_library(gift_web STATIC
  ${G}/src/camera/camera.cpp ${G}/src/camera/DoubleSphereCamera.cpp ${G}/src/camera/EquidistantCamera.cpp
  ${G}/src/camera/PinholeCamera.cpp ${G}/src/camera/StandardCamera.cpp ${G}/src/PointFeatureTracker.cpp
  ${G}/src/EgoMotion.cpp ${G}/src/Feature.cpp ${G}/src/ImagePyramid.cpp ${G}/src/GIFeatureTracker.cpp ${G}/src/RANSAC.cpp)
# GIFT includes <eigen3/Eigen/Eigen>: a directory holding only an 'eigen3' link (not /usr/include, which the WebAssembly
# build must not see)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/eigen_prefix)
file(CREATE_LINK ${EIGEN3_INCLUDE_DIR} ${CMAKE_BINARY_DIR}/eigen_prefix/eigen3 SYMBOLIC)
target_include_directories(gift_web PUBLIC ${G}/include ${EIGEN3_INCLUDE_DIR} ${CMAKE_BINARY_DIR}/eigen_prefix ${OpenCV_INCLUDE_DIRS})
target_include_directories(gift_web PRIVATE compat)  # opencv2/highgui/highgui.hpp (included, unused)
target_link_libraries(gift_web PUBLIC ${OpenCV_LIBS} yaml-cpp::yaml-cpp)

# ---- EqVIO filter (no data servers, writer, simulator or visualiser) + the frame-by-frame driver
set(E ${EQVIO_SRC}/src)
add_library(eqvio_web STATIC
  ${E}/VIOFilter.cpp ${E}/mathematical/VIOState.cpp ${E}/mathematical/VIOGroup.cpp ${E}/mathematical/VIO_eqf.cpp
  ${E}/mathematical/VisionMeasurement.cpp ${E}/mathematical/IMUVelocity.cpp ${E}/mathematical/EqFMatrices.cpp
  ${E}/mathematical/Geometry.cpp ${E}/mathematical/coordinateSuite/euclid.cpp ${E}/mathematical/coordinateSuite/invdepth.cpp
  ${E}/mathematical/coordinateSuite/normal.cpp ${E}/LoopTimer.cpp EqvioWeb.cc)
target_include_directories(eqvio_web PUBLIC ${EQVIO_SRC}/include ${EQVIO_SRC}/external/LiePP/include ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(eqvio_web PUBLIC EQVIO_SUPPORT_CONCEPTS=1 EQVIO_BUILD_VISUALISATION=0 EQVIO_BUILD_ROSBAG=0)
target_link_libraries(eqvio_web PUBLIC gift_web)

# ---- replay: native, or Node (NODERAWFS: the dataset paths as on the host)
add_executable(eqvio_run eqvio_run.cc)
target_link_libraries(eqvio_run PRIVATE eqvio_web)
if (BUILD_WASM)
  target_link_options(eqvio_run PRIVATE -sENVIRONMENT=node,worker -sNODERAWFS=1 -sPROXY_TO_PTHREAD=1 -sALLOW_MEMORY_GROWTH=1
    -sMAXIMUM_MEMORY=2GB -sSTACK_SIZE=4MB -sEXIT_RUNTIME=1)

  # browser module (web/eqvio.worker.js)
  add_executable(eqvio_web_js bindings.cc)
  target_link_libraries(eqvio_web_js PRIVATE eqvio_web)
  set_target_properties(eqvio_web_js PROPERTIES OUTPUT_NAME "eqvio")
  target_link_options(eqvio_web_js PRIVATE
    --bind -sENVIRONMENT=web,worker -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createEqvio
    -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=64MB -sMAXIMUM_MEMORY=1GB -sSTACK_SIZE=4MB
    "-sEXPORTED_RUNTIME_METHODS=['HEAPU8']")
endif()
