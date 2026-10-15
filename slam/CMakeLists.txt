cmake_minimum_required(VERSION 3.16)
project(orbslam3_web C CXX)

# ORB-SLAM3 (vendored, slam/orbslam3) built for the web, from ONE source tree in two flavours:
#   native (parity reference):
#     cmake -S slam -B build/slam-native -DOpenCV_DIR=/opt/slam/opencv-4.10.0-native/lib/cmake/opencv4
#   WebAssembly, Node harness (pthreads, NODERAWFS):
#     emcmake cmake -S slam -B build/slam-wasm -DBUILD_WASM=ON -DOpenCV_DIR=/opt/slam/opencv-4.10.0-wasm-mt/lib/cmake/opencv4
# Both define ORBSLAM3_WEB (viewer stubs, no Atlas save/load) so native and wasm compile identical code.
option(BUILD_WASM "Build for WebAssembly" OFF)
set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3" CACHE PATH "Eigen 3.4 headers")
set(BOOST_HEADERS_DIR "/opt/slam/boost-headers" CACHE PATH "Directory containing boost/ (headers only)")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_C_FLAGS_RELEASE "-O3")

set(ORB ${CMAKE_CURRENT_SOURCE_DIR}/orbslam3)
set(PORT ${CMAKE_CURRENT_SOURCE_DIR}/port)

if (BUILD_WASM)
  # Same flags as the OpenCV build: SIMD128, threads (shared memory). C++ exceptions abort (emscripten default);
  # ORB-SLAM3 only throws on fatal configuration errors. WEB_SIMD=OFF builds our code (ORB-SLAM3, g2o, DBoW2)
  # without -msimd128 auto-vectorisation (diagnostic); the prebuilt OpenCV keeps its SIMD.
  option(WEB_SIMD "Compile the port with -msimd128" ON)
  # Diagnostic: emscripten runtime assertions + stack-overflow checks (wasm32 does not trap stack overflows)
  option(WEB_CHECKS "Link with -sASSERTIONS=2 -sSTACK_OVERFLOW_CHECK=2" OFF)
  if (WEB_CHECKS)
    add_link_options(-sASSERTIONS=2 -sSTACK_OVERFLOW_CHECK=2)
  endif()
  set(WEB_FLAGS -pthread)
  if (WEB_SIMD)
    list(APPEND WEB_FLAGS -msimd128)
  endif()
  add_compile_options(${WEB_FLAGS})
  add_link_options(${WEB_FLAGS})
  if (OpenCV_DIR)
    list(APPEND CMAKE_FIND_ROOT_PATH "${OpenCV_DIR}/../../..")
  endif()
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc video features2d calib3d imgcodecs)  # video: KLT frames
message(STATUS "OpenCV ${OpenCV_VERSION} from ${OpenCV_DIR}")
find_package(Threads REQUIRED)

# Profiling build: ORB-SLAM3's own per-stage timers (REGISTER_TIMES). At Shutdown it writes TrackingTimeStats.txt,
# LocalMapTimeStats.txt, LBA_Stats.txt and ExecMean.txt to the working directory. Keep accuracy runs un-instrumented.
option(WEB_PROFILE "Build with ORB-SLAM3 REGISTER_TIMES per-stage timers" OFF)
if (WEB_PROFILE)
  add_compile_definitions(REGISTER_TIMES)
endif()

add_compile_definitions(COMPILEDWITHC11 ORBSLAM3_WEB
  # Boost 1.74 derives hash_base from std::unary_function, removed in C++17 (libc++); Boost's own switch avoids it.
  BOOST_NO_CXX98_FUNCTION_BASE)

# ---- g2o (exact source list of upstream Thirdparty/g2o/CMakeLists.txt) ----
set(G2O ${ORB}/Thirdparty/g2o/g2o)
add_library(g2o_web STATIC
  ${G2O}/core/batch_stats.cpp ${G2O}/core/cache.cpp ${G2O}/core/estimate_propagator.cpp ${G2O}/core/factory.cpp
  ${G2O}/core/hyper_dijkstra.cpp ${G2O}/core/hyper_graph.cpp ${G2O}/core/hyper_graph_action.cpp
  ${G2O}/core/jacobian_workspace.cpp ${G2O}/core/marginal_covariance_cholesky.cpp ${G2O}/core/matrix_structure.cpp
  ${G2O}/core/optimizable_graph.cpp ${G2O}/core/optimization_algorithm.cpp ${G2O}/core/optimization_algorithm_factory.cpp
  ${G2O}/core/optimization_algorithm_gauss_newton.cpp ${G2O}/core/optimization_algorithm_levenberg.cpp
  ${G2O}/core/optimization_algorithm_with_hessian.cpp ${G2O}/core/parameter.cpp ${G2O}/core/parameter_container.cpp
  ${G2O}/core/robust_kernel.cpp ${G2O}/core/robust_kernel_factory.cpp ${G2O}/core/robust_kernel_impl.cpp
  ${G2O}/core/solver.cpp ${G2O}/core/sparse_optimizer.cpp
  ${G2O}/stuff/os_specific.c ${G2O}/stuff/property.cpp ${G2O}/stuff/string_tools.cpp ${G2O}/stuff/timeutil.cpp
  ${G2O}/types/types_sba.cpp ${G2O}/types/types_seven_dof_expmap.cpp ${G2O}/types/types_six_dof_expmap.cpp)
target_include_directories(g2o_web PUBLIC ${ORB}/Thirdparty/g2o ${EIGEN3_INCLUDE_DIR})

# ---- DBoW2 ----
set(DB ${ORB}/Thirdparty/DBoW2)
add_library(dbow2_web STATIC
  ${DB}/DBoW2/BowVector.cpp ${DB}/DBoW2/FORB.cpp ${DB}/DBoW2/FeatureVector.cpp ${DB}/DBoW2/ScoringObject.cpp
  ${DB}/DUtils/Random.cpp ${DB}/DUtils/Timestamp.cpp)
target_include_directories(dbow2_web PUBLIC ${DB} ${OpenCV_INCLUDE_DIRS})
target_include_directories(dbow2_web SYSTEM PUBLIC ${BOOST_HEADERS_DIR})
target_link_libraries(dbow2_web PUBLIC ${OpenCV_LIBS})

# ---- ORB-SLAM3 core (Viewer.cc / MapDrawer.cc replaced by port/web/viewer_stubs.cc) ----
file(GLOB ORB_SRCS ${ORB}/src/*.cc ${ORB}/src/*.cpp ${ORB}/src/CameraModels/*.cpp)
list(REMOVE_ITEM ORB_SRCS ${ORB}/src/Viewer.cc ${ORB}/src/MapDrawer.cc)
add_library(orbslam3_web STATIC ${ORB_SRCS} ${PORT}/web/viewer_stubs.cc
  ${PORT}/features/teblid/beblid.cpp)  # TEBLID/BEBLID descriptors vendored from opencv_contrib (Web.Descriptor)
target_include_directories(orbslam3_web PUBLIC
  ${ORB} ${ORB}/include ${ORB}/include/CameraModels ${ORB}/Thirdparty/Sophus
  ${PORT}/stubs ${PORT}/features/teblid ${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_include_directories(orbslam3_web SYSTEM PUBLIC ${BOOST_HEADERS_DIR})
target_link_libraries(orbslam3_web PUBLIC g2o_web dbow2_web ${OpenCV_LIBS} Threads::Threads)

# ---- dataset harness: the same driver as the native ORB-SLAM3 baseline ----
add_executable(orbslam3_run ${CMAKE_CURRENT_SOURCE_DIR}/../baselines/orbslam3/orbslam3_run.cc)
target_link_libraries(orbslam3_run PRIVATE orbslam3_web)

if (BUILD_WASM)
  # Node: main() runs in a pthread (may block), host filesystem via NODERAWFS, memory grows up to 4 GB.
  target_link_options(orbslam3_run PRIVATE
    -sENVIRONMENT=node,worker -sNODERAWFS=1 -sPROXY_TO_PTHREAD=1 -sPTHREAD_POOL_SIZE=8
    -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB -sINITIAL_MEMORY=512MB
    -sSTACK_SIZE=8MB -sDEFAULT_PTHREAD_STACK_SIZE=8MB -sEXIT_RUNTIME=1)
endif()

# ---- diagnostics ----
add_executable(orb_dump ${PORT}/tests/orb_dump.cc)
target_link_libraries(orb_dump PRIVATE orbslam3_web)
if (BUILD_WASM)
  target_link_options(orb_dump PRIVATE -sENVIRONMENT=node,worker -sNODERAWFS=1 -sPROXY_TO_PTHREAD=1
    -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB -sSTACK_SIZE=8MB -sEXIT_RUNTIME=1)
endif()

# ---- tools (native only) ----
if (NOT BUILD_WASM)
  add_executable(vocab_convert ${PORT}/tools/vocab_convert.cc)
  target_link_libraries(vocab_convert PRIVATE orbslam3_web)
  add_executable(vocab_train ${PORT}/tools/vocab_train.cc)
  target_link_libraries(vocab_train PRIVATE orbslam3_web)
endif()

# ---- browser module (WebAssembly only): Embind API, loaded inside a Web Worker ----
if (BUILD_WASM)
  add_executable(orbslam3_web_js ${PORT}/web/bindings.cc)
  target_link_libraries(orbslam3_web_js PRIVATE orbslam3_web)
  set_target_properties(orbslam3_web_js PROPERTIES OUTPUT_NAME "orbslam3")
  target_link_options(orbslam3_web_js PRIVATE
    --bind -sENVIRONMENT=web,worker -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createOrbSlam3
    -sPTHREAD_POOL_SIZE=6 -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=256MB -sMAXIMUM_MEMORY=2GB
    -sSTACK_SIZE=4MB -sDEFAULT_PTHREAD_STACK_SIZE=4MB -sFORCE_FILESYSTEM=1
    "-sEXPORTED_RUNTIME_METHODS=['FS','HEAPU8']")
endif()

add_executable(numerics_check ${PORT}/tests/numerics_check.cc)
target_link_libraries(numerics_check PRIVATE orbslam3_web)
if (BUILD_WASM)
  target_link_options(numerics_check PRIVATE -sENVIRONMENT=node,worker -sNODERAWFS=1 -sPROXY_TO_PTHREAD=1
    -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB -sSTACK_SIZE=8MB -sEXIT_RUNTIME=1)
endif()
