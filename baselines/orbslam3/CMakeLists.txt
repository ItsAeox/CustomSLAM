cmake_minimum_required(VERSION 3.16)
project(orbslam3_baseline CXX)

# ORB-SLAM3 does not install a CMake package: point ORB_SLAM3_ROOT at the (built) source tree.
#   cmake -S baselines/orbslam3 -B build/orbslam3-run -DORB_SLAM3_ROOT=/mnt/h/Research/third_party/ORB_SLAM3
set(ORB_SLAM3_ROOT "" CACHE PATH "ORB-SLAM3 source tree (with lib/libORB_SLAM3.so built)")
if (NOT ORB_SLAM3_ROOT)
  message(FATAL_ERROR "Set -DORB_SLAM3_ROOT=<path to ORB_SLAM3>")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_definitions(-DCOMPILEDWITHC11)

find_package(OpenCV 4.4 REQUIRED)   # must match the OpenCV ORB-SLAM3 was built with (system 4.5.4)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

add_executable(orbslam3_run orbslam3_run.cc)
target_include_directories(orbslam3_run PRIVATE
  ${ORB_SLAM3_ROOT} ${ORB_SLAM3_ROOT}/include ${ORB_SLAM3_ROOT}/include/CameraModels
  ${ORB_SLAM3_ROOT}/Thirdparty/Sophus ${EIGEN3_INCLUDE_DIR} ${Pangolin_INCLUDE_DIRS})
target_link_libraries(orbslam3_run PRIVATE
  ${ORB_SLAM3_ROOT}/lib/libORB_SLAM3.so
  ${ORB_SLAM3_ROOT}/Thirdparty/DBoW2/lib/libDBoW2.so
  ${ORB_SLAM3_ROOT}/Thirdparty/g2o/lib/libg2o.so
  ${OpenCV_LIBS} ${Pangolin_LIBRARIES} -lboost_serialization -lcrypto)
set_target_properties(orbslam3_run PROPERTIES BUILD_RPATH
  "${ORB_SLAM3_ROOT}/lib;${ORB_SLAM3_ROOT}/Thirdparty/DBoW2/lib;${ORB_SLAM3_ROOT}/Thirdparty/g2o/lib")
