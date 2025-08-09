cmake_minimum_required(VERSION 3.20)
project(webar_vio CXX)

option(BUILD_WASM "Build for WebAssembly" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

# OpenCV minimal: core imgproc features2d calib3d
find_package(OpenCV REQUIRED COMPONENTS core imgproc features2d calib3d videoio highgui)

# Library with a few stub .cc files so it compiles
add_library(vio
  src/common/camera.cc
  src/frontend/feature_tracker.cc
  src/system.cc
)

# Public headers live here:
target_include_directories(vio PUBLIC
  src
  third_party/eigen
  third_party/sophus            # repo root; headers are under sophus/
)

target_link_libraries(vio PUBLIC
  ${OpenCV_LIBS}
)

if (BUILD_WASM)
  add_executable(vio_wasm src/wasm/bindings.cpp)
  target_link_libraries(vio_wasm PRIVATE vio)
  set_target_properties(vio_wasm PROPERTIES
    LINK_FLAGS "-s WASM=1 -s ALLOW_MEMORY_GROWTH=1 -s EXPORTED_FUNCTIONS='[_malloc,_free]' -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap'] -s FILESYSTEM=1"
  )
else()
  add_executable(vio_app src/main_native.cpp)
  target_link_libraries(vio_app PRIVATE vio)
endif()
