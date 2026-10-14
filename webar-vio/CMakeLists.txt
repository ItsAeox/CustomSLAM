cmake_minimum_required(VERSION 3.16)
project(webar_vio CXX)

# Build modes:
#   Native (headless dataset runner, used for accuracy evaluation):
#     cmake -S webar-vio -B build/native -DOpenCV_DIR=/opt/slam/opencv-4.10.0-native/lib/cmake/opencv4
#   WebAssembly (browser module, used for the web demo and timing):
#     emcmake cmake -S webar-vio -B build/wasm -DBUILD_WASM=ON -DOpenCV_DIR=/opt/slam/opencv-4.10.0-wasm/lib/cmake/opencv4
# Both builds must use the same OpenCV version (see scripts/setup_env.sh).
option(BUILD_WASM "Build for WebAssembly" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The only third-party dependency of the core is OpenCV.
set(VIO_OPENCV_MODULES core imgproc video features2d calib3d flann)
if (BUILD_WASM)
  # emscripten's toolchain restricts package search to its sysroot; allow the explicit OpenCV_DIR.
  if (OpenCV_DIR)
    list(APPEND CMAKE_FIND_ROOT_PATH "${OpenCV_DIR}/../../..")
  endif()
  find_package(OpenCV REQUIRED COMPONENTS ${VIO_OPENCV_MODULES})
else()
  find_package(OpenCV REQUIRED COMPONENTS ${VIO_OPENCV_MODULES} imgcodecs)
endif()
message(STATUS "OpenCV ${OpenCV_VERSION} from ${OpenCV_DIR}")

add_library(vio STATIC
  src/system.cc
  src/imu_preint.cc
  src/vio_backend.cc
  src/vio_init.cc
  src/kitti_calib.cc
  src/vio_initializer.cc
)
target_include_directories(vio PUBLIC src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vio PUBLIC ${OpenCV_LIBS})

if (BUILD_WASM)
  target_compile_options(vio PRIVATE -msimd128)

  add_executable(vio_wasm src/wasm/bindings.cpp)
  target_link_libraries(vio_wasm PRIVATE vio)
  target_link_options(vio_wasm PRIVATE
    -O3
    -flto
    -sWASM=1
    -sALLOW_MEMORY_GROWTH=1
    -sINITIAL_MEMORY=268435456
    -sFILESYSTEM=0
    -sMODULARIZE=1
    -sEXPORT_ES6=1
    -sEXPORT_NAME=createVioModule
    "-sEXPORTED_FUNCTIONS=['_malloc','_free']"
    "-sEXPORTED_RUNTIME_METHODS=['ccall','cwrap','HEAPU8','HEAP8','UTF8ToString','lengthBytesUTF8']"
    -sENVIRONMENT=web
    -sWASM_BIGINT=1
    --bind
    -msimd128
  )

  # Copy vio_wasm.js / vio_wasm.wasm next to index.html (repo root)
  get_filename_component(ROOT_DIR "${CMAKE_SOURCE_DIR}" DIRECTORY)
  add_custom_command(TARGET vio_wasm POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "$<TARGET_FILE_DIR:vio_wasm>/vio_wasm.wasm" "${ROOT_DIR}/vio_wasm.wasm"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "$<TARGET_FILE:vio_wasm>" "${ROOT_DIR}/vio_wasm.js"
  )
else()
  add_executable(vio_run tools/vio_run.cpp)
  target_link_libraries(vio_run PRIVATE vio)
endif()
