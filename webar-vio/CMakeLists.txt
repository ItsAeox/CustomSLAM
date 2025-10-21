cmake_minimum_required(VERSION 3.20)
project(webar_vio CXX)

option(BUILD_WASM "Build for WebAssembly" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ==========================================================
# HARD-CODED WASM PREFIX (change here if you move the repo)
# ==========================================================
# This is your confirmed path:
# /home/aeox/Documents/FinalYearResearch/CustomSLAM/webar-vio/third_party/wasm-install
set(WASM_PREFIX "/home/aeox/Documents/FinalYearResearch/CustomSLAM/webar-vio/third_party/wasm-install")

# Make CMake search here for CONFIG packages
list(APPEND CMAKE_PREFIX_PATH
  "${WASM_PREFIX}/absl"
  "${WASM_PREFIX}/ceres"
  "${WASM_PREFIX}/eigen"
  "${WASM_PREFIX}/opengv"
)

# Explicit hints for CONFIG packages
set(absl_DIR   "${WASM_PREFIX}/absl/lib/cmake/absl")
set(Ceres_DIR  "${WASM_PREFIX}/ceres/lib/cmake/Ceres")
set(Eigen3_DIR "${WASM_PREFIX}/eigen/share/eigen3/cmake")

# If you also want to hardcode OpenCV for WASM:
# (comment this if you prefer passing OpenCV_DIR manually)
if (BUILD_WASM)
  set(OpenCV_DIR "/home/aeox/Documents/FinalYearResearch/CustomSLAM/webar-vio/build_opencv_wasm/install/lib/cmake/opencv4")
endif()

# ---------------------------
# OpenCV (minimal for WASM)
# ---------------------------
if (BUILD_WASM)
  message(STATUS "Configuring for WASM; minimal OpenCV modules")
  find_package(OpenCV REQUIRED COMPONENTS core imgproc features2d calib3d)
else()
  message(STATUS "Configuring for native; add videoio/highgui")
  find_package(OpenCV REQUIRED COMPONENTS core imgproc features2d calib3d videoio highgui)
endif()

# ---------------------------
# Eigen / Sophus
# ---------------------------
# Use Eigen as a CONFIG package so Ceres finds it too.
# (Your Eigen install dropped a proper Eigen3Config.cmake in WASM_PREFIX)
find_package(Eigen3 3.4 CONFIG REQUIRED)  # gives target Eigen3::Eigen

# Keep Sophus header path if you use those headers in your codebase.
set(SOPHUS_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/sophus")

# ---------------------------
# Ceres (CONFIG package)
# ---------------------------
find_package(Ceres CONFIG REQUIRED)  # will use Ceres_DIR above, and see Eigen/absl

# ---------------------------
# OpenGV (manual find, robust for Emscripten)
# ---------------------------
set(_opengv_root "${WASM_PREFIX}/opengv")

# Headers (expect: ${_opengv_root}/include/opengv/absolute_pose/methods.hpp)
find_path(OPENGV_INCLUDE_DIR
  NAMES opengv/absolute_pose/methods.hpp
  PATHS "${_opengv_root}/include"
  NO_DEFAULT_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

# Library (expect: ${_opengv_root}/lib/libopengv.a)
find_library(OPENGV_LIBRARY
  NAMES opengv
  PATHS "${_opengv_root}/lib"
  NO_DEFAULT_PATH
  NO_CMAKE_FIND_ROOT_PATH
)

# Optional debug to verify what CMake sees
message(STATUS "OpenGV include hint: ${_opengv_root}/include (exists: $<BOOL:EXISTS;${_opengv_root}/include>)")
message(STATUS "OpenGV lib hint:     ${_opengv_root}/lib (exists: $<BOOL:EXISTS;${_opengv_root}/lib>)")
message(STATUS "OpenGV_INCLUDE_DIR: ${OPENGV_INCLUDE_DIR}")
message(STATUS "OpenGV_LIBRARY:     ${OPENGV_LIBRARY}")

# Fallbacks (let CMake search defaults too, just in case)
if (NOT OPENGV_INCLUDE_DIR)
  find_path(OPENGV_INCLUDE_DIR opengv/absolute_pose/methods.hpp)
endif()
if (NOT OPENGV_LIBRARY)
  find_library(OPENGV_LIBRARY opengv)
endif()

if (NOT OPENGV_INCLUDE_DIR OR NOT OPENGV_LIBRARY)
  message(FATAL_ERROR
    "OpenGV not found.\n"
    "Tried include: ${_opengv_root}/include\n"
    "Tried lib:     ${_opengv_root}/lib\n"
    "If OpenGV is elsewhere, update WASM_PREFIX."
  )
endif()

# ---------------------------
# Library (your core)
# ---------------------------
add_library(vio
  src/common/camera.cc
  src/frontend/feature_tracker.cc
  src/system.cc
  src/slam/imu_initializer.cc
)

target_include_directories(vio PUBLIC
  src
  ${SOPHUS_INCLUDE_DIR}
  ${OPENGV_INCLUDE_DIR}
)

target_link_libraries(vio PUBLIC
  ${OpenCV_LIBS}
  Eigen3::Eigen
  Ceres::ceres
  ${OPENGV_LIBRARY}
)

# Emscripten compile niceties (compile-time flags only)
if (BUILD_WASM AND EMSCRIPTEN)
  target_compile_options(vio PRIVATE -msimd128)
endif()

# ---------------------------
# Executables
# ---------------------------
if (BUILD_WASM)
  add_executable(vio_wasm src/wasm/bindings.cpp)
  target_link_libraries(vio_wasm PRIVATE vio)
  target_include_directories(vio_wasm PRIVATE src)

  if (EMSCRIPTEN)
    target_link_options(vio_wasm PRIVATE
      -sWASM=1
      -sALLOW_MEMORY_GROWTH=1
      -sFILESYSTEM=1
      -sMODULARIZE=1
      -sEXPORT_ES6=1
      -sEXPORT_NAME=createVioModule
      -sEXPORTED_FUNCTIONS=['_malloc','_free']
      -sEXPORTED_RUNTIME_METHODS=['ccall','cwrap','UTF8ToString','lengthBytesUTF8']
      -sENVIRONMENT=web
      --bind
      -msimd128
    )
  endif()

  # Copy outputs one level above webar-vio
  get_filename_component(ROOT_DIR "${CMAKE_SOURCE_DIR}" DIRECTORY)
  add_custom_command(TARGET vio_wasm POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${ROOT_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "$<TARGET_FILE_DIR:vio_wasm>/$<TARGET_FILE_BASE_NAME:vio_wasm>.wasm"
      "${ROOT_DIR}/$<TARGET_FILE_BASE_NAME:vio_wasm>.wasm"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "$<TARGET_FILE:vio_wasm>"
      "${ROOT_DIR}/$<TARGET_FILE_NAME:vio_wasm>"
  )
else()
  add_executable(vio_app src/main_native.cpp)
  target_link_libraries(vio_app PRIVATE vio)
endif()
