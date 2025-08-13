cmake_minimum_required(VERSION 3.20)
project(webar_vio CXX)

option(BUILD_WASM "Build for WebAssembly" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

# -------- OpenCV per-target --------
if (BUILD_WASM)
  message(STATUS "Configuring for WASM; minimal OpenCV modules")
  # If you built OpenCV for WASM in a custom prefix, allow passing OpenCV_DIR from cmd line
  # emcmake cmake ... -DOpenCV_DIR=/abs/path/to/build_opencv_wasm/install/lib/cmake/opencv4
  find_package(OpenCV REQUIRED COMPONENTS core imgproc features2d calib3d)
else()
  message(STATUS "Configuring for native; add videoio/highgui")
  find_package(OpenCV REQUIRED COMPONENTS core imgproc features2d calib3d videoio highgui)
endif()

# -------- Library --------
add_library(vio
  src/common/camera.cc
  src/frontend/feature_tracker.cc
  src/system.cc
)

# Headers
target_include_directories(vio PUBLIC
  src
  third_party/eigen
  third_party/sophus    # if your checkout is Sophus/ with include inside it
  # if headers are actually in sophus/include, use: third_party/sophus/include
)

target_link_libraries(vio PUBLIC ${OpenCV_LIBS})

# -------- Executables --------
if (BUILD_WASM)
  add_executable(vio_wasm src/wasm/bindings.cpp)
  target_link_libraries(vio_wasm PRIVATE vio)

  # Embind + runtime flags (MODULARIZE so you can `import createVioModule`)
  # You can use target_link_options (newer CMake) instead of LINK_FLAGS if you prefer.
  set_target_properties(vio_wasm PROPERTIES
    LINK_FLAGS
      "--bind \
      -s WASM=1 \
      -s ALLOW_MEMORY_GROWTH=1 \
      -s FILESYSTEM=1 \
      -s MODULARIZE=1 \
      -s EXPORT_ES6=1 \
      -s EXPORT_NAME=createVioModule \
      -s EXPORTED_FUNCTIONS=['_malloc','_free'] \
      -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap','UTF8ToString','lengthBytesUTF8'] \
      -s ENVIRONMENT=web"
  )

  # Make sure headers are visible even if propagation ever changes
  target_include_directories(vio_wasm PRIVATE src)

  # One level above webar-vio (i.e., .../CustomSLAM)
  get_filename_component(ROOT_DIR "${CMAKE_SOURCE_DIR}" DIRECTORY)

  # Ensure it exists (harmless if it already does)
  add_custom_command(TARGET vio_wasm POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${ROOT_DIR}"
    # Copy .wasm sidecar
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "$<TARGET_FILE_DIR:vio_wasm>/$<TARGET_FILE_BASE_NAME:vio_wasm>.wasm"
      "${ROOT_DIR}/$<TARGET_FILE_BASE_NAME:vio_wasm>.wasm"
    # Copy the main target file (usually .js for Emscripten)
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "$<TARGET_FILE:vio_wasm>"
      "${ROOT_DIR}/$<TARGET_FILE_NAME:vio_wasm>"
  )
else()
  add_executable(vio_app src/main_native.cpp)
  target_link_libraries(vio_app PRIVATE vio)
endif()

