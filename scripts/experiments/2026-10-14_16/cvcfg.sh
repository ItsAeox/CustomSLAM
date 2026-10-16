for f in /root/build/opencv-wasm-mt/CMakeCache.txt /root/build/opencv-wasm-mt-scalar/CMakeCache.txt /root/build/opencv-native/CMakeCache.txt; do
  echo "== $f"
  grep -E '^(CMAKE_CXX_FLAGS|CPU_BASELINE|CPU_DISPATCH|CV_ENABLE_INTRINSICS|CV_DISABLE_OPTIMIZATION|CMAKE_INSTALL_PREFIX|CMAKE_BUILD_TYPE)[:]' $f
done
grep -c "msimd128" /root/build/opencv-wasm-mt/CMakeFiles/*.txt /root/build/opencv-wasm-mt/modules/features2d/CMakeFiles/opencv_features2d.dir/flags.make 2>/dev/null | head -3
grep -m2 "CXX_FLAGS" /root/build/opencv-wasm-mt/modules/features2d/CMakeFiles/opencv_features2d.dir/flags.make 2>/dev/null | cut -c1-300
