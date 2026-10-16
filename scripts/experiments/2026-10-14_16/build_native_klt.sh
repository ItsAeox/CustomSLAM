cmake --build /root/build/slam-native -j2 --target orbslam3_run 2>&1 | grep -E "error|FAILED|warning: unused" | head -20
ls -la /root/build/slam-native/orbslam3_run
