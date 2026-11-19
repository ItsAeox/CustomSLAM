source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
cmake --build /root/build/slam-native -j2 --target orbslam3_run 2>&1 | grep -E "error|FAILED" || true
cmake --build /root/build/slam-wasm -j2 2>&1 | grep -E "error|FAILED" || true
cp /root/build/slam-wasm/orbslam3.js /root/build/slam-wasm/orbslam3.wasm /root/build/slam-wasm/orbslam3.worker.mjs /mnt/h/Research/CustomSLAM/web/build/
echo "built $(date +%T)"
