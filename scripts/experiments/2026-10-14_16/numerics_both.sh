#!/usr/bin/env bash
# Build + run numerics_check native (GCC) and WASM; diff.
S=/mnt/h/Research/results/scratch
cd /root/build/slam-native && cmake --build . --target numerics_check 2>&1 | grep -E "error|warning: unused" | head -20
./numerics_check /mnt/h/Research/vocab/ORBvoc.bin > $S/numerics_native.txt 2>&1
source /opt/slam/emsdk/emsdk_env.sh >/dev/null 2>&1
cd /root/build/slam-wasm && cmake --build . --target numerics_check 2>&1 | grep -E "error" | head -20
cd /mnt/h/Research/vocab && node /root/build/slam-wasm/numerics_check.js ORBvoc.bin > $S/numerics_wasm.txt 2>&1
diff $S/numerics_native.txt $S/numerics_wasm.txt
