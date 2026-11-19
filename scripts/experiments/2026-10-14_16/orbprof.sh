O=/mnt/h/Research/results/m1_profile/orbprof; mkdir -p $O
A="--dataset euroc --mode mono --seq /mnt/h/Research/datasets/euroc/MH_01_easy --vocab /mnt/h/Research/vocab/ORBvoc.bin --settings /mnt/h/Research/CustomSLAM/baselines/orbslam3/settings/EuRoC_mono.yaml --seed 1 --max-frames 610"
/root/build/slam-native-prof/orbslam3_run $A --out $O/native > $O/native.log 2>&1
/opt/slam/emsdk/node/24.19.0_64bit/bin/node /root/build/slam-wasm-prof/orbslam3_run.js $A --out $O/wasm > $O/wasm.log 2>&1
grep -h "ORB-PROF" $O/native.log | tail -1 | sed 's/^/native: /'; grep -h "ORB-PROF" $O/wasm.log | tail -1 | sed 's/^/wasm:   /'
