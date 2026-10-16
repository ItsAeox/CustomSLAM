set -uo pipefail
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
for b in slam-wasm slam-wasm-prof; do cmake --build /root/build/$b -j2 2>&1 | grep -E "error|FAILED" || true; done
cp /root/build/slam-wasm/orbslam3.js /root/build/slam-wasm/orbslam3.wasm /root/build/slam-wasm/orbslam3.worker.mjs /mnt/h/Research/CustomSLAM/web/build/
S=/mnt/c/Users/Aeox/AppData/Local/Temp/claude/H--Research/6624e451-5f38-4671-9a62-226588a85597/scratchpad
bash $S/orbhash.sh /mnt/h/Research/results/scratch/orbhash_after.txt > /dev/null
if diff -q /mnt/h/Research/results/scratch/orbhash_before.txt /mnt/h/Research/results/scratch/orbhash_after.txt; then echo "ORB descriptors bit-identical"; else echo "ORB OUTPUT CHANGED"; diff /mnt/h/Research/results/scratch/orbhash_before.txt /mnt/h/Research/results/scratch/orbhash_after.txt; fi
bash $S/orbprof.sh | grep wasm
