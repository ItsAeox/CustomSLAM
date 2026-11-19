#!/usr/bin/env bash
# Smoke test: the WASM runner (node) loads the map file saved by the native runner (portable format check)
O=/mnt/h/Research/results/scratch/smoke_map; mkdir -p $O/Bwasm; cd $O
/opt/slam/emsdk/node/24.19.0_64bit/bin/node /root/build/slam-wasm-ui/orbslam3_run.js --dataset euroc --mode mono \
  --seq /mnt/h/Research/datasets/euroc/MH_02_easy --vocab /mnt/h/Research/vocab/ORBvoc.bin \
  --settings /mnt/h/Research/results/settings_variants/EuRoC_mono_preset-fast.yaml --seed 1 --max-frames 600 \
  --out $O/Bwasm --load-map $O/A/map.bin > $O/Bwasm/run.log 2>&1
echo "exit $?"; grep -h 'WEBMAP\|Aborted\|RuntimeError' $O/Bwasm/run.log | head; grep '^maps\|^keyframes' $O/Bwasm/run_info.txt
