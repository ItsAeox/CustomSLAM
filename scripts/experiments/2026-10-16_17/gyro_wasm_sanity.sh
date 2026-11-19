#!/usr/bin/env bash
# The gyro prior must be active in the WASM snapshot that gyro_live uses: room1, --gyro, stock-gyro settings, ~90 s.
set -uo pipefail
O=/mnt/h/Research/results/gyro_sanity/wasm/tumvi_dataset-room1/wasm_seed1; mkdir -p $O
timeout 90 /opt/slam/emsdk/node/24.19.0_64bit/bin/node /root/build/snap-gyro1/wasm/orbslam3_run.js --dataset tumvi --mode mono \
  --seq /mnt/h/Research/datasets/tumvi/dataset-room1_512_16 --vocab /mnt/h/Research/vocab/ORBvoc.bin \
  --settings /mnt/h/Research/results/settings_variants/TUM-VI_mono_stock-gyro.yaml --out $O --seed 1 --clahe --gyro > $O/run.log 2>&1
echo "exit $? (124 = stopped by the 90 s timeout, expected)"
grep -o "Web config.*" $O/run.log | head -1 | cut -c1-220
grep -c "gyro prior off" $O/run.log | sed 's/^/"gyro prior off" lines: /'
grep "^frame" $O/run.log | tail -1 | cut -c1-100
