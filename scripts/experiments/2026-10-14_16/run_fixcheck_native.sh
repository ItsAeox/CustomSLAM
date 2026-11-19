#!/usr/bin/env bash
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; S=$PWD/baselines/orbslam3/settings
rm -rf /mnt/h/Research/results/m0_fixcheck/tumvi_dataset-room1/native_seed*
FLAVOURS=native JOBS=3 EXTRA="--realtime" bash scripts/seed_sweep.sh /mnt/h/Research/results/m0_fixcheck "1 2 3" \
  "tumvi_dataset-room1|tumvi|$D/tumvi/dataset-room1_512_16|$S/TUM-VI_mono.yaml|$G/tumvi_dataset-room1_512_16.txt|--clahe"
