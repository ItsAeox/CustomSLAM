#!/usr/bin/env bash
# Check the EdgeSE3ProjectXYZ dangling-reference fix: WASM on EuRoC MH01 / V102 + TUM-VI room1, seeds 1-3, real-time.
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; S=$PWD/baselines/orbslam3/settings
SPECS=("euroc_MH_01_easy|euroc|$D/euroc/MH_01_easy|$S/EuRoC_mono.yaml|$G/euroc_MH_01_easy.txt|"
       "euroc_V1_02_medium|euroc|$D/euroc/V1_02_medium|$S/EuRoC_mono.yaml|$G/euroc_V1_02_medium.txt|"
       "tumvi_dataset-room1|tumvi|$D/tumvi/dataset-room1_512_16|$S/TUM-VI_mono.yaml|$G/tumvi_dataset-room1_512_16.txt|--clahe")
echo "=== start $(date +%T)"
FLAVOURS=wasm JOBS=3 EXTRA="--realtime" bash scripts/seed_sweep.sh /mnt/h/Research/results/m0_fixcheck "1 2 3" "${SPECS[@]}"
echo "=== done $(date +%T)"
