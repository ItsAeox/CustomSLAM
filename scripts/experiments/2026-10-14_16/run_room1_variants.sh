#!/usr/bin/env bash
# TUM-VI room1 WASM: is the residual gap timing (mapping too slow -> fewer KFs) or numerics?
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; S=$PWD/baselines/orbslam3/settings
SPEC="tumvi_dataset-room1|tumvi|$D/tumvi/dataset-room1_512_16|$S/TUM-VI_mono.yaml|$G/tumvi_dataset-room1_512_16.txt|--clahe"
FLAVOURS=wasm JOBS=3 EXTRA="--realtime --rate-scale 0.5" bash scripts/seed_sweep.sh /mnt/h/Research/results/m0_room1_half "1 2 3" "$SPEC" &
FLAVOURS=wasm JOBS=3 EXTRA="--sync-mapping" bash scripts/seed_sweep.sh /mnt/h/Research/results/m0_room1_sync "1 2 3" "$SPEC" &
wait
for d in m0_room1_half m0_room1_sync; do grep -h LM-STATS /mnt/h/Research/results/$d/*/*/run.log; done
