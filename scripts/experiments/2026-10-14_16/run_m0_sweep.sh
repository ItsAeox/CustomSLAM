#!/usr/bin/env bash
# M0: native vs WASM ORB-SLAM3 port on EuRoC (11) + TUM-VI rooms (6), seeds 1-3, real-time pacing.
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; S=$PWD/baselines/orbslam3/settings
SPECS=()
for s in $D/euroc/*; do n=$(basename $s); SPECS+=("euroc_$n|euroc|$s|$S/EuRoC_mono.yaml|$G/euroc_$n.txt|"); done
for s in $D/tumvi/*; do n=$(basename $s); SPECS+=("tumvi_${n%_512_16}|tumvi|$s|$S/TUM-VI_mono.yaml|$G/tumvi_$n.txt|--clahe"); done
echo "=== start $(date +%T) (${#SPECS[@]} sequences)"
JOBS=3 EXTRA="--realtime" bash scripts/seed_sweep.sh /mnt/h/Research/results/m0_sweep "1 2 3" "${SPECS[@]}"
echo "=== summary $(date +%T)"
/opt/slam/venv/bin/python eval/summarize_sweep.py /mnt/h/Research/results/m0_sweep
echo "=== done $(date +%T)"
