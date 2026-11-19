#!/usr/bin/env bash
# Diagnostic (after the EdgeSE3ProjectXYZ fix): is WASM systematically worse on fisheye data?
# TUM-VI room1-6 + TUM-VIE mocap-desk/desk2, seeds 1-5, native + WASM, real-time pacing.
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; S=$PWD/baselines/orbslam3/settings; R=/mnt/h/Research/results/scratch
SPECS=("mocap-desk2:$R/settings/tumvie_mocap-desk2.yaml:$R/mocap-desk2/gt_tum.txt"
       "mocap-desk:$R/settings/tumvie_mocap-desk.yaml:$R/gt_mocap-desk.txt")
for s in $D/tumvi/*; do n=$(basename $s); SPECS+=("tumvi_${n%_512_16}|tumvi|$s|$S/TUM-VI_mono.yaml|$G/tumvi_$n.txt|--clahe"); done
echo "=== start $(date +%T) (${#SPECS[@]} sequences)"
JOBS=3 EXTRA="--realtime" bash scripts/seed_sweep.sh /mnt/h/Research/results/m0_fisheye5 "1 2 3 4 5" "${SPECS[@]}"
echo "=== summary $(date +%T)"
/opt/slam/venv/bin/python eval/summarize_sweep.py /mnt/h/Research/results/m0_fisheye5
echo "=== done $(date +%T)"
