#!/usr/bin/env bash
# What makes the slowest 5 % of frames slow? Browser build with per-frame stage timing (REGISTER_TIMES:
# /root/build/slam-wasm-prof, rebuilt first), first 1500 frames of MH01 and room1, one run at a time, real time:
# "Hybrid, fastest" (preset-fast) and fast-speed2 (quick solver + parallel ORB + 4-thread flow). Then p95_stages.py
# splits the frames above the 95th percentile of total time by stage (ORB extraction, pose prediction, local-map
# tracking, keyframe decision) and by kind (flow / ORB / new keyframe).
# Usage: prof_p95.sh
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
echo "=== build prof $(date +%T)"
nice -n 10 cmake --build /root/build/slam-wasm-prof -j4 --target orbslam3_run 2>&1 | grep -E "error|FAILED" | head
NODE=/opt/slam/emsdk/node/24.19.0_64bit/bin/node; JS=/root/build/slam-wasm-prof/orbslam3_run.js
D=/mnt/h/Research/datasets; V=/mnt/h/Research/results/settings_variants; VOC=/mnt/h/Research/vocab/ORBvoc.bin
O=/mnt/h/Research/results/prof_p95
for a in preset-fast fast-speed2; do
  for n in mh01 room1; do
    o=$O/$a/$n; mkdir -p $o
    if [ $n = mh01 ]; then args="--dataset euroc --seq $D/euroc/MH_01_easy --settings $V/EuRoC_mono_$a.yaml"
    else args="--dataset tumvi --seq $D/tumvi/dataset-room1_512_16 --settings $V/TUM-VI_mono_$a.yaml --clahe"; fi
    [ -s $o/TrackingTimeStats.txt ] || $NODE $JS $args --mode mono --vocab $VOC --seed 1 --rt-wall --max-frames 1500 --out $o > $o/run.log 2>&1
    echo "--- $a $n"; /opt/slam/venv/bin/python scripts/experiments/2026-10-18/p95_stages.py $o
  done
done
echo "=== done $(date +%T)"
