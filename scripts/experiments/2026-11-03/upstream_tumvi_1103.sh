#!/usr/bin/env bash
# Upstream ORB-SLAM3 monocular on the six TUM-VI rooms, 5 runs each (ISSUES G39; upstream_tumvi_job.sh), 3 at a time,
# rooms interleaved; machine load logged every 60 s. Results: results/upstream_tumvi_1103/<room>/run<N>/.
set -uo pipefail
R=/mnt/h/Research; J=$R/CustomSLAM/scripts/experiments/2026-11-03/upstream_tumvi_job.sh
mkdir -p $R/results/upstream_tumvi_1103
( while true; do echo "$(date +%T) $(cut -d' ' -f1-3 /proc/loadavg) runs $(pgrep -c mono_tum_vi)" >> $R/logs/upstream_tumvi_1103_load.log; sleep 60; done ) &
LOADER=$!
echo "=== start $(date +%T)" >> $R/logs/upstream_tumvi_1103_runs.log
for run in 1 2 3 4 5; do for room in room1 room2 room3 room4 room5 room6; do echo "$room $run"; done; done \
  | xargs -P3 -n2 bash $J
kill $LOADER
echo "=== done $(date +%T)" >> $R/logs/upstream_tumvi_1103_runs.log
