#!/usr/bin/env bash
# Rebuild native + wasm drivers with LocalMapping diagnostics, then sweep with --sync-mapping (and record LM stats).
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
echo "=== native build $(date +%T)"
cmake --build /root/build/slam-native --target orbslam3_run -j 2 2>&1 | grep -E "error|FAILED" || true
echo "=== wasm build $(date +%T)"
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
cmake --build /root/build/slam-wasm --target orbslam3_run -j 2 2>&1 | grep -E "error:|FAILED" || true
R=/mnt/h/Research/results/scratch
for mode in async sync; do
  EXTRA="--realtime"; [ "$mode" = sync ] && EXTRA="--realtime --sync-mapping"
  echo "=== sweep $mode $(date +%T)"
  EXTRA="$EXTRA" bash scripts/seed_sweep.sh "$R/seed_sweep_$mode" "1 2 3 4 5" \
    "mocap-desk2:$R/settings/tumvie_mocap-desk2.yaml:$R/mocap-desk2/gt_tum.txt"
  /opt/slam/venv/bin/python eval/summarize_sweep.py "$R/seed_sweep_$mode"
  for d in "$R/seed_sweep_$mode"/mocap-desk2/*; do echo "$(basename $d): $(grep -m1 LM-STATS $d/run.log)"; done
done
echo "=== done $(date +%T)"
