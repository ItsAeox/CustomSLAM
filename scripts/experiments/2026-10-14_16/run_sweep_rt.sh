#!/usr/bin/env bash
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
echo "=== relink $(date +%T)"
cmake --build /root/build/slam-native --target orbslam3_run -j 2 2>&1 | grep -E "error|FAILED" || true
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
cmake --build /root/build/slam-wasm --target orbslam3_run -j 2 2>&1 | grep -E "error:|FAILED" || true
R=/mnt/h/Research/results/scratch
echo "=== sweep $(date +%T)"
EXTRA="--realtime" bash scripts/seed_sweep.sh "$R/seed_sweep_rt" "1 2 3 4 5" \
  "mocap-desk2:$R/settings/tumvie_mocap-desk2.yaml:$R/mocap-desk2/gt_tum.txt" \
  "mocap-desk:$R/settings/tumvie_mocap-desk.yaml:$R/gt_mocap-desk.txt"
echo "=== summary $(date +%T)"
/opt/slam/venv/bin/python eval/summarize_sweep.py "$R/seed_sweep_rt"
echo "=== done $(date +%T)"
