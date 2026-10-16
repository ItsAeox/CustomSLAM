#!/usr/bin/env bash
# Wait for the real-time sweep, relink drivers, then run WASM at half speed (--realtime --rate-scale 2).
set -uo pipefail
until grep -q "=== done" /mnt/h/Research/logs/seed_sweep_rt.log 2>/dev/null; do sleep 10; done
cd /mnt/h/Research/CustomSLAM
echo "=== relink $(date +%T)"
cmake --build /root/build/slam-native --target orbslam3_run -j 2 2>&1 | grep -E "error|FAILED" || true
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
cmake --build /root/build/slam-wasm --target orbslam3_run -j 2 2>&1 | grep -E "error:|FAILED" || true
R=/mnt/h/Research/results/scratch
echo "=== half-speed sweep $(date +%T)"
FLAVOURS="wasm" EXTRA="--realtime --rate-scale 2" bash scripts/seed_sweep.sh "$R/seed_sweep_half" "1 2 3 4 5" \
  "mocap-desk2:$R/settings/tumvie_mocap-desk2.yaml:$R/mocap-desk2/gt_tum.txt"
/opt/slam/venv/bin/python eval/summarize_sweep.py "$R/seed_sweep_half"
echo "=== done $(date +%T)"
