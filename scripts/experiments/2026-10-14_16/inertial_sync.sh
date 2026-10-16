#!/usr/bin/env bash
# inertial_rt2 follow-up: WASM mono-inertial failed 9/18 on TUM-VI rooms (native 0/18) at real time. Speed or bug?
# Same rooms that failed most (1, 3, 5), WASM, --sync-mapping (each keyframe fully mapped before the next frame:
# mapping can't fall behind), unpaced. As m0_room1_sync for visual-only (2026-10-14).
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; R=/mnt/h/Research/results; P=/opt/slam/venv/bin/python
SET=/mnt/h/Research/CustomSLAM/baselines/orbslam3/settings
export WASM_JS=/root/build/slam-wasm-ui/orbslam3_run.js
echo "=== start $(date +%T)"
O=$R/inertial_sync; SPECS=()
for r in room1 room3 room5; do n=dataset-${r}_512_16
  SPECS+=("tumvi_dataset-$r|tumvi|$D/tumvi/$n|$SET/TUM-VI_mono_inertial.yaml|$G/tumvi_${n}_body.txt|--clahe --mode mono_inertial"); done
FLAVOURS=wasm JOBS=2 EXTRA="--sync-mapping" bash scripts/seed_sweep.sh $O/wasm_sync "1 2 3" "${SPECS[@]}" > /dev/null
$P eval/summarize_sweep.py $O/wasm_sync --bad-ate-mm 200 --bad-rot 1000
echo "=== done $(date +%T)"
