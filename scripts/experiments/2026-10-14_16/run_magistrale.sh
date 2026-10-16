#!/usr/bin/env bash
# TUM-VI magistrale1-6 (long walks; GT only at start/end in the mocap room -> ATE measures accumulated drift).
# Queued after the KLT gate experiment. WASM stock vs hybrid, native stock as reference; seeds 1-3, real time.
set -uo pipefail
until grep -q "=== done" /mnt/h/Research/logs/klt_gates.log 2>/dev/null; do sleep 60; done
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets/tumvi; G=/mnt/h/Research/results/gt; V=/mnt/h/Research/results/settings_variants
O=/mnt/h/Research/results/magistrale
export WASM_JS=/root/build/slam-wasm/orbslam3_run.js NATIVE=/root/build/slam-native/orbslam3_run
specs() { SPECS=(); for i in 1 2 3 4 5 6; do n=dataset-magistrale${i}_512_16; SPECS+=("tumvi_magistrale$i|tumvi|$D/$n|$V/TUM-VI_mono_$1.yaml|$G/tumvi_$n.txt|--clahe"); done; }
echo "=== start $(date +%T)"
specs stock; FLAVOURS="wasm native" JOBS=3 EXTRA="--realtime" bash scripts/seed_sweep.sh $O/stock "1 2 3" "${SPECS[@]}" > /dev/null
echo "=== stock done $(date +%T)"
specs klt;   FLAVOURS=wasm JOBS=3 EXTRA="--realtime" bash scripts/seed_sweep.sh $O/klt "1 2 3" "${SPECS[@]}" > /dev/null
echo "=== klt done $(date +%T)"
P=/opt/slam/venv/bin/python
for v in stock klt; do echo "--- $v"; $P eval/summarize_sweep.py $O/$v --bad-ate-mm 1000; done
$P eval/compare_builds.py --across $O/stock $O/klt --a wasm --bad-ate-mm 1000
$P eval/compare_builds.py $O/stock --a native --b wasm --bad-ate-mm 1000
echo "=== done $(date +%T)"
