#!/usr/bin/env bash
# Robustness of the hybrid on the hard sequences: 10 seeds, stock vs klt, native and WASM, real time.
# n = 3 per sequence was not enough to tell (native and WASM disagreed). Queued after the magistrale sweep.
set -uo pipefail
until grep -q "=== done" /mnt/h/Research/logs/magistrale.log 2>/dev/null; do sleep 60; done
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; V=/mnt/h/Research/results/settings_variants
O=/mnt/h/Research/results/hard10
export WASM_JS=/root/build/slam-wasm/orbslam3_run.js NATIVE=/root/build/slam-native/orbslam3_run
specs() {
  SPECS=()
  for e in MH_05_difficult V1_03_difficult V2_03_difficult; do SPECS+=("euroc_$e|euroc|$D/euroc/$e|$V/EuRoC_mono_$1.yaml|$G/euroc_$e.txt|"); done
  for r in room1 room5; do SPECS+=("tumvi_dataset-$r|tumvi|$D/tumvi/dataset-${r}_512_16|$V/TUM-VI_mono_$1.yaml|$G/tumvi_dataset-${r}_512_16.txt|--clahe"); done
}
echo "=== start $(date +%T)"
for var in stock klt; do
  specs $var
  FLAVOURS="wasm native" JOBS=3 EXTRA="--realtime" bash scripts/seed_sweep.sh $O/$var "1 2 3 4 5 6 7 8 9 10" "${SPECS[@]}" > /dev/null
  echo "=== $var done $(date +%T)"
done
P=/opt/slam/venv/bin/python
for fl in wasm native; do $P eval/compare_builds.py --across $O/stock $O/klt --a $fl --bad-ate-mm 200; done
echo "=== done $(date +%T)"
