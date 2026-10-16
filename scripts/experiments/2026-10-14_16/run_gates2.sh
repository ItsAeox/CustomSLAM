#!/usr/bin/env bash
# KLT gate experiment (queued after the live phone-emulation experiment): native (gates are in slam-native-teblid), real time,
# seeds 1-3, the sequences where the hybrid struggled + controls. Variants: stock, klt, klt-flow15, klt-frac07, klt-gated.
set -uo pipefail
until grep -q "=== done" /mnt/h/Research/logs/live.log 2>/dev/null; do sleep 60; done
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; V=/mnt/h/Research/results/settings_variants
O=/mnt/h/Research/results/klt_gates
export NATIVE=/root/build/slam-native-teblid/orbslam3_run
echo "=== start $(date +%T)"
for var in stock klt klt-flow15 klt-frac07 klt-gated; do
  SPECS=()
  for e in MH_01_easy MH_05_difficult V1_03_difficult V2_01_easy V2_03_difficult; do SPECS+=("euroc_$e|euroc|$D/euroc/$e|$V/EuRoC_mono_$var.yaml|$G/euroc_$e.txt|"); done
  for r in room1 room2 room5; do SPECS+=("tumvi_dataset-$r|tumvi|$D/tumvi/dataset-${r}_512_16|$V/TUM-VI_mono_$var.yaml|$G/tumvi_dataset-${r}_512_16.txt|--clahe"); done
  FLAVOURS=native JOBS=3 EXTRA="--realtime" bash scripts/seed_sweep.sh $O/$var "1 2 3" "${SPECS[@]}" > /dev/null
  echo "=== $var done $(date +%T)"
done
P=/opt/slam/venv/bin/python
for var in klt klt-flow15 klt-frac07 klt-gated; do $P eval/compare_builds.py --across $O/stock $O/$var --a native --bad-ate-mm 200 --bad-rot 1000; done
for var in stock klt klt-flow15 klt-frac07 klt-gated; do echo "--- $var"; $P eval/summarize_sweep.py $O/$var --bad-ate-mm 200 --bad-rot 1000; grep -h KLT-STATS $O/$var/*/native_seed1/run.log | head -3; done
echo "=== done $(date +%T)"
