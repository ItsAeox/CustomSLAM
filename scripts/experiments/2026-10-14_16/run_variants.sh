#!/usr/bin/env bash
# M3 ablation: WASM (profiling build: REGISTER_TIMES stats per run), real time, settings variants.
#   TUM-VI room1-6 x seeds 1-5 x {stock, light-map, light-orb, light-all}
#   EuRoC 6 sequences x seeds 1-3 x {stock, light-map, light-orb, light-all} + native stock
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; V=/mnt/h/Research/results/settings_variants
O=/mnt/h/Research/results/m3_variants
export WASM_JS=/root/build/slam-wasm-prof/orbslam3_run.js NATIVE=/root/build/slam-native-prof/orbslam3_run
EUROC="MH_01_easy MH_03_medium MH_05_difficult V1_01_easy V1_02_medium V2_02_medium"
echo "=== start $(date +%T)"
for var in light-map light-orb light-all stock; do
  SPECS=()
  for e in $EUROC; do SPECS+=("euroc_$e|euroc|$D/euroc/$e|$V/EuRoC_mono_$var.yaml|$G/euroc_$e.txt|"); done
  FL=wasm; [ $var = stock ] && FL="wasm native"
  FLAVOURS="$FL" JOBS=3 EXTRA="--realtime" bash scripts/seed_sweep.sh $O/$var "1 2 3" "${SPECS[@]}" | grep -c seed | sed "s/^/$var euroc runs: /"
  if true; then
    SPECS=()
    for s in $D/tumvi/*; do n=$(basename $s); SPECS+=("tumvi_${n%_512_16}|tumvi|$s|$V/TUM-VI_mono_$var.yaml|$G/tumvi_$n.txt|--clahe"); done
    FLAVOURS=wasm JOBS=3 EXTRA="--realtime" bash scripts/seed_sweep.sh $O/$var "1 2 3 4 5" "${SPECS[@]}" | grep -c seed | sed "s/^/$var tumvi runs: /"
  fi
  echo "=== $var done $(date +%T)"
done
for var in stock light-map light-orb light-all; do
  echo "=== summary $var"; /opt/slam/venv/bin/python eval/summarize_sweep.py $O/$var
done
echo "=== done $(date +%T)"
