#!/usr/bin/env bash
# Hybrid ORB+KLT vs stock ORB-SLAM3: EuRoC (11) + TUM-VI room1-6, seeds 1-3, real time.
# WASM (regular build) first, then native. Results: results/klt_sweep/<variant>/<seq>/<flavour>_seed<n>
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; V=/mnt/h/Research/results/settings_variants
O=/mnt/h/Research/results/klt_sweep
export WASM_JS=/root/build/slam-wasm/orbslam3_run.js NATIVE=/root/build/slam-native/orbslam3_run
echo "=== start $(date +%T)"
for fl in wasm native; do
  for var in klt stock; do
    SPECS=()
    for s in $D/euroc/*; do n=$(basename $s); SPECS+=("euroc_$n|euroc|$s|$V/EuRoC_mono_$var.yaml|$G/euroc_$n.txt|"); done
    for s in $D/tumvi/*; do n=$(basename $s); SPECS+=("tumvi_${n%_512_16}|tumvi|$s|$V/TUM-VI_mono_$var.yaml|$G/tumvi_$n.txt|--clahe"); done
    FLAVOURS=$fl JOBS=3 EXTRA="--realtime" bash scripts/seed_sweep.sh $O/$var "1 2 3" "${SPECS[@]}" | grep -c seed | sed "s/^/$fl $var runs: /"
    echo "=== $fl $var done $(date +%T)"
  done
done
for var in klt stock; do echo "=== summary $var"; /opt/slam/venv/bin/python eval/summarize_sweep.py $O/$var; done
echo "=== hybrid vs stock (WASM)"
for fl in wasm native; do /opt/slam/venv/bin/python eval/compare_builds.py --across $O/stock $O/klt --a $fl; done
echo "=== done $(date +%T)"
