#!/usr/bin/env bash
# After klt_sweep: (1) hybrid + light ORB settings; (2) light-map vs stock run side by side (identical load).
set -uo pipefail
until grep -q "=== done" /mnt/h/Research/logs/klt_sweep.log 2>/dev/null; do sleep 60; done
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; V=/mnt/h/Research/results/settings_variants
export WASM_JS=/root/build/slam-wasm/orbslam3_run.js NATIVE=/root/build/slam-native/orbslam3_run
specs() {  # variant -> SPECS array (EuRoC 11 + TUM-VI 6)
  SPECS=()
  for s in $D/euroc/*; do n=$(basename $s); SPECS+=("euroc_$n|euroc|$s|$V/EuRoC_mono_$1.yaml|$G/euroc_$n.txt|"); done
  for s in $D/tumvi/*; do n=$(basename $s); SPECS+=("tumvi_${n%_512_16}|tumvi|$s|$V/TUM-VI_mono_$1.yaml|$G/tumvi_$n.txt|--clahe"); done
}
echo "=== klt-orb $(date +%T)"
specs klt-orb
FLAVOURS=wasm JOBS=3 EXTRA="--realtime" bash scripts/seed_sweep.sh /mnt/h/Research/results/klt_sweep/klt-orb "1 2 3" "${SPECS[@]}" > /dev/null
echo "=== light-map vs stock, side by side $(date +%T)"
O=/mnt/h/Research/results/m3_clean
specs stock;     FLAVOURS=wasm JOBS=2 EXTRA="--realtime" bash scripts/seed_sweep.sh $O/stock "1 2 3" "${SPECS[@]}" > /dev/null &
specs light-map; FLAVOURS=wasm JOBS=2 EXTRA="--realtime" bash scripts/seed_sweep.sh $O/light-map "1 2 3" "${SPECS[@]}" > /dev/null &
wait
P=/opt/slam/venv/bin/python
echo "=== summaries"
for d in /mnt/h/Research/results/klt_sweep/klt-orb $O/stock $O/light-map; do echo "--- $d"; $P eval/summarize_sweep.py $d --bad-ate-mm 200; done
$P eval/compare_builds.py --across $O/stock $O/light-map --a wasm --bad-ate-mm 200
$P eval/compare_builds.py --across /mnt/h/Research/results/klt_sweep/stock /mnt/h/Research/results/klt_sweep/klt-orb --a wasm --bad-ate-mm 200
echo "=== done $(date +%T)"
