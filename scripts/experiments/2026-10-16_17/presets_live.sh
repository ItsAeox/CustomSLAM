#!/usr/bin/env bash
# presets_live: the AR page's visual-only presets on datasets under phone emulation (WASM --live, late frames dropped,
# --rate-scale 0.5 = a device 2x slower; the condition that separated arms in live_r0.5). Phone preset round
# (2026-10-16): "Hybrid, fastest" looked best (n = 1); this measures accuracy and robustness.
# Arms side by side (one run per arm at a time: identical load), seeds 1-3, TUM-VI rooms 1-6 (handheld) + the four
# hardest EuRoC sequences. Settings: results/settings_variants/*_preset-*.yaml (scripts/make_variants.py).
set -uo pipefail
until grep -q "=== done" /mnt/h/Research/logs/imu_sens.log 2>/dev/null; do sleep 60; done
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; V=/mnt/h/Research/results/settings_variants
P=/opt/slam/venv/bin/python
export WASM_JS=/root/build/snap-cea2e3b/wasm/orbslam3_run.js
RATE=0.5; O=/mnt/h/Research/results/presets_live
ARMS="preset-stock preset-hybrid preset-fast preset-light"
specs() {
  SPECS=()
  for n in MH_04_difficult MH_05_difficult V1_03_difficult V2_03_difficult; do
    SPECS+=("euroc_$n|euroc|$D/euroc/$n|$V/EuRoC_mono_$1.yaml|$G/euroc_$n.txt|"); done
  for r in 1 2 3 4 5 6; do n=dataset-room${r}_512_16
    SPECS+=("tumvi_dataset-room$r|tumvi|$D/tumvi/$n|$V/TUM-VI_mono_$1.yaml|$G/tumvi_$n.txt|--clahe"); done
}
echo "=== start $(date +%T) rate $RATE binary $WASM_JS"
for a in $ARMS; do
  specs $a; FLAVOURS=wasm JOBS=1 EXTRA="--live --rate-scale $RATE" bash scripts/seed_sweep.sh $O/$a "1 2 3" "${SPECS[@]}" > /dev/null &
done
wait
echo "=== summaries $(date +%T)"
for a in $ARMS; do echo "--- $a"; $P eval/summarize_sweep.py $O/$a --bad-ate-mm 200 --bad-rot 1000; done
echo "pooled bad runs and dropped frames per arm:"
for a in $ARMS; do
  bad=$($P eval/summarize_sweep.py $O/$a --bad-ate-mm 200 --bad-rot 1000 | awk '$2=="wasm"{b+=$5; n+=$3} END{print b "/" n}')
  drop=$(awk '/^frames/{f+=$2} /^dropped/{d+=$2} END{printf "%.3f", d/f}' $O/$a/*/*/run_info.txt)
  echo "  $a: bad $bad, dropped $drop"
done
$P eval/compare_builds.py --across $O/preset-stock $O/preset-fast --a wasm --bad-ate-mm 200 --bad-rot 1000
$P eval/compare_builds.py --across $O/preset-hybrid $O/preset-fast --a wasm --bad-ate-mm 200 --bad-rot 1000
echo "=== done $(date +%T)"
