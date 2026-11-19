#!/usr/bin/env bash
# G12: native memory curve on magistrale1 (peak RSS every 200 frames), stock + klt, seed 1, rt-wall.
# If native peaks above ~4 GB around loop closing, the WASM aborts are the 4 GB wasm32 heap limit.
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; V=/mnt/h/Research/results/settings_variants
export NATIVE=/root/build/slam-native-teblid/orbslam3_run
n=dataset-magistrale1_512_16
echo "=== start $(date +%T)"
for v in stock klt; do
  FLAVOURS=native JOBS=1 EXTRA="--rt-wall" bash scripts/seed_sweep.sh /mnt/h/Research/results/magistrale1_diag/native_$v "1" \
    "tumvi_magistrale1|tumvi|$D/tumvi/$n|$V/TUM-VI_mono_$v.yaml|$G/tumvi_$n.txt|--clahe" > /dev/null &
done
wait
for f in /mnt/h/Research/results/magistrale1_diag/native_*/tumvi_magistrale1/*/run.log; do
  echo "--- $f"
  grep -E "^frame|Loop detected|Merge detected" $f | awk '/Loop|Merge/{print "   " $0; next} {print}' | grep -B1 -A2 "Loop\|Merge" | cut -c1-110
  grep -E "^(mem_mb|map_points|keyframes|maps) " $(dirname $f)/run_info.txt | tr '\n' ' '; echo
done
echo "=== done $(date +%T)"
