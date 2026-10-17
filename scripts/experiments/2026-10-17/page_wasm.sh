#!/usr/bin/env bash
# The AR page's presets exactly as the page builds them (scripts/page_presets.mjs -> *_page-<key>.yaml), WASM (Node
# runner), phone emulation (--live --rate-scale 0.5), screen subset, seeds 1-3, arms side by side. Catches crashes and
# bad interactions of the combined presets before the phone round. Gyro presets: --gyro; voc small: own k10L5.
# Usage: page_wasm.sh <snapshot tag> <out name> <preset key...>   (the first key is the baseline)
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; V=/mnt/h/Research/results/settings_variants; P=/opt/slam/venv/bin/python
export WASM_JS=/root/build/snap-$1/wasm/orbslam3_run.js
O=/mnt/h/Research/results/$2; shift 2
echo "=== start $(date +%T) binary $WASM_JS presets $*"
for k in "$@"; do
  a=page-$k; SPECS=()
  for n in V1_03_difficult V2_03_difficult MH_04_difficult MH_05_difficult; do
    SPECS+=("euroc_$n|euroc|$D/euroc/$n|$V/EuRoC_mono_$a.yaml|$G/euroc_$n.txt|")
  done
  for r in 1 2 3 5; do n=dataset-room${r}_512_16; SPECS+=("tumvi_dataset-room$r|tumvi|$D/tumvi/$n|$V/TUM-VI_mono_$a.yaml|$G/tumvi_$n.txt|--clahe"); done
  g=""; grep -q "Web.Gyro.Prior: 1" $V/EuRoC_mono_$a.yaml && g="--gyro"
  voc=/mnt/h/Research/vocab/ORBvoc.bin; grep -q "Web.BoW.LevelsUp: 3" $V/EuRoC_mono_$a.yaml && voc=/mnt/h/Research/vocab/ORBvoc_own_k10L5.bin
  VOCAB=$voc FLAVOURS=wasm JOBS=1 EXTRA="--live --rate-scale 0.5 $g" bash scripts/seed_sweep.sh $O/$a "1 2 3" "${SPECS[@]}" > /dev/null &
done
wait
echo "=== done $(date +%T)"
$P eval/screen_table.py $O page-$1
grep -l "Assertion\|Aborted\|RuntimeError\|Segmentation\|terminate called" $O/*/*/*/run.log 2>/dev/null | head
