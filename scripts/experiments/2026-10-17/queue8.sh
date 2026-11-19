#!/usr/bin/env bash
# Queue 8 (2026-10-17): finish what the PC shutdown interrupted in queue 7.
#   outdoors_h3: outdoors4 seeds 2-3 (native + WASM) were cut off; re-run them, then score every map of every run
#   vocab_eval2 (not started)
set -uo pipefail
E=/mnt/h/Research/CustomSLAM/scripts/experiments/2026-10-16_17; L=/mnt/h/Research/logs
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; V=/mnt/h/Research/results/settings_variants; P=/opt/slam/venv/bin/python
export NATIVE=/root/build/snap-kf1/native/orbslam3_run WASM_JS=/root/build/snap-kf1/wasm/orbslam3_run.js
O=/mnt/h/Research/results/outdoors_h3
{
  echo "=== outdoors_h3 resume $(date +%T)"
  n=dataset-outdoors4_512_16
  FLAVOURS="native wasm" JOBS=2 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $O/stock "2 3" \
    "tumvi_outdoors4|tumvi|$D/tumvi/$n|$V/TUM-VI_mono_stock.yaml|$G/tumvi_$n.txt|--clahe" > /dev/null
  for f in $O/stock/*/*/; do
    q=$(basename $(dirname $f) | sed 's/tumvi_//')
    echo "$(basename $(dirname $f))/$(basename $f): $($P eval/evaluate_multimap.py --gt $G/tumvi_dataset-${q}_512_16.txt --run $f --out $f/metrics_multimap.json 2>&1 | tail -1)"
  done
  echo "=== outdoors_h3 done $(date +%T)"
} > $L/outdoors_h3.log 2>&1
bash $E/vocab_eval2.sh kf1 > $L/vocab_eval.log 2>&1
echo "=== queue8 done $(date +%T)"
