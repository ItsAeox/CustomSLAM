#!/usr/bin/env bash
# Queue v4 (after queue v3): Magistrale1 WASM abort diagnostic. In magistrale_rt, 3 WASM runs aborted (Aborted(),
# no message) right after "*Loop detected" late in magistrale1, while 4 long runs shared WSL's 7.8 GB. Rerun magistrale1
# WASM stock + hybrid, seeds 1-3, only 2 runs at a time, with the new runner (map size + memory every 200 frames):
# memory climbing to the limit before the abort -> memory; flat memory -> a code path in loop closing.
set -uo pipefail
until grep -q "queue3 finished" /mnt/h/Research/logs/queue3.log 2>/dev/null; do sleep 60; done
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets/tumvi; G=/mnt/h/Research/results/gt; V=/mnt/h/Research/results/settings_variants
export WASM_JS=/root/build/slam-wasm-ui/orbslam3_run.js
O=/mnt/h/Research/results/magistrale1_diag; L=/mnt/h/Research/logs/magistrale1_diag.log
{
  echo "=== start $(date +%T)"
  n=dataset-magistrale1_512_16
  for v in stock klt; do
    FLAVOURS=wasm JOBS=1 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $O/$v "1 2 3" \
      "tumvi_magistrale1|tumvi|$D/$n|$V/TUM-VI_mono_$v.yaml|$G/tumvi_$n.txt|--clahe" > /dev/null &
  done
  wait
  for f in $O/*/*/*/run.log; do
    echo "--- $f"
    grep -E "^frame (1[0-9]|[0-9])[0-9]00/|Loop detected|Merge|Abort|run failed" "$f" | grep -v GROWABLE | tail -8 | cut -c1-160
    grep -E "^(map_points|keyframes|maps|mem_mb|effective_rate) " "$(dirname $f)/run_info.txt" 2>/dev/null | tr '\n' ' '; echo
  done
  echo "=== done $(date +%T)"
} > $L 2>&1
echo "queue4 finished $(date +%T)" >> /mnt/h/Research/logs/queue4.log
