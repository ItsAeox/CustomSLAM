#!/usr/bin/env bash
# Pilot for the phone-emulation experiment: MH01 (WASM, --live) at rate-scale 1.0 / 0.5 / 0.33, stock vs tsm,
# seed 1. Measures how many frames are dropped (run_info.txt) and checks the time-scaled motion model runs.
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; V=/mnt/h/Research/results/settings_variants
export WASM_JS=/root/build/slam-wasm-ui/orbslam3_run.js
O=/mnt/h/Research/results/live_pilot
for r in 1.0 0.5 0.33; do
  for v in stock tsm; do
    FLAVOURS=wasm JOBS=1 EXTRA="--live --rate-scale $r" bash scripts/seed_sweep.sh $O/r$r/$v "1" \
      "euroc_MH_01_easy|euroc|$D/euroc/MH_01_easy|$V/EuRoC_mono_$v.yaml|$G/euroc_MH_01_easy.txt|" > /dev/null &
  done
  wait
  for v in stock tsm; do
    f=$O/r$r/$v/euroc_MH_01_easy/wasm_seed1
    echo "rate $r $v: $(grep -E '^(processed|dropped)' $f/run_info.txt | tr '\n' ' ') ATE sim3 $(python3 -c "import json;m=json.load(open('$f/metrics.json'));print(round(m['ate_sim3']['rmse']*1000,1),'mm cov',round(m['coverage'],3))" 2>/dev/null)"
  done
done
echo "=== pilot done $(date +%T)"
