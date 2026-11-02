#!/usr/bin/env bash
# WebVIO reference for the AlvaAR pilot: same sequence / protocols, current WASM build (slam-wasm-ui), stock EuRoC
# settings, with --trace so the online poses can be scored (eval/trace_to_tum.py) next to the final trajectory.
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; O=/mnt/h/Research/results/alvaar_pilot/webvio
S=baselines/orbslam3/settings; P=/opt/slam/venv/bin/python
export WASM_JS=/root/build/slam-wasm-ui/orbslam3_run.js FLAVOURS=wasm
ls -la --time-style=full-iso $WASM_JS
SPEC="euroc_MH_01_easy|euroc|$D/euroc/MH_01_easy|$S/EuRoC_mono.yaml|$G/euroc_MH_01_easy.txt|--trace"
echo "=== rt-wall seeds 1 2 3 $(date +%T)"
JOBS=3 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $O "1 2 3" "$SPEC"
echo "=== live@0.5 seed 1 $(date +%T)"
JOBS=1 EXTRA="--live --rate-scale 0.5" bash scripts/seed_sweep.sh $O/live_r0.5 "1" "$SPEC"
echo "=== online poses $(date +%T)"
for r in $O/euroc_MH_01_easy/wasm_seed* $O/live_r0.5/euroc_MH_01_easy/wasm_seed*; do
  $P eval/trace_to_tum.py --trace $r/trace.txt --seq $D/euroc/MH_01_easy --out $r/online_tum.txt
  $P eval/evaluate.py --gt $G/euroc_MH_01_easy.txt --est $r/online_tum.txt --out $r/metrics_online.json --label "online $(basename $r)" | tail -1
done
echo "=== done $(date +%T)"
