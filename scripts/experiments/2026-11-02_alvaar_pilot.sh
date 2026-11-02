#!/usr/bin/env bash
# AlvaAR baseline pilot (claude/PAPER.md gap 1): released bundle, EuRoC MH01, rt-wall x3 + live@0.5 x1,
# through seed_sweep.sh (checks that baselines/alvaar/alvaar_run.mjs is a drop-in for the WASM runner).
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; O=/mnt/h/Research/results/alvaar_pilot/alvaar
S=baselines/orbslam3/settings
export WASM_JS=/mnt/h/Research/CustomSLAM/baselines/alvaar/alvaar_run.mjs FLAVOURS=wasm
SPEC="euroc_MH_01_easy|euroc|$D/euroc/MH_01_easy|$S/EuRoC_mono.yaml|$G/euroc_MH_01_easy.txt|"
echo "=== rt-wall seeds 2 3 $(date +%T)"
JOBS=2 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $O "2 3" "$SPEC"
echo "=== live@0.5 seed 1 $(date +%T)"
JOBS=1 EXTRA="--live --rate-scale 0.5" bash scripts/seed_sweep.sh $O/live_r0.5 "1" "$SPEC"
echo "=== done $(date +%T)"
