#!/usr/bin/env bash
# AlvaAR baseline on EuRoC 11 (claude/PAPER.md gap 1): released bundle (fixed binary, so these runs stand for the
# main table as long as baselines/alvaar/alvaar_run.mjs is unchanged), 5 runs each (AlvaAR has no seed: run index),
# rt-wall and live@0.5. Waits for the WebVIO reference of the pilot to finish (pacing-sensitive runs alone).
set -uo pipefail
until grep -q "=== done" /mnt/h/Research/logs/alvaar_pilot_ref.log 2>/dev/null; do sleep 30; done
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; O=/mnt/h/Research/results/alvaar_euroc
S=baselines/orbslam3/settings; P=/opt/slam/venv/bin/python
export WASM_JS=/mnt/h/Research/CustomSLAM/baselines/alvaar/alvaar_run.mjs FLAVOURS=wasm
SPECS=()
for s in $D/euroc/*; do n=$(basename $s); SPECS+=("euroc_$n|euroc|$s|$S/EuRoC_mono.yaml|$G/euroc_$n.txt|"); done
echo "=== rt-wall $(date +%T)"
JOBS=3 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $O/rt "1 2 3 4 5" "${SPECS[@]}"
echo "=== live@0.5 $(date +%T)"
JOBS=3 EXTRA="--live --rate-scale 0.5" bash scripts/seed_sweep.sh $O/live_r0.5 "1 2 3 4 5" "${SPECS[@]}"
echo "=== summaries $(date +%T)"
for v in rt live_r0.5; do echo "--- $v"; $P eval/summarize_sweep.py $O/$v --bad-ate-mm 200 --bad-rot 1000; done
echo "=== done $(date +%T)"
