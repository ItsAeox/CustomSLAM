#!/usr/bin/env bash
# Phone-emulation experiment (queued after the TEBLID evaluation): WASM runner with --live pacing (late frames are
# dropped, as a live camera in the browser) at rate-scale RATE (< 1 = a slower device), stock motion model vs
# time-scaled (Web.Track.TimeScaledMotion). Both arms side by side (identical load), seeds 1-3,
# EuRoC 11 + TUM-VI rooms 6.
set -uo pipefail
RATE="${RATE:-0.5}"
until grep -q "=== done" /mnt/h/Research/logs/teblid_eval.log 2>/dev/null; do sleep 60; done
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; V=/mnt/h/Research/results/settings_variants
export WASM_JS=/root/build/slam-wasm-ui/orbslam3_run.js
O=/mnt/h/Research/results/live_r$RATE
specs() {
  SPECS=()
  for s in $D/euroc/*; do n=$(basename $s); SPECS+=("euroc_$n|euroc|$s|$V/EuRoC_mono_$1.yaml|$G/euroc_$n.txt|"); done
  for s in $D/tumvi/dataset-room*; do n=$(basename $s); SPECS+=("tumvi_${n%_512_16}|tumvi|$s|$V/TUM-VI_mono_$1.yaml|$G/tumvi_$n.txt|--clahe"); done
}
echo "=== start $(date +%T) rate $RATE"
specs stock; FLAVOURS=wasm JOBS=2 EXTRA="--live --rate-scale $RATE" bash scripts/seed_sweep.sh $O/stock "1 2 3" "${SPECS[@]}" > /dev/null &
specs tsm;   FLAVOURS=wasm JOBS=2 EXTRA="--live --rate-scale $RATE" bash scripts/seed_sweep.sh $O/tsm "1 2 3" "${SPECS[@]}" > /dev/null &
wait
P=/opt/slam/venv/bin/python
echo "=== summaries $(date +%T)"
for v in stock tsm; do echo "--- $v"; $P eval/summarize_sweep.py $O/$v --bad-ate-mm 200 --bad-rot 1000; done
$P eval/compare_builds.py --across $O/stock $O/tsm --a wasm --bad-ate-mm 200 --bad-rot 1000
echo "dropped frames (fraction of all frames), per arm:"
for v in stock tsm; do
  awk '/^frames/{f+=$2} /^dropped/{d+=$2} END{printf "  '$v': %.3f\n", d/f}' $O/$v/*/*/run_info.txt
done
echo "=== done $(date +%T)"
