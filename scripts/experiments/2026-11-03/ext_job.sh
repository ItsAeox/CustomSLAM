#!/usr/bin/env bash
# One run of ext_1103.sh: <arm> <kind: synth|phone> <sequence dir> <seed>. Same arms and binaries as main_job.sh, on the
# synthetic set (exact ground truth: results/gt/synth_<seq>.txt) and the user's Pixel 9 recordings (no ground truth:
# tracking metrics from trace.txt / frames.txt). Page presets are built from the sequence's own settings.yaml as the AR
# page builds them, plus the Pixel 9's IMU time offset (-88 ms, as every earlier synthetic / recording run) for presets
# that read the gyro. --rt-wall (ISSUES H4 does not apply in order).
set -uo pipefail
ARM="$1"; KIND="$2"; SEQ="$3"; SEED="$4"
R=/mnt/h/Research; REPO=$R/CustomSLAM; O0=$R/results/ext_1103
SNAP=/root/build/snap-84c1f17; NODE=/opt/slam/emsdk/node/24.19.0_64bit/bin/node; P=/opt/slam/venv/bin/python
n=$(basename "$SEQ"); NAME=${KIND}_$n; V=$O0/settings/$n
GT=""; [ "$KIND" = synth ] && GT=$R/results/gt/synth_$n.txt
case "$ARM" in
  wasm-stock) ST=$SEQ/settings.yaml; CMD=($NODE --report-on-signal $SNAP/wasm/orbslam3_run.js) ;;
  page-*)     ST=$V/${n}_$ARM.yaml; CMD=($NODE --report-on-signal $SNAP/wasm/orbslam3_run.js) ;;
  alvaar)     ST=$SEQ/settings.yaml; CMD=($NODE $REPO/baselines/alvaar/alvaar_run.mjs) ;;
esac
O=$O0/$ARM/$NAME/wasm_seed$SEED
mkdir -p "$O"
X=(--rt-wall)
if [ "$ARM" != alvaar ]; then
  X+=(--trace)
  grep -qE "^Web\.(Gyro|Scale)\." "$ST" && X+=(--gyro)
  CMD=("${CMD[0]}" --report-directory="$O" "${CMD[@]:1}")
fi
echo "$(date +%T) start $ARM $NAME seed$SEED" >> $R/logs/ext_1103_runs.log
timeout -k 30 -s USR2 1200 "${CMD[@]}" --dataset euroc --mode mono --seq "$SEQ" --vocab $R/vocab/ORBvoc.bin \
  --settings "$ST" --out "$O" --seed $SEED "${X[@]}" > "$O/run.log" 2>&1
rc=$?; [ $rc -ne 0 ] && echo "run failed (exit $rc)" >> "$O/run.log"
if [ -n "$GT" ]; then
  [ -s "$O/traj_tum.txt" ] && $P $REPO/eval/evaluate.py --gt $GT --est "$O/traj_tum.txt" --out "$O/metrics.json" --label "$ARM $NAME seed$SEED" > "$O/eval.log" 2>&1
  if [ -s "$O/trace.txt" ]; then
    $P $REPO/eval/trace_to_tum.py --trace "$O/trace.txt" --seq "$SEQ" --out "$O/online_tum.txt" >> "$O/eval.log" 2>&1
    [ -s "$O/online_tum.txt" ] && $P $REPO/eval/evaluate.py --gt $GT --est "$O/online_tum.txt" --out "$O/metrics_online.json" --label "online $ARM $NAME seed$SEED" >> "$O/eval.log" 2>&1
  fi
  if [ "$ARM" = alvaar ]; then PF=traj_seg; else PF=traj_map; fi
  ls "$O"/${PF}*_tum.txt > /dev/null 2>&1 && $P $REPO/eval/evaluate_multimap.py --gt $GT --run "$O" --out "$O/metrics_multimap.json" --prefix $PF >> "$O/eval.log" 2>&1
fi
echo "$(date +%T) end   $ARM $NAME seed$SEED rc=$rc" >> $R/logs/ext_1103_runs.log
