#!/usr/bin/env bash
# One run of the main-table batch (main_1103.sh): <arm> <dataset> <sequence dir> <seed>.
# Arms: native-stock / wasm-stock = the port with the stock ORB-SLAM3 settings (no Web.* keys; parity), page-<preset> =
# the AR page's preset exactly as the page builds it (scripts/page_presets.mjs), alvaar = AlvaAR's released bundle.
# Scores the final trajectory (metrics.json), the online poses from --trace (metrics_online.json) and every map /
# segment on its own (metrics_multimap.json). A run still going after 1200 s gets a Node report (SIGUSR2) and is
# killed 30 s later (replay hangs G23 / G33).
set -uo pipefail
ARM="$1"; DS="$2"; SEQ="$3"; SEED="$4"
R=/mnt/h/Research; REPO=$R/CustomSLAM; O0=$R/results/main_1103
SNAP=/root/build/snap-84c1f17; NODE=/opt/slam/emsdk/node/24.19.0_64bit/bin/node; P=/opt/slam/venv/bin/python
S=$REPO/baselines/orbslam3/settings; V=$O0/settings
n=$(basename "$SEQ"); NAME=${DS}_$n
if [ "$DS" = euroc ]; then BASE=EuRoC_mono; GT=$R/results/gt/euroc_$n.txt; C=""; else BASE=TUM-VI_mono; GT=$R/results/gt/tumvi_$n.txt; C=--clahe; fi
case "$ARM" in
  native-stock) FL=native; ST=$S/$BASE.yaml; CMD=($SNAP/native/orbslam3_run) ;;
  wasm-stock)   FL=wasm;   ST=$S/$BASE.yaml; CMD=($NODE --report-on-signal $SNAP/wasm/orbslam3_run.js) ;;
  page-*)       FL=wasm;   ST=$V/${BASE}_$ARM.yaml; CMD=($NODE --report-on-signal $SNAP/wasm/orbslam3_run.js) ;;
  alvaar)       FL=wasm;   ST=$S/$BASE.yaml; CMD=($NODE $REPO/baselines/alvaar/alvaar_run.mjs) ;;
esac
O=$O0/$ARM/$NAME/${FL}_seed$SEED
mkdir -p "$O"
X=(--rt-wall)
if [ "$ARM" != alvaar ]; then
  X+=(--trace); [ -n "$C" ] && X+=($C)
  grep -qE "^Web\.(Gyro|Scale)\." "$ST" && X+=(--gyro)   # presets that read the IMU get the dataset's
  [ "$FL" = wasm ] && CMD=("${CMD[0]}" --report-directory="$O" "${CMD[@]:1}")
fi
echo "$(date +%T) start $ARM $NAME seed$SEED" >> $R/logs/main_1103_runs.log
timeout -k 30 -s USR2 1200 "${CMD[@]}" --dataset $DS --mode mono --seq "$SEQ" --vocab $R/vocab/ORBvoc.bin \
  --settings "$ST" --out "$O" --seed $SEED "${X[@]}" > "$O/run.log" 2>&1
rc=$?; [ $rc -ne 0 ] && echo "run failed (exit $rc)" >> "$O/run.log"
if [ -s "$O/traj_tum.txt" ]; then
  $P $REPO/eval/evaluate.py --gt $GT --est "$O/traj_tum.txt" --out "$O/metrics.json" --label "$ARM $NAME seed$SEED" > "$O/eval.log" 2>&1
fi
if [ -s "$O/trace.txt" ]; then
  $P $REPO/eval/trace_to_tum.py --trace "$O/trace.txt" --seq "$SEQ" --out "$O/online_tum.txt" >> "$O/eval.log" 2>&1
  [ -s "$O/online_tum.txt" ] && $P $REPO/eval/evaluate.py --gt $GT --est "$O/online_tum.txt" --out "$O/metrics_online.json" --label "online $ARM $NAME seed$SEED" >> "$O/eval.log" 2>&1
fi
if [ "$ARM" = alvaar ]; then PF=traj_seg; else PF=traj_map; fi
ls "$O"/${PF}*_tum.txt > /dev/null 2>&1 && $P $REPO/eval/evaluate_multimap.py --gt $GT --run "$O" --out "$O/metrics_multimap.json" --prefix $PF >> "$O/eval.log" 2>&1
echo "$(date +%T) end   $ARM $NAME seed$SEED rc=$rc $(tail -1 $O/eval.log 2>/dev/null | cut -c1-120)" >> $R/logs/main_1103_runs.log
