#!/usr/bin/env bash
# One run of gyrobias_1103.sh: <arm> <EuRoC sequence name> <seed>. Arm = <settings>-<raw|cal>; scoring as main_job.sh.
set -uo pipefail
ARM="$1"; n="$2"; SEED="$3"
R=/mnt/h/Research; REPO=$R/CustomSLAM; O0=$R/results/gyrobias_1103
SNAP=/root/build/snap-84c1f17; NODE=/opt/slam/emsdk/node/24.19.0_64bit/bin/node; P=/opt/slam/venv/bin/python
ST=$O0/settings/${ARM%-*}.yaml; GT=$R/results/gt/euroc_$n.txt
case "$ARM" in *-cal) SEQ=/root/data/euroc_gyrocal/$n ;; *) SEQ=$R/datasets/euroc/$n ;; esac
O=$O0/$ARM/euroc_$n/wasm_seed$SEED
mkdir -p "$O"
echo "$(date +%T) start $ARM $n seed$SEED" >> $R/logs/gyrobias_1103_runs.log
timeout -k 30 -s USR2 1200 $NODE --report-on-signal --report-directory="$O" $SNAP/wasm/orbslam3_run.js --dataset euroc \
  --mode mono --seq "$SEQ" --vocab $R/vocab/ORBvoc.bin --settings "$ST" --out "$O" --seed $SEED --rt-wall --trace --gyro \
  > "$O/run.log" 2>&1
rc=$?; [ $rc -ne 0 ] && echo "run failed (exit $rc)" >> "$O/run.log"
[ -s "$O/traj_tum.txt" ] && $P $REPO/eval/evaluate.py --gt $GT --est "$O/traj_tum.txt" --out "$O/metrics.json" --label "$ARM $n seed$SEED" > "$O/eval.log" 2>&1
if [ -s "$O/trace.txt" ]; then
  $P $REPO/eval/trace_to_tum.py --trace "$O/trace.txt" --seq "$SEQ" --out "$O/online_tum.txt" >> "$O/eval.log" 2>&1
  [ -s "$O/online_tum.txt" ] && $P $REPO/eval/evaluate.py --gt $GT --est "$O/online_tum.txt" --out "$O/metrics_online.json" --label "online $ARM $n seed$SEED" >> "$O/eval.log" 2>&1
fi
ls "$O"/traj_map*_tum.txt > /dev/null 2>&1 && $P $REPO/eval/evaluate_multimap.py --gt $GT --run "$O" --out "$O/metrics_multimap.json" >> "$O/eval.log" 2>&1
echo "$(date +%T) end   $ARM $n seed$SEED rc=$rc" >> $R/logs/gyrobias_1103_runs.log
