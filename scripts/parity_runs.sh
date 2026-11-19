#!/usr/bin/env bash
# Repeat native vs WASM runs of the ORB-SLAM3 web port on one sequence (ORB-SLAM3 is multi-threaded and
# therefore non-deterministic: compare distributions, not single runs).
#
# Usage: scripts/parity_runs.sh <dataset> <seq_dir> <settings.yaml> <gt_tum.txt> <out_dir> [runs=5] [mode=mono] [extra driver args]
set -euo pipefail
DATASET="$1"; SEQ="$2"; SETTINGS="$3"; GT="$4"; OUT="$5"; RUNS="${6:-5}"; MODE="${7:-mono}"; shift 7 || true
EXTRA=("$@")
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VOCAB="${VOCAB:-$REPO/../third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt}"
NATIVE="${NATIVE:-$HOME/build/slam-native/orbslam3_run}"
WASM_JS="${WASM_JS:-$HOME/build/slam-wasm/orbslam3_run.js}"
NODE="${NODE:-/opt/slam/emsdk/node/24.19.0_64bit/bin/node}"
PY="${PY:-/opt/slam/venv/bin/python}"

for i in $(seq 1 "$RUNS"); do
  for flavour in native wasm; do
    O="$OUT/${flavour}_$i"
    mkdir -p "$O"
    if [ "$flavour" = native ]; then
      CMD=("$NATIVE")
    else
      CMD=("$NODE" "$WASM_JS")
    fi
    "${CMD[@]}" --dataset "$DATASET" --mode "$MODE" --seq "$SEQ" --vocab "$VOCAB" --settings "$SETTINGS" \
      --out "$O" "${EXTRA[@]}" > "$O/run.log" 2>&1 || echo "run failed: $flavour $i (exit $?)" | tee -a "$O/run.log"
    if [ -s "$O/traj_tum.txt" ]; then
      "$PY" "$REPO/eval/evaluate.py" --gt "$GT" --est "$O/traj_tum.txt" --out "$O/metrics.json" --label "${flavour}_$i"
    fi
  done
done
