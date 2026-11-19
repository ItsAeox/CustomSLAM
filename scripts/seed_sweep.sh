#!/usr/bin/env bash
# Seed sweep: ORB-SLAM3 (web port) builds over several RANSAC seeds and sequences.
# Monocular initialisation is seed-sensitive, so results must be judged on distributions over seeds.
#
# Usage: scripts/seed_sweep.sh <out_dir> "<seeds>" "<spec>" ["<spec>" ...]
#   spec = name|dataset|seq_dir|settings.yaml|gt_tum.txt|extra driver args (e.g. --clahe)
#   (legacy spec "name:settings:gt" = TUM-VIE sequence under $DATASETS with --clahe)
# Env:   FLAVOURS ("native wasm"), EXTRA (args for every run, e.g. "--realtime"), JOBS (parallel runs, default 1),
#        NATIVE, WASM_JS, NODE, PY, VOCAB, DATASETS
set -uo pipefail
OUT="$1"; SEEDS="$2"; shift 2
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATASETS="${DATASETS:-$REPO/../datasets}"
VOCAB="${VOCAB:-$REPO/../vocab/ORBvoc.bin}"
FLAVOURS="${FLAVOURS:-native wasm}"
JOBS="${JOBS:-1}"
NATIVE="${NATIVE:-$HOME/build/slam-native/orbslam3_run}"
WASM_JS="${WASM_JS:-$HOME/build/slam-wasm/orbslam3_run.js}"
NODE="${NODE:-/opt/slam/emsdk/node/24.19.0_64bit/bin/node}"
PY="${PY:-/opt/slam/venv/bin/python}"

run_one() {  # name dataset seq settings gt extra seed flavour
  local NAME="$1" DS="$2" SEQ="$3" SETTINGS="$4" GT="$5" XARGS="$6" seed="$7" fl="$8"
  local O="$OUT/$NAME/${fl}_seed$seed"
  mkdir -p "$O"
  local CMD
  if [ "$fl" = native ]; then CMD=("$NATIVE"); else CMD=("$NODE" "$WASM_JS"); fi
  # shellcheck disable=SC2086
  "${CMD[@]}" --dataset "$DS" --mode mono --seq "$SEQ" --vocab "$VOCAB" --settings "$SETTINGS" \
    --out "$O" --seed "$seed" $XARGS ${EXTRA:-} > "$O/run.log" 2>&1 || echo "run failed (exit $?)" >> "$O/run.log"
  if [ -s "$O/traj_tum.txt" ]; then
    "$PY" "$REPO/eval/evaluate.py" --gt "$GT" --est "$O/traj_tum.txt" --out "$O/metrics.json" \
          --label "$NAME ${fl} seed$seed" | tail -1
  else
    echo "$NAME ${fl} seed$seed: NO TRAJECTORY (failure)"
  fi
}

for spec in "$@"; do
  if [[ "$spec" == *"|"* ]]; then
    IFS='|' read -r NAME DS SEQ SETTINGS GT XARGS <<< "$spec"
  else  # legacy TUM-VIE spec
    IFS=: read -r NAME SETTINGS GT <<< "$spec"
    DS=tumvie; SEQ="$DATASETS/$NAME"; XARGS="--clahe"
  fi
  for seed in $SEEDS; do
    for fl in $FLAVOURS; do
      run_one "$NAME" "$DS" "$SEQ" "$SETTINGS" "$GT" "${XARGS:-}" "$seed" "$fl" &
      while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 1; done
    done
  done
done
wait
