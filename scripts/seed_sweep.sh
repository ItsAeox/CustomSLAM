#!/usr/bin/env bash
# Seed sweep: native vs WASM builds of the ORB-SLAM3 web port over several RANSAC seeds.
# Monocular initialisation is seed-sensitive, so parity must be judged on distributions over seeds.
#
# Usage: scripts/seed_sweep.sh <out_dir> <seeds e.g. "1 2 3 4 5"> <seq_name>:<settings.yaml>:<gt_tum.txt> [...]
# Env:   DATASETS (default ../datasets), VOCAB (default ../vocab/ORBvoc.bin), FLAVOURS (default "native wasm"),
#        EXTRA (extra driver args, e.g. "--realtime")
set -uo pipefail
OUT="$1"; SEEDS="$2"; shift 2
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATASETS="${DATASETS:-$REPO/../datasets}"
VOCAB="${VOCAB:-$REPO/../vocab/ORBvoc.bin}"
FLAVOURS="${FLAVOURS:-native wasm}"
NATIVE="${NATIVE:-$HOME/build/slam-native/orbslam3_run}"
WASM_JS="${WASM_JS:-$HOME/build/slam-wasm/orbslam3_run.js}"
NODE="${NODE:-/opt/slam/emsdk/node/24.19.0_64bit/bin/node}"
PY="${PY:-/opt/slam/venv/bin/python}"

for spec in "$@"; do
  IFS=: read -r SEQ SETTINGS GT <<< "$spec"
  for seed in $SEEDS; do
    for fl in $FLAVOURS; do
      O="$OUT/$SEQ/${fl}_seed$seed"
      mkdir -p "$O"
      if [ "$fl" = native ]; then CMD=("$NATIVE"); else CMD=("$NODE" "$WASM_JS"); fi
      "${CMD[@]}" --dataset tumvie --mode mono --seq "$DATASETS/$SEQ" --vocab "$VOCAB" --settings "$SETTINGS" \
        --out "$O" --clahe --seed "$seed" ${EXTRA:-} > "$O/run.log" 2>&1 || echo "run failed: $SEQ $fl seed $seed" | tee -a "$O/run.log"
      if [ -s "$O/traj_tum.txt" ]; then
        "$PY" "$REPO/eval/evaluate.py" --gt "$GT" --est "$O/traj_tum.txt" --out "$O/metrics.json" \
              --label "$SEQ ${fl} seed$seed" | tail -1
      else
        echo "$SEQ ${fl} seed$seed: NO TRAJECTORY (failure)"
      fi
    done
  done
done
