#!/usr/bin/env bash
# Run the native VIO runner on dataset sequences and evaluate against ground truth.
#
# Usage: scripts/run_vio_eval.sh kitti|tumvie <seq_dir> [<seq_dir> ...]
#
# Results go to $RESULTS_ROOT/<commit>/<dataset>/<sequence>/ :
#   traj_tum.txt legacy_poses.csv perf.csv run_info.txt   (runner)
#   gt_tum.txt                                            (ground truth)
#   metrics.json traj.png                                 (evaluation)
# <commit> is the short hash of HEAD, suffixed with "-dirty" if the working tree has uncommitted changes,
# so every number can be traced to a source state.
set -euo pipefail

DATASET="$1"; shift
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO/../results}"
BUILD_DIR="${BUILD_DIR:-$HOME/build/webar-vio-native}"
OPENCV_DIR="${OPENCV_DIR:-/opt/slam/opencv-4.10.0-native/lib/cmake/opencv4}"
PY="${PY:-/opt/slam/venv/bin/python}"

COMMIT="$(git -C "$REPO" rev-parse --short HEAD)"
git -C "$REPO" diff --quiet HEAD -- webar-vio eval scripts || COMMIT="$COMMIT-dirty"

cmake -S "$REPO/webar-vio" -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="$OPENCV_DIR" >/dev/null
cmake --build "$BUILD_DIR" --target vio_run

for SEQ in "$@"; do
  SEQ="$(cd "$SEQ" && pwd)"
  NAME="$(basename "$SEQ")"
  OUT="$RESULTS_ROOT/$COMMIT/$DATASET/$NAME"
  mkdir -p "$OUT"
  echo "=== $DATASET / $NAME -> $OUT"
  "$BUILD_DIR/vio_run" --dataset "$DATASET" --seq "$SEQ" --out "$OUT"
  if [ "$DATASET" = kitti ]; then
    "$PY" "$REPO/eval/gt_kitti_raw.py" --seq "$SEQ" --out "$OUT/gt_tum.txt"
  else
    "$PY" "$REPO/eval/gt_tumvie.py" --seq "$SEQ" --out "$OUT/gt_tum.txt"
  fi
  "$PY" "$REPO/eval/evaluate.py" --gt "$OUT/gt_tum.txt" --est "$OUT/traj_tum.txt" \
        --out "$OUT/metrics.json" --plot "$OUT/traj.png" --label "$NAME"
done
