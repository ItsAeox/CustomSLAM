#!/usr/bin/env bash
# 29 Oct living-room recording with a known motion (datasets/phone/scale_recordings.txt): WebVIO beta + online scale,
# WASM live@1.0, seeds 1-3, per-frame trace (columns 26-31: online scale state); as 2026-10-21/scale_1021.sh
cd /mnt/h/Research/CustomSLAM
export TRACE=1 RECS="rec_20261029_202938" SEEDS="1 2 3"
bash scripts/experiments/2026-10-18/replay_arms.sh 99436ab /mnt/h/Research/results/scale_1029 scripts/experiments/2026-10-18/arms_scale.txt \
  > /mnt/h/Research/results/scale_1029.log 2>&1
