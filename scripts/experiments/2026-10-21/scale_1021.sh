#!/usr/bin/env bash
# 21 Oct phone recordings with a known motion (claude/PHONE.md "Scale recordings 21 Oct"): WebVIO beta + online scale,
# WASM live@1.0, seeds 1-3, per-frame trace (columns 26-31: online scale state)
cd /mnt/h/Research/CustomSLAM
export TRACE=1 RECS="rec_20261021_070221 rec_20261021_070444" SEEDS="1 2 3"
bash scripts/experiments/2026-10-18/replay_arms.sh 99436ab /mnt/h/Research/results/scale_1021 scripts/experiments/2026-10-18/arms_scale.txt \
  > /mnt/h/Research/results/scale_1021.log 2>&1
