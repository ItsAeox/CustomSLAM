#!/usr/bin/env bash
# tracking ms of beta + window estimator vs beta + filter in one batch (snap-efa845e), 3 sequences x 2 seeds, live@1.0
cd /mnt/h/Research/CustomSLAM
export PH=/mnt/h/Research/datasets/synth SEEDS="1 2" RECS="gesture2 shake2 orbit2"
O=/mnt/h/Research/results/scale_cost_1030
bash scripts/experiments/2026-10-18/replay_arms.sh efa845e $O scripts/experiments/2026-10-30/arms_scale_cost.txt > $O.log 2>&1
for r in $RECS; do for a in beta_scale beta_scaleF; do for s in 1 2; do
  echo "$r $a seed$s: $(awk '{x+=$2; n++} END{printf "%.1f ms mean", x/n}' $O/$r/$a/seed$s/track_times.txt)"
done; done; done
