#!/usr/bin/env bash
# The pruned, stability-checked filter build (snap-efa845e) on the 7 screen sequences, both IMUs: cost per frame against
# synth_run's beta_scale on the same sequences, and the converged / error columns with the new reported 1-sigma
cd /mnt/h/Research/CustomSLAM
P="slide3 gesture2 shake2 rotate2 orbit2 walk2 ar2"
OUT=scaleF2_1030 RECS_A="$P" RECS_B="$(for r in $P; do echo ${r}__imuideal; done)" bash scripts/experiments/2026-10-30/scaleF_run.sh efa845e
