#!/usr/bin/env bash
# synthetic pilot (synth/README.md): WebVIO beta + online scale on the first synthetic sequences, WASM live@1.0,
# seeds 1-2, trace; then the online scale against the exact true scale per run (Sim3 of the tracked trajectory to GT)
cd /mnt/h/Research/CustomSLAM
export PH=/mnt/h/Research/datasets/synth TRACE=1 RECS="${RECS:-slide4 orbit3}" SEEDS="${SEEDS:-1 2}"
O=/mnt/h/Research/results/${OUT:-synth_pilot}
bash scripts/experiments/2026-10-18/replay_arms.sh 99436ab $O scripts/experiments/2026-10-18/arms_scale.txt > $O.log 2>&1
for r in $RECS; do for s in $SEEDS; do
  d=$O/$r/beta_scale/seed$s
  echo "$r seed$s: $(/opt/slam/venv/bin/python eval/scale_online_check.py $d $PH/$r --gt /mnt/h/Research/results/gt/synth_$r.txt 2>&1 | sed "s|$O/||")"
done; done
