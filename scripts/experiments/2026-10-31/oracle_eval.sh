#!/usr/bin/env bash
# anchor_return.py --anchor-oracle over a batch: anchor_oracle.txt per run (the best fixed anchor of the placement map)
# Usage: oracle_eval.sh <batch dir> [jobs]
O=$1; J=${2:-8}; D=/mnt/h/Research/datasets/synth
cd /mnt/h/Research/CustomSLAM
ls -d $O/*/*/seed* | xargs -P $J -I{} bash -c 'r={}; q=$(basename $(dirname $(dirname $r))); /opt/slam/venv/bin/python eval/anchor_return.py $r '$D'/$q --gt /mnt/h/Research/results/gt/synth_$q.txt --anchor-oracle > $r/anchor_oracle.txt 2>&1'
