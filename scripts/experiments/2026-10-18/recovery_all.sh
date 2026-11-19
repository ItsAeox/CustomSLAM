#!/usr/bin/env bash
# PLAN 4d b1: eval/recovery_anatomy.py on every recording of a replay batch (one arm, both seeds) -> <out>
# Usage: recovery_all.sh <batch dir> <arm> <out file>
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
B=$1 A=$2 OUT=$3 PH=/mnt/h/Research/datasets/phone
: > "$OUT"
for r in $(awk '!/^#/ && NF {print $1}' $PH/labels.txt); do
  for s in 1 2; do
    d=$B/$r/$A/seed$s
    [ -s $d/trace.txt ] || continue
    echo "== $r seed $s ($(awk -v r=$r '$1 == r {print $2}' $PH/labels.txt))" >> "$OUT"
    /opt/slam/venv/bin/python eval/recovery_anatomy.py $d $PH/$r >> "$OUT" 2>&1
  done
done
grep -c "^  loss" "$OUT"
