#!/usr/bin/env bash
# KLT tracking frames: native, real time, seeds 1-3; MH01 (and room2) with Web.KLT.Enable=1 vs stock.
# Usage: klt_test.sh <tag> <settings-variant> [sequences...]
set -uo pipefail
TAG=$1; VAR=$2; shift 2
SEQS=${*:-euroc/MH_01_easy}
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; V=/mnt/h/Research/results/settings_variants
O=/mnt/h/Research/results/klt_dev/$TAG
mkdir -p $O
for s in $SEQS; do
  ds=${s%%/*}; name=${s#*/}
  if [ $ds = euroc ]; then st=$V/EuRoC_mono_$VAR.yaml; gt=$G/euroc_$name.txt; x=""; else st=$V/TUM-VI_mono_$VAR.yaml; gt=$G/tumvi_$name.txt; x="--clahe"; fi
  for seed in 1 2 3; do
    o=$O/${name}_seed$seed
    /root/build/slam-native/orbslam3_run --dataset $ds --mode mono --seq $D/$ds/$name --vocab /mnt/h/Research/vocab/ORBvoc.bin \
      --settings $st $x --seed $seed --realtime --out $o > $o.log 2>&1 &
  done
  wait
  for seed in 1 2 3; do
    o=$O/${name}_seed$seed
    r=$(/opt/slam/venv/bin/python /mnt/h/Research/CustomSLAM/eval/evaluate.py --gt $gt --est $o/traj_tum.txt --out $o/metrics.json --label "$TAG $name s$seed" 2>&1 | tail -1)
    tt=$(awk '{print $2}' $o/track_times.txt | sort -n | awk '{a[NR]=$1} END {printf "track ms med %.1f p95 %.1f", a[int(NR/2)], a[int(NR*0.95)]}')
    echo "$r | $tt | $(grep -h 'KLT-STATS' $o.log | sed 's/KLT-STATS //')"
  done
done
