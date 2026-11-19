#!/usr/bin/env bash
# PLAN 4d b2: Web.Scale.Online on datasets with ground truth. WebVIO beta preset (+ online scale), native, seed 1,
# phone emulation (--live --rate-scale 0.5), --gyro --trace; then eval/scale_online_check.py per run against the GT.
# Usage: scale_online.sh <native binary> <out dir> [seeds]
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
B=$1 O=$2 SEEDS=${3:-1}
D=/mnt/h/Research/datasets G=/mnt/h/Research/results/gt V=/mnt/h/Research/results/settings_variants VOC=/mnt/h/Research/vocab/ORBvoc.bin
mkdir -p $O
jobs=()
for n in MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium V2_03_difficult; do
  for s in $SEEDS; do jobs+=("euroc_$n euroc $D/euroc/$n $V/EuRoC_mono_page-beta-scale.yaml $G/euroc_$n.txt $s -"); done
done
for r in 1 2 3 4 5 6; do
  n=dataset-room${r}_512_16
  for s in $SEEDS; do jobs+=("tumvi_room$r tumvi $D/tumvi/$n $V/TUM-VI_mono_page-beta-scale.yaml $G/tumvi_$n.txt $s --clahe"); done
done
run_one() {
  local name=$1 ds=$2 seq=$3 st=$4 gt=$5 s=$6 extra=$7 o=$O/$1/seed$6
  [ -s $o/trace.txt ] && return
  mkdir -p $o
  [ "$extra" = "-" ] && extra=""
  timeout 1800 $B --dataset $ds --mode mono --seq $seq --vocab $VOC --settings $st --seed $s --live --rate-scale 0.5 --gyro --trace $extra \
    --out $o > $o/run.log 2>&1 || echo "  $name seed $s: exit $?"
}
export -f run_one; export O B VOC
printf '%s\n' "${jobs[@]}" | xargs -P 4 -L 1 bash -c 'run_one $0 $1 $2 $3 $4 $5 $6'
for j in "${jobs[@]}"; do
  set -- $j
  echo "$(/opt/slam/venv/bin/python eval/scale_online_check.py $O/$1/seed$6 $3 --gt $5 2>&1 | sed "s|$O/||")"
  grep -h "SCALE-STATS" $O/$1/seed$6/run.log | sed 's/^/    /'
done
