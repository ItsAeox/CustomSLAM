#!/usr/bin/env bash
# Phone recordings of 2026-10-18 (datasets/phone/labels.txt) through every AR preset, as the page builds them: WASM
# runner (Node), phone emulation (--live --rate-scale 0.5), seeds 1-2, 6 runs at a time; then eval/recording_metrics.py
# per recording and one summary table per preset over all recordings (summary_presets.py). Skips finished runs.
# The revisit pair: the "fast" preset maps recording A (--save-map), then recording B loads it (--load-map).
# Usage: replay_all.sh <snapshot tag>
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
TAG=$1; NODE=/opt/slam/emsdk/node/24.19.0_64bit/bin/node; JS=/root/build/snap-$TAG/wasm/orbslam3_run.js
PH=/mnt/h/Research/datasets/phone; O=/mnt/h/Research/results/replays_1018; V=/mnt/h/Research/vocab
PRESETS="stock hybrid fast fastq fastqp gstart grecover small expo lean next fastflow rotate light"
RECS=$(awk '!/^#/ && NF {print $1}' $PH/labels.txt)
mkdir -p $O
echo "=== start $(date +%T) binary $JS"
for r in $RECS; do  # settings per preset from the recording's own settings.yaml (camera + phone IMU block)
  mkdir -p $O/$r/settings
  $NODE scripts/page_presets.mjs --base $PH/$r/settings.yaml --name $r $O/$r/settings $PRESETS
done
jobs_list=()
for r in $RECS; do for p in $PRESETS; do for s in 1 2; do jobs_list+=("$r $p $s"); done; done; done
run_one() {
  local r=$1 p=$2 s=$3 o=$O/$1/$2/seed$3
  [ -s $o/track_times.txt ] && return
  mkdir -p $o
  local st=$O/$r/settings/${r}_page-$p.yaml voc=$V/ORBvoc.bin g=""
  grep -q "Web.BoW.LevelsUp: 3" $st && voc=$V/ORBvoc_own_k10L5.bin
  grep -q "Web.Gyro.Prior: 1" $st && g="--gyro"
  timeout 1800 $NODE $JS --dataset euroc --mode mono --seq $PH/$r --vocab $voc --settings $st --seed $s --live --rate-scale 0.5 $g \
    --out $o > $o/run.log 2>&1 || echo "  $r $p seed $s: exit $?"
}
export -f run_one; export O PH V NODE JS
printf '%s\n' "${jobs_list[@]}" | xargs -P 6 -L 1 bash -c 'run_one $0 $1 $2'
echo "=== presets done $(date +%T)"
# revisit pair: map A, save; B loads it (fast preset, seeds 1-2)
A=$(awk '$2=="revisit-A"{print $1}' $PH/labels.txt); B=$(awk '$2=="revisit-B"{print $1}' $PH/labels.txt)
for s in 1 2; do
  oa=$O/revisit/A_seed$s; ob=$O/revisit/B_load_seed$s; mkdir -p $oa $ob
  [ -s $oa/map.bin ] || $NODE $JS --dataset euroc --mode mono --seq $PH/$A --vocab $V/ORBvoc.bin --settings $O/$A/settings/${A}_page-fast.yaml \
    --seed $s --live --rate-scale 0.5 --out $oa --save-map $oa/map.bin > $oa/run.log 2>&1
  [ -s $ob/track_times.txt ] || $NODE $JS --dataset euroc --mode mono --seq $PH/$B --vocab $V/ORBvoc.bin --settings $O/$B/settings/${B}_page-fast.yaml \
    --seed $s --live --rate-scale 0.5 --out $ob --load-map $oa/map.bin > $ob/run.log 2>&1
  echo "  revisit seed $s: $(grep -h 'WEBMAP' $oa/run.log $ob/run.log | tr '\n' ' ' | cut -c1-300)"
done
for r in $RECS; do echo "--- $r ($(awk -v r=$r '$1==r{print $2, $3, $4, $5, $6}' $PH/labels.txt))"; /opt/slam/venv/bin/python eval/recording_metrics.py $O/$r | tail -n +2; done
/opt/slam/venv/bin/python scripts/experiments/2026-10-18/summary_presets.py $O $PH/labels.txt
echo "=== done $(date +%T)"
