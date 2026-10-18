#!/usr/bin/env bash
# Phone recordings (datasets/phone/labels.txt) through settings variants made from a page preset plus extra lines, WASM
# runner, live@1.0, seeds 1-2, 4 at a time; then summary_presets.py. Each arm: name|preset|extra lines separated by ';'.
# Usage: [TRACE=1] replay_arms.sh <snapshot tag> <out dir> <arms file>   (one arm per line; # comments)
#   TRACE=1: per-frame trace (--trace) and eval/loss_anatomy.py per run (anatomy.txt in the run directory)
#   e.g. "fast_cov|fast|Web.KF.Coverage: 0.4"   "grecover_dt|grecover|Web.Gyro.TimeOffset: -0.088"
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
TAG=$1 O=$2 ARMSFILE=$3
NODE=/opt/slam/emsdk/node/24.19.0_64bit/bin/node; JS=/root/build/snap-$TAG/wasm/orbslam3_run.js
PH=/mnt/h/Research/datasets/phone; V=/mnt/h/Research/vocab
RECS=$(awk '!/^#/ && NF {print $1}' $PH/labels.txt)
mkdir -p $O
echo "=== start $(date +%T) binary $JS"
ARMS=()
while IFS= read -r spec; do
  [[ -z "$spec" || "$spec" == \#* ]] && continue
  IFS='|' read -r name preset extra <<< "$spec"; ARMS+=("$name")
  for r in $RECS; do
    mkdir -p $O/$r/settings
    [ -s $O/$r/settings/${r}_page-$preset.yaml ] || $NODE scripts/page_presets.mjs --base $PH/$r/settings.yaml --name $r $O/$r/settings $preset > /dev/null
    { cat $O/$r/settings/${r}_page-$preset.yaml; echo; echo "# replay_arms.sh: $name"; echo "$extra" | tr ';' '\n' | sed 's/^ *//'; } > $O/$r/settings/${r}_arm-$name.yaml
  done
done < $ARMSFILE
jobs_list=()
for r in $RECS; do for a in "${ARMS[@]}"; do for s in 1 2; do jobs_list+=("$r $a $s"); done; done; done
run_one() {
  local r=$1 a=$2 s=$3 o=$O/$1/$2/seed$3
  [ -s $o/run_info.txt ] && return
  mkdir -p $o
  local st=$O/$r/settings/${r}_arm-$a.yaml voc=$V/ORBvoc.bin g=""
  grep -q "Web.BoW.LevelsUp: 3" $st && voc=$V/ORBvoc_own_k10L5.bin
  grep -q "Web.Gyro.Prior: 1" $st && g="--gyro"
  timeout 1800 $NODE $JS --dataset euroc --mode mono --seq $PH/$r --vocab $voc --settings $st --seed $s --live --rate-scale 1.0 $g ${TRACE:+--trace} \
    --out $o > $o/run.log 2>&1 || echo "  $r $a seed $s: exit $?"
  if [ -n "${TRACE:-}" ]; then /opt/slam/venv/bin/python eval/loss_anatomy.py $o $PH/$r > $o/anatomy.txt 2>&1; fi
}
export -f run_one; export O PH V NODE JS TRACE
printf '%s\n' "${jobs_list[@]}" | xargs -P 4 -L 1 bash -c 'run_one $0 $1 $2'
echo "=== runs done $(date +%T)"
/opt/slam/venv/bin/python scripts/experiments/2026-10-18/summary_presets.py $O $PH/labels.txt
for a in "${ARMS[@]}"; do echo "$a: turn bridges $(grep -h BRIDGE-STATS $O/*/$a/*/run.log 2>/dev/null | sed 's/.*bridges=\([0-9]*\).*/\1/' | awk '{s+=$1; n++} END{printf "%.1f per run", n ? s/n : 0}')"; done
for a in "${ARMS[@]}"; do echo "$a: coverage keyframes $(grep -h COV-STATS $O/*/$a/*/run.log 2>/dev/null | sed 's/.*=//' | awk '{s+=$1; n++} END{printf "%.1f per run", n ? s/n : 0}')"; done
echo "=== done $(date +%T)"
