#!/usr/bin/env bash
# Paper scope (user, 3 Nov evening): no version from the scale work on (first scale code 99436ab, 19 Oct 08:07). Do the
# kept arms (stock / fast / next / beta) of the frozen build snap-84c1f17 run as in the last pre-scale build
# snap-f1a4a97 (19 Oct 01:34)? The code diff says every later change is gated off for them; this runs both binaries
# the same deterministic way (native, unpaced, --sync-mapping, seed 1, main_1103's settings files, first 1500 frames:
# full sequences take ~18 min per run this way) and compares the final trajectories byte for byte. The frozen build is run twice (is the mode deterministic at all?).
# Results: results/equiv_1103/<build>/<arm>/<seq>/; summary from equiv_compare (below).
set -uo pipefail
R=/mnt/h/Research; REPO=$R/CustomSLAM; O0=$R/results/equiv_1103; D=$R/datasets; P=/opt/slam/venv/bin/python
S=$REPO/baselines/orbslam3/settings; V=$R/results/main_1103/settings
job() {  # <build label> <binary dir> <arm> <dataset> <seq dir>
  local B="$1" BIN="$2" ARM="$3" DS="$4" SEQ="$5" n BASE C ST GT O X
  n=$(basename "$SEQ")
  if [ "$DS" = euroc ]; then BASE=EuRoC_mono; GT=$R/results/gt/euroc_$n.txt; C=""; else BASE=TUM-VI_mono; GT=$R/results/gt/tumvi_$n.txt; C=--clahe; fi
  if [ "$ARM" = stock ]; then ST=$S/$BASE.yaml; else ST=$V/${BASE}_page-$ARM.yaml; fi
  O=$O0/$B/$ARM/${DS}_$n; mkdir -p "$O"
  X=(--sync-mapping --seed 1 --max-frames 1500); [ -n "$C" ] && X+=($C)
  grep -qE "^Web\.(Gyro|Scale)\." "$ST" && X+=(--gyro)
  timeout 1800 $BIN/orbslam3_run --dataset $DS --mode mono --seq "$SEQ" --vocab $R/vocab/ORBvoc.bin --settings "$ST" \
    --out "$O" "${X[@]}" > "$O/run.log" 2>&1
  echo "rc=$?" >> "$O/run.log"
  [ -s "$O/traj_tum.txt" ] && $P $REPO/eval/evaluate.py --gt $GT --est "$O/traj_tum.txt" --out "$O/metrics.json" --label "$B $ARM $n" > "$O/eval.log" 2>&1
  echo "$(date +%T) $B $ARM $n done" >> $R/logs/equiv_1103.log
}
export -f job; export R REPO O0 D P S V
echo "=== start $(date +%T)" >> $R/logs/equiv_1103.log
for a in stock fast next beta; do
  for q in "euroc $D/euroc/MH_01_easy" "euroc $D/euroc/V1_02_medium" "tumvi $D/tumvi/dataset-room2_512_16"; do
    echo "pre /root/build/snap-f1a4a97/native $a $q"
    echo "frozen /root/build/snap-84c1f17/native $a $q"
    echo "frozen2 /root/build/snap-84c1f17/native $a $q"
  done
done | xargs -P3 -L1 bash -c 'job "$@"' _
echo "=== done $(date +%T)" >> $R/logs/equiv_1103.log
