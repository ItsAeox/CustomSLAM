#!/usr/bin/env bash
# map_reuse: room map save and reload (WebMapIO.cc; orbslam3_run --save-map / --load-map). Session A maps one sequence
# and saves its largest map; session B loads it and plays another sequence of the same place, relocalising into it.
# Pairs (ground truths in one world frame): EuRoC MH01 -> MH02, MH01 -> MH03, V1_01 -> V1_02, V1_01 -> V1_03;
# TUM-VI room1 -> room2, room1 -> room4. Arms per pair and seed (1-3), native, rt-wall, preset-fast (the phone's
# "Hybrid, fastest") and stock:
#   A       session A, map saved
#   Bload   session B with A's map loaded (Web.Map.LoadWait 20 s, then a new map)
#   Bfresh  session B alone (the usual start)
# Scored by eval/map_reuse.py: B's frames with a pose, ATE of B (own Sim3 fit) and B in A's frame (A's fit applied to B:
# small only if B really tracked in A's coordinates, which keeps AR anchors), time to the first relocalisation,
# file size, save / load time.
# Usage: map_reuse.sh <snapshot tag>
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
P=/opt/slam/venv/bin/python; D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt
V=/mnt/h/Research/results/settings_variants; VOC=/mnt/h/Research/vocab/ORBvoc.bin
B=/root/build/snap-$1/native/orbslam3_run; O=/mnt/h/Research/results/map_reuse
grep -c "load-map" $B > /dev/null || { echo "binary lacks --load-map: stop"; exit 1; }
echo "=== start $(date +%T) binary $B"
PAIRS="euroc:MH_01_easy:MH_02_easy euroc:MH_01_easy:MH_03_medium euroc:V1_01_easy:V1_02_medium euroc:V1_01_easy:V1_03_difficult
tumvi:dataset-room1_512_16:dataset-room2_512_16 tumvi:dataset-room1_512_16:dataset-room4_512_16"
spec() {  # dataset name variant -> "seq settings gt clahe"
  if [ $1 = euroc ]; then echo "$D/euroc/$2 $V/EuRoC_mono_$3.yaml $G/euroc_$2.txt"
  else echo "$D/tumvi/$2 $V/TUM-VI_mono_$3.yaml $G/tumvi_$2.txt --clahe"; fi
}
run() {  # out dataset name variant seed [extra]
  local o=$1 ds=$2 n=$3 var=$4 s=$5; shift 5
  read -r seq st gt cl <<< "$(spec $ds $n $var)"
  mkdir -p $o
  [ -s $o/track_times.txt ] && return
  $B --dataset $ds --mode mono --seq $seq --vocab $VOC --settings $st $cl --seed $s --rt-wall --out $o "$@" > $o/run.log 2>&1
}
for var in preset-fast stock; do
  # sessions A (one per first sequence and seed), then B with and without the map, 6 runs at a time
  for s in 1 2 3; do
    for a in euroc:MH_01_easy euroc:V1_01_easy tumvi:dataset-room1_512_16; do IFS=: read -r ds n <<< "$a"
      run $O/$var/A_${n}/s$s $ds $n $var $s --save-map $O/$var/A_${n}/s$s/map.bin &
    done
  done
  wait
  echo "  $var sessions A done $(date +%T)"
  for p in $PAIRS; do IFS=: read -r ds a b <<< "$p"
    for s in 1 2 3; do
      run $O/$var/${a}__${b}/s$s/Bload $ds $b $var $s --load-map $O/$var/A_${a}/s$s/map.bin &
      run $O/$var/${a}__${b}/s$s/Bfresh $ds $b $var $s &
    done
    wait
  done
  echo "  $var sessions B done $(date +%T)"
done
echo "=== results"
for var in preset-fast stock; do
  echo "--- $var"
  for p in $PAIRS; do IFS=: read -r ds a b <<< "$p"
    read -r _ _ gA _ <<< "$(spec $ds $a $var)"; read -r _ _ gB _ <<< "$(spec $ds $b $var)"
    for s in 1 2 3; do
      A=$O/$var/A_${a}/s$s; L=$O/$var/${a}__${b}/s$s/Bload; F=$O/$var/${a}__${b}/s$s/Bfresh
      nf=$(wc -l < $L/track_times.txt 2>/dev/null || echo 0)
      echo "  $($P eval/map_reuse.py --a-traj $A/traj_tum.txt --a-gt $gA --b-traj $L/traj_tum.txt --b-gt $gB --b-frames $nf --name "$a->$b s$s load ")" \
           "| reloc after $(grep -h 'WEBMAP-STATS' $L/run.log | sed 's/.*reloc_after_s=//') s | maps $(grep -h '^maps' $L/run_info.txt | awk '{print $2}')"
      echo "  $($P eval/map_reuse.py --a-traj $A/traj_tum.txt --a-gt $gA --b-traj $F/traj_tum.txt --b-gt $gB --b-frames $nf --name "$a->$b s$s fresh")"
    done
  done
  grep -h "WEBMAP saved" $O/$var/A_*/s*/run.log | sed 's/^/  /'
  grep -h "WEBMAP loaded" $O/$var/*__*/s*/Bload/run.log | sed 's/^/  /' | head -6
done
echo "=== done $(date +%T)"
