#!/usr/bin/env bash
# Clean tracking-time bench (nothing else running): one run at a time, --rt-wall, first 1500 frames of MH01 (pinhole) and
# room1 (fisheye), seed 1. Arms: preset-fast vs the speed candidates (g2o-free pose solver, warm rounds, parallel ORB).
# Usage: speed_bench.sh <snapshot tag> <flavour: native|wasm>
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; V=/mnt/h/Research/results/settings_variants; VOC=/mnt/h/Research/vocab/ORBvoc.bin
S=/root/build/snap-$1; FL=$2; O=/mnt/h/Research/results/speed_bench/$FL
NODE=/opt/slam/emsdk/node/24.19.0_64bit/bin/node
run() {  # arm name dataset seq settings extra
  local a=$1 n=$2 ds=$3 seq=$4 st=$5; shift 5
  mkdir -p $O/$a/$n
  if [ $FL = native ]; then B=($S/native/orbslam3_run); else B=($NODE $S/wasm/orbslam3_run.js); fi
  "${B[@]}" --dataset $ds --mode mono --seq $seq --vocab $VOC --settings $st --seed 1 --rt-wall --max-frames 1500 "$@" \
    --out $O/$a/$n > $O/$a/$n/run.log 2>&1
  awk -v a=$a -v n=$n '{v[NR]=$2; s+=$2} END{asort(v); printf "%-16s %-6s mean %6.2f  median %6.2f  p95 %6.2f ms\n", a, n, s/NR, v[int(NR/2)], v[int(NR*0.95)]}' $O/$a/$n/track_times.txt
  grep -h "KLT-STATS" $O/$a/$n/run.log | awk '{print "    " $5, $6, $7, $9}'
  grep -h "FASTPOSE-STATS" $O/$a/$n/run.log | awk '{print "    " $2, $3}'
}
echo "=== start $(date +%T) $FL $S"
for a in preset-fast fast-fastpose fast-fastposew fast-orbpar fast-speed; do
  run $a mh01 euroc $D/euroc/MH_01_easy $V/EuRoC_mono_$a.yaml
  run $a room1 tumvi $D/tumvi/dataset-room1_512_16 $V/TUM-VI_mono_$a.yaml --clahe
done
echo "=== done $(date +%T)"
