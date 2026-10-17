#!/usr/bin/env bash
# Optical flow on 4 threads (Web.KLT.Parallel), clean tracking-time bench (nothing else running): one run at a time, --rt-wall,
# first 1500 frames of MH01 and room1, seed 1. Arms: preset-fast, fast-kltpar (LK on 4 threads), fast-kltparc (serial and
# 4 threads on the same points each frame: mismatches + both times), fast-speed (parallel ORB + fast pose), fast-speed2 (+ kltpar).
# Usage: speed_bench_klt.sh <snapshot tag> <flavour: native|wasm>
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; V=/mnt/h/Research/results/settings_variants; VOC=/mnt/h/Research/vocab/ORBvoc.bin
S=/root/build/snap-$1; FL=$2; O=/mnt/h/Research/results/speed_bench_klt/$FL
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
  grep -h "KLTPAR-STATS" $O/$a/$n/run.log | sed "s/^/    /"
}
echo "=== start $(date +%T) $FL $S"
for a in ${ARMS:-preset-fast fast-kltpar fast-kltparc fast-speed fast-speed2}; do
  run $a mh01 euroc $D/euroc/MH_01_easy $V/EuRoC_mono_$a.yaml
  run $a room1 tumvi $D/tumvi/dataset-room1_512_16 $V/TUM-VI_mono_$a.yaml --clahe
done
echo "=== done $(date +%T)"
