#!/usr/bin/env bash
# orb_split (2026-10-18): where the "fast+octree" ORB time goes (FAST corners vs keypoint spreading: octree or SSC).
# Profiling builds (REGISTER_TIMES, ORB-PROF line: spread = the octree / SSC share, candidates = FAST corners before
# spreading, kept = keypoints after, per frame), every frame extracted (each recording's stock settings, no --live), one
# run at a time so the timings are not disturbed. WASM under Node is closer to the phone than native x86 (OpenCV FAST
# gets more SIMD speed-up on x86 than the scalar octree). Usage: orb_split.sh <out dir> <recording...>
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
O=$1; shift
NODE=/opt/slam/emsdk/node/24.19.0_64bit/bin/node PH=/mnt/h/Research/datasets/phone V=/mnt/h/Research/vocab
mkdir -p $O
echo "=== start $(date +%T)"
for r in "$@"; do
  for arm in octree ssc; do
    for fl in wasm native; do
      o=$O/$r/$arm-$fl; mkdir -p $o
      st=$o/settings.yaml; cp $PH/$r/settings.yaml $st
      [ $arm = ssc ] && { echo; echo "Web.ORB.SSC: 1"; } >> $st
      if [ $fl = wasm ]; then RUN="$NODE /root/build/slam-wasm-prof/orbslam3_run.js"; else RUN=/root/build/slam-native-prof/orbslam3_run; fi
      $RUN --dataset euroc --mode mono --seq $PH/$r --vocab $V/ORBvoc.bin --settings $st --seed 1 --out $o > $o/run.log 2>&1
      echo "$r $arm $fl: $(grep -c 'SSC keypoint selection' $o/run.log) ssc-config; $(grep ORB-PROF $o/run.log | tail -1)"
    done
  done
done
echo "=== done $(date +%T)"
