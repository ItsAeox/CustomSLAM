#!/usr/bin/env bash
# TEBLID experiment (queued): build native (separate build dir), then train ORB + TEBLID vocabularies on the same
# non-evaluation images (KITTI raw image_02 + TUM-VIE non-desk sequences, every 10th frame, no CLAHE, k=10 L=6 like
# ORBvoc.txt). Build and training run at the lowest CPU priority (the follow-up sweep is running meanwhile).
set -uo pipefail
LOG=/mnt/h/Research/logs/teblid.log
until grep -q "=== done" /mnt/h/Research/logs/klt_sweep.log 2>/dev/null; do sleep 60; done
echo "=== build $(date +%T)" >> $LOG
B=/root/build/slam-native-teblid
if [ ! -f $B/build.ninja ]; then
  cmake -S /mnt/h/Research/CustomSLAM/slam -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DOpenCV_DIR=/opt/slam/opencv-4.10.0-native/lib/cmake/opencv4 >> $LOG 2>&1
fi
nice -n 19 cmake --build $B -j2 --target orbslam3_run vocab_train 2>&1 | grep -E "error|FAILED|Linking" >> $LOG
ls -la $B/orbslam3_run $B/vocab_train >> $LOG 2>&1 || { echo "BUILD FAILED" >> $LOG; exit 1; }

echo "=== train vocabularies $(date +%T)" >> $LOG
D=/mnt/h/Research/datasets; V=/mnt/h/Research/vocab
DIRS="$(ls -d $D/2011_*_sync/image_02/data | tr '\n' ' ')"
for s in bike-easy loop-floor0 loop-floor1 office-maze running-easy skate-easy; do DIRS="$DIRS $D/$s/left_images"; done
nice -n 19 $B/vocab_train $V/ORBvoc_own_k10L6.bin orb 10 6 10 $DIRS > $LOG.orb 2>&1 &
nice -n 19 $B/vocab_train $V/TEBLIDvoc_own_k10L6.bin teblid 10 6 10 $DIRS > $LOG.teblid 2>&1 &
wait
tail -4 $LOG.orb $LOG.teblid >> $LOG
ls -la $V >> $LOG
echo "=== vocabularies done $(date +%T)" >> $LOG
