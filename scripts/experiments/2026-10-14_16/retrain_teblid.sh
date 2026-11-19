#!/usr/bin/env bash
# Rebuild vocab_train with the DBoW2 empty-cluster fix and train the TEBLID vocabulary (same images/params as ORB).
set -uo pipefail
LOG=/mnt/h/Research/logs/teblid.log
B=/root/build/slam-native-teblid
echo "=== retrain teblid (empty-cluster fix) $(date +%T)" >> $LOG
nice -n 19 cmake --build $B -j2 --target vocab_train orbslam3_run 2>&1 | grep -E "error|FAILED" >> $LOG
D=/mnt/h/Research/datasets; V=/mnt/h/Research/vocab
DIRS="$(ls -d $D/2011_*_sync/image_02/data | tr '\n' ' ')"
for s in bike-easy loop-floor0 loop-floor1 office-maze running-easy skate-easy; do DIRS="$DIRS $D/$s/left_images"; done
nice -n 19 $B/vocab_train $V/TEBLIDvoc_own_k10L6.bin teblid 10 6 10 $DIRS > $LOG.teblid 2>&1
echo "teblid exit=$?" >> $LOG
tail -2 $LOG.teblid >> $LOG
ls -la $V >> $LOG
echo "=== vocabularies done $(date +%T)" >> $LOG
