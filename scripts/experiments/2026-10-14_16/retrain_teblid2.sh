#!/usr/bin/env bash
# Retrain the TEBLID vocabulary with a vocab_train that has the DBoW2 empty-cluster fix (the 10:45 rebuild was
# OOM-killed: WSL has 7 GB). -j1, lowest priority. The "vocabularies done" marker is written only on success.
set -uo pipefail
LOG=/mnt/h/Research/logs/teblid.log
B=/root/build/slam-native-teblid
D=/mnt/h/Research/datasets; V=/mnt/h/Research/vocab
echo "=== retrain2 teblid $(date +%T): rebuild vocab_train -j1" >> $LOG
nice -n 19 cmake --build $B -j1 --target vocab_train 2>&1 | grep -E "error|FAILED" >> $LOG
if [ ! $B/vocab_train -nt /mnt/h/Research/CustomSLAM/slam/orbslam3/Thirdparty/DBoW2/DBoW2/TemplatedVocabulary.h ]; then
  echo "retrain2: vocab_train was not rebuilt, stopping" >> $LOG; exit 1
fi
echo "vocab_train built $(date +%T)" >> $LOG
DIRS="$(ls -d $D/2011_*_sync/image_02/data | tr '\n' ' ')"
for s in bike-easy loop-floor0 loop-floor1 office-maze running-easy skate-easy; do DIRS="$DIRS $D/$s/left_images"; done
nice -n 19 $B/vocab_train $V/TEBLIDvoc_own_k10L6.bin teblid 10 6 10 $DIRS > $LOG.teblid 2>&1
rc=$?
echo "teblid exit=$rc" >> $LOG
tail -2 $LOG.teblid >> $LOG
ls -la $V >> $LOG
if [ $rc -eq 0 ] && [ -s $V/TEBLIDvoc_own_k10L6.bin ]; then echo "=== vocabularies done $(date +%T)" >> $LOG; fi
