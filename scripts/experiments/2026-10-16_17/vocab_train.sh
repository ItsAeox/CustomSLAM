#!/usr/bin/env bash
# Smaller ORB vocabularies (claude/ISSUES.md G6: the 48.7 MB download): k=10, L=5 (~100k words) and L=4 (~10k words),
# trained like ORBvoc_own_k10L6 on non-evaluation images (KITTI raw image_02, TUM-VIE bike/loop-floor/office-maze/
# running/skate; every 10th image). vocab_train: slam/port/tools/vocab_train.cc.
set -uo pipefail
B=/root/build/slam-native-teblid; D=/mnt/h/Research/datasets; V=/mnt/h/Research/vocab
DIRS="$(ls -d $D/2011_*_sync/image_02/data | tr '\n' ' ')"
for s in bike-easy loop-floor0 loop-floor1 office-maze running-easy skate-easy; do DIRS="$DIRS $D/$s/left_images"; done
for L in 5 4; do
  echo "=== train k10 L$L $(date +%T)"
  nice -n 10 $B/vocab_train $V/ORBvoc_own_k10L$L.bin orb 10 $L 10 $DIRS 2>&1 | tail -3
done
ls -la $V
echo "=== vocab_train done $(date +%T)"
