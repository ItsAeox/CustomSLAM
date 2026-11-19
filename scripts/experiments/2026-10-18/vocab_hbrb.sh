#!/usr/bin/env bash
# vocab_hbrb: does better vocabulary training make the small (k10 L5, 5 MB) vocabulary as accurate as the official one?
# own-L5 (2026-10-16) had the same robustness as the official ORBvoc but 1.27x its ATE (claude/EXPERIMENTS.md, vocab_eval2).
# Two new k10 L5 vocabularies, trained on the same non-evaluation images as own-L5 (vocab_train.sh: KITTI raw
# image_02 + TUM-VIE bike/loop-floor/office-maze/running/skate):
#   hbrb-L5   every 10th image (as own-L5), --hbrb: real-valued k-means centres, binarised at the end (vocab_train.cc)
#   own2x-L5  every 5th image (2x the descriptors), DBoW2's standard k-majority: control (is it only more data?)
# Evaluated exactly as vocab_eval2 (same pinned binary snap-kf1, rt-wall, EuRoC 11 + TUM-VI rooms 6, seeds 1-3,
# Web.BoW.LevelsUp 3), so the existing official and own-L5 runs are the paired baselines.
# Usage: vocab_hbrb.sh   (nothing else heavy at the same time: training holds ~1-2 GB of descriptors)
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
source /opt/slam/emsdk/emsdk_env.sh > /dev/null 2>&1
B=/root/build/slam-native-teblid; D=/mnt/h/Research/datasets; VV=/mnt/h/Research/vocab; P=/opt/slam/venv/bin/python
G=/mnt/h/Research/results/gt; V=/mnt/h/Research/results/settings_variants; O=/mnt/h/Research/results/vocab_eval
echo "=== build vocab_train $(date +%T)"
nice -n 10 cmake --build $B -j4 --target vocab_train 2>&1 | grep -E "error|FAILED" | head -20
DIRS="$(ls -d $D/2011_*_sync/image_02/data | tr '\n' ' ')"
for s in bike-easy loop-floor0 loop-floor1 office-maze running-easy skate-easy; do DIRS="$DIRS $D/$s/left_images"; done
[ -s $VV/ORBvoc_hbrb_k10L5.bin ] || { echo "=== train hbrb-L5 $(date +%T)"; nice -n 10 $B/vocab_train $VV/ORBvoc_hbrb_k10L5.bin orb 10 5 10 --hbrb $DIRS 2>&1 | grep -v "images$" | tail -3; }
[ -s $VV/ORBvoc_own2x_k10L5.bin ] || { echo "=== train own2x-L5 $(date +%T)"; nice -n 10 $B/vocab_train $VV/ORBvoc_own2x_k10L5.bin orb 10 5 5 $DIRS 2>&1 | grep -v "images$" | tail -3; }
ls -la $VV
[ -s $VV/ORBvoc_hbrb_k10L5.bin ] && [ -s $VV/ORBvoc_own2x_k10L5.bin ] || { echo "training failed: stop"; exit 1; }

export NATIVE=/root/build/snap-kf1/native/orbslam3_run
echo "=== eval start $(date +%T) binary $NATIVE"
for arm in hbrb-L5:ORBvoc_hbrb_k10L5.bin own2x-L5:ORBvoc_own2x_k10L5.bin; do
  IFS=: read -r name voc <<< "$arm"
  SPECS=()
  for s in $D/euroc/*; do n=$(basename $s); [ -s $O/$name/euroc_$n/native_seed3/track_times.txt ] && continue
    SPECS+=("euroc_$n|euroc|$s|$V/EuRoC_mono_bow-lu3.yaml|$G/euroc_$n.txt|"); done
  for r in 1 2 3 4 5 6; do n=dataset-room${r}_512_16; [ -s $O/$name/tumvi_dataset-room$r/native_seed3/track_times.txt ] && continue
    SPECS+=("tumvi_dataset-room$r|tumvi|$D/tumvi/$n|$V/TUM-VI_mono_bow-lu3.yaml|$G/tumvi_$n.txt|--clahe"); done
  [ ${#SPECS[@]} -gt 0 ] && VOCAB=$VV/$voc FLAVOURS=native JOBS=1 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $O/$name "1 2 3" "${SPECS[@]}" > /dev/null &
done
wait
echo "=== summaries $(date +%T)"
for a in official own-L5 hbrb-L5 own2x-L5; do
  bad=$($P eval/summarize_sweep.py $O/$a --bad-ate-mm 200 --bad-rot 1000 | awk '$2=="native"{b+=$5; n+=$3} END{print b "/" n}')
  echo "  $a: bad $bad, loops $(cat $O/$a/*/*/run.log | grep -c 'Loop detected'), merges $(cat $O/$a/*/*/run.log | grep -c 'Merge detected'), relocalised $(cat $O/$a/*/*/run.log | grep -c 'Relocalized'), track ms mean $(cat $O/$a/*/*/track_times.txt | awk '{s+=$2;n++} END{printf "%.2f", s/n}')"
  $P eval/summarize_sweep.py $O/$a --bad-ate-mm 200 --bad-rot 1000 | awk -v a=$a '$2=="native"{split($0,p,"|"); split(p[3],m," "); v[++n]=m[1]} END{asort(v); printf "  %-9s median-of-medians ATE %.1f mm over %d sequences\n", a, v[int((n+1)/2)], n}'
done
for a in own-L5 hbrb-L5 own2x-L5; do $P eval/compare_builds.py --across $O/official $O/$a --a native --bad-ate-mm 200 --bad-rot 1000 | tail -2; done
for a in hbrb-L5 own2x-L5; do $P eval/compare_builds.py --across $O/own-L5 $O/$a --a native --bad-ate-mm 200 --bad-rot 1000 | tail -2; done
echo "=== done $(date +%T)"
