#!/usr/bin/env bash
# vocab_eval2: does a smaller vocabulary cost accuracy or robustness? (G6: the 48.7 MB download). BoW is used in tracking
# (reference-keyframe matching), relocalisation, loop closing and map merging. Each vocabulary indexes BoW at the same
# depth (Web.BoW.LevelsUp = L - 2; stock 4 for L = 6), so size is not confounded with SearchByBoW cost.
# Arms side by side, native (pinned snapshot), rt-wall, EuRoC 11 + TUM-VI rooms 6, seeds 1-3:
#   official (ORBvoc k10L6, stock) | own-L6 (stock) | own-L5 (bow-lu3) | own-L4 (bow-lu2)
# Usage: vocab_eval2.sh <snapshot tag>   (needs the Web.BoW.LevelsUp build)
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; V=/mnt/h/Research/results/settings_variants
VV=/mnt/h/Research/vocab; P=/opt/slam/venv/bin/python
export NATIVE=/root/build/snap-$1/native/orbslam3_run
grep -c "Web.BoW.LevelsUp" $NATIVE > /dev/null || { echo "binary lacks Web.BoW.LevelsUp: stop"; exit 1; }
O=/mnt/h/Research/results/vocab_eval
echo "=== start $(date +%T) binary $NATIVE"
for arm in official:ORBvoc.bin:stock own-L6:ORBvoc_own_k10L6.bin:stock own-L5:ORBvoc_own_k10L5.bin:bow-lu3 own-L4:ORBvoc_own_k10L4.bin:bow-lu2; do
  IFS=: read -r name voc var <<< "$arm"
  SPECS=()
  for s in $D/euroc/*; do n=$(basename $s); SPECS+=("euroc_$n|euroc|$s|$V/EuRoC_mono_$var.yaml|$G/euroc_$n.txt|"); done
  for r in 1 2 3 4 5 6; do n=dataset-room${r}_512_16; SPECS+=("tumvi_dataset-room$r|tumvi|$D/tumvi/$n|$V/TUM-VI_mono_$var.yaml|$G/tumvi_$n.txt|--clahe"); done
  VOCAB=$VV/$voc FLAVOURS=native JOBS=1 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $O/$name "1 2 3" "${SPECS[@]}" > /dev/null &
done
wait
echo "=== summaries $(date +%T)"
for a in official own-L6 own-L5 own-L4; do
  bad=$($P eval/summarize_sweep.py $O/$a --bad-ate-mm 200 --bad-rot 1000 | awk '$2=="native"{b+=$5; n+=$3} END{print b "/" n}')
  echo "  $a: bad $bad, loops $(cat $O/$a/*/*/run.log | grep -c 'Loop detected'), merges $(cat $O/$a/*/*/run.log | grep -c 'Merge detected'), relocalised $(cat $O/$a/*/*/run.log | grep -c 'Relocalized'), track ms mean $(cat $O/$a/*/*/track_times.txt | awk '{s+=$2;n++} END{printf "%.2f", s/n}')"
done
for a in own-L6 own-L5 own-L4; do $P eval/compare_builds.py --across $O/official $O/$a --a native --bad-ate-mm 200 --bad-rot 1000 | tail -2; done
ls -la $VV
echo "=== done $(date +%T)"
