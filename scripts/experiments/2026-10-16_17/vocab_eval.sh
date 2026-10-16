#!/usr/bin/env bash
# vocab_eval: does a smaller vocabulary cost accuracy or robustness? BoW is used in tracking (reference-keyframe
# matching), relocalisation, loop closing and map merging. Arms side by side (one run each at a time), native
# (pinned snapshot), rt-wall, stock settings, EuRoC 11 + TUM-VI rooms 6, seeds 1-3:
#   official (ORBvoc.bin, k10L6, 48.7 MB) | own-L6 (ORBvoc_own_k10L6) | own-L5 | own-L4
# Usage: vocab_eval.sh <snapshot tag>
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; V=/mnt/h/Research/results/settings_variants
VV=/mnt/h/Research/vocab; P=/opt/slam/venv/bin/python
export NATIVE=/root/build/snap-$1/native/orbslam3_run
O=/mnt/h/Research/results/vocab_eval
SPECS=()
for s in $D/euroc/*; do n=$(basename $s); SPECS+=("euroc_$n|euroc|$s|$V/EuRoC_mono_stock.yaml|$G/euroc_$n.txt|"); done
for r in 1 2 3 4 5 6; do n=dataset-room${r}_512_16; SPECS+=("tumvi_dataset-room$r|tumvi|$D/tumvi/$n|$V/TUM-VI_mono_stock.yaml|$G/tumvi_$n.txt|--clahe"); done
echo "=== start $(date +%T) binary $NATIVE"
for arm in official:ORBvoc.bin own-L6:ORBvoc_own_k10L6.bin own-L5:ORBvoc_own_k10L5.bin own-L4:ORBvoc_own_k10L4.bin; do
  VOCAB=$VV/${arm#*:} FLAVOURS=native JOBS=1 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $O/${arm%%:*} "1 2 3" "${SPECS[@]}" > /dev/null &
done
wait
echo "=== summaries $(date +%T)"
for a in official own-L6 own-L5 own-L4; do
  bad=$($P eval/summarize_sweep.py $O/$a --bad-ate-mm 200 --bad-rot 1000 | awk '$2=="native"{b+=$5; n+=$3} END{print b "/" n}')
  echo "  $a: bad $bad, loops $(cat $O/$a/*/*/run.log | grep -c 'Loop detected'), merges $(cat $O/$a/*/*/run.log | grep -c 'Merge detected'), relocalised $(cat $O/$a/*/*/run.log | grep -c 'Relocalized')"
done
for a in own-L6 own-L5 own-L4; do $P eval/compare_builds.py --across $O/official $O/$a --a native --bad-ate-mm 200 --bad-rot 1000 | tail -2; done
ls -la $VV
echo "=== done $(date +%T)"
