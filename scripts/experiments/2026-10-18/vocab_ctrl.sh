#!/usr/bin/env bash
# vocab_ctrl: official and own-L5 again under the same load as hbrb-L5 / own2x-L5 (two runs at a time). The vocab_eval2
# runs of official / own-L5 had four at a time (track ms mean ~30 vs ~18), so cross-batch ATE was confounded.
# Same binary (snap-kf1), protocol and sequences as vocab_hbrb.sh; output results/vocab_eval_r2/<arm>, then all four
# arms side by side (hbrb-L5 / own2x-L5 from results/vocab_eval).
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; VV=/mnt/h/Research/vocab; P=/opt/slam/venv/bin/python
G=/mnt/h/Research/results/gt; V=/mnt/h/Research/results/settings_variants; O=/mnt/h/Research/results/vocab_eval_r2
export NATIVE=/root/build/snap-kf1/native/orbslam3_run
echo "=== start $(date +%T) binary $NATIVE"
for arm in official:ORBvoc.bin:stock own-L5:ORBvoc_own_k10L5.bin:bow-lu3; do
  IFS=: read -r name voc var <<< "$arm"
  SPECS=()
  for s in $D/euroc/*; do n=$(basename $s); [ -s $O/$name/euroc_$n/native_seed3/track_times.txt ] && continue
    SPECS+=("euroc_$n|euroc|$s|$V/EuRoC_mono_$var.yaml|$G/euroc_$n.txt|"); done
  for r in 1 2 3 4 5 6; do n=dataset-room${r}_512_16; [ -s $O/$name/tumvi_dataset-room$r/native_seed3/track_times.txt ] && continue
    SPECS+=("tumvi_dataset-room$r|tumvi|$D/tumvi/$n|$V/TUM-VI_mono_$var.yaml|$G/tumvi_$n.txt|--clahe"); done
  [ ${#SPECS[@]} -gt 0 ] && VOCAB=$VV/$voc FLAVOURS=native JOBS=1 EXTRA="--rt-wall" bash scripts/seed_sweep.sh $O/$name "1 2 3" "${SPECS[@]}" > /dev/null &
done
wait
echo "=== summaries $(date +%T)"
for a in $O/official $O/own-L5 /mnt/h/Research/results/vocab_eval/hbrb-L5 /mnt/h/Research/results/vocab_eval/own2x-L5; do
  n=$(basename $a)
  bad=$($P eval/summarize_sweep.py $a --bad-ate-mm 200 --bad-rot 1000 | awk '$2=="native"{b+=$5; n+=$3} END{print b "/" n}')
  echo "  $n: bad $bad, relocalised $(cat $a/*/*/run.log | grep -c 'Relocalized'), track ms mean $(cat $a/*/*/track_times.txt | awk '{s+=$2;n++} END{printf "%.2f", s/n}')"
  $P eval/summarize_sweep.py $a --bad-ate-mm 200 --bad-rot 1000 | awk -v a=$n '$2=="native"{split($0,p,"|"); split(p[3],m," "); v[++k]=m[1]} END{asort(v); printf "  %-9s median-of-medians ATE %.1f mm over %d sequences\n", a, v[int((k+1)/2)], k}'
done
for b in /mnt/h/Research/results/vocab_eval/hbrb-L5 /mnt/h/Research/results/vocab_eval/own2x-L5 $O/own-L5; do
  $P eval/compare_builds.py --across $O/official $b --a native --bad-ate-mm 200 --bad-rot 1000 | tail -22
done
echo "=== done $(date +%T)"
