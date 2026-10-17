#!/usr/bin/env bash
# Finish a screen_a.sh-style screen that was interrupted: only the (sequence, seed) runs without metrics.json are run
# (a partial run directory is removed first), arms side by side, same protocol as screen_a.sh.
# Usage: resume_screen.sh <snapshot tag> <screen name> <arm...>
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt; V=/mnt/h/Research/results/settings_variants
export NATIVE=/root/build/snap-$1/native/orbslam3_run
O=/mnt/h/Research/results/$2; shift 2
echo "=== resume $(date +%T) binary $NATIVE arms $*"
for a in "$@"; do
  (
    g=""; [[ $a == *gyro* ]] && g="--gyro"
    for k in 1 2 3 4 5; do
      for n in V1_03_difficult V2_03_difficult MH_04_difficult MH_05_difficult dataset-room1_512_16 dataset-room2_512_16 dataset-room3_512_16 dataset-room5_512_16; do
        if [[ $n == dataset-* ]]; then name=tumvi_${n%_512_16}; spec="$name|tumvi|$D/tumvi/$n|$V/TUM-VI_mono_$a.yaml|$G/tumvi_$n.txt|--clahe"
        else name=euroc_$n; spec="$name|euroc|$D/euroc/$n|$V/EuRoC_mono_$a.yaml|$G/euroc_$n.txt|"; fi
        [ -f $O/$a/$name/native_seed$k/metrics.json ] && continue
        rm -rf $O/$a/$name/native_seed$k
        FLAVOURS=native JOBS=1 EXTRA="--live --rate-scale 0.5 $g" bash scripts/seed_sweep.sh $O/$a "$k" "$spec" > /dev/null
      done
    done
  ) &
done
wait
echo "=== done $(date +%T)"
