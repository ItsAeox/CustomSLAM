#!/usr/bin/env bash
set -uo pipefail
V=/mnt/h/Research/results/settings_variants/EuRoC_mono_stock.yaml; VOC=/mnt/h/Research/vocab/ORBvoc.bin
for seq in /root/datasets/euroc/MH_01_easy /mnt/h/Research/datasets/euroc/MH_01_easy; do
  O=/tmp/h1c_$RANDOM
  /root/build/slam-native/orbslam3_run --dataset euroc --mode mono --seq $seq --vocab $VOC --settings $V --out $O --seed 1 \
    --realtime --max-frames 600 2>&1 >/dev/null | while IFS= read -r l; do printf '%s %s\n' "$(date +%s.%N)" "$l"; done > $O.err
  a=$(grep " frame 200/" $O.err | cut -d' ' -f1); b=$(grep " frame 400/" $O.err | cut -d' ' -f1)
  echo "realtime $seq: frames 200->400 in $(python3 -c "print(round($b-$a,2))") s (real time 10.0 s)"
done
