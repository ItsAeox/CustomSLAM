#!/usr/bin/env bash
# Effective playback rate of the new pacing modes on MH01 (600 frames = 30 s of data): native + WASM, --rt-wall and
# --live, data on /mnt/h. Expect effective_rate ~1.0 with --rt-wall when tracking keeps up.
set -uo pipefail
D=/mnt/h/Research/datasets/euroc/MH_01_easy; V=/mnt/h/Research/results/settings_variants/EuRoC_mono_stock.yaml
VOC=/mnt/h/Research/vocab/ORBvoc.bin; NODE=/opt/slam/emsdk/node/24.19.0_64bit/bin/node
for fl in native wasm; do
  if [ $fl = wasm ]; then CMD="$NODE /root/build/slam-wasm-ui/orbslam3_run.js"; else CMD=/root/build/slam-native-teblid/orbslam3_run; fi
  for mode in "--rt-wall" "--live" "--live --rate-scale 0.5"; do
    O=/tmp/rtc_${fl}$(echo $mode | tr -d " -"); rm -rf $O
    $CMD --dataset euroc --mode mono --seq $D --vocab $VOC --settings $V --out $O --seed 1 $mode --max-frames 600 > $O.log 2>&1
    echo "$fl $mode: $(grep -E '^(processed|dropped|wall_s|seq_s|effective_rate)' $O/run_info.txt | tr '\n' ' ') | track ms mean $(awk '{t+=$2;n++} END{printf "%.1f", t/n}' $O/track_times.txt)"
  done
done
