#!/usr/bin/env bash
# Wall time vs summed TrackMonocular time for 300 MH01 frames, unpaced (WASM and native).
D=/mnt/h/Research/datasets/euroc/MH_01_easy; V=/mnt/h/Research/results/settings_variants/EuRoC_mono_stock.yaml
VOC=/mnt/h/Research/vocab/ORBvoc.bin; NODE=/opt/slam/emsdk/node/24.19.0_64bit/bin/node
for fl in wasm native; do
  O=/tmp/ovh_$fl; rm -rf $O
  if [ $fl = wasm ]; then CMD="$NODE /root/build/slam-wasm-ui/orbslam3_run.js"; else CMD=/root/build/slam-native/orbslam3_run; fi
  $CMD --dataset euroc --mode mono --seq $D --vocab $VOC --settings $V --out $O --seed 1 --max-frames 300 > $O.log 2>&1 &
  pid=$!
  # time from the first track_times line to the end (excludes vocabulary loading)
  until grep -q "^frame 0/" $O.log 2>/dev/null; do sleep 0.05; done; s=$(date +%s%N)
  wait $pid; e=$(date +%s%N)
  awk -v w=$(( (e - s) / 1000000 )) '{t+=$2; n++} END{printf "%s: %d frames, wall %d ms (%.1f ms/frame), TrackMonocular %.1f ms/frame, other %.1f ms/frame\n", "'$fl'", n, w, w/n, t/n, w/n - t/n}' $O/track_times.txt
done
