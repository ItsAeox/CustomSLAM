#!/usr/bin/env bash
# H1 check: effective playback rate of --realtime on /mnt/h vs ext4 (native, MH01, frames 200-400), and read pattern.
set -uo pipefail
SRC=/mnt/h/Research/datasets/euroc/MH_01_easy
E=/root/datasets/euroc/MH_01_easy
if [ ! -f $E/mav0/cam0/data.csv ]; then
  mkdir -p $E/mav0/cam0; cp -r $SRC/mav0/cam0/data $SRC/mav0/cam0/data.csv $E/mav0/cam0/; echo "copied MH01 cam0 to ext4"
fi
cd $SRC/mav0/cam0/data; ls | sed -n 2000,2099p > /tmp/l2.txt
s=$(date +%s%N); while read f; do dd if="$f" of=/dev/null bs=4k status=none; done < /tmp/l2.txt; e=$(date +%s%N); echo "dd bs=4k /mnt/h: $(( (e - s) / 1000000 / 100 )) ms/file"
ls | sed -n 2100,2199p > /tmp/l3.txt
s=$(date +%s%N); while read f; do dd if="$f" of=/dev/null bs=1M status=none; done < /tmp/l3.txt; e=$(date +%s%N); echo "dd bs=1M /mnt/h: $(( (e - s) / 1000000 / 100 )) ms/file"
cd $E/mav0/cam0/data
s=$(date +%s%N); while read f; do dd if="$f" of=/dev/null bs=4k status=none; done < /tmp/l2.txt; e=$(date +%s%N); echo "dd bs=4k ext4:   $(( (e - s) / 1000000 / 100 )) ms/file"
V=/mnt/h/Research/results/settings_variants/EuRoC_mono_stock.yaml; VOC=/mnt/h/Research/vocab/ORBvoc.bin
for seq in $SRC $E; do
  O=/tmp/h1_$(basename $(dirname $(dirname $(dirname $seq))))_$RANDOM
  /root/build/slam-native/orbslam3_run --dataset euroc --mode mono --seq $seq --vocab $VOC --settings $V --out $O --seed 1 \
    --realtime --max-frames 600 2>&1 >/dev/null | while IFS= read -r l; do printf '%s %s\n' "$(date +%s.%N)" "$l"; done > $O.err
  a=$(grep " frame 200/" $O.err | cut -d' ' -f1); b=$(grep " frame 400/" $O.err | cut -d' ' -f1)
  echo "realtime $seq: frames 200->400 in $(echo "$b - $a" | bc) s (real time: 10.0 s)"
done
