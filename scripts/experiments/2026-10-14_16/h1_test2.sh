#!/usr/bin/env bash
set -uo pipefail
SRC=/mnt/h/Research/datasets/euroc/MH_01_easy; E=/root/datasets/euroc/MH_01_easy
mkdir -p $E/mav0/imu0 && cp $SRC/mav0/imu0/data.csv $E/mav0/imu0/
f=/tmp/h1_Research_3118.err
a=$(grep " frame 200/" $f | cut -d' ' -f1); b=$(grep " frame 400/" $f | cut -d' ' -f1)
echo "realtime /mnt/h (earlier run): frames 200->400 in $(python3 -c "print(round($b-$a,2))") s (real time 10.0 s)"
cd $SRC/mav0/cam0/data; ls | sed -n 2000,2099p > /tmp/l2.txt; ls | sed -n 2100,2199p > /tmp/l3.txt
s=$(date +%s%N); while read f; do dd if="$f" of=/dev/null bs=4k status=none; done < /tmp/l2.txt; e=$(date +%s%N); echo "dd bs=4k /mnt/h: $(( (e - s) / 100000 )) /1000 ms per file x100"
s=$(date +%s%N); while read f; do dd if="$f" of=/dev/null bs=1M status=none; done < /tmp/l3.txt; e=$(date +%s%N); echo "dd bs=1M /mnt/h: $(( (e - s) / 100000 )) /1000 ms per file x100"
V=/mnt/h/Research/results/settings_variants/EuRoC_mono_stock.yaml; VOC=/mnt/h/Research/vocab/ORBvoc.bin
for seq in $E $SRC; do
  O=/tmp/h1b_$RANDOM
  /root/build/slam-native/orbslam3_run --dataset euroc --mode mono --seq $seq --vocab $VOC --settings $V --out $O --seed 1 \
    --realtime --max-frames 600 2>$O.err >/dev/null
  echo "done $seq"
done
