#!/usr/bin/env bash
# G18: mono-inertial segfault after "scale too small" (TUM-VI rooms). 8 short runs under gdb, rt-wall as in imu_calib.
cd /mnt/h/Research/CustomSLAM
B=/root/build/snap-rot2/native/orbslam3_run; D=/mnt/h/Research/datasets/tumvi; VOC=/mnt/h/Research/vocab/ORBvoc.bin
SET=baselines/orbslam3/settings/TUM-VI_mono_inertial.yaml; O=/mnt/h/Research/results/scratch/gdb_g18; rm -rf $O; mkdir -p $O
for r in 1 3; do for s in 1 2 3 4; do n=r${r}s$s; mkdir -p $O/$n
  gdb -q -batch -ex "set pagination off" -ex run -ex "bt 25" -ex "thread apply all bt 6" --args $B --dataset tumvi --mode mono_inertial \
    --seq $D/dataset-room${r}_512_16 --vocab $VOC --settings $SET --clahe --seed $s --rt-wall --max-frames 600 --out $O/$n > $O/bt_$n.txt 2>&1 &
done; done
wait
for f in $O/bt_*.txt; do if grep -q "SIGSEGV\|SIGABRT" $f; then echo "== $f"; grep -A26 "received signal" $f | grep "^#\|signal" | head -26; fi; done
echo "crashed: $(grep -l 'SIGSEGV\|SIGABRT' $O/bt_*.txt | wc -l) of 8"
