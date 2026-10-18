#!/usr/bin/env bash
# Sophus "SO3::exp failed" (NaN rotation) abort seen in vocab_eval_r2 own-L5 V1_03 seed 3 (monocular). 8 seeds under gdb.
cd /mnt/h/Research/CustomSLAM
B=/root/build/snap-kf1/native/orbslam3_run; D=/mnt/h/Research/datasets/euroc; VOC=/mnt/h/Research/vocab/ORBvoc_own_k10L5.bin
SET=/mnt/h/Research/results/settings_variants/EuRoC_mono_bow-lu3.yaml; O=/mnt/h/Research/results/scratch/gdb_nan; rm -rf $O; mkdir -p $O
for s in 1 2 3 4 5 6 7 8; do mkdir -p $O/s$s
  gdb -q -batch -ex "set pagination off" -ex run -ex "bt 25" --args $B --dataset euroc --mode mono \
    --seq $D/V1_03_difficult --vocab $VOC --settings $SET --seed $s --rt-wall --out $O/s$s > $O/bt_s$s.txt 2>&1 &
done
wait
for f in $O/bt_*.txt; do if grep -q "SIGSEGV\|SIGABRT" $f; then echo "== $f"; grep -A30 "received signal" $f | grep "^#\|signal" | grep -v "raise\|abort ()\|__pthread_kill" | head -14; fi; done
echo "crashed: $(grep -l 'SIGSEGV\|SIGABRT' $O/bt_*.txt | wc -l) of 8"
