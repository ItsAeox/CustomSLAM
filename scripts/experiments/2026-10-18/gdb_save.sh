#!/usr/bin/env bash
# Reproduce the session-A segfault (map save after Shutdown, rt-wall, 3 at once) under gdb; backtraces -> bt_*.txt
cd /mnt/h/Research/CustomSLAM
B=/root/build/snap-c59fc2e/native/orbslam3_run; D=/mnt/h/Research/datasets; V=/mnt/h/Research/results/settings_variants
VOC=/mnt/h/Research/vocab/ORBvoc.bin; O=/mnt/h/Research/results/scratch/gdb_save; rm -rf $O; mkdir -p $O
g() { local n=$1; shift; mkdir -p $O/$n; gdb -q -batch -ex "set pagination off" -ex run -ex "thread apply all bt 12" --args $B "$@" --out $O/$n --save-map $O/$n/map.bin > $O/bt_$n.txt 2>&1; }
g mh01 --dataset euroc --mode mono --seq $D/euroc/MH_01_easy --vocab $VOC --settings $V/EuRoC_mono_preset-fast.yaml --seed 2 --rt-wall &
g v101 --dataset euroc --mode mono --seq $D/euroc/V1_01_easy --vocab $VOC --settings $V/EuRoC_mono_preset-fast.yaml --seed 1 --rt-wall &
g room1 --dataset tumvi --mode mono --seq $D/tumvi/dataset-room1_512_16 --vocab $VOC --settings $V/TUM-VI_mono_preset-fast.yaml --clahe --seed 1 --rt-wall &
wait
for n in mh01 v101 room1; do echo "== $n"; grep -A14 "SIGSEGV\|signal SIG" $O/bt_$n.txt | head -40; grep -h "WEBMAP saved" $O/bt_$n.txt; done
