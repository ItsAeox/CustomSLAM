#!/usr/bin/env bash
# After the PreSave fix + varint format: 3 real-time sessions saving at once (crashed 6/9 before), then MH02 loading
# the new (v2) MH01 map and the old (v1) smoke-test map
cd /mnt/h/Research/CustomSLAM
B=/root/build/slam-native-teblid/orbslam3_run; D=/mnt/h/Research/datasets; V=/mnt/h/Research/results/settings_variants
VOC=/mnt/h/Research/vocab/ORBvoc.bin; O=/mnt/h/Research/results/scratch/verify_save; G=/mnt/h/Research/results/gt; rm -rf $O; mkdir -p $O
r() { local n=$1; shift; mkdir -p $O/$n; $B "$@" --out $O/$n > $O/$n/run.log 2>&1; echo "$n exit $?"; }
r mh01 --dataset euroc --mode mono --seq $D/euroc/MH_01_easy --vocab $VOC --settings $V/EuRoC_mono_preset-fast.yaml --seed 2 --rt-wall --save-map $O/mh01.map &
r v101 --dataset euroc --mode mono --seq $D/euroc/V1_01_easy --vocab $VOC --settings $V/EuRoC_mono_preset-fast.yaml --seed 1 --rt-wall --save-map $O/v101.map &
r room1 --dataset tumvi --mode mono --seq $D/tumvi/dataset-room1_512_16 --vocab $VOC --settings $V/TUM-VI_mono_preset-fast.yaml --clahe --seed 1 --rt-wall --save-map $O/room1.map &
wait
grep -h "WEBMAP saved" $O/*/run.log
r mh02v2 --dataset euroc --mode mono --seq $D/euroc/MH_02_easy --vocab $VOC --settings $V/EuRoC_mono_preset-fast.yaml --seed 1 --load-map $O/mh01.map &
r mh02v1 --dataset euroc --mode mono --seq $D/euroc/MH_02_easy --vocab $VOC --settings $V/EuRoC_mono_preset-fast.yaml --seed 1 --load-map /mnt/h/Research/results/scratch/smoke_map/A/map.bin &
wait
for n in mh02v2 mh02v1; do grep -h "WEBMAP" $O/$n/run.log | head -3; done
/opt/slam/venv/bin/python eval/map_reuse.py --a-traj $O/mh01/traj_tum.txt --a-gt $G/euroc_MH_01_easy.txt --b-traj $O/mh02v2/traj_tum.txt --b-gt $G/euroc_MH_02_easy.txt --b-frames $(wc -l < $O/mh02v2/track_times.txt) --name v2
