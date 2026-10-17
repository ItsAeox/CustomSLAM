#!/usr/bin/env bash
# Smoke tests (native build dir): MH01 -> save map -> MH02 with the map loaded; one fast-rot run (crash, ROT-STATS)
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
B=/root/build/slam-native-teblid/orbslam3_run; D=/mnt/h/Research/datasets; V=/mnt/h/Research/results/settings_variants
VOC=/mnt/h/Research/vocab/ORBvoc.bin; O=/mnt/h/Research/results/scratch/smoke_map; G=/mnt/h/Research/results/gt
rm -rf $O; mkdir -p $O/A $O/B $O/rot
$B --dataset euroc --mode mono --seq $D/euroc/MH_01_easy --vocab $VOC --settings $V/EuRoC_mono_preset-fast.yaml --seed 1 --out $O/A --save-map $O/A/map.bin > $O/A/run.log 2>&1
echo "A exit $?"; grep -h "WEBMAP" $O/A/run.log; ls -la $O/A/map.bin
$B --dataset euroc --mode mono --seq $D/euroc/MH_02_easy --vocab $VOC --settings $V/EuRoC_mono_preset-fast.yaml --seed 1 --out $O/B --load-map $O/A/map.bin > $O/B/run.log 2>&1
echo "B exit $?"; grep -h "WEBMAP" $O/B/run.log | head -5; grep -h "^maps\|^keyframes" $O/B/run_info.txt
/opt/slam/venv/bin/python eval/map_reuse.py --a-traj $O/A/traj_tum.txt --a-gt $G/euroc_MH_01_easy.txt --b-traj $O/B/traj_tum.txt --b-gt $G/euroc_MH_02_easy.txt --b-frames $(wc -l < $O/B/track_times.txt) --name smoke
$B --dataset tumvi --mode mono --seq $D/tumvi/dataset-room1_512_16 --vocab $VOC --settings $V/TUM-VI_mono_fast-rot.yaml --clahe --seed 1 --out $O/rot > $O/rot/run.log 2>&1
echo "rot exit $?"; grep -h "ROT-STATS\|LM-STATS" $O/rot/run.log; cat $O/rot/metrics.json 2>/dev/null | head -c 300
