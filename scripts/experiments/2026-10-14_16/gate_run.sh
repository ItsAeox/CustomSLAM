#!/usr/bin/env bash
# Gate for behaviour-neutral changes: native, synchronous mapping, stock settings, fixed seed.
# Usage: gate_run.sh <tag>   -> /mnt/h/Research/results/gate/<tag>/{mh01,room1}
set -uo pipefail
T=$1
D=/mnt/h/Research/datasets; S=/mnt/h/Research/CustomSLAM/baselines/orbslam3/settings; V=/mnt/h/Research/vocab/ORBvoc.bin
O=/mnt/h/Research/results/gate/$T
mkdir -p $O
/root/build/slam-native/orbslam3_run --dataset euroc --mode mono --seq $D/euroc/MH_01_easy --vocab $V \
  --settings $S/EuRoC_mono.yaml --seed 1 --sync-mapping --max-frames 1500 --out $O/mh01 > $O/mh01.log 2>&1 &
/root/build/slam-native/orbslam3_run --dataset tumvi --mode mono --seq $D/tumvi/dataset-room1_512_16 --vocab $V \
  --settings $S/TUM-VI_mono.yaml --clahe --seed 1 --sync-mapping --max-frames 1500 --out $O/room1 > $O/room1.log 2>&1 &
wait
for s in mh01 room1; do echo "$T $s: $(grep LM-STATS $O/$s.log) | $(wc -l < $O/$s/traj_tum.txt) poses | md5 $(md5sum < $O/$s/traj_tum.txt | cut -c1-12)"; done
