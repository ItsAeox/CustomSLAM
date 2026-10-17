#!/usr/bin/env bash
# Final binary: gyro two-view init fires (MH01, 400 frames, live@0.5, seed 1)
cd /mnt/h/Research/CustomSLAM; O=/mnt/h/Research/results/scratch/ginit_final; mkdir -p $O
timeout 200 /root/build/snap-$1/native/orbslam3_run --dataset euroc --mode mono --seq /mnt/h/Research/datasets/euroc/MH_01_easy \
  --vocab /mnt/h/Research/vocab/ORBvoc.bin --settings /mnt/h/Research/results/settings_variants/EuRoC_mono_fast-gyro-ginit.yaml \
  --seed 1 --live --rate-scale 0.5 --gyro --max-frames 400 --out $O > $O/run.log 2>&1
echo "exit $?"; grep 'Gyro two-view' $O/run.log | head -2; awk '$3==2||$3==5{print "frames before map", NR-1; exit}' $O/track_times.txt
