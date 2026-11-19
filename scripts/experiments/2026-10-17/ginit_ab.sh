#!/usr/bin/env bash
# Gyro two-view init versions side by side: snap-97b9685 (2 px RANSAC, gyro rotation as is), snap-a75dc17 (6 px RANSAC +
# joint R/t refinement), snap-<tag> (+ a prior keeping R near the gyro's) side by side (same load), dataset gyro and phone-like gyro, first 300 frames, live@0.5, seeds 1-3.
NEW=${1:?snapshot tag}
cd /mnt/h/Research/CustomSLAM
S=/mnt/h/Research/datasets_synth; D=/mnt/h/Research/datasets; V=/mnt/h/Research/results/settings_variants
VOC=/mnt/h/Research/vocab/ORBvoc.bin; O=/mnt/h/Research/results/ginit_ab
for src in orig phone; do for s in 1 2 3; do
  for q in "euroc V1_03_difficult" "euroc MH_04_difficult" "euroc MH_05_difficult" "tumvi dataset-room1_512_16"; do
    set -- $q; ds=$1; n=$2
    for tag in 97b9685 a75dc17 $NEW; do
      st=$V/EuRoC_mono_fast-gyro-ginit.yaml; c=""; [ $ds = tumvi ] && { st=$V/TUM-VI_mono_fast-gyro-ginit.yaml; c=--clahe; }
      seq=$D/$ds/$n; [ $src = phone ] && seq=$S/$ds/${n}_phonegyro
      o=$O/$tag/$src/${n}_s$s; mkdir -p $o
      /root/build/snap-$tag/native/orbslam3_run --dataset $ds --mode mono --seq $seq --vocab $VOC --settings $st $c --seed $s --live --rate-scale 0.5 --gyro --max-frames 300 --out $o > $o/run.log 2>&1 &
    done
  done; wait
done; done
for tag in 97b9685 a75dc17 $NEW; do for src in orig phone; do
  echo "$tag $src: frames before map (mean) $(for f in $O/$tag/$src/*/track_times.txt; do awk '$3==2||$3==5{print NR-1; exit}' $f; done | awk '{s+=$1;n++} END{printf "%.1f over %d runs", s/n, n}'), gyro inits $(cat $O/$tag/$src/*/run.log | grep -c 'Gyro two-view'), runs with a gyro init $(grep -l 'Gyro two-view' $O/$tag/$src/*/run.log | wc -l)"
done; done
