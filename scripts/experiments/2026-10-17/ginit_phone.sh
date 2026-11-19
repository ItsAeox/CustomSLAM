#!/usr/bin/env bash
# Does Web.Init.Gyro trigger with phone-like gyro (60 Hz, +30 ms, +-5 ms jitter: datasets_synth *_phonegyro)?
# First 300 frames, live@0.5, seeds 1-2; prints frames before the map and the gyro two-view lines.
cd /mnt/h/Research/CustomSLAM
S=/mnt/h/Research/datasets_synth; D=/mnt/h/Research/datasets; V=/mnt/h/Research/results/settings_variants
B=/root/build/snap-${1:-97b9685}/native/orbslam3_run; VOC=/mnt/h/Research/vocab/ORBvoc.bin; O=/mnt/h/Research/results/scratch/ginit_phone${1:+_$1}
for v in preset-fast-gyro fast-gyro-ginit; do for src in orig phone; do for s in 1 2; do
  for q in "euroc V1_03_difficult" "euroc MH_04_difficult" "tumvi dataset-room1_512_16"; do
    set -- $q; ds=$1; n=$2; st=$V/EuRoC_mono_$v.yaml; c=""; [ $ds = tumvi ] && { st=$V/TUM-VI_mono_$v.yaml; c=--clahe; }
    seq=$D/$ds/$n; [ $src = phone ] && seq=$S/$ds/${n}_phonegyro
    o=$O/$v/$src/${n}_s$s; mkdir -p $o
    $B --dataset $ds --mode mono --seq $seq --vocab $VOC --settings $st $c --seed $s --live --rate-scale 0.5 --gyro --max-frames 300 --out $o > $o/run.log 2>&1 &
  done; wait
done; done; done
for v in preset-fast-gyro fast-gyro-ginit; do for src in orig phone; do
  echo "$v $src: frames before map $(for f in $O/$v/$src/*/track_times.txt; do awk '$3==1||$3==0{n++} $3==2||$3==5{print n+0; exit}' $f; done | tr '\n' ' ') gyro-inits $(cat $O/$v/$src/*/run.log | grep -c 'Gyro two-view')"
done; done
