#!/usr/bin/env bash
# Sanity check of the gyro prior build (native, unpaced): room1 seed 1, stock vs stock-gyro (--gyro), and the prior with a
# phone-like gyro. Checks: the prior is reported active, no crash, accuracy in the same range.
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
P=/opt/slam/venv/bin/python; N=/root/build/slam-native-teblid/orbslam3_run
V=/mnt/h/Research/results/settings_variants; G=/mnt/h/Research/results/gt; n=dataset-room1_512_16
O=/mnt/h/Research/results/gyro_sanity
for arm in "stock|$V/TUM-VI_mono_stock.yaml|/mnt/h/Research/datasets/tumvi/$n|" \
           "gyro|$V/TUM-VI_mono_stock-gyro.yaml|/mnt/h/Research/datasets/tumvi/$n|--gyro" \
           "gyrophone|$V/TUM-VI_mono_stock-gyro.yaml|/mnt/h/Research/datasets_synth/tumvi/${n}_phonecombo|--gyro"; do
  IFS='|' read -r name set seq extra <<< "$arm"
  R=$O/$name/tumvi_dataset-room1/native_seed1; mkdir -p $R
  $N --dataset tumvi --mode mono --seq $seq --vocab /mnt/h/Research/vocab/ORBvoc.bin --settings $set --out $R --seed 1 --clahe $extra > $R/run.log 2>&1 &
done
wait
for name in stock gyro gyrophone; do
  R=$O/$name/tumvi_dataset-room1/native_seed1
  $P eval/evaluate.py --gt $G/tumvi_$n.txt --est $R/traj_tum.txt --out $R/metrics.json > /dev/null 2>&1
  echo "$name: $(grep -o 'gyro prior[^;]*' $R/run.log | head -1) | gyro=$(awk '/^gyro /{print $2}' $R/run_info.txt) | exit: $(tail -1 $R/run.log | cut -c1-40) | $(tr -d '\n ' < $R/metrics.json | grep -oE '"ate_sim3":\{"rmse":[0-9.e-]+' | grep -oE '[0-9.e-]+$') m Sim3"
done
