#!/usr/bin/env bash
# web/imuscale.js (browser) vs eval/scale_from_accel.py on the same visual-only runs (vocab_eval official, seed 1), 2 s windows
cd /mnt/h/Research/CustomSLAM
N=/opt/slam/emsdk/node/24.19.0_64bit/bin/node; P=/opt/slam/venv/bin/python; R=/mnt/h/Research/results/vocab_eval/official
D=/mnt/h/Research/datasets; G=/mnt/h/Research/results/gt
for s in MH_01_easy MH_04_difficult V1_02_medium V2_03_difficult; do
  tr=$R/euroc_$s/native_seed1/traj_tum.txt; imu=$D/euroc/$s/mav0/imu0/data.csv; y=baselines/orbslam3/settings/EuRoC_mono_inertial.yaml
  echo "$s py: $($P eval/scale_from_accel.py --traj $tr --imu $imu --tbc $y --gt $G/euroc_$s.txt --window 2 | sed 's/.*| s_est/s_est/' | cut -c1-60)"
  echo "$s js: $($N scripts/imuscale_check.mjs $tr $imu $y 2)"
done
for r in 1 3; do n=dataset-room${r}_512_16
  tr=$R/tumvi_dataset-room$r/native_seed1/traj_tum.txt; imu=$D/tumvi/$n/mav0/imu0/data.csv; y=baselines/orbslam3/settings/TUM-VI_mono_inertial.yaml
  echo "room$r py: $($P eval/scale_from_accel.py --traj $tr --imu $imu --tbc $y --gt $G/tumvi_$n.txt --window 2 | sed 's/.*| s_est/s_est/' | cut -c1-60)"
  echo "room$r js: $($N scripts/imuscale_check.mjs $tr $imu $y 2)"
done
