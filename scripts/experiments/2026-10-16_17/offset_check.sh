#!/usr/bin/env bash
# Validate eval/estimate_time_offset.py: a visual-only room1 trajectory against the original IMU (expect ~0 ms:
# TUM-VI is hardware-synchronised) and against copies shifted by known offsets (imu_sens synthetic data).
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
P=/opt/slam/venv/bin/python
T=/mnt/h/Research/results/ndebug_check/ndebug_native/tumvi_dataset-room1/native_seed1/traj_tum.txt
B=baselines/orbslam3/settings/TUM-VI_mono_inertial.yaml
S=/mnt/h/Research/datasets_synth/tumvi/dataset-room1_512_16
echo "true 0:";    $P eval/estimate_time_offset.py --traj $T --imu /mnt/h/Research/datasets/tumvi/dataset-room1_512_16/mav0/imu0/data.csv --tbc $B
for o in +20 -20 +50 -50 +100 -100; do
  echo "true $o:"; $P eval/estimate_time_offset.py --traj $T --imu ${S}_off$o/mav0/imu0/data.csv --tbc $B
done
