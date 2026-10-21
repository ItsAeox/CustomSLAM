#!/usr/bin/env bash
# window length sweep of the offline accelerometer scale (eval/scale_from_accel.py, now with the 1-sigma of s):
# the 21 Oct 1 m slides (3 seeds, slides part only, true scale from eval/scale_known_motion.py) and EuRoC (the
# scale_online_1019b trajectories, ground truth)
cd /mnt/h/Research/CustomSLAM
PY=/opt/slam/venv/bin/python R=/mnt/h/Research/results PH=/mnt/h/Research/datasets/phone r=rec_20261021_070221
for s in 1 2 3; do
  d=$R/scale_1021/$r/beta_scale/seed$s
  st=$($PY eval/scale_known_motion.py $d $PH/$r --t1 100 | grep TRUE | sed 's/.*TRUE SCALE \([0-9.]*\).*/\1/')
  for w in 0.5 0.75 1 1.5 2; do
    for off in -88 -120; do
      echo "slides seed$s w$w off$off: $($PY eval/scale_from_accel.py --traj $d/traj_tum.txt --imu $PH/$r/mav0/imu0/data.csv --tbc $PH/$r/settings.yaml --window $w --offset-ms $off --max-s 96 --s-true $st 2>&1 | sed 's/^[^:]*: //' | cut -c1-175)"
    done
  done
done
for n in MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium V2_03_difficult; do
  d=$R/scale_online_1019b/euroc_$n/seed1
  for w in 0.5 0.75 1 1.5 2; do
    echo "euroc $n w$w: $($PY eval/scale_from_accel.py --traj $d/traj_tum.txt --imu /mnt/h/Research/datasets/euroc/$n/mav0/imu0/data.csv --tbc baselines/orbslam3/settings/EuRoC_mono_inertial.yaml --gt $R/gt/euroc_$n.txt --window $w 2>&1 | sed 's/^[^:]*: //' | cut -c1-175)"
  done
done
