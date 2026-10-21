#!/usr/bin/env bash
# offline accelerometer scale (eval/scale_from_accel.py) on the 21 Oct known-motion runs: whole run vs the shake part
# (from ~98 s) vs the slides only, windows 1 / 2 s, IMU offset -88 ms; true scale from eval/scale_known_motion.py
cd /mnt/h/Research/CustomSLAM
R=/mnt/h/Research/results/scale_1021 PH=/mnt/h/Research/datasets/phone r=rec_20261021_070221
for s in 1 2 3; do
  d=$R/$r/beta_scale/seed$s
  echo "seed $s: $(/opt/slam/venv/bin/python eval/scale_known_motion.py $d $PH/$r --t1 100 | grep TRUE | sed 's/.*TRUE/TRUE/')"
  for part in "all|" "slides|--max-s 96" "shake|--from-s 96"; do
    IFS='|' read -r name opt <<< "$part"
    for w in 1 2; do
      echo "  $name w$w: $(/opt/slam/venv/bin/python eval/scale_from_accel.py --traj $d/traj_tum.txt --imu $PH/$r/mav0/imu0/data.csv --tbc $PH/$r/settings.yaml --window $w --offset-ms -88 $opt 2>&1 | sed 's/^[^:]*: //; s/ s_gt nan ratio nan//' | cut -c1-170)"
    done
  done
done
