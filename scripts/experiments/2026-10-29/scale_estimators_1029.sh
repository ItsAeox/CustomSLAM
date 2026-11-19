#!/usr/bin/env bash
# 29 Oct recording: true scale from the 1.00 m slides (eval/scale_known_motion.py), the online estimate, and the offline
# estimators against it: window estimator (0.5-2 s) and linear alignment (direct 0.5 / 1.0 s, inverse 0.5 s, rotation
# gate < 0.5 rad/s), IMU offset -88 ms (the 18 Oct measurement) and -120 ms
cd /mnt/h/Research/CustomSLAM
PY=/opt/slam/venv/bin/python R=/mnt/h/Research/results/scale_1029 PH=/mnt/h/Research/datasets/phone r=rec_20261029_202938
for s in 1 2 3; do
  d=$R/$r/beta_scale/seed$s
  $PY eval/scale_known_motion.py $d $PH/$r | grep -vE "^    t "
  st=$($PY eval/scale_known_motion.py $d $PH/$r | grep TRUE | sed 's/.*TRUE SCALE \([0-9.]*\).*/\1/')
  [ -z "$st" ] && { echo "  seed $s: no true scale"; continue; }
  for off in -88 -120; do
    for w in 0.5 0.75 1 1.5 2; do
      echo "  windows w$w off$off: $($PY eval/scale_from_accel.py --traj $d/traj_tum.txt --imu $PH/$r/mav0/imu0/data.csv --tbc $PH/$r/settings.yaml --window $w --offset-ms $off --s-true $st 2>&1 | sed 's/^[^:]*: //; s/ | span.*//; s/ | bias [0-9.]* m\/s2//' | cut -c1-150)"
    done
    for arm in "direct 0.5|--dt 0.5" "direct 1.0|--dt 1.0" "inverse 0.5|--dt 0.5 --inverse" "direct 0.5 gate|--dt 0.5 --rate-range 0,0.5" "inverse 0.5 gate|--dt 0.5 --inverse --rate-range 0,0.5"; do
      IFS='|' read -r an opt <<< "$arm"
      echo "  align $an off$off: $($PY eval/scale_linear_align.py --traj $d/traj_tum.txt --imu $PH/$r/mav0/imu0/data.csv --tbc $PH/$r/settings.yaml --offset-ms $off --s-true $st --bias-prior 0.1 $opt 2>&1 | tail -1 | sed 's/^[^|]*| //; s/ | rms.*//' | cut -c1-150)"
    done
  done
done
