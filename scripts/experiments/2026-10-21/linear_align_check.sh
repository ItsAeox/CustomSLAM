#!/usr/bin/env bash
# eval/scale_linear_align.py (velocities as unknowns, bias prior 0.1 m/s2, nodes every 0.2 s) on EuRoC (the
# scale_online_1019b trajectories, GT) and on the 21 Oct recording: slides (first 96 s) and shake part (from 96 s)
cd /mnt/h/Research/CustomSLAM
PY=/opt/slam/venv/bin/python R=/mnt/h/Research/results PH=/mnt/h/Research/datasets/phone r=rec_20261021_070221
for n in MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium V2_03_difficult; do
  d=$R/scale_online_1019b/euroc_$n/seed1
  echo "euroc $n: $($PY eval/scale_linear_align.py --traj $d/traj_tum.txt --imu /mnt/h/Research/datasets/euroc/$n/mav0/imu0/data.csv --tbc baselines/orbslam3/settings/EuRoC_mono_inertial.yaml --gt $R/gt/euroc_$n.txt --dt 0.2 --bias-prior 0.1 2>&1 | tail -1 | sed 's/^[^|]*| //; s/ | span.*//' | cut -c1-170)"
done
for s in 1 2 3; do
  st=$(echo "1 1.970 2 1.521 3 1.382" | tr ' ' '\n' | paste - - | awk -v s=$s '$1==s{print $2}')
  for part in "slides|--max-s 96" "shake|--from-s 96"; do
    IFS='|' read -r name opt <<< "$part"
    for off in -88 -120; do
      echo "phone seed$s $name off$off: $($PY eval/scale_linear_align.py --traj $R/scale_1021/$r/beta_scale/seed$s/traj_tum.txt --imu $PH/$r/mav0/imu0/data.csv --tbc $PH/$r/settings.yaml --offset-ms $off --s-true $st --dt 0.2 --bias-prior 0.1 $opt 2>&1 | tail -1 | sed 's/^[^|]*| //; s/ | span.*//' | cut -c1-170)"
    done
  done
done
