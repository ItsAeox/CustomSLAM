#!/usr/bin/env bash
# regression dilution check of eval/scale_linear_align.py: direct form at node spacing 0.2 / 0.5 / 1.0 s vs the inverse
# form (1/s, visual positions as observations) at 0.2 / 0.5 s; bias prior 0.1. EuRoC (GT) + the 21 Oct slides.
cd /mnt/h/Research/CustomSLAM
PY=/opt/slam/venv/bin/python R=/mnt/h/Research/results PH=/mnt/h/Research/datasets/phone r=rec_20261021_070221
arms=("direct 0.2|--dt 0.2" "direct 0.5|--dt 0.5" "direct 1.0|--dt 1.0" "inverse 0.2|--dt 0.2 --inverse" "inverse 0.5|--dt 0.5 --inverse")
for n in MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium V2_03_difficult; do
  d=$R/scale_online_1019b/euroc_$n/seed1
  for arm in "${arms[@]}"; do
    IFS='|' read -r an opt <<< "$arm"
    echo "euroc $n $an: $($PY eval/scale_linear_align.py --traj $d/traj_tum.txt --imu /mnt/h/Research/datasets/euroc/$n/mav0/imu0/data.csv --tbc baselines/orbslam3/settings/EuRoC_mono_inertial.yaml --gt $R/gt/euroc_$n.txt --bias-prior 0.1 $opt 2>&1 | tail -1 | sed 's/^[^|]*| //; s/ | rms.*//' | cut -c1-150)"
  done
done
for s in 1 2 3; do
  st=$(echo "1 1.970 2 1.521 3 1.382" | tr ' ' '\n' | paste - - | awk -v s=$s '$1==s{print $2}')
  for arm in "${arms[@]}"; do
    IFS='|' read -r an opt <<< "$arm"
    echo "phone slides seed$s $an: $($PY eval/scale_linear_align.py --traj $R/scale_1021/$r/beta_scale/seed$s/traj_tum.txt --imu $PH/$r/mav0/imu0/data.csv --tbc $PH/$r/settings.yaml --offset-ms -88 --s-true $st --max-s 96 --bias-prior 0.1 $opt 2>&1 | tail -1 | sed 's/^[^|]*| //; s/ | rms.*//' | cut -c1-150)"
  done
done
