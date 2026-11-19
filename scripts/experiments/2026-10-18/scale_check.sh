#!/usr/bin/env bash
# scale_check: metric scale of visual-only (monocular) runs from the accelerometer (eval/scale_from_accel.py), scored
# against the ground-truth Sim3 scale. Trajectories: the existing vocab_eval2 'official' runs (stock ORB-SLAM3 settings,
# official vocabulary, rt-wall, seeds 1-3; EuRoC 11 + TUM-VI rooms 6), no new SLAM runs. Arms:
#   all      whole trajectory, 1 s windows
#   w2       whole trajectory, 2 s windows
#   first10 / first30 / first60   only the first 10 / 30 / 60 s (how soon the AR page could know the scale)
#   r60      TUM-VI rooms 1/3/5 with the 60 Hz averaged IMU (phone rate; datasets_synth/tumvi/*_r60avg)
#   off40    the same rooms with IMU stamps 40 ms late, uncorrected (*_off+40, made by imu_calib.sh)
# EuRoC caveat: its raw IMU has biases (accelerometer bias is estimated here; the gyro is not used: the rotation is visual).
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
P=/opt/slam/venv/bin/python; D=/mnt/h/Research/datasets; S=/mnt/h/Research/datasets_synth/tumvi; G=/mnt/h/Research/results/gt
R=/mnt/h/Research/results/vocab_eval/official; O=/mnt/h/Research/results/scale_check
EU=baselines/orbslam3/settings/EuRoC_mono_inertial.yaml; TV=baselines/orbslam3/settings/TUM-VI_mono_inertial.yaml
mkdir -p $O
est() {  # <arm> <label> <traj> <imu> <tbc> <gt> [extra]
  local arm=$1 lab=$2 tr=$3 imu=$4 tbc=$5 gt=$6; shift 6
  [ -s $tr ] || { echo "$arm $lab: no trajectory"; return; }
  echo "$arm $($P eval/scale_from_accel.py --traj $tr --imu $imu --tbc $tbc --gt $gt --name $lab "$@")"
}
{
for s in 1 2 3; do
  for q in $D/euroc/*; do n=$(basename $q); tr=$R/euroc_$n/native_seed$s/traj_tum.txt
    est all $n.s$s $tr $q/mav0/imu0/data.csv $EU $G/euroc_$n.txt
    est w2 $n.s$s $tr $q/mav0/imu0/data.csv $EU $G/euroc_$n.txt --window 2
    for m in 10 30 60; do est first$m $n.s$s $tr $q/mav0/imu0/data.csv $EU $G/euroc_$n.txt --max-s $m; done
  done
  for r in 1 2 3 4 5 6; do n=dataset-room${r}_512_16; tr=$R/tumvi_dataset-room$r/native_seed$s/traj_tum.txt
    est all room$r.s$s $tr $D/tumvi/$n/mav0/imu0/data.csv $TV $G/tumvi_$n.txt
    est w2 room$r.s$s $tr $D/tumvi/$n/mav0/imu0/data.csv $TV $G/tumvi_$n.txt --window 2
    for m in 10 30 60; do est first$m room$r.s$s $tr $D/tumvi/$n/mav0/imu0/data.csv $TV $G/tumvi_$n.txt --max-s $m; done
    if [ $r = 1 ] || [ $r = 3 ] || [ $r = 5 ]; then
      [ -s $S/${n}_r60avg/mav0/imu0/data.csv ] && est r60 room$r.s$s $tr $S/${n}_r60avg/mav0/imu0/data.csv $TV $G/tumvi_$n.txt
      [ -s $S/${n}_off+40/mav0/imu0/data.csv ] && est off40 room$r.s$s $tr $S/${n}_off+40/mav0/imu0/data.csv $TV $G/tumvi_$n.txt
    fi
  done
done
} | tee $O/scale_check.txt
echo "=== per arm: |s_est / s_gt - 1| median, p90, share within 5 % / 10 %"
awk '{for(i=1;i<=NF;i++) if($i=="ratio") {r=$(i+1); a=$1; e=(r>1?r-1:1-r); v[a]=v[a] " " e}} END{for(a in v) print a, v[a]}' $O/scale_check.txt | \
  $P -c "
import sys, numpy as np
for l in sys.stdin:
    a, *e = l.split(); e = np.array([float(x) for x in e if x != 'nan'])
    print(f'{a:8s} n {len(e):3d}  median {np.median(e)*100:5.1f} %  p90 {np.percentile(e,90)*100:5.1f} %  within 5 %: {np.mean(e<0.05)*100:3.0f} %  within 10 %: {np.mean(e<0.10)*100:3.0f} %')
" | sort
echo "=== done $(date +%T)"
