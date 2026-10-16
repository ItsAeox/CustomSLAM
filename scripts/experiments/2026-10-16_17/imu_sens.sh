#!/usr/bin/env bash
# imu_sens: why does ORB-SLAM3 mono-inertial fail on the phone (claude/ISSUES.md G15)? One factor at a time from the
# clean 200 Hz baseline, on handheld sequences (TUM-VI rooms 1/3/5), native, unpaced (validated: g13_ctrl), 3 seeds.
# Phone symptoms to reproduce: IMU never refined (no VIBA2), "Not enough motion" resets, too-small scale.
#   base          original data and settings
#   off±20/50/100 IMU timestamps shifted by ± ms (reported = true + offset; + = IMU stamps late)
#   r60avg        IMU resampled to 60 Hz, averaged (anti-aliased), IMU.Frequency 60
#   fx0.9 / fx1.1 wrong focal length (fx, fy scaled): the phone runs uncalibrated (assumed 69° FOV)
#   noisephone    IMU noise model = the phone page's values (web/slamclient.js phoneImuSettings)
#   phonecombo    60 Hz averaged + 5 ms jitter + 2% drops + 50 ms offset + phone noise model
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
P=/opt/slam/venv/bin/python
D=/mnt/h/Research/datasets/tumvi; S=/mnt/h/Research/datasets_synth/tumvi; G=/mnt/h/Research/results/gt
R=/mnt/h/Research/results/imu_sens
SET=baselines/orbslam3/settings/TUM-VI_mono_inertial.yaml
export NATIVE=/root/build/snap-cea2e3b/native/orbslam3_run
mkdir -p $R/settings $S
echo "=== start $(date +%T) (binary $NATIVE)"
phone_noise() { sed -e 's/^IMU.NoiseGyro:.*/IMU.NoiseGyro: 6.0e-4/' -e 's/^IMU.NoiseAcc:.*/IMU.NoiseAcc: 1.0e-2/' \
                    -e 's/^IMU.GyroWalk:.*/IMU.GyroWalk: 2.0e-5/' -e 's/^IMU.AccWalk:.*/IMU.AccWalk: 3.0e-3/'; }
fscale() { awk -v s=$1 '/^Camera1\.f[xy]:/{split($0,a,":"); printf "%s: %.6f\n", a[1], a[2]*s; next} {print}'; }
cp $SET $R/settings/base.yaml
for c in off+20 off-20 off+50 off-50 off+100 off-100; do cp $SET $R/settings/$c.yaml; done
sed 's/^IMU.Frequency:.*/IMU.Frequency: 60.0/' $SET > $R/settings/r60avg.yaml
fscale 0.9 < $SET > $R/settings/fx0.9.yaml
fscale 1.1 < $SET > $R/settings/fx1.1.yaml
phone_noise < $SET > $R/settings/noisephone.yaml
phone_noise < $SET | sed 's/^IMU.Frequency:.*/IMU.Frequency: 60.0/' > $R/settings/phonecombo.yaml
grep -H "^Camera1.f[xy]\|^IMU.Noise\|^IMU.Frequency" $R/settings/fx0.9.yaml $R/settings/phonecombo.yaml
ROOMS="room1 room3 room5"
for r in $ROOMS; do n=dataset-${r}_512_16
  for o in 20 -20 50 -50 100 -100; do c=off$( [ $o -gt 0 ] && echo + )$o
    $P scripts/degrade_imu.py $D/$n $S/${n}_$c --rate 0 --offset-ms $o; done
  $P scripts/degrade_imu.py $D/$n $S/${n}_r60avg --rate 60 --average
  $P scripts/degrade_imu.py $D/$n $S/${n}_phonecombo --rate 60 --average --jitter-ms 5 --drop 0.02 --offset-ms 50 --seed 1
done
for c in base off+20 off-20 off+50 off-50 off+100 off-100 r60avg fx0.9 fx1.1 noisephone phonecombo; do
  SPECS=()
  for r in $ROOMS; do n=dataset-${r}_512_16
    seq=$D/$n; case $c in off*|r60avg|phonecombo) seq=$S/${n}_$c;; esac
    SPECS+=("tumvi_dataset-$r|tumvi|$seq|$R/settings/$c.yaml|$G/tumvi_${n}_body.txt|--clahe --mode mono_inertial")
  done
  FLAVOURS=native JOBS=4 bash scripts/seed_sweep.sh $R/$c "1 2 3" "${SPECS[@]}" > /dev/null
  echo "  $c done $(date +%T)"
done
$P scripts/experiments/2026-10-16_17/vi_table.py $R
echo "=== done $(date +%T)"
