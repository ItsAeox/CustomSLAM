#!/usr/bin/env bash
# After replay_all (live@1.0): fix the recordings' gyro axes (G19), re-run the gyro presets (their first runs had the
# permuted axes), then the phone's camera-IMU time offset and focal from camera-only replays + the fixed gyro.
set -uo pipefail
cd /mnt/h/Research/CustomSLAM
PH=/mnt/h/Research/datasets/phone; R=/mnt/h/Research/results/replays_1018_rt; P=/opt/slam/venv/bin/python
until grep -q "=== done" /mnt/h/Research/logs/replay_all_rt.log; do sleep 30; done
cp /mnt/h/Research/logs/replay_all_rt.log /mnt/h/Research/logs/replay_all_rt_firstpass.log
echo "=== fix axes $(date +%T)"
$P scripts/fix_rec_axes.py $(awk '!/^#/ && NF {print "'$PH'/" $1}' $PH/labels.txt)
for r in $(awk '!/^#/ && NF {print $1}' $PH/labels.txt); do
  for p in gstart grecover next; do [ -d $R/$r/$p ] && mv $R/$r/$p $R/$r/${p}_wrongaxes; done
done
echo "=== gyro presets again $(date +%T)"
RATE=1.0 PAR=4 bash scripts/experiments/2026-10-18/replay_all.sh e11b2e9 > /mnt/h/Research/logs/replay_all_rt.log 2>&1
echo "=== phone time offset + focal (camera-only runs, fixed gyro) $(date +%T)"
for r in $(awk '!/^#/ && NF && $2 != "turning" {print $1}' $PH/labels.txt); do for s in 1 2; do
  t=$R/$r/fast/seed$s/traj_tum.txt; [ -s $t ] || continue
  echo "$r s$s: $($P eval/estimate_time_offset.py --traj $t --imu $PH/$r/mav0/imu0/data.csv --tbc phone --focal 2>&1 | sed 's/^seed[0-9]*: //' | tr '\n' ' ' | cut -c1-330)"
done; done
echo "=== done $(date +%T)"
