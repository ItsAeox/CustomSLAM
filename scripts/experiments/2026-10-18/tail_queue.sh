#!/usr/bin/env bash
# After the overnight queue: build the points-at-infinity version (Web.Rot.FarPoints 2), screen it like rot_screen
# (alone: live@0.5 is load-sensitive), then the same-load vocabulary control. A failed build skips the screen.
L=/mnt/h/Research/logs; E=/mnt/h/Research/CustomSLAM/scripts/experiments/2026-10-18
until grep -q "imu_calib done" $L/overnight_queue.log; do sleep 30; done
echo "=== tail start $(date +%T)"
bash $E/build_cfg.sh > $L/build_rot2.log 2>&1
if grep -q "error\|FAILED" $L/build_rot2.log; then
  echo "  build failed: rot_screen2 skipped (see build_rot2.log)"
else
  bash /mnt/h/Research/CustomSLAM/scripts/experiments/2026-10-16_17/snapshot.sh rot2 > /dev/null
  bash $E/rot_screen.sh rot2 rot_screen2 preset-fast fast-rot2 > $L/rot_screen2.log 2>&1
  echo "  rot_screen2 done $(date +%T)"
fi
bash $E/vocab_ctrl.sh > $L/vocab_ctrl.log 2>&1
echo "  vocab_ctrl done $(date +%T)"
