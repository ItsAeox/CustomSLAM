#!/usr/bin/env bash
# Queue 6 (2026-10-16 evening): one CPU-heavy job at a time so live/rt-wall timing is not biased.
#   wait: imu_sens done + gyro build done -> snapshot "gyro1" -> vocabulary training -> presets_live -> gyro_live -> vocab_eval
set -uo pipefail
E=/mnt/h/Research/CustomSLAM/scripts/experiments/2026-10-16_17; L=/mnt/h/Research/logs
until grep -q "=== done" $L/imu_sens.log 2>/dev/null && grep -q "=== wasm-ui done" $L/build_gyro.log 2>/dev/null; do sleep 60; done
echo "=== queue6 start $(date +%T)"
bash $E/snapshot.sh gyro1
bash $E/vocab_train.sh > $L/vocab_train.log 2>&1
bash $E/presets_live.sh > $L/presets_live.log 2>&1
bash $E/gyro_live.sh gyro1 > $L/gyro_live.log 2>&1
bash $E/vocab_eval.sh gyro1 > $L/vocab_eval.log 2>&1
echo "=== queue6 done $(date +%T)"
