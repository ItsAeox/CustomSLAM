#!/usr/bin/env bash
# Native build of the working tree (low priority: sweeps run on pinned snapshots)
nice -n 15 cmake --build /root/build/slam-native-teblid -j2 --target orbslam3_run 2>&1 | grep -E "error|FAILED|warning: unused" | head -30
ls -la --time-style=+%T /root/build/slam-native-teblid/orbslam3_run
