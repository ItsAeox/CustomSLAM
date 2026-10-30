#!/usr/bin/env bash
# native runner only, -j3 (the -j4 native build was OOM-killed on 30 Oct 11:56 while 4 GB were in use); then into snap-<tag>
set -uo pipefail
echo "=== start $(date +%T)"
nice -n 10 cmake --build /root/build/slam-native-teblid -j3 --target orbslam3_run 2>&1 | { grep -E "error|FAILED|Killed" || true; } | head -20
ls -la --time-style=+%T /root/build/slam-native-teblid/orbslam3_run
cp /root/build/slam-native-teblid/orbslam3_run /root/build/snap-$1/native/ && rm -f /root/build/snap-$1/NOTE.txt
sha256sum /root/build/snap-$1/native/orbslam3_run
echo "=== done $(date +%T)"
