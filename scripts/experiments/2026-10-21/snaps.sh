#!/usr/bin/env bash
# list the newest WASM / native snapshot builds and the machine load
for d in $(ls -dt /root/build/snap-* | head -4); do
  ls -la --time-style=+%m-%d_%H:%M $d/wasm/orbslam3_run.wasm $d/native/orbslam3_run 2>/dev/null
done
ls -dt /root/build/*/ | head -8
nproc; free -g | head -2; uptime
