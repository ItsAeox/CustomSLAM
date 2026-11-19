#!/usr/bin/env bash
# Pin the runner binaries a sweep uses: /root/build/snap-<tag>/{native,wasm}. Sweeps point at the snapshot, so a
# rebuild during a sweep cannot change its binaries (2026-10-16: the G13 relink happened under a running sweep).
# Usage: snapshot.sh <tag>   (tag = the source commit the build directories were built from)
set -euo pipefail
T=/root/build/snap-$1
mkdir -p $T/native $T/wasm
cp /root/build/slam-native-teblid/orbslam3_run $T/native/
cp /root/build/slam-wasm-ui/orbslam3_run.* $T/wasm/
ls -la $T/native $T/wasm | grep -v "^total"
sha256sum $T/native/orbslam3_run $T/wasm/orbslam3_run.wasm
