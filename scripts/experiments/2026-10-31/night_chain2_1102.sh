#!/usr/bin/env bash
# 2 Nov night chain 2: after night_chain_1102.sh (loopfast), build + snapshot the working tree as <tag> (orbslam3_run
# --dump-local), stop on a failed build (a failed build keeps the old binary), then anchorfloor_run.sh <tag>.
# Usage: night_chain2_1102.sh <tag>   Log: results/night_chain2_1102.log
TAG=$1
L=/mnt/h/Research/results/night_chain_1102.log
until grep -qE '=== chain done|FAILED|not started|missing' $L 2>/dev/null; do sleep 60; done
echo "=== chain 1 ended $(date +%T): $(tail -1 $L); build $TAG"
B=$(bash /mnt/h/Research/CustomSLAM/scripts/experiments/2026-10-30/build_snap.sh $TAG 2>&1)
if echo "$B" | grep -q 'FAILED'; then echo "$B" | grep -E 'FAILED|error' | head; echo "=== build FAILED, anchorfloor not started"; exit 1; fi
H1=$(sha256sum /root/build/snap-$TAG/wasm/orbslam3_run.wasm | cut -d' ' -f1); H0=$(sha256sum /root/build/snap-591c5a8/wasm/orbslam3_run.wasm | cut -d' ' -f1)
if [ "$H1" = "$H0" ]; then echo "=== snapshot identical to 591c5a8: build did not take, anchorfloor not started"; exit 1; fi
strings /root/build/snap-$TAG/native/orbslam3_run | grep -q 'local_pts.bin' || { echo "=== --dump-local missing in the binary"; exit 1; }
echo "=== build ok $(date +%T); anchorfloor"
bash /mnt/h/Research/CustomSLAM/scripts/experiments/2026-10-31/anchorfloor_run.sh $TAG
echo "=== chain 2 done $(date +%T)"
