#!/usr/bin/env bash
# 2 Nov morning chain 5: after night_chain4_1102.sh (walkdiag), build + snapshot the working tree as <tag>
# (Web.Turn.IniFeatures), stop on a failed build (a failed build keeps the old binary), then bridgeini_run.sh <tag>.
# Usage: night_chain5_1102.sh <tag>   Log: results/night_chain5_1102.log
TAG=$1
L=/mnt/h/Research/results/night_chain4_1102.log
until grep -qE '=== chain 4 done|FAILED|not started|missing' $L 2>/dev/null; do sleep 30; done
echo "=== chain 4 ended $(date +%T): $(tail -1 $L | tr -d '\r'); build $TAG"
grep -qE 'FAILED|not started|missing' $L && { echo "=== chain 4 failed, bridgeini not started"; exit 1; }
B=$(bash /mnt/h/Research/CustomSLAM/scripts/experiments/2026-10-30/build_snap.sh $TAG 2>&1)
if echo "$B" | grep -q 'FAILED'; then echo "$B" | grep -E 'FAILED|error' | head; echo "=== build FAILED, bridgeini not started"; exit 1; fi
H1=$(sha256sum /root/build/snap-$TAG/wasm/orbslam3_run.wasm | cut -d' ' -f1); H0=$(sha256sum /root/build/snap-ee32f06/wasm/orbslam3_run.wasm | cut -d' ' -f1)
if [ "$H1" = "$H0" ]; then echo "=== snapshot identical to ee32f06: build did not take, bridgeini not started"; exit 1; fi
strings /root/build/snap-$TAG/native/orbslam3_run | grep -q 'features while lost' || { echo "=== IniFeatures missing in the binary"; exit 1; }
echo "=== build ok $(date +%T); bridgeini"
bash /mnt/h/Research/CustomSLAM/scripts/experiments/2026-10-31/bridgeini_run.sh $TAG
echo "=== chain 5 done $(date +%T)"
