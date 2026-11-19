#!/usr/bin/env bash
# 2 Nov night chain: after mapfirst_run.sh (both paces), build + snapshot 591c5a8 (Web.Loop.MinCoincidences), stop on a
# failed build (a failed build keeps the old binary), then loopfast_run.sh. Log: results/night_chain_1102.log
L=/mnt/h/Research/results/mapfirst_1101.log
until grep -q '=== phone done' $L 2>/dev/null; do sleep 60; done
echo "=== mapfirst done $(date +%T); build 591c5a8"
B=$(bash /mnt/h/Research/CustomSLAM/scripts/experiments/2026-10-30/build_snap.sh 591c5a8 2>&1)
if echo "$B" | grep -q 'FAILED'; then echo "$B" | grep -E 'FAILED|error' | head; echo "=== build FAILED, loopfast not started"; exit 1; fi
H1=$(sha256sum /root/build/snap-591c5a8/wasm/orbslam3_run.wasm | cut -d' ' -f1); H0=$(sha256sum /root/build/snap-9cba366/wasm/orbslam3_run.wasm | cut -d' ' -f1)
if [ "$H1" = "$H0" ]; then echo "=== snapshot identical to 9cba366: build did not take, loopfast not started"; exit 1; fi
strings /root/build/snap-591c5a8/native/orbslam3_run | grep -q 'coinciding keyframes' || { echo "=== new option missing in the binary"; exit 1; }
echo "=== build ok $(date +%T); loopfast"
bash /mnt/h/Research/CustomSLAM/scripts/experiments/2026-10-31/loopfast_run.sh
echo "=== chain done $(date +%T)"
