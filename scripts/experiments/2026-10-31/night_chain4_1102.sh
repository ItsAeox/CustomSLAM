#!/usr/bin/env bash
# 2 Nov morning chain 4: after night_chain3_1102.ps1 (flatfirst), build + snapshot the working tree as <tag> (ee32f06:
# walk-out diagnostics, Web.KF.AfterReloc), stop on a failed build (a failed build keeps the old binary), then
# walkdiag_run.sh <tag>.  Usage: night_chain4_1102.sh <tag>   Log: results/night_chain4_1102.log
TAG=$1
L=/mnt/h/Research/results/night_chain3_1102.log
until grep -qE '=== chain 3 done|did not build' $L 2>/dev/null; do sleep 60; done
echo "=== chain 3 ended $(date +%T): $(tail -1 $L | tr -d '\r'); build $TAG"
B=$(bash /mnt/h/Research/CustomSLAM/scripts/experiments/2026-10-30/build_snap.sh $TAG 2>&1)
if echo "$B" | grep -q 'FAILED'; then echo "$B" | grep -E 'FAILED|error' | head; echo "=== build FAILED, walkdiag not started"; exit 1; fi
H1=$(sha256sum /root/build/snap-$TAG/wasm/orbslam3_run.wasm | cut -d' ' -f1); H0=$(sha256sum /root/build/snap-1f4b239/wasm/orbslam3_run.wasm | cut -d' ' -f1)
if [ "$H1" = "$H0" ]; then echo "=== snapshot identical to 1f4b239: build did not take, walkdiag not started"; exit 1; fi
strings /root/build/snap-$TAG/native/orbslam3_run | grep -q 'few_matches' || { echo "=== bridge reasons missing in the binary"; exit 1; }
echo "=== build ok $(date +%T); walkdiag"
bash /mnt/h/Research/CustomSLAM/scripts/experiments/2026-10-31/walkdiag_run.sh $TAG
echo "=== chain 4 done $(date +%T)"
