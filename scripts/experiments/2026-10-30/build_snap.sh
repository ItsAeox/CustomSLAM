#!/usr/bin/env bash
# Build the working tree (native + WASM runner + browser module, -j4: see build_j4.sh) and pin it as snap-<tag>
# Usage: build_snap.sh <tag>
set -uo pipefail
bash /mnt/h/Research/CustomSLAM/scripts/experiments/2026-10-30/build_j3.sh
bash /mnt/h/Research/CustomSLAM/scripts/experiments/2026-10-16_17/snapshot.sh $1
