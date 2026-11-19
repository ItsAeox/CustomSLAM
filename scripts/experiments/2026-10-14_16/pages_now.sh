#!/usr/bin/env bash
set -euo pipefail
cd /mnt/h/Research/CustomSLAM
bash scripts/build_pages.sh
git status --short docs | head -20
