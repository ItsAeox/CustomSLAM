#!/usr/bin/env bash
cd /mnt/h/Research/CustomSLAM
P=/opt/slam/venv/bin/python; R=/mnt/h/Research/results
echo "##### m3_clean: light-map vs stock (side by side, WASM)"
$P eval/compare_builds.py --across $R/m3_clean/stock $R/m3_clean/light-map --a wasm --bad-ate-mm 200 --bad-rot 1000 2>&1 | tail -28
echo "##### light-map summary"
$P eval/summarize_sweep.py $R/m3_clean/light-map --bad-ate-mm 200 --bad-rot 1000 2>&1 | head -30
