#!/usr/bin/env bash
cd /mnt/h/Research/CustomSLAM
K=/mnt/h/Research/results/klt_sweep
for fl in wasm native; do echo "== $fl"; /opt/slam/venv/bin/python eval/compare_builds.py --across $K/stock $K/klt --a $fl --bad-ate-mm 200 --bad-rot 1000 2>&1 | tail -3; done
