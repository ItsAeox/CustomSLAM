#!/usr/bin/env bash
cd /mnt/h/Research/CustomSLAM && /opt/slam/venv/bin/python eval/collect_results.py > /dev/null 2>&1
echo "rows: $(( $(wc -l < /mnt/h/Research/results/all_runs.csv) - 1 ))"
cut -d, -f1 /mnt/h/Research/results/all_runs.csv | sort | uniq -c | grep -E "outdoors|magistrale1_diag|inertial|g13"
