#!/usr/bin/env bash
cd /mnt/h/Research/results/browser
/opt/slam/venv/bin/python /mnt/h/Research/CustomSLAM/eval/ar_log_summary.py ar_2026-10-16T08-40-22.log ar_2026-10-16T08-52-46.log ar_2026-10-16T08-53-49.log ar_2026-10-16T08-54-39.log ar_2026-10-16T08-55-26.log ar_2026-10-16T08-57-02.log 2>&1 | cut -c1-220
