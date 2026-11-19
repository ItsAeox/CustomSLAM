#!/usr/bin/env python3
"""Per-scenario summary of vio_filter_runs.py output (raw medians, no ranking): runs where the filter / the online
estimator converged, median |s/s_gt - 1| (per run median, then the median over runs), metric ATE of the scaled live
positions (filter / online, over their converged frames) and the true-scale floor.
  vio_filter_summary.py <vio_filter_<tag>.txt> [more files ...]
"""
import re
import sys
from collections import defaultdict

import numpy as np


def num(p, s):
    m = re.search(p, s)
    return float(m.group(1)) if m else None


for path in sys.argv[1:]:
    rows = defaultdict(list)
    for line in open(path, encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        seq, arm, seed, imu = line.split()[:4]
        imu = imu.rstrip(":")
        sc = re.match(r"[a-z]+", seq).group(0)
        var = "clean" if "_clean" in seq else ("ideal-run" if "__imuideal" in seq else "phone-run")
        on = line.split("| online:")[1] if "| online:" in line else ""
        main = line.split("| online:")[0]
        rows[(sc, var, imu)].append(dict(
            conv=num(r"converged (\d+) s", main) or 0, first=num(r"first after ([\d.]+) s", main),
            err=num(r"median ([\d.]+) %", main), ate=num(r"metric ATE ([\d.]+) cm", main),
            floor=num(r"floor ([\d.]+) cm", main), ate_on=num(r"online estimator ([\d.]+) cm", main),
            on_err=num(r"median ([\d.]+) %", on), on_conv=num(r"converged (\d+) s", on) or 0))
    print(f"# {path}\n")
    print("| Scenario | Run (WebVIO IMU) | Filter IMU | Runs | Filter converged in | first after s (median) | filter err median % | "
          "online converged in | online err median % | metric ATE cm: filter / online / true-scale floor |")
    print("|" + "---|" * 10)
    md = lambda v, fmt="{:.1f}": fmt.format(np.median(v)) if v else "-"
    for (sc, var, imu), rs in sorted(rows.items()):
        print(f"| {sc} | {var} | {imu} | {len(rs)} | {sum(r['conv'] > 0 for r in rs)} | {md([r['first'] for r in rs if r['first'] is not None])} | "
              f"{md([r['err'] for r in rs if r['err'] is not None])} | {sum(r['on_conv'] > 0 for r in rs)} | "
              f"{md([r['on_err'] for r in rs if r['on_err'] is not None])} | {md([r['ate'] for r in rs if r['ate'] is not None])} / "
              f"{md([r['ate_on'] for r in rs if r['ate_on'] is not None])} / {md([r['floor'] for r in rs if r['floor'] is not None])} |")
    print()
