#!/usr/bin/env python3
"""Summarise a seed sweep (scripts/seed_sweep.sh): per sequence and build, median/min/max of coverage,
Sim(3) ATE and rotation ATE, plus the number of runs with a bad result (rotation ATE > --bad-rot deg or failure).

Usage: summarize_sweep.py <sweep_dir> [--bad-rot 5]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", type=Path)
    ap.add_argument("--bad-rot", type=float, default=5.0)
    a = ap.parse_args()
    print(f"{'sequence':14s} {'build':7s} {'runs':>4s} {'fail':>4s} {'bad':>4s} | {'coverage':>18s} | "
          f"{'Sim3 ATE mm':>20s} | {'rot ATE deg':>18s}")
    for seq in sorted(p for p in a.sweep.iterdir() if p.is_dir()):
        for fl in ("native", "wasm"):
            runs = sorted(seq.glob(f"{fl}_seed*"))
            if not runs:
                continue
            ms = [json.loads((r / "metrics.json").read_text()) for r in runs if (r / "metrics.json").exists()]
            fail = len(runs) - len(ms)
            cov = np.array([m["coverage"] for m in ms])
            ate = np.array([m["ate_sim3"]["rmse"] * 1000 for m in ms])
            rot = np.array([m["ate_se3_rot_deg"]["rmse"] for m in ms])
            bad = int((rot > a.bad_rot).sum()) + fail
            fmt = lambda x, p=2: f"{np.median(x):.{p}f} [{x.min():.{p}f},{x.max():.{p}f}]" if len(x) else "-"
            print(f"{seq.name:14s} {fl:7s} {len(runs):4d} {fail:4d} {bad:4d} | {fmt(cov):>18s} | "
                  f"{fmt(ate, 1):>20s} | {fmt(rot):>18s}")


if __name__ == "__main__":
    main()
