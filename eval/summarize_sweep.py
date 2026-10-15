#!/usr/bin/env python3
"""Summarise a seed sweep (scripts/seed_sweep.sh): per sequence and build, median/min/max of coverage,
Sim(3) ATE and rotation ATE, pooled per-frame tracking time (median / p95), plus the number of runs with a bad result (rotation ATE > --bad-rot deg or failure).

Usage: summarize_sweep.py <sweep_dir> [--bad-rot 5]
"""
import argparse
import json
from pathlib import Path

import numpy as np


def track_times(runs):
    """Per-frame TrackMonocular times (ms) pooled over runs (track_times.txt: idx ms state)."""
    out = []
    for r in runs:
        f = r / "track_times.txt"
        if f.exists():
            out += [float(l.split()[1]) for l in f.read_text().splitlines()
                    if len(l.split()) >= 2 and not l.startswith("#")]  # tolerate a truncated last line
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", type=Path)
    ap.add_argument("--bad-rot", type=float, default=5.0)
    ap.add_argument("--bad-ate-mm", type=float, default=0.0,
                    help="also count runs with Sim(3) ATE above this as bad (0 = off). Use on EuRoC, whose GT "
                         "orientation has per-sequence offsets of a few degrees (claude/ISSUES.md G8)")
    a = ap.parse_args()
    print(f"{'sequence':22s} {'build':7s} {'runs':>4s} {'fail':>4s} {'bad':>4s} | {'coverage':>18s} | "
          f"{'Sim3 ATE mm':>20s} | {'rot ATE deg':>18s} | {'track ms med/p95':>16s}")
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
            badm = rot > a.bad_rot
            if a.bad_ate_mm > 0:
                badm = badm | (ate > a.bad_ate_mm)
            bad = int(badm.sum()) + fail
            fmt = lambda x, p=2: f"{np.median(x):.{p}f} [{x.min():.{p}f},{x.max():.{p}f}]" if len(x) else "-"
            tt = track_times(runs)
            tts = f"{np.median(tt):.1f} / {np.percentile(tt, 95):.1f}" if len(tt) else "-"
            print(f"{seq.name:22s} {fl:7s} {len(runs):4d} {fail:4d} {bad:4d} | {fmt(cov):>18s} | "
                  f"{fmt(ate, 1):>20s} | {fmt(rot):>18s} | {tts:>16s}")


if __name__ == "__main__":
    main()
