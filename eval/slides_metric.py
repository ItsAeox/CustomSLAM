#!/usr/bin/env python3
"""Metric check of a VIO trajectory (TUM: t x y z qx qy qz qw, metres) on a phone recording with a known back-and-forth
motion (rec_20261029_202938: 1.00 m slides; claude/PHONE.md): the positions over --t0..--t1 (s from the first pose) on their
main axis, turning points as in scale_known_motion.py, the median distance between consecutive ones against --dist.
Also the spread on the other two axes and the largest distance from the start (divergence check).
  slides_metric.py <traj_tum.txt> [--dist 1.0] [--t0 0] [--t1 1e9] [--min-gap 0.8] [--min-frac 0.5]
"""
import argparse

import numpy as np

from scale_known_motion import turning_points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("--dist", type=float, default=1.0)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=1e9)
    ap.add_argument("--min-gap", type=float, default=0.8)
    ap.add_argument("--min-frac", type=float, default=0.5)
    a = ap.parse_args()
    d = np.loadtxt(a.traj, comments="#", ndmin=2)
    d = d[d[:, 0] > 0]  # EqVIO writes time -1 (+ lag) for frames before its first IMU sample
    t, pos = d[:, 0] - d[0, 0], d[:, 1:4]
    far = np.linalg.norm(pos - pos[0], axis=1).max()
    sel = (t >= a.t0) & (t <= a.t1) & np.isfinite(pos).all(1)
    p, tt = pos[sel], t[sel]
    c = p - p.mean(0)
    _, S, Vt = np.linalg.svd(c, full_matrices=False)
    xs, ext, amps = turning_points(tt, c @ Vt[0], a.min_gap, a.min_frac)
    sd = S / np.sqrt(len(c))
    head = f"{a.traj}: {len(t)} poses, max distance from start {far:.2f} m, spread 1-sigma {sd[0]:.3f} / {sd[1]:.3f} / {sd[2]:.3f} m"
    if not len(amps):
        print(head + "; no back-and-forth found")
        return
    m = np.median(amps)
    print(head + f"; {len(ext)} turning points, distance median {m:.3f} m (25-75 % {np.percentile(amps, 25):.3f}-"
          f"{np.percentile(amps, 75):.3f}) -> {m / a.dist:.3f}x the true {a.dist} m")


if __name__ == "__main__":
    main()
