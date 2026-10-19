#!/usr/bin/env python3
"""Web.Scale.Online (slam/orbslam3/include/WebScale.h) against ground truth, or its stability without (phone).

trace.txt (orbslam3_run --trace) columns used: 0 frame, 1 state, 3-5 Twc t, 13 map keyframes, 26 s (metres per map
unit, 0 = no estimate yet), 27 its 1-sigma / s, 28 windows. Per frame with an estimate, the reference is the scale of
the same stretch of trajectory the estimator saw: Sim3 (Umeyama) of the tracked positions of the current map in the
last --keep s against the ground truth. Maps are split where the keyframe count drops to 0.

  scale_online_check.py <run dir> <sequence dir (mav0/cam0/data.csv)> [--gt gt.txt] [--keep 30] [--sigma 0.05]
Prints one line: tracked s with an estimate, converged (1-sigma < --sigma) after s, and with GT the error
|s / s_gt - 1| (median, p90) over converged frames and at the end; without GT the spread of s after convergence.
"""
import sys
from pathlib import Path

import numpy as np

a = [x for x in sys.argv[1:]]
opt = lambda k, d: a[a.index(k) + 1] if k in a else d
run, seq = Path(a[0]), Path(a[1])
gt_path, KEEP, SIG = opt("--gt", None), float(opt("--keep", 30)), float(opt("--sigma", 0.05))
rows = [l.split() for l in (run / "trace.txt").read_text().splitlines()]
tr = np.array([[float(x) for x in r] for r in rows if len(r) >= 29])
if not len(tr):
    sys.exit(f"{run}: no scale columns in trace.txt")
cam = np.loadtxt(seq / "mav0" / "cam0" / "data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64)
fr = tr[:, 0].astype(int)
t = cam[np.clip(fr, 0, len(cam) - 1)] * 1e-9
ok = np.isin(tr[:, 1].astype(int), [2, 5])
pos, kfs, s, rel, win = tr[:, 3:6], tr[:, 13], tr[:, 26], tr[:, 27], tr[:, 28]
mapid = np.cumsum(np.concatenate([[0], (kfs[1:] == 0) & (kfs[:-1] > 0)]))
gt = np.loadtxt(gt_path, comments="#", ndmin=2) if gt_path else None


def umeyama_s(src, dst):
    a0, b0 = src - src.mean(0), dst - dst.mean(0)
    U, D, Vt = np.linalg.svd(b0.T @ a0 / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    return np.trace(np.diag(D) @ S) / max((a0 ** 2).sum(1).mean(), 1e-12)


est = ok & (s > 0)
conv = est & (rel >= 0) & (rel < SIG)
dt = np.median(np.diff(t)) if len(t) > 1 else 0.05
first_conv = None
for m in np.unique(mapid):
    idx = np.where(mapid == m)[0]
    c = idx[conv[idx]]
    if len(c):
        t0 = t[idx[ok[idx]][0]] if ok[idx].any() else t[idx[0]]
        first_conv = t[c[0]] - t0 if first_conv is None else min(first_conv, t[c[0]] - t0)
line = f"{run}: tracked {ok.sum() * dt:.0f} s, with an estimate {est.sum() * dt:.0f} s, converged {conv.sum() * dt:.0f} s"
line += f" (first after {first_conv:.1f} s of its map)" if first_conv is not None else " (never)"
if gt is not None and conv.any():
    errs = []
    for k in np.where(conv)[0][::5]:
        w = np.where(ok & (mapid == mapid[k]) & (t >= t[k] - KEEP) & (t <= t[k]))[0]
        w = w[(t[w] >= gt[0, 0]) & (t[w] <= gt[-1, 0])]
        if len(w) < 30:
            continue
        g = np.stack([np.interp(t[w], gt[:, 0], gt[:, j]) for j in (1, 2, 3)], 1)
        if np.ptp(g, 0).max() < 0.3:  # under 30 cm of motion: the reference scale is not defined well
            continue
        errs.append(abs(s[k] / umeyama_s(pos[w], g) - 1))
    if errs:
        e = np.array(errs)
        line += f" | |s/s_gt - 1| median {100 * np.median(e):.1f} %, p90 {100 * np.percentile(e, 90):.1f} %, within 5 % {100 * (e < 0.05).mean():.0f} %, last {100 * e[-1]:.1f} % ({len(e)} checks)"
elif conv.any():
    sc = s[conv]
    line += f" | s median {np.median(sc):.4g}, spread (p10-p90)/median {100 * (np.percentile(sc, 90) - np.percentile(sc, 10)) / np.median(sc):.1f} %"
print(line)
