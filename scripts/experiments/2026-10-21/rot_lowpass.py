#!/usr/bin/env python3
"""Visual vs gyro angular rate per camera axis, low-passed (moving average over --win s, ~3 Hz): is the visual rotation
damped (stabilised image / tracker) or only missing the fast vibration the camera cannot sample?
Per frame interval: visual rate = log(R_i^T R_i+1) / dt (camera frame); gyro = the mean rate over the same interval
(IMU time = camera time + offset), rotated into the camera frame with R_cb = diag(1, -1, -1) (phone rear camera).
Prints, per time range, the least-squares slope visual = k * gyro and the correlation for each axis.
  rot_lowpass.py <traj_tum.txt> <imu csv> [--offset-ms -88] [--win 0.33] [--ranges 0-100,100-116]"""
import argparse

import numpy as np
from scipy.spatial.transform import Rotation

ap = argparse.ArgumentParser()
ap.add_argument("traj")
ap.add_argument("imu")
ap.add_argument("--offset-ms", type=float, default=-88.0)
ap.add_argument("--win", type=float, default=0.33)
ap.add_argument("--ranges", default="0-100,100-116")
a = ap.parse_args()
tr = np.loadtxt(a.traj, comments="#", ndmin=2)
tr = tr[np.argsort(tr[:, 0])]
imu = np.loadtxt(a.imu, delimiter=",", comments="#")
ti = imu[:, 0] * 1e-9 - a.offset_ms * 1e-3
w_c = imu[:, 1:4] * np.array([1.0, -1.0, -1.0])  # body -> camera
R = Rotation.from_quat(tr[:, 4:8])
t = tr[:, 0]
dt = np.diff(t)
keep = (dt > 0.01) & (dt < 0.15)  # skip tracking gaps
vis = (R[:-1].inv() * R[1:]).as_rotvec() / dt[:, None]
gyr = np.full_like(vis, np.nan)
cum = np.concatenate([[[0, 0, 0]], np.cumsum(w_c[1:] * np.diff(ti)[:, None], 0)])  # integral of the rate
for i in range(len(dt)):
    gyr[i] = (np.array([np.interp(t[i + 1], ti, cum[:, k]) - np.interp(t[i], ti, cum[:, k]) for k in range(3)])) / dt[i]
tm = 0.5 * (t[:-1] + t[1:]) - t[0]
# moving average over --win s (ignores gaps)
h = max(1, int(round(a.win / np.median(dt) / 2)))
lp = lambda x: np.array([x[max(0, i - h):i + h + 1].mean(0) for i in range(len(x))])
vl, gl = lp(vis), lp(gyr)
for rg in a.ranges.split(","):
    lo, hi = map(float, rg.split("-"))
    m = keep & (tm >= lo) & (tm < hi) & np.isfinite(gl).all(1)
    out = []
    for k, ax in enumerate("xyz"):
        g, v = gl[m, k], vl[m, k]
        slope = (g @ v) / max(g @ g, 1e-12)
        corr = np.corrcoef(g, v)[0, 1] if g.std() > 1e-9 else np.nan
        out.append(f"{ax}: k {slope:.2f} r {corr:.2f} gyro rms {np.degrees(g.std()):.1f} deg/s")
    raw = keep & (tm >= lo) & (tm < hi)
    ratio = np.linalg.norm(vis[raw], axis=1).sum() / max(np.linalg.norm(gyr[raw], axis=1).sum(), 1e-9)
    print(f"  {lo:5.0f}-{hi:5.0f} s ({m.sum()} intervals, low-pass {a.win} s): " + " | ".join(out) +
          f" | unfiltered magnitude ratio {ratio:.2f}")
