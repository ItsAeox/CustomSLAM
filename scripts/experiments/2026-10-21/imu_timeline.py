#!/usr/bin/env python3
"""Timeline of a phone recording from its IMU alone: still / moving segments (gyro + accelerometer), the motion's
linear acceleration (accelerometer minus the still-phase gravity, in the device frame) and its double integral between
still phases (a rough metric displacement, the "truth" the scale solver has to recover).
  imu_timeline.py <recording dir> [--still-gyro 0.03 rad/s] [--still-acc 0.15 m/s2]
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rec")
    ap.add_argument("--still-gyro", type=float, default=0.03)
    ap.add_argument("--still-acc", type=float, default=0.15)
    a = ap.parse_args()
    d = np.loadtxt(Path(a.rec) / "mav0/imu0/data.csv", delimiter=",", comments="#")
    t = (d[:, 0] - d[0, 0]) * 1e-9
    w, f = d[:, 1:4], d[:, 4:7]
    # still: |w| and the accelerometer's deviation from its 0.5 s mean both small over +-0.25 s
    n = len(t)
    still = np.zeros(n, bool)
    for i in range(n):
        m = (t >= t[i] - 0.25) & (t <= t[i] + 0.25)
        still[i] = np.abs(w[m]).max() < a.still_gyro and np.abs(f[m] - f[m].mean(0)).max() < a.still_acc
    # segments
    segs, s0 = [], 0
    for i in range(1, n + 1):
        if i == n or still[i] != still[s0]:
            segs.append((still[s0], s0, i - 1))
            s0 = i
    print(f"{a.rec}: {t[-1]:.1f} s, {n} samples ({n / t[-1]:.1f} Hz); accelerometer steps {np.unique(np.round(np.diff(np.sort(np.unique(f))), 3))[:3]}")
    g_prev = None
    for k, (st, i0, i1) in enumerate(segs):
        dur = t[i1] - t[i0]
        if st:
            g = f[i0:i1 + 1].mean(0)
            print(f"  {t[i0]:6.2f}-{t[i1]:6.2f} still  {dur:5.2f} s  gravity {np.round(g, 2)} |g| {np.linalg.norm(g):.2f}")
            g_prev = g
        else:
            # rough displacement: double integral of (f - gravity of the previous still phase), device frame, ignores
            # rotation (only meaningful for the near-pure translations of the bed / table slides)
            seg = slice(i0, i1 + 1)
            peak = np.abs(np.linalg.norm(f[seg], axis=1) - (np.linalg.norm(g_prev) if g_prev is not None else 9.81)).max()
            rot = np.degrees(np.abs(np.cumsum(w[seg] * np.gradient(t[seg])[:, None], 0)).max())
            disp = ""
            if g_prev is not None and k + 1 < len(segs) and dur < 6:
                lin = f[seg] - g_prev
                dt = np.gradient(t[seg])
                v = np.cumsum(lin * dt[:, None], 0)
                p = np.cumsum(v * dt[:, None], 0)
                disp = f" | double-integrated {np.round(p[-1], 2)} m (|{np.linalg.norm(p[-1]):.2f}|), end speed {np.linalg.norm(v[-1]):.2f} m/s"
            print(f"  {t[i0]:6.2f}-{t[i1]:6.2f} moving {dur:5.2f} s  peak |f|-|g| {peak:.2f} m/s2, max rotation {rot:.0f} deg{disp}")


if __name__ == "__main__":
    main()
