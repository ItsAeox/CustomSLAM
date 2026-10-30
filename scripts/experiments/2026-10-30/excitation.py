#!/usr/bin/env python3
"""Acceleration excitation of a run, independent of the scale: the IMU specific force rotated into the map frame with the
live visual rotation (trace.txt), minus its mean over a sliding window (gravity + slowly varying bias), RMS over the window.
Prints the percentiles of that RMS over the tracked frames. For the scale filter's convergence gate (slow slides:
unobservable, the filter claimed 3.7-4.8x on rec_20261021_070221).
  excitation.py <run dir> <sequence dir> [--window 10] [--offset-ms -88] [--imu csv]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "eval"))
from scale_from_accel import load_tbc  # noqa: E402

a = sys.argv[1:]
run, seq = Path(a[0]), Path(a[1])
W = float(a[a.index("--window") + 1]) if "--window" in a else 10.0
OFF = float(a[a.index("--offset-ms") + 1]) if "--offset-ms" in a else -88.0
imu_path = a[a.index("--imu") + 1] if "--imu" in a else seq / "mav0" / "imu0" / "data.csv"
t_i = np.loadtxt(imu_path, delimiter=",", comments="#", usecols=0, dtype=np.int64)
d = np.loadtxt(imu_path, delimiter=",", comments="#")
ti = (t_i - t_i[0]) * 1e-9 + t_i[0] * 1e-9 - OFF * 1e-3
f = d[:, 4:7]
Rbc = load_tbc(seq / "settings.yaml")[:3, :3]
tr = np.array([[float(x) for x in l.split()[:14]] for l in (run / "trace.txt").read_text().splitlines() if len(l.split()) >= 14])
cam = np.loadtxt(seq / "mav0" / "cam0" / "data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64)
tf = cam[np.clip(tr[:, 0].astype(int), 0, len(cam) - 1)] * 1e-9
ok = np.isin(tr[:, 1].astype(int), [2, 5])
tf, q = tf[ok], tr[ok, 6:10]
_, u = np.unique(tf, return_index=True)
tf, q = tf[u], q[u]
m = (ti >= tf[0]) & (ti <= tf[-1])
ti, f = ti[m], f[m]
R_vc = Slerp(tf, Rotation.from_quat(q))(ti)
af = (R_vc * Rotation.from_matrix(Rbc.T)).apply(f)  # R_vb f: specific force in the map frame
rms = []
for k in range(0, len(ti), 6):
    w = (ti > ti[k] - W) & (ti <= ti[k])
    if ti[k] - ti[0] < W:
        continue
    x = af[w] - af[w].mean(0)
    rms.append(np.sqrt((x ** 2).sum(1).mean()))
r = np.array(rms)
print(f"{run}: excitation RMS over {W:.0f} s windows (m/s^2): p10 {np.percentile(r, 10):.3f} median {np.median(r):.3f} p90 {np.percentile(r, 90):.3f} max {r.max():.3f}")
