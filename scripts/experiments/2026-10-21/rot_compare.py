#!/usr/bin/env python3
"""Per-second rotation of the visual trajectory vs the gyro (same clock: IMU time = camera time + offset), and the
visual speed: does the tracker follow the phone's rotation / motion?  rot_compare.py <traj_tum.txt> <imu csv> [offset ms]"""
import sys

import numpy as np
from scipy.spatial.transform import Rotation

tr = np.loadtxt(sys.argv[1], comments="#", ndmin=2)
tr = tr[np.argsort(tr[:, 0])]
imu = np.loadtxt(sys.argv[2], delimiter=",", comments="#")
off = float(sys.argv[3]) if len(sys.argv) > 3 else -88.0
ti = imu[:, 0] * 1e-9 - off * 1e-3
t0 = tr[0, 0]
R = Rotation.from_quat(tr[:, 4:8])
for b in np.arange(np.floor(t0), tr[-1, 0], 1.0):
    m = (tr[:, 0] >= b) & (tr[:, 0] < b + 1)
    idx = np.where(m)[0]
    if len(idx) < 3:
        print(f"  t {b - t0:6.1f}  visual: {len(idx)} poses")
        continue
    vis = sum(np.linalg.norm((R[i].inv() * R[i + 1]).as_rotvec()) for i in idx[:-1])
    span = tr[idx[-1], 0] - tr[idx[0], 0]
    mi = (ti >= tr[idx[0], 0]) & (ti <= tr[idx[-1], 0])
    gyr = np.sum(np.linalg.norm(imu[mi, 1:4], axis=1) * np.gradient(ti[mi])) if mi.sum() > 2 else np.nan
    path = np.sum(np.linalg.norm(np.diff(tr[idx, 1:4], axis=0), axis=1))
    print(f"  t {b - t0:6.1f}  rotation visual {np.degrees(vis):6.1f} deg  gyro {np.degrees(gyr):6.1f} deg  "
          f"(ratio {vis / gyr if gyr > 1e-3 else np.nan:4.2f})  path {path:.3f} units over {span:.2f} s, {len(idx)} poses")
