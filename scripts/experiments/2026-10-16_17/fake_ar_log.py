#!/usr/bin/env python3
"""Test input for estimate_time_offset.py --ar-log: an AR-page-style log ("imu t wx wy wz", "pose t tx ty tz qx qy qz qw")
made from a TUM-VI visual trajectory and a (shifted) TUM-VI IMU, with the gyro rotated into the phone's body axes
(camera -> body = diag(1, -1, -1), as web/slamclient.js assumes), so --tbc phone must recover the known offset.

  fake_ar_log.py <traj_tum.txt> <imu data.csv> <TUM-VI_mono_inertial.yaml> <out.log>
"""
import re
import sys
from pathlib import Path

import numpy as np

traj = np.loadtxt(sys.argv[1])
imu = np.loadtxt(sys.argv[2], delimiter=",", comments="#")
text = Path(sys.argv[3]).read_text()
data = re.search(r"IMU\.T_b_c1:\s*!!opencv-matrix.*?data:\s*\[(.*?)\]", text, re.S).group(1)
Rbc = np.array([float(v) for v in data.replace("\n", " ").split(",")]).reshape(4, 4)[:3, :3]
R_phone_cam = np.diag([1.0, -1.0, -1.0])
w = imu[:, 1:4] @ (R_phone_cam @ Rbc.T).T  # TUM-VI body -> camera -> phone body
lines = [f"00:00:00.000 imu {t * 1e-9:.4f} {x:.5f} {y:.5f} {z:.5f}" for t, (x, y, z) in zip(imu[:, 0], w)]
lines += [f"00:00:00.000 pose {r[0]:.4f} " + " ".join(f"{v:.6f}" for v in r[1:]) for r in traj]
Path(sys.argv[4]).write_text("\n".join(lines) + "\n")
print(f"{sys.argv[4]}: {imu.shape[0]} imu, {traj.shape[0]} pose lines")
