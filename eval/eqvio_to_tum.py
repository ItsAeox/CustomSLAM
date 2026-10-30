#!/usr/bin/env python3
"""EqVIO output (IMUState.csv: time, p, q w x y z of the IMU in the world; camera.csv: time, the camera pose in the IMU
frame, estimated online) -> camera poses in TUM (Hamilton x y z w), T_wc = T_wi T_ic at each IMU-state time.
  eqvio_to_tum.py <EqVIO output dir> <out traj_tum.txt>
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

d, out = Path(sys.argv[1]), Path(sys.argv[2])
imu = np.loadtxt(d / "IMUState.csv", delimiter=",", skiprows=1, ndmin=2)
cam = np.loadtxt(d / "camera.csv", delimiter=",", skiprows=1, ndmin=2)
n = min(len(imu), len(cam))
imu, cam = imu[:n], cam[:n]
R_wi = Rotation.from_quat(imu[:, [5, 6, 7, 4]])  # w x y z -> x y z w
R_ic = Rotation.from_quat(cam[:, [5, 6, 7, 4]])
p_wc = imu[:, 1:4] + R_wi.apply(cam[:, 1:4])
q_wc = (R_wi * R_ic).as_quat()
with open(out, "w", newline="\n") as f:
    f.write("# t tx ty tz qx qy qz qw (camera, EqVIO)\n")
    for k in range(n):
        f.write(f"{imu[k, 0]:.9f} " + " ".join(f"{v:.9f}" for v in (*p_wc[k], *q_wc[k])) + "\n")
print(f"{out}: {n} poses")
