#!/usr/bin/env python3
"""Test input for estimate_time_offset.py --ar-log: an AR-page-style log ("imu t wx wy wz", "pose t tx ty tz qx qy qz qw
[maps]") made from a TUM-VI visual trajectory and a (shifted) TUM-VI IMU, with the gyro rotated into the phone's body
axes (camera -> body = diag(1, -1, -1), as web/slamclient.js assumes), so --tbc phone must recover the known offset.

--jumps N: N map changes at evenly spaced times, each moving every later pose by a random rigid transform (as a new
map / a merge re-anchors the trajectory). --no-map-field writes old-format pose lines (no map count), so only the
robust cost protects the estimate.

  fake_ar_log.py <traj_tum.txt> <imu data.csv> <TUM-VI_mono_inertial.yaml> <out.log> [--jumps 3] [--no-map-field]
"""
import argparse
import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ap = argparse.ArgumentParser()
ap.add_argument("traj"); ap.add_argument("imu"); ap.add_argument("yaml"); ap.add_argument("out")
ap.add_argument("--jumps", type=int, default=0)
ap.add_argument("--no-map-field", action="store_true")
a = ap.parse_args()

traj = np.loadtxt(a.traj)
imu = np.loadtxt(a.imu, delimiter=",", comments="#")
text = Path(a.yaml).read_text()
data = re.search(r"IMU\.T_b_c1:\s*!!opencv-matrix.*?data:\s*\[(.*?)\]", text, re.S).group(1)
Rbc = np.array([float(v) for v in data.replace("\n", " ").split(",")]).reshape(4, 4)[:3, :3]
w = imu[:, 1:4] @ (np.diag([1.0, -1.0, -1.0]) @ Rbc.T).T  # TUM-VI body -> camera -> phone body

maps = np.ones(len(traj), dtype=int)
rng = np.random.default_rng(7)
for j in range(1, a.jumps + 1):  # re-anchor every pose after the jump: T' = J * Twc
    k = j * len(traj) // (a.jumps + 1)
    J = Rotation.from_rotvec(rng.normal(size=3))
    Rw = J * Rotation.from_quat(traj[k:, 4:8])
    traj[k:, 4:8] = Rw.as_quat()
    traj[k:, 1:4] = J.apply(traj[k:, 1:4]) + rng.normal(size=3)
    maps[k:] += 1
lines = [f"00:00:00.000 imu {t * 1e-9:.4f} {x:.5f} {y:.5f} {z:.5f}" for t, (x, y, z) in zip(imu[:, 0], w)]
lines += [f"00:00:00.000 pose {r[0]:.4f} " + " ".join(f"{v:.6f}" for v in r[1:8]) + ("" if a.no_map_field else f" {m}")
          for r, m in zip(traj, maps)]
Path(a.out).write_text("\n".join(lines) + "\n")
print(f"{a.out}: {imu.shape[0]} imu, {traj.shape[0]} pose lines, {a.jumps} map jumps, map field {not a.no_map_field}")
