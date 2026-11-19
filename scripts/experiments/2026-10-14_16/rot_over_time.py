# Rotation error over time after SE(3) alignment (evo), to tell a constant offset from drift.
import sys
import numpy as np
from evo.core import sync
from evo.tools import file_interface
gt = file_interface.read_tum_trajectory_file(sys.argv[1])
est = file_interface.read_tum_trajectory_file(sys.argv[2])
gt, est = sync.associate_trajectories(gt, est, max_diff=0.005)
est.align(gt, correct_scale=True)
R_e = np.array([p[:3, :3] for p in est.poses_se3]); R_g = np.array([p[:3, :3] for p in gt.poses_se3])
ang = np.array([np.degrees(np.arccos(np.clip((np.trace(g.T @ e) - 1) / 2, -1, 1))) for e, g in zip(R_e, R_g)])
n = len(ang)
for q in range(10):
    seg = ang[q * n // 10:(q + 1) * n // 10]
    print(f"decile {q}: rot err median {np.median(seg):.2f} deg")
# constant offset? rotation that best maps est to gt in the camera frame: mean of R_g^T R_e
M = sum(g.T @ e for e, g in zip(R_e, R_g)); U, _, Vt = np.linalg.svd(M); Rm = U @ Vt
print("mean camera-frame offset angle:", round(float(np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1)))), 2), "deg")
