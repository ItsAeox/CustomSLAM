#!/usr/bin/env python3
"""Metric scale of a monocular (visual-only) trajectory from the accelerometer, loosely coupled (claude/RESEARCH.md
"Metric scale without full VIO"; after Nuetzi et al. 2011 / arXiv 1707.07518, solved in closed form over windows).

Monocular SLAM has an unknown scale s (map units -> metres). Over a window [t0, t1] the IMU's specific force, rotated
into the SLAM world frame with the visual orientation, integrates to the velocity change minus gravity:
    s (v(t1) - v(t0)) = sum R_wb (f_b - b_a) dt + g_w (t1 - t0)
v: velocity of the visual camera centre (central differences of the smoothed positions); R_wb = R_wc R_cb (visual
rotation, camera -> IMU extrinsic); unknowns s, gravity g_w in the SLAM world frame (3) and the accelerometer bias b_a
(body frame, 3). Linear least squares over every window (length --window, half-overlapping), then a refinement with
|g_w| = 9.81 and a Huber loss. The camera-IMU lever arm (~cm) is ignored. Needs no ground truth; --gt only scores the
estimate: the Sim3 alignment of the trajectory to the ground truth gives the true scale s_gt.

  scale_from_accel.py --traj traj_tum.txt --imu mav0/imu0/data.csv --tbc EuRoC_mono_inertial.yaml --gt gt.txt
  options: --window 1.0 (s)  --offset-ms 0 (IMU time = camera time + offset)  --max-s 0 (first N s only; 0 = all)
Prints one line: name, windows, s_est, s_gt, s_est / s_gt, |g| of the linear solution, residual RMS (m/s).
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import segments_by_gap  # noqa: E402

G = 9.81


def load_tbc(path):
    text = Path(path).read_text()
    data = re.search(r"IMU\.T_b_c1:\s*!!opencv-matrix.*?data:\s*\[(.*?)\]", text, re.S).group(1)
    return np.array([float(v) for v in data.replace("\n", " ").split(",")]).reshape(4, 4)


def load_imu(path):  # ASL csv: t [ns], w (3), a (3)
    t = np.loadtxt(path, delimiter=",", comments="#", usecols=0, dtype=np.int64)
    d = np.loadtxt(path, delimiter=",", comments="#")
    return (t - t[0]) * 1e-9 + t[0] * 1e-9, d[:, 4:7]  # seconds; precision kept by subtracting before scaling


def umeyama_scale(src, dst):  # Sim3 dst ~ s R src + t: the scale s
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, D, Vt = np.linalg.svd(b.T @ a / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    return np.trace(np.diag(D) @ S) / (a ** 2).sum(1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--imu", required=True)
    ap.add_argument("--tbc", required=True)
    ap.add_argument("--gt")
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--offset-ms", type=float, default=0.0)
    ap.add_argument("--max-s", type=float, default=0.0)
    ap.add_argument("--name", default="")
    a = ap.parse_args()

    tr = np.loadtxt(a.traj, comments="#", ndmin=2)
    tr = tr[np.argsort(tr[:, 0])]
    tr = tr[np.concatenate([[True], np.diff(tr[:, 0]) > 1e-6])]
    if a.max_s > 0:
        tr = tr[tr[:, 0] <= tr[0, 0] + a.max_s]
    tc, pc = tr[:, 0], tr[:, 1:4]
    Rwc = Rotation.from_quat(tr[:, 4:8])
    ti, acc = load_imu(a.imu)
    ti = ti - a.offset_ms * 1e-3  # IMU times on the camera clock
    Rcb = load_tbc(a.tbc)[:3, :3].T  # T_b_c1: camera -> body; R_cb = R_bc^T

    rows, rhs = [], []
    name = a.name or Path(a.traj).parent.name
    for s0, s1 in segments_by_gap(tc, 0.2):  # tracking gaps break the velocity
        if tc[s1] - tc[s0] < 2 * a.window:
            continue
        t, p = tc[s0:s1 + 1], pc[s0:s1 + 1]
        k = 2  # smooth positions over +-2 frames, then central differences
        ps = np.array([p[max(0, i - k):i + k + 1].mean(0) for i in range(len(p))])
        v = np.gradient(ps, t, axis=0)
        slerp = Slerp(t, Rwc[s0:s1 + 1])
        i0 = k
        while True:
            i1 = np.searchsorted(t, t[i0] + a.window)
            if i1 >= len(t) - k:
                break
            m = (ti >= t[i0]) & (ti <= t[i1])
            if m.sum() < 5:
                i0 = np.searchsorted(t, t[i0] + a.window / 2)
                continue
            tk, fk = ti[m], acc[m]
            dt = np.gradient(tk)
            Rwb = (slerp(tk) * Rotation.from_matrix(Rcb)).as_matrix()
            A_f = np.einsum("kij,kj,k->i", Rwb, fk, dt)
            A_R = np.einsum("kij,k->ij", Rwb, dt)
            T = dt.sum()
            # s dv - g T + A_R b = A_f
            rows.append(np.hstack([(v[i1] - v[i0])[:, None], -T * np.eye(3), A_R]))
            rhs.append(A_f)
            i0 = max(i0 + 1, np.searchsorted(t, t[i0] + a.window / 2))
    if len(rows) < 10:
        print(f"{name}: too few windows ({len(rows)})")
        return
    A, b = np.vstack(rows), np.concatenate(rhs)
    x0 = np.linalg.lstsq(A, b, rcond=None)[0]
    g_lin = np.linalg.norm(x0[1:4])

    def res(x):
        return np.concatenate([A @ x - b, [10.0 * (np.linalg.norm(x[1:4]) - G)]])
    x1 = x0.copy()
    if g_lin > 1e-6:
        x1[1:4] *= G / g_lin
    if x1[0] <= 0:
        x1[0] = abs(x1[0]) + 1e-3
    sol = least_squares(res, x1, loss="huber", f_scale=0.1)
    s_est = sol.x[0]
    rms = np.sqrt(np.mean((A @ sol.x - b) ** 2))

    s_gt = np.nan
    if a.gt:
        gt = np.loadtxt(a.gt, comments="#", ndmin=2)
        ok = (tc >= gt[0, 0]) & (tc <= gt[-1, 0])
        g = np.stack([np.interp(tc[ok], gt[:, 0], gt[:, j]) for j in (1, 2, 3)], 1)
        s_gt = umeyama_scale(pc[ok], g)
    print(f"{name}: windows {len(rows)} | s_est {s_est:.4f} s_gt {s_gt:.4f} ratio {s_est / s_gt:.3f} | "
          f"|g| linear {g_lin:.2f} | bias {np.linalg.norm(sol.x[4:7]):.3f} m/s2 | rms {rms:.3f} m/s | span {tc[-1] - tc[0]:.0f} s")


if __name__ == "__main__":
    main()
