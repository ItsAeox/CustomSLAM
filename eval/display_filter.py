#!/usr/bin/env python3
"""Does a 1-euro filter on the displayed pose help the AR overlay? (RESEARCH.md D1, 2026-10-17)

The AR page renders every pose on the frame it was computed from, so the overlay error of a frame is that frame's pose
error. A causal 1-euro filter (Casiez et al., CHI 2012) on position and rotation removes high-frequency estimation noise
but lags real motion. For each run: Sim3-align the estimate to the ground truth (Umeyama on matched stamps), filter the
aligned poses causally, and compare with the ground truth per frame: position error (mm), rotation error (deg) and the
frame-to-frame jitter of the error (mm, deg), before and after filtering.

Usage: display_filter.py <run dir with traj_tum.txt> ... --gt-dir <dir of <seq>.txt GT files>
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / xs.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


class OneEuro:
    def __init__(self, mincutoff, beta, dcutoff=1.0):
        self.mc, self.beta, self.dc = mincutoff, beta, dcutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t, x):
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = max(t - self.t, 1e-4)
        dx = (x - self.x) / dt
        a_d = self.alpha(self.dc, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.mc + self.beta * np.linalg.norm(self.dx)
        a = self.alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x.copy()


def errors(t, p, q, gt_t, gt_p, gt_slerp):
    gp = np.stack([np.interp(t, gt_t, gt_p[:, i]) for i in range(3)], 1)
    ep = np.linalg.norm(p - gp, axis=1) * 1000
    er = np.degrees((gt_slerp(t).inv() * Rot.from_quat(q)).magnitude())
    jp = np.linalg.norm(np.diff(p - gp, axis=0), axis=1) * 1000
    rel = gt_slerp(t[1:]).inv() * gt_slerp(t[:-1])
    jr = np.degrees((Rot.from_quat(q[1:]).inv() * Rot.from_quat(q[:-1]) * rel.inv()).magnitude())
    return ep, er, jp, jr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--params", default="1.0:0.0,1.0:0.5,0.5:1.0,2.0:1.0,4.0:2.0", help="mincutoff:beta list (position, m/s)")
    a = ap.parse_args()
    params = [tuple(map(float, s.split(":"))) for s in a.params.split(",")]
    rows = {p: [] for p in [None] + params}
    for run in a.runs:
        run = Path(run)
        m = json.loads((run / "metrics.json").read_text())
        gt = np.loadtxt(Path(a.gt_dir) / Path(m["gt"]).name, comments="#")  # metrics.json holds the WSL path
        est = np.loadtxt(run / "traj_tum.txt", comments="#")
        est = est[(est[:, 0] >= gt[0, 0]) & (est[:, 0] <= gt[-1, 0])]
        if len(est) < 50:
            continue
        gt_t, gt_p = gt[:, 0], gt[:, 1:4]
        gs = Slerp(gt_t, Rot.from_quat(gt[:, 4:8]))
        t = est[:, 0]
        gp = np.stack([np.interp(t, gt_t, gt_p[:, i]) for i in range(3)], 1)
        s, R, tr = umeyama(est[:, 1:4], gp)
        p = (s * (R @ est[:, 1:4].T)).T + tr
        q = (Rot.from_matrix(R) * Rot.from_quat(est[:, 4:8])).as_quat()
        for k in range(1, len(q)):  # continuous hemisphere for component-wise filtering
            if np.dot(q[k], q[k - 1]) < 0:
                q[k] = -q[k]
        for prm in rows:
            if prm is None:
                fp, fq = p, q
            else:
                fpos, fquat = OneEuro(prm[0], prm[1]), OneEuro(prm[0], prm[1] * 2.0)  # rotation beta per rad/s
                fp = np.array([fpos(ti, pi) for ti, pi in zip(t, p)])
                fq = np.array([fquat(ti, qi) for ti, qi in zip(t, q)])
                fq /= np.linalg.norm(fq, axis=1, keepdims=True)
            rows[prm].append([np.sqrt(np.mean(e ** 2)) for e in errors(t, fp, fq, gt_t, gt_p, gs)])
    print(f"{len(rows[None])} runs; RMSE over frames, mean over runs: position mm, rotation deg, jitter mm, jitter deg")
    for prm, r in rows.items():
        r = np.array(r).mean(0)
        name = "unfiltered" if prm is None else f"1euro mincutoff {prm[0]} Hz beta {prm[1]}"
        print(f"  {name:32s} pos {r[0]:7.1f}  rot {r[1]:5.2f}  jitter {r[2]:6.2f} mm {r[3]:5.3f} deg")


if __name__ == "__main__":
    main()
