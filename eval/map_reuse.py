#!/usr/bin/env python3
"""Room map reuse (save in session A, reload in session B; orbslam3_run --save-map / --load-map, WebMapIO.cc).

A's trajectory is aligned to A's ground truth (Sim3, Umeyama on positions). If B relocalised into A's loaded map, B's
poses are in A's map frame, so A's alignment applied to B (no new fit) must match B's ground truth: that error is
"B in A's frame". B's own Sim3 fit is reported too (accuracy regardless of the frame), and the share of B's frames
with a pose. Ground truths of A and B must share one world frame (EuRoC machine hall, Vicon room 1, TUM-VI room).

  map_reuse.py --a-traj A/traj_tum.txt --a-gt gtA.txt --b-traj B/traj_tum.txt --b-gt gtB.txt [--b-frames N] [--name x]
Prints one line: B frames with a pose, ATE of A (own fit), B (own fit), B in A's frame (mm, RMSE).
"""
import argparse

import numpy as np


def load(path):
    try:
        d = np.loadtxt(path, comments="#", ndmin=2)
    except (OSError, ValueError):
        return np.zeros((0, 8))
    return d[np.argsort(d[:, 0])] if len(d) else np.zeros((0, 8))


def gt_at(gt, t):
    ok = (t >= gt[0, 0]) & (t <= gt[-1, 0])
    return ok, np.stack([np.interp(t[ok], gt[:, 0], gt[:, j]) for j in (1, 2, 3)], 1)


def umeyama(src, dst):  # dst ~ s R src + t
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, D, Vt = np.linalg.svd(b.T @ a / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / (a ** 2).sum(1).mean()
    return s, R, md - s * R @ ms


def rmse(e):
    return float(np.sqrt(np.mean(np.sum(e ** 2, 1)))) * 1000 if len(e) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-traj", required=True)
    ap.add_argument("--a-gt", required=True)
    ap.add_argument("--b-traj", required=True)
    ap.add_argument("--b-gt", required=True)
    ap.add_argument("--b-frames", type=int, default=0)
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    A, B = load(a.a_traj), load(a.b_traj)
    gA, gB = load(a.a_gt), load(a.b_gt)
    out = {"a": np.nan, "b": np.nan, "bA": np.nan}
    if len(A) > 10:
        ok, g = gt_at(gA, A[:, 0])
        s, R, t = umeyama(A[ok, 1:4], g)
        out["a"] = rmse((s * (R @ A[ok, 1:4].T)).T + t - g)
        if len(B) > 10:
            okb, gb = gt_at(gB, B[:, 0])
            out["bA"] = rmse((s * (R @ B[okb, 1:4].T)).T + t - gb)
    if len(B) > 10:
        okb, gb = gt_at(gB, B[:, 0])
        s2, R2, t2 = umeyama(B[okb, 1:4], gb)
        out["b"] = rmse((s2 * (R2 @ B[okb, 1:4].T)).T + t2 - gb)
    cov = len(B) / a.b_frames if a.b_frames else np.nan
    print(f"{a.name}: B posed {len(B)} ({cov:.3f}) | ATE mm: A {out['a']:.1f}  B own {out['b']:.1f}  B in A's frame {out['bA']:.1f}")


if __name__ == "__main__":
    main()
