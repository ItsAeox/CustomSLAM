#!/usr/bin/env python3
"""Metric scale of a monocular trajectory from the accelerometer by linear alignment with velocities as unknowns
(after VINS-Mono's visual-inertial initialisation, Qin et al. 2018, and Martinelli's closed form): no numeric
differentiation of the visual positions (eval/scale_from_accel.py differentiates them, which amplifies pose jitter).

Nodes = tracked frames every --dt s (segments split at tracking gaps > 0.2 s). Between consecutive nodes i -> j
(T = t_j - t_i), with the visual rotation R_wb = R_wc R_cb at every IMU sample:
    s (p_j - p_i) = v_i T + 1/2 g_w T^2 + dP_ij - JP_ij b_a          (position, 3 rows)
    v_j           = v_i   + g_w T       + dV_ij - JV_ij b_a          (velocity, 3 rows)
dP / dV: double / single integral of R_wb f; JP / JV: the same of R_wb (bias Jacobians). Unknowns: s, g_w (3), b_a (3)
and v (3 per node, metres / s). Linear weighted least squares (rows / their 1-sigma: accelerometer noise --sig-a,
visual position jitter --sig-p in metres at the estimated scale), then |g_w| = 9.81 (soft, Gauss-Newton).
The camera-IMU lever arm is ignored (as scale_from_accel.py).

  scale_linear_align.py --traj traj_tum.txt --imu mav0/imu0/data.csv --tbc settings.yaml [--gt gt.txt | --s-true S]
     [--dt 0.1] [--offset-ms 0] [--max-s 0] [--from-s 0] [--sig-a 0.15] [--sig-p 0.005]
Prints one line like scale_from_accel.py: nodes, s_est, true, ratio, 1-sigma, |g| linear, bias, rms.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import segments_by_gap  # noqa: E402
from scale_from_accel import load_imu, load_tbc, umeyama_scale  # noqa: E402

G = 9.81


def preintegrate(ti, acc, slerp, Rcb, t0, t1, step=0.005):
    """dV, dP (world, R_wb f integrated) and JV, JP over [t0, t1] on a fine grid (linear interpolation of f)."""
    n = max(2, int(np.ceil((t1 - t0) / step)) + 1)
    tg = np.linspace(t0, t1, n)
    fg = np.stack([np.interp(tg, ti, acc[:, k]) for k in range(3)], 1)
    Rg = (slerp(tg) * Rotation.from_matrix(Rcb)).as_matrix()
    h = np.diff(tg)
    am = 0.5 * (np.einsum("kij,kj->ki", Rg[:-1], fg[:-1]) + np.einsum("kij,kj->ki", Rg[1:], fg[1:]))  # world accel, midpoint
    Rm = 0.5 * (Rg[:-1] + Rg[1:])
    dV = np.zeros(3); dP = np.zeros(3); JV = np.zeros((3, 3)); JP = np.zeros((3, 3))
    for k in range(len(h)):
        dP += dV * h[k] + 0.5 * am[k] * h[k] ** 2
        JP += JV * h[k] + 0.5 * Rm[k] * h[k] ** 2
        dV += am[k] * h[k]
        JV += Rm[k] * h[k]
    return dV, dP, JV, JP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--imu", required=True)
    ap.add_argument("--tbc", required=True)
    ap.add_argument("--gt")
    ap.add_argument("--s-true", type=float, default=0.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--offset-ms", type=float, default=0.0)
    ap.add_argument("--max-s", type=float, default=0.0)
    ap.add_argument("--from-s", type=float, default=0.0)
    ap.add_argument("--sig-a", type=float, default=0.15)
    ap.add_argument("--sig-p", type=float, default=0.005)
    ap.add_argument("--bias-prior", type=float, default=0.0, help="1-sigma of an accelerometer bias prior (m/s2; 0 = none)")
    ap.add_argument("--inverse", action="store_true", help="solve for 1/s in map units: the noisy visual positions are "
                    "the observations, not a regressor (no regression dilution of s)")
    ap.add_argument("--rate-range", type=lambda v: tuple(float(x) for x in v.split(",")), default=None,
                    help="lo,hi: use only steps with a mean gyro rate in [lo, hi) rad/s")
    ap.add_argument("--name", default="")
    a = ap.parse_args()

    tr = np.loadtxt(a.traj, comments="#", ndmin=2)
    tr = tr[np.argsort(tr[:, 0])]
    tr = tr[np.concatenate([[True], np.diff(tr[:, 0]) > 1e-6])]
    if a.max_s > 0:
        tr = tr[tr[:, 0] <= tr[0, 0] + a.max_s]
    if a.from_s > 0:
        tr = tr[tr[:, 0] >= tr[0, 0] + a.from_s]
    tc, pc = tr[:, 0], tr[:, 1:4]
    Rwc = Rotation.from_quat(tr[:, 4:8])
    ti, acc = load_imu(a.imu)
    ti = ti - a.offset_ms * 1e-3
    gyr = np.loadtxt(a.imu, delimiter=",", comments="#", usecols=(1, 2, 3))
    Rcb = load_tbc(a.tbc)[:3, :3].T
    name = a.name or Path(a.traj).parent.name

    # nodes: every --dt s inside gap-free segments
    segs = []
    for s0, s1 in segments_by_gap(tc, 0.2):
        if tc[s1] - tc[s0] < 1.0:
            continue
        idx = [s0]
        for i in range(s0 + 1, s1 + 1):
            if tc[i] - tc[idx[-1]] >= a.dt - 1e-6:
                idx.append(i)
        if len(idx) >= 5:
            segs.append((s0, s1, idx))
    nn = sum(len(x[2]) for x in segs)
    if nn < 20:
        print(f"{name}: too few nodes ({nn})")
        return
    nv = 3 * nn
    NX = nv + 7  # v..., g (3), s, b (3)
    iG, iS, iB = nv, nv + 3, nv + 4
    rows, rhs, sig, kind = [], [], [], []
    vbase = 0
    for s0, s1, idx in segs:
        slerp = Slerp(tc[s0:s1 + 1], Rwc[s0:s1 + 1])
        for k in range(len(idx) - 1):
            i, j = idx[k], idx[k + 1]
            T = tc[j] - tc[i]
            m = (ti >= tc[i] - 0.05) & (ti <= tc[j] + 0.05)
            if m.sum() < 3:
                continue
            if a.rate_range:  # only steps whose mean gyro rate is in [lo, hi) rad/s (rotation-related error check)
                lo, hi = a.rate_range
                if not lo <= np.linalg.norm(gyr[m], axis=1).mean() < hi:
                    continue
            dV, dP, JV, JP = preintegrate(ti[m], acc[m], slerp, Rcb, tc[i], tc[j])
            vi, vj = vbase + 3 * k, vbase + 3 * (k + 1)
            dp = pc[j] - pc[i]
            for r in range(3):  # position
                row = np.zeros(NX)
                if a.inverse:  # map units: dp = v' T + 1/2 g' T^2 + q dP - JP b'   (v' = v / s, g' = g / s, b' = b / s, q = 1 / s)
                    row[iS] = dP[r]
                    row[vi + r] = T
                    row[iG + r] = 0.5 * T * T
                    row[iB:iB + 3] = -JP[r]
                    rows.append(row); rhs.append(dp[r]); kind.append(0)
                else:
                    row[iS] = dp[r]
                    row[vi + r] = -T
                    row[iG + r] = -0.5 * T * T
                    row[iB:iB + 3] = JP[r]
                    rows.append(row); rhs.append(dP[r]); kind.append(0)
                sig.append(np.hypot(0.5 * a.sig_a * T * T, np.sqrt(2) * a.sig_p))
            for r in range(3):  # velocity
                row = np.zeros(NX)
                row[vj + r] = 1.0
                row[vi + r] = -1.0
                row[iG + r] = -T
                row[iB:iB + 3] = JV[r]
                if a.inverse:  # v'_j - v'_i - g' T + JV b' - q dV = 0
                    row[iS] = -dV[r]
                    rows.append(row); rhs.append(0.0); kind.append(1)
                else:
                    rows.append(row); rhs.append(dV[r]); kind.append(1)
                sig.append(a.sig_a * T)
        vbase += 3 * len(idx)
    if a.bias_prior > 0:  # b_a ~ N(0, sigma^2): with little rotation gravity and bias are inseparable (ORB-SLAM3 uses a prior too)
        for r in range(3):
            row = np.zeros(NX); row[iB + r] = 1.0
            rows.append(row); rhs.append(0.0); sig.append(a.bias_prior); kind.append(2)
    A, b, w = np.array(rows), np.array(rhs), 1.0 / np.array(sig)
    Aw, bw = A * w[:, None], b * w
    H = Aw.T @ Aw
    x = np.linalg.solve(H + 1e-9 * np.eye(NX), Aw.T @ bw)
    to_s = (lambda q: 1.0 / q if q else np.inf) if a.inverse else (lambda v: v)  # unknown -> metres per map unit
    s_lin = to_s(x[iS])
    g_lin = np.linalg.norm(x[iG:iG + 3]) * (s_lin if a.inverse else 1.0)  # inverse: g' = g / s
    r = Aw @ x - bw
    chi2 = r @ r / max(1, len(bw) - NX)
    cov = np.linalg.inv(H + 1e-9 * np.eye(NX)) * chi2
    rel = np.sqrt(max(cov[iS, iS], 0)) / abs(x[iS]) if x[iS] else np.inf  # relative 1-sigma (same for s and 1/s)
    # refinement: |g| = 9.81 (inverse: |g'| = 9.81 q), soft (weight 1e3 in whitened units), Gauss-Newton
    if g_lin > 1e-6:
        x[iG:iG + 3] *= G / g_lin
    for _ in range(10):
        gn = np.linalg.norm(x[iG:iG + 3])
        J = np.zeros(NX); J[iG:iG + 3] = x[iG:iG + 3] / gn
        cval = gn - G
        if a.inverse:
            J[iS] = -G
            cval = gn - G * x[iS]
        c = 1e3
        Hc = H + c * c * np.outer(J, J)
        grad = Aw.T @ (Aw @ x - bw) + c * c * J * cval
        dx = -np.linalg.solve(Hc + 1e-9 * np.eye(NX), grad)
        x += dx
        if np.linalg.norm(dx) < 1e-12:
            break
    s_est = to_s(x[iS])
    pos, vel = np.array(kind) == 0, np.array(kind) == 1
    rms_p = np.sqrt(np.mean((A[pos] @ x - b[pos]) ** 2))
    rms_v = np.sqrt(np.mean((A[vel] @ x - b[vel]) ** 2))

    s_gt = a.s_true or np.nan
    if a.gt:
        gt = np.loadtxt(a.gt, comments="#", ndmin=2)
        ok = (tc >= gt[0, 0]) & (tc <= gt[-1, 0])
        g = np.stack([np.interp(tc[ok], gt[:, 0], gt[:, k]) for k in (1, 2, 3)], 1)
        s_gt = umeyama_scale(pc[ok], g)
    print(f"{name}: nodes {nn} | s_est {s_est:.4f} s_gt {s_gt:.4f} ratio {s_est / s_gt:.3f} | 1-sigma {100 * rel:.1f} % | "
          f"s linear {s_lin:.4f} | |g| linear {g_lin:.2f} | bias {np.linalg.norm(x[iB:iB + 3]) * (s_est if a.inverse else 1):.3f} m/s2 | "
          f"rms pos {rms_p:.4f} vel {rms_v:.4f} ({'map units' if a.inverse else 'm, m/s'}) | span {tc[-1] - tc[0]:.0f} s")


if __name__ == "__main__":
    main()
