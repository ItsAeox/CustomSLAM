#!/usr/bin/env python3
"""Loosely coupled visual-inertial filter, prototype (claude/PLAN.md 4d, "full IMU" approach B): metric scale, gravity,
velocity and accelerometer bias of a monocular map, causal, at IMU rate, from the accelerometer, the gyro and the camera
poses the tracker reported live (trace.txt, so later bundle adjustment does not help it). After Weiss & Siegwart (ICRA
2011) and Nuetzi et al. (JIRS 2011), who put the visual scale into an EKF; here the scale is not a filter state but a
grid of hypotheses, because an EKF on the scale diverges when its start value is far off.

Per map, in the map frame, metric units (camera centre; the camera-IMU lever arm, ~1 cm, is ignored):
    p' = v,   v' = R_vb (f - b_a) + g        R_vb = visual camera rotation x camera -> IMU (IMU.T_b_c1), carried by the
                                             gyro from the last tracked frame (also through losses)
    tracked frame: p_vis = p / s             p_vis in map units, noise --sig-vis map units; s = metres per map unit
For a fixed s this is linear in (p, v, g, b_a): a bank of Kalman filters over a log grid of s (--grid points between
--s-min and --s-max) gives each s its exact marginal likelihood. |g| = 9.81 is enforced on every filter after each
update (a constraint, not evidence). Posterior over s: the likelihoods with a flat prior in ln s; estimate = parabola
peak of the log-likelihood around the best grid point; 1-sigma = the posterior spread of ln s (at least half a grid
step). --forget T: the log-likelihoods decay with time constant T s (follows a slowly changing map scale; 0 = off).
A map's filter starts when the map has settled (--start-kfs keyframes, --start-time s: early bundle adjustment changes
the scale a lot); --restart: staggered banks over sliding windows follow later scale changes; --gate: visual updates
that do not fit the most likely scale (relocalisation jumps, bad poses) are skipped. A map switch (keyframe count back
to 0) drops the banks. Lost frames: IMU propagation only.

  vio_filter.py <run dir> <sequence dir> [--imu csv (default <seq>/mav0/imu0/data.csv)] [--offset-ms -88]
               [--gt gt.txt] [--keep 30] [--sigma 0.05] [--sig-a 0.15] [--sig-vis 0.005] [--forget 0] [--restart 30] [--gate 16.27]
               [--start-kfs 10] [--start-time 2] [--out file]
Prints one line: tracked s, converged s (first after s of its map), with GT |s / s_gt - 1| over converged frames
(the reference as eval/scale_online_check.py: Sim3 of the last --keep s of the map's live positions), and the metric
ATE (SE3 alignment, no scale) of the scaled live positions s(t) p_vis(t) over the converged frames, per largest map; the
same for the online estimator (trace columns, Web.Scale.Online) where it had converged. --out: per frame t, s, 1-sigma.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_from_accel import load_tbc, umeyama_scale  # noqa: E402

G = 9.81


def load_imu(path):  # ASL csv: t [ns], w (3), a (3)
    t = np.loadtxt(path, delimiter=",", comments="#", usecols=0, dtype=np.int64)
    d = np.loadtxt(path, delimiter=",", comments="#")
    return (t - t[0]) * 1e-9 + t[0] * 1e-9, d[:, 1:4], d[:, 4:7]


def so3exp(w):
    return Rotation.from_rotvec(w).as_matrix()


class Bank:
    """Kalman filters over a grid of scales; state x = p (3, m), v (3), g (3), b_a (3)."""

    def __init__(self, s_grid, p_vis, R_vb, f_mean, a):
        self.s, self.a = s_grid, a
        J = len(s_grid)
        self.x = np.zeros((J, 12))
        self.x[:, 0:3] = s_grid[:, None] * p_vis
        g0 = -R_vb @ f_mean
        self.x[:, 6:9] = G * g0 / max(np.linalg.norm(g0), 1e-6)
        sd = np.concatenate([[0.0] * 3, [1.0] * 3, [1.0] * 3, [a.sig_ba] * 3])
        self.P = np.repeat(np.diag(sd ** 2)[None], J, 0)
        self.P[:, 0:3, 0:3] = np.eye(3) * (a.sig_vis * s_grid[:, None, None]) ** 2
        self.L = np.zeros(J)
        self.n_upd = self.skips = 0
        self.t0 = self.t_last = None

    def propagate(self, R_vb, f, dt):
        if dt <= 0:
            return
        x, P = self.x, self.P
        acc = (R_vb @ (f[None, :] - x[:, 9:12]).T).T + x[:, 6:9]
        x[:, 0:3] += x[:, 3:6] * dt + 0.5 * acc * dt * dt
        x[:, 3:6] += acc * dt
        F = np.eye(12)
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = np.eye(3) * 0.5 * dt * dt
        F[0:3, 9:12] = -0.5 * dt * dt * R_vb
        F[3:6, 6:9] = np.eye(3) * dt
        F[3:6, 9:12] = -dt * R_vb
        q = self.a.qa ** 2  # accelerometer + model noise, continuous density (m/s^2)^2 / Hz
        Q = np.zeros((12, 12))
        Q[0:3, 0:3] = np.eye(3) * q * dt ** 3 / 3
        Q[0:3, 3:6] = Q[3:6, 0:3] = np.eye(3) * q * dt ** 2 / 2
        Q[3:6, 3:6] = np.eye(3) * q * dt
        Q[6:9, 6:9] = np.eye(3) * 1e-6 * dt
        Q[9:12, 9:12] = np.eye(3) * self.a.ba_walk ** 2 * dt
        self.P = F @ P @ F.T + Q

    def update(self, z):
        J = len(self.s)
        H = np.zeros((J, 3, 12))
        H[:, :, 0:3] = np.eye(3)[None] / self.s[:, None, None]
        nu = z[None, :] - self.x[:, 0:3] / self.s[:, None]
        S = H @ self.P @ H.transpose(0, 2, 1) + np.eye(3)[None] * self.a.sig_vis ** 2
        Si = np.linalg.inv(S)
        jb = int(np.argmax(self.L))  # gate on the most likely scale: a visual jump (relocalisation, bad pose) is skipped
        if self.a.gate > 0 and self.n_upd >= 30 and nu[jb] @ Si[jb] @ nu[jb] > self.a.gate:
            self.skips += 1
            return
        self.skips = 0
        self.n_upd += 1
        K = self.P @ H.transpose(0, 2, 1) @ Si
        self.x += np.einsum("jab,jb->ja", K, nu)
        IKH = np.eye(12)[None] - K @ H
        self.P = IKH @ self.P @ IKH.transpose(0, 2, 1) + K @ (np.eye(3)[None] * self.a.sig_vis ** 2) @ K.transpose(0, 2, 1)
        self.L += -0.5 * (np.einsum("ja,jab,jb->j", nu, Si, nu) + np.linalg.slogdet(S)[1] + 3 * np.log(2 * np.pi))
        for _ in range(2):  # |g| = 9.81 (constraint, not evidence)
            g = self.x[:, 6:9]
            n = np.linalg.norm(g, axis=1)
            Hg = np.zeros((J, 1, 12))
            Hg[:, 0, 6:9] = g / n[:, None]
            Sg = Hg @ self.P @ Hg.transpose(0, 2, 1) + 1e-4
            Kg = self.P @ Hg.transpose(0, 2, 1) / Sg
            self.x += Kg[:, :, 0] * (G - n)[:, None]
            self.P = (np.eye(12)[None] - Kg @ Hg) @ self.P
        self.P = 0.5 * (self.P + self.P.transpose(0, 2, 1))

    def estimate(self):
        L = self.L - self.L.max()
        w = np.exp(L)
        w /= w.sum()
        ls = np.log(self.s)
        j = int(np.argmax(L))
        est = ls[j]
        if 0 < j < len(ls) - 1:  # parabola through the best grid point and its neighbours
            y0, y1, y2 = L[j - 1], L[j], L[j + 1]
            den = y0 - 2 * y1 + y2
            if den < 0:
                est = ls[j] + 0.5 * (y0 - y2) / den * (ls[j + 1] - ls[j])
        mean = (w * ls).sum()
        sd = max(np.sqrt((w * (ls - mean) ** 2).sum()), 0.5 * (ls[1] - ls[0]))
        return float(np.exp(est)), float(sd), j in (0, len(ls) - 1)


def se3_ate(est, gt):
    """RMSE (m) after a rigid (no scale) alignment of est onto gt."""
    me, mg = est.mean(0), gt.mean(0)
    U, D, Vt = np.linalg.svd((gt - mg).T @ (est - me) / len(est))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    return float(np.sqrt((np.linalg.norm(gt - (est - me) @ R.T - mg, axis=1) ** 2).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("seq")
    ap.add_argument("--imu")
    ap.add_argument("--offset-ms", type=float, default=-88.0, help="IMU time = camera time + offset (as scale_from_accel.py)")
    ap.add_argument("--gt")
    ap.add_argument("--keep", type=float, default=30.0)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--sig-a", type=float, default=0.15, help="accelerometer + model noise per sample at the IMU rate (m/s^2)")
    ap.add_argument("--sig-vis", type=float, default=0.005, help="visual position noise (map units)")
    ap.add_argument("--sig-ba", type=float, default=0.1, help="accelerometer bias prior 1-sigma (m/s^2)")
    ap.add_argument("--ba-walk", type=float, default=0.002, help="accelerometer bias random walk (m/s^2 / sqrt(s))")
    ap.add_argument("--grid", type=int, default=300)
    ap.add_argument("--s-min", type=float, default=0.01)
    ap.add_argument("--s-max", type=float, default=1000.0)
    ap.add_argument("--forget", type=float, default=0.0)
    ap.add_argument("--start-kfs", type=int, default=10, help="a map's filter starts once it has this many keyframes ...")
    ap.add_argument("--start-time", type=float, default=2.0, help="... and is this many seconds old (ORB-SLAM3's IMU init: 10, 2 s)")
    ap.add_argument("--restart", type=float, default=30.0, help="a new bank every restart/2 s, the oldest reports, banks live "
                    "restart s (windows of restart/2 .. restart s; follows scale changes of the map); 0 = one bank per map")
    ap.add_argument("--gate", type=float, default=16.27, help="skip a visual update whose NIS (3 dof) on the most likely scale "
                    "exceeds this (99.9 %%); 0 = off")
    ap.add_argument("--max-skips", type=int, default=10, help="a bank that skipped this many updates in a row is dropped")
    ap.add_argument("--out")
    a = ap.parse_args()
    run, seq = Path(a.run), Path(a.seq)
    ti, w_imu, f_imu = load_imu(a.imu or seq / "mav0" / "imu0" / "data.csv")
    ti = ti - a.offset_ms * 1e-3  # camera clock
    a.qa = a.sig_a / np.sqrt(len(ti) / (ti[-1] - ti[0]))  # per-sample sigma -> density
    Rcb = load_tbc(seq / "settings.yaml")[:3, :3]  # IMU.T_b_c1: camera -> body; R_vb = R_vc R_bc^T

    rows = [l.split() for l in (run / "trace.txt").read_text().splitlines()]
    tr = np.array([[float(x) for x in r[:29]] for r in rows if len(r) >= 14 and r[0].lstrip("-").isdigit()], dtype=float)
    cam = np.loadtxt(seq / "mav0" / "cam0" / "data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64)
    fr = tr[:, 0].astype(int)
    tf = cam[np.clip(fr, 0, len(cam) - 1)] * 1e-9
    ok = np.isin(tr[:, 1].astype(int), [2, 5])
    pos, quat, kfs = tr[:, 3:6], tr[:, 6:10], tr[:, 13]
    s_on = tr[:, 26] if tr.shape[1] > 27 else np.zeros(len(tr))
    rel_on = tr[:, 27] if tr.shape[1] > 27 else -np.ones(len(tr))
    mapid = np.cumsum(np.concatenate([[0], (kfs[1:] == 0) & (kfs[:-1] > 0)]))
    s_grid = np.exp(np.linspace(np.log(a.s_min), np.log(a.s_max), a.grid))

    s_est = np.zeros(len(tr))
    sd_est = np.full(len(tr), -1.0)
    edge = np.zeros(len(tr), bool)
    k_imu = 0
    banks, R_vb, t_rot, cur_map, t_map0 = [], None, None, -1, None
    for k in range(len(tr)):
        t = tf[k]
        if mapid[k] != cur_map:
            banks, cur_map, t_map0 = [], mapid[k], None
        if t_map0 is None and kfs[k] > 0:
            t_map0 = t
        # IMU samples up to this frame: propagate the banks, carry the rotation by the gyro
        while k_imu < len(ti) and ti[k_imu] <= t:
            if R_vb is not None:
                dt = ti[k_imu] - t_rot
                if dt > 0:  # midpoint rule: rotation, rate and specific force at the middle of the interval
                    w_mid = 0.5 * (w_imu[max(k_imu - 1, 0)] + w_imu[k_imu])
                    R_mid = R_vb @ so3exp(w_mid * dt / 2)
                    f = 0.5 * (f_imu[max(k_imu - 1, 0)] + f_imu[k_imu])
                    for b in banks:
                        b.propagate(R_mid, f, ti[k_imu] - b.t_last)
                        b.t_last = ti[k_imu]
                    R_vb = R_vb @ so3exp(w_mid * dt)
                    t_rot = ti[k_imu]
            k_imu += 1
        if ok[k]:
            R_vb = Rotation.from_quat(quat[k]).as_matrix() @ Rcb.T
            t_rot = t
            for b in banks:
                b.propagate(R_vb, f_imu[max(k_imu - 1, 0)], t - b.t_last)  # last stretch to the frame time
                b.t_last = t
                if a.forget > 0 and k > 0:
                    b.L *= np.exp(-max(t - tf[k - 1], 0) / a.forget)
                b.update(pos[k])
            # banks that keep rejecting the visual pose (the map jumped) or are older than --restart are dropped
            banks = [b for b in banks if b.skips <= a.max_skips and (a.restart <= 0 or t - b.t0 <= a.restart)]
            settled = t_map0 is not None and kfs[k] >= a.start_kfs and t - t_map0 >= a.start_time
            if settled and (not banks or (a.restart > 0 and t - banks[-1].t0 >= a.restart / 2)):
                m = (ti > t - 0.5) & (ti <= t)
                f_mean = f_imu[m].mean(0) if m.any() else f_imu[max(k_imu - 1, 0)]
                b = Bank(s_grid, pos[k], R_vb, f_mean, a)
                b.t0 = b.t_last = t
                banks.append(b)
        if banks:  # the oldest bank (the longest window) reports
            s_est[k], sd_est[k], edge[k] = banks[0].estimate()

    if a.out:
        np.savetxt(a.out, np.c_[tf, s_est, sd_est, ok, mapid], fmt="%.6f", header="t s sigma_ln_s tracked map")

    dt_med = np.median(np.diff(tf)) if len(tf) > 1 else 0.033
    conv = ok & (sd_est >= 0) & (sd_est < a.sigma) & ~edge
    first = None
    for m in np.unique(mapid):
        idx = np.where(mapid == m)[0]
        c = idx[conv[idx]]
        if len(c) and ok[idx].any():
            d = tf[c[0]] - tf[idx[ok[idx]][0]]
            first = d if first is None else min(first, d)
    line = f"{run}: tracked {ok.sum() * dt_med:.0f} s, converged {conv.sum() * dt_med:.0f} s"
    line += f" (first after {first:.1f} s of its map)" if first is not None else " (never)"
    if a.gt:
        gt = np.loadtxt(a.gt, comments="#", ndmin=2)
        gpos = lambda tt: np.stack([np.interp(tt, gt[:, 0], gt[:, j]) for j in (1, 2, 3)], 1)
        conv_on = ok & (s_on > 0) & (rel_on >= 0) & (rel_on < a.sigma)

        def errs(sv, cv):
            out = []
            for k in np.where(cv)[0][::5]:
                wi = np.where(ok & (mapid == mapid[k]) & (tf >= tf[k] - a.keep) & (tf <= tf[k]))[0]
                wi = wi[(tf[wi] >= gt[0, 0]) & (tf[wi] <= gt[-1, 0])]
                if len(wi) < 30:
                    continue
                g = gpos(tf[wi])
                if np.ptp(g, 0).max() < 0.3:
                    continue
                out.append(abs(sv[k] / umeyama_scale(pos[wi], g) - 1))
            return np.array(out)

        e = errs(s_est, conv)
        if len(e):
            line += (f" | |s/s_gt - 1| median {100 * np.median(e):.1f} %, p90 {100 * np.percentile(e, 90):.1f} %, "
                     f"within 5 % {100 * (e < 0.05).mean():.0f} %, last {100 * e[-1]:.1f} % ({len(e)} checks)")
        # metric ATE of the scaled live positions over the converged frames of the largest map (the same frames for the
        # online estimator where it had converged, and for the true Sim3 scale of those frames as the floor)
        big = max(np.unique(mapid), key=lambda m: (ok & (mapid == m)).sum())
        sel = conv & (mapid == big) & (tf >= gt[0, 0]) & (tf <= gt[-1, 0])
        if sel.sum() >= 30:
            g = gpos(tf[sel])
            ate = se3_ate(s_est[sel, None] * pos[sel], g)
            s_true = umeyama_scale(pos[sel], g)
            line += f" | metric ATE {100 * ate:.1f} cm over {sel.sum() * dt_med:.0f} s (true-scale floor {100 * se3_ate(s_true * pos[sel], g):.1f} cm)"
            so = sel & conv_on
            if so.sum() >= 30:
                line += f", online estimator {100 * se3_ate(s_on[so, None] * pos[so], gpos(tf[so])):.1f} cm over {so.sum() * dt_med:.0f} s"
        eo = errs(s_on, conv_on)
        if len(eo):
            line += f" | online: median {100 * np.median(eo):.1f} %, converged {conv_on.sum() * dt_med:.0f} s"
    print(line)


if __name__ == "__main__":
    main()
