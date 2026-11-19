#!/usr/bin/env python3
"""How long can the AR page keep showing the tree through a tracking loss? (RESEARCH.md D3, 2026-10-17)

From many start times t0 along the ground truth, the display pose is propagated for a loss of duration T in two ways:
  freeze  the last good pose is kept (what a renderer would do without sensors)
  gyro    the last good rotation is integrated with the gyro (camera centre kept), as the dead-reckoning display would
Both are compared with the ground truth at t0 + T: rotation error (deg), camera displacement ignored by both (cm), and the
overlay error of an anchor 1.5 m in front of the camera (deg of view angle: rotation error + displacement / distance).
Gyro bias is not corrected (raw dataset gyro).

Usage: deadreckon_sim.py --gt <tum cam0 GT> --imu <mav0/imu0/data.csv> --settings <yaml with IMU.T_b_c1> [--step 0.5]
"""
import argparse
import re

import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp


def load_tbc(path):
    txt = open(path).read()
    m = re.search(r"IMU\.T_b_c1:\s*!!opencv-matrix.*?data:\s*\[(.*?)\]", txt, re.S)
    return np.array([float(v) for v in m.group(1).replace("\n", " ").split(",")]).reshape(4, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--imu", required=True)
    ap.add_argument("--settings", required=True)
    ap.add_argument("--step", type=float, default=0.5, help="seconds between loss start times")
    ap.add_argument("--dist", type=float, default=1.5, help="anchor distance (m) for the overlay error")
    a = ap.parse_args()

    gt = np.loadtxt(a.gt, comments="#")
    tg, pg, qg = gt[:, 0], gt[:, 1:4], gt[:, 4:8]
    slerp = Slerp(tg, Rot.from_quat(qg))
    imu = np.loadtxt(a.imu, delimiter=",", comments="#")
    ti, w = imu[:, 0] * 1e-9, imu[:, 1:4]
    Rbc = load_tbc(a.settings)[:3, :3]

    def gyro_rot(t0, t1):  # body rotation t0 -> t1, trapezoid on the samples inside
        k = (ti >= t0) & (ti <= t1)
        ts, ws = ti[k], w[k]
        R = Rot.identity()
        for j in range(len(ts) - 1):
            R = R * Rot.from_rotvec(0.5 * (ws[j] + ws[j + 1]) * (ts[j + 1] - ts[j]))
        return R

    Rcb = Rot.from_matrix(Rbc.T)
    print(f"{a.gt.split('/')[-1]}: loss start every {a.step} s, anchor at {a.dist} m")
    print(f"{'T (s)':>6} | {'freeze rot':>10} {'gyro rot':>9} | {'displ cm':>8} | {'freeze overlay':>14} {'gyro overlay':>12}  (median / p90, deg)")
    for T in (0.25, 0.5, 1.0, 2.0):
        fr, gr, dc, fo, go = [], [], [], [], []
        for t0 in np.arange(max(tg[0], ti[0]) + 0.1, min(tg[-1], ti[-1]) - T - 0.1, a.step):
            t1 = t0 + T
            R0, R1 = slerp([t0])[0], slerp([t1])[0]
            p0, p1 = (np.array([np.interp(t, tg, pg[:, i]) for i in range(3)]) for t in (t0, t1))
            Rrel = R0.inv() * R1                                  # true camera rotation over the loss (camera frame)
            dRc = Rcb * gyro_rot(t0, t1) * Rcb.inv()              # gyro rotation in the camera frame
            efr = np.degrees(Rrel.magnitude())
            egr = np.degrees((Rrel.inv() * dRc).magnitude())
            disp = np.linalg.norm(p1 - p0)
            ang = np.degrees(np.arctan2(disp, a.dist))
            fr.append(efr); gr.append(egr); dc.append(100 * disp); fo.append(efr + ang); go.append(egr + ang)
        q = lambda v: f"{np.median(v):5.1f}/{np.percentile(v, 90):5.1f}"
        print(f"{T:6.2f} | {q(fr):>10} {q(gr):>9} | {q(dc):>8} | {q(fo):>14} {q(go):>12}")


if __name__ == "__main__":
    main()
