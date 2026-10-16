#!/usr/bin/env python3
"""Estimate the camera-IMU time offset from a visual trajectory and the gyroscope (claude/ISSUES.md G15).

The camera's angular velocity, differentiated from the tracked poses (visual-only SLAM is enough: rotation needs no
scale), is compared with the gyro rate rotated into the camera frame, over a range of time shifts. The offset is the
shift that minimises the squared difference (and maximises the correlation of the rate magnitudes, reported as a
cross-check). Convention as scripts/degrade_imu.py: reported IMU time = camera time + offset (> 0: IMU stamps late),
i.e. the value for Web.Gyro.TimeOffset.

Inputs:
  --traj  TUM trajectory of the camera (t tx ty tz qx qy qz qw; Twc), e.g. orbslam3_run traj_tum.txt (mono)
  --imu   ASL IMU csv (t[ns], w_x, w_y, w_z, a_x, a_y, a_z) or a "t wx wy wz" text file (t in s)
  --tbc   settings yaml with IMU.T_b_c1 (camera -> IMU body), or "phone" for the browser rear camera diag(1,-1,-1)

  --ar-log  instead of --traj/--imu: an AR page session log recorded with the option "Record gyro + poses"
            ("imu t wx wy wz" and "pose t tx ty tz qx qy qz qw" lines on one clock; --tbc phone)

  estimate_time_offset.py --traj traj_tum.txt --imu mav0/imu0/data.csv --tbc TUM-VI_mono_inertial.yaml
  estimate_time_offset.py --ar-log results/browser/ar_<time>.log --tbc phone
"""
import argparse
import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def load_rbc(arg):
    if arg == "phone":
        return np.diag([1.0, -1.0, -1.0])
    text = Path(arg).read_text()
    data = re.search(r"IMU\.T_b_c1:\s*!!opencv-matrix.*?data:\s*\[(.*?)\]", text, re.S).group(1)
    return np.array([float(v) for v in data.replace("\n", " ").split(",")]).reshape(4, 4)[:3, :3]


def load_imu(path):
    p = Path(path)
    if p.suffix == ".csv":
        d = np.loadtxt(p, delimiter=",", comments="#")
        return d[:, 0] * 1e-9, d[:, 1:4]
    d = np.loadtxt(p)
    return d[:, 0], d[:, 1:4]


def load_ar_log(path):
    """(trajectory rows t tx ty tz qx qy qz qw, imu times, imu rates) from an AR session log (web/ar.js 'rec')."""
    poses, imu = [], []
    for line in Path(path).read_text(errors="replace").splitlines():
        p = line.split()
        if len(p) == 10 and p[1] == "pose":
            poses.append([float(v) for v in p[2:]])
        elif len(p) == 6 and p[1] == "imu":
            imu.append([float(v) for v in p[2:]])
    poses, imu = np.array(poses), np.array(imu)
    poses = poses[np.concatenate([[True], np.diff(poses[:, 0]) > 0])]  # one pose per frame time
    imu = imu[np.argsort(imu[:, 0], kind="stable")]
    imu = imu[np.concatenate([[True], np.diff(imu[:, 0]) > 0])]
    return poses, imu[:, 0], imu[:, 1:4]


def camera_rates(traj):
    """Angular velocity of the camera (camera frame) at the midpoints of consecutive poses."""
    t = traj[:, 0]
    R = Rotation.from_quat(traj[:, 4:8])  # Twc rotation
    dt = np.diff(t)
    keep = (dt > 1e-3) & (dt < 0.2)  # skip gaps (lost tracking, new maps)
    rel = R[:-1].inv() * R[1:]       # R_c(k) -> c(k+1), in camera k
    w = rel.as_rotvec() / dt[:, None]
    tm = 0.5 * (t[:-1] + t[1:])
    return tm[keep], w[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj")
    ap.add_argument("--imu")
    ap.add_argument("--ar-log")
    ap.add_argument("--tbc", required=True)
    ap.add_argument("--max-ms", type=float, default=200)
    ap.add_argument("--step-ms", type=float, default=1)
    a = ap.parse_args()

    if a.ar_log:
        traj, ti, wi = load_ar_log(a.ar_log)
        name = Path(a.ar_log).name
    else:
        traj = np.loadtxt(a.traj)
        ti, wi = load_imu(a.imu)
        name = Path(a.traj).parent.name
    tc, wc = camera_rates(traj)
    Rcb = load_rbc(a.tbc).T
    wib = wi @ Rcb.T  # gyro rate in the camera frame
    # smooth the gyro over each camera interval: average the IMU rate on a window of the frame period
    period = np.median(np.diff(np.unique(traj[:, 0])))
    lags = np.arange(-a.max_ms, a.max_ms + a.step_ms / 2, a.step_ms) / 1000
    cs = np.vstack([np.zeros((1, 3)), np.cumsum(wib * np.gradient(ti)[:, None], axis=0)])  # integral of the rate
    best = None
    rows = []
    for lag in lags:
        # mean gyro rate over [tc - period/2, tc + period/2] in IMU time (camera time + lag)
        lo, hi = tc - period / 2 + lag, tc + period / 2 + lag
        ok = (lo >= ti[0]) & (hi <= ti[-1])
        if ok.sum() < 50:
            rows.append((lag, np.nan, np.nan))
            continue
        il, ih = np.searchsorted(ti, lo[ok]), np.searchsorted(ti, hi[ok])
        wg = (cs[ih] - cs[il]) / (ti[np.minimum(ih, ti.size - 1)] - ti[il] + 1e-9)[:, None]
        err = np.mean(np.sum((wg - wc[ok]) ** 2, axis=1))
        corr = np.corrcoef(np.linalg.norm(wg, axis=1), np.linalg.norm(wc[ok], axis=1))[0, 1]
        rows.append((lag, err, corr))
        if best is None or err < best[1]:
            best = (lag, err, corr)
    rows = np.array(rows)
    c_best = rows[np.nanargmax(rows[:, 2])]
    print(f"{name}: offset {best[0] * 1000:+.0f} ms (rate error {best[1]:.4f} rad²/s², "
          f"magnitude correlation {best[2]:.3f}); correlation peak at {c_best[0] * 1000:+.0f} ms; "
          f"camera rates {tc.size}, frame period {period * 1000:.1f} ms")


if __name__ == "__main__":
    main()
