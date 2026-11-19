#!/usr/bin/env python3
"""Objective checks of sensor-frame conventions, independent of any SLAM output.

The idea: the gyroscope measures angular velocity in the IMU frame. Ground-truth orientation,
if expressed for the IMU frame with the right transform direction and clock offset, must
reproduce that angular velocity. Likewise, a specific-force accelerometer reads ~9.81 m/s^2
when the platform is not accelerating.

  kitti  : which OXTS gyro channels (wx,wy,wz vs wf,wl,wu) are in the OXTS-orientation body frame,
           and whether the accelerometer channels include gravity.         (ISSUES E6, E7)
  tumvie : whether T_imu_marker is imu-from-marker (Basalt) or the inverse, and the residual
           mocap<->IMU clock offset after applying the calibration offsets.  (ISSUES E15)

Usage: check_gt_frames.py kitti  --seq <drive_sync dir>
       check_gt_frames.py tumvie --seq <sequence dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from common import T_from_pose_dict, load_kitti_timestamps, load_tumvie_calib, segments_by_gap


def body_rates(t, R):
    """Angular velocity in the body frame between consecutive orientations: log(R_k^T R_k+1)/dt."""
    dt = np.diff(t)
    rel = Rotation.from_matrix(np.einsum("nji,njk->nik", R[:-1], R[1:]))
    return 0.5 * (t[:-1] + t[1:]), rel.as_rotvec() / dt[:, None]


def rms(x):
    return float(np.sqrt(np.mean(np.sum(x * x, axis=1))))


def check_kitti(seq: Path):
    files = sorted((seq / "oxts" / "data").glob("*.txt"))
    v = np.array([np.array(f.read_text().split()[:25], dtype=np.float64) for f in files])
    t = load_kitti_timestamps(seq / "oxts" / "timestamps.txt")[[int(f.stem) for f in files]]  # pair by file index
    roll, pitch, yaw = v[:, 3], v[:, 4], v[:, 5]
    R = Rotation.from_euler("ZYX", np.stack([yaw, pitch, roll], axis=1)).as_matrix()  # Rz*Ry*Rx
    tm, w_gt = body_rates(t, R)
    mid = lambda a: 0.5 * (a[:-1] + a[1:])
    print(f"KITTI {seq.name}: {len(t)} OXTS samples, mean rate {1/np.mean(np.diff(t)):.1f} Hz")
    print(f"  |w_gt| rms = {rms(w_gt):.4f} rad/s")
    for name, cols in (("wx,wy,wz (idx 17-19)", [17, 18, 19]), ("wf,wl,wu (idx 20-22)", [20, 21, 22])):
        w = mid(v[:, cols])
        print(f"  gyro {name}: rms(gyro - w_gt) = {rms(w - w_gt):.4f} rad/s")
    for name, cols in (("ax,ay,az (idx 11-13)", [11, 12, 13]), ("af,al,au (idx 14-16)", [14, 15, 16])):
        n = np.linalg.norm(v[:, cols], axis=1)
        print(f"  accel {name}: |a| mean = {n.mean():.3f}, median = {np.median(n):.3f} m/s^2 "
              f"(~9.8 => specific force incl. gravity, ~0 => gravity removed)")


def check_tumvie(seq: Path, max_gap: float = 0.1):
    cam = load_tumvie_calib(seq)
    raw = json.loads((seq / "mocap-imu-calibration.json").read_text())
    moc = raw.get("value0", raw)
    moc_off = float(moc.get("mocap_time_offset_ns", 0)) * 1e-9

    imu = np.array([l.split()[:7] for l in (seq / "imu_data.txt").read_text().splitlines()
                    if l.strip() and not l.startswith("#")], dtype=np.float64)
    t_imu, gyro, acc = imu[:, 0] * 1e-6, imu[:, 1:4], imu[:, 4:7]

    m = np.array([l.split()[:8] for l in (seq / "mocap_data.txt").read_text().splitlines()
                  if l.strip() and not l.startswith("#")], dtype=np.float64)
    m = m[np.argsort(m[:, 0])]
    t_moc = m[:, 0] * 1e-6 + moc_off
    R_mm = Rotation.from_quat(m[:, 4:8]).as_matrix()  # R_mocap_marker
    R_im = T_from_pose_dict(moc["T_imu_marker"])[:3, :3]

    print(f"TUM-VIE {seq.name}: {len(t_imu)} IMU samples ({1/np.median(np.diff(t_imu)):.0f} Hz), "
          f"{len(t_moc)} mocap samples ({1/np.median(np.diff(t_moc)):.0f} Hz), mocap_time_offset={moc_off*1e3:.3f} ms")
    an = np.linalg.norm(acc, axis=1)
    print(f"  accel |a| median = {np.median(an):.3f} m/s^2 (~9.8 => specific force)")

    segs = [(s, e) for s, e in segments_by_gap(t_moc, max_gap) if e - s > 50]
    for label, R_marker_imu in (("T_imu_marker = imu-from-marker (Basalt)", R_im.T),
                                ("T_imu_marker = marker-from-imu (inverse)", R_im)):
        best = None
        for shift in np.arange(-0.050, 0.0505, 0.001):
            errs = []
            for s, e in segs:
                R = np.einsum("nij,jk->nik", R_mm[s:e + 1], R_marker_imu)
                tm, w_gt = body_rates(t_moc[s:e + 1], R)
                tq = tm + shift
                ok = (tq >= t_imu[0]) & (tq <= t_imu[-1])
                w_imu = np.stack([np.interp(tq[ok], t_imu, gyro[:, k]) for k in range(3)], axis=1)
                errs.append(w_imu - w_gt[ok])
            err = np.concatenate(errs)
            r = rms(err)
            if best is None or r < best[1]:
                best = (shift, r)
        print(f"  {label}: best rms(gyro - w_gt) = {best[1]:.4f} rad/s at residual clock shift {best[0]*1e3:+.0f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["kitti", "tumvie"])
    ap.add_argument("--seq", required=True, type=Path)
    a = ap.parse_args()
    check_kitti(a.seq) if a.dataset == "kitti" else check_tumvie(a.seq)


if __name__ == "__main__":
    main()
