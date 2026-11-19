#!/usr/bin/env python3
"""Export KITTI raw (synced+rectified) ground truth as camera poses in TUM format.

Pose chain (standard KITTI devkit / pykitti convention, p_a = T_a_b p_b):
  T_w_imu      OXTS: Mercator-projected lat/lon/alt + R = Rz(yaw) Ry(pitch) Rx(roll), origin at first sample
  T_velo_imu   calib_imu_to_velo.txt (R, T)
  T_cam0_velo  calib_velo_to_cam.txt (R, T)       (unrectified cam0)
  T_cam2_velo  = T2 * R_rect_00 * T_cam0_velo     (rectified cam2; T2 = baseline from P_rect_02)
  T_w_cam2     = T_w_imu * inv(T_cam2_velo * T_velo_imu)

Output timestamps = image_02 timestamps (absolute seconds, same parsing as vio_run), GT interpolated
from OXTS samples (slerp/linear). Camera frame = OpenCV (x right, y down, z forward).

Usage: gt_kitti_raw.py --seq <drive_sync dir> --out gt_tum.txt
"""
import argparse
from pathlib import Path

import numpy as np

from common import (T_from_Rt, find_up, interp_poses, invT, load_kitti_timestamps,
                    read_kitti_calib, write_tum)


def oxts_files(oxts_dir: Path):
    """OXTS files and their frame indices (from the file name; files can be missing)."""
    files = sorted(oxts_dir.glob("*.txt"))
    return files, np.array([int(f.stem) for f in files])


def oxts_poses(files) -> np.ndarray:
    vals = np.array([np.array(f.read_text().split()[:6], dtype=np.float64) for f in files])
    lat, lon, alt, roll, pitch, yaw = vals.T
    er = 6378137.0
    scale = np.cos(np.deg2rad(lat[0]))
    tx = scale * np.deg2rad(lon) * er
    ty = scale * er * np.log(np.tan(np.deg2rad(90.0 + lat) / 2.0))
    t = np.stack([tx, ty, alt], axis=1)
    t -= t[0]
    T = []
    for i in range(len(vals)):
        cr, sr = np.cos(roll[i]), np.sin(roll[i])
        cp, sp = np.cos(pitch[i]), np.sin(pitch[i])
        cy, sy = np.cos(yaw[i]), np.sin(yaw[i])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        T.append(T_from_Rt(Rz @ Ry @ Rx, t[i]))
    return np.stack(T)


def T_cam2_imu(seq: Path) -> np.ndarray:
    c2c = read_kitti_calib(find_up(seq, "calib_cam_to_cam.txt"))
    v2c = read_kitti_calib(find_up(seq, "calib_velo_to_cam.txt"))
    i2v = read_kitti_calib(find_up(seq, "calib_imu_to_velo.txt"))
    T_velo_imu = T_from_Rt(i2v["R"].reshape(3, 3), i2v["T"])
    T_cam0_velo = T_from_Rt(v2c["R"].reshape(3, 3), v2c["T"])
    R_rect = np.eye(4)
    R_rect[:3, :3] = c2c["R_rect_00"].reshape(3, 3)
    P2 = c2c["P_rect_02"].reshape(3, 4)
    T2 = np.eye(4)
    T2[0, 3] = P2[0, 3] / P2[0, 0]
    return T2 @ R_rect @ T_cam0_velo @ T_velo_imu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    t_oxts_all = load_kitti_timestamps(a.seq / "oxts" / "timestamps.txt")
    t_img = load_kitti_timestamps(a.seq / "image_02" / "timestamps.txt")
    files, idx = oxts_files(a.seq / "oxts" / "data")
    files, idx = [f for f, k in zip(files, idx) if k < len(t_oxts_all)], idx[idx < len(t_oxts_all)]
    T_w_imu = oxts_poses(files)
    t_oxts = t_oxts_all[idx]
    if len(idx) < len(t_oxts_all):
        print(f"WARNING: {len(t_oxts_all) - len(idx)} OXTS samples missing; GT is interpolated across the gaps")

    T_w_cam2_oxts = T_w_imu @ invT(T_cam2_imu(a.seq))
    keep = (t_img >= t_oxts[0]) & (t_img <= t_oxts[-1])
    T_w_cam2 = interp_poses(t_oxts, T_w_cam2_oxts, t_img[keep])
    write_tum(a.out, t_img[keep], T_w_cam2, header=f"KITTI raw GT, cam2 (rectified), {a.seq.name}")
    print(f"wrote {keep.sum()} / {len(t_img)} poses -> {a.out}")


if __name__ == "__main__":
    main()
