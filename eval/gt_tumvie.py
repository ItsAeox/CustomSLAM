#!/usr/bin/env python3
"""Export TUM-VIE mocap ground truth as left-camera poses in TUM format.

Convention (Basalt calibration files, p_a = T_a_b p_b):
  mocap_data.txt               t_us px py pz qx qy qz qw  = T_mocap_marker(t)
  mocap-imu-calibration.json   T_imu_marker, mocap_time_offset_ns
  camera-calibration.json      T_imu_cam[0], cam_time_offset_ns
  T_mocap_cam(t) = T_mocap_marker(t) * inv(T_imu_marker) * T_imu_cam

Clocks: everything is mapped to the IMU clock: t_cam + cam_time_offset, t_mocap + mocap_time_offset
(the runner uses the same camera times). Direction of T_imu_marker and the offsets are verified
independently of any SLAM output by check_gt_frames.py (gyro vs. mocap angular velocity).

Output: one pose per left-camera frame that lies inside a continuous mocap segment
(gaps > --max-gap are not bridged). World frame = mocap frame.

Usage: gt_tumvie.py --seq <sequence dir> --out gt_tum.txt
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common import T_from_Rt, T_from_pose_dict, interp_poses, invT, load_tumvie_calib, segments_by_gap, write_tum
from scipy.spatial.transform import Rotation


def load_mocap(path: Path):
    rows = [l.split() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    a = np.array([r[:8] for r in rows if len(r) >= 8], dtype=np.float64)
    a = a[np.argsort(a[:, 0])]
    T = np.repeat(np.eye(4)[None], len(a), axis=0)
    T[:, :3, :3] = Rotation.from_quat(a[:, 4:8]).as_matrix()
    T[:, :3, 3] = a[:, 1:4]
    return a[:, 0] * 1e-6, T


def load_camera_times(seq: Path, cam_offset_s: float) -> np.ndarray:
    vals = [float(l) for l in (seq / "left_images" / "image_timestamps_left.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    return np.array(vals) * 1e-6 + cam_offset_s


def mocap_imu_calib(seq: Path) -> dict:
    raw = json.loads((seq / "mocap-imu-calibration.json").read_text())
    return raw.get("value0", raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-gap", type=float, default=0.1, help="mocap gap (s) that splits segments")
    a = ap.parse_args()

    cam = load_tumvie_calib(a.seq)
    moc = mocap_imu_calib(a.seq)
    cam_off = float(cam.get("cam_time_offset_ns", 0)) * 1e-9
    moc_off = float(moc.get("mocap_time_offset_ns", 0)) * 1e-9

    t_moc, T_mocap_marker = load_mocap(a.seq / "mocap_data.txt")
    t_moc = t_moc + moc_off
    T_imu_marker = T_from_pose_dict(moc["T_imu_marker"])
    T_imu_cam = T_from_pose_dict(cam["T_imu_cam"][0])
    T_mocap_cam = T_mocap_marker @ invT(T_imu_marker) @ T_imu_cam

    t_cam = load_camera_times(a.seq, cam_off)
    t_out, T_out = [], []
    for s, e in segments_by_gap(t_moc, a.max_gap):
        if e - s < 2:
            continue
        m = (t_cam >= t_moc[s]) & (t_cam <= t_moc[e])
        if m.any():
            t_out.append(t_cam[m])
            T_out.append(interp_poses(t_moc[s:e + 1], T_mocap_cam[s:e + 1], t_cam[m]))
    t_out = np.concatenate(t_out)
    T_out = np.concatenate(T_out)
    write_tum(a.out, t_out, T_out, header=f"TUM-VIE mocap GT, left camera, {a.seq.name}")
    print(f"wrote {len(t_out)} / {len(t_cam)} camera frames with mocap coverage -> {a.out}")


if __name__ == "__main__":
    main()
