#!/usr/bin/env python3
"""Convert an IMU-body trajectory (T_w_b, TUM format) to a camera trajectory: T_w_c = T_w_b * T_b_c.

ORB-SLAM3 saves IMU-body poses in inertial modes; ground truth here is the camera pose.
T_b_c is taken from the dataset calibration (TUM-VIE: camera-calibration.json T_imu_cam[0]).

Usage: body_to_cam.py --in f_body.txt --out f_cam.txt --tumvie-seq <sequence dir>
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from common import T_from_pose_dict, load_tumvie_calib, write_tum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tumvie-seq", required=True, type=Path)
    a = ap.parse_args()

    T_b_c = T_from_pose_dict(load_tumvie_calib(a.tumvie_seq)["T_imu_cam"][0])
    rows = np.array([l.split()[:8] for l in a.inp.read_text().splitlines()
                     if l.strip() and not l.startswith("#")], dtype=np.float64)
    T = np.repeat(np.eye(4)[None], len(rows), axis=0)
    T[:, :3, :3] = Rotation.from_quat(rows[:, 4:8]).as_matrix()
    T[:, :3, 3] = rows[:, 1:4]
    write_tum(a.out, rows[:, 0], T @ T_b_c, header=f"camera poses from {a.inp.name} (T_w_b * T_b_c)")
    print(f"wrote {len(rows)} camera poses -> {a.out}")


if __name__ == "__main__":
    main()
