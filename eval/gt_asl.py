#!/usr/bin/env python3
"""Export EuRoC / TUM-VI (EuRoC-format export) ground truth as cam0 poses in TUM format.

Both datasets give the pose of the IMU body B in a world frame W (p_RS_R, q_RS, wxyz):
  EuRoC   mav0/state_groundtruth_estimate0/data.csv   (Leica/Vicon fused, 200 Hz)
  TUM-VI  mav0/mocap0/data.csv                        (motion capture, 120 Hz)
Camera poses: T_W_C = T_W_B * T_B_C, with
  EuRoC   T_B_C = T_BS from mav0/cam0/sensor.yaml
  TUM-VI  T_B_C = IMU.T_b_c1 of ORB-SLAM3's Examples/Monocular-Inertial/TUM-VI.yaml (TUM-VI 512 calibration;
          the EuRoC-format export ships no sensor.yaml for cam0)
Output: one pose per cam0 frame inside a continuous GT segment (gaps > --max-gap not bridged), timestamps in seconds
(ns * 1e-9), matching orbslam3_run's output.

Usage: gt_asl.py --dataset euroc|tumvi --seq <sequence root (contains mav0)> --out gt_tum.txt
"""
import argparse
import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from common import interp_poses, segments_by_gap, write_tum

DEFAULT_TUMVI_YAML = Path(__file__).resolve().parents[2] / "third_party/ORB_SLAM3/Examples/Monocular-Inertial/TUM-VI.yaml"


def matrix_after(text: str, key: str) -> np.ndarray:
    """4x4 matrix from a YAML block 'key: ... data: [16 numbers]' (EuRoC sensor.yaml / OpenCV yaml)."""
    i = text.index(key)
    j = text.index("data:", i)
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text[j + 5:text.index("]", j)])
    return np.array([float(x) for x in nums[:16]]).reshape(4, 4)


def load_gt(csv: Path):
    a = np.array([l.split(",")[:8] for l in csv.read_text().splitlines() if l.strip() and not l.startswith("#")],
                 dtype=np.float64)
    a = a[np.argsort(a[:, 0])]
    T = np.repeat(np.eye(4)[None], len(a), axis=0)
    T[:, :3, :3] = Rotation.from_quat(a[:, [5, 6, 7, 4]]).as_matrix()  # wxyz -> xyzw
    T[:, :3, 3] = a[:, 1:4]
    return a[:, 0] * 1e-9, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["euroc", "tumvi"])
    ap.add_argument("--seq", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tumvi-yaml", type=Path, default=DEFAULT_TUMVI_YAML)
    ap.add_argument("--max-gap", type=float, default=0.1)
    a = ap.parse_args()

    mav0 = a.seq / "mav0"
    if a.dataset == "euroc":
        T_B_C = matrix_after((mav0 / "cam0" / "sensor.yaml").read_text(), "T_BS")
        t_gt, T_W_B = load_gt(mav0 / "state_groundtruth_estimate0" / "data.csv")
    else:
        T_B_C = matrix_after(a.tumvi_yaml.read_text(), "IMU.T_b_c1")
        t_gt, T_W_B = load_gt(mav0 / "mocap0" / "data.csv")
    T_W_C = T_W_B @ T_B_C

    t_cam = np.array([int(l.split(",")[0]) for l in (mav0 / "cam0" / "data.csv").read_text().splitlines()
                      if l.strip() and not l.startswith("#")], dtype=np.int64) * 1e-9
    t_out, T_out = [], []
    for s, e in segments_by_gap(t_gt, a.max_gap):
        if e - s < 2:
            continue
        m = (t_cam >= t_gt[s]) & (t_cam <= t_gt[e])
        if m.any():
            t_out.append(t_cam[m])
            T_out.append(interp_poses(t_gt[s:e + 1], T_W_C[s:e + 1], t_cam[m]))
    t_out, T_out = np.concatenate(t_out), np.concatenate(T_out)
    write_tum(a.out, t_out, T_out, header=f"{a.dataset} GT, cam0 (T_W_B * T_B_C), {a.seq.name}")
    print(f"wrote {len(t_out)} / {len(t_cam)} cam0 frames with GT coverage -> {a.out}")


if __name__ == "__main__":
    main()
