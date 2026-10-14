"""Shared helpers for ground-truth export and evaluation (TUM trajectory format)."""
from __future__ import annotations

import calendar
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def T_from_Rt(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invT(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def T_from_pose_dict(d) -> np.ndarray:
    """Basalt-style {px,py,pz,qx,qy,qz,qw}."""
    R = Rotation.from_quat([d["qx"], d["qy"], d["qz"], d["qw"]]).as_matrix()
    return T_from_Rt(R, [d["px"], d["py"], d["pz"]])


def write_tum(path: Path, t: np.ndarray, T: np.ndarray, header: str = "") -> None:
    q = Rotation.from_matrix(T[:, :3, :3]).as_quat()  # x y z w
    p = T[:, :3, 3]
    with open(path, "w") as f:
        if header:
            f.write(f"# {header}\n")
        f.write("# t tx ty tz qx qy qz qw\n")
        for i in range(len(t)):
            f.write(f"{t[i]:.9f} {p[i,0]:.9f} {p[i,1]:.9f} {p[i,2]:.9f} "
                    f"{q[i,0]:.12f} {q[i,1]:.12f} {q[i,2]:.12f} {q[i,3]:.12f}\n")


def interp_poses(t_src: np.ndarray, T_src: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """Linear translation + slerp rotation. Queries must lie inside [t_src[0], t_src[-1]]."""
    p = np.stack([np.interp(t_query, t_src, T_src[:, k, 3]) for k in range(3)], axis=1)
    rot = Slerp(t_src, Rotation.from_matrix(T_src[:, :3, :3]))(t_query).as_matrix()
    out = np.repeat(np.eye(4)[None], len(t_query), axis=0)
    out[:, :3, :3] = rot
    out[:, :3, 3] = p
    return out


def segments_by_gap(t: np.ndarray, max_gap_s: float):
    """Split a sorted time array into (start, end_inclusive) index ranges at gaps > max_gap_s."""
    cuts = np.where(np.diff(t) > max_gap_s)[0]
    starts = np.concatenate([[0], cuts + 1])
    ends = np.concatenate([cuts, [len(t) - 1]])
    return list(zip(starts.tolist(), ends.tolist()))


# ---------------------------------------------------------------- KITTI raw

def parse_kitti_timestamp(line: str) -> float:
    """'2011-09-26 13:02:25.964389445' -> seconds. UTC day + time-of-day, matching vio_run (timegm)."""
    date_part, time_part = line.strip().split(" ")
    hms, _, frac = time_part.partition(".")
    Y, M, D = (int(x) for x in date_part.split("-"))
    hh, mm, ss = (int(x) for x in hms.split(":"))
    frac = (frac + "000000000")[:9]
    return calendar.timegm((Y, M, D, 0, 0, 0)) + hh * 3600 + mm * 60 + ss + float("0." + frac)


def load_kitti_timestamps(path: Path) -> np.ndarray:
    return np.array([parse_kitti_timestamp(l) for l in Path(path).read_text().splitlines() if l.strip()])


def read_kitti_calib(path: Path) -> dict:
    out = {}
    for line in Path(path).read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        try:
            out[k.strip()] = np.array([float(x) for x in v.split()])
        except ValueError:
            pass  # e.g. calib_time
    return out


def find_up(seq: Path, name: str) -> Path:
    for d in (seq, seq.parent):
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(f"{name} not found in {seq} or its parent")


def load_tumvie_calib(seq: Path) -> dict:
    raw = json.loads((seq / "camera-calibration.json").read_text())
    return raw.get("value0", raw)
