#!/usr/bin/env python3
import os
import csv
import math
import glob
import datetime
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------
# Edit this if needed
# ---------------------------------------------------------------------
DATASET_ROOT = Path("dataset")

# Put only the sequences you want summarized here
SEQUENCES = [
    "2011_09_26_drive_0009_sync",
    "2011_09_26_drive_0018_sync",
    "2011_09_26_drive_0022_sync",
    "2011_09_26_drive_0023_sync",
    "2011_09_26_drive_0028_sync",
    "2011_09_26_drive_0029_sync",
    "2011_09_26_drive_0084_sync",
    "2011_09_26_drive_0086_sync",
    "2011_09_26_drive_0087_sync",
    "2011_09_26_drive_0101_sync",
    "2011_09_29_drive_0071_sync",
    "2011_10_03_drive_0047_sync",
]

OUTPUT_CSV = Path("kitti_results_summary.csv")
OUTPUT_TXT = Path("kitti_results_summary.txt")


# ---------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------
def parse_kitti_timestamp_line(line: str) -> float:
    line = line.strip()
    if not line:
        return None
    date_part, time_part = line.split(" ")
    if "." in time_part:
        hms, frac = time_part.split(".")
        frac_s = float("0." + frac)
    else:
        hms, frac_s = time_part, 0.0
    dt = datetime.datetime.fromisoformat(f"{date_part}T{hms}")
    return dt.timestamp() + frac_s


def load_kitti_timestamps(path: Path) -> np.ndarray:
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    t = np.array([parse_kitti_timestamp_line(l) for l in lines], dtype=np.float64)
    t -= t[0]
    return t


# ---------------------------------------------------------------------
# SO(3)/SE(3)
# ---------------------------------------------------------------------
def R_from_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def T_from_Rt(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invT(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -(R.T @ t)
    return Ti


def quat_from_R(R):
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m[2, 1] - m[1, 2]) / S
        y = (m[0, 2] - m[2, 0]) / S
        z = (m[1, 0] - m[0, 1]) / S
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        S = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / S
        x = 0.25 * S
        y = (m[0, 1] + m[1, 0]) / S
        z = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / S
        x = (m[0, 1] + m[1, 0]) / S
        y = 0.25 * S
        z = (m[1, 2] + m[2, 1]) / S
    else:
        S = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / S
        x = (m[0, 2] + m[2, 0]) / S
        y = (m[1, 2] + m[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def R_from_quat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def slerp(q0, q1, a):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(dot)
    s0 = math.sin((1 - a) * th) / math.sin(th)
    s1 = math.sin(a * th) / math.sin(th)
    return s0 * q0 + s1 * q1


def interp_se3(T_list, t_list, t_query):
    tq = np.asarray(t_query, dtype=np.float64)
    out = np.zeros((tq.size, 4, 4), dtype=np.float64)
    out[:] = np.eye(4)

    qs = np.array([quat_from_R(T_list[i, :3, :3]) for i in range(len(T_list))], dtype=np.float64)
    ps = np.array([T_list[i, :3, 3] for i in range(len(T_list))], dtype=np.float64)

    for k, t in enumerate(tq):
        if t <= t_list[0]:
            out[k] = T_list[0]
            continue
        if t >= t_list[-1]:
            out[k] = T_list[-1]
            continue
        j = np.searchsorted(t_list, t, side="right")
        i = j - 1
        t0, t1 = t_list[i], t_list[j]
        a = (t - t0) / max(1e-12, (t1 - t0))
        q = slerp(qs[i], qs[j], float(a))
        p = (1.0 - a) * ps[i] + a * ps[j]
        out[k] = T_from_Rt(R_from_quat(q), p)
    return out


# ---------------------------------------------------------------------
# KITTI calib parsing
# ---------------------------------------------------------------------
def parse_kitti_calib_txt(path: Path):
    d = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        d[k.strip()] = v.strip()
    return d


def mat_from_line(vals, rows, cols):
    a = np.fromstring(vals, sep=" ", dtype=np.float64)
    return a.reshape(rows, cols)


def load_T_imu_to_velo(path: Path):
    d = parse_kitti_calib_txt(path)
    R = mat_from_line(d["R"], 3, 3)
    t = mat_from_line(d["T"], 3, 1).reshape(3)
    return T_from_Rt(R, t)


def load_T_velo_to_cam0(path: Path):
    d = parse_kitti_calib_txt(path)
    R = mat_from_line(d["R"], 3, 3)
    t = mat_from_line(d["T"], 3, 1).reshape(3)
    return T_from_Rt(R, t)


def load_T_cam0_to_cam2(path: Path):
    d = parse_kitti_calib_txt(path)
    if "R_02" in d and "T_02" in d:
        R = mat_from_line(d["R_02"], 3, 3)
        t = mat_from_line(d["T_02"], 3, 1).reshape(3)
        return T_from_Rt(R, t)
    return np.eye(4, dtype=np.float64)


def build_T_imu_to_cam2(seq_dir: Path):
    T_i_v = load_T_imu_to_velo(seq_dir / "calib_imu_to_velo.txt")
    T_v_c0 = load_T_velo_to_cam0(seq_dir / "calib_velo_to_cam.txt")
    T_c0_c2 = load_T_cam0_to_cam2(seq_dir / "calib_cam_to_cam.txt")
    return T_c0_c2 @ T_v_c0 @ T_i_v


# ---------------------------------------------------------------------
# OXTS parsing
# ---------------------------------------------------------------------
def read_oxts_file(path: Path):
    vals = np.fromstring(path.read_text().strip(), sep=" ")
    if vals.size < 6:
        raise ValueError(f"Bad OXTS line in {path}")
    lat, lon, alt, roll, pitch, yaw = vals[:6]
    return float(lat), float(lon), float(alt), float(roll), float(pitch), float(yaw)


def latlon_to_enu_m(lat, lon, alt, lat0, lon0, alt0):
    R = 6378137.0
    latr = np.deg2rad(lat)
    lonr = np.deg2rad(lon)
    lat0r = np.deg2rad(lat0)
    lon0r = np.deg2rad(lon0)
    dlat = latr - lat0r
    dlon = lonr - lon0r
    east = R * dlon * np.cos(lat0r)
    north = R * dlat
    up = alt - alt0
    return np.array([east, north, up], dtype=np.float64)


def load_oxts_poses(oxts_dir: Path):
    files = sorted(glob.glob(str(oxts_dir / "*.txt")))
    if not files:
        raise FileNotFoundError(f"No OXTS files found in {oxts_dir}")
    first = read_oxts_file(Path(files[0]))
    lat0, lon0, alt0 = first[0], first[1], first[2]

    T_w_imu = []
    for f in files:
        lat, lon, alt, roll, pitch, yaw = read_oxts_file(Path(f))
        t = latlon_to_enu_m(lat, lon, alt, lat0, lon0, alt0)
        R = R_from_rpy(roll, pitch, yaw)
        T_w_imu.append(T_from_Rt(R, t))
    return np.stack(T_w_imu, axis=0)


# ---------------------------------------------------------------------
# Loading estimate + performance CSVs
# ---------------------------------------------------------------------
def load_est_csv(path: Path):
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in estimator CSV: {path}")

    req = ["frame", "t", "x", "y", "z"]
    for r in req:
        if r not in rows[0]:
            raise ValueError(f"Estimator CSV missing column '{r}'")

    has_rpy = all(k in rows[0] for k in ["yaw_rad", "pitch_rad", "roll_rad"])

    N = len(rows)
    frame = np.zeros(N, dtype=np.int64)
    t = np.zeros(N, dtype=np.float64)
    p = np.zeros((N, 3), dtype=np.float64)
    R = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], N, axis=0)

    for i, row in enumerate(rows):
        frame[i] = int(float(row["frame"]))
        t[i] = float(row["t"])
        p[i] = [float(row["x"]), float(row["y"]), float(row["z"])]
        if has_rpy:
            yaw = float(row["yaw_rad"])
            pitch = float(row["pitch_rad"])
            roll = float(row["roll_rad"])
            R[i] = R_from_rpy(roll, pitch, yaw)

    return frame, t, p, R, has_rpy


def load_perf_csv(path: Path):
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    out = {}
    keys = rows[0].keys()
    for k in keys:
        if k == "frame":
            out[k] = np.array([int(float(r[k])) for r in rows], dtype=np.int64)
        else:
            out[k] = np.array([float(r[k]) for r in rows], dtype=np.float64)
    return out


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def align_se3_by_first_frame(T_est, T_gt):
    A = T_gt[0] @ invT(T_est[0])
    T_al = np.einsum("ij,njk->nik", A, T_est)
    return T_al


def rmse(a):
    a = np.asarray(a, dtype=np.float64)
    return math.sqrt(np.mean(a * a))


def path_length(P):
    if len(P) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(P[1:] - P[:-1], axis=1)))


def stats(a):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def summarize_sequence(seq_name: str):
    seq_dir = DATASET_ROOT / seq_name

    est_csv = seq_dir / "kitti_poses.csv"
    perf_csv = seq_dir / "kitti_performance.csv"
    img_ts = seq_dir / "image_02" / "timestamps.txt"
    oxts_ts = seq_dir / "oxts" / "timestamps.txt"
    oxts_dir = seq_dir / "oxts" / "data"

    for p in [est_csv, img_ts, oxts_ts, oxts_dir]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required path for {seq_name}: {p}")

    t_cam = load_kitti_timestamps(img_ts)
    t_oxts = load_kitti_timestamps(oxts_ts)
    T_w_imu = load_oxts_poses(oxts_dir)
    T_i_c2 = build_T_imu_to_cam2(seq_dir)
    T_w_c2 = np.einsum("nij,jk->nik", T_w_imu, T_i_c2)
    T_gt_cam = interp_se3(T_w_c2, t_oxts, t_cam)

    frame, t_est, p_est, R_est, has_rpy = load_est_csv(est_csv)

    N = min(len(p_est), len(t_cam), len(T_gt_cam))
    t_use = t_cam[:N]

    T_est = np.zeros((N, 4, 4), dtype=np.float64)
    T_est[:] = np.eye(4)
    for i in range(N):
        if has_rpy:
            T_est[i, :3, :3] = R_est[i]
        T_est[i, :3, 3] = p_est[i]

    T_gt = T_gt_cam[:N]
    T_est_al = align_se3_by_first_frame(T_est, T_gt)

    gtP = T_gt[:, :3, 3]
    estP = T_est_al[:, :3, 3]

    err3d = np.linalg.norm(estP - gtP, axis=1)
    errxy = np.linalg.norm(estP[:, :2] - gtP[:, :2], axis=1)

    gt_len = path_length(gtP)
    est_len = path_length(estP)
    duration_s = float(t_use[-1] - t_use[0]) if N > 1 else 0.0

    row = {
        "sequence": seq_name,
        "frames": int(N),
        "duration_s": duration_s,
        "gt_path_length_m": gt_len,
        "est_path_length_m": est_len,
        "ate_rmse_m": rmse(err3d),
        "xy_rmse_m": rmse(errxy),
        "ate_pct_of_gt_path": (100.0 * rmse(err3d) / gt_len) if gt_len > 1e-9 else float("nan"),
        "xy_pct_of_gt_path": (100.0 * rmse(errxy) / gt_len) if gt_len > 1e-9 else float("nan"),
    }

    if perf_csv.exists():
        perf = load_perf_csv(perf_csv)
        if perf is not None:
            total_ms = perf["wasm_total_ms"] if "wasm_total_ms" in perf else np.array([])
            klt_ms = perf["wasm_klt_ms"] if "wasm_klt_ms" in perf else np.array([])
            imu_ms = perf["wasm_imu_ms"] if "wasm_imu_ms" in perf else np.array([])
            seed_ms = perf["wasm_seed_ms"] if "wasm_seed_ms" in perf else np.array([])
            num_keypoints = perf["num_keypoints"] if "num_keypoints" in perf else np.array([])
            num_kfs = perf["num_kfs"] if "num_kfs" in perf else np.array([])
            num_mps = perf["num_mps"] if "num_mps" in perf else np.array([])
            imu_hz = perf["imu_hz"] if "imu_hz" in perf else np.array([])

            total_stats = stats(total_ms)
            klt_stats = stats(klt_ms)
            imu_stats = stats(imu_ms)
            seed_stats = stats(seed_ms)
            kp_stats = stats(num_keypoints)
            imuhz_stats = stats(imu_hz)

            row.update({
                "mean_total_ms": total_stats["mean"],
                "median_total_ms": total_stats["median"],
                "mean_fps": (1000.0 / total_stats["mean"]) if np.isfinite(total_stats["mean"]) and total_stats["mean"] > 1e-9 else float("nan"),
                "mean_klt_ms": klt_stats["mean"],
                "mean_imu_ms": imu_stats["mean"],
                "mean_seed_ms": seed_stats["mean"],
                "mean_keypoints": kp_stats["mean"],
                "median_keypoints": kp_stats["median"],
                "final_num_kfs": int(num_kfs[-1]) if num_kfs.size else 0,
                "final_num_mps": int(num_mps[-1]) if num_mps.size else 0,
                "mean_imu_hz": imuhz_stats["mean"],
            })
    return row


def main():
    rows = []
    for seq in SEQUENCES:
        print(f"Summarizing {seq} ...")
        rows.append(summarize_sequence(seq))

    fieldnames = list(rows[0].keys())

    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(OUTPUT_TXT, "w") as f:
        f.write("=== KITTI Results Summary ===\n\n")
        for r in rows:
            f.write(f"Sequence: {r['sequence']}\n")
            f.write(f"  Frames:              {r['frames']}\n")
            f.write(f"  Duration [s]:        {r['duration_s']:.3f}\n")
            f.write(f"  GT path length [m]:  {r['gt_path_length_m']:.3f}\n")
            f.write(f"  Est path length [m]: {r['est_path_length_m']:.3f}\n")
            f.write(f"  ATE RMSE [m]:        {r['ate_rmse_m']:.3f}\n")
            f.write(f"  XY RMSE [m]:         {r['xy_rmse_m']:.3f}\n")
            f.write(f"  ATE / GT path [%]:   {r['ate_pct_of_gt_path']:.3f}\n")
            f.write(f"  XY / GT path [%]:    {r['xy_pct_of_gt_path']:.3f}\n")
            if "mean_fps" in r:
                f.write(f"  Mean FPS:            {r['mean_fps']:.3f}\n")
                f.write(f"  Mean total ms:       {r['mean_total_ms']:.3f}\n")
                f.write(f"  Mean KLT ms:         {r['mean_klt_ms']:.3f}\n")
                f.write(f"  Mean IMU ms:         {r['mean_imu_ms']:.3f}\n")
                f.write(f"  Mean seed ms:        {r['mean_seed_ms']:.3f}\n")
                f.write(f"  Mean keypoints:      {r['mean_keypoints']:.3f}\n")
                f.write(f"  Final keyframes:     {r['final_num_kfs']}\n")
                f.write(f"  Final map points:    {r['final_num_mps']}\n")
                f.write(f"  Mean IMU Hz:         {r['mean_imu_hz']:.3f}\n")
            f.write("\n")

    print(f"Wrote CSV summary to: {OUTPUT_CSV}")
    print(f"Wrote text summary to: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()