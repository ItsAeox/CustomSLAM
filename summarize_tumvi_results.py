#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------
# Edit if needed
# ---------------------------------------------------------------------
DATASET_ROOT = Path("dataset")

SEQUENCES = [
    "bike-easy",
    "loop-floor1",
    "mocap-desk",
    "mocap-desk2",
    "office-maze",
    "running-easy",
    "skate-easy",
]

OUTPUT_CSV = Path("tumvi_results_summary.csv")
OUTPUT_TXT = Path("tumvi_results_summary.txt")

CAM_INDEX = 0
T_IMU_CAM_IS_IMU_FROM_CAM = True
T_IMU_MARKER_IS_IMU_FROM_MARKER = False
T_MOCAP_WORLD_IS_MOCAP_FROM_WORLD = True

USE_SMALL_SENSOR_TIME_OFFSETS = True
USE_CAMERA_TIMESTAMPS = True
RPE_DELTAS = (1, 10, 20, 50)
MOCAP_GAP_THRESHOLD_S = 0.1


# ---------------------------------------------------------------------
# SE(3) helpers
# ---------------------------------------------------------------------
def T_from_Rt(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invT(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -(R.T @ t)
    return Ti


def rot_angle(R):
    c = 0.5 * (np.trace(R) - 1.0)
    c = min(1.0, max(-1.0, float(c)))
    return math.acos(c)


def quat_xyzw_to_wxyz(q_xyzw):
    qx, qy, qz, qw = [float(v) for v in q_xyzw]
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("Zero quaternion encountered")
    return q / n


def R_from_quat_wxyz(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def quat_from_R(R):
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m[2, 1] - m[1, 2]) / S
        y = (m[0, 2] - m[2, 0]) / S
        z = (m[1, 0] - m[0, 1]) / S
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        S = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / S
        x = 0.25 * S
        y = (m[0, 1] + m[1, 0]) / S
        z = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / S
        x = (m[0, 1] + m[1, 0]) / S
        y = 0.25 * S
        z = (m[1, 2] + m[2, 1]) / S
    else:
        S = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / S
        x = (m[0, 2] + m[2, 0]) / S
        y = (m[1, 2] + m[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


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
    s0 = math.sin((1.0 - a) * th) / math.sin(th)
    s1 = math.sin(a * th) / math.sin(th)
    return s0 * q0 + s1 * q1


def R_from_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------
def normalize_time_array(arr, unit="auto"):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return arr

    rel = arr - arr[0]

    if unit == "s":
        return rel
    elif unit == "ms":
        return rel * 1e-3
    elif unit == "us":
        return rel * 1e-6
    elif unit == "ns":
        return rel * 1e-9

    mag = float(np.max(np.abs(arr)))
    if mag > 1e15:
        return rel * 1e-9
    elif mag > 1e12:
        return rel * 1e-6
    elif mag > 1e9:
        return rel * 1e-3
    return rel


def load_json(path: Path):
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and len(obj) == 1 and "value0" in obj:
        return obj["value0"]
    return obj


def tf_from_pose_dict(d):
    q = quat_xyzw_to_wxyz([d["qx"], d["qy"], d["qz"], d["qw"]])
    R = R_from_quat_wxyz(q)
    t = np.array([d["px"], d["py"], d["pz"]], dtype=np.float64)
    return T_from_Rt(R, t)


def load_numeric_text_file(path: Path):
    vals = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            vals.append(float(s))
    return np.asarray(vals, dtype=np.float64)


def load_est_csv(path: Path):
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in estimator CSV: {path}")

    req = ("frame", "t", "x", "y", "z")
    for k in req:
        if k not in rows[0]:
            raise ValueError(f"Estimator CSV missing column '{k}'")

    has_rpy = all(k in rows[0] for k in ("yaw_rad", "pitch_rad", "roll_rad"))

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

    t = t - t[0]
    return frame, t, p, R, has_rpy


def load_perf_csv(path: Path):
    if not path.exists():
        return None

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    req = [
        "frame", "t",
        "num_kfs", "num_mps", "num_keypoints",
        "wasm_klt_ms", "wasm_total_ms", "wasm_imu_ms", "wasm_seed_ms",
        "imu_hz"
    ]
    for k in req:
        if k not in rows[0]:
            raise ValueError(f"Performance CSV missing column '{k}'")

    N = len(rows)
    out = {k: np.zeros(N, dtype=np.float64) for k in req}
    out["frame"] = np.zeros(N, dtype=np.int64)

    for i, row in enumerate(rows):
        out["frame"][i] = int(float(row["frame"]))
        for k in req[1:]:
            out[k][i] = float(row[k])

    return out


def load_mocap_data(path: Path):
    raw_t = []
    T_mocap_marker = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            vals = np.fromstring(s, sep=" ", dtype=np.float64)
            if vals.size < 8:
                continue
            raw_t.append(vals[0])
            p = vals[1:4]
            q = quat_xyzw_to_wxyz(vals[4:8])
            T_mocap_marker.append(T_from_Rt(R_from_quat_wxyz(q), p))

    if not raw_t:
        raise ValueError(f"No mocap rows found in {path}")

    raw_t = np.asarray(raw_t, dtype=np.float64)
    T_mocap_marker = np.stack(T_mocap_marker, axis=0)

    order = np.argsort(raw_t)
    raw_t = raw_t[order]
    T_mocap_marker = T_mocap_marker[order]

    return normalize_time_array(raw_t, unit="us"), T_mocap_marker


def load_left_camera_times(left_images_dir: Path):
    p1 = left_images_dir / "image_timestamps_left.txt"
    if p1.exists():
        t = normalize_time_array(load_numeric_text_file(p1), unit="us")
        if t.size:
            return t, str(p1)

    p2 = left_images_dir / "timestamps.txt"
    if p2.exists():
        t = normalize_time_array(load_numeric_text_file(p2), unit="us")
        if t.size:
            return t, str(p2)

    return None, None


def build_query_times(frame_idx, est_t, cam_t):
    if (not USE_CAMERA_TIMESTAMPS) or cam_t is None or len(cam_t) == 0:
        return np.asarray(est_t, dtype=np.float64), "estimator_csv_t"

    frame_idx = np.asarray(frame_idx, dtype=np.int64)
    valid = (frame_idx >= 0) & (frame_idx < len(cam_t))

    if np.all(valid):
        return cam_t[frame_idx], "camera_timestamps_by_frame_index"

    t = np.empty_like(est_t, dtype=np.float64)
    t[:] = np.nan
    t[valid] = cam_t[frame_idx[valid]]

    n_valid = int(np.sum(valid))
    if n_valid >= max(10, int(0.8 * len(frame_idx))):
        row_count = min(len(cam_t), len(t))
        row_fallback = cam_t[:row_count]
        mask = ~np.isfinite(t[:row_count])
        t[:row_count][mask] = row_fallback[mask]
        nan_mask = ~np.isfinite(t)
        t[nan_mask] = est_t[nan_mask]
        return t, f"camera_timestamps_mostly_by_frame_index ({n_valid}/{len(frame_idx)} valid)"

    N = min(len(cam_t), len(est_t))
    t = np.asarray(est_t, dtype=np.float64).copy()
    t[:N] = cam_t[:N]
    return t, f"camera_timestamps_by_row_order ({N} rows)"


# ---------------------------------------------------------------------
# Pose interpolation / GT construction
# ---------------------------------------------------------------------
def interp_se3(T_list, t_list, t_query):
    tq = np.asarray(t_query, dtype=np.float64)
    out = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], tq.size, axis=0)
    qs = np.array([quat_from_R(T[:3, :3]) for T in T_list], dtype=np.float64)
    ps = np.array([T[:3, 3] for T in T_list], dtype=np.float64)

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
        out[k] = T_from_Rt(R_from_quat_wxyz(q), p)
    return out


def build_constant_transforms(cam_calib, mocap_imu_calib):
    T_imu_cam = tf_from_pose_dict(cam_calib["T_imu_cam"][CAM_INDEX])
    T_imu_marker = tf_from_pose_dict(mocap_imu_calib["T_imu_marker"])
    T_mocap_world = tf_from_pose_dict(mocap_imu_calib["T_mocap_world"])

    if not T_IMU_CAM_IS_IMU_FROM_CAM:
        T_imu_cam = invT(T_imu_cam)
    if not T_IMU_MARKER_IS_IMU_FROM_MARKER:
        T_imu_marker = invT(T_imu_marker)
    if not T_MOCAP_WORLD_IS_MOCAP_FROM_WORLD:
        T_mocap_world = invT(T_mocap_world)

    T_world_mocap = invT(T_mocap_world)
    T_marker_imu = invT(T_imu_marker)
    return T_world_mocap, T_marker_imu, T_imu_cam


def build_world_cam_poses(T_mocap_marker_series, T_world_mocap, T_marker_imu, T_imu_cam):
    return np.einsum(
        "ij,njk,kl,lm->nim",
        T_world_mocap,
        T_mocap_marker_series,
        T_marker_imu,
        T_imu_cam,
    )


# ---------------------------------------------------------------------
# Mocap segment handling
# ---------------------------------------------------------------------
def build_mocap_segments(t_list, gap_thresh_s):
    t_list = np.asarray(t_list, dtype=np.float64)
    if len(t_list) == 0:
        return []

    dt = np.diff(t_list)
    cut_idx = np.where(dt > gap_thresh_s)[0]

    segments = []
    s = 0
    for i in cut_idx:
        segments.append((s, i))
        s = i + 1
    segments.append((s, len(t_list) - 1))
    return segments


def segment_id_for_queries(t_query, segments, t_mocap):
    seg_id = np.full(len(t_query), -1, dtype=np.int64)
    for sid, (a, b) in enumerate(segments):
        t0 = t_mocap[a]
        t1 = t_mocap[b]
        mask = (t_query >= t0) & (t_query <= t1)
        seg_id[mask] = sid
    return seg_id


# ---------------------------------------------------------------------
# RPE
# ---------------------------------------------------------------------
def compute_rpe_pairs(T_est, T_gt, t_query, seg_id, delta):
    rows = []
    N = len(T_est)
    for i in range(N - delta):
        j = i + delta

        if seg_id[i] < 0 or seg_id[j] < 0:
            continue
        if seg_id[i] != seg_id[j]:
            continue

        d_est = invT(T_est[i]) @ T_est[j]
        d_gt = invT(T_gt[i]) @ T_gt[j]
        d_err = invT(d_gt) @ d_est

        trans_err = float(np.linalg.norm(d_err[:3, 3]))
        rot_err_deg = float(rot_angle(d_err[:3, :3]) * 180.0 / math.pi)

        rows.append({
            "trans_err_m": trans_err,
            "rot_err_deg": rot_err_deg,
        })
    return rows


def summarize_rpe_rows(rows):
    if not rows:
        return None
    te = np.array([r["trans_err_m"] for r in rows], dtype=np.float64)
    re = np.array([r["rot_err_deg"] for r in rows], dtype=np.float64)
    return {
        "count": int(len(rows)),
        "trans_mean_m": float(np.mean(te)),
        "trans_median_m": float(np.median(te)),
        "trans_rmse_m": float(np.sqrt(np.mean(te * te))),
        "rot_mean_deg": float(np.mean(re)),
        "rot_median_deg": float(np.median(re)),
        "rot_rmse_deg": float(np.sqrt(np.mean(re * re))),
    }


# ---------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Per-sequence summarizer
# ---------------------------------------------------------------------
def summarize_sequence(seq_name: str):
    seq_dir = DATASET_ROOT / seq_name

    est_csv = seq_dir / "tumvi_poses.csv"
    perf_csv = seq_dir / "tumvi_performance.csv"
    mocap_txt = seq_dir / "mocap_data.txt"
    cam_calib_json = seq_dir / "camera-calibration.json"
    mocap_imu_calib_json = seq_dir / "mocap-imu-calibration.json"
    left_images_dir = seq_dir / "left_images"

    required = [est_csv, mocap_txt, cam_calib_json, mocap_imu_calib_json]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file for {seq_name}: {p}")

    frame, est_t, p_est, R_est, has_rpy = load_est_csv(est_csv)
    mocap_t_rel, T_mocap_marker = load_mocap_data(mocap_txt)
    perf = load_perf_csv(perf_csv)

    cam_calib = load_json(cam_calib_json)
    mocap_imu_calib = load_json(mocap_imu_calib_json)

    T_world_mocap, T_marker_imu, T_imu_cam = build_constant_transforms(cam_calib, mocap_imu_calib)
    T_world_cam_mocap = build_world_cam_poses(T_mocap_marker, T_world_mocap, T_marker_imu, T_imu_cam)

    cam_t_rel, cam_t_source = load_left_camera_times(left_images_dir)

    small_time_shift_s = 0.0
    if USE_SMALL_SENSOR_TIME_OFFSETS:
        cam_shift = float(cam_calib.get("cam_time_offset_ns", 0.0)) * 1e-9
        mocap_shift = float(mocap_imu_calib.get("mocap_time_offset_ns", 0.0)) * 1e-9
        small_time_shift_s = cam_shift - mocap_shift

    t_query_rel, t_query_source = build_query_times(frame, est_t, cam_t_rel)
    t_query_rel = np.asarray(t_query_rel, dtype=np.float64) + small_time_shift_s

    N = len(est_t)
    if len(t_query_rel) != N:
        M = min(N, len(t_query_rel))
        frame = frame[:M]
        est_t = est_t[:M]
        p_est = p_est[:M]
        R_est = R_est[:M]
        t_query_rel = t_query_rel[:M]
        N = M

    segments = build_mocap_segments(mocap_t_rel, MOCAP_GAP_THRESHOLD_S)
    seg_id = segment_id_for_queries(t_query_rel, segments, mocap_t_rel)
    valid_queries = seg_id >= 0

    t_query_clamped = np.clip(t_query_rel, mocap_t_rel[0], mocap_t_rel[-1])
    T_gt = interp_se3(T_world_cam_mocap, mocap_t_rel, t_query_clamped)
    T_est = np.stack([T_from_Rt(R_est[i], p_est[i]) for i in range(N)], axis=0)

    row = {
        "sequence": seq_name,
        "frames": int(N),
        "duration_s": float(t_query_rel[-1] - t_query_rel[0]) if N > 1 else 0.0,
        "mocap_rows": int(len(mocap_t_rel)),
        "mocap_segments": int(len(segments)),
        "valid_queries": int(np.sum(valid_queries)),
        "valid_query_ratio": float(np.sum(valid_queries) / N) if N > 0 else float("nan"),
        "camera_time_source": cam_t_source if cam_t_source else "none",
        "query_time_source": t_query_source,
        "small_time_shift_s": float(small_time_shift_s),
    }

    for delta in RPE_DELTAS:
        rows = compute_rpe_pairs(T_est, T_gt, t_query_rel, seg_id, delta)
        s = summarize_rpe_rows(rows)
        if s is None:
            row[f"rpe_count_d{delta}"] = 0
            row[f"rpe_trans_rmse_d{delta}_m"] = float("nan")
            row[f"rpe_rot_rmse_d{delta}_deg"] = float("nan")
            row[f"rpe_trans_mean_d{delta}_m"] = float("nan")
            row[f"rpe_rot_mean_d{delta}_deg"] = float("nan")
        else:
            row[f"rpe_count_d{delta}"] = s["count"]
            row[f"rpe_trans_rmse_d{delta}_m"] = s["trans_rmse_m"]
            row[f"rpe_rot_rmse_d{delta}_deg"] = s["rot_rmse_deg"]
            row[f"rpe_trans_mean_d{delta}_m"] = s["trans_mean_m"]
            row[f"rpe_rot_mean_d{delta}_deg"] = s["rot_mean_deg"]

    if perf is not None:
        total_s = stats(perf["wasm_total_ms"])
        klt_s = stats(perf["wasm_klt_ms"])
        imu_s = stats(perf["wasm_imu_ms"])
        seed_s = stats(perf["wasm_seed_ms"])
        kp_s = stats(perf["num_keypoints"])
        imuhz_s = stats(perf["imu_hz"])

        row.update({
            "mean_total_ms": total_s["mean"],
            "median_total_ms": total_s["median"],
            "mean_fps": (1000.0 / total_s["mean"]) if np.isfinite(total_s["mean"]) and total_s["mean"] > 1e-9 else float("nan"),
            "mean_klt_ms": klt_s["mean"],
            "mean_imu_ms": imu_s["mean"],
            "mean_seed_ms": seed_s["mean"],
            "mean_keypoints": kp_s["mean"],
            "median_keypoints": kp_s["median"],
            "final_num_kfs": int(perf["num_kfs"][-1]) if len(perf["num_kfs"]) else 0,
            "final_num_mps": int(perf["num_mps"][-1]) if len(perf["num_mps"]) else 0,
            "mean_imu_hz": imuhz_s["mean"],
        })

    return row


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
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
        f.write("=== TUM-VI Results Summary ===\n\n")
        for r in rows:
            f.write(f"Sequence: {r['sequence']}\n")
            f.write(f"  Frames:                {r['frames']}\n")
            f.write(f"  Duration [s]:          {r['duration_s']:.3f}\n")
            f.write(f"  Mocap rows:            {r['mocap_rows']}\n")
            f.write(f"  Mocap segments:        {r['mocap_segments']}\n")
            f.write(f"  Valid queries:         {r['valid_queries']}\n")
            f.write(f"  Valid query ratio:     {r['valid_query_ratio']:.3f}\n")
            f.write(f"  Camera time source:    {r['camera_time_source']}\n")
            f.write(f"  Query time source:     {r['query_time_source']}\n")
            f.write(f"  Small time shift [s]:  {r['small_time_shift_s']:.9f}\n")

            for delta in RPE_DELTAS:
                f.write(f"  RPE delta={delta}:\n")
                f.write(f"    Count:              {r[f'rpe_count_d{delta}']}\n")
                if np.isfinite(r[f"rpe_trans_rmse_d{delta}_m"]):
                    f.write(f"    Trans RMSE [m]:     {r[f'rpe_trans_rmse_d{delta}_m']:.6f}\n")
                    f.write(f"    Rot RMSE [deg]:     {r[f'rpe_rot_rmse_d{delta}_deg']:.6f}\n")
                    f.write(f"    Trans mean [m]:     {r[f'rpe_trans_mean_d{delta}_m']:.6f}\n")
                    f.write(f"    Rot mean [deg]:     {r[f'rpe_rot_mean_d{delta}_deg']:.6f}\n")
                else:
                    f.write("    No valid pairs\n")

            if "mean_fps" in r:
                f.write(f"  Mean FPS:             {r['mean_fps']:.3f}\n")
                f.write(f"  Mean total ms:        {r['mean_total_ms']:.3f}\n")
                f.write(f"  Mean KLT ms:          {r['mean_klt_ms']:.3f}\n")
                f.write(f"  Mean IMU ms:          {r['mean_imu_ms']:.3f}\n")
                f.write(f"  Mean seed ms:         {r['mean_seed_ms']:.3f}\n")
                f.write(f"  Mean keypoints:       {r['mean_keypoints']:.3f}\n")
                f.write(f"  Final keyframes:      {r['final_num_kfs']}\n")
                f.write(f"  Final map points:     {r['final_num_mps']}\n")
                f.write(f"  Mean IMU Hz:          {r['mean_imu_hz']:.3f}\n")
            f.write("\n")

    print(f"Wrote CSV summary to: {OUTPUT_CSV}")
    print(f"Wrote text summary to: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()