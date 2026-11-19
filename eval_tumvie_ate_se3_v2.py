#!/usr/bin/env python3
"""
SE(3)-aligned ATE evaluator for TUM-VIE sequences.

Designed to mirror the user's KITTI evaluator style while preserving the
TUM-VIE dataset layout and mocap handling from the existing RPE evaluator.

Key behavior:
- Uses mocap-supported queries only (segments split by timestamp gaps)
- Builds camera-frame GT poses from mocap + calibration
- Aligns estimated trajectory to GT with rigid SE(3) alignment (no scale)
- Exports per-sequence aligned poses, summary text/CSV, plots, and performance plots
- Processes the requested sequence list in one run

Outputs go to:
    dataset/tumvi_eval_v2/<sequence>/...
    dataset/tumvi_eval_v2/tumvie_ate_summary.csv
"""

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SEQUENCES = [
    "bike-easy",
    "loop-floor1",
    "mocap-desk",
    "mocap-desk2",
    "office-maze",
    "running-easy",
    "skate-easy",
]

OUTPUT_ROOT = Path("dataset/tumvi_eval_v2")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

CAM_INDEX = 0
T_IMU_CAM_IS_IMU_FROM_CAM = True
T_IMU_MARKER_IS_IMU_FROM_MARKER = False
T_MOCAP_WORLD_IS_MOCAP_FROM_WORLD = True
USE_SMALL_SENSOR_TIME_OFFSETS = True
USE_CAMERA_TIMESTAMPS = True
MOCAP_GAP_THRESHOLD_S = 0.1

# -----------------------------------------------------------------------------
# SE(3) helpers
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Alignment
# -----------------------------------------------------------------------------
def align_se3_umeyama_no_scale(P_est, P_gt):
    """Return R, t aligning Nx3 P_est to P_gt with rigid SE(3), no scale."""
    P_est = np.asarray(P_est, dtype=np.float64)
    P_gt = np.asarray(P_gt, dtype=np.float64)
    if P_est.shape != P_gt.shape or P_est.ndim != 2 or P_est.shape[1] != 3:
        raise ValueError("Point arrays must be Nx3 and shape-matched")
    if len(P_est) < 3:
        raise ValueError("Need at least 3 matched points for SE(3) alignment")

    mu_est = P_est.mean(axis=0)
    mu_gt = P_gt.mean(axis=0)
    X = P_est - mu_est
    Y = P_gt - mu_gt

    H = X.T @ Y / len(P_est)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = mu_gt - R @ mu_est
    return R, t


def apply_alignment(T_est, R, t):
    A = T_from_Rt(R, t)
    return np.einsum("ij,njk->nik", A, T_est), A


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
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


def load_json(path):
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


def load_numeric_text_file(path):
    vals = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            vals.append(float(s))
    return np.asarray(vals, dtype=np.float64)


def load_est_csv(path):
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


def load_perf_csv(path):
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in performance CSV: {path}")

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


def load_mocap_data(path):
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


def load_left_camera_times(dataset_root):
    left_images_dir = dataset_root / "left_images"
    ts_path = left_images_dir / "image_timestamps_left.txt"
    if ts_path.exists():
        t = normalize_time_array(load_numeric_text_file(ts_path), unit="us")
        if t.size:
            return t, str(ts_path)
    fallback = left_images_dir / "timestamps.txt"
    if fallback.exists():
        t = normalize_time_array(load_numeric_text_file(fallback), unit="us")
        if t.size:
            return t, str(fallback)
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


# -----------------------------------------------------------------------------
# GT and valid mocap segments
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def ate_translation_rmse(T_est, T_gt):
    e = T_est[:, :3, 3] - T_gt[:, :3, 3]
    return float(math.sqrt(np.mean(np.sum(e * e, axis=1))))


def ate_translation_rmse_xy(T_est, T_gt):
    e_xy = T_est[:, :2, 3] - T_gt[:, :2, 3]
    return float(math.sqrt(np.mean(np.sum(e_xy * e_xy, axis=1))))


def ate_rotation_rmse_deg(T_est, T_gt):
    ang = []
    for i in range(len(T_est)):
        dR = T_gt[i, :3, :3].T @ T_est[i, :3, :3]
        ang.append(rot_angle(dR))
    ang = np.array(ang, dtype=np.float64)
    return float(np.sqrt(np.mean((ang * 180.0 / math.pi) ** 2)))


def valid_ratio(seg_id):
    return float(np.mean(seg_id >= 0)) if len(seg_id) else 0.0


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
def write_aligned_csv(path, t_use, T_est_al, T_gt, valid_mask):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "i", "t", "est_x", "est_y", "est_z", "gt_x", "gt_y", "gt_z",
            "err_xy", "err_3d"
        ])
        idxs = np.where(valid_mask)[0]
        for i in idxs:
            ex, ey, ez = T_est_al[i, :3, 3]
            gx, gy, gz = T_gt[i, :3, 3]
            err_xy = math.sqrt((ex - gx) ** 2 + (ey - gy) ** 2)
            err_3d = math.sqrt((ex - gx) ** 2 + (ey - gy) ** 2 + (ez - gz) ** 2)
            w.writerow([i, f"{t_use[i]:.9f}", f"{ex:.9f}", f"{ey:.9f}", f"{ez:.9f}",
                        f"{gx:.9f}", f"{gy:.9f}", f"{gz:.9f}", f"{err_xy:.9f}", f"{err_3d:.9f}"])


def write_summary_txt(path, dataset_name, dataset_root, cam_t_source, t_query_source,
                      segments, metrics, valid_count, total_count, small_time_shift_s):
    with open(path, "w") as f:
        f.write("=== TUM-VIE SE(3)-aligned ATE evaluation ===\n\n")
        f.write(f"Sequence: {dataset_name}\n")
        f.write(f"DATASET_ROOT: {dataset_root}\n")
        f.write(f"Camera timestamp source: {cam_t_source}\n")
        f.write(f"Chosen query time source: {t_query_source}\n")
        f.write(f"Applied small time shift [s]: {small_time_shift_s:.9f}\n")
        f.write(f"Mocap continuity threshold [s]: {MOCAP_GAP_THRESHOLD_S}\n")
        f.write(f"Mocap segment count: {len(segments)}\n")
        f.write(f"Valid queries used for ATE: {valid_count} / {total_count}\n\n")

        f.write("Interpretation:\n")
        f.write("  This evaluator computes SE(3)-aligned ATE only over mocap-supported trajectory parts.\n")
        f.write("  Ground-truth discontinuities are handled by segmenting the mocap stream; only queries that fall\n")
        f.write("  inside valid mocap-supported segments are used in the final aligned ATE computation.\n")
        f.write("  No scale alignment is applied.\n\n")

        for sid, (a, b) in enumerate(segments):
            f.write(f"  segment {sid}: idx [{a}, {b}], samples={b-a+1}\n")

        f.write("\nMetrics:\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v}\n")


def plot_trajectory_valid(path, T_est_al, T_gt, valid_mask, title):
    idx = np.where(valid_mask)[0]
    if len(idx) == 0:
        return
    estP = T_est_al[idx, :3, 3]
    gtP = T_gt[idx, :3, 3]
    plt.figure()
    plt.plot(gtP[:, 0], gtP[:, 1], label="GT cam")
    plt.plot(estP[:, 0], estP[:, 1], label="Est aligned SE(3)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_up_over_time(path, t_use, T_est_al, T_gt, valid_mask):
    idx = np.where(valid_mask)[0]
    if len(idx) == 0:
        return
    plt.figure()
    plt.plot(t_use[idx], T_gt[idx, 2, 3], label="GT Z")
    plt.plot(t_use[idx], T_est_al[idx, 2, 3], label="Est Z")
    plt.grid(True)
    plt.legend()
    plt.title("Up/Z over valid mocap-supported time")
    plt.xlabel("time [s]")
    plt.ylabel("meters")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_position_error(path, t_use, T_est_al, T_gt, valid_mask, xy_only=False, title=""):
    idx = np.where(valid_mask)[0]
    if len(idx) == 0:
        return
    if xy_only:
        err = np.linalg.norm(T_est_al[idx, :2, 3] - T_gt[idx, :2, 3], axis=1)
    else:
        err = np.linalg.norm(T_est_al[idx, :3, 3] - T_gt[idx, :3, 3], axis=1)
    plt.figure()
    plt.plot(t_use[idx], err)
    plt.grid(True)
    plt.title(title)
    plt.xlabel("time [s]")
    plt.ylabel("meters")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_performance(output_dir, perf):
    if perf is None:
        return
    Np = min(
        len(perf["frame"]),
        len(perf["num_kfs"]),
        len(perf["num_mps"]),
        len(perf["num_keypoints"]),
        len(perf["wasm_klt_ms"]),
        len(perf["wasm_total_ms"]),
        len(perf["wasm_imu_ms"]),
        len(perf["wasm_seed_ms"]),
        len(perf["imu_hz"]),
    )
    pf = perf["frame"][:Np]

    plt.figure()
    plt.plot(pf, perf["num_keypoints"][:Np])
    plt.grid(True)
    plt.title("Tracked keypoints over frames")
    plt.xlabel("frame")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(output_dir / "perf_keypoints.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(pf, perf["num_kfs"][:Np], label="KFs")
    plt.plot(pf, perf["num_mps"][:Np], label="MPs")
    plt.grid(True)
    plt.legend()
    plt.title("Keyframes and map points over frames")
    plt.xlabel("frame")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(output_dir / "perf_map_structure.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(pf, perf["wasm_total_ms"][:Np], label="Total")
    plt.plot(pf, perf["wasm_klt_ms"][:Np], label="KLT")
    plt.plot(pf, perf["wasm_imu_ms"][:Np], label="IMU")
    plt.plot(pf, perf["wasm_seed_ms"][:Np], label="Seed")
    plt.grid(True)
    plt.legend()
    plt.title("WASM timing over frames")
    plt.xlabel("frame")
    plt.ylabel("ms")
    plt.tight_layout()
    plt.savefig(output_dir / "perf_wasm_timing.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(pf, perf["imu_hz"][:Np])
    plt.grid(True)
    plt.title("IMU rate over frames")
    plt.xlabel("frame")
    plt.ylabel("Hz")
    plt.tight_layout()
    plt.savefig(output_dir / "perf_imu_hz.png", dpi=160)
    plt.close()


def write_performance_summary(path, perf):
    if perf is None:
        return

    def stats(a):
        a = np.asarray(a, dtype=np.float64)
        return {
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
        }

    with open(path, "w") as f:
        f.write("=== TUM-VIE performance summary ===\n\n")
        fields = [
            ("num_keypoints", "Tracked keypoints"),
            ("num_kfs", "Keyframes"),
            ("num_mps", "Map points"),
            ("wasm_total_ms", "WASM total time [ms]"),
            ("wasm_klt_ms", "WASM KLT time [ms]"),
            ("wasm_imu_ms", "WASM IMU time [ms]"),
            ("wasm_seed_ms", "WASM seed time [ms]"),
            ("imu_hz", "IMU rate [Hz]"),
        ]
        for key, label in fields:
            s = stats(perf[key])
            f.write(f"{label}: min={s['min']:.3f}, max={s['max']:.3f}, mean={s['mean']:.3f}, median={s['median']:.3f}\n")


# -----------------------------------------------------------------------------
# Per-sequence evaluation
# -----------------------------------------------------------------------------
def evaluate_sequence(dataset_name):
    dataset_root = Path("dataset") / dataset_name
    output_dir = OUTPUT_ROOT / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    est_csv = dataset_root / "tumvi_poses.csv"
    perf_csv = dataset_root / "tumvi_performance.csv"
    mocap_txt = dataset_root / "mocap_data.txt"
    cam_calib_json = dataset_root / "camera-calibration.json"
    mocap_imu_calib_json = dataset_root / "mocap-imu-calibration.json"

    required = [est_csv, mocap_txt, cam_calib_json, mocap_imu_calib_json]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file for {dataset_name}: {p}")

    frame, est_t, p_est, R_est, has_rpy = load_est_csv(est_csv)
    perf = load_perf_csv(perf_csv) if perf_csv.exists() else None
    mocap_t_rel, T_mocap_marker = load_mocap_data(mocap_txt)
    cam_calib = load_json(cam_calib_json)
    mocap_imu_calib = load_json(mocap_imu_calib_json)

    T_world_mocap, T_marker_imu, T_imu_cam = build_constant_transforms(cam_calib, mocap_imu_calib)
    T_world_cam_mocap = build_world_cam_poses(T_mocap_marker, T_world_mocap, T_marker_imu, T_imu_cam)

    cam_t_rel, cam_t_source = load_left_camera_times(dataset_root)

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

    idx = np.where(valid_queries)[0]
    if len(idx) < 3:
        raise RuntimeError(f"Not enough valid mocap-supported queries for ATE in {dataset_name}: {len(idx)}")

    P_est_valid = T_est[idx, :3, 3]
    P_gt_valid = T_gt[idx, :3, 3]
    R_align, t_align = align_se3_umeyama_no_scale(P_est_valid, P_gt_valid)
    T_est_al, A = apply_alignment(T_est, R_align, t_align)

    ate_3d = ate_translation_rmse(T_est_al[idx], T_gt[idx])
    ate_xy = ate_translation_rmse_xy(T_est_al[idx], T_gt[idx])
    rot_rmse_deg = ate_rotation_rmse_deg(T_est_al[idx], T_gt[idx]) if has_rpy else None

    metrics = {
        "valid_ratio": f"{valid_ratio(seg_id):.6f}",
        "valid_query_count": int(len(idx)),
        "total_query_count": int(N),
        "ate_rmse_3d_m": f"{ate_3d:.9f}",
        "ate_rmse_xy_m": f"{ate_xy:.9f}",
        "rot_rmse_deg": f"{rot_rmse_deg:.9f}" if rot_rmse_deg is not None else "NA",
        "alignment_type": "SE(3) rigid, no scale",
    }

    write_aligned_csv(output_dir / "poses_aligned_se3.csv", t_query_rel, T_est_al, T_gt, valid_queries)
    write_summary_txt(
        output_dir / "summary.txt",
        dataset_name,
        dataset_root,
        cam_t_source if cam_t_source else "none",
        t_query_source,
        segments,
        metrics,
        int(len(idx)),
        int(N),
        small_time_shift_s,
    )

    plot_trajectory_valid(
        output_dir / "traj_EN_valid.png",
        T_est_al,
        T_gt,
        valid_queries,
        f"Trajectory over valid mocap-supported parts, ATE={ate_3d:.3f} m",
    )
    plot_up_over_time(output_dir / "up_over_time_valid.png", t_query_rel, T_est_al, T_gt, valid_queries)
    plot_position_error(
        output_dir / "pos_error_xy_valid.png",
        t_query_rel,
        T_est_al,
        T_gt,
        valid_queries,
        xy_only=True,
        title=f"Bird's-eye position error (XY), RMSE={ate_xy:.3f} m",
    )
    plot_position_error(
        output_dir / "pos_error_3d_valid.png",
        t_query_rel,
        T_est_al,
        T_gt,
        valid_queries,
        xy_only=False,
        title=f"Position error 3D, ATE RMSE={ate_3d:.3f} m",
    )

    plot_performance(output_dir, perf)
    write_performance_summary(output_dir / "performance_summary.txt", perf)

    print(f"=== TUM-VIE ATE eval: {dataset_name} ===")
    print(f"Valid queries:      {len(idx)} / {N} ({valid_ratio(seg_id):.3f})")
    print(f"ATE RMSE 3D [m]:    {ate_3d:.6f}")
    print(f"ATE RMSE XY [m]:    {ate_xy:.6f}")
    if rot_rmse_deg is not None:
        print(f"Rot RMSE [deg]:     {rot_rmse_deg:.6f}")
    print(f"Output dir:         {output_dir}")

    return {
        "sequence": dataset_name,
        "valid_ratio": valid_ratio(seg_id),
        "valid_query_count": int(len(idx)),
        "total_query_count": int(N),
        "ate_rmse_3d_m": ate_3d,
        "ate_rmse_xy_m": ate_xy,
        "rot_rmse_deg": rot_rmse_deg if rot_rmse_deg is not None else float("nan"),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def write_global_summary(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "sequence", "valid_ratio", "valid_query_count", "total_query_count",
            "ate_rmse_3d_m", "ate_rmse_xy_m", "rot_rmse_deg"
        ])
        for r in rows:
            w.writerow([
                r["sequence"],
                f"{r['valid_ratio']:.6f}",
                r["valid_query_count"],
                r["total_query_count"],
                f"{r['ate_rmse_3d_m']:.9f}",
                f"{r['ate_rmse_xy_m']:.9f}",
                "NA" if np.isnan(r["rot_rmse_deg"]) else f"{r['rot_rmse_deg']:.9f}",
            ])


def main():
    results = []
    failures = []
    for seq in SEQUENCES:
        try:
            results.append(evaluate_sequence(seq))
        except Exception as e:
            failures.append((seq, str(e)))
            print(f"FAILED {seq}: {e}")

    write_global_summary(OUTPUT_ROOT / "tumvie_ate_summary.csv", results)

    with open(OUTPUT_ROOT / "run_summary.txt", "w") as f:
        f.write("=== TUM-VIE ATE batch run summary ===\n\n")
        f.write(f"Sequences requested: {len(SEQUENCES)}\n")
        f.write(f"Succeeded: {len(results)}\n")
        f.write(f"Failed: {len(failures)}\n\n")
        if results:
            f.write("Successful sequences:\n")
            for r in results:
                f.write(
                    f"  {r['sequence']}: valid_ratio={r['valid_ratio']:.3f}, "
                    f"ATE3D={r['ate_rmse_3d_m']:.6f} m, ATEXY={r['ate_rmse_xy_m']:.6f} m\n"
                )
        if failures:
            f.write("\nFailures:\n")
            for seq, msg in failures:
                f.write(f"  {seq}: {msg}\n")

    print(f"\nWrote global summary: {OUTPUT_ROOT / 'tumvie_ate_summary.csv'}")
    print(f"Wrote run summary:    {OUTPUT_ROOT / 'run_summary.txt'}")


if __name__ == "__main__":
    main()
