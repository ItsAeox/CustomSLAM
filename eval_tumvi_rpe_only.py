#!/usr/bin/env python3
"""
RPE-focused evaluator for a TUM-VI / TUM-VI-like sequence.

This version avoids global ATE/trajectory conclusions and instead evaluates
local relative pose consistency against mocap-derived camera poses, while
explicitly handling mocap discontinuities by splitting the mocap stream into
continuous time segments.

Outputs go to:
    DATASET_ROOT / "outputs_rpe"
so this can coexist with your existing evaluator.
"""

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
DATASET_ROOT = Path("dataset/office-maze")
EST_CSV = DATASET_ROOT / "tumvi_poses.csv"
MOCAP_TXT = DATASET_ROOT / "mocap_data.txt"
CAM_CALIB_JSON = DATASET_ROOT / "camera-calibration.json"
MOCAP_IMU_CALIB_JSON = DATASET_ROOT / "mocap-imu-calibration.json"
LEFT_IMAGES_DIR = DATASET_ROOT / "left_images"
IMAGE_TIMESTAMPS_LEFT = LEFT_IMAGES_DIR / "image_timestamps_left.txt"
IMAGE_EXPOSURES_LEFT = LEFT_IMAGES_DIR / "image_exposures_left.txt"

OUTPUT_DIR = DATASET_ROOT / "outputs_rpe"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Assumptions / knobs
# -----------------------------------------------------------------------------
CAM_INDEX = 0
T_IMU_CAM_IS_IMU_FROM_CAM = True
T_IMU_MARKER_IS_IMU_FROM_MARKER = False
T_MOCAP_WORLD_IS_MOCAP_FROM_WORLD = True

USE_SMALL_SENSOR_TIME_OFFSETS = True
RPE_DELTAS = (1, 10, 20, 50)

# Any mocap timestamp jump larger than this starts a new valid segment.
# The user's debug run showed a huge 117.575 s jump, so segmenting is essential.
MOCAP_GAP_THRESHOLD_S = 0.1

# Whether to use the left image timestamps as the primary timebase.
USE_CAMERA_TIMESTAMPS = True


# -----------------------------------------------------------------------------
# Helpers: SE(3)
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


# -----------------------------------------------------------------------------
# Loading helpers
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


def load_left_camera_times():
    if IMAGE_TIMESTAMPS_LEFT.exists():
        t = normalize_time_array(load_numeric_text_file(IMAGE_TIMESTAMPS_LEFT), unit="us")
        if t.size:
            return t, f"{IMAGE_TIMESTAMPS_LEFT}"

    fallback = LEFT_IMAGES_DIR / "timestamps.txt"
    if fallback.exists():
        t = normalize_time_array(load_numeric_text_file(fallback), unit="us")
        if t.size:
            return t, f"{fallback}"

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
# Pose interpolation
# -----------------------------------------------------------------------------
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
# GT construction
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


# -----------------------------------------------------------------------------
# Mocap segment handling
# -----------------------------------------------------------------------------
def build_mocap_segments(t_list, gap_thresh_s):
    """
    Returns list of (start_idx, end_idx_inclusive) continuous segments.
    """
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
    """
    For each query time, return the segment id it belongs to, or -1 if none.
    """
    seg_id = np.full(len(t_query), -1, dtype=np.int64)
    for sid, (a, b) in enumerate(segments):
        t0 = t_mocap[a]
        t1 = t_mocap[b]
        mask = (t_query >= t0) & (t_query <= t1)
        seg_id[mask] = sid
    return seg_id


# -----------------------------------------------------------------------------
# RPE
# -----------------------------------------------------------------------------
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

        gt_rel_dist = float(np.linalg.norm(d_gt[:3, 3]))
        est_rel_dist = float(np.linalg.norm(d_est[:3, 3]))

        rows.append({
            "i": i,
            "j": j,
            "t_i": float(t_query[i]),
            "t_j": float(t_query[j]),
            "delta_frames": int(delta),
            "delta_t_s": float(t_query[j] - t_query[i]),
            "segment_id": int(seg_id[i]),
            "gt_rel_dist_m": gt_rel_dist,
            "est_rel_dist_m": est_rel_dist,
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


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
def write_pairs_csv(path, rows):
    if not rows:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "i", "j", "t_i", "t_j", "delta_frames", "delta_t_s", "segment_id",
                "gt_rel_dist_m", "est_rel_dist_m", "trans_err_m", "rot_err_deg"
            ])
        return

    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_summary_txt(path, dataset_root, cam_t_source, t_query_source, segments, summaries):
    with open(path, "w") as f:
        f.write("=== TUM-VI RPE-focused evaluation ===\n")
        f.write(f"DATASET_ROOT: {dataset_root}\n")
        f.write(f"Camera timestamp source: {cam_t_source}\n")
        f.write(f"Chosen query time source: {t_query_source}\n")
        f.write(f"Mocap continuity threshold [s]: {MOCAP_GAP_THRESHOLD_S}\n")
        f.write(f"Mocap segment count: {len(segments)}\n")
        f.write("\nSegments:\n")
        for sid, (a, b) in enumerate(segments):
            f.write(f"  segment {sid}: idx [{a}, {b}], samples={b-a+1}\n")

        f.write("\nRPE summaries:\n")
        for delta in RPE_DELTAS:
            s = summaries.get(delta)
            if s is None:
                f.write(f"  delta={delta}: no valid pairs\n")
                continue
            f.write(
                f"  delta={delta}: count={s['count']}, "
                f"trans_rmse={s['trans_rmse_m']:.6f} m, "
                f"trans_mean={s['trans_mean_m']:.6f} m, "
                f"rot_rmse={s['rot_rmse_deg']:.6f} deg, "
                f"rot_mean={s['rot_mean_deg']:.6f} deg\n"
            )


def plot_rpe_series(rows, delta):
    if not rows:
        return

    t = np.array([r["t_i"] for r in rows], dtype=np.float64)
    te = np.array([r["trans_err_m"] for r in rows], dtype=np.float64)
    re = np.array([r["rot_err_deg"] for r in rows], dtype=np.float64)
    gt_d = np.array([r["gt_rel_dist_m"] for r in rows], dtype=np.float64)
    est_d = np.array([r["est_rel_dist_m"] for r in rows], dtype=np.float64)

    plt.figure()
    plt.plot(t, te)
    plt.grid(True)
    plt.title(f"RPE translation error over time (delta={delta})")
    plt.xlabel("time [s]")
    plt.ylabel("translation error [m]")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"rpe_trans_delta_{delta}.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(t, re)
    plt.grid(True)
    plt.title(f"RPE rotation error over time (delta={delta})")
    plt.xlabel("time [s]")
    plt.ylabel("rotation error [deg]")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"rpe_rot_delta_{delta}.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(t, gt_d, label="GT relative distance")
    plt.plot(t, est_d, label="Est relative distance")
    plt.grid(True)
    plt.legend()
    plt.title(f"Relative motion magnitude over time (delta={delta})")
    plt.xlabel("time [s]")
    plt.ylabel("relative translation magnitude [m]")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"rel_motion_delta_{delta}.png", dpi=160)
    plt.close()


def plot_z_over_time(T_est, T_gt, t_query, seg_id):
    valid = seg_id >= 0
    if not np.any(valid):
        return

    plt.figure()
    plt.plot(t_query[valid], T_gt[valid, 2, 3], label="GT Z")
    plt.plot(t_query[valid], T_est[valid, 2, 3], label="Est Z")
    plt.grid(True)
    plt.legend()
    plt.title("Z over time (valid mocap-covered queries only)")
    plt.xlabel("time [s]")
    plt.ylabel("Z [m]")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "z_over_time_valid_only.png", dpi=160)
    plt.close()


def plot_estimate_reference(T_est):
    """
    Optional reference-only plots of the estimator trajectory.
    No GT overlay here, since this evaluator is intentionally RPE-focused.
    """
    P = T_est[:, :3, 3]

    plt.figure()
    plt.plot(P[:, 0], P[:, 1])
    plt.axis("equal")
    plt.grid(True)
    plt.title("Estimator trajectory reference (X-Y)")
    plt.xlabel("X [units from estimator]")
    plt.ylabel("Y [units from estimator]")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "est_ref_xy.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(P[:, 2])
    plt.grid(True)
    plt.title("Estimator Z reference over frame index")
    plt.xlabel("frame")
    plt.ylabel("Z [units from estimator]")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "est_ref_z_over_frame.png", dpi=160)
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    for p in [EST_CSV, MOCAP_TXT, CAM_CALIB_JSON, MOCAP_IMU_CALIB_JSON]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    frame, est_t, p_est, R_est, has_rpy = load_est_csv(EST_CSV)
    mocap_t_rel, T_mocap_marker = load_mocap_data(MOCAP_TXT)
    dt = np.diff(mocap_t_rel)
    print("Mocap dt min/max [s]:", dt.min(), dt.max())
    print("Mocap negative dt count:", np.sum(dt < 0))
    print("Mocap zero dt count:", np.sum(dt == 0))

    cam_calib = load_json(CAM_CALIB_JSON)
    mocap_imu_calib = load_json(MOCAP_IMU_CALIB_JSON)

    T_world_mocap, T_marker_imu, T_imu_cam = build_constant_transforms(cam_calib, mocap_imu_calib)
    T_world_cam_mocap = build_world_cam_poses(T_mocap_marker, T_world_mocap, T_marker_imu, T_imu_cam)

    cam_t_rel, cam_t_source = load_left_camera_times()

    small_time_shift_s = 0.0
    if USE_SMALL_SENSOR_TIME_OFFSETS:
        cam_shift = float(cam_calib.get("cam_time_offset_ns", 0.0)) * 1e-9
        mocap_shift = float(mocap_imu_calib.get("mocap_time_offset_ns", 0.0)) * 1e-9
        small_time_shift_s = cam_shift - mocap_shift

    t_query_rel, t_query_source = build_query_times(frame, est_t, cam_t_rel)
    t_query_rel = np.asarray(t_query_rel, dtype=np.float64) + small_time_shift_s

    print("Unique query times:", len(np.unique(np.round(t_query_rel, 6))))
    print("First 20 query times:", t_query_rel[:20])

    N = len(est_t)
    if len(t_query_rel) != N:
        M = min(N, len(t_query_rel))
        frame = frame[:M]
        est_t = est_t[:M]
        p_est = p_est[:M]
        R_est = R_est[:M]
        t_query_rel = t_query_rel[:M]
        N = M

    # Build valid mocap segments and mark which segment each query belongs to.
    segments = build_mocap_segments(mocap_t_rel, MOCAP_GAP_THRESHOLD_S)
    seg_id = segment_id_for_queries(t_query_rel, segments, mocap_t_rel)

    valid_queries = seg_id >= 0
    print("Valid query count within mocap-covered segments:", int(np.sum(valid_queries)))
    print("Mocap segment count:", len(segments))

    # Interpolate GT for all queries; rows outside valid segments are simply ignored later.
    # Clamp only for numerical safety at the outer edges.
    t_query_clamped = np.clip(t_query_rel, mocap_t_rel[0], mocap_t_rel[-1])
    T_gt = interp_se3(T_world_cam_mocap, mocap_t_rel, t_query_clamped)

    T_est = np.stack([T_from_Rt(R_est[i], p_est[i]) for i in range(N)], axis=0)

    # No global alignment for RPE. Relative transforms are invariant to a fixed SE(3) offset.
    plot_estimate_reference(T_est)
    plot_z_over_time(T_est, T_gt, t_query_rel, seg_id)

    summaries = {}
    for delta in RPE_DELTAS:
        rows = compute_rpe_pairs(T_est, T_gt, t_query_rel, seg_id, delta)
        write_pairs_csv(OUTPUT_DIR / f"rpe_pairs_delta_{delta}.csv", rows)
        plot_rpe_series(rows, delta)
        summaries[delta] = summarize_rpe_rows(rows)

    write_summary_txt(
        OUTPUT_DIR / "summary.txt",
        DATASET_ROOT,
        cam_t_source if cam_t_source else "none",
        t_query_source,
        segments,
        summaries,
    )

    print("=== TUM-VI RPE-focused evaluation ===")
    print(f"DATASET_ROOT:                 {DATASET_ROOT}")
    print(f"Estimator CSV:                {EST_CSV}")
    print(f"Mocap TXT:                    {MOCAP_TXT}")
    print(f"Camera timestamp source:      {cam_t_source if cam_t_source else 'none'}")
    print(f"Chosen query time source:     {t_query_source}")
    print(f"Mocap rows:                   {len(mocap_t_rel)}")
    print(f"Mocap span [s]:               {mocap_t_rel[0]:.6f} -> {mocap_t_rel[-1]:.6f}")
    print(f"Valid queries in segments:    {int(np.sum(valid_queries))} / {N}")
    print(f"Applied small time shift [s]: {small_time_shift_s:.9f}")
    print(f"Gap threshold [s]:            {MOCAP_GAP_THRESHOLD_S}")
    if IMAGE_EXPOSURES_LEFT.exists():
        try:
            exp = load_numeric_text_file(IMAGE_EXPOSURES_LEFT)
            print(f"Exposure rows found:          {len(exp)} (not used in interpolation)")
        except Exception:
            print("Exposure rows found:          unreadable")

    for delta in RPE_DELTAS:
        s = summaries.get(delta)
        if s is None:
            print(f"RPE delta={delta:>3d}: no valid pairs")
            continue
        print(
            f"RPE delta={delta:>3d}: "
            f"count={s['count']}, "
            f"trans_rmse={s['trans_rmse_m']:.6f} m, "
            f"rot_rmse={s['rot_rmse_deg']:.6f} deg"
        )

    print(f"Wrote outputs to:             {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
