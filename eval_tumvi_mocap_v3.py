#!/usr/bin/env python3
"""
Evaluate a TUM-VI / TUM-VI-like trajectory CSV against mocap ground truth.

This version is tuned for your loop-floor0 layout:
    DATASET_ROOT = Path("dataset/loop-floor0")

Expected files:
  dataset/loop-floor0/
    tumvi_poses.csv
    mocap_data.txt
    camera-calibration.json
    mocap-imu-calibration.json
    left_images/
      image_timestamps_left.txt
      image_exposures_left.txt   # optional, only used for diagnostics

Key fixes:
  1) Uses left_images/image_timestamps_left.txt as the primary camera time source.
  2) Uses estimator CSV 'frame' values to index camera timestamps when possible.
     This is much better than assuming CSV row i == image timestamp i.
  3) Falls back gracefully when frame indexing is not possible.
  4) Prints debug info so it is easier to catch timebase disasters.
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
DATASET_ROOT = Path("dataset/mocap-desk")
EST_CSV = DATASET_ROOT / "tumvi_poses.csv"
MOCAP_TXT = DATASET_ROOT / "mocap_data.txt"
CAM_CALIB_JSON = DATASET_ROOT / "camera-calibration.json"
MOCAP_IMU_CALIB_JSON = DATASET_ROOT / "mocap-imu-calibration.json"
LEFT_IMAGES_DIR = DATASET_ROOT / "left_images"
IMAGE_TIMESTAMPS_LEFT = LEFT_IMAGES_DIR / "image_timestamps_left.txt"
IMAGE_EXPOSURES_LEFT = LEFT_IMAGES_DIR / "image_exposures_left.txt"
OUTPUT_DIR = DATASET_ROOT / "outputs"
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

    # fallback only if truly unknown
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
    """
    Prefer camera timestamps addressed by frame index.
    Fallback to row-order if needed.
    Fallback to estimator time if camera timestamps unavailable.
    """
    if cam_t is None or len(cam_t) == 0:
        return np.asarray(est_t, dtype=np.float64), "estimator_csv_t"

    frame_idx = np.asarray(frame_idx, dtype=np.int64)
    valid = (frame_idx >= 0) & (frame_idx < len(cam_t))

    # Best case: every frame index maps cleanly.
    if np.all(valid):
        return cam_t[frame_idx], "camera_timestamps_by_frame_index"

    # Partial mapping: if most are valid, keep only valid timestamp lookup and fill
    # missing entries from row-order or estimator time.
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

    # Otherwise row order is safer than bogus frame lookup.
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
# Metrics
# -----------------------------------------------------------------------------
def align_se3_by_first_frame(T_est, T_gt):
    A = T_gt[0] @ invT(T_est[0])
    return np.einsum("ij,njk->nik", A, T_est), A


def ate_translation_rmse(T_est, T_gt):
    e = T_est[:, :3, 3] - T_gt[:, :3, 3]
    return math.sqrt(np.mean(np.sum(e * e, axis=1)))


def ate_rotation_rmse_deg(T_est, T_gt):
    ang = []
    for i in range(len(T_est)):
        dR = T_gt[i, :3, :3].T @ T_est[i, :3, :3]
        ang.append(rot_angle(dR))
    ang = np.asarray(ang, dtype=np.float64)
    return float(np.sqrt(np.mean((ang * 180.0 / math.pi) ** 2)))


def rpe(T_est, T_gt, delta=1):
    trans = []
    rot = []
    for i in range(len(T_est) - delta):
        d_est = invT(T_est[i]) @ T_est[i + delta]
        d_gt = invT(T_gt[i]) @ T_gt[i + delta]
        d_err = invT(d_gt) @ d_est
        trans.append(np.linalg.norm(d_err[:3, 3]))
        rot.append(rot_angle(d_err[:3, :3]) * 180.0 / math.pi)
    return np.asarray(trans), np.asarray(rot)


def path_length(T):
    p = T[:, :3, 3]
    if len(p) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def axis_rmse(estP, gtP):
    d = estP - gtP
    return np.sqrt(np.mean(d * d, axis=0))


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
# Output helpers
# -----------------------------------------------------------------------------
def write_aligned_csv(path, t, T_est_al, T_gt):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "i", "t",
            "est_x", "est_y", "est_z",
            "gt_x", "gt_y", "gt_z",
            "pos_err_m", "rot_err_deg",
        ])
        for i in range(len(T_est_al)):
            estp = T_est_al[i, :3, 3]
            gtp = T_gt[i, :3, 3]
            perr = float(np.linalg.norm(estp - gtp))
            rerr = float(rot_angle(T_gt[i, :3, :3].T @ T_est_al[i, :3, :3]) * 180.0 / math.pi)
            w.writerow([
                i, f"{t[i]:.9f}",
                *[f"{v:.9f}" for v in estp],
                *[f"{v:.9f}" for v in gtp],
                f"{perr:.9f}", f"{rerr:.9f}",
            ])


def make_plots(T_est_al, T_gt, pos_err, rot_err_deg):
    estP = T_est_al[:, :3, 3]
    gtP = T_gt[:, :3, 3]

    plt.figure()
    plt.plot(gtP[:, 0], gtP[:, 1], label="GT")
    plt.plot(estP[:, 0], estP[:, 1], label="Est aligned")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("Trajectory projection (X-Y)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "traj_xy.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(gtP[:, 0], gtP[:, 2], label="GT")
    plt.plot(estP[:, 0], estP[:, 2], label="Est aligned")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("Trajectory projection (X-Z)")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "traj_xz.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(pos_err)
    plt.grid(True)
    plt.title("Position error over time")
    plt.xlabel("frame")
    plt.ylabel("meters")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pos_error.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(rot_err_deg)
    plt.grid(True)
    plt.title("Rotation error over time")
    plt.xlabel("frame")
    plt.ylabel("degrees")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rot_error.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(gtP[:, 0], gtP[:, 1], label="GT only")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("GT only (X-Y)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "traj_xy_gt_only.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(gtP[:, 0], gtP[:, 2], label="GT only")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("GT only (X-Z)")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "traj_xz_gt_only.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(gtP[:, 1], gtP[:, 2], label="GT only")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title("GT only (Y-Z)")
    plt.xlabel("Y (m)")
    plt.ylabel("Z (m)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "traj_yz_gt_only.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(gtP[:, 2], label="GT Z")
    plt.plot(estP[:, 2], label="Est aligned Z")
    plt.grid(True)
    plt.legend()
    plt.title("Z over time")
    plt.xlabel("frame")
    plt.ylabel("Z (m)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "z_over_time.png", dpi=160)
    plt.close()

def save_debug_proj(T_series, prefix):
    P = T_series[:, :3, 3]

    plt.figure()
    plt.plot(P[:, 0], P[:, 1])
    plt.axis("equal")
    plt.grid(True)
    plt.title(f"{prefix} (X-Y)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{prefix.lower().replace(' ', '_')}_xy.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(P[:, 0], P[:, 2])
    plt.axis("equal")
    plt.grid(True)
    plt.title(f"{prefix} (X-Z)")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{prefix.lower().replace(' ', '_')}_xz.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(P[:, 1], P[:, 2])
    plt.axis("equal")
    plt.grid(True)
    plt.title(f"{prefix} (Y-Z)")
    plt.xlabel("Y (m)")
    plt.ylabel("Z (m)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{prefix.lower().replace(' ', '_')}_yz.png", dpi=160)
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
    # Build intermediate stage
    T_world_marker = np.einsum("ij,njk->nik", T_world_mocap, T_mocap_marker)

    # Debug plots (VERY IMPORTANT — before interpolation)
    save_debug_proj(T_mocap_marker, "Raw Mocap Marker")
    save_debug_proj(T_world_marker, "World Marker")
    save_debug_proj(T_world_cam_mocap, "World Cam Mocap")

    cam_t_rel, cam_t_source = load_left_camera_times()

    small_time_shift_s = 0.0
    if USE_SMALL_SENSOR_TIME_OFFSETS:
        cam_shift = float(cam_calib.get("cam_time_offset_ns", 0.0)) * 1e-9
        mocap_shift = float(mocap_imu_calib.get("mocap_time_offset_ns", 0.0)) * 1e-9
        small_time_shift_s = cam_shift - mocap_shift

    t_query_rel, t_query_source = build_query_times(frame, est_t, cam_t_rel)
    t_query_rel = np.asarray(t_query_rel, dtype=np.float64) + small_time_shift_s
    t_query_rel = np.clip(t_query_rel, mocap_t_rel[0], mocap_t_rel[-1])

    print("Unique query times:", len(np.unique(np.round(t_query_rel, 3))))
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

    T_gt = interp_se3(T_world_cam_mocap, mocap_t_rel, t_query_rel)
    T_est = np.stack([T_from_Rt(R_est[i], p_est[i]) for i in range(N)], axis=0)
    T_est_al, A_est0_to_gt0 = align_se3_by_first_frame(T_est, T_gt)

    pos_err = np.linalg.norm(T_est_al[:, :3, 3] - T_gt[:, :3, 3], axis=1)
    rot_err_deg = np.array([
        rot_angle(T_gt[i, :3, :3].T @ T_est_al[i, :3, :3]) * 180.0 / math.pi
        for i in range(N)
    ], dtype=np.float64)

    ate_t = ate_translation_rmse(T_est_al, T_gt)
    ate_r = ate_rotation_rmse_deg(T_est_al, T_gt)
    est_len = path_length(T_est_al)
    gt_len = path_length(T_gt)
    ax_rmse = axis_rmse(T_est_al[:, :3, 3], T_gt[:, :3, 3])

    print("=== TUM-VI mocap evaluation ===")
    print(f"DATASET_ROOT:                {DATASET_ROOT}")
    print(f"Estimator CSV:               {EST_CSV}")
    print(f"Mocap TXT:                   {MOCAP_TXT}")
    print(f"Camera timestamp source:     {cam_t_source if cam_t_source else 'none'}")
    print(f"Chosen query time source:    {t_query_source}")
    print(f"Camera count:                {0 if cam_t_rel is None else len(cam_t_rel)}")
    print(f"Estimator rows:              {N}")
    print(f"Mocap rows:                  {len(mocap_t_rel)}")
    print(f"Mocap span [s]:              {mocap_t_rel[0]:.6f} -> {mocap_t_rel[-1]:.6f}")
    print(f"Query span [s]:              {t_query_rel[0]:.6f} -> {t_query_rel[-1]:.6f}")
    print(f"Applied small time shift [s]: {small_time_shift_s:.9f}")
    if IMAGE_EXPOSURES_LEFT.exists():
        try:
            exp = load_numeric_text_file(IMAGE_EXPOSURES_LEFT)
            print(f"Exposure rows found:         {len(exp)} (not used in interpolation)")
        except Exception:
            print("Exposure rows found:         unreadable")
    print(f"GT path length [m]:          {gt_len:.6f}")
    print(f"Est path length [m]:         {est_len:.6f}")
    print(f"ATE transl RMSE [m]:         {ate_t:.6f}")
    print(f"ATE rot RMSE [deg]:          {ate_r:.6f}")
    print(f"Axis RMSE [m]:               x={ax_rmse[0]:.6f}, y={ax_rmse[1]:.6f}, z={ax_rmse[2]:.6f}")

    for d in RPE_DELTAS:
        if N > d:
            tr, rr = rpe(T_est_al, T_gt, delta=d)
            print(
                f"RPE delta={d:>3d}: "
                f"trans_rmse={np.sqrt(np.mean(tr * tr)):.6f} m, "
                f"rot_rmse={np.sqrt(np.mean(rr * rr)):.6f} deg"
            )

    write_aligned_csv(OUTPUT_DIR / "poses_aligned_se3.csv", est_t[:N], T_est_al, T_gt)
    make_plots(T_est_al, T_gt, pos_err, rot_err_deg)
    print(f"Wrote outputs to:            {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
