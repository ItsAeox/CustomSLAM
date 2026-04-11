#!/usr/bin/env python3
import os, glob, math, datetime
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Paths (edit if needed)
# --------------------------

DATASET_NAME = "2011_09_26_drive_0101_sync"
DATASET_ROOT = f"dataset/{DATASET_NAME}"
EST_CSV      = os.path.join(DATASET_ROOT, "kitti_poses.csv")
PERF_CSV     = os.path.join(DATASET_ROOT, "kitti_performance.csv")
IMG_TS       = os.path.join(DATASET_ROOT, "image_02", "timestamps.txt")
OXTS_TS      = os.path.join(DATASET_ROOT, "oxts", "timestamps.txt")
OXTS_DIR     = os.path.join(DATASET_ROOT, "oxts", "data")

CALIB_CAM2CAM = os.path.join(DATASET_ROOT, "calib_cam_to_cam.txt")
CALIB_VELO2CAM = os.path.join(DATASET_ROOT, "calib_velo_to_cam.txt")
CALIB_IMU2VELO = os.path.join(DATASET_ROOT, "calib_imu_to_velo.txt")

OUTPUT_DIR =  f"dataset/kitti_eval_v2/{DATASET_NAME}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------
# Utils: timestamps
# --------------------------
def parse_kitti_timestamp_line(line: str) -> float:
    # "2011-09-26 13:02:39.123456789"
    # Parse to epoch seconds (float). Keep fractional seconds.
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
    # KITTI timestamps are in local time; for relative timing we just use differences.
    return dt.timestamp() + frac_s

def load_kitti_timestamps(path: str) -> np.ndarray:
    lines = [l.strip() for l in open(path, "r").read().splitlines() if l.strip()]
    t = np.array([parse_kitti_timestamp_line(l) for l in lines], dtype=np.float64)
    # normalize to start at 0
    t -= t[0]
    return t

# --------------------------
# Utils: SO(3)/SE(3)
# --------------------------
def R_from_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # KITTI uses roll (x), pitch (y), yaw (z) in ENU-like navigation convention.
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]], dtype=np.float64)
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]], dtype=np.float64)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]], dtype=np.float64)
    return Rz @ Ry @ Rx

def T_from_Rt(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3, 3] = t
    return T

def invT(T):
    R = T[:3,:3]
    t = T[:3,3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3,:3] = R.T
    Ti[:3, 3] = -(R.T @ t)
    return Ti

def rot_angle(R):
    # robust angle from rotation matrix
    c = (np.trace(R) - 1.0) * 0.5
    c = min(1.0, max(-1.0, c))
    return math.acos(c)

def quat_from_R(R):
    # returns [w,x,y,z]
    m = R
    tr = m[0,0] + m[1,1] + m[2,2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m[2,1] - m[1,2]) / S
        y = (m[0,2] - m[2,0]) / S
        z = (m[1,0] - m[0,1]) / S
    elif (m[0,0] > m[1,1]) and (m[0,0] > m[2,2]):
        S = math.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2
        w = (m[2,1] - m[1,2]) / S
        x = 0.25 * S
        y = (m[0,1] + m[1,0]) / S
        z = (m[0,2] + m[2,0]) / S
    elif m[1,1] > m[2,2]:
        S = math.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2
        w = (m[0,2] - m[2,0]) / S
        x = (m[0,1] + m[1,0]) / S
        y = 0.25 * S
        z = (m[1,2] + m[2,1]) / S
    else:
        S = math.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2
        w = (m[1,0] - m[0,1]) / S
        x = (m[0,2] + m[2,0]) / S
        y = (m[1,2] + m[2,1]) / S
        z = 0.25 * S
    q = np.array([w,x,y,z], dtype=np.float64)
    return q / np.linalg.norm(q)

def R_from_quat(q):
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=np.float64)

def slerp(q0, q1, a):
    # shortest-path slerp
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        q = q0 + a*(q1-q0)
        return q / np.linalg.norm(q)
    th = math.acos(dot)
    s0 = math.sin((1-a)*th) / math.sin(th)
    s1 = math.sin(a*th) / math.sin(th)
    return s0*q0 + s1*q1

# --------------------------
# Parse KITTI calib files
# --------------------------
def parse_kitti_calib_txt(path):
    d = {}
    for line in open(path,"r").read().splitlines():
        line = line.strip()
        if not line or ":" not in line: 
            continue
        k, v = line.split(":", 1)
        d[k.strip()] = v.strip()
    return d

def mat_from_line(vals, rows, cols):
    a = np.fromstring(vals, sep=" ", dtype=np.float64)
    return a.reshape(rows, cols)

def load_T_imu_to_velo(path):
    d = parse_kitti_calib_txt(path)
    R = mat_from_line(d["R"], 3, 3)
    t = mat_from_line(d["T"], 3, 1).reshape(3)
    return T_from_Rt(R, t)

def load_T_velo_to_cam0(path):
    d = parse_kitti_calib_txt(path)
    R = mat_from_line(d["R"], 3, 3)
    t = mat_from_line(d["T"], 3, 1).reshape(3)
    return T_from_Rt(R, t)

def load_T_cam0_to_cam2(path):
    # calib_cam_to_cam.txt contains R_02 and T_02 (rectified) and P_rect_02, etc.
    d = parse_kitti_calib_txt(path)
    # For raw sequences, the extrinsic between cam0 and cam2 is given by R_02, T_02 (often identity-ish for reference),
    # but depending on file version, you may need "R_02" and "T_02" keys.
    # We handle both patterns robustly.
    keyR = "R_02"
    keyT = "T_02"
    if keyR in d and keyT in d:
        R = mat_from_line(d[keyR], 3, 3)
        t = mat_from_line(d[keyT], 3, 1).reshape(3)
        return T_from_Rt(R, t)
    # If not present, assume cam2 is the reference (some dumps use cam2 as main), fall back to identity.
    return np.eye(4, dtype=np.float64)

def build_T_imu_to_cam2():
    T_i_v = load_T_imu_to_velo(CALIB_IMU2VELO)
    T_v_c0 = load_T_velo_to_cam0(CALIB_VELO2CAM)
    T_c0_c2 = load_T_cam0_to_cam2(CALIB_CAM2CAM)
    # imu -> velo -> cam0 -> cam2
    return T_c0_c2 @ T_v_c0 @ T_i_v

# --------------------------
# OXTS -> local metric pose
# --------------------------
def read_oxts_file(path: str):
    vals = np.fromstring(open(path, "r").read().strip(), sep=" ")
    if vals.size < 6:
        raise ValueError(f"Bad OXTS line in {path}")
    lat, lon, alt, roll, pitch, yaw = vals[:6]
    return float(lat), float(lon), float(alt), float(roll), float(pitch), float(yaw)

def latlon_to_enu_m(lat, lon, alt, lat0, lon0, alt0):
    # local tangent plane approximation around the first point (good + standard for KITTI-style local frames)
    R = 6378137.0
    latr = np.deg2rad(lat)
    lonr = np.deg2rad(lon)
    lat0r = np.deg2rad(lat0)
    lon0r = np.deg2rad(lon0)
    dlat = latr - lat0r
    dlon = lonr - lon0r
    east  = R * dlon * np.cos(lat0r)
    north = R * dlat
    up    = alt - alt0
    return np.array([east, north, up], dtype=np.float64)

def load_oxts_poses(oxts_dir: str):
    files = sorted(glob.glob(os.path.join(oxts_dir, "*.txt")))
    if not files:
        raise FileNotFoundError(f"No OXTS files found in {oxts_dir}")
    first = read_oxts_file(files[0])
    lat0, lon0, alt0 = first[0], first[1], first[2]

    T_w_imu = []
    for f in files:
        lat, lon, alt, roll, pitch, yaw = read_oxts_file(f)
        t = latlon_to_enu_m(lat, lon, alt, lat0, lon0, alt0)  # ENU meters
        R = R_from_rpy(roll, pitch, yaw)
        T_w_imu.append(T_from_Rt(R, t))
    return np.stack(T_w_imu, axis=0), files

# --------------------------
# Interpolate SE(3) poses by timestamp
# --------------------------
def interp_se3(T_list, t_list, t_query):
    # T_list: Nx4x4, t_list: N
    # For each query time, find bracketing indices and interpolate:
    # translation linear, rotation slerp
    tq = np.asarray(t_query, dtype=np.float64)
    out = np.zeros((tq.size, 4, 4), dtype=np.float64)
    out[:] = np.eye(4)

    qs = np.array([quat_from_R(T_list[i,:3,:3]) for i in range(len(T_list))], dtype=np.float64)
    ps = np.array([T_list[i,:3,3] for i in range(len(T_list))], dtype=np.float64)

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

# --------------------------
# Read your estimated CSV (no pandas)
# --------------------------
def load_est_csv(path):
    # expects header: frame,t,x,y,z,yaw_rad,pitch_rad,roll_rad
    # (yaw/pitch/roll optional; if missing, rotation eval will be skipped)
    lines = [l.strip() for l in open(path,"r").read().splitlines() if l.strip()]
    header = lines[0].split(",")
    idx = {name:i for i,name in enumerate(header)}
    req = ["frame","t","x","y","z"]
    for r in req:
        if r not in idx:
            raise ValueError(f"Estimator CSV missing column '{r}'")

    has_rpy = all(k in idx for k in ["yaw_rad","pitch_rad","roll_rad"])
    N = len(lines)-1
    frame = np.zeros(N, dtype=np.int64)
    t = np.zeros(N, dtype=np.float64)
    p = np.zeros((N,3), dtype=np.float64)
    R = None
    if has_rpy:
        R = np.zeros((N,3,3), dtype=np.float64)

    for i in range(N):
        parts = lines[i+1].split(",")
        frame[i] = int(float(parts[idx["frame"]]))
        t[i] = float(parts[idx["t"]])
        p[i,0] = float(parts[idx["x"]])
        p[i,1] = float(parts[idx["y"]])
        p[i,2] = float(parts[idx["z"]])
        if has_rpy:
            yaw = float(parts[idx["yaw_rad"]])
            pitch = float(parts[idx["pitch_rad"]])
            roll = float(parts[idx["roll_rad"]])
            # Your system comment said OpenCV-ish axes; this R is only used for eval if you want.
            # If your yaw/pitch/roll definition differs from KITTI nav rpy, rotation eval can be misleading.
            R[i] = R_from_rpy(roll, pitch, yaw)

    return frame, t, p, R, has_rpy

def load_perf_csv(path):
    lines = [l.strip() for l in open(path, "r").read().splitlines() if l.strip()]
    header = lines[0].split(",")
    idx = {name:i for i,name in enumerate(header)}

    req = [
        "frame", "t",
        "num_kfs", "num_mps", "num_keypoints",
        "wasm_klt_ms", "wasm_total_ms", "wasm_imu_ms", "wasm_seed_ms",
        "imu_hz"
    ]
    for r in req:
        if r not in idx:
            raise ValueError(f"Performance CSV missing column '{r}'")

    N = len(lines) - 1
    out = {k: np.zeros(N, dtype=np.float64) for k in req}
    out["frame"] = np.zeros(N, dtype=np.int64)

    for i in range(N):
        parts = lines[i+1].split(",")
        out["frame"][i] = int(float(parts[idx["frame"]]))
        for k in req[1:]:
            out[k][i] = float(parts[idx[k]])

    return out

# --------------------------
# Alignment: SE(3) only (no scale)
# --------------------------

def align_se3_by_first_frame(T_est, T_gt):
    # Align so that first pose matches exactly: T_est_aligned = A * T_est, where A = T_gt0 * inv(T_est0)
    A = T_gt[0] @ invT(T_est[0])
    T_al = np.einsum("ij,njk->nik", A, T_est)
    return T_al, A

# --------------------------
# Metrics: ATE (SE(3)), RPE
# --------------------------
def ate_translation_rmse(T_est, T_gt):
    e = T_est[:, :3, 3] - T_gt[:, :3, 3]
    return math.sqrt(np.mean(np.sum(e*e, axis=1)))

def ate_rotation_rmse_deg(T_est, T_gt):
    ang = []
    for i in range(len(T_est)):
        dR = T_gt[i,:3,:3].T @ T_est[i,:3,:3]
        ang.append(rot_angle(dR))
    ang = np.array(ang, dtype=np.float64)
    return float(np.sqrt(np.mean((ang*180.0/math.pi)**2)))

def rpe(T_est, T_gt, delta=1):
    # relative pose error between i and i+delta
    trans = []
    rot = []
    for i in range(len(T_est)-delta):
        d_est = invT(T_est[i]) @ T_est[i+delta]
        d_gt  = invT(T_gt[i])  @ T_gt[i+delta]
        d_err = invT(d_gt) @ d_est
        trans.append(np.linalg.norm(d_err[:3,3]))
        rot.append(rot_angle(d_err[:3,:3]) * 180.0/math.pi)
    return np.array(trans), np.array(rot)

def ate_translation_rmse_xy(T_est, T_gt):
    e_xy = T_est[:, :2, 3] - T_gt[:, :2, 3]
    return math.sqrt(np.mean(np.sum(e_xy * e_xy, axis=1)))

# --------------------------
# Main
# --------------------------
def main():
    # Load timestamps
    t_cam = load_kitti_timestamps(IMG_TS)
    t_oxts = load_kitti_timestamps(OXTS_TS)

    # Load OXTS world->IMU poses
    T_w_imu, _ = load_oxts_poses(OXTS_DIR)

    # Convert to world->cam2 poses using calib
    T_i_c2 = build_T_imu_to_cam2()           # imu -> cam2
    T_w_c2 = np.einsum("nij,jk->nik", T_w_imu, T_i_c2)

    # Interpolate GT cam pose to camera timestamps
    # (OXTS is usually same length and near-synced, but we do it correctly)
    T_gt_cam = interp_se3(T_w_c2, t_oxts, t_cam)

    # Load estimator
    frame, t_est, p_est, R_est, has_rpy = load_est_csv(EST_CSV)

    # Load performance CSV if present
    perf = None
    if os.path.exists(PERF_CSV):
        perf = load_perf_csv(PERF_CSV)

    # Choose estimator timestamps:
    # If your CSV t is synthetic, you MUST switch to camera timestamps to be rigorous.
    # Here we map each CSV row i -> t_cam[i] (the real camera timebase).
    N = min(len(p_est), len(t_cam), len(T_gt_cam))
    t_use = t_cam[:N]

    # Build estimator SE(3). If you don't have reliable rotation, we still evaluate translation.
    T_est = np.zeros((N,4,4), dtype=np.float64)
    T_est[:] = np.eye(4)
    for i in range(N):
        if has_rpy:
            T_est[i,:3,:3] = R_est[i]
        T_est[i,:3, 3] = p_est[i]

    T_gt = T_gt_cam[:N]

    # SE(3) alignment (no scale): fix initial frame
    T_est_al, A = align_se3_by_first_frame(T_est, T_gt)

    # Metrics
    ate_t = ate_translation_rmse(T_est_al, T_gt)
    ate_xy = ate_translation_rmse_xy(T_est_al, T_gt)
    print("=== KITTI Raw VIO eval (SE(3), timestamps, cam frame) ===")
    print(f"Frames used: {N}")
    print(f"ATE trans RMSE 3D (m): {ate_t:.4f}")
    print(f"ATE trans RMSE XY (m): {ate_xy:.4f}")
    
    if has_rpy:
        ate_r = ate_rotation_rmse_deg(T_est_al, T_gt)
        print(f"ATE rot  RMSE (deg): {ate_r:.4f}")
    else:
        print("Rotation eval: SKIPPED (estimator CSV missing yaw/pitch/roll columns or they are not reliable).")

    # RPE at multiple deltas (1 frame, 10 frames ~1s if 10Hz)
    for d in [1, 10, 50]:
        if N > d+1:
            te, re = rpe(T_est_al, T_gt, delta=d)
            print(f"RPE delta={d}: trans mean={te.mean():.4f} m, trans rmse={math.sqrt(np.mean(te*te)):.4f} m, "
                  f"rot mean={re.mean():.4f} deg, rot rmse={math.sqrt(np.mean(re*re)):.4f} deg")

    # Save aligned trajectories
    out_csv = os.path.join(OUTPUT_DIR, "poses_aligned_se3.csv")
    with open(out_csv, "w") as f:
        f.write("i,t,est_x,est_y,est_z,gt_x,gt_y,gt_z,err_xy,err_3d\n")
        for i in range(N):
            ex,ey,ez = T_est_al[i,:3,3]
            gx,gy,gz = T_gt[i,:3,3]
            err_xy = math.sqrt((ex - gx)**2 + (ey - gy)**2)
            err_3d = math.sqrt((ex - gx)**2 + (ey - gy)**2 + (ez - gz)**2)
            f.write(f"{i},{t_use[i]:.9f},{ex:.9f},{ey:.9f},{ez:.9f},{gx:.9f},{gy:.9f},{gz:.9f},{err_xy:.9f},{err_3d:.9f}\n")
    print(f"Wrote: {out_csv}")

    # Plots
    estP = T_est_al[:,:3,3]
    gtP  = T_gt[:,:3,3]
    err = np.linalg.norm(estP - gtP, axis=1)

    # Top-down EN (East-North)
    plt.figure()
    plt.plot(gtP[:,0], gtP[:,1], label="GT cam (ENU)")
    plt.plot(estP[:,0], estP[:,1], label="Est aligned SE(3)")
    plt.axis("equal"); plt.grid(True); plt.legend()
    plt.title(f"Trajectory top-down (East-North), XY RMSE={ate_xy:.3f} m")
    plt.savefig(os.path.join(OUTPUT_DIR, "traj_EN.png"), dpi=160)

    # Up component
    plt.figure()
    plt.plot(gtP[:,2], label="GT Up")
    plt.plot(estP[:,2], label="Est Up")
    plt.grid(True); plt.legend()
    plt.title("Up (altitude) over time")
    plt.savefig(os.path.join(OUTPUT_DIR, "up_over_time.png"), dpi=160)

    # Error over time
    err_xy = np.linalg.norm(estP[:, :2] - gtP[:, :2], axis=1)

    plt.figure()
    plt.plot(err_xy)
    plt.grid(True)
    plt.title(f"Bird's-eye position error (XY), RMSE={ate_xy:.3f} m")
    plt.xlabel("frame")
    plt.ylabel("meters")
    plt.savefig(os.path.join(OUTPUT_DIR, "pos_error_xy.png"), dpi=160)

    plt.figure()
    plt.plot(err)
    plt.grid(True)
    plt.title(f"Position error 3D ||p_est - p_gt|| (m), ATE RMSE={ate_t:.3f}")
    plt.xlabel("frame")
    plt.ylabel("meters")
    plt.savefig(os.path.join(OUTPUT_DIR, "pos_error_3d.png"), dpi=160)

    if perf is not None:
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
        plt.savefig(os.path.join(OUTPUT_DIR, "perf_keypoints.png"), dpi=160)

        plt.figure()
        plt.plot(pf, perf["num_kfs"][:Np], label="KFs")
        plt.plot(pf, perf["num_mps"][:Np], label="MPs")
        plt.grid(True)
        plt.legend()
        plt.title("Keyframes and map points over frames")
        plt.xlabel("frame")
        plt.ylabel("count")
        plt.savefig(os.path.join(OUTPUT_DIR, "perf_map_structure.png"), dpi=160)

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
        plt.savefig(os.path.join(OUTPUT_DIR, "perf_wasm_timing.png"), dpi=160)

        plt.figure()
        plt.plot(pf, perf["imu_hz"][:Np])
        plt.grid(True)
        plt.title("IMU rate over frames")
        plt.xlabel("frame")
        plt.ylabel("Hz")
        plt.savefig(os.path.join(OUTPUT_DIR, "perf_imu_hz.png"), dpi=160)

    print(f"Wrote plots into: {OUTPUT_DIR}/")
    print("Trajectory/Error plots: traj_EN.png, up_over_time.png, pos_error_xy.png, pos_error_3d.png")
    if perf is not None:
        print("Performance plots: perf_keypoints.png, perf_map_structure.png, perf_wasm_timing.png, perf_imu_hz.png")

if __name__ == "__main__":
    main()
