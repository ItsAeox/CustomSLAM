#!/usr/bin/env python3
"""
Generate an offline stereo+IMU reference trajectory for a TUM-VIE/TUM-VI-like
folder layout.

Important:
- This does NOT create official ground truth.
- It creates an offline reference trajectory using stereo visual odometry with
  gyro-based orientation prior / fallback.
- Output format matches the user's browser pipeline CSV:
      frame,t,x,y,z,yaw_rad,pitch_rad,roll_rad

Expected folder layout:
  dataset/loop-floor0/
    left_images/
      image_timestamps_left.txt
      image_exposures_left.txt      # optional, ignored by solver
      00000.jpg
      00001.jpg
      ...
    right_images/
      image_timestamps_right.txt
      image_exposures_right.txt     # optional, ignored by solver
      00000.jpg
      00001.jpg
      ...
    camera-calibration.json
    imu_data.txt

Calibration assumptions:
- camera-calibration.json contains T_imu_cam and intrinsics arrays.
- cam index 0 = left visual camera
- cam index 1 = right visual camera
- intrinsics use Kannala-Brandt 4 coefficients (OpenCV fisheye model).

Runtime dependencies:
  pip install numpy opencv-python scipy

Usage examples:
  python3 generate_offline_reference_tumvie.py \
      --dataset-root dataset/loop-floor0 \
      --output-csv dataset/loop-floor0/offline_reference_poses.csv

Optional debugging:
  --save-preview-dir dataset/loop-floor0/offline_debug
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


def T_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invT(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -(R.T @ t)
    return Ti


def compose(T_a_b: np.ndarray, T_b_c: np.ndarray) -> np.ndarray:
    return T_a_b @ T_b_c


def quat_xyzw_to_wxyz(q_xyzw: Sequence[float]) -> np.ndarray:
    qx, qy, qz, qw = [float(v) for v in q_xyzw]
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("Zero quaternion encountered")
    return q / n


def R_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def euler_yaw_pitch_roll_from_R(R: np.ndarray) -> Tuple[float, float, float]:
    """Return yaw(Z), pitch(Y), roll(X), matching the user's CSV naming."""
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return yaw, pitch, roll


def rodrigues_to_R(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return R.astype(np.float64)


def R_to_rodrigues(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(3).astype(np.float64)


def normalize_time_array_us(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return arr
    return (arr - arr[0]) * 1e-6


def load_json(path: Path):
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and len(obj) == 1 and "value0" in obj:
        return obj["value0"]
    return obj


def tf_from_pose_dict(d: dict) -> np.ndarray:
    q = quat_xyzw_to_wxyz([d["qx"], d["qy"], d["qz"], d["qw"]])
    R = R_from_quat_wxyz(q)
    t = np.array([d["px"], d["py"], d["pz"]], dtype=np.float64)
    return T_from_Rt(R, t)


def load_numeric_text_file(path: Path) -> np.ndarray:
    vals: List[float] = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            vals.append(float(s))
    return np.asarray(vals, dtype=np.float64)


@dataclass
class CameraModel:
    K: np.ndarray
    D: np.ndarray
    resolution: Tuple[int, int]
    T_imu_cam: np.ndarray


@dataclass
class StereoRig:
    left: CameraModel
    right: CameraModel
    T_left_right: np.ndarray
    baseline_m: float


@dataclass
class ImuPacket:
    t: float
    gyro: np.ndarray
    accel: np.ndarray
    temperature: Optional[float]


@dataclass
class FramePacket:
    index: int
    t: float
    left_path: Path
    right_path: Path


class TumVieSequence:
    def __init__(self, dataset_root: Path):
        self.dataset_root = Path(dataset_root)
        self.left_dir = self.dataset_root / "left_images"
        self.right_dir = self.dataset_root / "right_images"
        self.calib_json = self.dataset_root / "camera-calibration.json"
        self.imu_txt = self.dataset_root / "imu_data.txt"

        self.left_ts_txt = self.left_dir / "image_timestamps_left.txt"
        self.right_ts_txt = self.right_dir / "image_timestamps_right.txt"

        for p in [self.left_dir, self.right_dir, self.calib_json, self.imu_txt,
                  self.left_ts_txt, self.right_ts_txt]:
            if not p.exists():
                raise FileNotFoundError(f"Missing required path: {p}")

    def load_stereo_rig(self) -> StereoRig:
        calib = load_json(self.calib_json)
        if "T_imu_cam" not in calib or "intrinsics" not in calib:
            raise ValueError("camera-calibration.json missing T_imu_cam / intrinsics")

        def build_cam(i: int) -> CameraModel:
            intr = calib["intrinsics"][i]["intrinsics"]
            K = np.ascontiguousarray(np.array([
                [float(intr["fx"]), 0.0, float(intr["cx"])],
                [0.0, float(intr["fy"]), float(intr["cy"])],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64))
            D = np.ascontiguousarray(np.array([
                float(intr["k1"]),
                float(intr["k2"]),
                float(intr["k3"]),
                float(intr["k4"]),
            ], dtype=np.float64).reshape(1, 4))
            res = tuple(int(v) for v in calib["resolution"][i])
            T_imu_cam = tf_from_pose_dict(calib["T_imu_cam"][i])
            return CameraModel(K=K, D=D, resolution=res, T_imu_cam=T_imu_cam)

        left = build_cam(0)
        right = build_cam(1)
        T_left_imu = invT(left.T_imu_cam)
        T_imu_right = right.T_imu_cam
        T_left_right = T_left_imu @ T_imu_right
        baseline = float(np.linalg.norm(T_left_right[:3, 3]))
        return StereoRig(left=left, right=right, T_left_right=T_left_right, baseline_m=baseline)

    def load_imu(self) -> List[ImuPacket]:
        packets: List[ImuPacket] = []
        raw_rows = []
        with open(self.imu_txt, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                vals = np.fromstring(s, sep=" ", dtype=np.float64)
                if vals.size < 7:
                    continue
                raw_rows.append(vals)

        if not raw_rows:
            raise ValueError(f"No IMU data found in {self.imu_txt}")

        t0_us = float(raw_rows[0][0])
        for vals in raw_rows:
            t = float((vals[0] - t0_us) * 1e-6)
            gyro = vals[1:4].astype(np.float64)
            accel = vals[4:7].astype(np.float64)
            temp = float(vals[7]) if vals.size >= 8 else None
            packets.append(ImuPacket(t=t, gyro=gyro, accel=accel, temperature=temp))
        return packets

    def load_frames(self) -> List[FramePacket]:
        left_ts = normalize_time_array_us(load_numeric_text_file(self.left_ts_txt))
        right_ts = normalize_time_array_us(load_numeric_text_file(self.right_ts_txt))

        left_files = sorted(self.left_dir.glob("*.jpg"))
        right_files = sorted(self.right_dir.glob("*.jpg"))

        if not left_files or not right_files:
            raise ValueError("No JPG files found in left_images/right_images")

        n = min(len(left_ts), len(right_ts), len(left_files), len(right_files))
        if n < 2:
            raise ValueError("Need at least 2 synchronized stereo frames")

        frames: List[FramePacket] = []
        for i in range(n):
            t = 0.5 * (left_ts[i] + right_ts[i])
            frames.append(FramePacket(index=i, t=float(t), left_path=left_files[i], right_path=right_files[i]))
        return frames


class ImuIntegrator:
    def __init__(self, packets: Sequence[ImuPacket]):
        self.packets = list(packets)
        self.times = np.array([p.t for p in self.packets], dtype=np.float64)

    def integrate_delta_rotation(self, t0: float, t1: float) -> np.ndarray:
        """
        Gyro-only rotation preintegration in the IMU frame.
        Returns R_i0_i1.
        """
        if t1 <= t0:
            return np.eye(3, dtype=np.float64)

        mask = (self.times >= t0) & (self.times <= t1)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            i0 = max(0, int(np.searchsorted(self.times, t0) - 1))
            i1 = min(len(self.packets) - 1, int(np.searchsorted(self.times, t1)))
            idx = np.arange(i0, max(i0 + 1, i1 + 1))

        R = np.eye(3, dtype=np.float64)
        last_t = t0
        for j in idx:
            pkt = self.packets[j]
            cur_t = min(max(pkt.t, t0), t1)
            dt = cur_t - last_t
            if dt > 0:
                omega = pkt.gyro * dt
                R = R @ rodrigues_to_R(omega)
                last_t = cur_t
        if last_t < t1:
            pkt = self.packets[idx[-1]]
            dt = t1 - last_t
            omega = pkt.gyro * dt
            R = R @ rodrigues_to_R(omega)
        return R

    def estimate_initial_world_R_imu(self, duration_s: float = 0.5) -> np.ndarray:
        """
        Crude gravity alignment from early accelerometer average.
        World convention: +Y up, gravity = [0, -1, 0] in world.
        """
        end_t = min(self.times[-1], duration_s)
        sel = [p.accel for p in self.packets if p.t <= end_t]
        if len(sel) < 5:
            return np.eye(3, dtype=np.float64)

        a = np.mean(np.asarray(sel, dtype=np.float64), axis=0)
        na = np.linalg.norm(a)
        if na < 1e-6:
            return np.eye(3, dtype=np.float64)
        a = a / na

        target = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        v = np.cross(a, target)
        c = float(np.dot(a, target))
        if np.linalg.norm(v) < 1e-8:
            return np.eye(3, dtype=np.float64) if c > 0 else np.diag([1, -1, -1])
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ], dtype=np.float64)
        s = np.linalg.norm(v)
        R = np.eye(3, dtype=np.float64) + vx + vx @ vx * ((1 - c) / (s * s))
        return R


class StereoRectifier:
    def __init__(self, rig: StereoRig):
        self.rig = rig
        width, height = rig.left.resolution
        image_size = (int(width), int(height))

        R = np.ascontiguousarray(rig.T_left_right[:3, :3].astype(np.float64))
        t = np.ascontiguousarray(rig.T_left_right[:3, 3].astype(np.float64).reshape(3, 1))

        K1 = np.ascontiguousarray(rig.left.K.astype(np.float64))
        D1 = np.ascontiguousarray(rig.left.D.astype(np.float64).reshape(1, 4))
        K2 = np.ascontiguousarray(rig.right.K.astype(np.float64))
        D2 = np.ascontiguousarray(rig.right.D.astype(np.float64).reshape(1, 4))

        flags = cv2.CALIB_ZERO_DISPARITY
        self.R1, self.R2, self.P1, self.P2, self.Q = cv2.fisheye.stereoRectify(
            K1, D1,
            K2, D2,
            image_size,
            R, t,
            flags=flags,
            newImageSize=image_size,
            balance=0.0,
            fov_scale=1.0,
        )

        self.map1_l, self.map2_l = cv2.fisheye.initUndistortRectifyMap(
            K1, D1, self.R1, self.P1, image_size, cv2.CV_32FC1
        )
        self.map1_r, self.map2_r = cv2.fisheye.initUndistortRectifyMap(
            K2, D2, self.R2, self.P2, image_size, cv2.CV_32FC1
        )

        self.fx = float(self.P1[0, 0])
        self.fy = float(self.P1[1, 1])
        self.cx = float(self.P1[0, 2])
        self.cy = float(self.P1[1, 2])
        self.Tx = float(-self.P2[0, 3] / self.P2[0, 0])
        self.baseline = abs(self.Tx)

        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16 * 12,
            blockSize=7,
            P1=8 * 1 * 7 * 7,
            P2=32 * 1 * 7 * 7,
            disp12MaxDiff=1,
            uniquenessRatio=8,
            speckleWindowSize=100,
            speckleRange=2,
            preFilterCap=31,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def read_and_rectify(self, left_path: Path, right_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
        if left is None or right is None:
            raise FileNotFoundError(f"Failed to read stereo pair: {left_path} / {right_path}")
        left_r = cv2.remap(left, self.map1_l, self.map2_l, interpolation=cv2.INTER_LINEAR)
        right_r = cv2.remap(right, self.map1_r, self.map2_r, interpolation=cv2.INTER_LINEAR)
        return left_r, right_r

    def disparity_to_depth(self, disp: np.ndarray) -> np.ndarray:
        depth = np.full(disp.shape, np.nan, dtype=np.float32)
        valid = disp > 0.5
        depth[valid] = self.fx * self.baseline / disp[valid]
        return depth

    def compute_disparity(self, left_rect: np.ndarray, right_rect: np.ndarray) -> np.ndarray:
        disp = self.sgbm.compute(left_rect, right_rect).astype(np.float32) / 16.0
        return disp

    def backproject_points(self, uv: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        H, W = depth.shape[:2]
        pts_3d = np.zeros((len(uv), 3), dtype=np.float64)
        valid = np.zeros(len(uv), dtype=bool)

        for i, (u, v) in enumerate(uv):
            x = int(round(float(u)))
            y = int(round(float(v)))
            if x < 0 or x >= W or y < 0 or y >= H:
                continue
            z = float(depth[y, x])
            if not np.isfinite(z) or z <= 0.05 or z > 50.0:
                continue
            X = (u - self.cx) * z / self.fx
            Y = (v - self.cy) * z / self.fy
            pts_3d[i] = [X, Y, z]
            valid[i] = True
        return pts_3d, valid


@dataclass
class SolverConfig:
    dataset_root: Path
    output_csv: Path
    max_features: int = 2500
    min_pnp_points: int = 40
    save_preview_dir: Optional[Path] = None
    preview_every: int = 25


class OfflineStereoImuReference:
    def __init__(self, cfg: SolverConfig):
        self.cfg = cfg
        self.seq = TumVieSequence(cfg.dataset_root)
        self.rig = self.seq.load_stereo_rig()
        self.imu_packets = self.seq.load_imu()
        self.frames = self.seq.load_frames()
        self.imu_integrator = ImuIntegrator(self.imu_packets)
        self.rectifier = StereoRectifier(self.rig)

        self.orb = cv2.ORB_create(
            nfeatures=cfg.max_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=19,
            patchSize=31,
            fastThreshold=15,
        )
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        if cfg.save_preview_dir is not None:
            cfg.save_preview_dir.mkdir(parents=True, exist_ok=True)

    def _detect(self, img: np.ndarray):
        return self.orb.detectAndCompute(img, None)

    def _match(self, des_prev: np.ndarray, des_cur: np.ndarray, ratio: float = 0.8):
        if des_prev is None or des_cur is None or len(des_prev) == 0 or len(des_cur) == 0:
            return []
        knn = self.bf.knnMatch(des_prev, des_cur, k=2)
        out = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                out.append(m)
        return out

    def run(self) -> List[Tuple[int, float, np.ndarray]]:
        world_R_imu0 = self.imu_integrator.estimate_initial_world_R_imu()
        T_world_left = compose(T_from_Rt(world_R_imu0, np.zeros(3)), self.rig.left.T_imu_cam)

        trajectories: List[Tuple[int, float, np.ndarray]] = []
        trajectories.append((0, self.frames[0].t, T_world_left.copy()))

        prev_left, prev_right = self.rectifier.read_and_rectify(self.frames[0].left_path, self.frames[0].right_path)
        prev_disp = self.rectifier.compute_disparity(prev_left, prev_right)
        prev_depth = self.rectifier.disparity_to_depth(prev_disp)
        prev_kp, prev_des = self._detect(prev_left)

        if prev_kp is None or len(prev_kp) < 80:
            raise RuntimeError("Too few features in first rectified frame")

        print(f"[INFO] dataset_root={self.cfg.dataset_root}")
        print(f"[INFO] frames={len(self.frames)} imu_packets={len(self.imu_packets)} baseline={self.rig.baseline_m:.5f}m")
        print(f"[INFO] rectified baseline={self.rectifier.baseline:.5f}m fx={self.rectifier.fx:.3f}")

        T_world_left_prev = T_world_left.copy()

        for i in range(1, len(self.frames)):
            cur_frame = self.frames[i]
            prev_frame = self.frames[i - 1]
            cur_left, cur_right = self.rectifier.read_and_rectify(cur_frame.left_path, cur_frame.right_path)
            cur_disp = self.rectifier.compute_disparity(cur_left, cur_right)
            cur_depth = self.rectifier.disparity_to_depth(cur_disp)
            cur_kp, cur_des = self._detect(cur_left)

            if cur_kp is None or cur_des is None or len(cur_kp) < 60:
                trajectories.append((cur_frame.index, cur_frame.t, T_world_left_prev.copy()))
                prev_left, prev_right, prev_disp, prev_depth, prev_kp, prev_des = cur_left, cur_right, cur_disp, cur_depth, cur_kp, cur_des
                continue

            matches = self._match(prev_des, cur_des)

            T_left_imu = invT(self.rig.left.T_imu_cam)
            dR_imu = self.imu_integrator.integrate_delta_rotation(prev_frame.t, cur_frame.t)
            dR_left_prior = T_left_imu[:3, :3] @ dR_imu @ self.rig.left.T_imu_cam[:3, :3]

            object_pts = []
            image_pts = []

            if prev_kp is not None:
                uv_prev = np.array([prev_kp[m.queryIdx].pt for m in matches], dtype=np.float64)
                uv_cur = np.array([cur_kp[m.trainIdx].pt for m in matches], dtype=np.float64)
                pts3_prev, valid = self.rectifier.backproject_points(uv_prev, prev_depth)
                for k, ok in enumerate(valid):
                    if not ok:
                        continue
                    object_pts.append(pts3_prev[k])
                    image_pts.append(uv_cur[k])

            object_pts = np.asarray(object_pts, dtype=np.float64)
            image_pts = np.asarray(image_pts, dtype=np.float64)

            used_method = "hold"
            T_prev_cur = np.eye(4, dtype=np.float64)

            if len(object_pts) >= self.cfg.min_pnp_points:
                K_rect = np.array([
                    [self.rectifier.fx, 0.0, self.rectifier.cx],
                    [0.0, self.rectifier.fy, self.rectifier.cy],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float64)
                rvec_guess = R_to_rodrigues(dR_left_prior)
                tvec_guess = np.zeros(3, dtype=np.float64)

                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    object_pts.reshape(-1, 1, 3),
                    image_pts.reshape(-1, 1, 2),
                    K_rect,
                    None,
                    rvec=rvec_guess.reshape(3, 1),
                    tvec=tvec_guess.reshape(3, 1),
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                    iterationsCount=150,
                    reprojectionError=2.5,
                    confidence=0.999,
                )

                if ok and inliers is not None and len(inliers) >= self.cfg.min_pnp_points:
                    R_prev_cur = rodrigues_to_R(rvec)
                    t_prev_cur = tvec.reshape(3)
                    T_prev_cur = T_from_Rt(R_prev_cur, t_prev_cur)
                    used_method = f"pnp({len(inliers)})"
                else:
                    T_prev_cur = T_from_Rt(dR_left_prior, np.zeros(3))
                    used_method = "gyro_fallback"
            else:
                T_prev_cur = T_from_Rt(dR_left_prior, np.zeros(3))
                used_method = "gyro_fallback"

            T_world_left_cur = T_world_left_prev @ T_prev_cur
            trajectories.append((cur_frame.index, cur_frame.t, T_world_left_cur.copy()))

            if i % 20 == 0 or i == len(self.frames) - 1:
                pos = T_world_left_cur[:3, 3]
                print(f"[INFO] frame={i:05d}/{len(self.frames)-1:05d} method={used_method:<16} pos=({pos[0]: .3f}, {pos[1]: .3f}, {pos[2]: .3f})")

            prev_left, prev_right, prev_disp, prev_depth, prev_kp, prev_des = cur_left, cur_right, cur_disp, cur_depth, cur_kp, cur_des
            T_world_left_prev = T_world_left_cur

        return trajectories


def write_reference_csv(path: Path, traj: Sequence[Tuple[int, float, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t", "x", "y", "z", "yaw_rad", "pitch_rad", "roll_rad"])
        t0 = traj[0][1] if traj else 0.0
        for frame_idx, t_abs, T in traj:
            t_rel = float(t_abs - t0)
            x, y, z = T[:3, 3]
            yaw, pitch, roll = euler_yaw_pitch_roll_from_R(T[:3, :3])
            w.writerow([
                int(frame_idx),
                f"{t_rel:.9f}",
                f"{x:.9f}",
                f"{y:.9f}",
                f"{z:.9f}",
                f"{yaw:.9f}",
                f"{pitch:.9f}",
                f"{roll:.9f}",
            ])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate offline stereo+IMU reference trajectory CSV")
    p.add_argument("--dataset-root", default="dataset/loop-floor0", help="Relative or absolute dataset root")
    p.add_argument("--output-csv", default=None, help="Output CSV path. Defaults to <dataset-root>/offline_reference_poses.csv")
    p.add_argument("--max-features", type=int, default=2500)
    p.add_argument("--min-pnp-points", type=int, default=40)
    p.add_argument("--save-preview-dir", default=None, help="Optional directory for preview images")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_csv = Path(args.output_csv) if args.output_csv else dataset_root / "offline_reference_poses.csv"
    save_preview_dir = Path(args.save_preview_dir) if args.save_preview_dir else None

    cfg = SolverConfig(
        dataset_root=dataset_root,
        output_csv=output_csv,
        max_features=int(args.max_features),
        min_pnp_points=int(args.min_pnp_points),
        save_preview_dir=save_preview_dir,
    )

    solver = OfflineStereoImuReference(cfg)
    traj = solver.run()
    write_reference_csv(cfg.output_csv, traj)
    print(f"[DONE] Wrote offline reference trajectory to: {cfg.output_csv}")
    print("[NOTE] This file is an offline reference trajectory, not official benchmark ground truth.")


if __name__ == "__main__":
    main()
