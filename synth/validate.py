#!/usr/bin/env python3
"""Validation ladder of a synthetic sequence (synth/README.md "Validation"):
 (a) ground-truth camera trajectory + the ideal IMU (variants/imu_ideal.csv) -> eval/scale_from_accel.py and
     eval/scale_linear_align.py must give s = 1 (metric) and |g| = 9.81: conventions (axes, R_bc, specific force sign,
     time base) are right;
 (b) ground truth + the Chrome-like IMU (mav0/imu0/data.csv) -> eval/estimate_time_offset.py must recover the injected
     offset (meta.json synthetic.imu.offset, -88 ms), and the scale estimators must stay near 1 with --offset-ms;
 (c) the 1 m calibration board: detected corners (cv2.findChessboardCorners) vs the ground-truth projection, in frames
     where the whole board is in view and the camera is nearly still (no rolling shutter / blur effect).
Writes <seq>/validation.json and prints one summary line.  validate.py <seq dir> [...]
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

EVAL = Path(__file__).resolve().parent.parent / "eval"
PY = sys.executable


def run(args):
    r = subprocess.run([PY] + [str(a) for a in args], capture_output=True, text=True)
    return (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else ""


def grab(pat, s, cast=float):
    m = re.search(pat, s)
    return cast(m.group(1)) if m else None


def frame_rate_gt(seq, tmp):
    """camera ground truth at ~30 Hz on the true clock (TUM) from gt/camera_dense.csv"""
    d = np.loadtxt(seq / "gt" / "camera_dense.csv", comments="#")
    d = d[::7]  # 200 Hz -> ~28.6 Hz
    np.savetxt(tmp, d, fmt="%.9f")
    return tmp


def board_check(seq, meta, scene):
    from scipy.spatial.transform import Rotation
    K = meta["intrinsics"]
    b = scene["known"]["board_1m"]
    c0 = np.array(b["corner"]); sq = b["square_m"]
    n = int(round(1.0 / sq)) - 3  # inner corners per side (one-square white border): 7
    grid = np.array([[c0[0] + (i + 2) * sq, c0[1] + (j + 2) * sq, c0[2]] for j in range(n) for i in range(n)])
    cam = [l.split(",") for l in (seq / "variants" / "cam_exact.csv").read_text().splitlines() if l and l[0] != "#"]
    dense = np.loadtxt(seq / "gt" / "camera_dense.csv", comments="#")
    errs, used = [], 0
    for k in range(0, len(cam), 5):
        t = int(cam[k][0]) * 1e-9
        j = np.argmin(np.abs(dense[:, 0] - t))
        if abs(dense[j, 0] - t) > 0.003:
            continue
        if j > 0 and j + 1 < len(dense):  # nearly still: < 2 mm and < 0.1 deg per 5 ms
            dp = np.linalg.norm(dense[j + 1, 1:4] - dense[j - 1, 1:4])
            dq = Rotation.from_quat(dense[j - 1, 4:8]).inv() * Rotation.from_quat(dense[j + 1, 4:8])
            if dp > 0.002 or dq.magnitude() > np.radians(0.1):
                continue
        R = Rotation.from_quat(dense[j, 4:8]).as_matrix(); o = dense[j, 1:4]
        pc = (grid - o) @ R
        if (pc[:, 2] < 0.2).any():
            continue
        uv = np.stack([K["fx"] * pc[:, 0] / pc[:, 2] + K["cx"], K["fy"] * pc[:, 1] / pc[:, 2] + K["cy"]], 1)
        if (uv < 20).any() or (uv[:, 0] > 460).any() or (uv[:, 1] > 620).any():
            continue
        img = cv2.imread(str(seq / "mav0" / "cam0" / "data" / cam[k][1]), cv2.IMREAD_GRAYSCALE)
        ok, cor = cv2.findChessboardCorners(img, (n, n), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            continue
        cor = cv2.cornerSubPix(img, cor, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01))[:, 0]
        d = np.linalg.norm(cor[:, None, :] - uv[None], axis=2).min(1)  # nearest projected corner (ordering-free)
        errs.append(d); used += 1
        if used >= 8:
            break
    if not errs:
        return {"frames": 0}
    e = np.concatenate(errs)
    return {"frames": used, "corners": int(len(e)), "mean_px": round(float(e.mean()), 3), "max_px": round(float(e.max()), 3)}


def main():
    for arg in sys.argv[1:]:
        seq = Path(arg)
        meta = json.loads((seq / "meta.json").read_text())
        scene = json.loads((seq / "scene.json").read_text())
        off_ms = meta["synthetic"]["imu"]["offset"] * 1e3
        tmp = seq / "gt" / "camera_30hz.txt"
        gt = frame_rate_gt(seq, tmp)
        st = seq / "settings.yaml"
        out = {}
        # (a) ideal IMU
        l = run([EVAL / "scale_from_accel.py", "--traj", gt, "--imu", seq / "variants" / "imu_ideal.csv", "--tbc", st, "--window", "1", "--s-true", "1"])
        out["a_window"] = {"ratio": grab(r"ratio ([\d.]+)", l), "g": grab(r"\|g\| linear ([\d.]+)", l), "line": l}
        l = run([EVAL / "scale_linear_align.py", "--traj", gt, "--imu", seq / "variants" / "imu_ideal.csv", "--tbc", st, "--dt", "0.5", "--s-true", "1"])
        out["a_align"] = {"ratio": grab(r"ratio ([\d.]+)", l), "g": grab(r"\|g\| linear ([\d.]+)", l), "line": l}
        # (b) Chrome-like IMU
        l = run([EVAL / "estimate_time_offset.py", "--traj", gt, "--imu", seq / "mav0" / "imu0" / "data.csv", "--tbc", "phone"])
        out["b_offset"] = {"ms": grab(r"offset ([+-]?\d+) ms", l, int), "injected_ms": off_ms, "line": l}
        l = run([EVAL / "scale_from_accel.py", "--traj", gt, "--imu", seq / "mav0" / "imu0" / "data.csv", "--tbc", st, "--window", "1", "--s-true", "1", "--offset-ms", off_ms])
        out["b_window"] = {"ratio": grab(r"ratio ([\d.]+)", l), "g": grab(r"\|g\| linear ([\d.]+)", l), "line": l}
        l = run([EVAL / "scale_linear_align.py", "--traj", gt, "--imu", seq / "mav0" / "imu0" / "data.csv", "--tbc", st, "--dt", "0.5", "--s-true", "1", "--offset-ms", off_ms, "--bias-prior", "0.1"])
        out["b_align"] = {"ratio": grab(r"ratio ([\d.]+)", l), "g": grab(r"\|g\| linear ([\d.]+)", l), "line": l}
        # (c) calibration board
        out["c_board"] = board_check(seq, meta, scene)
        (seq / "validation.json").write_text(json.dumps(out, indent=1))
        f = lambda d, k: d.get(k) if d.get(k) is not None else float("nan")
        print(f"{seq.name}: (a) ideal IMU scale {f(out['a_window'], 'ratio')} / {f(out['a_align'], 'ratio')} |g| {f(out['a_window'], 'g')} / {f(out['a_align'], 'g')}"
              f" | (b) offset {out['b_offset']['ms']} ms (injected {off_ms:.0f}), Chrome IMU scale {f(out['b_window'], 'ratio')} / {f(out['b_align'], 'ratio')}"
              f" | (c) board {out['c_board']}", flush=True)


if __name__ == "__main__":
    main()
