#!/usr/bin/env python3
"""Score eqvio_checks.sh / msceqf_checks.sh (EqVIO / MSCEqF native on the synthetic set, the phone recordings and with
dropped frames). Per run: the camera trajectory on the camera clock (EqVIO: from IMUState.csv + camera.csv, T_wc = T_wi T_ic
as eval/eqvio_to_tum.py, + the 0.088 s camera lag unless the ideal IMU -> traj_camclock.txt; MSCEqF: msceqf_run writes
traj_camclock.txt), then
  synthetic   eval/evaluate.py against results/gt/synth_<seq>.txt (the _k2 / _k3 runs: the full sequence's ground truth,
              matched at the frames they kept): SE3 / Sim3 ATE and scale
  real        eval/slides_metric.py on the 1 m slide recordings; the others: largest distance from the start, distance
              start -> end, path length (no ground truth)
  timing      timing.csv: EqVIO (--timing): median / p90 ms of GIFT feature tracking ('features') and of the filter
              ('total vision update' = preprocessing + propagation + correction); MSCEqF: the camera update (tracking +
              filter update, 'camera_ms') and the IMU propagation since the previous frame ('imu_ms')
Prints markdown tables. Run with the WSL venv (evo):
  /opt/slam/venv/bin/python eqvio_checks_score.py /mnt/h/Research/results/eqvio_checks_1030
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

R = Path(sys.argv[1])
EVAL = Path(__file__).resolve().parents[3] / "eval"
GT = Path("/mnt/h/Research/results/gt")
LAG = 0.088
DIVERGED = 1.0  # SE3 ATE (m) above which a run is marked diverged


def trajectory(o, ideal):
    if not (o / "IMUState.csv").exists():  # MSCEqF
        rows = [l.split() for l in (o / "traj_camclock.txt").read_text().splitlines() if l and l[0] != "#"]
        d = np.array([[float(x) for x in r] for r in rows if len(r) == 8], ndmin=2)  # a run cut by its timeout: last line
        d = d[np.isfinite(d).all(1)]
        return d[:, 0], d[:, 1:4]
    imu = np.loadtxt(o / "IMUState.csv", delimiter=",", skiprows=1, ndmin=2)
    cam = np.loadtxt(o / "camera.csv", delimiter=",", skiprows=1, ndmin=2)
    n = min(len(imu), len(cam))
    imu, cam = imu[:n], cam[:n]
    R_wi = Rotation.from_quat(imu[:, [5, 6, 7, 4]])
    R_ic = Rotation.from_quat(cam[:, [5, 6, 7, 4]])
    p = imu[:, 1:4] + R_wi.apply(cam[:, 1:4])
    q = (R_wi * R_ic).as_quat()
    t = imu[:, 0] + (0.0 if ideal else LAG)
    ok = np.isfinite(p).all(1) & np.isfinite(q).all(1) & (imu[:, 0] > 0)  # time -1: frames before the first IMU sample
    with open(o / "traj_camclock.txt", "w", newline="\n") as f:
        f.write("# t tx ty tz qx qy qz qw (camera, EqVIO, camera clock)\n")
        for k in np.where(ok)[0]:
            f.write(f"{t[k]:.9f} " + " ".join(f"{v:.9f}" for v in (*p[k], *q[k])) + "\n")
    return t[ok], p[ok]


def timing(o):
    f = o / "timing.csv"
    if not f.exists():
        return None
    try:
        d = np.genfromtxt(f, delimiter=",", names=True, invalid_raise=False)  # a crashed run: cut last line
    except (IndexError, ValueError):  # empty (crashed before the first write)
        return None
    if "camera_ms" in d.dtype.names:  # MSCEqF (ms; the frames after its initialisation)
        d = d[d["initialised"] > 0]
        return {k: (np.nanmedian(d[k]), np.nanpercentile(d[k], 90)) for k in ("camera_ms", "imu_ms")} if len(d) else None
    return {k: (np.nanmedian(d[k]) * 1e3, np.nanpercentile(d[k], 90) * 1e3) for k in ("features", "total_vision_update")}


rows, real, tim = {}, [], {}
for o in sorted(p for p in R.iterdir() if p.is_dir()):
    n = o.name
    f = o / "IMUState.csv" if (o / "IMUState.csv").exists() else o / "traj_camclock.txt"
    if not f.exists() or f.stat().st_size < 100:
        rows[n] = "no output"
        continue
    ideal = n.endswith("__imuideal")
    t, p = trajectory(o, ideal)
    tim[n] = timing(o)
    if n.startswith("rec_"):
        if "20261029_202938" in n or "20261021_070221" in n:
            out = subprocess.run([sys.executable, str(EVAL / "slides_metric.py"), str(o / "traj_camclock.txt")],
                                 capture_output=True, text=True, cwd=EVAL).stdout.strip()
            real.append((n, out.split(": ", 1)[-1]))
        else:
            far = np.linalg.norm(p - p[0], axis=1).max()
            real.append((n, f"{len(t)} poses, max distance from start {far:.2f} m, start -> end {np.linalg.norm(p[-1] - p[0]):.2f} m, "
                            f"path {np.linalg.norm(np.diff(p, axis=0), axis=1).sum():.1f} m"))
        continue
    seq = re.sub(r"_k\d$", "", n)
    subprocess.run([sys.executable, str(EVAL / "evaluate.py"), "--gt", str(GT / f"synth_{seq}.txt"), "--est",
                    str(o / "traj_camclock.txt"), "--out", str(o / "metrics.json")], capture_output=True, cwd=EVAL)
    if not (o / "metrics.json").exists():
        rows[n] = "eval failed"
        continue
    m = json.loads((o / "metrics.json").read_text())
    rows[n] = (m["ate_se3"]["rmse"], m["sim3_scale"], m["ate_sim3"]["rmse"], m["matched_poses"])


def cell(n):
    r = rows.get(n)
    if r is None:
        return "-"
    if isinstance(r, str):
        return r
    se3, s, sim3, _ = r
    return ("**diverged** " if se3 > DIVERGED else "") + f"{100 * se3:.1f} cm (s {s:.2f})"


print(f"### Synthetic: SE3 ATE (Sim3 scale); diverged = SE3 ATE > {DIVERGED} m\n")
print("| Scenario | IMU | v0 | v1 | v2 | v3 | v4 | v5 | v0 clean | diverged | median SE3 ATE of the rest |\n|---|---|---|---|---|---|---|---|---|---|---|")
for sc in ("ar", "gesture", "orbit", "rotate", "shake", "slide", "walk"):
    for imu, suf in (("phone", ""), ("ideal", "__imuideal")):
        names = [f"{sc}{v}{suf}" for v in range(6)]
        vals = [rows.get(x) for x in names]
        ok = [v[0] for v in vals if isinstance(v, tuple) and v[0] <= DIVERGED]
        div = sum(1 for v in vals if isinstance(v, tuple) and v[0] > DIVERGED)
        clean = cell(f"{sc}0_clean") if suf == "" else "-"
        print(f"| {sc} | {imu} | " + " | ".join(cell(x) for x in names) + f" | {clean} | {div}/6 | "
              + (f"{100 * np.median(ok):.1f} cm" if ok else "-") + " |")
print("\n### Dropped frames (phone IMU): every frame / every 2nd (15 fps) / every 3rd (10 fps)\n")
print("| Sequence | 30 fps | 15 fps | 10 fps |\n|---|---|---|---|")
for s in ("ar2", "gesture2", "orbit2", "rotate2", "shake2", "slide3", "walk2"):
    print(f"| {s} | {cell(s)} | {cell(s + '_k2')} | {cell(s + '_k3')} |")
print("\n### Phone recordings (no ground truth)\n")
for n, line in real:
    print(f"- {n}: {line}")
print("\n### Cost (native, runs in parallel; ms per frame: range of the run medians, their median, the largest p90)\n")
labels = {"features": "feature tracking (GIFT)", "total_vision_update": "filter (propagation + update)",
          "camera_ms": "camera update (tracking + filter update)", "imu_ms": "IMU propagation per frame"}
for k, lab in labels.items():
    m = np.array([v[k][0] for v in tim.values() if v and k in v])
    m9 = np.array([v[k][1] for v in tim.values() if v and k in v])
    if len(m):
        print(f"- {lab}: {np.min(m):.2f}-{np.max(m):.2f}, median {np.median(m):.2f}, p90 up to {np.max(m9):.2f}")
