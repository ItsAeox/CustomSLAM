#!/usr/bin/env python3
"""Build the synthetic WebVIO dataset (synth/README.md): rooms (scene.py) x motions (traj.py) rendered (render.py) with
the phone IMU model (imu.py), written in the phone recording layout (record.html) so every tool runs unchanged:

  <out>/<name>/mav0/cam0/data/frame_<i>.jpg, mav0/cam0/data.csv (reported = jittered stamps), mav0/imu0/data.csv
  (DeviceMotion-like: Chrome tick stamps, rounded), accel_gs.csv / gyro_gs.csv (sensor stamps), touches.csv,
  settings.yaml, meta.json (every parameter; assumptions flagged), scene.json (geometry, known-size objects),
  mav0/state_groundtruth_estimate0/data.csv (EuRoC body ground truth at 200 Hz: p, q wxyz, v, bw, ba),
  gt/camera_dense.csv (camera poses at 200 Hz, true clock, TUM order), variants/ (no re-render needed):
    imu_ideal.csv (true signal at the 100 Hz sample times, exact stamps), imu_chrome_sensorstamps.csv (DeviceMotion
    values with the sensor stamps), cam_exact.csv (frame stamps without jitter)
  <gt dir>/synth_<name>.txt: camera ground truth (TUM, Twc) keyed by the reported frame stamps, pose at the true
  capture time (no speed x jitter error from timestamp association).

  make_dataset.py [--out H:/Research/datasets/synth] [--gt H:/Research/results/gt] [--only slide0,ar3] [--procs 8]
                  [--clean]   (also the clean render of variant 0 of every scenario: no blur, rolling shutter, noise)
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):  # one thread per render process
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from synth import imu as imu_mod  # noqa: E402
from synth.geom import R_BC, quat_from_R  # noqa: E402
from synth.render import Renderer  # noqa: E402
from synth.scene import make_room  # noqa: E402
from synth.traj import SCENARIOS, check_motion  # noqa: E402

W, H = 480, 640
FX = FY = 465.60288917518244  # = the phone's assumed 69 deg horizontal FOV (here exact)
CX, CY = 240.0, 320.0
FPS = 29.97                    # measured capture period 33.37 ms
CAM_JITTER = 0.0037            # measured stamp jitter (s, 1-sigma)
BASE_NS = 1_800_000_000_000_000_000
ROOMS = ["living", "bedroom", "office"]


def specs():
    out = []
    for si, sc in enumerate(SCENARIOS):
        for k in range(6):
            out.append({"name": f"{sc}{k}", "scenario": sc, "variant": k, "room": "flat" if sc == "flat" else ROOMS[(si + k) % 3], "room_seed": 1,
                        "seed": 1000 * si + k, "index": len(out)})
    return out


def ns(t, base):
    return base + np.round(np.asarray(t) * 1e9).astype(np.int64)


def write_csv(path, header, rows, fmt):
    with open(path, "w", newline="\n") as fh:
        fh.write(header + "\n")
        for r in rows:
            fh.write(fmt(r) + "\n")


def settings_yaml(fps):
    return f"""%YAML:1.0
File.version: "1.0"
Camera.type: "PinHole"
Camera1.fx: {FX:.6f}
Camera1.fy: {FY:.6f}
Camera1.cx: {CX:.6f}
Camera1.cy: {CY:.6f}
Camera1.k1: 0.000000
Camera1.k2: 0.000000
Camera1.p1: 0.000000
Camera1.p2: 0.000000
Camera.width: {W}
Camera.height: {H}
Camera.fps: {int(round(fps))}
Camera.RGB: 1
ORBextractor.nFeatures: 1000
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0

# IMU (phone, DeviceMotion): web/slamclient.js phoneImuSettings (synthetic: exact camera-IMU rotation, no lever arm)
IMU.T_b_c1: !!opencv-matrix
   rows: 4
   cols: 4
   dt: f
   data: [1.0, 0.0, 0.0, 0.0,
          0.0, -1.0, 0.0, 0.0,
          0.0, 0.0, -1.0, 0.0,
          0.0, 0.0, 0.0, 1.0]
IMU.InsertKFsWhenLost: 0
IMU.NoiseGyro: 6.0e-4
IMU.NoiseAcc: 1.0e-2
IMU.GyroWalk: 2.0e-5
IMU.AccWalk: 3.0e-3
IMU.Frequency: 60.0
"""


def git_rev():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip()
    except Exception:
        return "?"


def build(spec, out_root, gt_dir, clean=False, readout=0.02, exposure=0.012):
    t_start = time.time()
    name = spec["name"] + ("_clean" if clean else "")
    room = make_room(spec["room"], spec["room_seed"])
    fn = SCENARIOS[spec["scenario"]]
    for attempt in range(30):
        rng = np.random.default_rng(spec["seed"] + 100000 * attempt)
        m = fn(room, rng, spec["variant"])
        bad = check_motion(room, m, table_ok=spec["scenario"] == "slide")
        if not bad:
            break
    else:
        raise RuntimeError(f"{name}: no collision-free motion ({bad})")
    rng = np.random.default_rng(spec["seed"] * 7 + 13)  # sensors / stamps (independent of the motion draw)
    seq = Path(out_root) / name
    (seq / "mav0" / "cam0" / "data").mkdir(parents=True, exist_ok=True)
    (seq / "mav0" / "imu0").mkdir(parents=True, exist_ok=True)
    (seq / "mav0" / "state_groundtruth_estimate0").mkdir(parents=True, exist_ok=True)
    (seq / "gt").mkdir(exist_ok=True)
    (seq / "variants").mkdir(exist_ok=True)
    base = BASE_NS + spec["index"] * 10 ** 12

    # ---- camera: true capture times, reported (jittered) stamps, rendering
    T = 1.0 / FPS
    margin = max(readout, exposure) + 0.05
    t_cam = np.arange(0.2, m.duration - margin, T)
    jit = np.clip(rng.normal(0, CAM_JITTER, len(t_cam)), -0.01, 0.01)
    t_rep = t_cam + jit
    order_ok = np.all(np.diff(t_rep) > 0.005)
    if not order_ok:  # keep stamps increasing
        t_rep = np.maximum.accumulate(t_rep + np.arange(len(t_rep)) * 1e-9)
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
    r = Renderer(room, K, W, H, readout=0.0 if clean else readout, exposure=0.0 if clean else exposure,
                 noise=not clean, seed=spec["seed"] + 5)
    samples = []
    for i, t in enumerate(t_cam):
        img, S = r.frame(m.pose, t)
        samples.append(S)
        cv2.imwrite(str(seq / "mav0" / "cam0" / "data" / f"frame_{i}.jpg"), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    write_csv(seq / "mav0" / "cam0" / "data.csv", "#timestamp [ns],filename",
              [(n_, i) for i, n_ in enumerate(ns(t_rep, base))], lambda r_: f"{r_[0]},frame_{r_[1]}.jpg")
    write_csv(seq / "variants" / "cam_exact.csv", "#timestamp [ns],filename",
              [(n_, i) for i, n_ in enumerate(ns(t_cam, base))], lambda r_: f"{r_[0]},frame_{r_[1]}.jpg")

    # ---- IMU
    P = imu_mod.SensorParams()
    truth = imu_mod.body_truth(m)
    s = imu_mod.sensor(truth, rng, P, 0.0, m.duration)
    ch = imu_mod.chrome(s, rng, P)
    idl = imu_mod.ideal(truth, s)
    hdr = ("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
           "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]")
    fmt6 = lambda r_: f"{r_[0]},{r_[1]:.6f},{r_[2]:.6f},{r_[3]:.6f},{r_[4]:.6f},{r_[5]:.6f},{r_[6]:.6f}"
    write_csv(seq / "mav0" / "imu0" / "data.csv", hdr, zip(ns(ch["t_event"], base), *ch["w"].T, *ch["f"].T), fmt6)
    write_csv(seq / "variants" / "imu_chrome_sensorstamps.csv", hdr, zip(ns(ch["t_sensor"], base), *ch["w"].T, *ch["f"].T), fmt6)
    write_csv(seq / "variants" / "imu_ideal.csv", hdr, zip(ns(idl["t"], base), *idl["w"].T, *idl["f"].T), fmt6)
    fmt3 = lambda r_: f"{r_[0]},{r_[1]:.6f},{r_[2]:.6f},{r_[3]:.6f}"
    write_csv(seq / "accel_gs.csv", "#timestamp [ns],x,y,z", zip(ns(ch["t_sensor"], base), *ch["f"].T), fmt3)
    write_csv(seq / "gyro_gs.csv", "#timestamp [ns],x,y,z", zip(ns(ch["t_sensor"], base), *ch["w"].T), fmt3)

    # ---- ground truth
    td = np.arange(m.t[0], m.t[-1], 0.005)
    Rd, pd = m.pose(td)
    qd = quat_from_R(Rd)
    write_csv(seq / "gt" / "camera_dense.csv", "#t [s, true clock = camera clock] tx ty tz qx qy qz qw (Twc, camera)",
              zip(td + base * 1e-9, *pd.T, *qd.T), lambda r_: " ".join(f"{x:.9f}" if i == 0 else f"{x:.7f}" for i, x in enumerate(r_)))
    Rc, pc = m.pose(t_cam)
    qc = quat_from_R(Rc)
    Path(gt_dir).mkdir(parents=True, exist_ok=True)
    write_csv(Path(gt_dir) / f"synth_{name}.txt", "# synthetic ground truth: t = reported frame stamp (s), pose at the true capture time (Twc, camera)",
              zip(ns(t_rep, base) * 1e-9, *pc.T, *qc.T), lambda r_: " ".join(f"{x:.9f}" if i == 0 else f"{x:.7f}" for i, x in enumerate(r_)))
    tb = np.arange(m.t[0], m.t[-1], 0.005)
    ib = np.clip(np.round(tb * 1000).astype(int), 0, len(m.t) - 1)
    qb = quat_from_R(truth["R_wb"][ib])
    bg = np.stack([np.interp(tb, s["t"], s["bg"][:, k]) for k in range(3)], 1)
    ba = np.stack([np.interp(tb, s["t"], s["ba"][:, k]) for k in range(3)], 1)
    write_csv(seq / "mav0" / "state_groundtruth_estimate0" / "data.csv",
              "#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], "
              "v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1], b_w_RS_S_x [rad s^-1], b_w_RS_S_y [rad s^-1], "
              "b_w_RS_S_z [rad s^-1], b_a_RS_S_x [m s^-2], b_a_RS_S_y [m s^-2], b_a_RS_S_z [m s^-2]",
              zip(ns(tb, base), *truth["p"][ib].T, qb[:, 3], qb[:, 0], qb[:, 1], qb[:, 2], *truth["v"][ib].T, *bg.T, *ba.T),
              lambda r_: ",".join([str(r_[0])] + [f"{x:.7f}" for x in r_[1:]]))

    # ---- taps (AR scenario: the floor spot during the "place" phase), settings, metadata, scene
    taps = []
    if spec["scenario"] in ("ar", "flat"):
        ph = m.info["phases_s"]["place"]
        tt = 0.5 * (ph[0] + ph[1])
        Rt, pt = m.pose([tt])
        pcam = Rt[0].T @ (np.array(m.info["spot"]) - pt[0])
        if pcam[2] > 0.1:
            u, v = FX * pcam[0] / pcam[2] + CX, FY * pcam[1] / pcam[2] + CY
            if 0 <= u < W and 0 <= v < H:
                taps.append((ns([tt], base)[0], u, v))
    write_csv(seq / "touches.csv", "#timestamp [ns],u [px],v [px]", taps, lambda r_: f"{r_[0]},{r_[1]:.1f},{r_[2]:.1f}")
    (seq / "settings.yaml").write_text(settings_yaml(FPS))
    meta = {
        "name": name, "notes": f"synthetic: {spec['scenario']} variant {spec['variant']} in the {spec['room']} room",
        "recorded": "synthetic", "userAgent": "synth/make_dataset.py", "image": {"width": W, "height": H},
        "fps": FPS, "frames": len(t_cam), "droppedByRecorder": 0, "jpegQuality": 0.92,
        "intrinsics": {"fx": FX, "fy": FY, "cx": CX, "cy": CY, "k1": 0, "k2": 0, "p1": 0, "p2": 0},
        "hfovAssumed": 69, "calibrated": True,
        "imu": {"source": "synthetic DeviceMotion (synth/imu.py)", "rotationRateOrder": "chrome", "samples": len(ch["t_event"]),
                "hz": P.tick_hz, "genericSensor": {"gyroscope": len(ch["t_sensor"]), "accelerometer": len(ch["t_sensor"])}},
        "cameraToImu": "IMU.T_b_c1 = diag(1, -1, -1) (rear camera, portrait), no lever arm",
        "taps": len(taps), "durationS": round(float(t_cam[-1] - t_cam[0]), 2),
        "synthetic": {
            "generator": f"CustomSLAM/synth (git {git_rev()})", "scenario": spec["scenario"], "variant": spec["variant"],
            "room": spec["room"], "room_seed": spec["room_seed"], "seed": spec["seed"], "motion_attempt": attempt,
            "motion": m.info, "clean": clean,
            "camera": {"fps": FPS, "stamp_jitter_s": CAM_JITTER, "readout_s": 0.0 if clean else readout,
                       "exposure_s": 0.0 if clean else exposure, "noise": not clean, "jpeg_quality": 92,
                       "assumed": ["readout_s", "exposure_s", "noise"],
                       "time_convention": "frame time = middle row, middle of the exposure; GT pose at that time"},
            "imu": P.as_dict(), "imu_true_bias": {"acc": s["ba"][0].tolist(), "gyr": s["bg"][0].tolist()},
            "imu_acc_scale_factor": s["acc_scale"].tolist(),
            "event_minus_sample_ms_p5_p50_p95": ch["event_minus_sample_ms"],
            "blur_samples_mean": float(np.mean(samples)), "render_s": round(time.time() - t_start, 1),
            "world": "z up, gravity (0, 0, -9.81), metres, origin at a floor corner of the room",
        },
    }
    (seq / "meta.json").write_text(json.dumps(meta, indent=1))
    (seq / "scene.json").write_text(json.dumps({
        "room": room.name, "size_m": room.size, "known": room.known, "targets": {k: v.tolist() for k, v in room.targets.items()},
        "boxes": room.boxes, "lights": room.lights.tolist(),
        "faces": [{"name": f.name, "corner": f.c.tolist(), "u": f.u.tolist(), "v": f.v.tolist(), "tex_px": list(f.tex.shape)} for f in room.faces],
    }, indent=1))
    return name, len(t_cam), round(time.time() - t_start, 1), float(np.mean(samples))


def _job(a):
    spec, out, gt, clean = a
    try:
        return build(spec, out, gt, clean)
    except Exception as e:  # report and continue with the others
        return spec["name"], -1, 0, repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="H:/Research/datasets/synth")
    ap.add_argument("--gt", default="H:/Research/results/gt")
    ap.add_argument("--only", default="")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--clean", action="store_true", help="also the clean render of variant 0 of every scenario")
    ap.add_argument("--clean-only", action="store_true")
    a = ap.parse_args()
    sp = specs()
    if a.only:
        keep = set(a.only.split(","))
        sp = [s for s in sp if s["name"] in keep]
    jobs = [] if a.clean_only else [(s, a.out, a.gt, False) for s in sp]
    if a.clean or a.clean_only:
        jobs += [(s, a.out, a.gt, True) for s in sp if s["variant"] == 0]
    jobs.sort(key=lambda j: -{"slide": 60, "rotate": 35, "ar": 40, "bedside": 80, "rapid": 70, "longar": 150, "flat": 60}.get(j[0]["scenario"], 30))  # long ones first
    t0 = time.time()
    with Pool(a.procs) as pool:
        for name, n, sec, info in pool.imap_unordered(_job, jobs):
            print(f"{name}: {n} frames, {sec} s, blur samples {info}  [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
