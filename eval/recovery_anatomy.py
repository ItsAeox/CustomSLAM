#!/usr/bin/env python3
"""What limits recovery after a tracking loss: motion blur, or not knowing where the camera is? (PLAN 4d b1)

Per loss in an orbslam3_run --trace run (trace.txt: frame, state, tracked points, Twc t xyz, q xyzw, ...; state 2 / 5 =
tracking), on the recording's own frames (all of them, also the ones the live replay dropped) and gyro:
  sharpness  variance of the Laplacian of each frame (half resolution), relative to the median of the tracked frames
             0.3-2 s before the loss; "sharp again" = the first frame after the loss at >= SHARP of that, for 3 frames
  lag        time from sharp again to tracking again (only losses that end in the same map)
  turned     gyro rotation from the last tracked frame to sharp again, and to the recovery (angle of the integrated
             rotation, degrees); gyro rate at sharp again (still moving?)
  visual     rotation and camera-centre distance between the last tracked pose and the recovered pose (same map)
Small lags = recovery waits for sharp frames (blur-limited); large lags with sharp frames = the tracker does not find
the map although the image is fine (prediction / relocalisation / unmapped view).

  recovery_anatomy.py <run dir> <recording dir> [--sharp 0.6] [--imudt -88]   (one line per loss + a summary line)
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

args = [a for a in sys.argv[1:] if not a.startswith("--")]
SHARP = float(sys.argv[sys.argv.index("--sharp") + 1]) if "--sharp" in sys.argv else 0.6
run, rec = Path(args[0]), Path(args[1])
tr = np.array([[float(x) for x in l.split()] for l in (run / "trace.txt").read_text().splitlines() if len(l.split()) >= 20])
cam = np.loadtxt(rec / "mav0" / "cam0" / "data.csv", delimiter=",", comments="#", dtype=str)
t_cam = (cam[:, 0].astype(np.int64) - int(cam[0, 0])) * 1e-9
files = [rec / "mav0" / "cam0" / "data" / f.strip() for f in cam[:, 1]]
imu = np.loadtxt(rec / "mav0" / "imu0" / "data.csv", delimiter=",", comments="#")
IMUDT = float(sys.argv[sys.argv.index("--imudt") + 1]) if "--imudt" in sys.argv else -88.0  # ms, as the AR page's imudt
t_imu = (imu[:, 0] - int(cam[0, 0])) * 1e-9 - IMUDT * 1e-3  # IMU time = recorded - imudt
gyro = imu[:, 1:4]
fr = tr[:, 0].astype(int)
st = tr[:, 1].astype(int)
ok = np.isin(st, [2, 5])
kfs = tr[:, 13]
pos = tr[:, 3:6]
quat = tr[:, 6:10]  # x y z w (Twc)

sharp_cache = {}
def sharpness(k):
    if k not in sharp_cache:
        im = Image.open(files[k]).convert("L")
        a = np.asarray(im.resize((im.width // 2, im.height // 2), Image.BOX), dtype=np.float32)
        lap = a[1:-1, :-2] + a[1:-1, 2:] + a[:-2, 1:-1] + a[2:, 1:-1] - 4 * a[1:-1, 1:-1]  # 4-neighbour Laplacian
        sharp_cache[k] = float(lap.var())
    return sharp_cache[k]

def gyro_angle(t0, t1):  # angle of the rotation integrated from t0 to t1 (body frame), degrees
    m = (t_imu >= t0) & (t_imu < t1)
    if m.sum() < 2:
        return 0.0
    ts, ws = t_imu[m], gyro[m]
    R = np.eye(3)
    for i in range(len(ts) - 1):
        w = ws[i] * (ts[i + 1] - ts[i])
        th = np.linalg.norm(w)
        if th > 1e-12:
            K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]) / th
            R = R @ (np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K)
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))

def gyro_rate(t):
    k = int(np.clip(np.searchsorted(t_imu, t), 0, len(t_imu) - 1))
    return float(np.degrees(np.linalg.norm(gyro[k])))

def qangle(qa, qb):
    d = abs(float(np.dot(qa / np.linalg.norm(qa), qb / np.linalg.norm(qb))))
    return float(np.degrees(2 * np.arccos(min(1.0, d))))

losses = [i for i in range(1, len(ok)) if ok[i - 1] and not ok[i]]
rows = []
for i in losses:
    j = next((k for k in range(i, len(ok)) if ok[k]), None)
    last = i - 1
    f_last, t_last = fr[last], t_cam[min(fr[last], len(t_cam) - 1)]
    same_map = j is not None and np.all(kfs[i:j + 1] > 0)
    # sharpness baseline: tracked frames 0.3-2 s before the loss
    base = [k for k in range(len(t_cam)) if t_last - 2.0 <= t_cam[k] <= t_last - 0.3]
    b = np.median([sharpness(k) for k in base]) if base else np.nan
    end_f = fr[j] if j is not None else len(t_cam) - 1
    sharp_again = None
    run3 = 0
    for k in range(f_last + 1, min(end_f + 1, len(t_cam))):
        run3 = run3 + 1 if sharpness(k) >= SHARP * b else 0
        if run3 == 3:
            sharp_again = k - 2
            break
    t_sharp = t_cam[sharp_again] if sharp_again is not None else None
    t_rec = t_cam[min(fr[j], len(t_cam) - 1)] if j is not None else None
    dip = min(sharpness(k) for k in range(f_last, min(end_f + 1, len(t_cam)))) / b if b == b else np.nan
    r = {"t": t_last, "dur": (t_rec - t_last) if t_rec is not None else t_cam[-1] - t_last, "end": "same map" if same_map else
         ("new map / reset" if j is not None else "end of recording"), "dip": dip,
         "to_sharp": (t_sharp - t_last) if t_sharp is not None else None,
         "lag": (t_rec - t_sharp) if (t_sharp is not None and t_rec is not None and same_map) else None,
         "turn_sharp": gyro_angle(t_last, t_sharp) if t_sharp is not None else None,
         "rate_sharp": gyro_rate(t_sharp) if t_sharp is not None else None,
         "turn_rec": gyro_angle(t_last, t_rec) if (t_rec is not None and same_map) else None,
         "vis_rot": qangle(quat[last], quat[j]) if same_map else None,
         "vis_dist": float(np.linalg.norm(pos[j] - pos[last])) if same_map else None}
    rows.append(r)
    f = lambda v, fmt: (fmt % v) if v is not None else "-"
    print(f"  loss t {r['t']:.1f} s, lost {r['dur']:.1f} s ({r['end']}): sharpness dip x{r['dip']:.2f}, sharp again after "
          f"{f(r['to_sharp'], '%.2f')} s (turned {f(r['turn_sharp'], '%.0f')} deg, still {f(r['rate_sharp'], '%.0f')} deg/s), "
          f"lag to recovery {f(r['lag'], '%.2f')} s; at recovery turned {f(r['turn_rec'], '%.0f')} deg by gyro / "
          f"{f(r['vis_rot'], '%.0f')} deg visually, moved {f(r['vis_dist'], '%.3f')} map units")
lags = [r["lag"] for r in rows if r["lag"] is not None]
never = sum(1 for r in rows if r["to_sharp"] is None)
print(f"SUMMARY {run}: {len(rows)} losses, {sum(r['end'] == 'same map' for r in rows)} recovered in the same map; "
      f"never sharp again before the end of the loss {never}; lag sharp -> recovered (same map) "
      + (f"median {np.median(lags):.2f} s, under 0.3 s {sum(l < 0.3 for l in lags)}/{len(lags)}, over 1 s {sum(l > 1 for l in lags)}/{len(lags)}" if lags else "n/a"))
