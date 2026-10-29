#!/usr/bin/env python3
"""Statistics of real phone recordings (record.html, Pixel 9 Chrome) for the synthetic IMU / camera model
(synth/README.md "Sensor model"): DeviceMotion event timing, rounding, repeated samples, still-phase noise and bias,
the Generic Sensor streams (sensor timestamps) against DeviceMotion (latency, sample-and-hold), camera frame timing.

  imu_stats.py <recording dir> [...]   (datasets/phone/rec_*)
"""
import sys
from pathlib import Path

import numpy as np


def load_csv(p, cols=None):
    return np.loadtxt(p, delimiter=",", comments="#", ndmin=2, usecols=cols)


def pct(x, q=(5, 50, 95)):
    return " / ".join(f"{np.percentile(x, k):.2f}" for k in q)


def still_mask(t, w, f, win=0.5, gmax=0.03, amax=0.25):
    """samples inside a still phase: over +-win/2 the gyro stays below gmax rad/s and the accelerometer within amax"""
    m = np.zeros(len(t), bool)
    for i in range(len(t)):
        s = (t >= t[i] - win / 2) & (t <= t[i] + win / 2)
        m[i] = np.abs(w[s]).max() < gmax and np.ptp(f[s], axis=0).max() <= amax + 1e-9
    return m


def main():
    for rec in sys.argv[1:]:
        rec = Path(rec)
        d = load_csv(rec / "mav0/imu0/data.csv")
        t = (d[:, 0] - d[0, 0]) * 1e-9
        w, f = d[:, 1:4], d[:, 4:7]
        dt = np.diff(t) * 1e3
        print(f"== {rec.name}: {t[-1]:.1f} s")
        print(f"  DeviceMotion: {len(t)} events, {len(t) / t[-1]:.2f} Hz; interval ms p5/p50/p95 {pct(dt)}, min {dt.min():.2f} max {dt.max():.1f}; "
              f"> 25 ms: {np.mean(dt > 25) * 100:.2f} %")
        for name, v, step in (("accel", f, 0.1), ("gyro", w, np.radians(0.1))):
            q = v / step
            print(f"  {name}: off-grid samples (|v/step - round| > 0.01): {np.mean(np.abs(q - np.round(q)) > 0.01) * 100:.2f} %")
        same = np.all(np.diff(d[:, 1:7], axis=0) == 0, axis=1)
        same_w = np.all(np.diff(w, axis=0) == 0, axis=1)
        same_f = np.all(np.diff(f, axis=0) == 0, axis=1)
        mov = np.abs(w[1:]).max(1) > 0.3  # clearly rotating
        print(f"  consecutive identical events: all 6 values {same.mean() * 100:.1f} %, gyro {same_w.mean() * 100:.1f} %, accel {same_f.mean() * 100:.1f} %; "
              f"gyro identical while rotating > 0.3 rad/s: {same_w[mov].mean() * 100 if mov.any() else float('nan'):.1f} % of {mov.sum()}")
        st = still_mask(t, w, f)
        if st.sum() > 30:
            # per contiguous still segment: mean / std (remove the segment mean)
            seg = np.split(np.where(st)[0], np.where(np.diff(np.where(st)[0]) > 1)[0] + 1)
            seg = [s for s in seg if len(s) >= 30]
            if seg:
                fs = np.concatenate([f[s] - f[s].mean(0) for s in seg]); ws = np.concatenate([w[s] - w[s].mean(0) for s in seg])
                gm = np.array([w[s].mean(0) for s in seg]); gn = np.array([np.linalg.norm(f[s].mean(0)) for s in seg])
                print(f"  still: {len(seg)} segments, {sum(len(s) for s in seg)} samples; accel std per axis {np.round(fs.std(0), 4)} m/s2 "
                      f"(value changes per sample {np.mean(np.any(np.diff(np.concatenate([f[s] for s in seg]), axis=0) != 0, axis=1)) * 100:.0f} %); "
                      f"|f| {np.round(gn, 3)}; gyro std {np.round(np.degrees(ws.std(0)), 3)} deg/s, gyro mean (bias) {np.round(np.degrees(gm), 3).tolist()} deg/s")
        # Generic Sensor streams (sensor timestamps)
        for nm, col in (("accel_gs.csv", f), ("gyro_gs.csv", w)):
            p = rec / nm
            if not p.exists():
                continue
            g = load_csv(p)
            tg = (g[:, 0] - d[0, 0]) * 1e-9
            dg = np.diff(tg) * 1e3
            vg = g[:, 1:4]
            # latency DeviceMotion (event time) vs Generic Sensor (reading time): shift that best matches the norms
            sig_dm = np.linalg.norm(col, axis=1); sig_gs = np.linalg.norm(vg, axis=1)
            best = None
            for lag in np.arange(-0.4, 0.40001, 0.001):
                x = np.interp(t, tg + lag, sig_gs)
                c = np.mean((x - sig_dm) ** 2)
                if best is None or c < best[1]:
                    best = (lag, c)
            print(f"  {nm}: {len(tg)} readings, {len(tg) / (tg[-1] - tg[0]):.2f} Hz; interval ms p5/p50/p95 {pct(dg)}; "
                  f"DeviceMotion event time - reading time ~ {best[0] * 1e3:+.0f} ms")
        cam = load_csv(rec / "mav0/cam0/data.csv", cols=0).ravel()
        tc = (cam - cam[0]) * 1e-9
        dc = np.diff(tc) * 1e3
        print(f"  camera: {len(tc)} frames, {len(tc) / tc[-1]:.2f} fps; interval ms p5/p50/p95 {pct(dc)}, max {dc.max():.1f}; "
              f"first frame {(cam[0] - d[0, 0]) * 1e-9:+.3f} s after the first IMU event")


if __name__ == "__main__":
    main()
