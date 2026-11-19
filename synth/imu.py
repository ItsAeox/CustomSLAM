"""IMU of the synthetic dataset (claude/RESEARCH.md "Phone sensor model, measured").

1. Truth from the motion (1 kHz): body (device) pose R_wb = R_wc R_bc^T, p_b = p_c - R_wb t_bc (t_bc: camera centre in
   the body frame, the lever arm), angular velocity in the body frame, specific force f_b = R_wb^T (a_w - g_w)
   (at rest it points up: (0, +9.81, 0) for an upright portrait phone, as the real recordings).
2. The phone's sensor (`SensorParams`): samples on a 10 ms grid (measured), each the mean of the true signal over its
   10 ms period (assumed internal averaging), with per-axis scale factor, bias (+ random walk) and white noise
   (all assumed: typical phone IMU data sheet values, to be replaced by a still calibration recording).
3. Chrome's DeviceMotion (measured / from the Chromium source): a 60 Hz timer (tick jitter measured) reads the latest
   sample that has arrived (assumed 2 ms availability latency), rounds to 0.1 m/s^2 and 0.1 deg/s, and stamps the
   event with the tick time; the whole camera-IMU pipeline offset is `offset` (reported - true, measured -88 ms).
"""
from dataclasses import asdict, dataclass

import numpy as np

from .geom import G_W, R_BC, so3_log
from .traj import RATE


@dataclass
class SensorParams:
    rate: float = 100.0              # measured: sensor stamps on a 10 ms grid
    avg_window: float = 0.01         # assumed: each sample = mean over its period
    acc_noise: float = 0.012         # assumed: m/s^2 per sample (~120 ug/sqrt(Hz) at 50 Hz bandwidth)
    gyr_noise: float = 0.0006        # assumed: rad/s per sample (0.035 deg/s)
    acc_bias_sigma: float = 0.05     # assumed: m/s^2 per axis (constant part)
    gyr_bias_sigma: float = 0.002    # assumed: rad/s per axis (0.11 deg/s)
    acc_bias_walk: float = 1e-4      # assumed: m/s^2 / sqrt(s)
    gyr_bias_walk: float = 2e-5      # assumed: rad/s / sqrt(s)
    acc_scale_sigma: float = 0.005   # assumed: per-axis scale factor error (0.5 %)
    tick_hz: float = 60.0            # measured / Chromium: DeviceMotion timer
    tick_jitter: float = 0.0004      # measured: event interval p5/p95 16.0 / 17.3 ms
    avail_latency: float = 0.002     # assumed: sample -> shared memory
    round_acc: float = 0.1           # Chromium kAccelerometerRoundingMultiple (m/s^2)
    round_gyr: float = 0.00174532925199432963  # Chromium kGyroscopeRoundingMultiple (rad/s = 0.1 deg/s)
    offset: float = -0.088           # measured: reported IMU time - true time on the camera clock (s)

    def assumed(self):
        return ["avg_window", "acc_noise", "gyr_noise", "acc_bias_sigma", "gyr_bias_sigma", "acc_bias_walk",
                "gyr_bias_walk", "acc_scale_sigma", "avail_latency"]

    def as_dict(self):
        d = asdict(self); d["assumed"] = self.assumed(); return d


def d1(x, h):
    """first derivative, 5-point stencil (4th order); 2nd-order at the ends"""
    y = np.gradient(x, h, axis=0)
    y[2:-2] = (-x[4:] + 8 * x[3:-1] - 8 * x[1:-3] + x[:-4]) / (12 * h)
    return y


def d2(x, h):
    y = np.gradient(np.gradient(x, h, axis=0), h, axis=0)
    y[2:-2] = (-x[4:] + 16 * x[3:-1] - 30 * x[2:-2] + 16 * x[1:-3] - x[:-4]) / (12 * h * h)
    return y


def body_truth(m, lever=(0.0, 0.0, 0.0)):
    """dense truth of the IMU body: t, R_wb, p_b, v_b, w_b (body rad/s), f_b (body m/s^2)"""
    h = 1.0 / RATE
    R_wb = m.R @ R_BC.T
    p_b = m.p - R_wb @ np.asarray(lever, float)
    v = d1(p_b, h)
    a = d2(p_b, h)
    w = np.zeros_like(p_b)
    w[1:-1] = so3_log(np.einsum("nji,njk->nik", R_wb[:-2], R_wb[2:])) / (2 * h)  # R_{i-1}^T R_{i+1}
    w[0], w[-1] = w[1], w[-2]
    f = np.einsum("nji,nj->ni", R_wb, a - G_W)
    return {"t": m.t, "R_wb": R_wb, "p": p_b, "v": v, "w": w, "f": f}


def window_mean(t, x, ts, win):
    """mean of x (dense, uniform t) over (ts - win, ts] via the cumulative integral"""
    c = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum((x[1:] + x[:-1]) * 0.5 * np.diff(t)[:, None], 0)])
    hi = np.stack([np.interp(ts, t, c[:, k]) for k in range(x.shape[1])], 1)
    lo = np.stack([np.interp(ts - win, t, c[:, k]) for k in range(x.shape[1])], 1)
    return (hi - lo) / win


def sensor(truth, rng, P: SensorParams, t0, t1):
    """the phone's 100 Hz samples in [t0, t1]: sample times (true clock), w, f, and the per-sample biases"""
    period = 1.0 / P.rate
    ts = t0 + rng.uniform(0, period) + np.arange(int((t1 - t0) / period) - 1) * period
    ts = ts[ts - P.avg_window >= truth["t"][0]]
    w = window_mean(truth["t"], truth["w"], ts, P.avg_window)
    f = window_mean(truth["t"], truth["f"], ts, P.avg_window)
    n = len(ts)
    ba = rng.normal(0, P.acc_bias_sigma, 3) + np.cumsum(rng.normal(0, P.acc_bias_walk * np.sqrt(period), (n, 3)), 0)
    bg = rng.normal(0, P.gyr_bias_sigma, 3) + np.cumsum(rng.normal(0, P.gyr_bias_walk * np.sqrt(period), (n, 3)), 0)
    sf = rng.normal(0, P.acc_scale_sigma, 3)
    f = f * (1 + sf) + ba + rng.normal(0, P.acc_noise, (n, 3))
    w = w + bg + rng.normal(0, P.gyr_noise, (n, 3))
    return {"t": ts, "w": w, "f": f, "ba": ba, "bg": bg, "acc_scale": sf}


def chrome(s, rng, P: SensorParams):
    """DeviceMotion events: tick times, the sample each tick reads, rounded values, reported stamps (tick + const,
    so that mean(reported - sample time) = P.offset) and the sample's own stamp + offset (Generic Sensor-like)"""
    ts = s["t"]
    dt = 1.0 / P.tick_hz
    ticks = ts[0] + P.avail_latency + 0.003 + np.arange(int((ts[-1] - ts[0]) / dt)) * dt
    ticks = ticks + rng.normal(0, P.tick_jitter, len(ticks))
    idx = np.searchsorted(ts + P.avail_latency, ticks, side="right") - 1
    ok = idx >= 0
    ticks, idx = ticks[ok], idx[ok]
    w = np.round(s["w"][idx] / P.round_gyr) * P.round_gyr
    f = np.round(s["f"][idx] / P.round_acc) * P.round_acc
    c = P.offset - np.mean(ticks - ts[idx])
    return {"t_event": ticks + c, "t_sensor": ts[idx] + P.offset, "idx": idx, "w": w, "f": f,
            "event_minus_sample_ms": [float(np.percentile((ticks - ts[idx]) * 1e3, q)) for q in (5, 50, 95)]}


def ideal(truth, s):
    """the true signal sampled instantaneously at the sensor times, exact stamps (validation variant)"""
    w = np.stack([np.interp(s["t"], truth["t"], truth["w"][:, k]) for k in range(3)], 1)
    f = np.stack([np.interp(s["t"], truth["t"], truth["f"][:, k]) for k in range(3)], 1)
    return {"t": s["t"], "w": w, "f": f}
