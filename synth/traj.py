"""Camera trajectories for the synthetic dataset: dense (1 kHz), smooth (C2: minimum-jerk segments, cubic splines,
band-limited noise), so the IMU derived from them (synth/imu.py) is exact. Seven scenario types (`SCENARIOS`), six
variants each; every variant is fully determined by (scenario, variant index, seed).

A `Motion` holds the camera centre p (world, m) and camera->world rotation R per dense sample; `pose(ts)` interpolates
(linear / slerp) for the renderer's per-row times. Hand-held motion carries a physiological tremor (6-12 Hz, sub-mm,
<0.1 deg) and a slow wobble; table-resting motion carries only friction vibration while sliding.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation, Slerp

from .geom import ypr_rotation

RATE = 1000.0  # Hz


@dataclass
class Motion:
    t: np.ndarray            # (N,) s, starting at 0
    p: np.ndarray            # (N, 3) camera centre, world
    R: np.ndarray            # (N, 3, 3) camera -> world
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        self._slerp = Slerp(self.t, Rotation.from_matrix(self.R))

    def pose(self, ts):
        ts = np.clip(np.atleast_1d(np.asarray(ts, float)), self.t[0], self.t[-1])
        p = np.stack([np.interp(ts, self.t, self.p[:, k]) for k in range(3)], 1)
        return self._slerp(ts).as_matrix(), p

    @property
    def duration(self):
        return float(self.t[-1])


# ------------------------------------------------------------------------------------------------ building blocks
def timeline(duration):
    return np.arange(0.0, duration + 0.5 / RATE, 1.0 / RATE)


def minjerk(tau):
    tau = np.clip(tau, 0, 1)
    return tau ** 3 * (10 - 15 * tau + 6 * tau ** 2)


def segments(t, keys):
    """piecewise minimum-jerk between key values with holds: keys = [(t_start, t_end, v_from, v_to), ...] (scalar or
    vector values); before the first / after the last segment the value is held"""
    v0 = np.asarray(keys[0][2], float)
    out = np.broadcast_to(v0, t.shape + v0.shape).copy()
    for (t0, t1, a, b) in keys:
        a, b = np.asarray(a, float), np.asarray(b, float)
        m = t >= t0
        s = minjerk((t[m] - t0) / max(t1 - t0, 1e-9))
        out[m] = a + (b - a) * s[(...,) + (None,) * a.ndim]
    return out


def bandnoise(t, rng, lo, hi, rms, dims=1, order=4):
    """zero-phase band-limited Gaussian noise (lo = 0: low-pass), scaled to the given RMS per dimension"""
    n = rng.standard_normal((len(t) + 4000, dims))
    if lo <= 0:
        sos = butter(order, hi, "lowpass", fs=RATE, output="sos")
    else:
        sos = butter(order, [lo, hi], "bandpass", fs=RATE, output="sos")
    x = sosfiltfilt(sos, n, axis=0)[2000:2000 + len(t)]
    x *= rms / np.maximum(x.std(0), 1e-12)
    return x if dims > 1 else x[:, 0]


def envelope(t, t0, t1, ramp=0.5):
    """smooth 0 -> 1 -> 0 window over [t0, t1] (minimum-jerk ramps)"""
    return minjerk((t - t0) / ramp) * minjerk((t1 - t) / ramp)


def look_angles(p, target):
    d = np.asarray(target, float) - p
    yaw = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    pitch = np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1]))
    return yaw, pitch


def smooth(x, hz=2.0):
    sos = butter(2, hz, "lowpass", fs=RATE, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def hand(t, rng, still_scale=1.0):
    """physiological tremor + slow hand wobble: translation (N,3) m and rotation (N,3) rad (yaw, pitch, roll)"""
    # tremor 0.05 mm / 0.02 deg RMS at 6-12 Hz (~0.15 m/s^2, ~1 deg/s RMS), wobble 2.5 mm / 0.5 deg below 1 Hz
    # (still phases then match the real hand-held recordings: gyro 1.3-5.5 deg/s, accel 0.07-0.25 m/s^2 std)
    tr = bandnoise(t, rng, 6, 12, 0.00005, 3) + bandnoise(t, rng, 0, 1.0, 0.0025 * still_scale, 3)
    ro = bandnoise(t, rng, 6, 12, np.radians(0.02), 3) + bandnoise(t, rng, 0, 1.0, np.radians(0.5) * still_scale, 3)
    return tr, ro


def gait(t, speed, rng, amp_v=0.02, amp_l=0.012):
    """walking: vertical bob at the step frequency and lateral sway at half of it, scaled by the walking speed"""
    f = 1.8 + 0.2 * rng.standard_normal()
    k = smooth(np.clip(speed / 0.8, 0, 1.2), 1.0)  # amplitude follows the (smoothed) speed; the phase runs continuously
    return amp_v * k * np.sin(2 * np.pi * f * t), amp_l * k * np.sin(np.pi * f * t)


def finish(t, p, yaw, pitch, roll, info):
    return Motion(t, p, ypr_rotation(yaw, pitch, roll), info)


# ------------------------------------------------------------------------------------------------ scenarios
def slide_rig(room):
    """a table edge to slide the phone along (camera ~8 cm above the top, facing into the room): A, unit direction,
    yaw, height"""
    if room.name == "office":
        return np.array([0.5, 1.13, 0.83]), np.array([1.0, 0, 0]), np.pi / 2
    if room.name == "bedroom":
        sx = room.size[0]
        return np.array([sx - 0.63, 0.4, 0.83]), np.array([0, 1.0, 0]), np.pi
    return np.array([5.47, 2.8, 0.68]), np.array([0, 1.0, 0]), np.pi


def scn_slide(room, rng, k):
    """S1: phone resting on a table edge, slides of exactly 1.00 m between two marks (the user's 21 / 29 Oct test);
    slide duration per variant 13 / 6 / 4 / 3 / 2 / 1.5 s; small yaw wobble; friction vibration while sliding"""
    D = [13.0, 6.0, 4.0, 3.0, 2.0, 1.5][k]
    n = 4 if D >= 6 else 10
    L = 1.0
    A, u, yaw0 = slide_rig(room)
    t_still = 3.0
    keys, tc = [], t_still
    pos = 0.0
    for i in range(n):
        nxt = L - pos
        keys.append((tc, tc + D, [pos], [nxt]))
        tc += D + rng.uniform(1.5, 3.0)
        pos = nxt
    dur = tc + 2.0
    t = timeline(dur)
    s = segments(t, keys)[:, 0]
    moving = np.abs(np.gradient(s, 1 / RATE)) > 0.02
    vib = bandnoise(t, rng, 15, 45, 1.5e-6 * (1 + k), 3) * smooth(moving.astype(float), 5)[:, None]
    p = A[None] + s[:, None] * u[None] + vib
    yaw = yaw0 + bandnoise(t, rng, 0.2, 2.0, np.radians(0.6), 1) * smooth(moving.astype(float), 3)
    pitch = np.radians(-5) + bandnoise(t, rng, 0.2, 2.0, np.radians(0.2), 1) * smooth(moving.astype(float), 3)
    roll = bandnoise(t, rng, 0.2, 2.0, np.radians(0.2), 1) * smooth(moving.astype(float), 3)
    return finish(t, p, yaw, pitch, roll, {"slide_m": L, "slide_s": D, "slides": n, "rig": room.name,
                                           "marks": [A.tolist(), (A + L * u).tolist()]})


def stand_point(room, rng, target, dist=(1.4, 2.4)):
    x0, y0, x1, y1 = room.free
    for _ in range(200):
        p = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
        if dist[0] <= np.linalg.norm(p - target[:2]) <= dist[1]:
            return p
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2])


def scn_gesture(room, rng, k):
    """S2: hand-held 'scale gesture': still, a short sideways init move, then brisk side-to-side translation
    (amplitude 0.1-0.3 m, 0.7-2 Hz) with some wrist rotation, then looking around"""
    target = room.targets["board" if k % 2 else "table" if "table" in room.targets else "centre"]
    xy = stand_point(room, rng, target)
    z = rng.uniform(1.2, 1.5)
    amp = [0.1, 0.15, 0.2, 0.25, 0.3, 0.12][k]   # peak acceleration A (2 pi f)^2: 6-10 m/s^2
    f = [1.6, 1.2, 1.0, 0.9, 0.7, 1.4][k]
    t_still, t_init, t_g = 2.5, 1.5, rng.uniform(15, 25)
    dur = t_still + t_init + 1.0 + t_g + 10.0
    t = timeline(dur)
    base = np.array([xy[0], xy[1], z])
    fwd = target - base
    right = np.array([fwd[1], -fwd[0], 0]); right /= np.linalg.norm(right)
    init = segments(t, [(t_still, t_still + t_init, 0.0, 0.25)])
    g0 = t_still + t_init + 1.0
    osc = amp * np.sin(2 * np.pi * f * (t - g0)) * envelope(t, g0, g0 + t_g, 0.6)
    tr, ro = hand(t, rng)
    p = base[None] + (init + osc)[:, None] * right[None] + tr
    yaw, pitch = look_angles(p, target)
    yaw = smooth(yaw, 3) + np.radians([0, 2, 4, 6, 3, 1][k]) * osc / max(amp, 1e-6) + ro[:, 0]
    look = segments(t, [(g0 + t_g + 1, g0 + t_g + 4, 0.0, np.radians(40)), (g0 + t_g + 5, g0 + t_g + 9, np.radians(40), np.radians(-30))])
    yaw = yaw + look
    return finish(t, p, yaw, smooth(pitch, 3) + ro[:, 1], ro[:, 2],
                  {"gesture_amp_m": amp, "gesture_hz": f, "gesture_s": round(t_g, 1)})


def scn_shake(room, rng, k):
    """S3: hand-held random shaking bursts (translation 3-10 cm RMS, rotation 2-10 deg RMS, 0.3-4 Hz) between calm"""
    target = room.targets["centre"]
    xy = stand_point(room, rng, target, (1.2, 2.2))
    base = np.array([xy[0], xy[1], rng.uniform(1.2, 1.5)])
    tr_rms = [0.015, 0.02, 0.025, 0.03, 0.02, 0.035][k]   # 0.3-3 Hz
    ro_rms = np.radians([2, 4, 6, 8, 10, 3][k])
    t_still = 2.5
    bursts, tc = [], t_still + 2.5
    for _ in range(3):
        L = rng.uniform(6, 8)
        bursts.append((tc, tc + L)); tc += L + 3.0
    dur = tc + 1.0
    t = timeline(dur)
    init = segments(t, [(t_still, t_still + 1.5, 0.0, 0.25)])
    env = sum(envelope(t, a, b, 0.4) for a, b in bursts)
    sh_t = bandnoise(t, rng, 0.3, 3.0, tr_rms, 3) * env[:, None]
    sh_r = bandnoise(t, rng, 0.3, 3.0, ro_rms, 3) * env[:, None]
    tr, ro = hand(t, rng)
    fwd = target - base
    right = np.array([fwd[1], -fwd[0], 0]); right /= np.linalg.norm(right)
    p = base[None] + init[:, None] * right[None] + sh_t + tr
    yaw, pitch = look_angles(np.repeat(base[None], len(t), 0), target)
    return finish(t, p, yaw + sh_r[:, 0] + ro[:, 0], pitch + sh_r[:, 1] + ro[:, 1], sh_r[:, 2] * 0.5 + ro[:, 2],
                  {"shake_trans_rms_m": tr_rms, "shake_rot_rms_deg": round(float(np.degrees(ro_rms)), 1)})


def scn_rotate(room, rng, k):
    """S4: standing on the spot, turning the phone: pans (peak 30-240 deg/s), tilts, rolls, one full turn; the camera
    pivots ~0.35 m in front of the body axis (rotation with the translation that comes with it)"""
    peak = np.radians([30, 60, 90, 120, 180, 240][k])
    xy = np.array([(room.free[0] + room.free[2]) / 2, (room.free[1] + room.free[3]) / 2]) + rng.uniform(-0.3, 0.3, 2)
    z = rng.uniform(1.25, 1.5)
    yaw0 = rng.uniform(-np.pi, np.pi)
    t_still = 2.5
    keys_y, keys_p, keys_r = [], [], []
    tc, y = t_still + 2.5, 0.0
    moves = [rng.uniform(60, 150) * s for s in (1, -1, -1, 1)] + ([360.0] if k in (2, 4) else [])
    for dy in moves:
        D = 1.875 * np.radians(abs(dy)) / peak
        keys_y.append((tc, tc + D, y, y + np.radians(dy))); y += np.radians(dy); tc += D + rng.uniform(1.0, 2.0)
    for dp in (-25, 25):
        D = 1.875 * np.radians(abs(dp)) / peak
        keys_p.append((tc, tc + D, 0.0 if dp < 0 else np.radians(-25), np.radians(-25) if dp < 0 else 0.0)); tc += D + 1.5
    for dr in (25, -25):
        D = 1.875 * np.radians(abs(dr)) / peak
        keys_r.append((tc, tc + D, 0.0 if dr > 0 else np.radians(25), np.radians(25) if dr > 0 else 0.0)); tc += D + 1.5
    dur = tc + 1.5
    t = timeline(dur)
    yaw = yaw0 + segments(t, keys_y)
    pitch = np.radians(-15) + segments(t, keys_p)
    roll = segments(t, keys_r)
    tr, ro = hand(t, rng)
    init = segments(t, [(t_still, t_still + 1.5, 0.0, 0.3)])
    arm = 0.35
    p = np.stack([xy[0] + arm * np.cos(yaw), xy[1] + arm * np.sin(yaw), np.full_like(t, z)], 1)
    side = np.stack([np.sin(yaw0) * np.ones_like(t), -np.cos(yaw0) * np.ones_like(t), 0 * t], 1)
    p = p + init[:, None] * side + tr
    return finish(t, p, yaw + ro[:, 0], pitch + ro[:, 1], roll + ro[:, 2],
                  {"peak_rate_dps": round(float(np.degrees(peak))), "full_turn": k in (2, 4)})


def scn_orbit(room, rng, k):
    """S5: walking a circle arc around a point of interest while looking at it (radius 0.7-1.5 m, 180-360 deg,
    0.25-0.6 m/s), some variants back and forth"""
    name = ["board", "centre", "board", "centre", "board", "centre"][k]
    c = room.targets[name].copy()
    r = [0.8, 1.0, 1.2, 0.9, 1.4, 1.1][k]
    speed = [0.25, 0.35, 0.45, 0.6, 0.3, 0.5][k]
    ext = np.radians([180, 270, 360, 200, 240, 300][k])
    x0, y0, x1, y1 = room.free
    r = min(r, (c[0] - x0) * 0.98 + 0.3, (x1 - c[0]) * 0.98 + 0.3, (c[1] - y0) * 0.98 + 0.3, (y1 - c[1]) * 0.98 + 0.3)
    r = max(r, 0.6)
    z = rng.uniform(1.1, 1.5)
    a0 = rng.uniform(-np.pi, np.pi)
    t_still = 2.5
    D = ext * r / speed  # mean speed `speed` (minimum-jerk peak: 1.875 x)
    back = True
    keys = [(t_still + 2.0, t_still + 2.0 + D, 0.0, ext)]
    if back:
        keys.append((t_still + 3.0 + D, t_still + 3.0 + 2 * D, ext, 0.0))
    dur = keys[-1][1] + 2.0
    t = timeline(dur)
    a = a0 + segments(t, keys)
    spd = np.abs(np.gradient(a, 1 / RATE)) * r
    bob_v, bob_l = gait(t, spd, rng)
    tr, ro = hand(t, rng)
    init = segments(t, [(t_still, t_still + 1.5, 0.0, 0.25)])
    p = np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a), z + bob_v], 1)
    tang = np.stack([-np.sin(a), np.cos(a), 0 * a], 1)
    p = p + (bob_l + init)[:, None] * tang + tr
    yaw, pitch = look_angles(p, c)
    return finish(t, p, smooth(yaw, 3) + ro[:, 0], smooth(pitch, 3) + ro[:, 1], ro[:, 2],
                  {"target": name, "radius_m": round(float(r), 2), "speed_mps": speed,
                   "arc_deg": round(float(np.degrees(ext))), "back_and_forth": back})


def free_path(room, rng, n):
    x0, y0, x1, y1 = room.free
    pts = [np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])]
    while len(pts) < n:
        q = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
        if np.linalg.norm(q - pts[-1]) > 0.8:
            pts.append(q)
    return np.array(pts)


def scn_walk(room, rng, k):
    """S6: walking through the room along a smooth path (0.4-1.0 m/s) with gait bob / sway, head turns, short stops"""
    speed = [0.4, 0.5, 0.6, 0.7, 0.85, 1.0][k]
    pts = free_path(room, rng, 8)
    seglen = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    t_still = 2.5
    tw = t_still + 2.0 + np.concatenate([[0.0], np.cumsum(seglen / speed)])
    dur = tw[-1] + 2.5
    t = timeline(dur)
    cs = CubicSpline(tw, pts, bc_type="clamped")  # C2 path through the waypoints, at rest at both ends
    xy = smooth(cs(np.clip(t, tw[0], tw[-1])), 1.5)
    v = np.gradient(xy, 1 / RATE, axis=0)
    spd = np.linalg.norm(v, axis=1)
    heading = np.unwrap(np.arctan2(smooth(v[:, 1], 0.5), smooth(v[:, 0], 0.5)))
    heading = np.where(spd > 0.05, heading, np.nan)
    idx = np.where(~np.isnan(heading))[0]
    heading = np.interp(np.arange(len(t)), idx, heading[idx]) if len(idx) else np.zeros(len(t))
    bob_v, bob_l = gait(t, spd, rng)
    z = rng.uniform(1.3, 1.5)
    tr, ro = hand(t, rng)
    side = np.stack([np.sin(heading), -np.cos(heading)], 1)
    p = np.concatenate([xy + bob_l[:, None] * side, (z + bob_v)[:, None]], 1) + tr
    look = bandnoise(t, rng, 0, 0.25, np.radians(30), 1)
    yaw = smooth(heading, 1.0) + look + ro[:, 0]
    pitch = np.radians(-20) + bandnoise(t, rng, 0, 0.3, np.radians(8), 1) + ro[:, 1]
    return finish(t, p, yaw, pitch, ro[:, 2], {"speed_mps": speed, "waypoints": pts.round(2).tolist()})


def scn_ar(room, rng, k):
    """S7: an AR session: still, sideways init, look down at a floor spot (tap / place), walk a half circle around it
    looking at it, look away 90 deg and back, a fast 180 deg turn and back, return to the start"""
    c = room.targets["board"].copy(); c[2] = 0.0
    r = [1.0, 1.2, 1.1, 1.3, 1.0, 1.2][k]
    turn = np.radians([120, 150, 180, 200, 240, 160][k])
    x0, y0, x1, y1 = room.free
    a0 = rng.uniform(-np.pi, np.pi)
    z = rng.uniform(1.25, 1.5)
    t_still = 2.5
    tc = t_still
    T = {}
    T["init"] = (tc, tc + 1.5); tc += 2.5
    T["place"] = (tc, tc + 2.0); tc += 2.5
    arc = np.pi * 0.9
    D = arc * r / 0.3
    T["arc"] = (tc, tc + D); tc += D + 0.5
    T["away"] = (tc, tc + 90 / 120 * 1.875); tc += T["away"][1] - T["away"][0] + 3.0
    T["back"] = (tc, tc + 90 / 120 * 1.875); tc += T["back"][1] - T["back"][0] + 1.5
    Dt = 1.875 * np.pi / turn
    T["turn"] = (tc, tc + Dt); tc += Dt + 1.0
    T["turnback"] = (tc, tc + Dt); tc += Dt + 1.0
    T["return"] = (tc, tc + D * 0.6); tc += D * 0.6 + 2.0
    dur = tc
    t = timeline(dur)
    a = a0 + segments(t, [(T["arc"][0], T["arc"][1], 0.0, arc), (T["return"][0], T["return"][1], arc, 0.0)])
    spd = np.abs(np.gradient(a, 1 / RATE)) * r
    bob_v, bob_l = gait(t, spd, rng)
    init = segments(t, [(T["init"][0], T["init"][1], 0.0, 0.3)])
    p = np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a), z + bob_v], 1)
    tang = np.stack([-np.sin(a), np.cos(a), 0 * a], 1)
    tr, ro = hand(t, rng)
    p = p + (init + bob_l)[:, None] * tang + tr
    yaw, pitch = look_angles(p, c)
    off = segments(t, [(T["away"][0], T["away"][1], 0.0, np.radians(90)), (T["back"][0], T["back"][1], np.radians(90), 0.0),
                       (T["turn"][0], T["turn"][1], 0.0, np.pi), (T["turnback"][0], T["turnback"][1], np.pi, 0.0)])
    poff = segments(t, [(T["away"][0], T["away"][1], 0.0, np.radians(25)), (T["back"][0], T["back"][1], np.radians(25), 0.0),
                        (T["turn"][0], T["turn"][1], 0.0, np.radians(25)), (T["turnback"][0], T["turnback"][1], np.radians(25), 0.0)])
    return finish(t, p, smooth(yaw, 3) + off + ro[:, 0], smooth(pitch, 3) + poff + ro[:, 1], ro[:, 2],
                  {"spot": c.round(3).tolist(), "radius_m": r, "turn_peak_dps": round(float(np.degrees(turn))),
                   "phases_s": {k_: [round(v[0], 2), round(v[1], 2)] for k_, v in T.items()}})


SCENARIOS = {"slide": scn_slide, "gesture": scn_gesture, "shake": scn_shake, "rotate": scn_rotate,
             "orbit": scn_orbit, "walk": scn_walk, "ar": scn_ar}


def check_motion(room, m: Motion, table_ok=False):
    """camera inside the room (0.25 m from walls, 0.3 m above the floor) and outside every furniture box (0.12 m
    margin; `table_ok`: no margin, for a phone resting on a table); returns a list of problems"""
    sx, sy, sz = room.size
    p = m.p[::20]
    bad = []
    if (p[:, 0] < 0.25).any() or (p[:, 0] > sx - 0.25).any() or (p[:, 1] < 0.25).any() or (p[:, 1] > sy - 0.25).any():
        bad.append("near a wall")
    if (p[:, 2] < 0.3).any() or (p[:, 2] > sz - 0.2).any():
        bad.append("height")
    mg = 0.0 if table_ok else 0.12
    for (name, x0, y0, z0, bx, by, bz) in room.boxes:
        inside = ((p[:, 0] > x0 - mg) & (p[:, 0] < x0 + bx + mg) & (p[:, 1] > y0 - mg) & (p[:, 1] < y0 + by + mg) &
                  (p[:, 2] > z0 - mg) & (p[:, 2] < z0 + bz + mg))
        if inside.any():
            bad.append(f"inside {name}")
    return bad
