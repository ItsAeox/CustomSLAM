#!/usr/bin/env python3
"""Phone-like IMU for an ASL (EuRoC / TUM-VI) sequence: lower rate, timestamp jitter, dropped samples.

Writes <out>/mav0/imu0/data.csv and links <out>/mav0/cam0 to the original images, so orbslam3_run can play it
(--dataset euroc --seq <out>). Used to reproduce the phone mono-inertial hang (claude/ISSUES.md G13): the phone
delivers ~60 Hz DeviceMotion for a 30 fps camera (~2 samples per frame), EuRoC 200 Hz for 20 fps (~10).

Sampling: target times t0 + k / rate (+ uniform jitter), each takes the nearest original sample's values (no
averaging, like an event that reports the latest sensor reading); a fraction of samples is dropped.
--rate 0 keeps the original samples (for one-factor tests: offset or jitter only).
--offset-ms shifts every IMU timestamp: reported time = true time + offset. Positive = the IMU stamps are late
relative to the camera (as when a browser stamps a motion event at delivery, not at measurement).

  degrade_imu.py <seq> <out> --rate 40 --jitter-ms 5 --drop 0.05 --seed 1
  degrade_imu.py <seq> <out> --rate 0 --offset-ms 50
"""
import argparse
import os
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('seq', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--rate', type=float, default=0.0, help='IMU rate after resampling (Hz); 0 = keep the original samples')
    ap.add_argument('--offset-ms', type=float, default=0.0, help='added to every IMU timestamp (ms)')
    ap.add_argument('--jitter-ms', type=float, default=0.0, help='uniform timestamp jitter, +- this many ms')
    ap.add_argument('--drop', type=float, default=0.0, help='fraction of samples dropped at random')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--average', action='store_true',
                    help='each sample = mean of the original samples since the previous one (anti-aliasing, as a '
                         'phone sensor filters before it reports) instead of the nearest original sample')
    a = ap.parse_args()

    src = a.seq / 'mav0' / 'imu0' / 'data.csv'
    rows = np.loadtxt(src, delimiter=',', comments='#')  # t [ns], w_xyz, a_xyz
    # times relative to the first sample: absolute ns (~1.5e18) exceed float64's exact range (2^53)
    t_abs = np.loadtxt(src, delimiter=',', comments='#', usecols=0, dtype=np.int64)
    t0 = int(t_abs[0])
    t = (t_abs - t0).astype(np.float64)
    rng = np.random.default_rng(a.seed)
    if a.rate <= 0:  # original samples (jitter / drop / offset only)
        tq = t + rng.uniform(-a.jitter_ms, a.jitter_ms, t.size) * 1e6
        vals = rows[:, 1:]
    elif a.average:  # values sampled at the true times, averaged over the preceding interval
        step = 1e9 / a.rate
        tq = np.arange(t[0], t[-1], step)
        hi = np.searchsorted(t, tq, side='right')
        lo = np.searchsorted(t, tq - step, side='right')
        csum = np.vstack([np.zeros((1, 6)), np.cumsum(rows[:, 1:], axis=0)])
        vals = (csum[hi] - csum[lo]) / np.maximum(hi - lo, 1)[:, None]
        tq = tq + rng.uniform(-a.jitter_ms, a.jitter_ms, tq.size) * 1e6  # reported time jitters (event delivery)
    else:  # the nearest original sample to the (jittered) time
        step = 1e9 / a.rate
        tq = np.arange(t[0], t[-1], step)
        tq = tq + rng.uniform(-a.jitter_ms, a.jitter_ms, tq.size) * 1e6
        idx = np.clip(np.searchsorted(t, tq), 1, t.size - 1)
        idx = np.where(np.abs(t[idx - 1] - tq) < np.abs(t[idx] - tq), idx - 1, idx)
        vals = rows[idx, 1:]
    out = np.column_stack([tq, vals])
    out = out[(out[:, 0] >= t[0]) & (out[:, 0] <= t[-1])]
    if a.drop > 0:
        out = out[rng.random(out.shape[0]) >= a.drop]
    out = out[np.argsort(out[:, 0], kind='stable')]
    out = out[np.concatenate([[True], np.diff(out[:, 0]) > 0])]  # strictly increasing timestamps
    out[:, 0] += a.offset_ms * 1e6

    imu_dir = a.out / 'mav0' / 'imu0'
    imu_dir.mkdir(parents=True, exist_ok=True)
    with open(imu_dir / 'data.csv', 'w') as f:
        f.write('#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],'
                'a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n')
        for r in out:
            f.write(f'{t0 + int(round(r[0]))},' + ','.join(f'{v:.17g}' for v in r[1:]) + '\n')
    cam = a.out / 'mav0' / 'cam0'
    if not cam.exists():
        os.symlink((a.seq / 'mav0' / 'cam0').resolve(), cam, target_is_directory=True)
    dt = np.diff(out[:, 0]) / 1e9
    print(f'{a.out}: {out.shape[0]} samples (from {rows.shape[0]}), mean rate {1 / dt.mean():.1f} Hz, '
          f'dt min/max {dt.min() * 1e3:.1f}/{dt.max() * 1e3:.1f} ms')


if __name__ == '__main__':
    main()
