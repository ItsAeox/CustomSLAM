#!/usr/bin/env python3
"""Stage times of the slowest 5 % of frames vs all frames, from a REGISTER_TIMES run directory
(TrackingTimeStats.txt: per frame image rect, resize, ORB extraction, stereo, IMU, pose prediction, local-map tracking,
keyframe decision, total [ms]) and track_times.txt (state per frame: 5 = flow (KLT) frame).

  p95_stages.py <run dir>
"""
import sys
from pathlib import Path

import numpy as np

d = Path(sys.argv[1])
rows = [l.split(",") for l in (d / "TrackingTimeStats.txt").read_text().splitlines() if l and not l.startswith("#")]
t = np.array([[float(x) for x in r] for r in rows if len(r) >= 9])
names = ["rect", "resize", "ORB extraction", "stereo", "IMU", "pose prediction", "local-map tracking", "keyframe decision", "total"]
total = t[:, 8]
p95 = np.percentile(total, 95)
slow = total >= p95
st = [l.split() for l in (d / "track_times.txt").read_text().splitlines() if len(l.split()) >= 3]
kind = np.array([int(x[2]) for x in st])[: len(total)] if st else np.zeros(len(total), int)
print(f"frames {len(total)}, p95 {p95:.1f} ms; slowest 5 %: {slow.sum()} frames, of which flow frames {np.sum(kind[slow] == 5)}")
print(f"{'stage':20s} {'all frames':>10s} {'slowest 5 %':>12s} {'share of slow':>14s}")
for k in (2, 5, 6, 7, 8):
    a, b = t[:, k].mean(), t[slow, k].mean()
    print(f"{names[k]:20s} {a:10.2f} {b:12.2f} {100 * b / t[slow, 8].mean():13.0f}%")
other = t[slow, 8] - t[slow][:, [2, 5, 6, 7]].sum(1)
print(f"{'other (not split)':20s} {'':10s} {other.mean():12.2f} {100 * other.mean() / t[slow, 8].mean():13.0f}%")
