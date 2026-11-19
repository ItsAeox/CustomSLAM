#!/usr/bin/env python3
"""Stage times of the slowest 5 % of frames vs all frames, from a profiling-build run directory (stage_times.txt written
by orbslam3_run under REGISTER_TIMES: frame, total ms, ORB extraction, pose prediction, local-map tracking, keyframe
decision + creation, flow frame, keyframe made). ORB time counts only on ORB frames (flow frames carry a stale copy).
Frames without a pose-prediction and local-map stage (before the first map: two-view start; relocalisation frames do
have a pose stage) are reported on their own line and left out of the split: on room1 the first ~95 frames, ~50 ms each.

  p95_stages.py <run dir>
"""
import sys
from pathlib import Path

import numpy as np

d = Path(sys.argv[1])
t = np.loadtxt(d / "stage_times.txt", ndmin=2)
nostage = (t[:, 3] == 0) & (t[:, 4] == 0)
print(f"frames {len(t)}: {nostage.sum()} without a tracking stage (before the first map): mean {t[nostage, 1].mean() if nostage.any() else 0:.1f} ms,"
      f" max {t[nostage, 1].max() if nostage.any() else 0:.1f} ms; all frames p95 {np.percentile(t[:, 1], 95):.1f} ms")
t = t[~nostage]
total, orb, pose, lm, kf, flow, made = t[:, 1], t[:, 2].copy(), t[:, 3], t[:, 4], t[:, 5], t[:, 6] > 0, t[:, 7] > 0
orb[flow] = 0
p95 = np.percentile(total, 95)
slow = total >= p95
other = total - orb - pose - lm - kf
print(f"tracked frames {len(total)}: mean {total.mean():.1f} ms, p95 {p95:.1f} ms | slowest 5 % ({slow.sum()} frames): "
      f"{100 * np.mean(~flow[slow]):.0f} % ORB frames (all {100 * np.mean(~flow):.0f} %), "
      f"{100 * np.mean(made[slow]):.0f} % made a keyframe (all {100 * np.mean(made):.0f} %)")
print(f"  {'stage (ms)':32s} {'all frames':>10s} {'ORB frames':>10s} {'slowest 5 %':>12s} {'share of slow':>14s}")
for name, v in (("ORB extraction", orb), ("pose prediction", pose), ("local-map tracking", lm),
                ("keyframe decision + creation", kf), ("other (flow, image, bookkeeping)", other)):
    print(f"  {name:32s} {v.mean():10.2f} {v[~flow].mean():10.2f} {v[slow].mean():12.2f} {100 * v[slow].mean() / total[slow].mean():13.0f}%")
