#!/usr/bin/env python3
"""Why did tracking get lost? Per loss, the 2 s before it, from an orbslam3_run --trace run (trace.txt columns: frame,
state, tracked points, Twc t xyz, q xyzw, then keypoints, flow frame, local map size, map KFs, map points, mapping queue,
mapping accepts KFs, new map points so far, textured cells, textured cells without a tracked point) and the recording's
gyro (rotation rate at each frame).

Per loss: frame / time; tracked points 2 s, 1 s, 0.3 s before and just before; keyframes added in those 2 s and the last
time one was added; new map points triangulated in those 2 s; share of frames where mapping was busy (not accepting);
untracked textured cells just before; gyro rate (median / max deg/s); flow frames share.

  loss_anatomy.py <run dir> <recording dir>
"""
import sys
from pathlib import Path

import numpy as np

run, rec = Path(sys.argv[1]), Path(sys.argv[2])
tr = np.array([[float(x) for x in l.split()] for l in (run / "trace.txt").read_text().splitlines() if len(l.split()) >= 20])
if not len(tr):
    sys.exit("no extended trace (needs the --trace build of 18 Oct)")
cam = np.loadtxt(rec / "mav0" / "cam0" / "data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64)
imu = np.loadtxt(rec / "mav0" / "imu0" / "data.csv", delimiter=",", comments="#")
t_cam = (cam - cam[0]) * 1e-9
t_imu = (imu[:, 0] - cam[0]) * 1e-9
rate = np.degrees(np.linalg.norm(imu[:, 1:4], axis=1))
fr = tr[:, 0].astype(int)
t = t_cam[np.clip(fr, 0, len(t_cam) - 1)]
st = tr[:, 1].astype(int)
ok = np.isin(st, [2, 5])
tracked, klt, lmap, kfs, mps, queue, accept, newmp, tex, untr = (tr[:, 2], tr[:, 11], tr[:, 12], tr[:, 13], tr[:, 14], tr[:, 15],
                                                                 tr[:, 16], tr[:, 17], tr[:, 18], tr[:, 19])
losses = [i for i in range(1, len(ok)) if ok[i - 1] and not ok[i]]
# a map ends when the current map's keyframe count drops to 0 (culling only removes a few): kept in the atlas if it had
# >= 10 keyframes (new map started), else discarded (reset, e.g. a failed start)
ends = [i for i in range(1, len(kfs)) if kfs[i] == 0 and kfs[i - 1] > 0]
kept = sum(1 for i in ends if kfs[i - 1] >= 10)
print(f"{run}: {len(tr)} frames, {len(losses)} losses, new maps {kept} (+ {len(ends) - kept} resets), current map at the end {int(kfs[-1])} KFs")
for i in losses:
    w = (t >= t[i] - 2.0) & (np.arange(len(t)) < i)
    def at(dt):
        j = np.searchsorted(t, t[i] - dt)
        return int(tracked[min(j, i - 1)])
    kf_add = np.sum(np.diff(kfs[w]) > 0) if w.sum() > 1 else 0
    kf_idx = np.where((np.diff(kfs[: i]) > 0))[0]
    last_kf = t[i] - t[kf_idx[-1] + 1] if len(kf_idx) else np.nan
    new = newmp[i - 1] - newmp[w][0] if w.any() else 0
    g = rate[(t_imu >= t[i] - 2.0) & (t_imu <= t[i])]
    print(f"  loss at frame {fr[i]} (t {t[i]:.1f} s): tracked 2 s / 1 s / 0.3 s / last {at(2.0)} / {at(1.0)} / {at(0.3)} / {int(tracked[i - 1])}"
          f" | KFs added {kf_add} (last {last_kf:.1f} s before) | new points {int(new)} | mapping busy {100 * np.mean(accept[w] == 0):.0f}%"
          f" (queue max {int(queue[w].max()) if w.any() else 0}) | untracked textured cells {int(untr[i - 1])}/{int(tex[i - 1])}"
          f" | gyro median {np.median(g):.0f} max {np.max(g):.0f} deg/s | flow frames {100 * np.mean(klt[w]):.0f}%")
    j = i
    while j < len(ok) and not ok[j]:
        j += 1
    end = [e for e in ends if i <= e <= j]
    how = "end of recording" if j == len(ok) else ("tracking again in a new map" if any(kfs[e - 1] >= 10 for e in end) else
                                                   "tracking again after a reset" if end else "tracking again in the same map")
    print(f"    lost {t[min(j, len(t) - 1)] - t[i]:.1f} s, then {how}")
lost_s = np.sum(np.diff(t)[~ok[1:]])
print(f"  lost {lost_s:.1f} s of {t[-1] - t[0]:.1f} s; tracked {100 * ok.mean():.0f}% of frames")
if tr.shape[1] >= 22:  # turn bridge columns (Web.Turn.Bridge, 18 Oct): bridges so far, tracked points from before the latest bridge
    nb, old = tr[:, 20], tr[:, 21]
    print(f"  turn bridges {int(nb[-1])}; frames after a bridge tracking mostly points from before it: "
          f"{int(np.sum((nb > 0) & (old > 0.5 * tracked)))}")
