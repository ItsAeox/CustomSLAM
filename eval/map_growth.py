#!/usr/bin/env python3
"""How the map grows (ISSUES G22): from an orbslam3_run --trace run (trace.txt with the 18 Oct counters: ..., new points so
far (col 17), bridges, old points, new points culled for a low found ratio (22) / too few observations (23), keyframes
culled (24), keyframes due while local mapping was busy (25)), per time window: tracking state share, keyframes and map
points in the current map, points triangulated, points culled (by reason), keyframes culled, keyframes refused because
mapping was busy, mapping busy share. Totals at the end. The window table shows where the map stops growing.

  map_growth.py <run dir> <recording dir> [window s = 5]
"""
import sys
from pathlib import Path

import numpy as np

run, rec = Path(sys.argv[1]), Path(sys.argv[2])
win = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
tr = np.array([[float(x) for x in l.split()] for l in (run / "trace.txt").read_text().splitlines() if len(l.split()) >= 26])
if not len(tr):
    sys.exit("no trace with the map-growth counters (needs the build of 18 Oct evening)")
cam = np.loadtxt(rec / "mav0" / "cam0" / "data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64)
t = (cam[np.clip(tr[:, 0].astype(int), 0, len(cam) - 1)] - cam[0]) * 1e-9
ok = np.isin(tr[:, 1].astype(int), [2, 5])
kfs, mps, accept, newmp = tr[:, 13], tr[:, 14], tr[:, 16], tr[:, 17]
cr, co, ckf, busy = tr[:, 22], tr[:, 23], tr[:, 24], tr[:, 25]
tracked = tr[:, 2]
print(f"{run}: {len(tr)} frames, {t[-1]:.0f} s | totals: points triangulated {int(newmp[-1])}, culled for found ratio {int(cr[-1])}, "
      f"for observations {int(co[-1])}, keyframes culled {int(ckf[-1])}, keyframes due while mapping busy {int(busy[-1])}; "
      f"map at the end {int(kfs[-1])} KFs, {int(mps[-1])} points")
print(f"  {'window':>11s} {'tracked':>7s} {'pts/frm':>7s} {'KFs':>9s} {'points':>11s} {'+tri':>5s} {'-ratio':>6s} {'-obs':>5s} {'-KF':>4s} {'busy':>5s} {'map busy':>8s}")
for a in np.arange(0, t[-1], win):
    w = (t >= a) & (t < a + win)
    if w.sum() < 2:
        continue
    i0, i1 = np.where(w)[0][[0, -1]]
    d = lambda v: int(v[i1] - v[max(i0 - 1, 0)])
    print(f"  {a:4.0f}-{a + win:4.0f} s {100 * ok[w].mean():6.0f}% {np.median(tracked[w]):7.0f} {int(kfs[i0]):4d}->{int(kfs[i1]):<4d} "
          f"{int(mps[i0]):5d}->{int(mps[i1]):<5d} {d(newmp):5d} {d(cr):6d} {d(co):5d} {d(ckf):4d} {d(busy):5d} {100 * np.mean(accept[w] == 0):7.0f}%")
