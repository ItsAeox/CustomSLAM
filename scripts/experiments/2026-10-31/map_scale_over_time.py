#!/usr/bin/env python3
"""ISSUES G32: true scale of a live WebVIO run over time (Sim3 of 8 s windows of tracked positions vs GT, >= 0.15 m of motion), every 2 s; "." = too little motion.
  map_scale_over_time.py <live results dir> <arm> <window s> <seq ...>"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "H:/Research/CustomSLAM/eval")
from scale_from_accel import umeyama_scale
L, arm, W = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])
for q in sys.argv[4:]:
    run = L / q / arm / "seed1"
    tr = np.array([[float(x) for x in l.split()[:14]] for l in (run / "trace.txt").read_text().splitlines() if len(l.split()) >= 14])
    cam = np.loadtxt(f"H:/Research/datasets/synth/{q}/mav0/cam0/data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64)
    t = cam[np.clip(tr[:, 0].astype(int), 0, len(cam) - 1)] * 1e-9
    ok = np.isin(tr[:, 1].astype(int), [2, 5]); pos, kfs = tr[:, 3:6], tr[:, 13]
    mapid = np.cumsum(np.concatenate([[0], (kfs[1:] == 0) & (kfs[:-1] > 0)]))
    gt = np.loadtxt(f"H:/Research/results/gt/synth_{q}.txt", comments="#", ndmin=2)
    out = []
    for tc in np.arange(t[0] + W, t[-1], 2.0):
        w = np.where(ok & (t > tc - W) & (t <= tc))[0]
        if len(w) < 30 or len(set(mapid[w])) > 1: out.append("  -  "); continue
        g = np.stack([np.interp(t[w], gt[:, 0], gt[:, j]) for j in (1, 2, 3)], 1)
        if np.ptp(g, 0).max() < 0.15: out.append("  .  "); continue
        out.append(f"{umeyama_scale(pos[w], g):5.2f}")
    print(f"{q}: " + " ".join(out))
