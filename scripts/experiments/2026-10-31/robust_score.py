#!/usr/bin/env python3
"""ISSUES G32: filter estimate (replay output, converged frames) vs a robust true scale of the same map: the median of the Sim3 scales
of 8 s windows (>= 0.15 m of motion) of the map's tracked positions vs GT (the 30 s Sim3 of replay_score.py is corrupted by bad poses in shaking / whip pans).
  robust_score.py <replay dir> <live results dir> <seq glob> [arm]   (replay outputs <seq>__<arm>__<seed>.txt; arm default beta_scaleF)"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "H:/Research/CustomSLAM/eval")
from scale_from_accel import umeyama_scale
R, L, pat = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
arm = sys.argv[4] if len(sys.argv) > 4 else "beta_scaleF"
allr = {}
for f in sorted(R.glob(f"{pat}__{arm}__seed*.txt")):
    q, _, seed = f.stem.split("__")
    run = L / q / arm / seed
    tr = np.array([[float(x) for x in l.split()[:14]] for l in (run / "trace.txt").read_text().splitlines() if len(l.split()) >= 14])
    fl = np.loadtxt(f, comments="#"); n = min(len(tr), len(fl)); tr, fl = tr[:n], fl[:n]
    cam = np.loadtxt(f"H:/Research/datasets/synth/{q}/mav0/cam0/data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64)
    t = cam[np.clip(tr[:, 0].astype(int), 0, len(cam) - 1)] * 1e-9
    ok = np.isin(tr[:, 1].astype(int), [2, 5]); pos, kfs = tr[:, 3:6], tr[:, 13]
    mapid = np.cumsum(np.concatenate([[0], (kfs[1:] == 0) & (kfs[:-1] > 0)]))
    gt = np.loadtxt(f"H:/Research/results/gt/synth_{q}.txt", comments="#", ndmin=2)
    conv = ok & (fl[:, 1] > 0) & (fl[:, 2] >= 0) & (fl[:, 2] < 0.05)
    ratios = []
    for m in np.unique(mapid[conv]):
        ws = []
        for tc in np.arange(t[0] + 8, t[-1], 1.0):
            w = np.where(ok & (mapid == m) & (t > tc - 8) & (t <= tc))[0]
            w = w[(t[w] >= gt[0, 0]) & (t[w] <= gt[-1, 0])]
            if len(w) < 30: continue
            g = np.stack([np.interp(t[w], gt[:, 0], gt[:, j]) for j in (1, 2, 3)], 1)
            if np.ptp(g, 0).max() < 0.15: continue
            ws.append(umeyama_scale(pos[w], g))
        if len(ws) < 3: continue
        ref = np.median(ws)
        ratios += list(fl[conv & (mapid == m), 1] / ref)
    if ratios:
        r = np.array(ratios); key = q.rstrip("0123456789"); allr.setdefault(key, []).append(np.median(np.abs(r - 1)))
        print(f"{q} {seed}: est / robust true median {np.median(r):.3f} (p10 {np.percentile(r, 10):.3f}, p90 {np.percentile(r, 90):.3f}), > 10 % off {100 * np.mean(np.abs(r - 1) > 0.1):.0f} % of {len(r)} frames")
for k, v in allr.items():
    print(f"== {k}: runs {len(v)}, median of per-run median |error| {100 * np.median(v):.1f} %")
