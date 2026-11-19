#!/usr/bin/env python3
"""Score a scale_filter_replay output (t s 1-sigma tracked map) of a synthetic run against the ground truth as
eval/scale_online_check.py does (Sim3 of the last 30 s of the map's live positions, >= 30 cm of motion), converged =
reported 1-sigma < 5 %. Prints: converged frames, first after s, median / p90 error, last error.
  replay_score.py <run dir> <seq name> <replay output>
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "eval"))
from scale_from_accel import umeyama_scale  # noqa: E402

run, seq, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
tr = np.array([[float(x) for x in l.split()[:14]] for l in (run / "trace.txt").read_text().splitlines() if len(l.split()) >= 14])
f = np.loadtxt(out, comments="#")
n = min(len(tr), len(f))
tr, f = tr[:n], f[:n]
cam = np.loadtxt(f"H:/Research/datasets/synth/{seq}/mav0/cam0/data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64)
t = cam[np.clip(tr[:, 0].astype(int), 0, len(cam) - 1)] * 1e-9
ok = np.isin(tr[:, 1].astype(int), [2, 5])
pos, kfs = tr[:, 3:6], tr[:, 13]
mapid = np.cumsum(np.concatenate([[0], (kfs[1:] == 0) & (kfs[:-1] > 0)]))
s, sd = f[:, 1], f[:, 2]
gt = np.loadtxt(f"H:/Research/results/gt/synth_{seq}.txt", comments="#", ndmin=2)
conv = ok & (s > 0) & (sd >= 0) & (sd < 0.05)
errs = []
for k in np.where(conv)[0][::5]:
    w = np.where(ok & (mapid == mapid[k]) & (t >= t[k] - 30) & (t <= t[k]))[0]
    w = w[(t[w] >= gt[0, 0]) & (t[w] <= gt[-1, 0])]
    if len(w) < 30:
        continue
    g = np.stack([np.interp(t[w], gt[:, 0], gt[:, j]) for j in (1, 2, 3)], 1)
    if np.ptp(g, 0).max() < 0.3:
        continue
    errs.append(abs(s[k] / umeyama_scale(pos[w], g) - 1))
e = np.array(errs)
first = t[np.where(conv)[0][0]] - t[np.where(ok)[0][0]] if conv.any() else None
print(f"{out}: converged {conv.sum()} frames" + (f", first after {first:.1f} s" if first is not None else "") +
      (f", error median {100 * np.median(e):.1f} % p90 {100 * np.percentile(e, 90):.1f} % last {100 * e[-1]:.1f} % ({len(e)} checks, {int((e > 0.1).sum())} over 10 %)" if len(e) else ", no error checks"))
