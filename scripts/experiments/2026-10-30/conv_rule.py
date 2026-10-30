#!/usr/bin/env python3
"""Convergence rules for the online scale (trace columns 26 s, 27 1-sigma) of a synthetic results folder, against the
ground truth as eval/scale_online_check.py (Sim3 of the last 30 s of the map's live positions, >= 30 cm of motion):
  basic   1-sigma < 5 %
  stable  basic and the estimate within --tol of its value --back s earlier (same map, also converged then)
Per scenario and IMU: runs that converged, median of the per-run median errors, median time to the first converged frame,
and the share of the checked converged frames that were more than 10 % off (confident wrong claims).
  conv_rule.py <results dir> <arm substring> [--tol 0.03] [--back 3]
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "eval"))
from scale_from_accel import umeyama_scale  # noqa: E402

a = sys.argv[1:]
R, ARM = Path(a[0]), a[1]
TOL = float(a[a.index("--tol") + 1]) if "--tol" in a else 0.03
BACK = float(a[a.index("--back") + 1]) if "--back" in a else 3.0
D, G = Path("H:/Research/datasets/synth"), Path("H:/Research/results/gt")
rows = {}
for run in sorted(R.glob(f"*/*{ARM}*/seed*")):
    if not (run / "trace.txt").exists():
        continue
    seq = run.parts[-3]
    tr = np.array([[float(x) for x in l.split()[:29]] for l in (run / "trace.txt").read_text().splitlines() if len(l.split()) >= 29])
    cam = np.loadtxt(D / seq / "mav0" / "cam0" / "data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64)
    t = cam[np.clip(tr[:, 0].astype(int), 0, len(cam) - 1)] * 1e-9
    ok = np.isin(tr[:, 1].astype(int), [2, 5])
    pos, kfs, s, rel = tr[:, 3:6], tr[:, 13], tr[:, 26], tr[:, 27]
    mapid = np.cumsum(np.concatenate([[0], (kfs[1:] == 0) & (kfs[:-1] > 0)]))
    gt = np.loadtxt(G / f"synth_{seq}.txt", comments="#", ndmin=2)
    basic = ok & (s > 0) & (rel >= 0) & (rel < 0.05)
    stable = basic.copy()
    for k in np.where(basic)[0]:
        j = np.searchsorted(t, t[k] - BACK)
        stable[k] = j < k and mapid[j] == mapid[k] and basic[j] and abs(s[k] / s[j] - 1) < TOL
    res = {}
    for name, conv in (("basic", basic), ("stable", stable)):
        errs, first = [], None
        for k in np.where(conv)[0][::5]:
            w = np.where(ok & (mapid == mapid[k]) & (t >= t[k] - 30) & (t <= t[k]))[0]
            w = w[(t[w] >= gt[0, 0]) & (t[w] <= gt[-1, 0])]
            if len(w) < 30:
                continue
            g = np.stack([np.interp(t[w], gt[:, 0], gt[:, j]) for j in (1, 2, 3)], 1)
            if np.ptp(g, 0).max() < 0.3:
                continue
            errs.append(abs(s[k] / umeyama_scale(pos[w], g) - 1))
        if conv.any():
            m0 = np.where(ok & (mapid == mapid[np.where(conv)[0][0]]))[0][0]
            first = t[np.where(conv)[0][0]] - t[m0]
        res[name] = (conv.any(), np.median(errs) if errs else None, first, sum(e > 0.1 for e in errs), len(errs))
    key = (re.match(r"[a-z]+", seq).group(0), "ideal" if "__imuideal" in seq else ("clean" if "_clean" in seq else "phone"))
    rows.setdefault(key, []).append(res)
print(f"rule 'stable': within {100 * TOL:.0f} % of the value {BACK:.0f} s earlier")
print("| Scenario | IMU | Runs | basic: converged, err median %, first after s, > 10 % off | stable: converged, err median %, first after s, > 10 % off |\n|---|---|---|---|---|")
for k in sorted(rows):
    cells = []
    for name in ("basic", "stable"):
        rs = [r[name] for r in rows[k]]
        e = [x[1] for x in rs if x[1] is not None]
        f = [x[2] for x in rs if x[2] is not None]
        bad, chk = sum(x[3] for x in rs), sum(x[4] for x in rs)
        off = f"{100 * bad / chk:.0f} % of {chk}" if chk else "-"
        cells.append(f"{sum(1 for x in rs if x[0])}, {np.median(e) * 100:.1f}, {np.median(f):.1f}, {off}" if e else
                     f"{sum(1 for x in rs if x[0])}, -, {np.median(f):.1f}, -" if f else "0, -, -, -")
    print(f"| {k[0]} | {k[1]} | {len(rows[k])} | {cells[0]} | {cells[1]} |")
