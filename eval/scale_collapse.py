#!/usr/bin/env python3
"""ISSUES G35: does a map's scale collapse (its camera stops translating while the real camera walks)? Per run, per map
(trace map id column, orbslam3_run aa3763f+): the Sim3 scale (metres per map unit) of every 3 s window of tracked
frames with >= 0.15 m of true camera motion; the map's spread = largest / smallest window scale. Healthy maps stay
within ~1.5x; a collapse shows as 5-500x (the map's camera barely moves). Per run: the largest spread over its maps
with >= 3 windows, and whether it exceeds --ratio (default 5). A map merge (the atlas map count drops) re-expresses the
surviving map in the merged map's coordinates under the same id, so each map is split at merges (1 Nov: before, every
merged run read as a collapse).
  scale_collapse.py <results dir> [--ratio 5]   (<dir>/<seq>/<arm>/seed<N>/trace.txt, synthetic sequences)
Prints per-run lines and a table per scenario and arm.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_from_accel import umeyama_scale

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--ratio", type=float, default=5.0)
ap.add_argument("--data", default="H:/Research/datasets/synth")
ap.add_argument("--gt", default="H:/Research/results/gt")
a = ap.parse_args()
rows = {}
for tf in sorted(Path(a.root).glob("*/*/seed*/trace.txt")):
    run = tf.parent
    if not (run / "run_info.txt").exists():
        continue
    seq, arm = run.parts[-3], run.parts[-2]
    tr = np.array([[float(x) for x in l.split()[:6]] + [float(l.split()[-2]), float(l.split()[-1])] for l in tf.read_text().splitlines() if len(l.split()) >= 34])
    if not len(tr):
        continue
    st = np.loadtxt(f"{a.data}/{seq}/mav0/cam0/data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64)
    t = st[np.clip(tr[:, 0].astype(int), 0, len(st) - 1)] * 1e-9
    gt = np.loadtxt(f"{a.gt}/synth_{seq}.txt", comments="#")
    g = gt[np.clip(np.searchsorted(gt[:, 0], t - 1e-6), 0, len(gt) - 1), 1:4]
    ok = np.isin(tr[:, 1].astype(int), [2, 5])
    epoch = np.concatenate([[0], np.cumsum(np.diff(tr[:, 7]) < 0)])  # +1 at every merge
    worst, nwin = 1.0, 0
    for m, e in sorted(set(zip(tr[ok, 6], epoch[ok]))):
        sel = ok & (tr[:, 6] == m) & (epoch == e)
        ts = t[sel]
        sc = []
        for w0 in np.arange(ts[0], ts[-1] - 1.5, 1.5):
            w = sel & (t >= w0) & (t < w0 + 3.0)
            if w.sum() < 20 or np.linalg.norm(np.ptp(g[w], 0)) < 0.15:
                continue
            s = umeyama_scale(tr[w, 3:6], g[w])
            if np.isfinite(s) and s > 0:
                sc.append(s)
        if len(sc) >= 3:
            nwin += len(sc)
            worst = max(worst, max(sc) / min(sc))
    rows.setdefault((re.sub(r"\d+$", "", seq), arm), []).append((seq, run.parts[-1], worst, nwin))
    print(f"{seq} {arm} {run.parts[-1]}: spread {worst:.1f}x over {nwin} windows{' COLLAPSE' if worst > a.ratio else ''}")
print(f"\n| Scenario | Arm | Runs | Collapsed (spread > {a.ratio:g}x) | Spread median | Spread per run |")
print("|---|---|---|---|---|---|")
for (sc, arm), v in sorted(rows.items()):
    sp = [x[2] for x in v if x[3] > 0]
    print(f"| {sc} | {arm} | {len(v)} | {sum(1 for x in sp if x > a.ratio)} | {np.median(sp) if sp else float('nan'):.1f} | "
          f"{' '.join(f'{x:.0f}' for x in sorted(sp))} |")
