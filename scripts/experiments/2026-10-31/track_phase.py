"""Tracked % per motion phase (meta.json phases_s) per scenario and arm, medians over the runs of a batch (trace.txt:
state 2 = OK, 5 = OK after relocalisation). The walk phase is the way out and back through the flat: the map can only
grow where tracking holds. Usage: track_phase.py <batch dir> <scenario> [arm ...]"""
import sys, json, re, numpy as np
from pathlib import Path
from collections import defaultdict

root, sc = Path(sys.argv[1]), sys.argv[2]; arms = sys.argv[3:]
D = Path("H:/Research/datasets/synth")
PH = ("look", "survey", "walk", "view", "toStart", "final")
res = defaultdict(lambda: defaultdict(list))
for sd in sorted(root.glob(f"{sc}[0-9]/*/seed*")):
    seq, arm = sd.parts[-3], sd.parts[-2]
    if arms and arm not in arms: continue
    tf = sd / "trace.txt"
    if not tf.exists(): continue
    rows = [r.split() for r in tf.read_text().splitlines() if r and r[0] != "#"]
    rows = [r for r in rows if len(r) >= 34]
    if not rows: continue
    fr = np.array([int(r[0]) for r in rows]); st = np.array([int(r[1]) for r in rows])
    stamps = np.loadtxt(D / seq / "mav0/cam0/data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64)
    t0 = float(open(D / seq / "gt/camera_dense.csv").readlines()[1].split()[0])
    t = stamps[np.clip(fr, 0, len(stamps) - 1)] * 1e-9 - t0
    ph = json.loads((D / seq / "meta.json").read_text())["synthetic"]["motion"]["phases_s"]
    for name in PH:
        if name not in ph: continue
        m = (t >= ph[name][0]) & (t < ph[name][1])
        if m.any(): res[arm][name].append(100 * np.isin(st[m], [2, 5]).mean())
print(f"## {sc}: tracked % per phase (median over runs; min-max)\n")
print("| Arm | Runs | " + " | ".join(n for n in PH) + " |")
print("|---|---|" + "---|" * len(PH))
for arm, d in sorted(res.items()):
    cells = [f"{np.median(d[n]):.0f} ({min(d[n]):.0f}-{max(d[n]):.0f})" if d.get(n) else "-" for n in PH]
    print(f"| {arm} | {len(d['walk'])} | " + " | ".join(cells) + " |")
