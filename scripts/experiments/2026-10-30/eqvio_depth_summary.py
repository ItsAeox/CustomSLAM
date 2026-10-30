#!/usr/bin/env python3
"""Compact table of eqvio_depth_sweep.sh (after eqvio_checks_score.py has scored every results/eqvio_depth_1030/d<depth>):
per start depth, the real recordings (1 m slides: distance between turning points / 1 m; the others: diverged = more
than 20 m from the start) and the synthetic set (phone IMU: diverged runs, median SE3 ATE and Sim3 scale of the rest).
Also the base EuRoC config (5 m) from results/eqvio_checks_1030 when given.
  eqvio_depth_summary.py <results/eqvio_depth_1030> [<results/eqvio_checks_1030>]
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

R = Path(sys.argv[1])
dirs = sorted(R.glob("d*"), key=lambda p: float(p.name[1:]))
if len(sys.argv) > 2:
    dirs.append(Path(sys.argv[2]))
EVAL = Path(__file__).resolve().parents[3] / "eval"
print("| Start depth | 1 m slides (29 Oct) | slow 1 m slides (21 Oct) | other recordings diverged (> 20 m) | synthetic: diverged / runs | median SE3 ATE | median scale |")
print("|---|---|---|---|---|---|---|")
for d in dirs:
    label = d.name[1:] + " m" if d.name.startswith("d") else "5 m (EuRoC config)"
    slides = []
    for rec in ("rec_20261029_202938", "rec_20261021_070221"):
        f = d / rec / "traj_camclock.txt"
        out = subprocess.run([sys.executable, str(EVAL / "slides_metric.py"), str(f)], capture_output=True, text=True, cwd=EVAL).stdout if f.exists() else ""
        slides.append(out.split("->")[-1].split("the true")[0].strip() if "->" in out else ("no back-and-forth" if out else "-"))
    div_real, n_real = 0, 0
    for o in sorted(d.glob("rec_*")):
        if o.name in ("rec_20261029_202938", "rec_20261021_070221") or "_k" in o.name or not (o / "traj_camclock.txt").exists():
            continue
        p = np.loadtxt(o / "traj_camclock.txt", comments="#", ndmin=2)[:, 1:4]
        n_real += 1
        div_real += np.linalg.norm(p - p[0], axis=1).max() > 20
    ate, sc, div, n = [], [], 0, 0
    for sc_ in ("ar", "gesture", "orbit", "rotate", "shake", "slide", "walk"):
        for v in range(6):
            m = d / f"{sc_}{v}" / "metrics.json"
            if not m.exists():
                continue
            j = json.loads(m.read_text())
            n += 1
            if j["ate_se3"]["rmse"] > 1.0:
                div += 1
            else:
                ate.append(j["ate_se3"]["rmse"])
                sc.append(j["sim3_scale"])
    print(f"| {label} | {slides[0]} | {slides[1]} | {div_real} / {n_real} | {div} / {n} | "
          f"{100 * np.median(ate):.1f} cm | {np.median(sc):.2f} |" if ate else f"| {label} | {slides[0]} | {slides[1]} | {div_real} / {n_real} | {div} / {n} | - | - |")
