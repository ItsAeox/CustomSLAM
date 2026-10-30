#!/usr/bin/env python3
"""Per-scenario summary of replay_all.sh outputs (the final filter on scaleF_1030's live trajectories), scored with
replay_score.py's method: runs that converged, median of per-run median errors, median first-converged time.
  replay_summary.py <replay dir> <live results dir>
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

R, L = Path(sys.argv[1]), Path(sys.argv[2])
S = Path(__file__).with_name("replay_score.py")
rows = {}
for f in sorted(R.glob("*.txt")):
    q, arm, seed = f.stem.split("__") if f.stem.count("__") == 2 else (None, None, None)
    if q is None:  # <seq>__imuideal__<arm>__<seed>
        parts = f.stem.split("__")
        q, arm, seed = "__".join(parts[:2]), parts[2], parts[3]
    line = subprocess.run([sys.executable, str(S), str(L / q / arm / seed), q, str(f)], capture_output=True, text=True).stdout
    num = lambda p: float(re.search(p, line).group(1)) if re.search(p, line) else None
    key = (re.match(r"[a-z]+", q).group(0), "ideal" if "__imuideal" in q else ("clean" if "_clean" in q else "phone"))
    rows.setdefault(key, []).append((num(r"converged (\d+) frames") or 0, num(r"error median ([\d.]+) %"), num(r"first after ([\d.]+) s")))
print("| Scenario | IMU | Runs | Converged (runs) | Error median % | First after s |\n|---|---|---|---|---|---|")
for k in sorted(rows):
    rs = rows[k]
    e = [x[1] for x in rs if x[1] is not None]
    fa = [x[2] for x in rs if x[2] is not None]
    print(f"| {k[0]} | {k[1]} | {len(rs)} | {sum(1 for x in rs if x[0] > 0)} | {np.median(e):.1f} | {np.median(fa):.1f} |" if e else
          f"| {k[0]} | {k[1]} | {len(rs)} | {sum(1 for x in rs if x[0] > 0)} | - | {np.median(fa) if fa else float('nan'):.1f} |")
