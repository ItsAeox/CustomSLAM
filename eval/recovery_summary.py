#!/usr/bin/env python3
"""Aggregate recovery_anatomy.py output (recovery_all.sh): per loss outcome, blur duration, lag after sharp frames and
where the camera pointed (gyro angle from the last tracked view when frames were sharp again).
  recovery_summary.py <recovery_all output> [level ...]   (levels from labels.txt, default all)"""
import re
import sys

import numpy as np

text = open(sys.argv[1]).read().splitlines()
want = set(sys.argv[2:])
L = []
level = None
for l in text:
    m = re.match(r"== \S+ seed \d \((\S+)\)", l)
    if m:
        level = m.group(1)
        continue
    m = re.match(r"  loss t ([\d.]+) s, lost ([\d.]+) s \(([^)]+)\): sharpness dip x([\d.]+|nan), sharp again after (\S+) s \(turned (\S+) deg, still (\S+) deg/s\), lag to recovery (\S+) s; at recovery turned (\S+) deg", l)
    if m and (not want or level in want):
        v = lambda s: None if s == "-" else float(s)
        L.append(dict(level=level, dur=float(m.group(2)), end=m.group(3), to_sharp=v(m.group(5)), turn=v(m.group(6)),
                      rate=v(m.group(7)), lag=v(m.group(8)), turn_rec=v(m.group(9))))
n = len(L)
same = [x for x in L if x["end"] == "same map"]
print(f"{n} losses: same map {len(same)}, new map / reset {sum(x['end'] == 'new map / reset' for x in L)}, "
      f"end of recording {sum(x['end'] == 'end of recording' for x in L)}; never sharp again during the loss "
      f"{sum(x['to_sharp'] is None for x in L)}")
ts = [x["to_sharp"] for x in L if x["to_sharp"] is not None]
print(f"time to sharp frames after the loss: median {np.median(ts):.2f} s, under 0.3 s {sum(t < 0.3 for t in ts)}/{len(ts)}, "
      f"over 1 s {sum(t > 1 for t in ts)}/{len(ts)}")
lags = [x for x in same if x["lag"] is not None]
for name, sel in [("facing the last view (turned < 30 deg when sharp again)", [x for x in lags if x["turn"] < 30]),
                  ("turned away (>= 30 deg when sharp again)", [x for x in lags if x["turn"] >= 30])]:
    if not sel:
        print(f"same-map recoveries, {name}: none")
        continue
    lg = [x["lag"] for x in sel]
    tr = [x["turn_rec"] for x in sel if x["turn_rec"] is not None]
    print(f"same-map recoveries, {name}: {len(sel)}; lag sharp -> tracking median {np.median(lg):.2f} s, under 0.3 s "
          f"{sum(t < 0.3 for t in lg)}, over 1 s {sum(t > 1 for t in lg)}; turned at recovery median {np.median(tr):.0f} deg")
lost_time = sum(x["dur"] for x in L)
for name, sel in [("never sharp again", [x for x in L if x["to_sharp"] is None]),
                  ("sharp, facing the last view", [x for x in L if x["to_sharp"] is not None and x["turn"] < 30]),
                  ("sharp, turned away", [x for x in L if x["to_sharp"] is not None and x["turn"] >= 30])]:
    print(f"lost time {name}: {sum(x['dur'] for x in sel):.0f} s of {lost_time:.0f} s ({len(sel)} losses)")
