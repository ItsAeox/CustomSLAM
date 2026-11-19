#!/usr/bin/env python3
"""return_1104: is a far-off object (floor placement, anchor_floor.txt px_med) a collapsed map (COLLAPSE.md, eval/scale_collapse.py)?
Per batch and scenario, the placed runs split by the collapse flag: runs, median px, runs > 50 / > 15 px.
Usage: collapse_px_1104.py <results dir, e.g. H:/Research/results>  (prints; return_1104/COLLAPSE_PX.md keeps the output)
"""
import re
import statistics as st
import sys
from pathlib import Path

root = Path(sys.argv[1])
for d in ("return_1104", "return_1104_phone"):
    for sc in ("flat", "flatmap", "flatfirst"):
        g = {True: [], False: []}
        for l in (root / d / "COLLAPSE.md").read_text(encoding="utf-8").splitlines():
            m = re.match(rf"^({sc}\d) (\S+) (seed\d+): .*", l)
            if not m:
                continue
            f = root / d / m[1] / m[2] / m[3] / "anchor_floor.txt"
            kv = dict(re.findall(r"(\w+)=(\S+)", f.read_text(encoding="utf-8")))
            if kv.get("placed") != "1" or kv.get("px_med", "-") == "-":
                continue
            g["COLLAPSE" in l].append(float(kv["px_med"]))
        for c in (True, False):
            v = g[c]
            if v:
                print(f"{d} {sc} {'collapsed' if c else 'not collapsed'}: runs {len(v)}, px median {st.median(v):.1f}, "
                      f"> 50 px {sum(x > 50 for x in v)}, > 15 px {sum(x > 15 for x in v)}")
