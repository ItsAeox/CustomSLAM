#!/usr/bin/env python3
"""One row per AR session log (results/browser/ar_*.log, web/ar.js): preset, length, fps and tracking time, losses
and how they ended, time spent lost, new maps, relocalisations, coasting recoveries, turn bridges, trees placed, and
for profiling runs (?prof=1) the last stage line. For comparing a phone round preset by preset.

  ar_runs_table.py <log> [<log> ...]
"""
import re
import statistics as st
import sys
from datetime import datetime
from pathlib import Path


def ts(line):
    m = re.match(r"(\d\d):(\d\d):(\d\d\.\d+)", line)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else None


def row(path):
    L = path.read_text(encoding="utf-8", errors="replace").splitlines()
    t = [x for x in map(ts, L) if x is not None]
    start = next((l for l in L if " start: " in l), "")
    preset = (re.search(r"preset (\S+)", start) or [None, "?"])[1]
    fps, med, p95 = [], [], []
    for l in L:
        m = re.search(r"frame \d+: .*?\| ([\d.]+) fps \| track ms med ([\d.]+) p95 ([\d.]+)", l)
        if m:
            fps.append(float(m.group(1))); med.append(float(m.group(2))); p95.append(float(m.group(3)))
    losses = sum(1 for l in L if re.search(r"state tracking( \(KLT\))? -> (recently lost|lost)", l))
    lost_s = [float(x) for x in re.findall(r"was lost ([\d.]+) s", "\n".join(L))]
    peaks = [float(x) for l in L if re.search(r"-> (recently lost|lost)", l) for x in re.findall(r"gyro peak ([\d.]+)", l)]
    # time without tracking: from each loss to the next recovery (or the end)
    down, t_loss = 0.0, None
    for l in L:
        if re.search(r"state \S.* -> (recently lost|lost)", l) and t_loss is None:
            t_loss = ts(l)
        elif re.search(r"state (recently lost|lost) -> tracking", l) and t_loss is not None:
            down += ts(l) - t_loss; t_loss = None
    end_lost = 0.0  # a final loss that never recovered
    if t_loss is not None and t:
        down += t[-1] - t_loss
        end_lost = t[-1] - t_loss
    klt = [float(x) for l in L for x in re.findall(r"KLT ms ([\d.]+) \|", l)]
    prof = [l for l in L if "tracking (ms, mean" in l]
    pr = ""
    if prof:
        m = re.search(r"ORB ([\d.]+) pose ([\d.]+) local map ([\d.]+).*local BA ([\d.]+).*total ([\d.]+) \|", prof[-1])
        if m:
            pr = f" | profile: ORB {m.group(1)} ms, local map {m.group(3)} ms, mapping {m.group(5)} ms/KF"
    cnt = lambda pat: sum(1 for l in L if re.search(pat, l))
    bridges = cnt(r"TURN bridge \d")
    dur = (t[-1] - t[0]) if len(t) > 1 else 0
    return (f"{path.name[3:22]} {preset:9s} {dur:5.0f}s  fps {st.median(fps) if fps else 0:4.1f}  track ms {st.median(med) if med else 0:5.1f}"
            f" (p95 worst {max(p95) if p95 else 0:5.0f})  losses {losses:2d}, lost {down:5.1f}s ({100 * down / dur if dur else 0:3.0f}%),"
            f" recovered {len(lost_s)} (med {st.median(lost_s) if lost_s else 0:4.1f}s max {max(lost_s) if lost_s else 0:5.1f}s),"
            f" ended lost for {end_lost:4.1f}s  KLT frame ms {klt[-1] if klt else 0:4.1f}"
            f"  gyro@loss med {st.median(peaks) if peaks else 0:4.0f}  maps {cnt(r'Creation of new map with id') :d}"
            f"  reloc {cnt(r'Relocalized')} coast {cnt(r'Recovered by coasting')} bridge {bridges}"
            f"  tree {cnt(r'tree placed:')}{pr}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(row(Path(p)))
