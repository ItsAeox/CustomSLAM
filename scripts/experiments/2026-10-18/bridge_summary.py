#!/usr/bin/env python3
"""Turn bridge replays (replay_arms.sh with TRACE=1, arms_bridge.txt): one block per arm, per difficulty level (labels.txt)
and over all recordings, mean over runs: share of frames tracked, maps per run ("Creation of new map" / 2 as
summary_presets), seconds lost per run, loss episodes (median length, share that ended tracking again in the same
map), turn bridges per run, frames after a bridge tracking mostly points from before it (back on the original map),
tracking ms on lost frames (the bridge's matching and two-view solve run there). Reads run.log, track_times.txt and
anatomy.txt (eval/loss_anatomy.py) of every run.

  bridge_summary.py <replays dir> <labels.txt>
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "eval"))
from recording_metrics import run_metrics  # noqa: E402

root, labels = Path(sys.argv[1]), Path(sys.argv[2])
lv = {}
for line in labels.read_text().splitlines():
    if line.strip() and not line.startswith("#"):
        f = line.split()
        lv[f[0]] = f[1].split("-")[0]
levels = ["easy", "medium", "hard", "turning", "revisit"]
arms = sorted({p.name for r in lv if (root / r).is_dir() for p in (root / r).iterdir() if p.is_dir() and p.name != "settings"})


def run(d):
    m = run_metrics(d)
    log = (d / "run.log").read_text(errors="ignore") if (d / "run.log").exists() else ""
    b = re.search(r"BRIDGE-STATS bridges=(\d+)", log)
    m["bridges"] = int(b.group(1)) if b else 0
    an = (d / "anatomy.txt").read_text(errors="ignore") if (d / "anatomy.txt").exists() else ""
    ep = [(float(x), how) for x, how in re.findall(r"lost ([\d.]+) s, then (.+)", an)]
    m["episodes"] = [x for x, how in ep if how != "end of recording"]
    m["same"] = [how == "tracking again in the same map" for x, how in ep if how != "end of recording"]
    s = re.search(r"lost ([\d.]+) s of ([\d.]+) s", an)
    m["lost_s"] = float(s.group(1)) if s else np.nan
    o = re.search(r"tracking mostly points from before it: (\d+)", an)
    m["back"] = int(o.group(1)) if o else 0
    tt = [l.split() for l in (d / "track_times.txt").read_text().splitlines() if len(l.split()) >= 3] if (d / "track_times.txt").exists() else []
    lm = [float(x[1]) for x in tt if x[2] == "3"]
    m["lost_ms"] = np.mean(lm) if lm else np.nan
    return m


R = {a: {lvl: [] for lvl in levels} for a in arms}
for rec, level in lv.items():
    for a in arms:
        d = root / rec / a
        if d.is_dir():
            R[a][level] += [run(s) for s in sorted(d.iterdir()) if s.is_dir() and (s / "run_info.txt").exists()]  # finished runs only

print(f"{root.name}: mean over runs; tracked = share of frames, maps = maps per run (1 = never a new one), lost s = seconds without tracking per run (also before the first map)")
print(f"{'arm':22s} {'level':8s} {'runs':>4s} {'tracked':>7s} {'maps':>5s} {'lost s':>6s} {'episodes':>8s} {'median s':>8s} {'same map':>8s} "
      f"{'bridges':>7s} {'back':>5s} {'lost ms':>7s}")
for a in arms:
    for lvl in levels + ["all"]:
        rs = R[a][lvl] if lvl != "all" else [r for l in levels for r in R[a][l]]
        if not rs:
            continue
        ep = [x for r in rs for x in r["episodes"]]
        same = [x for r in rs for x in r["same"]]
        print(f"{a:22s} {lvl:8s} {len(rs):4d} {np.mean([r['tracked'] for r in rs]):7.2f} {np.mean([r['maps'] for r in rs]):5.2f} "
              f"{np.nanmean([r['lost_s'] for r in rs]):6.1f} {len(ep) / len(rs):8.1f} {np.median(ep) if ep else 0:8.1f} "
              f"{100 * np.mean(same) if same else 0:7.0f}% {np.mean([r['bridges'] for r in rs]):7.1f} "
              f"{np.mean([r['back'] for r in rs]):5.0f} {np.nanmean([r['lost_ms'] for r in rs]):7.1f}")
    print()
