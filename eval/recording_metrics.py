#!/usr/bin/env python3
"""Score replays of a phone recording (record.html; no ground truth) for comparing presets on the same movement.

Per run directory (orbslam3_run --out): frames tracked / relocalising / before the first map (track_times state column),
number of losses and recoveries, maps, tracking time (mean, p95), and the return-to-start check: the recording protocol
starts and ends at the same spot with the phone still, so the distance between the first and the last tracked camera
position of the same map, relative to the path length, is the drift (scale-free; rotation drift in degrees too).

Usage: recording_metrics.py <replay dir>   (layout: <replay dir>/<preset>/seed<k>/)
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot


def run_metrics(d: Path):
    tt = [l.split() for l in (d / "track_times.txt").read_text().splitlines() if len(l.split()) >= 3] if (d / "track_times.txt").exists() else []  # a killed run can leave a half-written line
    st = np.array([int(x[2]) for x in tt]) if tt else np.array([])
    ms = np.array([float(x[1]) for x in tt]) if tt else np.array([])
    ok = np.isin(st, [2, 5])
    lost = np.isin(st, [3, 4])
    losses = int(np.sum(ok[:-1] & ~ok[1:])) if len(ok) > 1 else 0
    recov = int(np.sum(~ok[:-1] & ok[1:] & (np.cumsum(ok)[:-1] > 0))) if len(ok) > 1 else 0
    first = int(np.argmax(ok)) if ok.any() else len(ok)
    log = (d / "run.log").read_text(errors="ignore") if (d / "run.log").exists() else ""
    out = dict(frames=len(st), tracked=ok.mean() if len(st) else 0, lost=lost.mean() if len(st) else 0,
               before_map=first, losses=losses, recoveries=recov, maps=log.count("Creation of new map") / 2,
               ms_mean=ms.mean() if len(ms) else 0, ms_p95=np.percentile(ms, 95) if len(ms) else 0,
               drift=np.nan, drift_deg=np.nan)
    # return-to-start: the trajectory file of the largest map (traj_tum.txt), first vs last pose
    tr = d / "traj_tum.txt"
    if tr.exists() and tr.stat().st_size:
        p = np.loadtxt(tr, comments="#", ndmin=2)
        if len(p) > 10:
            path = np.sum(np.linalg.norm(np.diff(p[:, 1:4], axis=0), axis=1))
            out["drift"] = np.linalg.norm(p[-1, 1:4] - p[0, 1:4]) / max(path, 1e-9)
            out["drift_deg"] = np.degrees((Rot.from_quat(p[0, 4:8]).inv() * Rot.from_quat(p[-1, 4:8])).magnitude())
    return out


def main():
    root = Path(sys.argv[1])
    print(f"{root.name}: per preset, mean over seeds (drift = |end - start| / path length; start and end at the same spot)")
    print(f"{'preset':14s} {'runs':>4s} {'tracked':>8s} {'lost':>6s} {'losses':>6s} {'recov':>6s} {'maps':>5s} {'1st map':>7s} {'ms':>6s} {'p95':>6s} {'drift':>6s} {'deg':>6s}")
    for pre in sorted(p for p in root.iterdir() if p.is_dir()):
        runs = [run_metrics(s) for s in sorted(pre.iterdir()) if s.is_dir()]
        if not runs:
            continue
        m = lambda k: np.nanmean([r[k] for r in runs])
        print(f"{pre.name:14s} {len(runs):4d} {m('tracked'):8.3f} {m('lost'):6.3f} {m('losses'):6.1f} {m('recoveries'):6.1f} "
              f"{m('maps'):5.2f} {m('before_map'):7.1f} {m('ms_mean'):6.1f} {m('ms_p95'):6.1f} {m('drift'):6.3f} {m('drift_deg'):6.1f}")


if __name__ == "__main__":
    main()
