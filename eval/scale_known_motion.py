#!/usr/bin/env python3
"""True metric scale of a phone run from a known back-and-forth motion (claude/PHONE.md "Scale recordings 21 Oct": the
phone slid between two marks a known distance apart, e.g. 1.00 m along a table edge), and the online estimate against it.

The tracked camera positions (trace.txt columns 3-5, map units) of the largest map are projected on their main axis
(PCA over --t0..--t1); the turning points of that coordinate (local extrema at least --min-gap s apart that move at
least --min-frac of the range) are the marks; the median distance between consecutive turning points = the known
distance in map units -> true metres per map unit. Then the online scale (columns 26-28) over time against it.

  scale_known_motion.py <run dir> <recording dir> [--dist 1.0] [--t0 0] [--t1 1e9] [--min-gap 0.8] [--min-frac 0.5]
"""
import argparse
from pathlib import Path

import numpy as np


def turning_points(tt, x, min_gap, min_frac):
    """Main-axis coordinate x(tt) smoothed over +-2 frames, its turning points (local extrema at least min_gap s apart
    that move at least min_frac of the range) as (index, kind) and the distances between consecutive ones."""
    xs = np.array([x[max(0, i - 2):i + 3].mean() for i in range(len(x))])
    rng = xs.max() - xs.min()
    ext = []  # (index, kind)
    i = 0
    while i < len(xs):
        w = (tt >= tt[i] - min_gap / 2) & (tt <= tt[i] + min_gap / 2)
        if xs[i] == xs[w].max() or xs[i] == xs[w].min():
            kind = 1 if xs[i] == xs[w].max() else -1
            if ext and ext[-1][1] == kind:  # same kind: keep the more extreme
                if kind * xs[i] > kind * xs[ext[-1][0]]:
                    ext[-1] = (i, kind)
            elif not ext or abs(xs[i] - xs[ext[-1][0]]) >= min_frac * rng:
                ext.append((i, kind))
        i += 1
    amps = np.array([abs(xs[ext[k + 1][0]] - xs[ext[k][0]]) for k in range(len(ext) - 1)])
    return xs, ext, amps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("rec")
    ap.add_argument("--dist", type=float, default=1.0)
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=1e9)
    ap.add_argument("--min-gap", type=float, default=0.8)
    ap.add_argument("--min-frac", type=float, default=0.5)
    ap.add_argument("--timeline", action="store_true", help="print the main-axis coordinate every 0.5 s")
    a = ap.parse_args()
    rows = [l.split() for l in (Path(a.run) / "trace.txt").read_text().splitlines()]
    tr = np.array([[float(x) for x in r] for r in rows if len(r) >= 29])
    cam = np.loadtxt(Path(a.rec) / "mav0/cam0/data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64)
    fr = tr[:, 0].astype(int)
    t = (cam[np.clip(fr, 0, len(cam) - 1)] - cam[0]) * 1e-9
    ok = np.isin(tr[:, 1].astype(int), [2, 5])
    pos, kfs, s, rel, win = tr[:, 3:6], tr[:, 13], tr[:, 26], tr[:, 27], tr[:, 28]
    mapid = np.cumsum(np.concatenate([[0], (kfs[1:] == 0) & (kfs[:-1] > 0)]))
    maps = [(np.sum(ok & (mapid == m)), m) for m in np.unique(mapid)]
    n_ok, m = max(maps)
    sel = ok & (mapid == m) & (t >= a.t0) & (t <= a.t1)
    print(f"{a.run}: tracked {ok.mean() * 100:.0f} % of {len(t)} frames, {len(maps)} map(s); largest map {m}: "
          f"{n_ok} tracked frames, t {t[ok & (mapid == m)][0]:.1f}-{t[ok & (mapid == m)][-1]:.1f} s")
    p, tt = pos[sel], t[sel]
    c = p - p.mean(0)
    U, S, Vt = np.linalg.svd(c, full_matrices=False)
    x = c @ Vt[0]
    print(f"  main axis {np.round(Vt[0], 3)}: spread 1-sigma {S[0] / np.sqrt(len(c)):.3f} / {S[1] / np.sqrt(len(c)):.3f} / "
          f"{S[2] / np.sqrt(len(c)):.3f} units (axes 1 / 2 / 3)")
    xs, ext, amps = turning_points(tt, x, a.min_gap, a.min_frac)
    if a.timeline:
        for k in range(0, len(tt), max(1, int(0.5 / np.median(np.diff(tt))))):
            print(f"    t {tt[k]:6.2f}  x {xs[k]:+.3f}")
    print(f"  turning points: {len(ext)} at t = {', '.join(f'{tt[i]:.1f}' for i, _ in ext)}")
    if not len(amps):
        print("  no back-and-forth found")
        return
    unit = np.median(amps)
    s_true = a.dist / unit
    print(f"  distance between turning points (units): {np.round(amps, 3).tolist()}")
    print(f"  -> {a.dist} m = {unit:.3f} units (median; spread {np.percentile(amps, 25):.3f}-{np.percentile(amps, 75):.3f})"
          f" -> TRUE SCALE {s_true:.3f} m per unit")
    # online estimate over time (same map)
    e = (mapid == m) & ok & (s > 0)
    if e.any():
        conv = e & (rel < 0.05)
        print(f"  online estimate: present {e.sum()} frames from t {t[e][0]:.1f} s; converged (<5 %) {conv.sum()} frames")
        for k in np.where(e)[0][:: max(1, e.sum() // 12)]:
            print(f"    t {t[k]:6.1f}  s {s[k]:.3f} ({s[k] / s_true:.2f}x true)  1-sigma {100 * rel[k]:.0f} %  windows {int(win[k])}")
    else:
        print("  online estimate: never valid in this map")


if __name__ == "__main__":
    main()
