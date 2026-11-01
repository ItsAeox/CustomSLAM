"""Why the kept map stops growing on the way out (ISSUES G22): per run of a batch on the flat scenarios, the tracking
losses after the tap (state recently lost / lost for >= 0.3 s; not initialised yet is not a loss) with what led into them, from the trace (orbslam3_run --trace) and the
true camera motion: motion-clock time and phase, tracked points 1 s / 0.5 s / 0.1 s before, textured 4x4 cells and those
without a tracked point just before, map keyframes, keyframes and new map points added in the 1 s before, mapping queue
and whether mapping accepted keyframes, keyframes due while mapping was busy (so far), true turn rate (deg/s) and walking speed (m/s) over the 0.5 s before, the loss length
and what ended it (same map = relocalised / recovered, new map, never). Then per arm: the median of each column at the
first loss in the walk. Usage: walk_loss.py <batch dir> <arm> [scenario=flat] [runs to list=6]"""
import sys, json, re, numpy as np
from pathlib import Path

root, arm = Path(sys.argv[1]), sys.argv[2]
SC = sys.argv[3] if len(sys.argv) > 3 else "flat"; NLIST = int(sys.argv[4]) if len(sys.argv) > 4 else 6
D = Path("H:/Research/datasets/synth"); G = Path("H:/Research/results/gt")


def quat_to_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


first, lines = [], []
for sd in sorted(root.glob(f"{SC}[0-9]*/{arm}/seed*")):
    seq = sd.parts[-3]
    if re.sub(r"\d+$", "", seq) != SC or not (sd / "trace.txt").exists(): continue
    rows = [l.split() for l in (sd / "trace.txt").read_text().splitlines()]
    rows = [r for r in rows if len(r) >= 34]
    if not rows: continue
    tr = np.array([[float(x) for x in r[:26]] + [float(r[-2]), float(r[-1])] for r in rows])
    fr, st, ntr = tr[:, 0].astype(int), tr[:, 1].astype(int), tr[:, 2]
    tex, unt, kfs, que, acc, busy, mid = tr[:, 18], tr[:, 19], tr[:, 13], tr[:, 15], tr[:, 16], tr[:, 25], tr[:, 26].astype(int)
    stamps = np.loadtxt(D / seq / "mav0/cam0/data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64) * 1e-9
    t = stamps[np.clip(fr, 0, len(stamps) - 1)]
    t0 = float(open(D / seq / "gt/camera_dense.csv").readlines()[1].split()[0])
    tm = t - t0
    ph = json.loads((D / seq / "meta.json").read_text())["synthetic"]["motion"]["phases_s"]
    gt = np.loadtxt(G / f"synth_{seq}.txt", comments="#", ndmin=2)
    gtm = gt[:, 0] - t0
    taps = [l for l in (D / seq / "touches.csv").read_text().splitlines() if l and l[0] != "#"]
    ttap = int(taps[0].split(",")[0]) * 1e-9 - t0 if taps else 0.5 * sum(ph["place"])

    def phase(x):
        best = "-"
        for k, (a, b) in ph.items():
            if k in ("out", "return"): continue
            if a <= x < b: best = k
        return best

    def motion(x):  # true turn rate and speed over [x-0.5, x]
        i0, i1 = np.searchsorted(gtm, [x - 0.5, x])
        i1 = min(i1, len(gt) - 1)
        if i1 <= i0: return np.nan, np.nan
        R0, R1 = quat_to_R(gt[i0, 4:8]), quat_to_R(gt[i1, 4:8])
        ang = np.degrees(np.arccos(np.clip((np.trace(R0.T @ R1) - 1) / 2, -1, 1)))
        dt = gtm[i1] - gtm[i0]
        return ang / dt, np.linalg.norm(gt[i1, 1:4] - gt[i0, 1:4]) / dt

    def at(x, col):  # value at the last trace frame at or before x
        j = np.searchsorted(tm, x, side="right") - 1
        return col[j] if j >= 0 else np.nan

    ok = (st != 3) & (st != 4)
    nmp = tr[:, 17]
    i, n = 0, len(st)
    seen_walk = False
    while i < n:
        if tm[i] <= ttap or ok[i]: i += 1; continue
        j = i
        while j < n and not ok[j]: j += 1
        dur = (tm[j] if j < n else tm[-1]) - tm[i]
        if dur < 0.3: i = j; continue
        tl = tm[i]
        end = "never" if j >= n else ("same map" if mid[j] == mid[i - 1] else "new map")
        w, v = motion(tl)
        rec = dict(seq=seq, seed=sd.name, t=tl, phase=phase(tl), n1=at(tl - 1, ntr), n05=at(tl - 0.5, ntr),
                   n01=at(tl - 0.1, ntr), dkf=kfs[i - 1] - at(tl - 1, kfs), dmp=nmp[i - 1] - at(tl - 1, nmp), tex=tex[i - 1], unt=unt[i - 1], kfs=kfs[i - 1], que=que[i - 1], acc=acc[i - 1],
                   busy=busy[i - 1], turn=w, speed=v, dur=dur, end=end)
        lines.append(rec)
        if rec["phase"] == "walk" and not seen_walk:
            seen_walk = True; first.append(rec)
        i = j

hdr = "| Seq | Seed | Loss at s | Phase | Tracked 1 s / 0.5 s / 0.1 s before | Textured cells / untracked | Map KFs | KFs / new points in the 1 s before | Queue, accepting | KFs due while busy (so far) | Turn deg/s | Speed m/s | Lost for s | Ended by |"
print(f"# Losses after the tap: {root.name} {arm} ({SC})\n\n{hdr}\n" + "|---" * 14 + "|")
shown = {}
for r in lines:
    k = (r["seq"], r["seed"])
    if len(shown) >= NLIST and k not in shown: continue
    shown[k] = 1
    print(f"| {r['seq']} | {r['seed']} | {r['t']:.1f} | {r['phase']} | {r['n1']:.0f} / {r['n05']:.0f} / {r['n01']:.0f} | "
          f"{r['tex']:.0f} / {r['unt']:.0f} | {r['kfs']:.0f} | {r['dkf']:.0f} / {r['dmp']:.0f} | {r['que']:.0f}, {r['acc']:.0f} | {r['busy']:.0f} | "
          f"{r['turn']:.0f} | {r['speed']:.2f} | {r['dur']:.1f} | {r['end']} |")
if first:
    def med(k): return np.nanmedian([r[k] for r in first])
    ends = {e: sum(r["end"] == e for r in first) for e in ("same map", "new map", "never")}
    print(f"\nFirst loss in the walk, {len(first)} runs (medians): at {med('t'):.1f} s (walk starts {ph['walk'][0]:.1f}), "
          f"tracked {med('n1'):.0f} / {med('n05'):.0f} / {med('n01'):.0f}, cells {med('tex'):.0f} / untracked "
          f"{med('unt'):.0f}, map KFs {med('kfs'):.0f}, added in the 1 s before {med('dkf'):.0f} KFs / {med('dmp'):.0f} points, queue {med('que'):.0f}, accepting {med('acc'):.0f}, "
          f"turn {med('turn'):.0f} deg/s, speed {med('speed'):.2f} m/s, lost {med('dur'):.1f} s; ended: {ends}")
    print(f"All losses after the tap: {len(lines)} in {len({(r['seq'], r['seed']) for r in lines})} runs; per phase: "
          + ", ".join(f"{p} {sum(r['phase'] == p for r in lines)}" for p in dict.fromkeys(r['phase'] for r in lines)))
