"""Are local map points in the view matched? (ISSUES G22, ee32f06+ traces). Over the ORB frames after the tap of a
batch's runs (one arm, one scenario), the local map points in the view and the share matched by projection, binned by
the true turn rate (deg/s over the 0.2 s before) and by whether a tracking loss follows within 1 s; plus the tracked
points from the 10 newest keyframes; with 40-column traces (the runner after df7b5b2) also the whole current map in the
view: points projecting into the image, those passing the frustum test's angle / distance checks, and the 4x4 cells
they cover (do the new view's points exist anywhere?). Usage: local_match.py <batch dir> <arm> [scenario=flat]"""
import sys, re, numpy as np
from pathlib import Path

root, arm = Path(sys.argv[1]), sys.argv[2]
SC = sys.argv[3] if len(sys.argv) > 3 else "flat"
D = Path("H:/Research/datasets/synth"); G = Path("H:/Research/results/gt")


def quat_to_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


BINS = [(0, 20), (20, 50), (50, 100), (100, 1e9)]
acc = {(b, pre): [] for b in range(len(BINS)) for pre in (0, 1)}
nruns = 0
for sd in sorted(root.glob(f"{SC}[0-9]*/{arm}/seed*")):
    seq = sd.parts[-3]
    if re.sub(r"\d+$", "", seq) != SC or not (sd / "trace.txt").exists(): continue
    rows = [l.split() for l in (sd / "trace.txt").read_text().splitlines()]
    rows = [r for r in rows if len(r) >= 37]
    if not rows: continue
    nruns += 1
    fr = np.array([int(r[0]) for r in rows]); st = np.array([int(r[1]) for r in rows])
    dg = np.array([[float(r[-5]), float(r[-4]), float(r[-3])] + ([float(r[-8]), float(r[-7]), float(r[-6])] if len(r) >= 40 else [np.nan] * 3)
                   for r in rows])
    stamps = np.loadtxt(D / seq / "mav0/cam0/data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64) * 1e-9
    t0 = float(open(D / seq / "gt/camera_dense.csv").readlines()[1].split()[0])
    tm = stamps[np.clip(fr, 0, len(stamps) - 1)] - t0
    taps = [l for l in (D / seq / "touches.csv").read_text().splitlines() if l and l[0] != "#"]
    ttap = int(taps[0].split(",")[0]) * 1e-9 - t0 if taps else 0
    gt = np.loadtxt(G / f"synth_{seq}.txt", comments="#", ndmin=2); gtm = gt[:, 0] - t0
    lost = (st == 3) | (st == 4)
    loss_t = tm[1:][lost[1:] & ~lost[:-1]]  # loss onsets
    for i in np.where((tm > ttap) & (st == 2) & (dg[:, 0] > 0))[0]:
        g0, g1 = np.searchsorted(gtm, [tm[i] - 0.2, tm[i]]); g1 = min(g1, len(gt) - 1)
        if g1 <= g0: continue
        R0, R1 = quat_to_R(gt[g0, 4:8]), quat_to_R(gt[g1, 4:8])
        w = np.degrees(np.arccos(np.clip((np.trace(R0.T @ R1) - 1) / 2, -1, 1))) / (gtm[g1] - gtm[g0])
        b = next(k for k, (lo, hi) in enumerate(BINS) if lo <= w < hi)
        pre = int(((loss_t > tm[i]) & (loss_t <= tm[i] + 1)).any())
        acc[(b, pre)].append(dg[i])
print(f"# Local map matching on ORB frames after the tap: {root.name} {arm} ({SC}, {nruns} runs)\n")
print("| True turn rate | Frames (no loss within 1 s / loss within 1 s) | In view, median | Matched, median | Matched share, median | From the 10 newest KFs, median | Whole map: projecting / valid / cells |")
print("|---|---|---|---|---|---|---|")
for b, (lo, hi) in enumerate(BINS):
    for pre in (0, 1):
        a = np.array(acc[(b, pre)])
        lab = f"{lo}-{hi:.0f} deg/s" if hi < 1e8 else f">= {lo} deg/s"
        if not len(a):
            print(f"| {lab} | {'loss follows' if pre else 'no loss'}: 0 | - | - | - | - | - |"); continue
        print(f"| {lab} | {'loss follows' if pre else 'no loss'}: {len(a)} | {np.median(a[:, 0]):.0f} | {np.median(a[:, 1]):.0f} | "
              f"{np.median(a[:, 1] / np.maximum(a[:, 0], 1)):.2f} | {np.median(a[:, 2]):.0f} | "
              f"{np.nanmedian(a[:, 3]) if np.isfinite(a[:, 3]).any() else float('nan'):.0f} / "
              f"{np.nanmedian(a[:, 4]) if np.isfinite(a[:, 4]).any() else float('nan'):.0f} / "
              f"{np.nanmedian(a[:, 5]) if np.isfinite(a[:, 5]).any() else float('nan'):.0f} |")
