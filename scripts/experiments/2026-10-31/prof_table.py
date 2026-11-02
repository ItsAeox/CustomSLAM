"""prof_1102 (prof_1102.sh): per run (<dir>/<arm>/<seq>, a profiling build's stage_times.txt + LocalMapTimeStats.txt +
run.log + run_info.txt), the tracking thread's ms per frame by kind, median / p90:
- map start (frames before the first map: no pose / local-map stage): count, seconds to the first map, total and its parts
  (ORB extraction, matching to the first frame, two-view reconstruction, the first map, the rest);
- flow frames (KLT), ORB frames without a keyframe, frames that made a keyframe: total and stages (ORB, pose prediction,
  local-map tracking, keyframe decision + creation, the rest);
- frames over 25 ms (the phone's budget is ~25 ms at ~2x desktop WASM, so the desktop share over 12.5 ms too);
- whole-map relocalisation tries (run.log: count, ms median / p90), dropped frames (run_info);
- LocalMapping per keyframe: total and its stages (insert, point culling, point creation, keyframe culling, the rest =
  local BA + the hand-off to loop closing). Upstream's LocalMapTimeStats.txt (and LBA_Stats.txt) wrote the
  keyframe-culling time in the LBA column too (vdLBASync_ms got timeKFCulling_ms; fixed 2 Nov, LocalMapping.cc).
Usage: prof_table.py <dir>"""
import sys, re
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])


def mp(v):
    v = np.asarray(v, float)
    return f"{np.median(v):.1f} / {np.percentile(v, 90):.1f}" if len(v) else "-"


print(f"# {root.name}: tracking-thread ms per frame (median / p90), desktop WASM, live@1.0, seed 1\n")
rows_start, rows_kind, rows_lm = [], [], []
for st in sorted(root.glob("*/*/stage_times.txt")):
    d = st.parent; arm, seq = d.parts[-2], d.parts[-1]
    t = np.loadtxt(st, ndmin=2)
    if t.shape[1] < 11:  # a build without the map-start timers
        print(f"WARNING: {st} has {t.shape[1]} columns (no map-start timers), padded with 0\n")
        t = np.hstack([t, np.zeros((len(t), 11 - t.shape[1]))])
    fr, tot, orb, pose, lm, kf, flow, made, im, itv, imap = (t[:, k] for k in range(11))
    flow = flow > 0; made = made > 0
    first = np.flatnonzero((pose > 0) | (lm > 0))
    n0 = first[0] if len(first) else len(t)
    ini = slice(0, n0)
    info = dict(l.split(" ", 1) for l in (d / "run_info.txt").read_text().splitlines() if " " in l) if (d / "run_info.txt").exists() else {}
    fps = float(info.get("frames", 0)) / max(float(info.get("seq_s", 1)), 1e-9)
    s0 = fr[n0] / fps if n0 < len(t) and fps else float("nan")
    rest = tot[ini] - orb[ini] - im[ini] - itv[ini] - imap[ini]
    rows_start.append(f"| {seq} | {arm} | {n0} | {s0:.1f} | {mp(tot[ini])} | {mp(orb[ini])} | {mp(im[ini][im[ini] > 0])} | "
                      f"{mp(itv[ini][itv[ini] > 0])} | {imap[ini].max():.0f} | {mp(rest)} |")
    a = slice(n0, None)
    T, O, P, L, K, F, M = tot[a], orb[a].copy(), pose[a], lm[a], kf[a], flow[a], made[a]
    O[F] = 0
    other = T - O - P - L - K
    for kind, m in (("flow", F), ("ORB", ~F & ~M), ("keyframe", M)):
        if m.any():
            rows_kind.append(f"| {seq} | {arm} | {kind} | {m.sum()} | {mp(T[m])} | {mp(O[m])} | {mp(P[m])} | {mp(L[m])} | "
                             f"{mp(K[m])} | {mp(other[m])} |")
    over = f"{100 * (T > 25).mean():.1f} / {100 * (T > 12.5).mean():.1f}"
    tries = [float(x) for x in re.findall(r"whole-map reloc try \d+:.*\(([\d.]+) ms\)", (d / "run.log").read_text(errors="replace"))]
    rows_kind.append(f"| {seq} | {arm} | all tracked frames: % over 25 / 12.5 ms {over}; whole-map tries {len(tries)} "
                     f"({mp(tries)} ms); dropped {info.get('dropped', '?')} of {info.get('frames', '?')} | | | | | | | |")
    lmf = d / "LocalMapTimeStats.txt"
    if lmf.exists():
        v = np.loadtxt(lmf, delimiter=",", comments="#", ndmin=2)
        if len(v):
            rows_lm.append(f"| {seq} | {arm} | {len(v)} | {mp(v[:, 5])} | {mp(v[:, 0])} | {mp(v[:, 1])} | {mp(v[:, 2])} | "
                           f"{mp(v[:, 4])} | {mp(v[:, 5] - v[:, :3].sum(1) - v[:, 4])} |")

print("## Map start (frames before the first map)\n")
print("| Seq | Arm | Frames | First map s | Total | ORB | Matching (when run) | Two-view (when run) | First map ms | Rest |")
print("|---|---|---|---|---|---|---|---|---|---|")
print("\n".join(rows_start))
print("\n## Tracked frames by kind\n")
print("| Seq | Arm | Kind | Frames | Total | ORB | Pose prediction | Local-map tracking | Keyframe | Rest (KLT, gyro, reloc ...) |")
print("|---|---|---|---|---|---|---|---|---|---|")
print("\n".join(rows_kind))
print("\n## LocalMapping per keyframe (ms, median / p90)\n")
print("| Seq | Arm | Keyframes | Total | Insert | Point culling | Point creation | Keyframe culling | Rest (local BA ...) |")
print("|---|---|---|---|---|---|---|---|---|")
print("\n".join(rows_lm))
