"""return_1104: the runs where the object is shown again on the return but never right (ret_snap.py: < 15 px for 2 s of
shown frames), sorted by rules fixed before counting, and the return counts the page's own rules would give.
Inputs per run: eval/anchor_return.py --place floor (anchor_floor.txt, anchor_dump_place_floor.txt: the page's floor
placement; 4 Nov fields ret_onscreen_*, stash_drop_s, dump column 6 = shown and in the image), trace.txt, the batch's
COLLAPSE.md (eval/scale_collapse.py).

Per run that is back (the page's visible flag on a return frame with the spot in view) and not right again:
  - before: px median of the shown frames from the placement to the start of the walk (right before leaving?)
  - r place: the map's camera translation scale at the placement against its start-up: Sim3 scale (true m per map
    unit) of 3 s windows (every 1.5 s, >= 20 tracked frames, >= 0.15 m true motion; split at merges, as
    eval/scale_collapse.py), r = s(first 3 windows) / s(windows within 6 s of the placement), median. 1 = the camera
    moves in the map as at the start; 0.2 = 5x less (scale_collapse.py's 5x rule).
  - merge partner (runs whose object was stashed and restored, i.e. the atlas map count dropped): a merge keeps the
    current map's id in the older map's coordinates, so the merged map's camera rotations line up with the truth through
    the older map's fixed rotation. Per candidate older map, the mean R_true R_est^-1 over its tracked frames before the
    merge (those within 1 m of the post-merge frames' true positions if >= 10), then the median rotation error of the
    current map's first 150 frames after the merge under it. Clean = the best <= 5 deg and every other >= 30 deg.
Classes, first rule that holds: (1) merged with another map (clean partner, not the object's map); (2) collapsed at
placement (r place <= 0.2); (3) wrong before leaving (before >= 15 px); (4) near miss (return px median < 50);
(5) other. Off screen (no return frame with the object shown and in the image) is counted on top.
Page rules: 'on screen' = back with the object in the image on some return frame; 'page stash' = also not after the
page's stashAnchors replaced the stash (stash_drop_s before the first such frame).
Usage: ret_wrong_1104.py [--root H:/Research] [--px 15] [--all] [--out results/return_1104/RET_WRONG.md under --root]"""
import argparse, json, re
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.spatial.transform import Rotation as Rot

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="H:/Research")
ap.add_argument("--px", type=float, default=15.0)
ap.add_argument("--all", action="store_true", help="print every run back on the return, not only the not-right ones")
ap.add_argument("--out", help="also write the output here (default <root>/results/return_1104/RET_WRONG.md; '-': no file)")
a = ap.parse_args()
R = Path(a.root); D = R / "datasets/synth"
OUT = []
def say(x=""):
    print(x); OUT.append(x)
say("# return_1104: back on the return but not right, and the page's own rules (ret_wrong_1104.py)")
say()
say("Rules and fields: the docstring of scripts/experiments/2026-11-04/ret_wrong_1104.py. Floor placement (the pre-cut "
    "page's). One line per run back and not right" + (" (--all: every run back)" if a.all else "") + ", then the counts.")
say()
CLASSES = ("merged with another map", "collapsed at placement", "wrong before leaving", "near miss", "other")


def sim3_scale(src, dst):  # Sim3 dst ~ s R src + t: s (eval/scale_from_accel.py umeyama_scale)
    a_, b_ = src - src.mean(0), dst - dst.mean(0)
    U, S, Vt = np.linalg.svd(b_.T @ a_ / len(src))
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: E[2, 2] = -1
    return np.trace(np.diag(S) @ E) / (a_ ** 2).sum(1).mean()


def right_again(s, px):  # ret_snap.py's rule on the shown return frames (t, px): the first time it holds, else None
    good = s[:, 1] < px
    for i in np.where(good)[0]:
        w = (s[:, 0] >= s[i, 0]) & (s[:, 0] < s[i, 0] + 2)
        if good[w].all(): return s[i, 0]
    return None


summ = defaultdict(lambda: defaultdict(int))
for batch in ("return_1104", "return_1104_phone"):
    pace = "phone" if "phone" in batch else "live"
    coll = {}
    for l in (R / "results" / batch / "COLLAPSE.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(\w+\d) (\S+) (seed\d+): .*", l)
        if m: coll[m.groups()] = "COLLAPSE" in l
    for sd in sorted((R / "results" / batch).glob("flat*/*/seed*")):
        seq, arm, seed = sd.parts[-3], sd.parts[-2], sd.parts[-1]
        kv = dict(re.findall(r"(\w+)=(\S+)", (sd / "anchor_floor.txt").read_text(encoding="utf-8")))
        if kv.get("placed") != "1": continue
        sc = re.sub(r"\d+$", "", seq)
        c = summ[(pace, sc, arm)]
        c["placed"] += 1
        d = np.loadtxt(sd / "anchor_dump_place_floor.txt", comments="#", ndmin=2)
        ph = json.loads((D / seq / "meta.json").read_text())["synthetic"]["motion"]["phases_s"]
        r0, w0 = ph["return"][0], ph["walk"][0]
        iv = d[:, 1] > 0
        retv = iv & (d[:, 0] >= r0)
        s = d[retv & (d[:, 2] > 0)]
        if not retv.any() or not len(s): continue
        c["back"] += 1
        tin = d[retv, 0][0]
        on = s[s[:, 5] > 0]
        drop = float(kv["stash_drop_s"]) if kv.get("stash_drop_s", "-") != "-" else None
        hit = right_again(s[:, [0, 3]], a.px)
        if len(on):
            c["onscreen"] += 1
            if drop is None or drop > on[0, 0]: c["onscreen_page"] += 1
        if hit is not None:
            c["right"] += 1
            if drop is None or drop > hit: c["right_page"] += 1
            if not a.all: continue
        # the trace: time on the motion's clock, state, pose, map id / count
        tr = np.array([[float(x) for x in l.split()] for l in (sd / "trace.txt").read_text().splitlines()
                       if l and l[0] != "#" and len(l.split()) >= 34])
        st = np.loadtxt(D / seq / "mav0/cam0/data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64)
        t0 = float(open(D / seq / "gt/camera_dense.csv").readlines()[1].split()[0])
        tt = st[np.clip(tr[:, 0].astype(int), 0, len(st) - 1)] * 1e-9 - t0
        ok = np.isin(tr[:, 1].astype(int), [2, 5])
        mid, nm = tr[:, -2].astype(int), tr[:, -1].astype(int)
        gt = np.loadtxt(R / f"results/gt/synth_{seq}.txt", comments="#", ndmin=2)
        gi = np.clip(np.searchsorted(gt[:, 0], tt + t0 - 1e-6), 0, len(gt) - 1)
        gp, gq = gt[gi, 1:4], gt[gi, 4:8]
        pm = int(kv["place_map"])
        # how it came back: the last lost frame before the first shown return frame
        j1 = int(np.argmin(np.abs(tt - s[0, 0])))
        lost = np.where(~ok[:j1])[0]
        how = "tracked in" if not len(lost) or tt[lost[-1]] < r0 else f"recovered {tt[lost[-1] + 1] - tin:+.1f}s"
        rmaps = sorted(set(mid[[int(np.argmin(np.abs(tt - x))) for x in s[:, 0]]].tolist()))
        # r place
        epoch = np.concatenate([[0], np.cumsum(np.diff(nm) < 0)])
        jpl = int(np.argmin(np.abs(tt - d[0, 0])))
        sel = ok & (mid == pm) & (epoch == epoch[jpl])
        win = []
        for c0 in np.arange(tt[sel][0], tt[sel][-1] - 1.5, 1.5):
            w = sel & (tt >= c0) & (tt < c0 + 3)
            if w.sum() >= 20 and np.linalg.norm(np.ptp(gp[w], 0)) >= 0.15:
                win.append((c0 + 1.5, sim3_scale(tr[w, 3:6], gp[w])))
        win = np.array(win).reshape(-1, 2)
        near = np.abs(win[:, 0] - d[0, 0]) < 6
        rpl = float(np.median(np.median(win[:3, 1]) / win[near, 1])) if len(win) >= 3 and near.any() else None
        # merge partner
        part, perr = None, {}
        ev = kv.get("events", "-")
        if "restore@" in ev:
            tre = float(re.findall(r"restore@([\d.]+)", ev)[0])
            Rg, Re = Rot.from_quat(gq), Rot.from_quat(tr[:, 6:10])
            aft = np.where(ok & (tt >= tre - 1e-3))[0]
            aft = aft[mid[aft] == mid[aft[0]]][:150]
            for m_ in sorted(set(mid[ok & (tt < tre - 1e-3)].tolist())):
                if m_ == mid[aft[0]]: continue
                w = np.where(ok & (mid == m_) & (tt < tre - 1e-3))[0]
                nr = w[np.min(np.linalg.norm(gp[w][:, None] - gp[aft][None], axis=2), axis=1) < 1.0]
                if len(nr) >= 10: w = nr
                if len(w) < 10: continue
                A = (Rg[w] * Re[w].inv()).mean()
                perr[m_] = float(np.median(np.degrees((Rg[aft].inv() * A * Re[aft]).magnitude())))
            if perr:
                b = min(perr, key=perr.get)
                clean = perr[b] <= 5 and all(e >= 30 for m_, e in perr.items() if m_ != b)
                part = (b, clean)
        before = d[iv & (d[:, 2] > 0) & (d[:, 0] < w0)]
        bpx = float(np.median(before[:, 3])) if len(before) else None
        rpx = float(np.median(s[:, 3]))
        if part and part[1] and part[0] != pm: cl = 0
        elif rpl is not None and rpl <= 0.2: cl = 1
        elif bpx is not None and bpx >= 15: cl = 2
        elif rpx < 50: cl = 3
        else: cl = 4
        if hit is None:
            c[CLASSES[cl]] += 1
            if not len(on): c["offscreen"] += 1
        pt = "-" if part is None else (f"{'object' if part[0] == pm else 'OTHER'} map {part[0]}{'' if part[1] else ' (unclear)'} ["
                                       + " ".join(f"{m_}:{e:.0f}" for m_, e in sorted(perr.items())) + "]")
        say(f"- {pace} {seq} {arm} {seed}: {CLASSES[cl] if hit is None else 'RIGHT'} | before "
              f"{'-' if bpx is None else f'{bpx:.1f}'}px | ret {rpx:.0f}px on screen {100 * len(on) / len(s):.0f}% "
              f"first {s[0, 0] - tin:+.1f}s {how} | map {pm}->{rmaps} n{nm.max()} | r place "
              f"{'-' if rpl is None else f'{rpl:.2f}'} | merge {pt} | stash drop {kv.get('stash_drop_s', '-')} | "
              f"collapse {int(coll.get((seq, arm, seed), False))}" + (f" | right {hit - tin:+.1f}s" if hit is not None else ""))
say()
say(f"| Pace | Scenario | Arm | Placed | Back (shown flag) | On screen | On screen, page stash | Right again "
      f"(< {a.px:.0f} px for 2 s) | Right again, page stash | Back, not right | " + " | ".join(CLASSES) + " | of them off screen |")
say("|---" * (11 + len(CLASSES)) + "|")
for (pace, sc, arm), c in sorted(summ.items()):
    say(f"| {pace} | {sc} | {arm} | {c['placed']} | {c['back']} | {c['onscreen']} | {c['onscreen_page']} | {c['right']} | "
          f"{c['right_page']} | {c['back'] - c['right']} | " + " | ".join(str(c[k]) for k in CLASSES) + f" | {c['offscreen']} |")
out = a.out or str(R / "results/return_1104" / ("RET_WRONG_ALL.md" if a.all else "RET_WRONG.md"))
if out != "-":
    Path(out).write_text("\n".join(OUT) + "\n", encoding="utf-8")
