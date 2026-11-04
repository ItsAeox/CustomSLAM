"""g40_1104 (ISSUES G40 / G41): the AR page as it was vs with the fixes (option stash=fixed), on the same runs.
Per run (results/return_g40_1104[_phone]/<seq>/<arm>/seed<N>, floor placement):
  - page as it was: eval/anchor_return.py default replay (back at any merge) with the G41 rule applied (stash_drop_s:
    gone from a second new map whose predecessor had found a floor);
  - scorer default: the same without the G41 rule (what return_1104's main counts are);
  - fixed page: --page-fix (stash kept; back only when the merge absorbed the tree's map, from the runner's WEBMERGE).
On screen again: a return frame with the spot in view and the tree shown inside the image; right again: ret_snap.py's
rule (< 15 px for 2 s of shown frames). Also checks the rotation rule of ret_wrong_1104.py (merge partner from camera
rotations, clean = best <= 5 deg and every other >= 30 deg) against the WEBMERGE records.
Usage: g40_table_1104.py [--root H:/Research]"""
import argparse, json, re
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.spatial.transform import Rotation as Rot

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="H:/Research")
a = ap.parse_args()
R = Path(a.root); D = R / "datasets/synth"


def kvs(f):
    return dict(re.findall(r"(\w+)=(\S+)", f.read_text(encoding="utf-8"))) if f.exists() else {}


def ret_counts(dump, r0, drop=None):  # (on screen again, right again) on the return; drop: tree gone from this time
    d = np.loadtxt(dump, comments="#", ndmin=2)
    s = d[(d[:, 0] >= r0) & (d[:, 1] > 0) & (d[:, 2] > 0)]
    if drop is not None: s = s[s[:, 0] < drop]
    on = bool((s[:, 5] > 0).any()) if len(s) else False
    good, right = s[:, 3] < 15, False
    for i in np.where(good)[0]:
        w = (s[:, 0] >= s[i, 0]) & (s[:, 0] < s[i, 0] + 2)
        if good[w].all(): right = True; break
    return on, right


def rot_partner(sd, seq, t_merge):  # ret_wrong_1104.py's rule: (best map, clean) from the camera rotations
    tr = np.array([[float(x) for x in l.split()] for l in (sd / "trace.txt").read_text().splitlines()
                   if l and l[0] != "#" and len(l.split()) >= 34])
    st = np.loadtxt(D / seq / "mav0/cam0/data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64)
    t0 = float(open(D / seq / "gt/camera_dense.csv").readlines()[1].split()[0])
    tt = st[np.clip(tr[:, 0].astype(int), 0, len(st) - 1)] * 1e-9 - t0
    ok = np.isin(tr[:, 1].astype(int), [2, 5]); mid = tr[:, -2].astype(int)
    gt = np.loadtxt(R / f"results/gt/synth_{seq}.txt", comments="#", ndmin=2)
    gi = np.clip(np.searchsorted(gt[:, 0], tt + t0 - 1e-6), 0, len(gt) - 1)
    gp, Rg, Re = gt[gi, 1:4], Rot.from_quat(gt[gi, 4:8]), Rot.from_quat(tr[:, 6:10])
    aft = np.where(ok & (tt >= t_merge - 1e-3))[0]
    if not len(aft): return None
    aft = aft[mid[aft] == mid[aft[0]]][:150]
    err = {}
    for m in sorted(set(mid[ok & (tt < t_merge - 1e-3)].tolist())):
        if m == mid[aft[0]]: continue
        w = np.where(ok & (mid == m) & (tt < t_merge - 1e-3))[0]
        nr = w[np.min(np.linalg.norm(gp[w][:, None] - gp[aft][None], axis=2), axis=1) < 1.0]
        if len(nr) >= 10: w = nr
        if len(w) < 10: continue
        A = (Rg[w] * Re[w].inv()).mean()
        err[m] = float(np.median(np.degrees((Rg[aft].inv() * A * Re[aft]).magnitude())))
    if not err: return None
    b = min(err, key=err.get)
    return b, err[b] <= 5 and all(e >= 30 for m, e in err.items() if m != b)


T = defaultdict(lambda: defaultdict(int))
lines, val = [], defaultdict(int)
for batch in ("return_g40_1104", "return_g40_1104_phone"):
    pace = "phone" if "phone" in batch else "live"
    for sd in sorted((R / "results" / batch).glob("flat*/*/seed*")):
        seq, arm, seed = sd.parts[-3], sd.parts[-2], sd.parts[-1]
        old, fix = kvs(sd / "anchor_floor.txt"), kvs(sd / "anchor_floor_fix.txt")
        sc = re.sub(r"\d+$", "", seq); c = T[(pace, sc, arm)]
        c["runs"] += 1
        if old.get("placed") != "1": continue
        c["placed"] += 1
        r0 = json.loads((D / seq / "meta.json").read_text())["synthetic"]["motion"]["phases_s"]["return"][0]
        drop = float(old["stash_drop_s"]) if old.get("stash_drop_s", "-") != "-" else None
        o_on, o_ri = ret_counts(sd / "anchor_dump_place_floor.txt", r0, drop)
        s_on, s_ri = ret_counts(sd / "anchor_dump_place_floor.txt", r0)
        f_on, f_ri = ret_counts(sd / "anchor_dump_place_floor_fix.txt", r0)
        for k, v in (("old_on", o_on), ("old_right", o_ri), ("def_on", s_on), ("def_right", s_ri), ("fix_on", f_on), ("fix_right", f_ri)):
            c[k] += v
        merges = re.findall(r"WEBMERGE frame (\d+) \(([\d.]+) s\).*? map (-?\d+) absorbed into map (-?\d+)",
                            (sd / "run.log").read_text(errors="replace"))
        c["merges"] += len(merges)
        c["othermerge"] += fix.get("events", "").count("othermerge@")
        c["missing"] += int(fix.get("merge_record_missing", "0"))
        # the rotation rule against the records: every merge the old replay restored at
        tm0 = float(open(D / seq / "gt/camera_dense.csv").readlines()[1].split()[0]) - \
            np.loadtxt(D / seq / "mav0/cam0/data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64)[0] * 1e-9
        stash_map = None
        for ev in old.get("events", "-").split(","):
            if ev.startswith("restore@"):
                t_r = float(ev.split("@")[1])
                # the record printed at or right after the count drop (record times are on the camera clock from frame 0)
                rec = [m for m in merges if float(m[1]) - tm0 >= t_r - 0.2]
                if not rec: val["no record"] += 1; continue
                truth = int(rec[0][2])
                rp = rot_partner(sd, seq, t_r)
                pm = int(old.get("place_map", -1))
                if rp is None: val["no rotation call"] += 1; continue
                val[f"{'clean' if rp[1] else 'unclear'} call, {'right' if rp[0] == truth else 'wrong'}"] += 1
                if rp[1] and rp[0] != truth: lines.append(f"rotation rule wrong: {pace} {seq} {arm} {seed}: called {rp[0]}, record {truth}")
        if (o_on, o_ri) != (f_on, f_ri):
            lines.append(f"- {pace} {seq} {arm} {seed}: page as it was on screen {int(o_on)} right {int(o_ri)} -> fixed "
                         f"{int(f_on)} {int(f_ri)} | events old {old.get('events')} stash drop {old.get('stash_drop_s')} | "
                         f"fixed {fix.get('events')} | merges " + "; ".join(f"{m[2]}->{m[3]} @{m[1]}s" for m in merges))
print("# g40_1104: the AR page as it was vs with the G40 / G41 fixes, same runs (g40_table_1104.py)")
print()
print("Floor placement. On screen again / right again (< 15 px for 2 s) are counts of the placed runs. 'As it was': "
      "back at any merge, gone at a second new map once the map left has a floor. 'Scorer default': back at any merge "
      "(return_1104's main counts). 'Fixed': option stash=fixed.")
print()
print("| pace | scenario | arm | runs | placed | on screen: as it was / scorer default / fixed | right again: as it was / "
      "scorer default / fixed | merges | merges of other maps (fixed: tree stays stashed) | count drops without a record |")
print("|---|---|---|---|---|---|---|---|---|---|")
for (pace, sc, arm), c in sorted(T.items()):
    print(f"| {pace} | {sc} | {arm} | {c['runs']} | {c['placed']} | {c['old_on']} / {c['def_on']} / {c['fix_on']} | "
          f"{c['old_right']} / {c['def_right']} / {c['fix_right']} | {c['merges']} | {c['othermerge']} | {c['missing']} |")
print()
print("Rotation rule (ret_wrong_1104.py) vs the WEBMERGE records, every merge the old replay restored at: "
      + ", ".join(f"{k} {v}" for k, v in sorted(val.items())))
print()
print("Runs where the fixes change the result:")
print()
print("\n".join(lines) if lines else "(none)")
