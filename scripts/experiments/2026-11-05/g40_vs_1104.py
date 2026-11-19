"""Does g40_table_1104.py count 'on screen again' the way return_1104's tables do, and do the same seeds agree?"""
import json, re
import numpy as np
from pathlib import Path
from collections import defaultdict

R = Path("H:/Research"); D = R / "datasets/synth"


def kvs(f):
    return dict(re.findall(r"(\w+)=(\S+)", f.read_text(encoding="utf-8"))) if f.exists() else {}


def ret_counts(dump, r0, drop=None):  # copied from g40_table_1104.py
    d = np.loadtxt(dump, comments="#", ndmin=2)
    s = d[(d[:, 0] >= r0) & (d[:, 1] > 0) & (d[:, 2] > 0)]
    if drop is not None: s = s[s[:, 0] < drop]
    on = bool((s[:, 5] > 0).any()) if len(s) else False
    good, right = s[:, 3] < 15, False
    for i in np.where(good)[0]:
        w = (s[:, 0] >= s[i, 0]) & (s[:, 0] < s[i, 0] + 2)
        if good[w].all(): right = True; break
    return on, right


T = defaultdict(lambda: defaultdict(int)); per = {}
for old, new in (("return_1104", "return_g40_1104"), ("return_1104_phone", "return_g40_1104_phone")):
    pace = "phone" if "phone" in old else "live"
    for b in (old, new):
        for sd in sorted((R / "results" / b).glob("flat*/*/seed*")):
            seq, arm, seed = sd.parts[-3], sd.parts[-2], sd.parts[-1]
            if arm not in ("stock", "beta_newmap"): continue
            f = kvs(sd / "anchor_floor.txt"); sc = re.sub(r"\d+$", "", seq)
            c = T[(pace, sc, arm, "old" if b == old else "g40")]
            c["runs"] += 1
            if f.get("placed") != "1": per[(b == old, pace, seq, arm, seed)] = None; continue
            c["placed"] += 1
            r0 = json.loads((D / seq / "meta.json").read_text())["synthetic"]["motion"]["phases_s"]["return"][0]
            on, ri = ret_counts(sd / "anchor_dump_place_floor.txt", r0)
            c["on_g40rule"] += on; c["right_g40rule"] += ri
            c["on_pct>0"] += float(f.get("ret_onscreen_pct", "0")) > 0
            per[(b == old, pace, seq, arm, seed)] = (on, ri, f.get("ret_px_med"), f.get("maps", f.get("n_maps")))
print("| pace | scen | arm | batch | runs | placed | on screen (g40 ret_counts) | on screen (ret_onscreen_pct>0) | right (g40) |")
for k, c in sorted(T.items()):
    print(f"| {' | '.join(k)} | {c['runs']} | {c['placed']} | {c['on_g40rule']} | {c['on_pct>0']} | {c['right_g40rule']} |")
same = diff = 0; dl = []
for (isold, pace, seq, arm, seed), v in per.items():
    if not isold: continue
    w = per.get((False, pace, seq, arm, seed), "missing")
    if v is None or w is None or w == "missing":
        if (v is None) != (w is None): diff += 1; dl.append((pace, seq, arm, seed, v, w))
        else: same += 1
        continue
    if v[0] == w[0]: same += 1
    else: diff += 1; dl.append((pace, seq, arm, seed, v, w))
print(f"\nper seed, on screen (or not placed) same {same}, different {diff}")
for x in dl[:60]: print(x)
