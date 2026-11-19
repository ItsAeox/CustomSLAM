"""h5_1105 (ISSUES H5): do the AR return counts repeat? Three batches of stock and beta_newmap, same scenarios, seeds,
settings and pacing: return_1104 (frozen snap-84c1f17, 4 Nov), return_h5_1105 (the same frozen build again, 5 Nov) and
return_g40_1104 (snap-a30b538: the post-cut code, gated off for these arms, plus the merge record).
Per run, floor placement as return_1104 / VII section I: placed; on screen again (a return frame with the spot in view
and the object shown inside the image); right again (ret_snap.py's rule, < 15 px for 2 s of shown frames); the same two
with the page's stash rule (object gone from stash_drop_s on, G41). Per-run agreement: same sequence, arm and seed in
two batches; as a reference, two different seeds of the same sequence and arm inside one batch.
Usage: h5_table_1105.py [--root H:/Research]"""
import argparse, json, re
import numpy as np
from itertools import combinations
from pathlib import Path
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="H:/Research")
a = ap.parse_args()
R = Path(a.root); D = R / "datasets/synth"
B = [("A", "return_1104", "frozen, 4 Nov"), ("B", "return_h5_1105", "frozen again, 5 Nov"), ("G", "return_g40_1104", "a30b538, 4 Nov")]
ARMS = ("stock", "beta_newmap")


def kvs(f):
    return dict(re.findall(r"(\w+)=(\S+)", f.read_text(encoding="utf-8"))) if f.exists() else {}


def ret_counts(dump, r0, drop=None):  # as g40_table_1104.py: (on screen again, right again) on the return
    d = np.loadtxt(dump, comments="#", ndmin=2)
    s = d[(d[:, 0] >= r0) & (d[:, 1] > 0) & (d[:, 2] > 0)]
    if drop is not None: s = s[s[:, 0] < drop]
    on = bool((s[:, 5] > 0).any()) if len(s) else False
    good, right = s[:, 3] < 15, False
    for i in np.where(good)[0]:
        w = (s[:, 0] >= s[i, 0]) & (s[:, 0] < s[i, 0] + 2)
        if good[w].all(): right = True; break
    return on, right


run = {}  # (batch key, pace, seq, arm, seed) -> outcome dict, or None when not placed
for k, b, _ in B:
    for pace, suf in (("live", ""), ("phone", "_phone")):
        for sd in sorted((R / "results" / (b + suf)).glob("flat*/*/seed*")):
            seq, arm, seed = sd.parts[-3], sd.parts[-2], sd.parts[-1]
            if arm not in ARMS or not (sd / "anchor_floor.txt").exists(): continue
            f = kvs(sd / "anchor_floor.txt")
            if f.get("placed") != "1": run[(k, pace, seq, arm, seed)] = None; continue
            r0 = json.loads((D / seq / "meta.json").read_text())["synthetic"]["motion"]["phases_s"]["return"][0]
            drop = float(f["stash_drop_s"]) if f.get("stash_drop_s", "-") != "-" else None
            on, ri = ret_counts(sd / "anchor_dump_place_floor.txt", r0)
            son, sri = ret_counts(sd / "anchor_dump_place_floor.txt", r0, drop)
            run[(k, pace, seq, arm, seed)] = dict(on=on, right=ri, son=son, sright=sri)

T = defaultdict(lambda: defaultdict(int))
for (k, pace, seq, arm, seed), o in run.items():
    c = T[(pace, re.sub(r"\d+$", "", seq), arm, k)]
    c["runs"] += 1
    if o is None: continue
    c["placed"] += 1
    for f in ("on", "right", "son", "sright"): c[f] += o[f]

print("# h5_1105: do the AR return counts repeat? (h5_table_1105.py)")
print()
print("Batches: " + "; ".join(f"**{k}** `{b}` ({d})" for k, b, d in B) + ". Same arms, scenarios, seeds, settings and "
      "pacing. Floor placement, counts of runs (live: 6 sequences x 3 seeds = 18 per cell; phone pace 6 x 2 = 12). "
      "Page stash rule: the object gone from the second new map on once the map left has a floor (G41).")
print()
print("| pace | scenario | arm | runs A / B / G | placed A / B / G | on screen again A / B / G | right again A / B / G | "
      "page stash rule, on screen A / B / G | page stash rule, right A / B / G |")
print("|---|---|---|---|---|---|---|---|---|")
cells = sorted({k[:3] for k in T})
for cell in cells:
    g = [T.get(cell + (k,), {}) for k, _, _ in B]
    f = lambda n: " / ".join(str(x.get(n, 0)) if x else "-" for x in g)
    print(f"| {' | '.join(cell)} | {f('runs')} | {f('placed')} | {f('on')} | {f('right')} | {f('son')} | {f('sright')} |")


def outcome(o, f):
    return "not placed" if o is None else bool(o[f])


print()
print("**Per run, same sequence, arm and seed in two batches** (outcome = not placed / on screen again or not; "
      "agree = same outcome):")
print()
print("| pair | pace | runs in both | agree, on screen again | agree, right again | cell counts: largest difference "
      "(on screen again) | sum of differences over the cells |")
print("|---|---|---|---|---|---|---|")
for (k1, b1, _), (k2, b2, _) in combinations(B, 2):
    for pace in ("live", "phone"):
        keys = [x[1:] for x in run if x[0] == k1 and x[1] == pace and (k2,) + x[1:] in run]
        if not keys: continue
        ag = {f: sum(outcome(run[(k1,) + x], f) == outcome(run[(k2,) + x], f) for x in keys) for f in ("on", "right")}
        dif = [abs(T[c + (k1,)]["on"] - T[c + (k2,)]["on"]) for c in cells if c[0] == pace]
        print(f"| {k1} vs {k2} | {pace} | {len(keys)} | {ag['on']} ({100 * ag['on'] / len(keys):.0f} %) | "
              f"{ag['right']} ({100 * ag['right'] / len(keys):.0f} %) | {max(dif)} | {sum(dif)} |")
print()
print("**Reference: two different seeds of the same sequence and arm inside one batch:**")
print()
print("| batch | pace | seed pairs | agree, on screen again | agree, right again |")
print("|---|---|---|---|---|")
for k, b, _ in B:
    for pace in ("live", "phone"):
        grp = defaultdict(list)
        for x in run:
            if x[0] == k and x[1] == pace: grp[(x[2], x[3])].append(x)
        pairs = [(p, q) for v in grp.values() for p, q in combinations(sorted(v), 2)]
        if not pairs: continue
        ag = {f: sum(outcome(run[p], f) == outcome(run[q], f) for p, q in pairs) for f in ("on", "right")}
        print(f"| {k} | {pace} | {len(pairs)} | {ag['on']} ({100 * ag['on'] / len(pairs):.0f} %) | "
              f"{ag['right']} ({100 * ag['right'] / len(pairs):.0f} %) |")
