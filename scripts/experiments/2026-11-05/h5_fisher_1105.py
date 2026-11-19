"""h5_1105 (ISSUES H5): two-sided Fisher exact tests (as score_main_1103.py) for the arm comparisons of VII section I,
with stock and beta_newmap counted in both frozen-build batches (A = return_1104, B = return_h5_1105; fast / next / beta
ran in A only), and A vs B for the same arm as the repeat check. Counts of the placed runs, floor placement; on screen
again and right again as h5_table_1105.py.
Usage: h5_fisher_1105.py [--root H:/Research]"""
import argparse, json, re
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import fisher_exact

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="H:/Research")
a = ap.parse_args()
R = Path(a.root); D = R / "datasets/synth"
KEEP, NEW = ("fast", "next", "beta"), ("stock", "beta_newmap")


def kvs(f):
    return dict(re.findall(r"(\w+)=(\S+)", f.read_text(encoding="utf-8"))) if f.exists() else {}


def ret_counts(dump, r0):  # as h5_table_1105.py
    d = np.loadtxt(dump, comments="#", ndmin=2)
    s = d[(d[:, 0] >= r0) & (d[:, 1] > 0) & (d[:, 2] > 0)]
    on = bool((s[:, 5] > 0).any()) if len(s) else False
    good, right = s[:, 3] < 15, False
    for i in np.where(good)[0]:
        w = (s[:, 0] >= s[i, 0]) & (s[:, 0] < s[i, 0] + 2)
        if good[w].all(): right = True; break
    return on, right


C = defaultdict(lambda: defaultdict(int))  # (batch, pace, scenario, arm) -> placed / on / right
for k, b in (("A", "return_1104"), ("B", "return_h5_1105")):
    for pace, suf in (("live", ""), ("phone", "_phone")):
        for sd in sorted((R / "results" / (b + suf)).glob("flat*/*/seed*")):
            seq, arm = sd.parts[-3], sd.parts[-2]
            f = kvs(sd / "anchor_floor.txt")
            if f.get("placed") != "1": continue
            r0 = json.loads((D / seq / "meta.json").read_text())["synthetic"]["motion"]["phases_s"]["return"][0]
            on, ri = ret_counts(sd / "anchor_dump_place_floor.txt", r0)
            c = C[(k, pace, re.sub(r"\d+$", "", seq), arm)]
            c["placed"] += 1; c["on"] += on; c["right"] += ri


def p(x, y, f):
    return fisher_exact([[x[f], x["placed"] - x[f]], [y[f], y["placed"] - y[f]]])[1]


def fmt(x, f):
    return f"{x[f]}/{x['placed']}"


print("# h5_1105: Fisher exact tests for VII section I (h5_fisher_1105.py)")
print()
print("Two-sided Fisher exact test on counts of the placed runs (as score_main_1103.py). A = return_1104, B = "
      "return_h5_1105 (the same frozen build again; stock and beta_newmap only). Keep-map arms from A.")
for f, name in (("on", "on screen again"), ("right", "right again (< 15 px for 2 s)")):
    print()
    print(f"## {name}: keep-map arm vs new-map arm")
    print()
    print("| pace | scenario | keep-map arm (A) | vs stock A | p | vs stock B | p | vs beta_newmap A | p | vs beta_newmap B | p |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for pace in ("live", "phone"):
        for sc in ("flat", "flatmap", "flatfirst"):
            for arm in KEEP:
                x = C[("A", pace, sc, arm)]
                row = [pace, sc, f"{arm} {fmt(x, f)}"]
                for na in NEW:
                    for k in ("A", "B"):
                        y = C[(k, pace, sc, na)]
                        row += [fmt(y, f), f"{p(x, y, f):.3f}"]
                print("| " + " | ".join(row) + " |")
print()
print("## Repeat check: the same arm, A vs B")
print()
print("| pace | scenario | arm | on screen again A / B | p | right again A / B | p |")
print("|---|---|---|---|---|---|---|")
for pace in ("live", "phone"):
    for sc in ("flat", "flatmap", "flatfirst"):
        for arm in NEW:
            x, y = C[("A", pace, sc, arm)], C[("B", pace, sc, arm)]
            print(f"| {pace} | {sc} | {arm} | {fmt(x, 'on')} / {fmt(y, 'on')} | {p(x, y, 'on'):.3f} | "
                  f"{fmt(x, 'right')} / {fmt(y, 'right')} | {p(x, y, 'right'):.3f} |")
