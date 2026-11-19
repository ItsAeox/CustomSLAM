"""Per scenario and arm of a batch: the AR object's error under each placement the runs were scored with
(anchor.txt: the scorer's centre depth; anchor_floor.txt: the page's floor plane; anchor_refine.txt: + the tap ray
re-intersected with later floors; anchor_oracle.txt: the best fixed anchor of the placement map). Medians over the runs
placed by that method. Usage: place_table.py <batch dir>"""
import sys, re, numpy as np
from pathlib import Path
from collections import defaultdict

root = Path(sys.argv[1])
FILES = (("centre", "anchor.txt"), ("floor", "anchor_floor.txt"), ("floor+refine", "anchor_refine.txt"), ("best fixed", "anchor_oracle.txt"))
R = defaultdict(lambda: defaultdict(list))
for sd in sorted(root.glob("*/*/seed*")):
    sc, arm = re.sub(r"\d+$", "", sd.parts[-3]), sd.parts[-2]
    for name, fn in FILES:
        f = sd / fn
        if not f.exists(): continue
        kv = dict(re.findall(r"(\w+)=(\S+)", f.read_text()))
        R[(sc, arm, name)]["runs"].append(1)
        if kv.get("placed") != "1": continue
        R[(sc, arm, name)]["placed"].append(1)
        if kv.get("anchor_method") == "tapplane": R[(sc, arm, name)]["tapplane"].append(1)
        if kv.get("px_med", "-") != "-":
            R[(sc, arm, name)]["px"].append(float(kv["px_med"]))
        if kv.get("ret_px_med", "-") != "-" and float(kv.get("ret_shown_pct", 0)) > 0:
            R[(sc, arm, name)]["rpx"].append(float(kv["ret_px_med"]))
        if "refine_updates" in kv:
            R[(sc, arm, name)]["upd"].append(int(kv["refine_updates"]))
            R[(sc, arm, name)]["ratio"].append(float(kv["refine_depth_ratio"]))
print("| Scenario | Arm | Placement | Runs | Placed (tap plane) | px med | px p75 | runs > 50 px | Return px med | Refine: updates / last depth ratio (med) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for (sc, arm, name), d in sorted(R.items(), key=lambda kv: (kv[0][0], kv[0][1], [n for n, _ in FILES].index(kv[0][2]))):
    px, rpx = d["px"], d["rpx"]
    m = lambda x: f"{np.median(x):.1f}" if x else "-"
    ref = f"{np.median(d['upd']):.0f} / {np.median(d['ratio']):.2f}" if d["upd"] else "-"
    print(f"| {sc} | {arm} | {name} | {len(d['runs'])} | {len(d['placed'])} ({len(d['tapplane'])}) | {m(px)} | "
          f"{np.percentile(px, 75) if px else float('nan'):.1f} | {sum(x > 50 for x in px)} | {m(rpx)} | {ref} |")
