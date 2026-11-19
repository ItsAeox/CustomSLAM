"""Per scenario and arm from the anchor.txt files of a batch (eval/anchor_return.py): placed, back on the return (shown
at all in the return phase), the wait from the spot first in view on the return to the object first shown (ret_wait_s,
medians over the runs that came back), the return px median, and the runs with the object shown > 50 px off.
Usage: ret_table.py <batch dir> [scenario ...]"""
import sys, re, numpy as np
from pathlib import Path
from collections import defaultdict

root = Path(sys.argv[1]); only = sys.argv[2:]
rows = defaultdict(list)
for a in sorted(root.glob("*/*/seed*/anchor.txt")):
    kv = dict(re.findall(r"(\w+)=(\S+)", a.read_text()))
    seq, arm = a.parts[-4], a.parts[-3]
    sc = re.sub(r"\d+$", "", seq)
    if only and sc not in only: continue
    rows[(sc, arm)].append(kv)
print("| Scenario | Arm | Runs | Placed | Back on return | Wait s med (min-max) | Return px med | Wrong runs (px med > 50) | Wrong on return (ret px med > 50) |")
print("|---|---|---|---|---|---|---|---|---|")
for (sc, arm), rs in sorted(rows.items()):
    pl = [r for r in rs if r.get("placed") == "1"]
    back = [r for r in pl if float(r.get("ret_shown_pct", 0)) > 0]
    w = [float(r["ret_wait_s"]) for r in back if float(r.get("ret_wait_s", -1)) >= 0]
    rpx = [float(r["ret_px_med"]) for r in back if r.get("ret_px_med", "-") != "-"]
    wrong = sum(float(r["px_med"]) > 50 for r in pl if r.get("px_med", "-") != "-")
    rwrong = sum(x > 50 for x in rpx)
    ws = f"{np.median(w):.1f} ({min(w):.1f}-{max(w):.1f})" if w else "-"
    print(f"| {sc} | {arm} | {len(rs)} | {len(pl)} | {len(back)} | {ws} | {np.median(rpx) if rpx else float('nan'):.1f} | {wrong} | {rwrong} |")
