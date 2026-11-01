"""Coming back: when is the object right again? Per run of a batch (eval/anchor_return.py --dump, made here if missing):
from the first return frame with the spot in view, the time until the object is shown within <px> px of the spot and
stays so for 2 s of shown frames (-1: never), and the median px of the shown frames in the first 5 s of the return.
Covers both ways back: relocalisation / merge (shown late) and tracking through with drift until a loop closure
(shown early but off). Medians per scenario and arm. Usage: ret_snap.py <batch dir> [px=15]"""
import sys, json, re, subprocess, numpy as np
from pathlib import Path
from collections import defaultdict

root = Path(sys.argv[1]); PX = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
D = Path("H:/Research/datasets/synth"); here = Path(__file__).resolve().parents[3]
R = defaultdict(lambda: {"ok": [], "never": 0, "early": [], "n": 0})
for sd in sorted(root.glob("*/*/seed*")):
    seq, arm = sd.parts[-3], sd.parts[-2]
    if not (D / seq / "meta.json").exists(): continue
    dump = sd / "anchor_dump.txt"
    if not dump.exists():
        subprocess.run([sys.executable, str(here / "eval/anchor_return.py"), str(sd), str(D / seq),
                        "--gt", f"H:/Research/results/gt/synth_{seq}.txt", "--dump", str(dump)], capture_output=True)
    if not dump.exists(): continue
    a = np.loadtxt(dump, comments="#", ndmin=2)
    ph = json.loads((D / seq / "meta.json").read_text())["synthetic"]["motion"]["phases_s"]
    if "return" not in ph or not len(a): continue
    sc = re.sub(r"\d+$", "", seq); r = R[(sc, arm)]; r["n"] += 1
    m = (a[:, 0] >= ph["return"][0]) & (a[:, 1] > 0)
    if not m.any(): continue
    t0 = a[m, 0][0]
    s = a[m & (a[:, 2] > 0)]
    e5 = s[s[:, 0] < t0 + 5, 3]
    if len(e5): r["early"].append(np.median(e5))
    good = s[:, 3] < PX
    hit = -1.0
    for i in np.where(good)[0]:
        w = (s[:, 0] >= s[i, 0]) & (s[:, 0] < s[i, 0] + 2)
        if good[w].all(): hit = s[i, 0] - t0; break
    if hit < 0: r["never"] += 1
    else: r["ok"].append(hit)
print(f"| Scenario | Arm | Runs | Right again (< {PX:.0f} px for 2 s) | Time to it s, med (min-max) | px med in the first 5 s shown |")
print("|---|---|---|---|---|---|")
for (sc, arm), r in sorted(R.items()):
    ok = r["ok"]
    ts = f"{np.median(ok):.1f} ({min(ok):.1f}-{max(ok):.1f})" if ok else "-"
    print(f"| {sc} | {arm} | {r['n']} | {len(ok)} | {ts} | {np.median(r['early']) if r['early'] else float('nan'):.1f} |")
