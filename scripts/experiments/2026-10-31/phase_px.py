"""Per run of a batch: the AR object's px error (median while shown) and shown % per motion phase (meta.json phases_s),
from eval/anchor_return.py --dump. Shows where a wrong object goes wrong: at placement, during the survey, on the
way out, or on the return. Usage: phase_px.py <batch dir> <scenario> [arm ...]"""
import sys, json, subprocess, numpy as np
from pathlib import Path

root, sc = Path(sys.argv[1]), sys.argv[2]; arms = sys.argv[3:]
D = Path("H:/Research/datasets/synth"); here = Path(__file__).resolve().parents[3]
for sd in sorted(root.glob(f"{sc}[0-9]/*/seed*")):
    seq, arm = sd.parts[-3], sd.parts[-2]
    if arms and arm not in arms: continue
    dump = sd / "anchor_dump.txt"
    if not dump.exists():
        subprocess.run([sys.executable, str(here / "eval/anchor_return.py"), str(sd), str(D / seq),
                        "--gt", f"H:/Research/results/gt/synth_{seq}.txt", "--dump", str(dump)], capture_output=True)
    if not dump.exists():
        print(f"{seq} {arm} {sd.name}: not placed"); continue
    a = np.loadtxt(dump, comments="#", ndmin=2)
    if not len(a):
        print(f"{seq} {arm} {sd.name}: empty"); continue
    ph = json.loads((D / seq / "meta.json").read_text())["synthetic"]["motion"]["phases_s"]
    cells = []
    for name in ("look", "survey", "walk", "view", "toStart", "final"):
        if name not in ph: continue
        t0, t1 = ph[name]
        m = (a[:, 0] >= t0) & (a[:, 0] < t1) & (a[:, 1] > 0)
        s = m & (a[:, 2] > 0)
        cells.append(f"{name} {100 * s.sum() / max(m.sum(), 1):.0f}%/{np.median(a[s, 3]) if s.any() else float('nan'):.0f}px")
    print(f"{seq} {arm} {sd.name}: " + "  ".join(cells))
