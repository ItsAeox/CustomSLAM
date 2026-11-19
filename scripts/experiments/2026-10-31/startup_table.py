"""The map start per scenario and arm (initf_1102; any batch with trace.txt + track_times.txt): time from the first frame
to the first tracked frame (s, sequence clock), the frames processed per sequence second until then (the frame rate a
device at this pace sees while the map starts), their median track time (ms), the first map's points ("New Map created
with N points"), maps started in the run, and tracked % of the run. Medians over the runs. Usage: startup_table.py <batch>"""
import sys, re, numpy as np
from pathlib import Path
from collections import defaultdict

root = Path(sys.argv[1])
D = Path("H:/Research/datasets/synth") if Path("H:/Research/datasets/synth").exists() else Path("/mnt/h/Research/datasets/synth")
R = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for sd in sorted(root.glob("*/*/seed*")):
    seq, arm = sd.parts[-3], sd.parts[-2]
    tt, tr = sd / "track_times.txt", sd / "trace.txt"
    if not (sd / "run_info.txt").exists() or not tt.exists() or not tr.exists() or not (D / seq).exists(): continue
    T = np.loadtxt(tt, ndmin=2)
    st = {int(r.split()[0]): int(r.split()[1]) for r in tr.read_text().splitlines() if r.strip()}
    fr, ms = T[:, 0].astype(int), T[:, 1]
    stamps = np.loadtxt(D / seq / "mav0/cam0/data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64) * 1e-9
    t = stamps[np.clip(fr, 0, len(stamps) - 1)] - stamps[0]
    ok = np.array([st.get(int(f), 0) in (2, 5) for f in fr])
    log = (sd / "run.log").read_text(errors="replace")
    m = re.search(r"New Map created with (\d+) points", log)
    sc = re.sub(r"\d+$", "", seq)
    d = R[sc][arm]
    d["tracked"].append(100 * ok.mean())
    d["maps"].append(len(re.findall(r"New Map created", log)))
    d["pts"].append(int(m.group(1)) if m else np.nan)
    if ok.any():
        i0 = int(np.argmax(ok))
        d["first"].append(t[i0])
        d["ms"].append(np.median(ms[:i0]) if i0 else np.nan)
        d["fps"].append(i0 / t[i0] if t[i0] > 0 else np.nan)
    else:
        d["never"].append(1)
med = lambda v: f"{np.nanmedian(v):.1f}" if len(v) and np.isfinite(np.array(v, float)).any() else "-"
print(f"# Map start: {root.name}\n")
print("| Scenario | Arm | Runs | First map (s) | Frame rate until then (fps) | Track ms until then | First map points | Maps started | Tracked % | Never tracked |")
print("|---|---|---|---|---|---|---|---|---|---|")
for sc in sorted(R):
    for arm in sorted(R[sc]):
        d = R[sc][arm]
        print(f"| {sc} | {arm} | {len(d['tracked'])} | {med(d['first'])} | {med(d['fps'])} | {med(d['ms'])} | {med(d['pts'])} | "
              f"{med(d['maps'])} | {med(d['tracked'])} | {len(d['never'])} |")
