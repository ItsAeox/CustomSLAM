"""combo_1102 (any replay_arms batch on the synthetic set): per scenario and arm, medians over the runs, then per arm vs the
same batch's baseline route by route (per sequence: the median over its seeds; better / worse / equal routes and the
mean difference). Per run: tracked % (whole run; the walk phase where the sequence has one), p90 track ms and dropped
frames (run_info), and with the object (eval/anchor_return.py --place floor --refine, written here if missing, with its
per-frame dump): placed, px median, back on the return (shown %), right again (< 15 px held 2 s of shown frames, s from the
spot in view; -1 never), wrong % (> 50 px), phantom frames (shown, in the image, spot not in view).
Usage: combo_table.py <batch dir> [workers=6]"""
import sys, re, json, subprocess, numpy as np
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

root = Path(sys.argv[1]); W = int(sys.argv[2]) if len(sys.argv) > 2 else 6
D = Path("H:/Research/datasets/synth"); G = Path("H:/Research/results/gt")
here = Path(__file__).resolve().parents[3]
BASE = {"keep_g005t5", "g005t5"}
runs = [sd for sd in sorted(root.glob("*/*/seed*")) if (sd / "run_info.txt").exists() and (D / sd.parts[-3] / "meta.json").exists()]


def score(sd):
    out, dump = sd / "anchor_refine.txt", sd / "anchor_dump_place_floor_refine.txt"
    if (sd / "local_pts.bin").exists() and not (out.exists() and dump.exists()):
        r = subprocess.run([sys.executable, str(here / "eval/anchor_return.py"), str(sd), str(D / sd.parts[-3]), "--gt",
                            str(G / f"synth_{sd.parts[-3]}.txt"), "--place", "floor", "--refine", "--dump", str(dump)],
                           capture_output=True, text=True)
        out.write_text(r.stdout + r.stderr)


with ThreadPoolExecutor(W) as ex:
    list(ex.map(score, runs))


def snap(dump, ph, px=15.0):
    if not dump.exists() or "return" not in ph: return np.nan
    a = np.loadtxt(dump, comments="#", ndmin=2)
    m = (a[:, 0] >= ph["return"][0]) & (a[:, 1] > 0) if len(a) else np.zeros(0, bool)
    if not m.any(): return np.nan
    t0, s = a[m, 0][0], a[m & (a[:, 2] > 0)]
    good = s[:, 3] < px
    for i in np.where(good)[0]:
        w = (s[:, 0] >= s[i, 0]) & (s[:, 0] < s[i, 0] + 2)
        if good[w].all(): return s[i, 0] - t0
    return -1.0


R = defaultdict(list)  # (scenario, arm) -> per-run dicts
for sd in runs:
    seq, arm = sd.parts[-3], sd.parts[-2]
    meta = json.loads((D / seq / "meta.json").read_text())["synthetic"]
    ph = meta["motion"].get("phases_s", {})
    rows = [r.split() for r in (sd / "trace.txt").read_text().splitlines() if r.strip()] if (sd / "trace.txt").exists() else []
    fr = np.array([int(r[0]) for r in rows]); st = np.array([int(r[1]) for r in rows])
    stamps = np.loadtxt(D / seq / "mav0/cam0/data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64) * 1e-9
    t0 = float(open(D / seq / "gt/camera_dense.csv").readlines()[1].split()[0])
    t = stamps[np.clip(fr, 0, len(stamps) - 1)] - t0 if len(fr) else np.zeros(0)
    ok = np.isin(st, [2, 5])
    info = dict(l.split(" ", 1) for l in (sd / "run_info.txt").read_text().splitlines() if " " in l)
    tt = np.loadtxt(sd / "track_times.txt", ndmin=2) if (sd / "track_times.txt").exists() else np.zeros((0, 3))
    d = {"seq": seq, "tracked": 100 * ok.mean() if len(ok) else np.nan,
         "walk": 100 * ok[(t >= ph["walk"][0]) & (t < ph["walk"][1])].mean() if "walk" in ph and ((t >= ph["walk"][0]) & (t < ph["walk"][1])).any() else np.nan,
         "p90": np.percentile(tt[:, 1], 90) if len(tt) else np.nan, "dropped": float(info.get("dropped", "nan"))}
    a = sd / "anchor_refine.txt"
    if a.exists():
        kv = dict(re.findall(r"(\w+)=(\S+)", a.read_text()))
        f = lambda k: float(kv[k]) if k in kv and kv[k] not in ("-", "None") else np.nan
        d.update(placed=f("placed"), px=f("px_med"), retshown=f("ret_shown_pct"), wrong=f("wrong_pct"), phantom=f("phantom_frames"),
                 snap=snap(sd / "anchor_dump_place_floor_refine.txt", ph))
    R[(re.sub(r"\d+$", "", seq), arm)].append(d)

med = lambda v: f"{np.nanmedian(v):.1f}" if np.isfinite(np.array(v, float)).any() else "-"
print(f"# {root.name}: per scenario and arm (medians over runs)\n")
print("| Scenario | Arm | Runs | Tracked % | Walk tracked % | p90 track ms | Dropped frames | Placed | Object px | Return shown % | Right again s (never) | Wrong % | Phantom frames (runs with any) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for (sc, arm), rs in sorted(R.items()):
    g = lambda k: np.array([r.get(k, np.nan) for r in rs], float)
    sn = g("snap")
    print(f"| {sc} | {arm} | {len(rs)} | {med(g('tracked'))} | {med(g('walk'))} | {med(g('p90'))} | {med(g('dropped'))} | "
          f"{int(np.nansum(g('placed'))) if np.isfinite(g('placed')).any() else '-'} | {med(g('px'))} | {med(g('retshown'))} | "
          f"{med(np.where(sn >= 0, sn, np.nan))} ({int((sn < 0).sum())}) | {med(g('wrong'))} | "
          f"{med(g('phantom'))} ({int((g('phantom') > 0).sum())}) |")

print("\n# Route by route vs the same batch's baseline (per sequence: median over its seeds)\n")
print("| Scenario | Arm | Metric | Routes better / worse / equal | Mean difference (arm - baseline) |")
print("|---|---|---|---|---|")
for (sc, arm), rs in sorted(R.items()):
    if arm in BASE: continue
    base = next((R[(sc, b)] for b in BASE if (sc, b) in R), None)
    if not base: continue
    for k, lab, hi in (("tracked", "tracked %", 1), ("walk", "walk tracked %", 1), ("p90", "p90 track ms", -1),
                       ("px", "object px", -1), ("retshown", "return shown %", 1), ("wrong", "wrong %", -1)):
        diffs = []
        for seq in sorted({r["seq"] for r in rs}):
            av = [r.get(k, np.nan) for r in rs if r["seq"] == seq]; bv = [r.get(k, np.nan) for r in base if r["seq"] == seq]
            if np.isfinite(av).any() and np.isfinite(bv).any(): diffs.append(np.nanmedian(av) - np.nanmedian(bv))
        if not diffs: continue
        diffs = np.array(diffs)
        print(f"| {sc} | {arm} | {lab} | {int((hi * diffs > 0).sum())} / {int((hi * diffs < 0).sum())} / {int((diffs == 0).sum())} | {diffs.mean():+.1f} |")
