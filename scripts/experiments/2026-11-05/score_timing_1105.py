"""timing_1105: runtime of the frozen build, one run at a time (timing_1105.sh). Per run: tracking-thread time per frame
(track_times.txt; AlvaAR: its whole per-frame call, which includes its inline mapping), CPU time over all threads per
processed frame and peak resident memory (/usr/bin/time -v), effective playback rate, local-BA time per call
(LBA-PROFILE), keyframes, KLT / ORB frame split (KLT-STATS). Writes TIMING.md (stdout) and runs.csv.
Usage: score_timing_1105.py [--root H:/Research]"""
import argparse, csv, re
import numpy as np
from pathlib import Path
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="H:/Research")
a = ap.parse_args()
O = Path(a.root) / "results/timing_1105"
ARMS = ["native-stock", "wasm-stock", "native-fast", "page-fast", "native-next", "page-next", "native-beta", "page-beta",
        "alvaar"]
SEQS = ["euroc_MH_01_easy", "euroc_V1_02_medium", "tumvi_dataset-room2_512_16"]


def kv(f):
    return dict(l.split(" ", 1) for l in f.read_text().splitlines() if " " in l) if f.exists() else {}


rows = []
for arm in ARMS:
    for seq in SEQS:
        for rd in sorted((O / arm / seq).glob("run*")):
            r = dict(arm=arm, seq=seq, run=rd.name)
            tt = rd / "track_times.txt"
            if tt.exists() and tt.stat().st_size:
                ms = np.array([float(l.split()[1]) for l in tt.read_text().splitlines()
                               if l and l[0] != "#" and len(l.split()) >= 3])  # a run still writing ends mid-line
                r.update(frames_timed=len(ms), track_med=np.median(ms), track_p95=np.percentile(ms, 95),
                         over33=100 * np.mean(ms > 1000 / 30), over25=100 * np.mean(ms > 25))
            info = kv(rd / "run_info.txt")
            for k in ("frames", "processed", "dropped", "effective_rate", "keyframes", "maps", "mem_mb", "wall_s"):
                if k in info: r[k] = float(info[k])
            log = (rd / "run.log").read_text(errors="replace") if (rd / "run.log").exists() else ""
            m = re.search(r"LBA-PROFILE runs=(\d+) .*?total_ms=([\d.]+)", log)
            if m: r.update(lba_runs=int(m.group(1)), lba_ms=float(m.group(2)))
            m = re.search(r"KLT-STATS klt_frames=(\d+) orb_frames=(\d+) .*?klt_ms_mean=([\d.]+)", log)
            if m: r.update(klt_frames=int(m.group(1)), orb_frames=int(m.group(2)), klt_ms=float(m.group(3)))
            m = re.search(r"frames (\d+) processed (\d+) dropped (\d+) tracked (\d+) resets (\d+) longest (\d+) wall ([\d.]+) s", log)
            if m: r.update(frames=float(m.group(1)), processed=float(m.group(2)), dropped=float(m.group(3)),
                           wall_s=float(m.group(7)), resets=int(m.group(5)))
            tv = (rd / "time.txt").read_text() if (rd / "time.txt").exists() else ""
            u = re.search(r"User time \(seconds\): ([\d.]+)", tv); s = re.search(r"System time \(seconds\): ([\d.]+)", tv)
            mx = re.search(r"Maximum resident set size \(kbytes\): (\d+)", tv)
            if u and s and r.get("processed"):
                r["cpu_ms_per_frame"] = 1000 * (float(u.group(1)) + float(s.group(1))) / r["processed"]
            if mx: r["rss_mb"] = int(mx.group(1)) / 1024
            r["failed"] = "run failed" in log
            rows.append(r)

keys = sorted({k for r in rows for k in r})
with open(O / "runs.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)

G = defaultdict(list)
for r in rows: G[(r["arm"], r["seq"])].append(r)


def f(rs, k, fmt="{:.1f}", sep=" / "):
    v = [r.get(k) for r in rs]
    return sep.join(fmt.format(x) if x is not None else "–" for x in v) if rs else "–"


print("# timing_1105: runtime of the frozen build, one run at a time (score_timing_1105.py)")
print()
print("Frozen snap-84c1f17, --rt-wall (frames never before their time), 1 run at a time; values per run (run 1 / run 2). "
      "*track ms*: the tracking thread's time per frame (mapping and loop closing run on their own threads); AlvaAR: its "
      "whole per-frame call, which includes its mapping (inline). *CPU ms / frame*: user + system time of all threads "
      "divided by the frames processed, including process start-up and loading the 48.7 MB vocabulary. *> 33 ms*: share of frames over the 30 fps frame time. *LBA ms*: mean time of one "
      "local bundle adjustment. *RSS*: peak resident memory of the process (WASM: Node plus the WebAssembly heap).")
for seq in SEQS:
    print()
    print(f"## {seq}")
    print()
    print("| arm | track ms median | track ms p95 | > 33 ms (%) | effective rate | CPU ms / frame | keyframes | LBA ms | "
          "KLT / ORB frames | RSS MB | WASM heap MB |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in ARMS:
        rs = G.get((arm, seq), [])
        if not rs: continue
        print(f"| {arm} | {f(rs, 'track_med')} | {f(rs, 'track_p95')} | {f(rs, 'over33')} | {f(rs, 'effective_rate', '{:.2f}')} "
              f"| {f(rs, 'cpu_ms_per_frame')} | {f(rs, 'keyframes', '{:.0f}')} | {f(rs, 'lba_ms')} | "
              f"{' / '.join(str(r.get('klt_frames', '–')) + ':' + str(r.get('orb_frames', '–')) for r in rs)} | "
              f"{f(rs, 'rss_mb', '{:.0f}')} | {f(rs, 'mem_mb', '{:.0f}')} |")
fails = [f"{r['arm']} {r['seq']} {r['run']}" for r in rows if r.get("failed")]
print()
print("Failed runs: " + (", ".join(fails) if fails else "none"))
