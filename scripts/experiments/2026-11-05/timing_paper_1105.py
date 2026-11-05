"""timing_1105 -> TIMING_PAPER.md: the paper's §VII-B (PC part) text block, built from results/timing_1105/runs.csv
(score_timing_1105.py). Values are the mean of the runs of each arm and sequence (2 per cell); the per-run values are
in TIMING.md / runs.csv.
Usage: timing_paper_1105.py [--root H:/Research]"""
import argparse, csv
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="H:/Research")
a = ap.parse_args()
O = Path(a.root) / "results/timing_1105"
rows = list(csv.DictReader(open(O / "runs.csv", encoding="utf-8")))
G = defaultdict(list)
for r in rows: G[(r["arm"], r["seq"])].append(r)
ARMS = [("native-stock", "stock, native"), ("wasm-stock", "stock, WASM"), ("native-fast", "fast, native"),
        ("page-fast", "fast, WASM"), ("native-next", "next, native"), ("page-next", "next, WASM"),
        ("native-beta", "beta, native"), ("page-beta", "beta, WASM"), ("alvaar", "AlvaAR (WASM)")]
SEQ = [("euroc_MH_01_easy", "MH01"), ("euroc_V1_02_medium", "V1_02"), ("tumvi_dataset-room2_512_16", "room2")]


def m(rs, k, fmt="{:.1f}"):
    v = [float(r[k]) for r in rs if r.get(k) not in (None, "")]
    return fmt.format(sum(v) / len(v)) if v else "–"


L = ["**Table XII. Runtime of the frozen build on the PC, one run at a time** (mean of 2 runs per cell; per-run values "
     "in the supplement). Real-time pacing (§VI-C). *Track*: the tracking call's time per frame (median, 95th "
     "percentile; frame loading and logging are outside it); WebVIO maps on its own threads, while AlvaAR's call "
     "includes its mapping, which runs inline. *> 33 ms*: share of frames over the 30 fps frame time. *CPU / frame*: "
     "user + system time of all threads per processed frame, including process start-up and loading the vocabulary. "
     "*LBA*: mean time of one local bundle adjustment. *RSS*: peak resident memory (WASM: Node.js plus the module's "
     "heap). AlvaAR processes EuRoC's 752×480 frames, the ORB-SLAM3-based arms 600×350 (1.7× fewer pixels). Desktop "
     "PC (§VI-F), WASM in Node.js: not the phone. The sequences are 20 Hz, so real-time pacing gives 50 ms per frame; the > 33 ms column refers to a 30 fps phone camera, not to the replay. AlvaAR runs on EuRoC only (§VI-A).", "",
     "| seq. | arm | track ms, median | track ms, p95 | > 33 ms (%) | CPU ms / frame | keyframes | LBA ms | RSS MB |",
     "|---|---|---|---|---|---|---|---|---|"]
for s, sn in SEQ:
    for k, an in ARMS:
        rs = G.get((k, s))
        if not rs: continue
        L.append(f"| {sn} | {an} | {m(rs, 'track_med')} | {m(rs, 'track_p95')} | {m(rs, 'over33')} | "
                 f"{m(rs, 'cpu_ms_per_frame')} | {m(rs, 'keyframes', '{:.0f}')} | {m(rs, 'lba_ms')} | {m(rs, 'rss_mb', '{:.0f}')} |")
fails = [f"{r['arm']} {r['seq']} {r['run']}" for r in rows if r.get("failed") == "True"]
L += ["", f"Runs: {len(rows)}; failed: {', '.join(fails) if fails else 'none'}."]
(O / "TIMING_PAPER.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
print("\n".join(L))

# Derived values quoted in the §VII-B prose (not part of the paper; a source for check_numbers.py).
def mean(rs, k):
    v = [float(r[k]) for r in rs if r.get(k) not in (None, "")]
    return sum(v) / len(v) if v else None


D = ["# timing_1105: values derived from the means of Table XII (timing_paper_1105.py)", ""]
for s, sn in SEQ:
    g = lambda k, f: mean(G.get((k, s), []), f)
    for k in ("track_med", "track_p95", "lba_ms"):
        for nat, wasm in (("native-stock", "wasm-stock"), ("native-fast", "page-fast"), ("native-next", "page-next"),
                          ("native-beta", "page-beta")):
            a, b = g(nat, k), g(wasm, k)
            if a and b: D.append(f"- {sn} {k} WASM/native {wasm}/{nat}: {b / a:.2f}")
    for side in ("native", "page"):
        a, b = g(f"{side}-fast", "lba_ms"), g(f"{side}-beta", "lba_ms")
        D.append(f"- {sn} lba_ms beta/fast {side}: {b / a:.2f}")
    for k in ("native-fast", "page-fast", "native-next", "page-next", "native-beta", "page-beta"):
        rs = G.get((k, s), [])
        sh = [float(r["orb_frames"]) / (float(r["orb_frames"]) + float(r["klt_frames"])) for r in rs]
        D.append(f"- {sn} {k} ORB-frame share per run: " + ", ".join(f"{100 * x:.1f}%" for x in sh))
spread = []
for (k, s), rs in G.items():
    v = [float(r["track_med"]) for r in rs]
    if len(v) == 2: spread.append((abs(v[0] - v[1]) / min(v), k, s, v))
import re
for s, sn in SEQ:
    for side in ("native", "page"):
        a_, b_ = mean(G.get((f"{side}-fast", s), []), "lba_ms"), mean(G.get((f"{side}-beta", s), []), "lba_ms")
        D.append(f"- {sn} local BA speed-up fast/beta {side}: {a_ / b_:.2f}x")
    for k in ("native-stock", "native-fast", "native-beta", "page-fast", "page-beta"):
        sz = []
        for r in G.get((k, s), []):
            log = O / k / s / r["run"] / "run.log"
            m = re.search(r"LBA-PROFILE runs=\d+ .* edges=(\d+) points=(\d+) kfs=(\d+)", log.read_text(errors="replace")) if log.exists() else None
            if m: sz.append(tuple(int(x) for x in m.groups()))
        if sz:
            D.append(f"- {sn} {k} local BA size per adjustment (edges, points, keyframes) per run: " + "; ".join(f"{e}, {pt}, {kf}" for e, pt, kf in sz))
for s, sn in SEQ:
    for k, an in ARMS:
        c = mean(G.get((k, s), []), "cpu_ms_per_frame")
        if c: D.append(f"- {sn} {k} cores busy on average (CPU ms / 50 ms frame): {c / 50:.2f}")
spread.sort(reverse=True)
D += ["", "Run-to-run spread of the tracking median, |run1 − run2| / smaller run (largest first):"]
D += [f"- {k} {s}: {x * 100:.0f}% ({v[0]:.1f} / {v[1]:.1f})" for x, k, s, v in spread]
eff = [float(r["effective_rate"]) for r in rows]
D += ["", f"Frames processed / frames: all equal: {all(r['frames'] == r['processed'] for r in rows)}; "
      f"dropped total: {sum(float(r['dropped'] or 0) for r in rows):.0f}; effective rate min {min(eff):.4f}."]
(O / "TIMING_DERIVED.md").write_text("\n".join(D) + "\n", encoding="utf-8", newline="\n")
