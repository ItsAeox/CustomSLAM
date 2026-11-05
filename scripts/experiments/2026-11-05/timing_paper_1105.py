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
ARMS = [("native-stock", "native-stock"), ("wasm-stock", "wasm-stock"), ("native-fast", "fast, native"),
        ("page-fast", "fast, WASM"), ("native-next", "next, native"), ("page-next", "next, WASM"),
        ("native-beta", "beta, native"), ("page-beta", "beta, WASM"), ("alvaar", "AlvaAR (WASM)")]
SEQ = [("euroc_MH_01_easy", "MH01"), ("euroc_V1_02_medium", "V1_02"), ("tumvi_dataset-room2_512_16", "room2")]


def m(rs, k, fmt="{:.1f}"):
    v = [float(r[k]) for r in rs if r.get(k) not in (None, "")]
    return fmt.format(sum(v) / len(v)) if v else "–"


L = ["**Table XII. Runtime of the frozen build on the PC, one run at a time** (mean of 2 runs per cell; per-run values "
     "in the supplement). Real-time pacing as in §VII-A. *Track*: the tracking call's time per frame (median, 95th "
     "percentile; frame loading and logging are outside it); WebVIO maps on its own threads, while AlvaAR's call "
     "includes its mapping, which runs inline. *> 33 ms*: share of frames over the 30 fps frame time. *CPU / frame*: "
     "user + system time of all threads per processed frame, including process start-up and loading the vocabulary. "
     "*LBA*: mean time of one local bundle adjustment. *RSS*: peak resident memory (WASM: Node.js plus the module's "
     "heap). AlvaAR processes EuRoC's 752×480 frames, the ORB-SLAM3-based arms 600×350 (1.7× fewer pixels). Desktop "
     "PC (§VI-F), WASM in Node.js: not the phone.", "",
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
