#!/usr/bin/env python3
"""Scores ext_1103 (the main_1103 arms and AlvaAR on the synthetic set's 42 core sequences and the 15 Pixel 9
recordings of 18 Oct). No ranking rule was fixed for this batch before the results, so this prints raw tables only:
no bad-run threshold, no ties, no ranks.

Per run, the same definitions for WebVIO and AlvaAR:
- tracked share: frames with a pose / frames (WebVIO: tracking state OK in trace.txt; RECENTLY_LOST frames, where
  the coast option keeps showing a predicted pose, are not counted; AlvaAR: frames.txt status 1);
- largest-map share: tracked frames of the map with the most of them (WebVIO: trace map id; AlvaAR: longest segment,
  as its driver keeps it) / frames. synth_results.py ("longest live map") splits maps where the keyframe count drops
  to 0 instead of by map id; the two differ only after a map merge;
- maps per run: maps with at least one tracked frame (AlvaAR: segments; it has no relocalisation and resets on loss).
Synthetic set only (exact ground truth): online Sim3 ATE of the largest map (metrics_online.json; AlvaAR metrics.json),
the ground truth's RMS spread over the same frames, and synth_results.py's flags: 'short' = spread < 5 cm (scale
undefined, ATE not meaningful), 'poor fit' = ATE > 0.3 x spread (a collapsed or wrong map can have a small ATE on a
small scene). Medians of ATE leave out runs without a trajectory and short ones; poor fits are counted, not dropped.

Usage: score_ext_1103.py [--root results/ext_1103] [--load logs/ext_1103_load.log]
Writes runs.csv and RESULTS.md into the batch folder.
"""
import argparse
import csv
import json
import re
import statistics as st
from collections import Counter
from pathlib import Path

import numpy as np

ARMS = ["alvaar", "wasm-stock", "page-fast", "page-next", "page-beta", "page-betagyro", "page-mapfirst"]
SCEN = ["ar", "gesture", "orbit", "rotate", "shake", "slide", "walk"]
R = Path("/mnt/h/Research")


def jload(p):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def info(p):
    d = {}
    if p.exists():
        for l in p.read_text().splitlines():
            k, _, v = l.partition(" ")
            d[k] = v
    return d


def tum(p):
    if not p.exists():
        return None
    a = np.loadtxt(p, comments="#", ndmin=2)
    return a if len(a) else None


def gt_spread(est, gt):
    """RMS distance of the ground-truth positions from their centroid over the estimate's frames."""
    if est is None:
        return None
    t = est[:, 0]
    t = t[(t >= gt[0, 0]) & (t <= gt[-1, 0])]
    if len(t) < 3:
        return None
    g = np.stack([np.interp(t, gt[:, 0], gt[:, j]) for j in (1, 2, 3)], 1)
    return float(np.sqrt(((g - g.mean(0)) ** 2).sum(1).mean()))


def web_track(d, frames):
    """tracked / largest-map frames, maps used, first tracked frame index, from trace.txt (state 2 = OK; 5 = OK_KLT)."""
    per_map, first, n = Counter(), None, 0
    for l in (d / "trace.txt").read_text().splitlines() if (d / "trace.txt").exists() else []:
        f = l.split()
        if len(f) < 10:
            continue
        n += 1
        if int(f[1]) in (2, 5):
            per_map[int(f[-2]) if len(f) > 12 else 0] += 1
            first = int(f[0]) if first is None else first
    return sum(per_map.values()), max(per_map.values(), default=0), len(per_map), first


def alva_track(d, ri):
    first = None
    if (d / "frames.txt").exists():
        for l in (d / "frames.txt").read_text().splitlines():
            f = l.split()
            if len(f) >= 3 and not l.startswith("#") and f[2] == "1":
                first = int(f[0])
                break
    return int(ri.get("tracked_frames", 0)), int(ri.get("longest_segment", 0)), int(ri.get("segments", 0)), first


def run_row(arm, name, d, gts):
    kind, _, seq = name.partition("_")
    ri = info(d / "run_info.txt")
    frames = int(ri["frames"]) if "frames" in ri else None
    log = (d / "run.log").read_text(errors="replace") if (d / "run.log").exists() else ""
    rc = re.search(r"run failed \(exit (\d+)\)", log)
    tr, big, maps, first = alva_track(d, ri) if arm == "alvaar" else web_track(d, frames)
    tt = []
    if (d / "track_times.txt").exists():
        for l in (d / "track_times.txt").read_text().splitlines():
            f = l.split()
            if len(f) >= 2 and not l.startswith("#"):
                tt.append(float(f[1]))
    fps = frames / float(ri["seq_s"]) if frames and "seq_s" in ri else None
    r = {"arm": arm, "kind": kind, "seq": seq, "scen": re.sub(r"\d+$", "", seq) if kind == "synth" else seq,
         "run": d.name, "frames": frames, "processed": int(ri["processed"]) if "processed" in ri else None,
         "rate": float(ri["effective_rate"]) if "effective_rate" in ri else None,
         "exit": int(rc.group(1)) if rc else 0, "error": ri.get("error", "none"), "track_ms": st.mean(tt) if tt else None,
         "tracked": tr / frames if frames else 0.0, "largest": big / frames if frames else 0.0, "maps": maps,
         "first_s": first / fps if first is not None and fps else None,
         "online_ate": None, "online_cov": None, "spread": None, "flag": None, "final_ate": None, "final_cov": None}
    if kind == "synth":
        on = jload(d / ("metrics.json" if arm == "alvaar" else "metrics_online.json"))
        if on:
            r["online_ate"], r["online_cov"] = on["ate_sim3"]["rmse"], on["coverage"]
            sp = gt_spread(tum(d / ("traj_tum.txt" if arm == "alvaar" else "online_tum.txt")), gts[seq])
            r["spread"] = sp
            r["flag"] = "short" if sp is not None and sp < 0.05 else (
                "poor fit" if sp is not None and r["online_ate"] > 0.3 * sp else "")
        m = jload(d / "metrics.json")
        if arm != "alvaar" and m:
            r["final_ate"], r["final_cov"] = m["ate_sim3"]["rmse"], m["coverage"]
    return r


def med(v):
    v = [x for x in v if x is not None]
    return st.median(v) if v else None


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}"


def grid(rows, arms, groups, cell, title, gkey="scen", label=lambda g: g):
    L = [f"### {title}", "", "| | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for g in groups:
        L.append(f"| {label(g)} | " + " | ".join(cell([r for r in rows if r["arm"] == a and r[gkey] == g]) for a in arms) + " |")
    L.append(f"| **all** | " + " | ".join(cell([r for r in rows if r["arm"] == a]) for a in arms) + " |")
    return L + [""]


def c_share(key):
    return lambda rr: f"{pct(med([r[key] for r in rr]))}" if rr else ""


def c_maps(rr):
    return f"{med([r['maps'] for r in rr]):.0f} ({max(r['maps'] for r in rr)})" if rr else ""


def c_ate(rr):
    if not rr:
        return ""
    ok = [r for r in rr if r["online_ate"] is not None and r["flag"] != "short"]
    v = med([r["online_ate"] for r in ok])
    pf = sum(r["flag"] == "poor fit" for r in ok)
    none = sum(r["online_ate"] is None for r in rr)
    sh = sum(r["flag"] == "short" for r in rr)
    tail = ", ".join(x for x in (f"{pf} pf" if pf else "", f"{none} none" if none else "", f"{sh} short" if sh else "") if x)
    return ("–" if v is None else f"{100 * v:.1f}") + (f" ({tail})" if tail else "")


def c_final(rr):
    v = med([r["final_ate"] for r in rr])
    c = med([r["final_cov"] for r in rr])
    return "" if not rr or rr[0]["arm"] == "alvaar" else ("–" if v is None else f"{100 * v:.1f} ({pct(c)} %)")


def paired(rows, arms, seqs, key, ref="alvaar", higher_better=True):
    """Per sequence, the median of 2 runs vs the reference arm's: sequences better / worse / equal."""
    L = []
    for a in arms:
        if a == ref:
            continue
        b = w = e = 0
        for s in seqs:
            x = med([r[key] for r in rows if r["arm"] == a and r["seq"] == s])
            y = med([r[key] for r in rows if r["arm"] == ref and r["seq"] == s])
            if x is None or y is None:
                continue
            if abs(x - y) < 1e-9:
                e += 1
            elif (x > y) == higher_better:
                b += 1
            else:
                w += 1
        L.append(f"| {a} | {b} | {w} | {e} |")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=R / "results/ext_1103")
    ap.add_argument("--load", type=Path, default=R / "logs/ext_1103_load.log")
    a = ap.parse_args()

    gts = {p.stem[len("synth_"):]: np.loadtxt(p, comments="#", ndmin=2) for p in (R / "results/gt").glob("synth_*.txt")}
    labels = {}
    for l in (R / "datasets/phone/labels.txt").read_text().splitlines():
        if l.strip() and not l.startswith("#"):
            f = l.split(None, 2)
            labels[f[0]] = f"{f[1]}: {f[2].split('  ')[0].strip()}" if len(f) > 2 else f[1]
    rows = []
    for arm in ARMS:
        for sd in sorted((a.root / arm).glob("*_*")):
            for d in sorted(sd.glob("wasm_seed*")):
                rows.append(run_row(arm, sd.name, d, gts))
    with (a.root / "runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    syn = [r for r in rows if r["kind"] == "synth"]
    ph = [r for r in rows if r["kind"] == "phone"]
    sseq = sorted({r["seq"] for r in syn}, key=lambda s: (SCEN.index(re.sub(r"\d+$", "", s)), s))
    pseq = sorted({r["seq"] for r in ph})

    L = ["# ext_1103 results", "",
         "Generated by scripts/experiments/2026-11-03/score_ext_1103.py. Raw tables only: no ranking rule was fixed "
         "for this batch before the results. Frozen build snap-84c1f17 (WASM), AlvaAR released bundle; --rt-wall, "
         "4 runs at a time, run order shuffled; seeds / runs 1–2. Tracked = frames with a pose (WebVIO: state OK; "
         "coasting RECENTLY_LOST frames not counted), largest map = the map / segment with the most tracked frames, "
         "both as a share of all frames. ATE = Sim3 RMSE of the online poses of the largest map; pf = poor fit "
         "(ATE > 0.3 × the ground truth's spread over the same frames), short = spread < 5 cm (ATE not meaningful). "
         "Track ms are with 4 runs at a time: not timing results.", ""]
    L += ["## Run health", "", "| arm | runs | exit ≠ 0 | stopped by an error | effective rate min / median | frames processed (min share) | "
          "mean track ms (median run) |", "|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        rr = [r for r in rows if r["arm"] == arm]
        rates = [r["rate"] for r in rr if r["rate"] is not None]
        sh = [r["processed"] / r["frames"] for r in rr if r["frames"]]
        L.append(f"| {arm} | {len(rr)} | {sum(r['exit'] != 0 for r in rr)} | {sum(r['error'] != 'none' for r in rr)} | {min(rates):.3f} / {st.median(rates):.3f} | "
                 f"{min(sh):.3f} | {med([r['track_ms'] for r in rr]):.1f} |")
    L += [""] + [f"Stopped by an error: {r['arm']} {r['kind']}_{r['seq']} {r['run']}: {r['error']} after "
                 f"{r['processed']} of {r['frames']} frames (the frames after it count as untracked)." for r in rows
                 if r["error"] != "none"]
    if a.load.exists():
        la, run = [], []
        for l in a.load.read_text().splitlines():
            m = re.match(r"--- \S+ (\S+) \S+ \S+ running=(\d+)", l)
            if m:
                la.append(float(m.group(1)))
                run.append(int(m.group(2)))
        if la:
            L += ["", f"Machine load ({len(la)} samples): 1-min load average median {st.median(la):.2f}, max "
                  f"{max(la):.2f}; runs in flight median {st.median(run) / 2:g}, max {max(run) / 2:g} (pgrep counts the "
                  "timeout wrapper and node: 2 processes per run).", ""]

    L += ["## Synthetic set (42 sequences, 7 scenarios × 6; 12 runs per scenario and arm)", ""]
    L += grid(syn, ARMS, SCEN, c_share("tracked"), "Tracked share, median (%)")
    L += grid(syn, ARMS, SCEN, c_share("largest"), "Largest-map share, median (%)")
    L += grid(syn, ARMS, SCEN, c_maps, "Maps (segments) per run, median (max)")
    L += grid(syn, ARMS, SCEN, c_ate, "Online Sim3 ATE of the largest map, median (cm); (n pf / none / short)")
    L += grid(syn, ARMS[1:], SCEN, c_final, "Final Sim3 ATE of the largest map (after BA / loop closure), median cm (its coverage, median)")
    L += ["Shake: the ground truth's spread is under 5 cm in every run, so its final ATE is not meaningful either.", ""]
    L += grid(syn, ARMS, SCEN, lambda rr: "" if not rr else ("–" if med([r["first_s"] for r in rr]) is None else
              f"{med([r['first_s'] for r in rr]):.1f}"), "First tracked frame, median (s)")
    L += ["### Per sequence vs AlvaAR (median of 2 runs each)", "",
          "| arm | largest-map share higher | lower | equal |", "|---|---|---|---|"]
    L += paired(syn, ARMS, sseq, "largest")
    L += ["", "| arm | tracked share higher | lower | equal |", "|---|---|---|---|"]
    L += paired(syn, ARMS, sseq, "tracked")
    L += ["", "### Per sequence: largest-map share % / online ATE cm (median of 2 runs)", "",
          "| sequence | " + " | ".join(ARMS) + " |", "|---|" + "---|" * len(ARMS)]
    for s in sseq:
        cells = []
        for arm in ARMS:
            rr = [r for r in syn if r["arm"] == arm and r["seq"] == s]
            cells.append(f"{pct(med([r['largest'] for r in rr]))} / {c_ate(rr)}")
        L.append(f"| {s} | " + " | ".join(cells) + " |")

    L += ["", "## Pixel 9 recordings (15, 18 Oct; no ground truth; 2 runs per recording and arm)", "",
          "Labels from datasets/phone/labels.txt. Gyro presets (next, beta, betagyro, mapfirst) read the recorded "
          "DeviceMotion gyro with the Pixel 9's −88 ms offset; AlvaAR and stock use the images only.", ""]
    lab = lambda s: f"{s.replace('rec_20261018_', '')} {labels.get(s, '')}"
    L += grid(ph, ARMS, pseq, c_share("tracked"), "Tracked share, median of 2 (%)", "seq", lab)
    L += grid(ph, ARMS, pseq, c_share("largest"), "Largest-map share, median of 2 (%)", "seq", lab)
    L += grid(ph, ARMS, pseq, c_maps, "Maps (segments) per run, median (max)", "seq", lab)
    L += grid(ph, ARMS, pseq, lambda rr: "" if not rr else ("–" if med([r["first_s"] for r in rr]) is None else
              f"{med([r['first_s'] for r in rr]):.1f}"), "First tracked frame, median (s)", "seq", lab)
    L += ["### Per recording vs AlvaAR (median of 2 runs each)", "",
          "| arm | largest-map share higher | lower | equal |", "|---|---|---|---|"]
    L += paired(ph, ARMS, pseq, "largest")
    L += ["", "| arm | tracked share higher | lower | equal |", "|---|---|---|---|"]
    L += paired(ph, ARMS, pseq, "tracked")
    L += ["", "### Per recording vs page-beta (the gyro attitude options on the real phone gyro)", "",
          "| arm | largest-map share higher | lower | equal |", "|---|---|---|---|"]
    L += paired(ph, ARMS, pseq, "largest", ref="page-beta")
    L += ["", "| arm | fewer maps | more | equal |", "|---|---|---|---|"]
    L += paired(ph, ARMS, pseq, "maps", ref="page-beta", higher_better=False)
    (a.root / "RESULTS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
