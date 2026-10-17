#!/usr/bin/env python3
"""Round-2 screen table (2026-10-17): one row per arm of results/<screen>/<arm>/<sequence>/<run>/.

Columns: bad runs (failure, coverage < 0.5, Sim3 ATE > 200 mm), paired accuracy vs the baseline arm (per sequence:
median Sim3 ATE of the good runs; ratio arm/baseline, median over sequences and how many sequences improve), frames
tracked / relocalising / before the first map (track_times state column), maps per run, mean tracking ms, recoveries.

Usage: screen_table.py <screen dir> [baseline arm, default preset-fast]
"""
import json
import statistics as st
import sys
from pathlib import Path


def run_stats(r):
    m = json.loads((r / "metrics.json").read_text()) if (r / "metrics.json").exists() else {}
    ate = m.get("ate_sim3", {}).get("rmse")
    cov = m.get("coverage", 0)
    bad = ate is None or cov < 0.5 or ate * 1000 > 200
    tt = [l.split() for l in (r / "track_times.txt").read_text().splitlines()] if (r / "track_times.txt").exists() else []
    st_ = [int(x[2]) for x in tt if len(x) >= 3]
    ms = [float(x[1]) for x in tt if len(x) >= 2]
    log = (r / "run.log").read_text(errors="ignore") if (r / "run.log").exists() else ""
    rec = {}
    for line in log.splitlines():
        if line.startswith("RECOVERY-STATS"):
            rec = dict(kv.split("=") for kv in line.split()[1:])
    first_ok = next((i for i, s in enumerate(st_) if s in (2, 5)), len(st_))
    return dict(bad=bad, ate=None if bad else ate * 1000, n=len(st_), ok=sum(s in (2, 5) for s in st_),
                lost=sum(s in (3, 4) for s in st_), init=first_ok, ms=sum(ms) / len(ms) if ms else 0,
                maps=log.count("Creation of new map") / 2, coast=int(rec.get("coast_recoveries", 0)),
                reloc=int(rec.get("reloc_recoveries", 0)))


def main():
    root = Path(sys.argv[1])
    base = sys.argv[2] if len(sys.argv) > 2 else "preset-fast"
    arms = sorted(p.name for p in root.iterdir() if p.is_dir())
    data = {a: {s.name: [run_stats(r) for r in sorted(s.iterdir()) if r.is_dir()] for s in (root / a).iterdir() if s.is_dir()}
            for a in arms}

    def seq_med(a, s):
        v = [x["ate"] for x in data[a].get(s, []) if x["ate"] is not None]
        return st.median(v) if v else None

    print(f"{root.name}: baseline {base}")
    print(f"{'arm':20s} {'bad':>7s} {'ATE ratio':>10s} {'better':>7s} {'tracked':>8s} {'lost':>6s} {'init fr':>8s} "
          f"{'maps':>5s} {'ms':>6s} {'coast':>6s} {'reloc':>6s}")
    for a in [base] + [x for x in arms if x != base]:
        runs = [x for v in data[a].values() for x in v]
        if not runs:
            continue
        ratios, better = [], 0
        for s in data[a]:
            b, m = seq_med(base, s), seq_med(a, s)
            if b and m:
                ratios.append(m / b)
                better += m < b
        n = sum(x["n"] for x in runs)
        print(f"{a:20s} {sum(x['bad'] for x in runs):3d}/{len(runs):<3d} "
              f"{(st.median(ratios) if ratios and a != base else 1.0):10.2f} {f'{better}/{len(ratios)}' if a != base else '':>7s} "
              f"{sum(x['ok'] for x in runs) / n:8.3f} {sum(x['lost'] for x in runs) / n:6.3f} "
              f"{st.mean(x['init'] for x in runs):8.1f} {st.mean(x['maps'] for x in runs):5.2f} "
              f"{st.mean(x['ms'] for x in runs):6.2f} {sum(x['coast'] for x in runs):6d} {sum(x['reloc'] for x in runs):6d}")


if __name__ == "__main__":
    main()
