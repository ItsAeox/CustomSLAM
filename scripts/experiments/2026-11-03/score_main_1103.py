#!/usr/bin/env python3
"""Scores main_1103 with the ranking rule written in claude/EXPERIMENTS.md before any result:
(1) bad-run rate on online poses (Sim3 ATE > 200 mm or no trajectory) over all 17 sequences, (2) median online Sim3
ATE, (3) online coverage; tuning subset and held-out sequences separately; two arms are tied when their bad-run counts
do not differ (Fisher p > 0.05) and their ATE ratio (median over sequences of the per-sequence median of good runs,
as eval/screen_table.py) is inside the ±45 % noise floor.

AlvaAR runs only on EuRoC (no fisheye model), so it is ranked against the others on the 11 EuRoC sequences only.

Writes runs.csv and RESULTS.md into the batch folder.
Usage: score_main_1103.py [--root results/main_1103] [--load logs/main_1103_load.log]
"""
import argparse
import csv
import json
import re
import statistics as st
from pathlib import Path

from scipy.stats import fisher_exact

ARMS = ["native-stock", "wasm-stock", "page-fast", "page-next", "page-beta", "page-betagyro", "page-mapfirst", "alvaar"]
TUNING = {"euroc_MH_04_difficult", "euroc_MH_05_difficult", "euroc_V1_03_difficult", "euroc_V2_03_difficult",
          "tumvi_dataset-room1_512_16", "tumvi_dataset-room2_512_16", "tumvi_dataset-room3_512_16",
          "tumvi_dataset-room5_512_16"}
BAD_M = 0.200
NOISE = 1.45


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


def short(seq):
    s = seq.replace("euroc_", "").replace("tumvi_dataset-", "").replace("_512_16", "")
    return re.sub(r"_(easy|medium|difficult)$", "", s)


def run_row(arm, seq, d):
    m = jload(d / "metrics.json")
    on = m if arm == "alvaar" else jload(d / "metrics_online.json")
    mm = jload(d / "metrics_multimap.json")
    ri = info(d / "run_info.txt")
    log = (d / "run.log").read_text(errors="replace") if (d / "run.log").exists() else ""
    rc = re.search(r"run failed \(exit (\d+)\)", log)
    tt = []
    if (d / "track_times.txt").exists():
        for l in (d / "track_times.txt").read_text().split("\n"):
            f = l.split()
            if len(f) >= 2:
                tt.append(float(f[1]))
    r = {
        "arm": arm, "seq": seq, "run": d.name,
        "online_ate": on["ate_sim3"]["rmse"] if on else None,
        "online_cov": on["coverage"] if on else 0.0,
        "final_ate": None if arm == "alvaar" or not m else m["ate_sim3"]["rmse"],
        "final_cov": None if arm == "alvaar" or not m else m["coverage"],
        "maps": mm["maps_total"] if mm else None,
        "union_cov": mm["coverage_union"] if mm else None,
        "pooled_ate": mm["ate_sim3_pooled"] if mm else None,
        "rate": float(ri["effective_rate"]) if "effective_rate" in ri else None,
        "frames": int(ri["frames"]) if "frames" in ri else None,
        "processed": int(ri["processed"]) if "processed" in ri else None,
        "exit": int(rc.group(1)) if rc else 0,
        "track_ms": st.mean(tt) if tt else None,
    }
    r["bad"] = r["online_ate"] is None or r["online_ate"] > BAD_M
    return r


def med(v):
    v = [x for x in v if x is not None]
    return st.median(v) if v else None


def fmt(x, k=1000, nd=0):
    return "–" if x is None else f"{x * k:.{nd}f}"


def ate_ratio(rows, a, b, seqs):
    """Median over sequences of (median good-run online ATE of a) / (same of b), and how many sequences a is lower."""
    rs = []
    for s in seqs:
        ma = med([r["online_ate"] for r in rows if r["arm"] == a and r["seq"] == s and not r["bad"]])
        mb = med([r["online_ate"] for r in rows if r["arm"] == b and r["seq"] == s and not r["bad"]])
        if ma and mb:
            rs.append(ma / mb)
    return (st.median(rs) if rs else None), sum(1 for x in rs if x < 1), len(rs)


def summary(rows, arms, seqs):
    out = []
    for a in arms:
        rr = [r for r in rows if r["arm"] == a and r["seq"] in seqs]
        if not rr:
            continue
        good = [r for r in rr if not r["bad"]]
        # median with failures counted as worst (sensitivity check; not the rule)
        allv = sorted((r["online_ate"] if r["online_ate"] is not None else float("inf")) for r in rr)
        out.append({
            "arm": a, "n": len(rr), "bad": sum(r["bad"] for r in rr), "notraj": sum(r["online_ate"] is None for r in rr),
            "med_good": med([r["online_ate"] for r in good]), "med_all": st.median(allv),
            "cov": st.mean(r["online_cov"] for r in rr),
            "final_med": med([r["final_ate"] for r in rr if r["final_ate"] is not None and r["final_ate"] <= BAD_M]),
            "final_bad": sum(1 for r in rr if r["final_ate"] is None or r["final_ate"] > BAD_M) if a != "alvaar" else None,
            "maps": st.mean(r["maps"] for r in rr if r["maps"] is not None) if any(r["maps"] is not None for r in rr) else None,
        })
    out.sort(key=lambda s: (s["bad"] / s["n"], s["med_good"] if s["med_good"] is not None else 9, -s["cov"]))
    return out


def ties(rows, ranked, seqs, ref=None):
    """Each arm against ref (default: the leader). Ties are not transitive, so neighbours are not chained."""
    lines = []
    a = next(x for x in ranked if x["arm"] == ref) if ref else ranked[0]
    for b in ranked:
        if b is a:
            continue
        p = fisher_exact([[a["bad"], a["n"] - a["bad"]], [b["bad"], b["n"] - b["bad"]]])[1]
        ratio, lower, n = ate_ratio(rows, b["arm"], a["arm"], seqs)
        tied = p > 0.05 and ratio is not None and 1 / NOISE <= ratio <= NOISE
        lines.append(f"| {a['arm']} vs {b['arm']} | {a['bad']}/{a['n']} vs {b['bad']}/{b['n']} | {p:.3f} | "
                     f"{'–' if ratio is None else f'{ratio:.2f}'} ({lower}/{n} sequences lower for {b['arm']}) | "
                     f"{'**tied**' if tied else 'separated'} |")
    return lines


def table(rows, arms, seqs, title, extra_refs=()):
    extra_refs = list(extra_refs)
    s = summary(rows, arms, seqs)
    L = [f"### {title}", "",
         "| rank | arm | runs | bad (no traj.) | online ATE median, good runs (mm) | median incl. failures (mm) | "
         "online coverage | final ATE median (mm) | final bad | maps / run |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for i, x in enumerate(s, 1):
        L.append(f"| {i} | {x['arm']} | {x['n']} | {x['bad']} ({x['notraj']}) | {fmt(x['med_good'], nd=1)} | "
                 f"{'fail' if x['med_all'] == float('inf') else fmt(x['med_all'], nd=1)} | {x['cov']:.3f} | "
                 f"{fmt(x['final_med'], nd=1)} | {'–' if x['final_bad'] is None else x['final_bad']} | "
                 f"{'–' if x['maps'] is None else format(x['maps'], '.2f')} |")
    for ref in [None] + extra_refs:
        if ref and not any(x["arm"] == ref for x in s):
            continue
        L += ["", f"Each arm against {ref or 'the leader'} (Fisher's exact test on bad runs; ATE ratio = other / "
              f"{ref or 'leader'}, median over sequences of the per-sequence median of good runs; tied = p > 0.05 and "
              "ratio inside 1/1.45–1.45):", "", "| pair | bad | Fisher p | ATE ratio | |", "|---|---|---|---|---|"]
        L += ties(rows, s, seqs, ref)
    return L + [""], s


def per_seq(rows, arms, seqs, key, title, unit_k=1000, nd=0, all_runs=False):
    """all_runs: median over every run (no trajectory = worst), as ORB-SLAM3's paper reports the median of executions."""
    L = [f"### {title}", "", "| sequence | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for s in seqs:
        cells = []
        for a in arms:
            rr = [r for r in rows if r["arm"] == a and r["seq"] == s]
            if not rr:
                cells.append("")
                continue
            nb = sum(1 for r in rr if r[key] is None or r[key] > BAD_M)
            if all_runs:
                m = st.median(r[key] if r[key] is not None else float("inf") for r in rr)
                cells.append(("fail" if m == float("inf") else fmt(m, unit_k, nd)) + (f" ({nb}✗)" if nb else ""))
                continue
            v = [r[key] for r in rr if r[key] is not None and r[key] <= BAD_M]
            cells.append(f"{fmt(med(v), unit_k, nd)}" + (f" ({nb}✗)" if nb else ""))
        L.append(f"| {short(s)}{' †' if s in TUNING else ''} | " + " | ".join(cells) + " |")
    return L + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/mnt/h/Research/results/main_1103"))
    ap.add_argument("--load", type=Path, default=Path("/mnt/h/Research/logs/main_1103_load.log"))
    ap.add_argument("--arms", nargs="+", default=ARMS, help="arm folders, in table order")
    ap.add_argument("--title", default="main_1103 results")
    a = ap.parse_args()
    arms_all = a.arms

    rows = []
    for arm in arms_all:
        for sd in sorted((a.root / arm).glob("*_*")):
            for d in sorted(sd.glob("*_seed*")):
                rows.append(run_row(arm, sd.name, d))
    with (a.root / "runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    seqs = sorted({r["seq"] for r in rows if r["arm"] != "alvaar"})
    euroc = [s for s in seqs if s.startswith("euroc")]
    web = [x for x in arms_all if x != "alvaar"]
    page = [x for x in web if x.startswith("page-")]
    held = [s for s in seqs if s not in TUNING]

    L = [f"# {a.title}", "",
         "Generated by scripts/experiments/2026-11-03/score_main_1103.py. Online = the poses the tracker reported at "
         "each frame (largest map; AlvaAR: longest segment), final = ORB-SLAM3's trajectory after BA / loop closure "
         "(largest map). ATE = RMSE after Sim(3) alignment. Bad = online ATE > 200 mm or no trajectory. Medians of "
         "ATE are over good runs only, so read them after the bad-run counts. † = tuning subset. 4 runs at a time: "
         "times here are not timing results.", ""]

    # health
    L += ["## Run health", "", "| arm | runs | exit ≠ 0 | effective rate min / median | frames processed (min share) | "
          "mean track ms (median run, 4 at a time) |", "|---|---|---|---|---|---|"]
    for arm in arms_all:
        rr = [r for r in rows if r["arm"] == arm]
        rates = [r["rate"] for r in rr if r["rate"] is not None]
        shares = [r["processed"] / r["frames"] for r in rr if r["frames"]]
        L.append(f"| {arm} | {len(rr)} | {sum(r['exit'] != 0 for r in rr)} | "
                 f"{min(rates):.3f} / {st.median(rates):.3f} | {min(shares):.3f} | "
                 f"{fmt(med([r['track_ms'] for r in rr]), 1, 1)} |" if rates else f"| {arm} | {len(rr)} | | | | |")
    if a.load.exists():
        la, run = [], []
        for l in a.load.read_text().splitlines():
            m = re.match(r"--- \S+ (\S+) \S+ \S+ running=(\d+)", l)
            if m:
                la.append(float(m.group(1)))
                run.append(int(m.group(2)))
        L += ["", f"Machine load (logs/main_1103_load.log, {len(la)} samples): 1-min load average median "
              f"{st.median(la):.2f}, max {max(la):.2f}; runs in flight median {st.median(run) / 2:g}, max {max(run) / 2:g} "
              "(pgrep counts the timeout wrapper and node: 2 processes per run).", ""]

    L += ["## Ranking (the rule fixed before the batch)", ""]
    t, s_all = table(rows, web, seqs, "WebVIO arms, all 17 sequences")
    L += t
    L += table(rows, web, [s for s in seqs if s in TUNING], "WebVIO arms, tuning subset (8)")[0]
    L += table(rows, web, [s for s in seqs if s not in TUNING], "WebVIO arms, held-out (9)")[0]
    if "alvaar" in arms_all:
        L += table(rows, arms_all, euroc, "All arms incl. AlvaAR, EuRoC only (11)", ["alvaar"])[0]
    if len(page) > 1 and len(page) < len(web):
        L += ["## Phone pool: the page presets only (what the phone can run)", ""]
        L += table(rows, page, seqs, "Page presets, all 17 sequences")[0]
        L += table(rows, page, held, "Page presets, held-out (9)")[0]
        L += table(rows, page, [s for s in seqs if s.startswith("tumvi")], "Page presets, TUM-VI rooms only (6)")[0]

    L += ["## Per sequence", ""]
    L += per_seq(rows, arms_all, seqs, "online_ate", "Online Sim3 ATE, median of good runs (mm); (n✗) = bad runs of 5")
    L += per_seq(rows, web, seqs, "final_ate", "Final Sim3 ATE, median of runs ≤ 200 mm (mm); (n✗) = runs over 200 mm or none")
    L += per_seq(rows, arms_all, seqs, "online_ate", "Online Sim3 ATE, median of all 5 runs (mm; failures included, "
                 "as the median of executions in ORB-SLAM3's paper)", all_runs=True)
    L += per_seq(rows, web, seqs, "final_ate", "Final Sim3 ATE, median of all 5 runs (mm; failures included)", all_runs=True)
    L += ["### Online coverage (mean over runs) and maps per run (mean)", "",
          "| sequence | " + " | ".join(arms_all) + " |", "|---|" + "---|" * len(arms_all)]
    for s in seqs:
        cells = []
        for arm in arms_all:
            rr = [r for r in rows if r["arm"] == arm and r["seq"] == s]
            if not rr:
                cells.append("")
                continue
            mp = [r["maps"] for r in rr if r["maps"] is not None]
            cells.append(f"{st.mean(r['online_cov'] for r in rr):.2f} / {st.mean(mp) if mp else 0:.1f}")
        L.append(f"| {short(s)} | " + " | ".join(cells) + " |")
    # final trajectory (after BA / loop closure) vs the online poses, per arm
    L += ["", "## Final vs online trajectory", "",
          "Median over sequences of (median final ATE / median online ATE), runs ≤ 200 mm on both; < 1 = bundle "
          "adjustment / loop closure improve on what the tracker showed.", "",
          "| arm | final / online | sequences where final is lower |", "|---|---|---|"]
    for arm in web:
        rs = []
        for s in seqs:
            rr = [r for r in rows if r["arm"] == arm and r["seq"] == s and not r["bad"]
                  and r["final_ate"] is not None and r["final_ate"] <= BAD_M]
            if rr:
                rs.append(med([r["final_ate"] for r in rr]) / med([r["online_ate"] for r in rr]))
        if rs:
            L.append(f"| {arm} | {st.median(rs):.2f} | {sum(x < 1 for x in rs)}/{len(rs)} |")
    if "native-stock" not in arms_all or "wasm-stock" not in arms_all:
        (a.root / "RESULTS.md").write_text("\n".join(L) + "\n")
        print("\n".join(L))
        return
    L += ["", "## Parity: native-stock vs wasm-stock (same code and settings, browser runtime vs native)", "",
          "| sequence | native final (mm) | wasm final (mm) | native online (mm) | wasm online (mm) | bad n / w |",
          "|---|---|---|---|---|---|"]
    for s in seqs:
        n = [r for r in rows if r["arm"] == "native-stock" and r["seq"] == s]
        w = [r for r in rows if r["arm"] == "wasm-stock" and r["seq"] == s]
        g = lambda rr, k: med([r[k] for r in rr if r[k] is not None and r[k] <= BAD_M])
        L.append(f"| {short(s)} | {fmt(g(n, 'final_ate'), nd=1)} | {fmt(g(w, 'final_ate'), nd=1)} | "
                 f"{fmt(g(n, 'online_ate'), nd=1)} | {fmt(g(w, 'online_ate'), nd=1)} | "
                 f"{sum(r['bad'] for r in n)} / {sum(r['bad'] for r in w)} |")
    for k in ("final_ate", "online_ate"):
        rs = []
        for s in seqs:
            g = lambda arm: med([r[k] for r in rows if r["arm"] == arm and r["seq"] == s and r[k] is not None and r[k] <= BAD_M])
            if g("native-stock") and g("wasm-stock"):
                rs.append((g("wasm-stock") / g("native-stock"), short(s)))
        rs.sort()
        L += ["", f"{k.split('_')[0]}: wasm / native per sequence, median {st.median(x for x, _ in rs):.2f} over {len(rs)} "
              f"sequences; range {rs[0][0]:.2f} ({rs[0][1]}) – {rs[-1][0]:.2f} ({rs[-1][1]}); over 1.45: "
              + (", ".join(f"{n} {x:.2f}" for x, n in rs if x > NOISE) or "none")]
    wr = [r for r in rows if r["arm"] == "wasm-stock"]
    lag = [r for r in wr if r["rate"] is not None and r["rate"] < 0.98]
    L += ["", f"wasm-stock runs slower than real time (effective rate < 0.98): {len(lag)} of {len(wr)}; bad among them "
          f"{sum(r['bad'] for r in lag)}, among the rest {sum(r['bad'] for r in wr if r not in lag)} of {len(wr) - len(lag)}. "
          "Slowest runs: " + ", ".join(f"{short(r['seq'])} {r['run'].split('_')[-1]} {r['rate']:.2f}"
                                       f" ({fmt(r['online_ate'])} mm)" for r in sorted(lag, key=lambda r: r["rate"])[:8])]
    (a.root / "RESULTS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
