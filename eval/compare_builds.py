#!/usr/bin/env python3
"""Compare two builds (default: native vs wasm) over a seed sweep (scripts/seed_sweep.sh output).

Monocular ORB-SLAM3 is not deterministic (seeds, thread timing), so single runs prove nothing. This reports, per
sequence, the median and bad-run count of each build, and pooled over all sequences:
  - bad-run rate (rotation ATE > --bad-rot deg, or no trajectory), Fisher's exact test (two-sided);
  - per-sequence Mann-Whitney U on rotation ATE (alignment-free of scale) and Sim(3) ATE.

Usage: compare_builds.py <sweep_dir> [<sweep_dir> ...] [--a native] [--b wasm] [--bad-rot 5]
       compare_builds.py --across <sweep_A> <sweep_B> --a wasm   (same build, two settings variants A vs B)
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu


def collect(seq: Path, flavour: str):
    rows = []
    for r in sorted(seq.glob(f"{flavour}_seed*")):
        m = r / "metrics.json"
        log = r / "run.log"
        if m.exists():
            d = json.loads(m.read_text())
            rows.append((d["ate_se3_rot_deg"]["rmse"], d["ate_sim3"]["rmse"] * 1000, d["coverage"]))
        elif log.exists() and ("run failed" in log.read_text(errors="replace") or (r / "run_info.txt").exists()
                               and not (r / "traj_tum.txt").exists() and "Saving" in log.read_text(errors="replace")):
            rows.append((np.inf, np.inf, 0.0))  # failure: crashed or finished without a trajectory
        # else: still running -> skipped
    return np.array(rows).reshape(-1, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweeps", type=Path, nargs="+")
    ap.add_argument("--a", default="native")
    ap.add_argument("--b", default="wasm")
    ap.add_argument("--bad-rot", type=float, default=5.0)
    ap.add_argument("--across", action="store_true", help="compare build --a between two sweep dirs (A vs B)")
    a = ap.parse_args()
    if a.across:
        return across(a)

    seqs = {}
    for sw in a.sweeps:
        for s in sorted(p for p in sw.iterdir() if p.is_dir()):
            seqs.setdefault(s.name, []).append(s)

    print(f"{'sequence':22s} | {a.a:>26s} | {a.b:>26s} | MWU p (rot)")
    print(f"{'':22s} | {'n bad  rot° med  Sim3 mm med':>26s} | {'n bad  rot° med  Sim3 mm med':>26s} |")
    tot = {a.a: [0, 0], a.b: [0, 0]}
    for name, dirs in seqs.items():
        A = np.vstack([collect(d, a.a) for d in dirs])
        B = np.vstack([collect(d, a.b) for d in dirs])
        if not len(A) or not len(B):
            continue
        cells = []
        for fl, X in ((a.a, A), (a.b, B)):
            bad = int((X[:, 0] > a.bad_rot).sum())
            tot[fl][0] += bad
            tot[fl][1] += len(X)
            cells.append(f"{len(X):2d} {bad:2d}  {np.median(X[:, 0]):7.2f}  {np.median(X[:, 1]):10.1f}")
        p = mannwhitneyu(A[:, 0], B[:, 0], alternative="two-sided").pvalue
        print(f"{name:22s} | {cells[0]:>26s} | {cells[1]:>26s} | {p:.3f}")

    (ba, na), (bb, nb) = tot[a.a], tot[a.b]
    _, p = fisher_exact([[ba, na - ba], [bb, nb - bb]])
    print(f"\npooled bad runs (rot ATE > {a.bad_rot}° or failure): {a.a} {ba}/{na} ({100 * ba / max(na, 1):.0f}%), "
          f"{a.b} {bb}/{nb} ({100 * bb / max(nb, 1):.0f}%), Fisher exact p = {p:.3f}")


def across(a):
    """Same build (--a) in two sweep dirs: variant A (first dir) vs variant B (second dir)."""
    A, B = a.sweeps[0], a.sweeps[1]
    fl = a.a
    print(f"{fl}: {A.name} (A) vs {B.name} (B)")
    print(f"{'sequence':22s} | {'n bad  rot° med  Sim3 mm med':>26s} | {'n bad  rot° med  Sim3 mm med':>26s} | MWU p (rot)")
    tot = {"A": [0, 0], "B": [0, 0]}
    for seq in sorted(p.name for p in A.iterdir() if p.is_dir()):
        if not (B / seq).is_dir():
            continue
        XA, XB = collect(A / seq, fl), collect(B / seq, fl)
        if not len(XA) or not len(XB):
            continue
        cells = []
        for key, X in (("A", XA), ("B", XB)):
            bad = int((X[:, 0] > a.bad_rot).sum())
            tot[key][0] += bad
            tot[key][1] += len(X)
            cells.append(f"{len(X):2d} {bad:2d}  {np.median(X[:, 0]):7.2f}  {np.median(X[:, 1]):10.1f}")
        p = mannwhitneyu(XA[:, 0], XB[:, 0], alternative="two-sided").pvalue
        print(f"{seq:22s} | {cells[0]:>26s} | {cells[1]:>26s} | {p:.3f}")
    (ba, na), (bb, nb) = tot["A"], tot["B"]
    _, p = fisher_exact([[ba, na - ba], [bb, nb - bb]])
    print(f"\npooled bad runs (rot ATE > {a.bad_rot}° or failure): A {ba}/{na} ({100 * ba / max(na, 1):.0f}%), "
          f"B {bb}/{nb} ({100 * bb / max(nb, 1):.0f}%), Fisher exact p = {p:.3f}")


if __name__ == "__main__":
    main()
