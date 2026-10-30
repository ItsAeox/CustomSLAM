#!/usr/bin/env python3
"""eval/vio_filter.py (loosely coupled filter, approach B) on every visual WebVIO run of a synthetic results folder, with
the phone-like IMU (mav0/imu0, -88 ms) and the ideal IMU (variants/imu_ideal.csv, 0 ms) on the same live trajectories.
One process (live replays may be running).   vio_filter_runs.py <results dir> <tag> [vio_filter.py options ...]
Writes <results dir>/vio_filter_<tag>.txt: one line per run and IMU.
"""
import subprocess
import sys
from pathlib import Path

R, TAG, EXTRA = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
D, GT = Path("H:/Research/datasets/synth"), Path("H:/Research/results/gt")
F = Path(__file__).resolve().parents[3] / "eval" / "vio_filter.py"
out = R / f"vio_filter_{TAG}.txt"
with open(out, "w", encoding="utf-8") as fh:
    fh.write(f"# vio_filter.py {' '.join(EXTRA)}\n")
    for run in sorted(R.glob("*/beta_scale*/seed*")):
        if not (run / "trace.txt").exists():
            continue
        seq = run.parts[-3]
        base = seq.split("__")[0]
        for name, imu, off in (("phone", D / base / "mav0" / "imu0" / "data.csv", "-88"),
                               ("ideal", D / base / "variants" / "imu_ideal.csv", "0")):
            if seq.endswith("__imuideal") and name == "phone":
                continue
            r = subprocess.run([sys.executable, str(F), str(run), str(D / seq), "--imu", str(imu), "--offset-ms", off,
                                "--gt", str(GT / f"synth_{seq}.txt")] + EXTRA, capture_output=True, text=True)
            line = (r.stdout.strip() or r.stderr.strip().splitlines()[-1]).replace(str(run), "")
            fh.write(f"{seq} {run.parts[-2]} {run.parts[-1]} {name}{line}\n")
            fh.flush()
print(out)
