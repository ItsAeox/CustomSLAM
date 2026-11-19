#!/usr/bin/env python3
"""G32: where does the scale filter's upward bias on the rapid scenario come from? eval/vio_filter.py on each run's
(a) live WebVIO track and (b) the ground-truth camera track (fake trace: GT poses at the frame times, map units = metres
/ 2, always tracked, 20 keyframes), each with the phone-like IMU (-88 ms) and the ideal IMU (variants/imu_ideal.csv, 0 ms).
  bias_split.py <results dir> <arm> <scenario regex> [vio_filter.py options ...]
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

R, ARM, PAT, EXTRA = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]
D = Path("H:/Research/datasets/synth")
F = Path(__file__).resolve().parents[3] / "eval" / "vio_filter.py"


def gt_run(seq, out):
    """trace.txt of the ground-truth camera track at the frame stamps (reported stamps; pose at the true capture time)"""
    g = np.loadtxt(f"H:/Research/results/gt/synth_{seq}.txt", comments="#", ndmin=2)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "trace.txt", "w", newline="\n") as f:
        for k, r in enumerate(g):
            f.write(f"{k} 2 0 {r[1] / 2:.7f} {r[2] / 2:.7f} {r[3] / 2:.7f} {r[4]:.7f} {r[5]:.7f} {r[6]:.7f} {r[7]:.7f} 0 0 0 20\n")
    return out


for run in sorted(R.glob(f"*/{ARM}/seed1")):
    seq = run.parts[-3]
    if not re.match(PAT, seq):
        continue
    fake = gt_run(seq, Path(f"C:/Users/Aeox/AppData/Local/Temp/claude/H--Research/6624e451-5f38-4671-9a62-226588a85597/scratchpad/gtrun/{seq}"))
    for track, rd in (("live", run), ("gt", fake)):
        for name, imu, off in (("phone", D / seq / "mav0" / "imu0" / "data.csv", "-88"), ("ideal", D / seq / "variants" / "imu_ideal.csv", "0")):
            line = subprocess.run([sys.executable, str(F), str(rd), str(D / seq), "--imu", str(imu), "--offset-ms", off,
                                   "--gt", f"H:/Research/results/gt/synth_{seq}.txt", *EXTRA], capture_output=True, text=True).stdout.strip()
            print(f"{seq} {track:4s} {name:5s}: {line.split(': ', 1)[-1][:170]}", flush=True)
