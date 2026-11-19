#!/usr/bin/env python3
"""Band-limit check of the synthetic motions (synth/README.md "Validation"): regenerates each sequence's 1 kHz motion
(same seeds and collision retries as make_dataset.py) and prints the content above 40 Hz of the true acceleration and
rotation rate. The phone samples at 100 Hz, so motion content near or above 50 Hz is unrealistic (and the ideal-IMU
check (a) of validate.py stops giving s = 1). Only the slide table vibration (15-45 Hz, deliberate) shows up.
band_check.py [name,name,...]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from synth import imu as imu_mod  # noqa: E402
from synth.make_dataset import specs  # noqa: E402
from synth.scene import make_room  # noqa: E402
from synth.traj import RATE, SCENARIOS, check_motion  # noqa: E402


def main():
    only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None
    sos = butter(4, 40, "highpass", fs=RATE, output="sos")
    for sp in specs():
        if only and sp["name"] not in only:
            continue
        room = make_room(sp["room"], sp["room_seed"])
        for attempt in range(30):
            m = SCENARIOS[sp["scenario"]](room, np.random.default_rng(sp["seed"] + 100000 * attempt), sp["variant"])
            if not check_motion(room, m, table_ok=sp["scenario"] == "slide"):
                break
        tr = imu_mod.body_truth(m)
        inner = slice(int(RATE), -int(RATE))  # filter edges (1 s)
        a = np.linalg.norm(sosfiltfilt(sos, imu_mod.d2(tr["p"], 1.0 / RATE), axis=0), axis=1)[inner]
        w = np.linalg.norm(sosfiltfilt(sos, tr["w"], axis=0), axis=1)[inner]
        print(f"{sp['name']:9s} accel > 40 Hz: max {a.max():.3f} m/s2, p99.9 {np.percentile(a, 99.9):.4f} | "
              f"rotation rate > 40 Hz: max {np.degrees(w.max()):.3f} deg/s", flush=True)


if __name__ == "__main__":
    main()
