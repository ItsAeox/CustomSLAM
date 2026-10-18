#!/usr/bin/env python3
"""One row per AR preset over all phone recordings (replay_all.sh): share of frames tracked per difficulty level
(labels.txt), losses per minute, maps per run, frames before the first map, tracking ms mean / p95 (WASM, phone
emulation). Uses eval/recording_metrics.run_metrics.

  summary_presets.py <replays dir> <labels.txt>
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "eval"))
from recording_metrics import run_metrics  # noqa: E402


def main():
    root, labels = Path(sys.argv[1]), Path(sys.argv[2])
    lv = {}
    for line in labels.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            f = line.split()
            lv[f[0]] = f[1].split("-")[0]  # revisit-A / -B count as "revisit"
    levels = ["easy", "medium", "hard", "turning", "revisit"]
    rows = {}
    for rec, level in lv.items():
        d = root / rec
        if not d.is_dir():
            continue
        for pre in sorted(p for p in d.iterdir() if p.is_dir() and p.name != "settings"):
            for s in sorted(x for x in pre.iterdir() if x.is_dir()):
                if not (s / "run_info.txt").exists():  # unfinished (killed) run
                    continue
                m = run_metrics(s)
                if not m["frames"]:
                    continue
                m["level"] = level
                info = dict(l.split(None, 1) for l in (s / "run_info.txt").read_text().splitlines()
                            if len(l.split(None, 1)) == 2) if (s / "run_info.txt").exists() else {}
                m["dur_min"] = float(info.get("seq_s", m["frames"] / 29.4)) / 60  # recording length, not processed frames
                m["dropped"] = float(info.get("dropped", 0)) / max(float(info.get("frames", 1)), 1)
                rows.setdefault(pre.name, []).append(m)
    print("tracked = share of frames with a pose (mean over runs); losses / min of recording; maps per run; first map = frames"
          " before the first map; ms = tracking time per frame (WASM); dropped = frames skipped because tracking was busy")
    print(f"{'preset':10s} {'runs':>4s} " + " ".join(f"{l:>8s}" for l in levels) + f" {'losses/min':>10s} {'maps':>5s} {'1st map':>7s} {'ms':>5s} {'p95':>5s} {'dropped':>7s}")
    for p, ms in sorted(rows.items(), key=lambda kv: -np.mean([m["tracked"] for m in kv[1]])):
        per = [np.mean([m["tracked"] for m in ms if m["level"] == l]) if any(m["level"] == l for m in ms) else np.nan for l in levels]
        lpm = np.mean([m["losses"] / max(m["dur_min"], 1e-6) for m in ms])
        print(f"{p:10s} {len(ms):4d} " + " ".join(f"{x:8.3f}" for x in per) +
              f" {lpm:10.1f} {np.mean([m['maps'] for m in ms]):5.2f} {np.mean([m['before_map'] for m in ms]):7.1f}"
              f" {np.mean([m['ms_mean'] for m in ms]):5.1f} {np.mean([m['ms_p95'] for m in ms]):5.1f} {100 * np.mean([m['dropped'] for m in ms]):6.0f}%")


if __name__ == "__main__":
    main()
