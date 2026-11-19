#!/usr/bin/env python3
"""One table of the synthetic dataset (synth/README.md): per sequence the room, duration, frames, motion statistics
from the ground truth (path length, peak / 95th percentile speed, acceleration, rotation rate), blur samples and the
validation ladder (synth/validate.py). Writes <dataset>/SUMMARY.md and prints it.   summary.py [dataset dir]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def stats(seq):
    d = np.loadtxt(seq / "gt" / "camera_dense.csv", comments="#")
    t, p = d[:, 0] - d[0, 0], d[:, 1:4]
    dt = np.diff(t).mean()
    v = np.gradient(p, dt, axis=0)
    a = np.gradient(v, dt, axis=0)
    R = Rotation.from_quat(d[:, 4:8])
    w = (R[:-1].inv() * R[1:]).magnitude() / np.diff(t)
    sp, ac = np.linalg.norm(v, axis=1), np.linalg.norm(a, axis=1)
    return {"dur": t[-1], "path": float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()),
            "v95": np.percentile(sp, 95), "vmax": sp.max(), "a95": np.percentile(ac, 95), "amax": ac.max(),
            "w95": np.degrees(np.percentile(w, 95)), "wmax": np.degrees(w.max())}


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "H:/Research/datasets/synth")
    rows = []
    for seq in sorted(p for p in root.iterdir() if (p / "meta.json").exists() and "__" not in p.name):
        m = json.loads((seq / "meta.json").read_text())
        st = stats(seq)
        val = json.loads((seq / "validation.json").read_text()) if (seq / "validation.json").exists() else {}
        g = lambda k, f, fmt="{:.3f}": "-" if val.get(k, {}).get(f) is None else fmt.format(val[k][f])
        b = val.get("c_board", {})
        rows.append(f"| {seq.name} | {m['synthetic']['room']} | {st['dur']:.0f} | {m['frames']} | {st['path']:.1f} | "
                    f"{st['v95']:.2f} / {st['vmax']:.2f} | {st['a95']:.1f} / {st['amax']:.1f} | {st['w95']:.0f} / {st['wmax']:.0f} | "
                    f"{m['synthetic']['blur_samples_mean']:.1f} | {g('a_align', 'ratio')} / {g('a_window', 'ratio')} | "
                    f"{g('b_offset', 'ms', '{}')} | {g('b_align', 'ratio')} / {g('b_window', 'ratio')} | "
                    f"{'{:.2f} / {:.2f}'.format(b['mean_px'], b['max_px']) if b.get('frames') else '-'} |")
    head = ("| Sequence | Room | s | Frames | Path m | Speed p95 / max m/s | Accel p95 / max m/s² | Rotation p95 / max °/s | "
            "Blur samples | (a) scale, ideal IMU: alignment / window | (b) offset ms (−88) | (b) scale, phone IMU: alignment / window | "
            "(c) board px mean / max |\n|" + "---|" * 13)
    text = "# Synthetic dataset summary (synth/summary.py)\n\n" + head + "\n" + "\n".join(rows) + "\n"
    (root / "SUMMARY.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
