#!/usr/bin/env python3
"""Variants of a rendered synthetic sequence without re-rendering (synth/README.md): the frames are hard links, the
changed parts are the sensor streams or the settings. One flaw at a time:

  --imu ideal          the true IMU at the 100 Hz sensor times, exact stamps, no rounding / noise / bias
  --imu sensorstamps   the DeviceMotion values with the sensor's own stamps instead of the 60 Hz tick stamps
  --cam exact          frame stamps without the camera stamp jitter
  --focal 0.95         settings.yaml focal length x 0.95 (an uncalibrated camera; images unchanged)

  variant.py <seq dir> [--imu ideal|sensorstamps] [--cam exact] [--focal F] [--gt H:/Research/results/gt]
Creates <seq dir>__<tags> next to it and results/gt/synth_<new name>.txt; prints the new name.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq")
    ap.add_argument("--imu", choices=["ideal", "sensorstamps"])
    ap.add_argument("--cam", choices=["exact"])
    ap.add_argument("--focal", type=float)
    ap.add_argument("--gt", default="H:/Research/results/gt")
    a = ap.parse_args()
    src = Path(a.seq)
    tags = []
    if a.imu:
        tags.append(f"imu{a.imu}")
    if a.cam:
        tags.append(f"cam{a.cam}")
    if a.focal:
        tags.append(f"f{a.focal:g}")
    if not tags:
        raise SystemExit("nothing to change")
    dst = src.parent / f"{src.name}__{'_'.join(tags)}"
    if dst.exists():
        shutil.rmtree(dst)
    for p in src.rglob("*"):
        q = dst / p.relative_to(src)
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=True)
        elif p.suffix == ".jpg":
            q.parent.mkdir(parents=True, exist_ok=True)
            os.link(p, q)  # hard link: no copy of the frames
        else:
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)
    if a.imu == "ideal":
        shutil.copy2(src / "variants" / "imu_ideal.csv", dst / "mav0" / "imu0" / "data.csv")
    elif a.imu == "sensorstamps":
        shutil.copy2(src / "variants" / "imu_chrome_sensorstamps.csv", dst / "mav0" / "imu0" / "data.csv")
    gt_src = Path(a.gt) / f"synth_{src.name}.txt"
    gt = np.loadtxt(gt_src, comments="#")
    if a.cam == "exact":
        shutil.copy2(src / "variants" / "cam_exact.csv", dst / "mav0" / "cam0" / "data.csv")
        exact = [int(l.split(",")[0]) for l in (src / "variants" / "cam_exact.csv").read_text().splitlines() if l and l[0] != "#"]
        gt[:, 0] = np.array(exact[:len(gt)]) * 1e-9  # same poses (true capture times), keyed by the exact stamps
    if a.focal:
        st = (dst / "settings.yaml").read_text().splitlines()
        st = [(f"{l.split(':')[0]}: {float(l.split(':')[1]) * a.focal:.6f}" if l.startswith(("Camera1.fx:", "Camera1.fy:")) else l) for l in st]
        (dst / "settings.yaml").write_text("\n".join(st) + "\n")
    meta = json.loads((dst / "meta.json").read_text())
    meta["name"] = dst.name
    meta["synthetic"]["variant_of"] = src.name
    meta["synthetic"]["variant_changes"] = tags
    if a.focal:
        meta["intrinsics"]["fx"] *= a.focal; meta["intrinsics"]["fy"] *= a.focal; meta["calibrated"] = False
    (dst / "meta.json").write_text(json.dumps(meta, indent=1))
    with open(Path(a.gt) / f"synth_{dst.name}.txt", "w", newline="\n") as fh:
        fh.write(f"# synthetic ground truth ({src.name}, variant {'_'.join(tags)}): t = frame stamp in data.csv, pose at the true capture time (Twc)\n")
        for r in gt:
            fh.write(" ".join(f"{x:.9f}" if i == 0 else f"{x:.7f}" for i, x in enumerate(r)) + "\n")
    print(dst.name)


if __name__ == "__main__":
    main()
