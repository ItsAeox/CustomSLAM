#!/usr/bin/env python3
"""TUM-VI frames for pinhole-only baselines (AlvaAR): CLAHE + fisheye rectification to a pinhole camera.

AlvaAR has only pinhole + radtan and CLAHE compiled off, so on TUM-VI it gets frames prepared here, the same way
WebVIO's input is prepared (orbslam3_run --clahe: cv::imread grayscale, CLAHE 3.0 / 8x8 tiles) and then undistorted
with the Kannala-Brandt calibration from the settings file to a pinhole camera with the same focal length and the
principal point at the image centre (~107 degrees across a 512 px image). This is an accommodation for the baseline
only; WebVIO uses the raw fisheye frames.

Writes <out>/<sequence>/mav0/cam0/data/*.png + data.csv (ASL layout, same timestamps) and <out>/TUM-VI_rect.yaml.

Usage: make_rectified_tumvi.py --settings TUM-VI_mono.yaml --out DIR SEQ_DIR [SEQ_DIR ...]
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np


def read_settings(p: Path):
    txt = p.read_text()
    g = lambda k: float(re.search(rf"^{re.escape(k)}:\s*([-\d.eE+]+)", txt, re.M).group(1))
    K = np.array([[g("Camera1.fx"), 0, g("Camera1.cx")], [0, g("Camera1.fy"), g("Camera1.cy")], [0, 0, 1]])
    D = np.array([g("Camera1.k1"), g("Camera1.k2"), g("Camera1.k3"), g("Camera1.k4")])
    return K, D, int(g("Camera.width")), int(g("Camera.height"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("seqs", nargs="+", type=Path)
    a = ap.parse_args()

    K, D, w, h = read_settings(a.settings)
    f = K[0, 0]
    P = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), P, (w, h), cv2.CV_16SC2)
    clahe = cv2.createCLAHE(3.0, (8, 8))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "TUM-VI_rect.yaml").write_text(
        "%YAML:1.0\n# TUM-VI cam0 after scripts/make_rectified_tumvi.py (CLAHE + fisheye -> pinhole, no distortion)\n"
        'File.version: "1.0"\nCamera.type: "PinHole"\n'
        f"Camera1.fx: {f}\nCamera1.fy: {f}\nCamera1.cx: {w / 2}\nCamera1.cy: {h / 2}\n"
        "Camera1.k1: 0.0\nCamera1.k2: 0.0\nCamera1.p1: 0.0\nCamera1.p2: 0.0\n"
        f"Camera.width: {w}\nCamera.height: {h}\nCamera.fps: 20\nCamera.RGB: 1\n")
    for seq in a.seqs:
        src = seq / "mav0" / "cam0"
        dst = a.out / seq.name / "mav0" / "cam0"
        (dst / "data").mkdir(parents=True, exist_ok=True)
        rows = [l for l in (src / "data.csv").read_text().splitlines() if l.strip()]
        (dst / "data.csv").write_text("\n".join(rows) + "\n")
        n = 0
        for l in rows:
            if l.startswith("#"):
                continue
            name = l.split(",", 1)[1].strip()
            im = cv2.imread(str(src / "data" / name), cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            im = cv2.remap(clahe.apply(im), m1, m2, cv2.INTER_LINEAR)
            cv2.imwrite(str(dst / "data" / name), im)
            n += 1
        print(f"{seq.name}: {n} frames -> {dst}")


if __name__ == "__main__":
    main()
