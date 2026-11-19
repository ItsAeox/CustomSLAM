#!/usr/bin/env python3
"""Diagnostic for gyrobias_1103 (not a main-table input): a copy of an EuRoC sequence whose gyro rates have the
dataset's estimated gyro bias (state_groundtruth_estimate0, b_w, nearest sample) subtracted, i.e. what a calibrated
gyroscope would deliver. A phone browser gets the Android calibrated gyroscope (drift-compensated); EuRoC's ADIS16448
rates are raw, with a static-start mean of 4.6-5.0 deg/s. Accelerometer columns and the images are unchanged (cam0 is
a symlink to the original).

Usage: gyro_debias.py --seq <EuRoC sequence dir> --out <new sequence dir>
"""
import argparse
import bisect
import csv
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    gt = [r for r in csv.reader((a.seq / "mav0/state_groundtruth_estimate0/data.csv").open()) if r and not r[0].startswith("#")]
    gt_t = [int(r[0]) for r in gt]
    bw = [[float(x) for x in r[11:14]] for r in gt]  # b_w_RS_S x y z [rad/s]

    (a.out / "mav0/imu0").mkdir(parents=True, exist_ok=True)
    cam = a.out / "mav0/cam0"
    if not cam.exists():
        os.symlink((a.seq / "mav0/cam0").resolve(), cam)
    n = 0
    with (a.seq / "mav0/imu0/data.csv").open() as fi, (a.out / "mav0/imu0/data.csv").open("w") as fo:
        for l in fi:
            if l.startswith("#") or not l.strip():
                fo.write(l)
                continue
            f = l.strip().split(",")
            t = int(f[0])
            i = min(max(bisect.bisect_left(gt_t, t), 0), len(gt_t) - 1)
            if i > 0 and abs(gt_t[i - 1] - t) < abs(gt_t[i] - t):
                i -= 1
            w = [float(f[k + 1]) - bw[i][k] for k in range(3)]
            fo.write(",".join([f[0]] + [f"{x:.9f}" for x in w] + f[4:7]) + "\n")
            n += 1
    print(f"{a.out}: {n} IMU rows, gyro bias at start {[round(x, 4) for x in bw[0]]} rad/s removed")


if __name__ == "__main__":
    main()
