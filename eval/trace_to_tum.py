#!/usr/bin/env python3
"""Online (live) trajectory from an orbslam3_run --trace file, as TUM, for comparing with systems that only have
online poses (AlvaAR). traj_tum.txt is ORB-SLAM3's final trajectory (each frame relative to its reference keyframe
after bundle adjustment / loop closure); the trace has the pose the tracker reported at that frame, which is what an
AR app shows.

trace.txt columns: frame index, tracking state, tracked points, Twc (tx ty tz qx qy qz qw), ...; web builds end with
the current map id and the number of maps. Frame times come from the sequence (orbslam3_run's loadAsl: every cam0
row of data.csv that has an image file).

Kept: frames in the given tracking states (default 2 = OK). With map ids, only the map with the most such frames
(the counterpart of "largest map"; AlvaAR's driver keeps its longest segment).

Usage: trace_to_tum.py --trace trace.txt --seq <sequence dir> --out online_tum.txt [--states 2 3]
"""
import argparse
from collections import Counter
from pathlib import Path


def asl_times(seq: Path):
    cam = seq / "mav0" / "cam0"
    rows = []
    for l in (cam / "data.csv").read_text().splitlines():
        l = l.strip()
        if not l or l.startswith("#"):
            continue
        ns, name = l.split(",", 1)
        if (cam / "data" / name.strip()).exists():
            rows.append(int(ns))
    return rows  # integer ns; frame i of orbslam3_run = rows[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--seq", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--states", type=int, nargs="+", default=[2])
    a = ap.parse_args()

    ns = asl_times(a.seq)
    rows = []
    for l in a.trace.read_text().splitlines():
        f = l.split()
        if len(f) < 10:
            continue
        i, state = int(f[0]), int(f[1])
        if state not in a.states:
            continue
        mid = int(f[-2]) if len(f) > 12 else 0
        rows.append((i, mid, f[3:10]))
    maps = Counter(m for _, m, _ in rows)
    keep = maps.most_common(1)[0][0] if maps else None
    with a.out.open("w") as o:
        o.write("# t tx ty tz qx qy qz qw (online poses from %s, states %s, map %s)\n" % (a.trace.name, a.states, keep))
        for i, m, p in rows:
            if m == keep:
                o.write("%.9f %s\n" % (ns[i] * 1e-9, " ".join(p)))
    print(f"{a.out}: {sum(1 for _, m, _ in rows if m == keep)} poses (map {keep}; maps with tracked frames: {dict(maps)})")


if __name__ == "__main__":
    main()
