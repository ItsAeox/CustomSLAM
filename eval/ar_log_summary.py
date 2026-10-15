#!/usr/bin/env python3
"""Summarise an AR session log (web/ar.js: Log button -> Copy/Download, or results/browser/ar_*.log).

Reports the device/options line, tracking losses (how long, what the tracker saw before, gyro peak), relocalisations,
new maps / resets / merges, floor estimates (tilt to gravity), cactus placements and rejected taps, and per-30-frame
throughput (fps, track ms).

Usage: ar_log_summary.py <log> [<log> ...]
"""
import re
import statistics as st
import sys
from pathlib import Path


def summarise(path: Path) -> None:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    print(f"=== {path.name} ({len(lines)} lines)")
    for l in lines:
        if " start: " in l:
            print("  " + l.split(" start: ", 1)[1][:400])
    losses, recover, fps, track, floors, events = [], [], [], [], [], []
    for l in lines:
        m = re.search(r"state (\S.*?) -> (recently lost|lost) \| before: matches (\d+), track (\d+) ms \|.*?KFs (\d+).*?gyro peak (\S+)", l)
        if m:
            losses.append({"from": m.group(1), "matches": int(m.group(3)), "track": int(m.group(4)), "kfs": int(m.group(5)),
                           "gyro": None if m.group(6) == "n/a" else float(m.group(6))})
        m = re.search(r"-> (tracking[^|]*?) \|.*was lost ([\d.]+) s", l)
        if m:
            recover.append(float(m.group(2)))
        m = re.search(r"frame \d+: .*?\| ([\d.]+) fps \| track ms med ([\d.]+) p95 ([\d.]+)", l)
        if m:
            fps.append(float(m.group(1))); track.append((float(m.group(2)), float(m.group(3))))
        m = re.search(r"floor (found|updated): camera height ([\d.]+).*?(\d+) of (\d+) points.*?tilt to gravity ([\d.]+) deg", l)
        if m:
            floors.append((m.group(1), float(m.group(2)), int(m.group(3)), int(m.group(4)), float(m.group(5))))
        if not re.search(r" frame \d+: ", l) and re.search(
                r"Relocalized|Creation of new map|Reseting|Merge detected|anchors (stashed|restored|reset)|"
                r"keep map|(cactus|tree) (placed|removed)|tap rejected|map reset requested|ERROR", l):
            events.append(l[:200])
    cam = [re.search(r"camera (\d+) frames, (\d+)% dropped", l) for l in lines]
    cam = [m for m in cam if m]
    if cam:
        print(f"  camera: {cam[-1].group(1)} frames presented, {cam[-1].group(2)}% dropped (replaced before tracking)")
    for l in lines:
        if "frame clock fallback" in l:
            print("  " + l[13:])
    if fps:
        print(f"  throughput: {st.median(fps):.1f} fps median (min {min(fps):.1f}); track ms median-of-medians "
              f"{st.median(t[0] for t in track):.1f}, worst p95 {max(t[1] for t in track):.1f}")
    print(f"  losses: {len(losses)}; recovered {len(recover)} (lost for median "
          f"{st.median(recover) if recover else float('nan'):.1f} s, max {max(recover) if recover else float('nan'):.1f} s)")
    for x in losses:
        g = "n/a" if x["gyro"] is None else f"{x['gyro']:.0f} deg/s"
        print(f"    from {x['from']}: matches before {x['matches']}, track {x['track']} ms, KFs {x['kfs']}, gyro peak {g}")
    gy = [x["gyro"] for x in losses if x["gyro"] is not None]
    if gy:
        print(f"  gyro peak at loss: median {st.median(gy):.0f} deg/s, {sum(g > 90 for g in gy)}/{len(gy)} above 90 deg/s "
              f"(fast rotation)")
    for f in floors:
        print(f"  floor {f[0]}: camera height {f[1]:.3f}, {f[2]}/{f[3]} points, tilt {f[4]:.1f} deg")
    print("  events:")
    for e in events:
        print("    " + e)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarise(Path(p))
