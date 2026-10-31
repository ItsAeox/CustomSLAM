#!/usr/bin/env python3
"""The user's AR test on the synthetic set (ISSUES G22, 31 Oct): an object placed on the floor spot, then - shown the way
the AR page shows it (web/ar.js: hidden while not tracking, stashed when a new map starts, restored when the map count
drops (merge), deleted when tracking restarts from scratch) - where it appears on screen for the rest of the run vs
where it should be.

Anchor: at the placement frame (the first tracked frame at / after the tap; synthetic ar: touches.csv, longar: the middle
of the 'place' phase) the true spot in that frame's true camera coordinates, scaled to map units with the map's local
Sim3 scale (tracked frames of that map from 8 s before to 1 s after), put into the map with the estimated pose: on screen
exactly at the spot when placed, like a tap. Then, for every processed frame after it where the spot is in the true view
(in the image, >= 0.1 m in front): shown or not, screen error (px), and the error at the spot (cm: where the object
appears in the room from that viewpoint, via the placement scale). The gyro hold (< 0.4 s) is not modelled.

trace.txt needs the map id / map count columns (orbslam3_run aa3763f+: the last two).
  anchor_return.py <run dir> <sequence dir> [--gt results/gt/synth_<seq>.txt] [--wrong-px 50]
Prints one line of key=value fields (anchor_summary.py reads them).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scale_from_accel import umeyama_scale


def quat_to_R(q):  # x y z w
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("seq")
    ap.add_argument("--gt", help="per-frame camera GT (t = reported frame stamp; default results/gt/synth_<seq>.txt)")
    ap.add_argument("--wrong-px", type=float, default=50.0, help="shown more than this far off = a wrong placement")
    a = ap.parse_args()
    run, seq = Path(a.run), Path(a.seq)
    name = seq.name
    gtp = Path(a.gt) if a.gt else Path("H:/Research/results/gt") / f"synth_{name}.txt"
    rows = [l.split() for l in (run / "trace.txt").read_text().splitlines()]
    rows = [r for r in rows if len(r) >= 34]
    if not rows:
        print(f"{name} run={run} error=no trace rows with map id columns"); return
    tr = np.array([[float(x) for x in r[:10]] + [float(r[-2]), float(r[-1])] for r in rows])
    frame, state = tr[:, 0].astype(int), tr[:, 1].astype(int)
    pos, quat, mapid, nmaps = tr[:, 3:6], tr[:, 6:10], tr[:, 10].astype(int), tr[:, 11].astype(int)
    stamps = np.loadtxt(seq / "mav0/cam0/data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64)
    t = stamps[np.clip(frame, 0, len(stamps) - 1)] * 1e-9
    gt = np.loadtxt(gtp, comments="#", ndmin=2)
    gidx = np.clip(np.searchsorted(gt[:, 0], t - 1e-6), 0, len(gt) - 1)
    gpos, gq = gt[gidx, 1:4], gt[gidx, 4:8]
    t0 = float(open(seq / "gt/camera_dense.csv").readlines()[1].split()[0])  # the motion's clock origin (phases_s)
    meta = json.loads((seq / "meta.json").read_text())["synthetic"]
    motion = meta["motion"]
    spot = np.array(motion["spot"], float)
    ph = motion.get("phases_s", {})
    st =(seq / "settings.yaml").read_text()
    def yv(key):
        for l in st.splitlines():
            if l.startswith(key + ":"):
                return float(l.split(":")[1])
    fx, fy, cx, cy = yv("Camera1.fx"), yv("Camera1.fy"), yv("Camera1.cx"), yv("Camera1.cy")
    W, H = int(yv("Camera.width")), int(yv("Camera.height"))
    taps = [l for l in (seq / "touches.csv").read_text().splitlines() if l and l[0] != "#"] if (seq / "touches.csv").exists() else []
    if taps:
        t_tap = int(taps[0].split(",")[0]) * 1e-9
    elif "place" in ph:
        t_tap = t0 + 0.5 * (ph["place"][0] + ph["place"][1])
    else:
        print(f"{name} run={run} error=no tap / place phase"); return
    if meta["scenario"] == "ar":
        ret = (t0 + ph["return"][0], t0 + ph["return"][1])
    elif meta["scenario"] == "longar":
        ret = (t0 + ph["in"][0], t0 + ph["orbit2"][1])
    elif meta["scenario"] == "flat":  # back in the bedroom (by the other door in the loop routes) to the end
        ret = (t0 + ph["return"][0], t0 + ph["return"][1])
    else:
        ret = None
    ok = np.isin(state, [2, 5])
    # placement: the first tracked frame at / after the tap
    cand = np.where(ok & (t >= t_tap - 1e-6))[0]
    out = {"seq": name, "run": str(run)}
    if not len(cand):
        out.update(placed=0)
        print(" ".join(f"{k}={v}" for k, v in out.items())); return
    p = cand[0]
    w = np.where(ok & (mapid == mapid[p]) & (t >= t[p] - 8) & (t <= t[p] + 1))[0]
    if len(w) < 20 or np.ptp(gpos[w], 0).max() < 0.05:
        out.update(placed=0, why="no local scale")
        print(" ".join(f"{k}={v}" for k, v in out.items())); return
    s = umeyama_scale(pos[w], gpos[w])  # metres per map unit
    Rcw_true = quat_to_R(gq[p]).T
    pc = Rcw_true @ (spot - gpos[p]) / s
    anchor = quat_to_R(quat[p]) @ pc + pos[p]
    out.update(placed=1, place_delay_s=round(t[p] - t_tap, 2), place_map=int(mapid[p]), scale_mpu=round(s, 4))
    # the page's anchor state after placement (ar.js: lastMaps / lastState logic)
    exists, stashed = True, False
    shown = np.zeros(len(t), bool)
    events = []
    for i in range(p, len(t)):
        if i > p:
            if nmaps[i] > nmaps[i - 1]:
                if exists and not stashed: stashed = True; events.append(f"stash@{t[i] - t0:.1f}")
            elif nmaps[i] < nmaps[i - 1]:
                if stashed: stashed = False; events.append(f"restore@{t[i] - t0:.1f}")
            elif state[i - 1] >= 2 and state[i] == 1 and not stashed and exists:
                exists = False; events.append(f"reset@{t[i] - t0:.1f}")
        shown[i] = exists and not stashed and ok[i]
    # true view of the spot, screen and room errors
    idx = np.arange(p + 1, len(t))
    Rcw_t = np.array([quat_to_R(q).T for q in gq[idx]])
    pt = np.einsum("nij,nj->ni", Rcw_t, spot - gpos[idx])
    inview = (pt[:, 2] > 0.1)
    ut = np.stack([fx * pt[:, 0] / np.maximum(pt[:, 2], 1e-9) + cx, fy * pt[:, 1] / np.maximum(pt[:, 2], 1e-9) + cy], 1)
    inview &= (ut[:, 0] >= 0) & (ut[:, 0] < W) & (ut[:, 1] >= 0) & (ut[:, 1] < H)
    Rcw_e = np.array([quat_to_R(q).T for q in quat[idx]])
    pe = np.einsum("nij,nj->ni", Rcw_e, anchor - pos[idx])
    ue = np.stack([fx * pe[:, 0] / np.where(pe[:, 2] > 1e-9, pe[:, 2], 1e-9) + cx,
                   fy * pe[:, 1] / np.where(pe[:, 2] > 1e-9, pe[:, 2], 1e-9) + cy], 1)
    epx = np.where(pe[:, 2] > 1e-6, np.linalg.norm(ue - ut, axis=1), np.inf)
    seen = gpos[idx] + np.einsum("nji,nj->ni", Rcw_t, pe * s)  # where the object appears in the room (Rwc = Rcw^T)
    ecm = 100 * np.linalg.norm(seen - spot, axis=1)
    sh = shown[idx]
    v, vs = inview, inview & sh
    def stats(m, pre=""):
        d = {f"{pre}view_frames": int(m.sum()), f"{pre}shown_pct": round(100 * (m & sh).sum() / max(m.sum(), 1), 1)}
        e, c = epx[m & sh], ecm[m & sh]
        if len(e):
            d.update({f"{pre}px_med": round(float(np.median(e)), 1), f"{pre}px_p90": round(float(np.percentile(e, 90)), 1),
                      f"{pre}cm_med": round(float(np.median(c)), 1), f"{pre}cm_p90": round(float(np.percentile(c, 90)), 1),
                      f"{pre}wrong_pct": round(100 * float(np.mean(e > a.wrong_px)), 1)})
        return d
    out.update(stats(v))
    if ret:
        tt = t[idx]
        out.update(stats(v & (tt >= ret[0]) & (tt <= ret[1]), "ret_"))
    # gaps: stretches of in-view frames where the object is not shown
    gaps, cur = [], None
    for j in range(len(idx)):
        if v[j] and not sh[j]:
            if cur is None: cur = t[idx[j]]
        elif v[j] and sh[j] and cur is not None:
            gaps.append(t[idx[j]] - cur); cur = None
    out.update(gaps=len(gaps) + (cur is not None), gap_max_s=round(max(gaps + ([t[-1] - cur] if cur is not None else []), default=0), 1),
               unrecovered_at_end=int(cur is not None), end_shown=int(bool(shown[-1])), anchor_deleted=int(not exists),
               maps_max=int(nmaps.max()), events=",".join(events) or "-")
    print(" ".join(f"{k}={v}" for k, v in out.items()))


if __name__ == "__main__":
    main()
