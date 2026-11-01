#!/usr/bin/env python3
"""The user's AR test on the synthetic set (ISSUES G22, 31 Oct): an object placed on the floor spot, then - shown the way
the AR page shows it (web/ar.js: hidden while not tracking, stashed when a new map starts, restored when the map count
drops (merge), deleted when tracking restarts from scratch) - where it appears on screen for the rest of the run vs
where it should be.

Anchor: where the page puts it when the run logged the map's depth (orbslam3_run --tap-at, center_depth.txt): the tap ray
through the spot at the median depth of the tracked map points around the image centre, at the first tracked frame from
the tap with the spot in the middle of the view (the user taps once the map is there); otherwise the spot as the map
itself reconstructs it - triangulated (DLT) from the estimated camera poses of the placement
map and the spot's true pixel in every tracked frame of that map from 8 s before the tap (the first tracked frame at /
after it; synthetic ar / flat: touches.csv, longar: the middle of the 'place' phase) to 2 / 5 / 10 / 20 s after, the
shortest window whose rays span >= 3 deg (the page puts the object on the floor the map has triangulated there; a
map-scale fit made the score depend on the map's scale errors). Then, for every processed frame after the placement
where the spot is in the true view (in the image, >= 0.1 m in front): shown or not, screen error (px), and that error at
the spot's true distance (cm: how far beside the spot the object appears). The gyro hold (< 0.4 s) is not modelled.

trace.txt needs the map id / map count columns (orbslam3_run aa3763f+: the last two).
  anchor_return.py <run dir> <sequence dir> [--gt results/gt/synth_<seq>.txt] [--wrong-px 50]
Prints one line of key=value fields (anchor_summary.py reads them).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np



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
    ap.add_argument("--dump", help="per-frame file: t (motion clock, s), spot in view, object shown, px error, cm error")
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
    elif meta["scenario"] in ("flat", "flatmap"):  # back in the bedroom (by the other door in the loop routes) to the end
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
    K3 = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    def true_px(j):
        pc_ = quat_to_R(gq[j]).T @ (spot - gpos[j])
        if pc_[2] < 0.1: return None
        u_ = K3 @ (pc_ / pc_[2])
        return u_[:2] if 0 <= u_[0] < W and 0 <= u_[1] < H else None
    anchor, used = None, None
    # the page's placement (orbslam3_run --tap-at writes center_depth.txt: per tracked frame the median depth of the tracked
    # map points within 60 px of the image centre): the first tracked frame from the tap with the true spot within 40 px
    # of the centre and >= 5 such points - the user taps the spot once the map is there; the tap ray through the spot's
    # pixel at that depth
    cdp = run / "center_depth.txt"
    if cdp.exists():
        cd = {int(r_[0]): (float(r_[1]), int(r_[2])) for r_ in (l.split() for l in cdp.read_text().splitlines() if l and l[0] != "#")}
        for j in np.where(ok & (t >= t_tap - 1e-6))[0]:
            u_ = true_px(j)
            if u_ is None or np.hypot(u_[0] - cx, u_[1] - cy) > 40 or cd.get(int(frame[j]), (0, 0))[1] < 5:
                continue
            p = int(j)
            ray = np.linalg.inv(K3) @ np.array([u_[0], u_[1], 1.0])
            anchor = quat_to_R(quat[p]) @ (ray * cd[int(frame[p])][0]) + pos[p]
            out.update(placed=1, place_delay_s=round(t[p] - t_tap, 2), place_map=int(mapid[p]), anchor_method="tap",
                       tap_points=cd[int(frame[p])][1])
            break
        if anchor is None:
            out.update(placed=0, why="no map at the spot")
            print(" ".join(f"{k}={v}" for k, v in out.items())); return
    for T_after in ((2, 5, 10, 20) if anchor is None else ()):
        w = [j for j in np.where(ok & (mapid == mapid[p]) & (t >= t[p] - 8) & (t <= t[p] + T_after))[0] if true_px(j) is not None]
        if len(w) < 10: continue
        A = []
        for j in w:
            Rcw = quat_to_R(quat[j]).T
            P = K3 @ np.hstack([Rcw, (-Rcw @ pos[j])[:, None]])
            u_ = true_px(j)
            A += [u_[0] * P[2] - P[0], u_[1] * P[2] - P[1]]
        X = np.linalg.svd(np.array(A))[2][-1]
        if abs(X[3]) < 1e-12: continue
        X = X[:3] / X[3]
        d = X[None, :] - pos[w]; d /= np.linalg.norm(d, axis=1, keepdims=True)
        sub = d[:: max(1, len(d) // 60)]
        ang = np.degrees(np.arccos(np.clip(sub @ sub.T, -1, 1))).max()
        in_front = all((quat_to_R(quat[j]).T @ (X - pos[j]))[2] > 0 for j in w)
        if ang >= 3.0 and in_front:
            anchor, used = X, (T_after, len(w), ang); break
    if anchor is None:
        out.update(placed=0, why="collapsed")  # the map cannot triangulate the spot: its poses barely move (scale collapsed)
        print(" ".join(f"{k}={v}" for k, v in out.items())); return
    if used is not None:
        out.update(placed=1, place_delay_s=round(t[p] - t_tap, 2), place_map=int(mapid[p]), anchor_method="triangulated",
                   anchor_window_s=used[0], anchor_frames=used[1], anchor_rays_deg=round(float(used[2]), 1))
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
    ecm = 100 * epx / fx * pt[:, 2]  # beside the spot at its true distance
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
        # how long the user looks at the spot after coming back before the object shows again
        rv = np.where(v & (tt >= ret[0]) & (tt <= ret[1]))[0]
        if len(rv):
            rs = rv[sh[rv]]
            out.update(ret_wait_s=round(float(tt[rs[0]] - tt[rv[0]]), 1) if len(rs) else -1)
    if a.dump:
        with open(a.dump, "w") as fh:
            fh.write("# t_s in_view shown px cm\n")
            for j in range(len(idx)):
                fh.write(f"{t[idx[j]] - t0:.3f} {int(v[j])} {int(sh[j])} {epx[j]:.1f} {ecm[j]:.1f}\n")
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
