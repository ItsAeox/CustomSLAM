"""Split the AR object's screen error into rotation and the rest (translation / placement depth).
For each run: placement frame p as in eval/anchor_return.py (tap file + center_depth.txt rule), then over the frames
after p with the spot in view: rotation error of the estimated relative rotation p->j vs the true one (deg), the
px error as scored, and the px error with the estimated rotation replaced by the true relative rotation (what is left
is translation + placement depth). Also the placement depth ratio vs the map's local scale over p..p+3 s."""
import sys, json, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot

def one(run, seq):
    run, seq = Path(run), Path(seq)
    rows = [r.split() for r in (run / "trace.txt").read_text().splitlines()]
    rows = [r for r in rows if len(r) >= 34]
    tr = np.array([[float(x) for x in r[:10]] + [float(r[-2])] for r in rows])
    frame, state = tr[:, 0].astype(int), tr[:, 1].astype(int)
    pos, quat, mapid = tr[:, 3:6], tr[:, 6:10], tr[:, 10].astype(int)
    stamps = np.loadtxt(seq / "mav0/cam0/data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64)
    t = stamps[np.clip(frame, 0, len(stamps) - 1)] * 1e-9
    gt = np.loadtxt(f"H:/Research/results/gt/synth_{seq.name}.txt", comments="#", ndmin=2)
    gi = np.clip(np.searchsorted(gt[:, 0], t - 1e-6), 0, len(gt) - 1)
    gpos, gq = gt[gi, 1:4], gt[gi, 4:8]
    meta = json.loads((seq / "meta.json").read_text())["synthetic"]
    spot = np.array(meta["motion"]["spot"], float)
    st = (seq / "settings.yaml").read_text()
    yv = lambda k: float([l for l in st.splitlines() if l.startswith(k + ":")][0].split(":")[1])
    fx, fy, cx, cy = yv("Camera1.fx"), yv("Camera1.fy"), yv("Camera1.cx"), yv("Camera1.cy")
    W, H = int(yv("Camera.width")), int(yv("Camera.height"))
    taps = [l for l in (seq / "touches.csv").read_text().splitlines() if l and l[0] != "#"] if (seq / "touches.csv").exists() else []
    t0m = float(open(seq / "gt/camera_dense.csv").readlines()[1].split()[0])
    t_tap = int(taps[0].split(",")[0]) * 1e-9 if taps else t0m + 0.5 * sum(meta["motion"]["phases_s"]["place"])
    ok = np.isin(state, [2, 5])
    Re, Rt = Rot.from_quat(quat), Rot.from_quat(gq)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    def true_px(j):
        pc = Rt[j].inv().apply(spot - gpos[j])
        if pc[2] < 0.1: return None
        u = K @ (pc / pc[2])
        return u[:2] if 0 <= u[0] < W and 0 <= u[1] < H else None
    cd = {int(r[0]): (float(r[1]), int(r[2])) for r in (l.split() for l in (run / "center_depth.txt").read_text().splitlines() if l and l[0] != "#")}
    p = None
    for j in np.where(ok & (t >= t_tap - 1e-6))[0]:
        u = true_px(j)
        if u is None or np.hypot(u[0] - cx, u[1] - cy) > 40 or cd.get(int(frame[j]), (0, 0))[1] < 5: continue
        p = int(j); break
    if p is None: return None
    u = true_px(p)
    ray = np.linalg.inv(K) @ np.array([u[0], u[1], 1.0])
    dmap = cd[int(frame[p])][0]
    anchor = Re[p].apply(ray * dmap) + pos[p]
    true_depth = Rt[p].inv().apply(spot - gpos[p])[2]
    # local scale: true / est path length over p..p+3 s in the same map
    w = np.where(ok & (mapid == mapid[p]) & (t >= t[p]) & (t <= t[p] + 3))[0]
    le = np.linalg.norm(np.diff(pos[w], axis=0), axis=1).sum(); lt = np.linalg.norm(np.diff(gpos[w], axis=0), axis=1).sum()
    s = lt / le if le > 1e-9 else np.nan
    depth_ratio = dmap * s / true_depth
    # Sim3 (Umeyama) scale of the placement map over p-2 .. p+5 s: robust to jitter, unlike the path length
    w2 = np.where(ok & (mapid == mapid[p]) & (t >= t[p] - 2) & (t <= t[p] + 5))[0]
    A, B = pos[w2], gpos[w2]
    if len(w2) >= 10 and np.linalg.norm(B - B.mean(0), axis=1).max() > 0.05:
        Ac, Bc = A - A.mean(0), B - B.mean(0)
        U, D, Vt = np.linalg.svd(Bc.T @ Ac / len(w2))
        S = np.eye(3); S[2, 2] = np.sign(np.linalg.det(U @ Vt))
        s3 = np.trace(np.diag(D) @ S) / (Ac ** 2).sum(1).mean()
        depth_ratio3 = dmap * s3 / true_depth
    else:
        depth_ratio3 = np.nan
    rot, epx, epx_rt, dctr, etop = [], [], [], [], []
    # a 30 cm object standing on the spot: its top in the map from the placement frame's own pose (the page's "up" is the
    # phone's gravity turned into the map by the current camera rotation; height in map units from the placement depth)
    up_map = (Re[p] * Rt[p].inv()).apply([0, 0, 1.0])
    top_map = anchor + up_map * 0.3 * dmap / true_depth
    top_true = spot + np.array([0, 0, 0.3])
    for j in np.where(ok & (mapid == mapid[p]))[0]:
        if j <= p: continue
        ut = true_px(j)
        if ut is None: continue
        dE = Re[p].inv() * Re[j]; dT = Rt[p].inv() * Rt[j]
        rot.append(np.degrees((dT.inv() * dE).magnitude()))
        dctr.append(np.hypot(ut[0] - cx, ut[1] - cy))
        pt_, pe_ = Rt[j].inv().apply(top_true - gpos[j]), Re[j].inv().apply(top_map - pos[j])
        etop.append(np.hypot(*((K @ (pe_ / pe_[2]))[:2] - (K @ (pt_ / pt_[2]))[:2])) if pt_[2] > 0.1 and pe_[2] > 1e-6 else np.inf)
        for R_j, lst in ((Re[j], epx), (Re[p] * dT, epx_rt)):
            pc = R_j.inv().apply(anchor - pos[j])
            if pc[2] <= 1e-6: lst.append(np.inf); continue
            ue = K @ (pc / pc[2]); lst.append(np.hypot(*(ue[:2] - ut)))
    if not rot: return None
    epx_a, dctr = np.array(epx), np.array(dctr)
    bins = {b: (np.median(epx_a[m]) if m.sum() >= 5 else np.nan, int(m.sum())) for b, m in
            (("c", dctr < 60), ("m", (dctr >= 60) & (dctr < 150)), ("o", dctr >= 150))}
    return dict(top=np.median(etop), bins=bins, depth_ratio3=depth_ratio3, delay=t[p] - t_tap, pts=cd[int(frame[p])][1], depth_ratio=depth_ratio, rot_med=np.median(rot),
                rot_p90=np.percentile(rot, 90), px=np.median(epx), px_truerot=np.median(epx_rt))

if __name__ == "__main__":
    root = Path(sys.argv[1]); scen = sys.argv[2:] or ["ar", "longar"]
    import collections
    agg = collections.defaultdict(list)
    for sq in sorted(root.iterdir()):
        if not sq.is_dir() or not any(sq.name.startswith(s) and sq.name[len(s):].isdigit() for s in scen): continue
        for arm in sorted(sq.iterdir()):
            for sd in sorted(arm.glob("seed*")):
                try: r = one(sd, Path("H:/Research/datasets/synth") / sq.name)
                except Exception as e: print(sq.name, arm.name, sd.name, "error", e); continue
                if r is None: print(sq.name, arm.name, sd.name, "not placed"); continue
                print(f"{sq.name:8s} {arm.name:20s} {sd.name}: delay {r['delay']:5.2f}s pts {r['pts']:3d} depth x{r['depth_ratio']:5.2f} sim3 x{r['depth_ratio3']:5.2f} "
                      f"rot med {r['rot_med']:5.2f} p90 {r['rot_p90']:5.2f} deg  px {r['px']:6.1f}  px with true rotation {r['px_truerot']:6.1f}  top {r['top']:6.1f}  by distance from centre <60 / 60-150 / >150 px: " + " / ".join(f"{v[0]:.0f} ({v[1]})" for v in r['bins'].values()))
                agg[(sq.name.rstrip('0123456789'), arm.name)].append(r)
    print()
    print("| Scenario | Arm | Runs | Rotation err median deg | px median | px with true rotation | px, top of a 30 cm object | px, spot < 60 px from centre | 60-150 px | > 150 px | Placement depth / true (Sim3, median; |log| median) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (sc, arm), rs in sorted(agg.items()):
        dr = np.array([r['depth_ratio'] for r in rs])
        print(f"| {sc} | {arm} | {len(rs)} | {np.median([r['rot_med'] for r in rs]):.2f} | {np.median([r['px'] for r in rs]):.1f} | "
              f"{np.median([r['px_truerot'] for r in rs]):.1f} | {np.median([r['top'] for r in rs]):.1f} | " + " | ".join(f"{np.nanmedian([r['bins'][b][0] for r in rs]):.1f}" for b in "cmo") +
              f" | {np.nanmedian([r['depth_ratio3'] for r in rs]):.2f} ({np.nanmedian(np.abs(np.log([r['depth_ratio3'] for r in rs]))):.2f}) |")
