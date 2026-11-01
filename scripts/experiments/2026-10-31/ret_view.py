"""On return: where the camera is when the spot first comes into view vs when the object shows again, relative to the
kept map's keyframes (true poses at the keyframe times): distance to the nearest keyframe, viewing-direction angle to
the most similar keyframe within 1 m, distance to the spot."""
import sys, json, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
for run in sys.argv[1:]:
    run = Path(run); q = run.parts[-3]; seq = Path("H:/Research/datasets/synth") / q
    kv = dict(x.split("=", 1) for x in open(run / "anchor.txt").read().split() if "=" in x)
    gt = np.loadtxt(f"H:/Research/results/gt/synth_{q}.txt", comments="#")
    kf = np.loadtxt(run / "kf_traj_tum.txt", comments="#", ndmin=2)
    gk = gt[np.clip(np.searchsorted(gt[:, 0], kf[:, 0] - 1e-6), 0, len(gt) - 1)]
    kpos, kdir = gk[:, 1:4], Rot.from_quat(gk[:, 4:8]).apply([0, 0, 1])
    meta = json.loads((seq / "meta.json").read_text())["synthetic"]["motion"]; spot = np.array(meta["spot"])
    t0 = float(open(seq / "gt/camera_dense.csv").readlines()[1].split()[0])
    # the trace: first frame with the spot in view in the return phase, then the first tracked frame after it
    rows = [l.split() for l in open(run / "trace.txt") if l[0] != "#" and len(l.split()) >= 34]
    fr = np.array([int(r[0]) for r in rows]); stt = np.array([int(r[1]) for r in rows])
    st = np.loadtxt(seq / "mav0/cam0/data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64)
    t = st[fr] * 1e-9; g = gt[np.clip(np.searchsorted(gt[:, 0], t - 1e-6), 0, len(gt) - 1)]
    R = Rot.from_quat(g[:, 4:8]); pc = R.inv().apply(spot - g[:, 1:4])
    K = 465.6; W, H = 480, 640
    u = K * pc[:, 0] / np.maximum(pc[:, 2], 1e-9) + 240; v = K * pc[:, 1] / np.maximum(pc[:, 2], 1e-9) + 320
    inview = (pc[:, 2] > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    ret = (t - t0 >= meta["phases_s"]["return"][0])
    iv = np.where(inview & ret)[0]
    if not len(iv): continue
    i0 = iv[0]
    ok = np.isin(stt, [2, 5]) & ret & (t >= t[i0])
    i1 = np.where(ok)[0][0] if ok.any() else None
    def desc(i):
        p, d = g[i, 1:4], R[i].apply([0, 0, 1])
        dist = np.linalg.norm(kpos - p, axis=1)
        near = dist < 1.0
        ang = np.degrees(np.arccos(np.clip(kdir[near] @ d, -1, 1))).min() if near.any() else np.nan
        return f"{t[i]-t0:5.1f}s: {dist.min():.2f} m from the nearest KF, best view angle to a KF within 1 m {ang:5.1f} deg, {np.linalg.norm(spot - p):.2f} m from the spot"
    print(f"{run}: wait {kv.get('ret_wait_s')} s")
    print("   spot in view  " + desc(i0))
    vm = t0 + 0.5 * sum(meta["phases_s"]["view"])
    iw = int(np.argmin(np.abs(t - vm)))
    sp = np.linalg.norm(np.diff(g[max(iw - 5, 0):iw + 6, 1:4], axis=0), axis=1).sum() / (t[min(iw + 5, len(t) - 1)] - t[max(iw - 5, 0)])
    print(f"   still at the door, mid 'view' (state {stt[iw]}, {sp:.2f} m/s) " + desc(iw))
    print("   tracked again " + (desc(i1) if i1 is not None else "never"))
