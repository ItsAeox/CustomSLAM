#!/usr/bin/env python3
"""Evaluate an estimated TUM trajectory against a GT TUM trajectory with evo.

Reports, for the poses that can be associated by timestamp:
  - ATE (translation) after SE(3) Umeyama alignment          -> metric VIO accuracy
  - ATE after Sim(3) Umeyama alignment (+ the fitted scale)  -> shape accuracy (monocular, scale-free)
  - ATE rotation (deg) after SE(3) alignment
  - RPE translation / rotation for fixed frame deltas (no alignment needed)
  - coverage: fraction of GT frames that have an estimate (a system that loses tracking
    and outputs nothing for some frames must not look better than one that keeps going)

Usage: evaluate.py --gt gt_tum.txt --est traj_tum.txt --out metrics.json [--plot traj.png] [--label name]
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from evo.core import metrics, sync
from evo.tools import file_interface


def stats(metric):
    s = metric.get_all_statistics()
    return {k: float(s[k]) for k in ("rmse", "mean", "median", "std", "min", "max")}


def ape(ref, est, relation):
    m = metrics.APE(relation)
    m.process_data((ref, est))
    return stats(m)


def rpe(ref, est, relation, delta):
    m = metrics.RPE(relation, delta=delta, delta_unit=metrics.Unit.frames, all_pairs=False)
    m.process_data((ref, est))
    return stats(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--est", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-diff", type=float, default=0.005, help="max timestamp difference for association (s)")
    ap.add_argument("--rpe-deltas", type=int, nargs="+", default=[1, 10])
    ap.add_argument("--plot", type=Path)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    ref_all = file_interface.read_tum_trajectory_file(str(a.gt))
    est_all = file_interface.read_tum_trajectory_file(str(a.est))
    ref, est = sync.associate_trajectories(ref_all, est_all, max_diff=a.max_diff)
    n = ref.num_poses

    res = {
        "label": a.label, "gt": str(a.gt), "est": str(a.est),
        "gt_poses": ref_all.num_poses, "est_poses": est_all.num_poses, "matched_poses": n,
        "coverage": n / ref_all.num_poses if ref_all.num_poses else float("nan"),
        "gt_path_length_m": float(ref.path_length),
    }

    est_se3 = copy.deepcopy(est)
    est_se3.align(ref, correct_scale=False)
    res["ate_se3"] = ape(ref, est_se3, metrics.PoseRelation.translation_part)
    res["ate_se3_rot_deg"] = ape(ref, est_se3, metrics.PoseRelation.rotation_angle_deg)

    est_sim3 = copy.deepcopy(est)
    _, _, scale = est_sim3.align(ref, correct_scale=True)
    res["sim3_scale"] = float(scale)
    res["ate_sim3"] = ape(ref, est_sim3, metrics.PoseRelation.translation_part)

    res["est_path_length_m_raw"] = float(est.path_length)
    for d in a.rpe_deltas:
        if n > d + 1:
            res[f"rpe_trans_d{d}"] = rpe(ref, est, metrics.PoseRelation.translation_part, d)
            res[f"rpe_rot_deg_d{d}"] = rpe(ref, est, metrics.PoseRelation.rotation_angle_deg, d)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2))
    print(f"{a.label or a.est}: matched {n}/{ref_all.num_poses} | "
          f"ATE SE3 {res['ate_se3']['rmse']:.3f} m | ATE Sim3 {res['ate_sim3']['rmse']:.3f} m (s={scale:.3f}) | "
          f"rot {res['ate_se3_rot_deg']['rmse']:.2f} deg")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        P = ref.positions_xyz
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (tr, name) in zip(axes, ((est_se3, "SE(3)"), (est_sim3, f"Sim(3), s={scale:.3f}"))):
            Q = tr.positions_xyz
            ax.plot(P[:, 0], P[:, 1], "k-", lw=1.5, label="ground truth")
            ax.plot(Q[:, 0], Q[:, 1], "-", lw=1, label=f"estimate ({name})")
            ax.set_aspect("equal"); ax.grid(True); ax.legend(); ax.set_title(f"{a.label} — {name} aligned")
        fig.tight_layout()
        fig.savefig(a.plot, dpi=150)


if __name__ == "__main__":
    main()
