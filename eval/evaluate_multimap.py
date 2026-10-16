#!/usr/bin/env python3
"""Evaluate a multi-map run: every map's trajectory (traj_map<id>_tum.txt, orbslam3_run) against the GT.

ORB-SLAM3 starts a new map after it loses tracking; the standard trajectory file keeps only the largest map
(claude/ISSUES.md H3). On sequences with partial GT (TUM-VI outdoors / Magistrale: start and end in the mocap
room) the largest map can lie entirely outside the GT, which made such runs unscorable. Here each map is aligned to
the GT on its own (Sim(3); monocular maps have independent scales) and reported, plus:
  - coverage_union: fraction of GT poses covered by any map
  - ate_sim3_pooled: RMS over every matched pose of every map (each map aligned separately), weighted by poses
  - maps_scored: maps with >= --min-poses matched poses (shorter maps are listed but not aligned)

Usage: evaluate_multimap.py --gt gt_tum.txt --run <run dir> --out metrics_multimap.json [--min-poses 20]
"""
import argparse
import copy
import json
import re
from pathlib import Path

import numpy as np
from evo.core import metrics, sync
from evo.tools import file_interface


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-diff", type=float, default=0.005)
    ap.add_argument("--min-poses", type=int, default=20)
    a = ap.parse_args()

    gt = file_interface.read_tum_trajectory_file(str(a.gt))
    gt_t = gt.timestamps
    covered = np.zeros(gt_t.size, dtype=bool)
    maps, sq_sum, n_sum = [], 0.0, 0
    files = sorted(a.run.glob("traj_map*_tum.txt"), key=lambda p: int(re.search(r"traj_map(\d+)_", p.name).group(1)))
    for f in files:
        mid = int(re.search(r"traj_map(\d+)_", f.name).group(1))
        est_all = file_interface.read_tum_trajectory_file(str(f))
        entry = {"map": mid, "est_poses": est_all.num_poses, "matched_poses": 0}
        try:
            ref, est = sync.associate_trajectories(gt, est_all, max_diff=a.max_diff)
        except sync.SyncException:
            maps.append(entry)
            continue
        n = ref.num_poses
        entry["matched_poses"] = n
        idx = np.searchsorted(gt_t, ref.timestamps)
        covered[np.clip(idx, 0, gt_t.size - 1)] = True
        if n >= a.min_poses:
            e = copy.deepcopy(est)
            r, t, s = e.align(ref, correct_scale=True)
            m = metrics.APE(metrics.PoseRelation.translation_part)
            m.process_data((ref, e))
            err = m.error
            entry.update({"ate_sim3_rmse": float(np.sqrt(np.mean(err ** 2))), "sim3_scale": float(s),
                          "t_first": float(ref.timestamps[0]), "t_last": float(ref.timestamps[-1])})
            sq_sum += float(np.sum(err ** 2))
            n_sum += err.size
        maps.append(entry)
    res = {
        "gt": str(a.gt), "run": str(a.run), "gt_poses": int(gt_t.size), "maps": maps,
        "maps_total": len(maps), "maps_scored": sum("ate_sim3_rmse" in m for m in maps),
        "coverage_union": float(covered.mean()) if gt_t.size else float("nan"),
        "ate_sim3_pooled": float(np.sqrt(sq_sum / n_sum)) if n_sum else float("nan"),
    }
    a.out.write_text(json.dumps(res, indent=1))
    print(f"{a.run.name}: maps {res['maps_total']} scored {res['maps_scored']} coverage {res['coverage_union']:.2f} "
          f"pooled Sim3 ATE {res['ate_sim3_pooled'] * 1000:.1f} mm")


if __name__ == "__main__":
    main()
