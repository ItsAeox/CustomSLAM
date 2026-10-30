#!/usr/bin/env python3
"""Runs on the synthetic set (synth/README.md; replay_arms.sh with PH=datasets/synth) against the exact ground truth.

Per run (<results>/<sequence>/<arm>/seed<n>):
  - tracking (eval/recording_metrics.run_metrics): tracked share, maps, frames before the first map, tracking ms;
  - every map (traj_map<k>_tum.txt) against the ground truth, evo as eval/evaluate.py: Sim3 scale (metres per map unit;
    for the mono-inertial arms the map is meant to be metric, so 1.0 = right), ATE after Sim3 and after SE3 (cm),
    share of the sequence's frames; flags: "short" (ground-truth motion of the map < 5 cm RMS about its centre: scale
    undefined), "poor fit" (Sim3 ATE > 30 % of that RMS: the map's shape is wrong, e.g. a rotation-only map);
  - online scale (Web.Scale.Online, trace columns): eval/scale_online_check.py line, parsed;
  - offline estimators on the largest map's trajectory, each with the phone-like IMU (mav0/imu0/data.csv of the base
    sequence, offset -88 ms) and the ideal IMU (variants/imu_ideal.csv, offset 0): linear alignment
    (scale_linear_align.py --dt 0.5 --bias-prior 0.1) and windows (scale_from_accel.py --window 1 / 2); ratio =
    estimate / the Sim3 scale of that map to the ground truth. Visual (non-inertial) arms only.
Writes <results>/scores.json and <results>/RESULTS.md (raw rows per scenario, every arm; per-scenario medians).

  synth_results.py <results dir> [--data datasets/synth] [--gt results/gt] [--jobs 8] [--offset-ms -88]
"""
import argparse
import copy
import json
import re
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from evo.core import sync
from evo.tools import file_interface

EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL))
from recording_metrics import run_metrics  # noqa: E402

A = None


def num(pat, text, cast=float):
    m = re.search(pat, text)
    return cast(m.group(1)) if m else None


def run_tool(args):
    r = subprocess.run([sys.executable] + [str(x) for x in args], capture_output=True, text=True, timeout=900)
    out = (r.stdout.strip().splitlines() or [""])[-1]
    return out if r.returncode == 0 else f"error: {(r.stderr.strip().splitlines() or [''])[-1]}"


def map_score(f, gt_traj, n_frames):
    est = file_interface.read_tum_trajectory_file(str(f))
    if est.num_poses < 10:
        return {"file": f.name, "poses": int(est.num_poses), "flag": "tiny"}
    ref, est = sync.associate_trajectories(copy.deepcopy(gt_traj), est, max_diff=0.005)
    g = ref.positions_xyz
    rms = float(np.sqrt(((g - g.mean(0)) ** 2).sum(1).mean()))
    out = {"file": f.name, "poses": int(est.num_poses), "share": est.num_poses / n_frames, "gt_rms_m": rms,
           "gt_path_m": float(np.linalg.norm(np.diff(g, axis=0), axis=1).sum()), "t0": float(ref.timestamps[0]),
           "t1": float(ref.timestamps[-1])}
    se3 = copy.deepcopy(est)
    se3.align(ref, correct_scale=False)
    out["ate_se3_cm"] = 100 * float(np.sqrt((np.linalg.norm(se3.positions_xyz - g, axis=1) ** 2).mean()))
    sim = copy.deepcopy(est)
    _, _, s = sim.align(ref, correct_scale=True)
    out["sim3_scale"] = float(s)
    out["ate_sim3_cm"] = 100 * float(np.sqrt((np.linalg.norm(sim.positions_xyz - g, axis=1) ** 2).mean()))
    out["flag"] = "short" if rms < 0.05 else ("poor fit" if out["ate_sim3_cm"] / 100 > 0.3 * rms else "")
    return out


def offline(traj, imu, tbc, gt, off):
    res = {}
    for key, tool, extra in (("align", "scale_linear_align.py", ["--dt", "0.5", "--bias-prior", "0.1"]),
                             ("w1", "scale_from_accel.py", ["--window", "1"]),
                             ("w2", "scale_from_accel.py", ["--window", "2"])):
        line = run_tool([EVAL / tool, "--traj", traj, "--imu", imu, "--tbc", tbc, "--gt", gt, "--offset-ms", off] + extra)
        res[key] = {"ratio": num(r"ratio ([\d.]+)", line), "g": num(r"\|g\| linear ([\d.]+)", line),
                    "sigma_pct": num(r"1-sigma ([\d.]+) %", line), "line": line}
    return res


def score(run):
    run = Path(run)
    seq_name, arm, seed = run.parts[-3], run.parts[-2], run.parts[-1]
    base = seq_name.split("__")[0]
    seq, bseq = A.data / seq_name, A.data / base
    gt = A.gt / f"synth_{seq_name}.txt"
    n_frames = sum(1 for l in (seq / "mav0" / "cam0" / "data.csv").read_text().splitlines() if l and l[0] != "#")
    out = {"run": str(run), "seq": seq_name, "base": base, "scenario": re.match(r"[a-z]+", base).group(0),
           "variant": seq_name[len(base):].lstrip("_") or ("clean" if base.endswith("_clean") else "phone"),
           "arm": arm, "seed": seed, "inertial": (run / "run_info.txt").exists() and
           "mono_inertial" in (run / "run_info.txt").read_text(), "finished": (run / "run_info.txt").exists()}
    log = (run / "run.log").read_text(errors="ignore") if (run / "run.log").exists() else ""
    out["aborted"] = (not out["finished"]) or bool(re.search(r"Aborted|Segmentation|RuntimeError|abort\(", log))
    # ORB-SLAM3's own messages: map resets (all causes), resets because the IMU was not initialised yet on a loss, the
    # local mapper's bad-IMU flag, and the visual-inertial BA stages of the IMU initialisation (VIBA 1 / 2 = done)
    out["log"] = {k: log.count(s) for k, s in (("resets", "Reseting active map in monocular case"),
                                                ("reset_imu_not_init", "IMU is not or recently initialized"),
                                                ("reset_bad_imu", "bad imu flag"), ("viba1", "start VIBA 1"),
                                                ("viba2", "start VIBA 2"))}
    m = run_metrics(run)
    out.update({k: (float(v) if v == v else None) for k, v in m.items()})
    gtt = file_interface.read_tum_trajectory_file(str(gt))
    out["maps_scored"] = [map_score(f, gtt, n_frames) for f in sorted(run.glob("traj_map*_tum.txt"))]
    good = [x for x in out["maps_scored"] if "sim3_scale" in x]
    big = max(good, key=lambda x: x["poses"]) if good else None
    out["largest"] = big
    if (run / "trace.txt").exists() and not out["inertial"]:
        line = run_tool([EVAL / "scale_online_check.py", run, seq, "--gt", gt])
        out["online"] = {"line": line, "tracked_s": num(r"tracked (\d+) s", line), "est_s": num(r"with an estimate (\d+) s", line),
                         "conv_s": num(r"converged (\d+) s", line), "first_after_s": num(r"first after ([\d.]+) s", line),
                         "err_med_pct": num(r"median ([\d.]+) %", line), "err_p90_pct": num(r"p90 ([\d.]+) %", line),
                         "within5_pct": num(r"within 5 % (\d+) %", line), "err_last_pct": num(r"last ([\d.]+) %", line)}
    if big and big["flag"] == "" and not out["inertial"]:
        traj, tbc = run / big["file"], seq / "settings.yaml"
        out["offline_phone"] = offline(traj, bseq / "mav0" / "imu0" / "data.csv", tbc, gt, A.offset_ms)
        out["offline_ideal"] = offline(traj, bseq / "variants" / "imu_ideal.csv", tbc, gt, 0)
    print(f"{seq_name} {arm} {seed}: tracked {100 * (out['tracked'] or 0):.0f} %, maps {len(good)}, "
          f"largest s={big['sim3_scale']:.3f} ATE sim3 {big['ate_sim3_cm']:.1f} cm {big['flag']}" if big else
          f"{seq_name} {arm} {seed}: no map", flush=True)
    return out


def f(v, fmt="{:.3f}"):
    return "-" if v is None else fmt.format(v)


def table(rows):
    lines = ["| Sequence | IMU | Arm | Seed | Tracked % | Maps | 1st map (frame) | Largest map: frames % | Sim3 scale | "
             "ATE Sim3 cm | ATE SE3 cm | Flag | Map resets (IMU not init / bad IMU), IMU init done | Online: converged s (after s) | Online err median / p90 / last % | "
             "Offline phone IMU: align / w1 / w2 | Offline ideal IMU: align / w1 / w2 |", "|" + "---|" * 17]
    for r in rows:
        b = r["largest"] or {}
        o = r.get("online", {})
        op, oi = r.get("offline_phone", {}), r.get("offline_ideal", {})
        trip = lambda d: " / ".join(f(d.get(k, {}).get("ratio")) for k in ("align", "w1", "w2")) if d else "-"
        lines.append(
            f"| {r['base']} | {r['variant']} | {r['arm']} | {r['seed'][4:]} | {f(100 * (r['tracked'] or 0), '{:.0f}')} | "
            f"{len([x for x in r['maps_scored'] if 'sim3_scale' in x])} | {f(r.get('before_map'), '{:.0f}')} | "
            f"{f(100 * b['share'], '{:.0f}') if b else '-'} | {f(b.get('sim3_scale'))} | {f(b.get('ate_sim3_cm'), '{:.1f}')} | "
            f"{f(b.get('ate_se3_cm'), '{:.1f}')} | {b.get('flag', '') if b else ('aborted' if r['aborted'] else 'no map')}"
            f"{' (aborted)' if b and r['aborted'] else ''} | "
            f"{'{} ({} / {}), {}'.format(r['log']['resets'], r['log']['reset_imu_not_init'], r['log']['reset_bad_imu'], 'yes' if r['log']['viba2'] else 'no') if r['inertial'] else '-'} | "
            f"{f(o.get('conv_s'), '{:.0f}')} ({f(o.get('first_after_s'), '{:.1f}')}) | "
            f"{f(o.get('err_med_pct'), '{:.1f}')} / {f(o.get('err_p90_pct'), '{:.1f}')} / {f(o.get('err_last_pct'), '{:.1f}')} | "
            f"{trip(op)} | {trip(oi)} |")
    return "\n".join(lines)


def medians(rows):
    """Per arm and IMU over a scenario's runs: medians of the raw columns (no ranking)."""
    out = ["| Arm | IMU | Runs | Tracked % | Maps | Sim3 scale (largest, unflagged) | ATE Sim3 cm | ATE SE3 cm | "
           "Online converged in runs | Online err median % | Offline phone align / w1 / w2 | Offline ideal align / w1 / w2 |",
           "|" + "---|" * 12]
    keys = sorted({(r["arm"], r["variant"]) for r in rows})
    md = lambda xs, fmt="{:.3f}": f(float(np.median(xs)), fmt) if xs else "-"
    for arm, var in keys:
        rs = [r for r in rows if r["arm"] == arm and r["variant"] == var]
        bl = [r["largest"] for r in rs if r["largest"] and r["largest"]["flag"] == ""]
        on = [r["online"] for r in rs if r.get("online")]
        trip = lambda key: " / ".join(md([r[key][k]["ratio"] for r in rs if r.get(key) and r[key][k]["ratio"] is not None])
                                      for k in ("align", "w1", "w2"))
        out.append(f"| {arm} | {var} | {len(rs)} | {md([100 * (r['tracked'] or 0) for r in rs], '{:.0f}')} | "
                   f"{md([len([x for x in r['maps_scored'] if 'sim3_scale' in x]) for r in rs], '{:.0f}')} | "
                   f"{md([b['sim3_scale'] for b in bl]) if any(r['inertial'] for r in rs) else '-'} | "  # visual maps: arbitrary units
                   f"{md([b['ate_sim3_cm'] for b in bl], '{:.1f}')} | "
                   f"{md([b['ate_se3_cm'] for b in bl], '{:.1f}')} | "
                   f"{sum(1 for o in on if o.get('conv_s'))} of {len(on)} | {md([o['err_med_pct'] for o in on if o.get('err_med_pct') is not None], '{:.1f}')} | "
                   f"{trip('offline_phone')} | {trip('offline_ideal')} |")
    return "\n".join(out)


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--data", default="/mnt/h/Research/datasets/synth")
    ap.add_argument("--gt", default="/mnt/h/Research/results/gt")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--offset-ms", type=float, default=-88)
    A = ap.parse_args()
    A.results, A.data, A.gt = Path(A.results), Path(A.data), Path(A.gt)
    runs = sorted(str(p.parent) for p in A.results.glob("*/*/seed*/run.log"))
    with Pool(A.jobs) as pool:
        rows = pool.map(score, runs)
    (A.results / "scores.json").write_text(json.dumps(rows, indent=1))
    order = {"phone": 0, "clean": 1, "imuideal": 2}
    rows.sort(key=lambda r: (r["scenario"], r["base"], order.get(r["variant"], 3), r["arm"], r["seed"]))
    text = ["# Synthetic set: runs against the exact ground truth (eval/synth_results.py)", "",
            "Sim3 scale: metres per map unit of the largest map (mono-inertial arms: 1.0 = metric). Online / offline "
            "errors and ratios: estimate / that map's true scale. Flags: short = < 5 cm of motion (scale undefined), "
            "poor fit = Sim3 ATE > 30 % of the map's motion (wrong shape); flagged maps get no offline estimate. "
            f"Offline phone IMU at {A.offset_ms:.0f} ms, ideal IMU at 0 ms.", ""]
    for sc in sorted({r["scenario"] for r in rows}):
        rs = [r for r in rows if r["scenario"] == sc]
        text += [f"## {sc}", "", medians(rs), "", table(rs), ""]
    (A.results / "RESULTS.md").write_text("\n".join(text), encoding="utf-8")
    print(f"{len(rows)} runs -> {A.results / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
