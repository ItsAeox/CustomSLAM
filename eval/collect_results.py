#!/usr/bin/env python3
"""Collect every evaluated run under results/ into one table (results/all_runs.csv) for the paper's tables and graphs.

A run is a directory with metrics.json (eval/evaluate.py), usually laid out as
  results/<experiment>/[<variant>/]<sequence>/<flavour>_seed<k>/
with run_info.txt (orbslam3_run) and track_times.txt next to it. One row per run:
  experiment, variant, sequence, dataset, flavour, seed, path,
  ATE (Sim3 / SE3 RMSE, mm), rotation ATE (deg), Sim3 scale, coverage, matched poses,
  tracking time (median / p95 / mean ms over processed frames), frames processed / dropped,
  pacing protocol (see below), effective playback rate, settings file, vocabulary, descriptor (from the settings).

Pacing protocol (ISSUES H1): "rt-wall" (honest real time), "live@<rate>" (camera emulation, late frames dropped),
"realtime-legacy" (ORB-SLAM3 examples' protocol, which played at ~0.5-0.7x on /mnt/h: comparable within a sweep only),
"unpaced".

Usage: collect_results.py [results_dir] [--out all_runs.csv]
"""
import argparse
import csv
import json
import re
import statistics as st
from pathlib import Path


def read_info(p: Path) -> dict:
    out = {}
    if p.exists():
        for l in p.read_text(errors="replace").splitlines():
            k, _, v = l.partition(" ")
            out[k] = v.strip()
    return out


def track_stats(p: Path):
    if not p.exists():
        return None
    ms = []
    for l in p.read_text(errors="replace").splitlines():
        f = l.split()
        if len(f) >= 2:
            try:
                ms.append(float(f[1]))
            except ValueError:
                pass
    if not ms:
        return None
    s = sorted(ms)
    return st.median(s), s[min(len(s) - 1, int(0.95 * len(s)))], st.mean(s), len(s)


def protocol(info: dict) -> str:
    if info.get("rt_wall") == "1":
        return "rt-wall"
    if info.get("live") == "1":
        return f"live@{float(info.get('rate_scale', '1')):g}"
    if info.get("realtime") == "1":
        return "realtime-legacy"
    return "unpaced"


def settings_tags(path: str) -> dict:
    tags = {"descriptor": "orb", "web_keys": ""}
    p = Path(path) if path else None
    if p is not None and p.is_file():
        txt = p.read_text(errors="replace")
        keys = re.findall(r"^(Web\.[\w.]+):\s*(\S+)", txt, re.M)
        tags["web_keys"] = ";".join(f"{k}={v}" for k, v in keys)
        if any(k == "Web.Descriptor" and v == "1" for k, v in keys):
            tags["descriptor"] = "teblid"
    return tags


def wsl_to_win(p: str) -> str:  # /mnt/h/Research/... -> H:/Research/...
    m = re.match(r"/mnt/([a-z])/(.*)", p)
    return f"{m.group(1).upper()}:/{m.group(2)}" if m else p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default=str(Path(__file__).resolve().parents[2] / "results"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.results)
    out = Path(a.out) if a.out else root / "all_runs.csv"
    rows = []
    settings_cache = {}
    # a run has metrics.json (largest map, eval/evaluate.py) and/or metrics_multimap.json (every map,
    # eval/evaluate_multimap.py; ISSUES H3): a multi-map run whose largest map misses the GT has only the latter
    runs = sorted({p.parent for p in root.rglob("metrics.json")} | {p.parent for p in root.rglob("metrics_multimap.json")})
    for run in runs:
        rel = run.relative_to(root).parts
        m = re.match(r"(native|wasm)_seed(\d+)$", rel[-1])
        flavour, seed = (m.group(1), int(m.group(2))) if m else ("", None)
        experiment = rel[0]
        sequence = rel[-2] if len(rel) >= 3 else ""
        variant = "/".join(rel[1:-2]) if len(rel) >= 4 else ""
        met, mm = {}, {}
        try:
            if (run / "metrics.json").exists():
                met = json.loads((run / "metrics.json").read_text())
            if (run / "metrics_multimap.json").exists():
                mm = json.loads((run / "metrics_multimap.json").read_text())
        except Exception:
            continue
        nan = float("nan")
        info = read_info(run / "run_info.txt")
        ts = track_stats(run / "track_times.txt")
        sset = info.get("settings", "")
        if sset not in settings_cache:
            settings_cache[sset] = settings_tags(wsl_to_win(sset))
        tags = settings_cache[sset]
        ds = info.get("dataset") or (sequence.split("_")[0] if "_" in sequence else "")
        rows.append({
            "experiment": experiment, "variant": variant, "sequence": sequence, "dataset": ds,
            "flavour": flavour, "seed": seed, "path": str(run.relative_to(root)).replace("\\", "/"),
            "ate_sim3_mm": round(met["ate_sim3"]["rmse"] * 1000, 2) if met else "",
            "ate_se3_mm": round(met["ate_se3"]["rmse"] * 1000, 2) if met else "",
            "rot_ate_deg": round(met["ate_se3_rot_deg"]["rmse"], 3) if met else "",
            "sim3_scale": round(met.get("sim3_scale", nan), 4) if met else "",
            "coverage": round(met.get("coverage", nan), 4) if met else "",
            "matched_poses": met.get("matched_poses"), "gt_poses": met.get("gt_poses"),
            "track_ms_med": round(ts[0], 2) if ts else "", "track_ms_p95": round(ts[1], 2) if ts else "",
            "track_ms_mean": round(ts[2], 2) if ts else "", "frames_tracked_calls": ts[3] if ts else "",
            "frames": info.get("frames", ""), "processed": info.get("processed", ""), "dropped": info.get("dropped", ""),
            "protocol": protocol(info), "rate_scale": info.get("rate_scale", ""),
            "effective_rate": info.get("effective_rate", ""), "sync_mapping": info.get("sync_mapping", ""),
            "map_points": info.get("map_points", ""), "keyframes": info.get("keyframes", ""),
            "maps": info.get("maps", ""), "mem_mb": info.get("mem_mb", ""),  # final map size; WASM heap / native peak RSS
            "settings": sset, "descriptor": tags["descriptor"], "web_keys": tags["web_keys"],
            "clahe": info.get("clahe", ""), "gyro": info.get("gyro", ""), "saved_maps": info.get("saved_maps", ""),
            # every map scored separately (H3); empty when the run predates per-map trajectories
            "mm_maps_scored": mm.get("maps_scored", ""),
            "mm_coverage": round(mm["coverage_union"], 4) if mm else "",
            "mm_ate_sim3_pooled_mm": round(mm["ate_sim3_pooled"] * 1000, 2) if mm else "",
        })
    if not rows:
        print("no runs found")
        return
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    by = {}
    for r in rows:
        by.setdefault((r["experiment"], r["protocol"]), 0)
        by[(r["experiment"], r["protocol"])] += 1
    print(f"{len(rows)} runs -> {out}")
    for (e, p), n in sorted(by.items()):
        print(f"  {e:18s} {p:16s} {n:4d}")


if __name__ == "__main__":
    main()
