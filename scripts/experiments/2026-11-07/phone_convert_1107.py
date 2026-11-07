"""phone_1107: turns each phone run (web/viewer.html files in results/browser/ph1107_*) into orbslam3_run's run-folder
layout, then scores it with the same tools as main_1103 / ext_1103 (eval/trace_to_tum.py, eval/evaluate.py).

Per run -> results/phone_1107/<arm>/<euroc_|synth_><seq>/web_seed<k>/:
  trace.txt       idx state 0 tx ty tz qx qy qz qw 0 0 map_id maps   (state 2 = tracked: the viewer's OK and OK (KLT);
                  the pose is the live pose the page got for that frame, from <base>_live.txt in frame order)
  traj_tum.txt    ORB-SLAM3's final trajectory (<base>_final.txt)
  track_times.txt idx track_ms state
  run_info.txt    frames / processed / dropped / wall_s / seq_s / effective_rate (+ browser fields) from <base>_info.json
  online_tum.txt, metrics_online.json, metrics.json (as the PC batches)
Differences from the PC runs: no metrics_multimap.json (the page does not save each map's trajectory); maps per run
here = distinct map ids that have tracked frames (column map_id).
Usage: python phone_convert_1107.py [--only <substring>]"""
import argparse, json, subprocess
from pathlib import Path

R = Path("H:/Research")
SRC = R / "results/browser"
OUT = R / "results/phone_1107"
EVAL = R / "CustomSLAM/eval"
EUROC = {"MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult", "V1_01_easy", "V1_02_medium",
         "V1_03_difficult", "V2_01_easy", "V2_02_medium", "V2_03_difficult"}


def convert(info: Path):
    base = info.name[: -len("_info.json")]
    tag, rest = base.split("_", 2)[0:2], base.split("_", 2)[2]  # ph1107, <arm>, <seq>_viewer_seed<k>
    arm = tag[1]
    seq, seed = rest.rsplit("_viewer_seed", 1)
    ds = "euroc" if seq in EUROC else "synth"
    d = OUT / arm / f"{ds}_{seq}" / f"web_seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    j = json.loads(info.read_text())
    timing = [l.split(",") for l in (SRC / f"{base}_timing.csv").read_text().splitlines()[1:] if l.strip()]
    live = [l.split() for l in (SRC / f"{base}_live.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
    k = 0
    tr, tt = [], []
    for row in timing:
        idx, state, ms = int(row[0]), int(row[2]), float(row[3])
        mid, nmaps = (int(row[5]), int(row[6])) if len(row) >= 7 else (0, 1)
        tt.append(f"{idx} {ms} {state}")
        if state in (2, 5):
            p = live[k]; k += 1
            tr.append(f"{idx} 2 0 {' '.join(p[1:8])} 0 0 {mid} {nmaps}")
        else:
            tr.append(f"{idx} {state} 0 0 0 0 0 0 0 1 0 0 {mid} {nmaps}")
    if k != len(live):
        raise SystemExit(f"{base}: {k} tracked frames vs {len(live)} live poses")
    (d / "trace.txt").write_text("\n".join(tr) + "\n")
    (d / "track_times.txt").write_text("\n".join(tt) + "\n")
    (d / "traj_tum.txt").write_text((SRC / f"{base}_final.txt").read_text())
    keys = ["frames", "processed", "dropped", "wall_s", "seq_s", "effective_rate", "seed", "settings", "gyro", "pacing",
            "tracked", "maps", "threads", "init_ms", "ua"]
    (d / "run_info.txt").write_text("".join(f"{k_} {j.get(k_)}\n" for k_ in keys))
    seqdir = R / "datasets" / ("euroc" if ds == "euroc" else "synth") / seq
    gt = R / "results/gt" / f"{ds}_{seq}.txt"
    # the eval tools run in WSL with the same Python (and evo) as the PC batches
    wsl = lambda p: "/mnt/h/" + str(p).replace("\\", "/").split(":/", 1)[-1].split(":\\", 1)[-1]
    run = lambda args: subprocess.run(["wsl", "-d", "Ubuntu-22.04", "-e", "/opt/slam/venv/bin/python", *args], stdout=log, stderr=log)
    log = open(d / "eval.log", "w")
    run([wsl(EVAL / "trace_to_tum.py"), "--trace", wsl(d / "trace.txt"), "--seq", wsl(seqdir), "--out", wsl(d / "online_tum.txt")])
    if (d / "online_tum.txt").exists() and (d / "online_tum.txt").stat().st_size > 80:
        run([wsl(EVAL / "evaluate.py"), "--gt", wsl(gt), "--est", wsl(d / "online_tum.txt"), "--out", wsl(d / "metrics_online.json"),
             "--label", f"online phone {arm} {seq} seed{seed}"])
    if (d / "traj_tum.txt").stat().st_size > 40:
        run([wsl(EVAL / "evaluate.py"), "--gt", wsl(gt), "--est", wsl(d / "traj_tum.txt"), "--out", wsl(d / "metrics.json"),
             "--label", f"phone {arm} {seq} seed{seed}"])
    log.close()
    on = json.loads((d / "metrics_online.json").read_text()) if (d / "metrics_online.json").exists() else None
    mids = {r.split()[-2] for r in tr if r.split()[1] == "2"}
    return arm, seq, seed, on, len(mids), j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    for info in sorted(SRC.glob("ph1107_*_info.json")):
        if a.only and a.only not in info.name:
            continue
        arm, seq, seed, on, nm, j = convert(info)
        ate = f"{on['ate_sim3']['rmse'] * 1000:.1f} mm cov {on['coverage']:.2f}" if on else "no online trajectory"
        print(f"{arm:6s} {seq:16s} seed{seed}: online ATE {ate}, maps with tracked frames {nm}, rate {j['effective_rate']:.3f}")


if __name__ == "__main__":
    main()
