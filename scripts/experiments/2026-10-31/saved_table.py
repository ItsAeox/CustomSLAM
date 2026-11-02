"""Saved room (saved_room_run.sh): session A (map the bedroom, place, save) and session B (a later run that loads the
map) per arm. A: the object's error before saving, the saved map vs the object's map, the map size. B per start ('start':
the first frame, in the bedroom; 'out': 3 s after leaving it): relocalised (WEBMAP-STATS reloc_after_s), the object first
shown (first_shown_s from B's first frame) and its px in the 2 s after, coming back (ret_wait_s, ret_px_med), wrong
(> 50 px) and phantom frames (shown, in the image, spot not in view). Medians over routes x saved maps, then every run.
Usage: saved_table.py <batch dir>"""
import sys, re, numpy as np
from pathlib import Path
from collections import defaultdict

root = Path(sys.argv[1])
kv = lambda p: dict(re.findall(r"(\w+)=(\S+)", p.read_text(errors="replace"))) if p.exists() else {}
num = lambda d, k: float(d[k]) if k in d and d[k] not in ("-", "None") else np.nan
print(f"# Saved room: {root.name}\n")
print("## Session A (flatfirst0 to 38.5 s: map the bedroom, place, save)\n")
print("| Arm | Seed | Placed | Object px (median, look) | Object map / saved map | Saved keyframes / points / MB |")
print("|---|---|---|---|---|---|")
for sd in sorted(root.glob("A/*/*/seed*")):
    d = kv(sd / "anchor_refine.txt")
    log = (sd / "run.log").read_text(errors="replace") if (sd / "run.log").exists() else ""
    m = re.search(r"WEBMAP saved map (\d+): (\d+) keyframes, (\d+) map points, (\d+) bytes", log)
    print(f"| {sd.parts[-2]} | {sd.name[4:]} | {d.get('placed', '-')} | {d.get('px_med', '-')} | {d.get('place_map', '-')} / "
          f"{m.group(1) if m else 'not saved'} | {f'{m.group(2)} / {m.group(3)} / {int(m.group(4)) / 1e6:.1f}' if m else '-'} |")
rows = defaultdict(list)
runs = []
for sd in sorted(root.glob("B/*/*/seed1")):
    route, (arm, mode, sa) = sd.parts[-3], sd.parts[-2].split("__")
    d = kv(sd / "anchor_saved.txt")
    log = (sd / "run.log").read_text(errors="replace") if (sd / "run.log").exists() else ""
    m = re.search(r"WEBMAP-STATS reloc_after_s=(\S+)", log)
    r = {"route": route, "arm": arm, "mode": mode, "map": sa, "reloc": float(m.group(1)) if m else np.nan,
         "first": num(d, "first_shown_s"), "first2": num(d, "first2s_px_med"), "wait": num(d, "ret_wait_s"),
         "retpx": num(d, "ret_px_med"), "retshown": num(d, "ret_shown_pct"), "shown": num(d, "shown_pct"),
         "wrong": num(d, "wrong_pct"), "phantom": num(d, "phantom_frames"), "ok": "placed=1" in (sd / "anchor_saved.txt").read_text(errors="replace") if (sd / "anchor_saved.txt").exists() else False}
    rows[(mode, arm)].append(r); runs.append(r)
med = lambda v: f"{np.nanmedian(v):.1f}" if np.isfinite(v).any() else "-"
for mode, lab in (("start", "from the first frame (in the bedroom, best case)"), ("out", "from 3 s after leaving the bedroom")):
    print(f"\n## Session B {lab}: medians over 6 routes x 2 saved maps\n")
    print("| Arm | Runs | Relocalised (runs / median s) | Object first shown (s from B's start; never) | px in the 2 s after | "
          "Back in the bedroom: wait s / px / shown % | Wrong % (> 50 px) | Phantom frames (median / runs with any) |")
    print("|---|---|---|---|---|---|---|---|")
    for (m_, arm), rs in sorted(rows.items()):
        if m_ != mode: continue
        g = lambda k: np.array([r[k] for r in rs], float)
        rel, first = g("reloc"), g("first")
        print(f"| {arm} | {len(rs)} | {int((rel >= 0).sum())} / {med(np.where(rel >= 0, rel, np.nan))} | "
              f"{med(np.where(first >= 0, first, np.nan))}; never {int((first < 0).sum() + np.isnan(first).sum())} | {med(g('first2'))} | "
              f"{med(np.where(g('wait') >= 0, g('wait'), np.nan))} / {med(g('retpx'))} / {med(g('retshown'))} | {med(g('wrong'))} | "
              f"{med(g('phantom'))} / {int((g('phantom') > 0).sum())} |")
print("\n## Every session B run\n")
print("| Start | Arm | Route | Saved map (A seed) | Reloc s | First shown s | px 2 s | Return wait s | Return px | Return shown % | Wrong % | Phantom |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in sorted(runs, key=lambda r: (r["mode"], r["arm"], r["route"], r["map"])):
    f = lambda k: "-" if np.isnan(r[k]) else f"{r[k]:g}"
    print(f"| {r['mode']} | {r['arm']} | {r['route']} | {r['map']} | {f('reloc')} | {f('first')} | {f('first2')} | {f('wait')} | "
          f"{f('retpx')} | {f('retshown')} | {f('wrong')} | {f('phantom')} |")
