#!/usr/bin/env python3
"""Tables of eval/anchor_return.py over a replay results dir (<dir>/<seq>/<arm>/seed<N>/anchor.txt), per scenario and
arm: runs with the object placed, share of the in-view frames it is shown, screen error (median of the per-run medians),
wrong placements (shown > 50 px off), the return phase, and how runs end (object still lost when the spot is in view,
deleted by a map reset). Raw per-run lines below the tables.
  anchor_summary.py <results dir>
"""
import re
import sys
from pathlib import Path

import numpy as np


def parse(f):
    d = {}
    for kv in f.read_text().split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            try:
                d[k] = float(v)
            except ValueError:
                d[k] = v
    return d


def med(xs):
    xs = [x for x in xs if x is not None]
    return f"{np.median(xs):.1f}" if xs else "-"


root = Path(sys.argv[1])
runs = {}
for f in sorted(root.glob("*/*/seed*/anchor.txt")):
    seq, arm = f.parts[-4], f.parts[-3]
    runs.setdefault((re.sub(r"\d+$", "", seq), arm), []).append((seq, f.parts[-2], parse(f)))
print(f"# Object placed, leave, come back (eval/anchor_return.py): {root.name}\n")
print("Per scenario and arm (medians over runs; shown % = of the frames after placement where the spot is in the true view;"
      " px / cm = the object's screen / room error while shown; wrong = shown > 50 px off; return = ar 'return' phase,"
      " longar 'in' .. 'orbit2', flat: back in the bedroom .. the end):\n")
print("| Scenario | Arm | Runs | Placed | Shown % | px | cm | Wrong % | Return: shown % | Return px | Lost at the end | Deleted | Longest gap s |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for (sc, arm), v in sorted(runs.items()):
    d = [x[2] for x in v]
    pl = [x for x in d if x.get("placed") == 1]
    g = lambda k: [x.get(k) for x in pl if isinstance(x.get(k), float)]
    print(f"| {sc} | {arm} | {len(d)} | {len(pl)} | {med(g('shown_pct'))} | {med(g('px_med'))} | {med(g('cm_med'))} | "
          f"{med(g('wrong_pct'))} | {med(g('ret_shown_pct'))} | {med(g('ret_px_med'))} | "
          f"{int(sum(g('unrecovered_at_end')))} of {len(pl)} | {int(sum(g('anchor_deleted')))} | {med(g('gap_max_s'))} |")
print("\nPer run:\n")
print("| Sequence | Arm | Seed | Shown % | px med / p90 | cm med | Wrong % | Return shown % / px | Gaps (longest s) | Lost at end | Maps | Events |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for (sc, arm), v in sorted(runs.items()):
    for seq, seed, x in v:
        if x.get("placed") != 1:
            print(f"| {seq} | {arm} | {seed} | not placed ({x.get('why', 'not tracking at the tap')}) |||||||||"); continue
        print(f"| {seq} | {arm} | {seed} | {x.get('shown_pct', '-')} | {x.get('px_med', '-')} / {x.get('px_p90', '-')} | "
              f"{x.get('cm_med', '-')} | {x.get('wrong_pct', '-')} | {x.get('ret_shown_pct', '-')} / {x.get('ret_px_med', '-')} | "
              f"{int(x.get('gaps', 0))} ({x.get('gap_max_s', '-')}) | {int(x.get('unrecovered_at_end', 0))} | "
              f"{int(x.get('maps_max', 0))} | {x.get('events', '-')} |")
