#!/usr/bin/env python3
"""Visual-inertial summary per condition of a sweep: <root>/<condition>/<sequence>/<flavour>_seed<k>/.

Per condition: runs, runs with a trajectory, IMU refined (VIBA2 reached), resets ("IMU is not or recently
initialized", "Not enough motion"), SE3 ATE (metric: no scale fitted), Sim3 ATE, scale error |s - 1|, coverage,
bad runs (no trajectory or Sim3 ATE > 200 mm). The phone symptoms (claude/ISSUES.md G15) are: no VIBA2, many resets,
a wrong scale.

  vi_table.py <root> [condition ...]
"""
import json
import statistics as st
import sys
from pathlib import Path


def med(v):
    return st.median(v) if v else float('nan')


def main():
    root = Path(sys.argv[1])
    conds = sys.argv[2:] or sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != 'settings')
    print(f'{"condition":12} {"runs":>4} {"traj":>4} {"VIBA2":>5} {"resets/run":>10} {"no-motion/run":>13} '
          f'{"SE3 m med":>9} {"Sim3 mm med":>11} {"|s-1| med":>9} {"cov med":>7} {"bad":>4}')
    for c in conds:
        runs = sorted((root / c).glob('*/*_seed*'))
        if not runs:
            continue
        traj = viba2 = resets = nomotion = bad = 0
        se3, sim3, serr, cov = [], [], [], []
        for d in runs:
            log = (d / 'run.log').read_text(errors='replace') if (d / 'run.log').exists() else ''
            viba2 += 'end VIBA 2' in log
            resets += log.count('IMU is not or recently initialized')
            nomotion += log.count('Not enough motion')
            m = d / 'metrics.json'
            if not m.exists():
                bad += 1
                continue
            j = json.loads(m.read_text())
            traj += 1
            se3.append(j['ate_se3']['rmse'])
            sim3.append(j['ate_sim3']['rmse'] * 1000)
            serr.append(abs(j['sim3_scale'] - 1))
            cov.append(j['coverage'])
            bad += j['ate_sim3']['rmse'] * 1000 > 200
        n = len(runs)
        print(f'{c:12} {n:4d} {traj:4d} {viba2:5d} {resets / n:10.1f} {nomotion / n:13.1f} {med(se3):9.3f} '
              f'{med(sim3):11.1f} {med(serr):9.3f} {med(cov):7.2f} {bad:4d}')


if __name__ == '__main__':
    main()
