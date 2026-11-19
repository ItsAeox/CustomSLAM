#!/usr/bin/env python3
"""Fix the gyro axes of phone recordings made before 2026-10-18 13:00 (web/record.js wrote DeviceMotion rotationRate as
(beta, gamma, alpha) = spec (x, y, z), but Chrome on Android reports alpha = x, beta = y, gamma = z; confirmed against the
Generic Sensor Gyroscope stream gyro_gs.csv, sample for sample). Rewrites mav0/imu0/data.csv with the gyro columns
(x, y, z) = (alpha, beta, gamma) = old columns (3, 1, 2); keeps the original as data_rr_spec_order.csv; notes it in
meta.json. Skips recordings already fixed or recorded with the new recorder (meta imu.rotationRateOrder).

  fix_rec_axes.py <recording dir> [...]
"""
import json
import shutil
import sys
from pathlib import Path

for arg in sys.argv[1:]:
    d = Path(arg)
    meta_p, imu_p = d / "meta.json", d / "mav0" / "imu0" / "data.csv"
    meta = json.loads(meta_p.read_text())
    if meta.get("imu", {}).get("rotationRateOrder"):
        print(f"{d.name}: already {meta['imu']['rotationRateOrder']}, skipped")
        continue
    keep = d / "mav0" / "imu0" / "data_rr_spec_order.csv"
    if not keep.exists():
        shutil.copy(imu_p, keep)
    lines = keep.read_text().splitlines()
    out = [lines[0]]
    for l in lines[1:]:
        if not l.strip():
            continue
        f = l.split(",")
        out.append(",".join([f[0], f[3], f[1], f[2]] + f[4:]))  # gyro x, y, z = alpha, beta, gamma
    imu_p.write_text("\n".join(out) + "\n")
    meta.setdefault("imu", {})["rotationRateOrder"] = "chrome (fixed 2026-10-18 by scripts/fix_rec_axes.py; original: data_rr_spec_order.csv)"
    meta_p.write_text(json.dumps(meta, indent=2))
    print(f"{d.name}: gyro columns fixed ({len(out) - 1} samples)")
