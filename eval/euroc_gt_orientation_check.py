"""Usage: euroc_gt_orientation_check.py <EuRoC sequence dir> [...]

EuRoC Vicon rooms: compare state_groundtruth_estimate0 body orientation with raw Vicon (vicon0 marker pose composed
with vicon0/sensor.yaml T_BS) at the same timestamps. A per-sequence constant offset here would mean the fused GT
orientation, not the SLAM estimate, carries the camera-frame rotation offset seen in rotation ATE."""
import sys
from pathlib import Path

import numpy as np


def quat_wxyz_to_R(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def ang(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def load_csv(p):
    rows = [l.split(",") for l in Path(p).read_text().splitlines() if l and not l.startswith("#")]
    return np.array([[float(v) for v in r[:8]] for r in rows])


def tbs(p):
    txt = Path(p).read_text()
    data = txt[txt.index("data:", txt.index("T_BS")):]
    nums = [float(v) for v in data[data.index("[") + 1:data.index("]")].replace("\n", " ").split(",")]
    return np.array(nums).reshape(4, 4)


for seq in sys.argv[1:]:
    m = Path(seq) / "mav0"
    est = load_csv(m / "state_groundtruth_estimate0" / "data.csv")
    vic = load_csv(m / "vicon0" / "data.csv")
    R_BS = tbs(m / "vicon0" / "sensor.yaml")[:3, :3]
    R_SB = R_BS.T
    vt = vic[:, 0]
    offs = []
    for r in est[::50]:
        j = np.searchsorted(vt, r[0])
        if j >= len(vt) or abs(vt[j] - r[0]) > 5e6:
            continue
        R_wb_est = quat_wxyz_to_R(*r[4:8])
        R_wb_vic = quat_wxyz_to_R(*vic[j, 4:8]) @ R_SB
        offs.append(R_wb_est.T @ R_wb_vic)
    if len(offs) < 10:
        print(f"{Path(seq).name}: too few time-matched samples ({len(offs)})")
        continue
    # constant offset (body frame) between the two and the residual around it
    U, _, Vt = np.linalg.svd(sum(offs))
    C = U @ Vt
    resid = np.median([ang(o @ C.T) for o in offs])
    print(f"{Path(seq).name}: n={len(offs)} median |offset| {np.median([ang(o) for o in offs]):.2f} deg, "
          f"constant part {ang(C):.2f} deg, residual {resid:.2f} deg")
