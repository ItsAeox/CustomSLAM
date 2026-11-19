"""Usage: imu_per_frame.py <ASL sequence dir>...

IMU samples per frame interval for ASL sequences, as ORB-SLAM3's PreintegrateIMU sees them:
samples with t in [t_prev - 1 ms, t] (mImuPer = 0.001). The G13 hold only acts when this count is 0 or 1."""
import sys
from pathlib import Path

import numpy as np

for seq in sys.argv[1:]:
    seq = Path(seq)
    cam = np.loadtxt(seq / 'mav0' / 'cam0' / 'data.csv', delimiter=',', comments='#', usecols=0)
    imu = np.loadtxt(seq / 'mav0' / 'imu0' / 'data.csv', delimiter=',', comments='#', usecols=0)
    lo = np.searchsorted(imu, cam[:-1] - 1e6, side='left')
    hi = np.searchsorted(imu, cam[1:], side='right')
    n = hi - lo
    print(f'{seq.name}: {cam.size} frames, IMU samples per frame min {n.min()} median {int(np.median(n))}, '
          f'frames with <= 1 sample: {(n <= 1).sum()}')
