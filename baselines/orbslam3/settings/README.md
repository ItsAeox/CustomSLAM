# ORB-SLAM3 settings used for evaluation

Copied unmodified from ORB-SLAM3 v1.0 (`4452a3c`, GPL-3.0), so that baseline and port runs use exactly the
configuration published by the ORB-SLAM3 authors:

| File | Source |
|---|---|
| `EuRoC_mono.yaml` | `Examples/Monocular/EuRoC.yaml` (pinhole + radtan, resized to 600x350, 1000 ORB features) |
| `TUM-VI_mono.yaml` | `Examples/Monocular/TUM-VI.yaml` (Kannala-Brandt, 512x512, 1500 ORB features; run with CLAHE as `mono_tum_vi.cc`) |

TUM-VIE settings are generated from each sequence's calibration by `../make_settings.py`.
