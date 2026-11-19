# OpenVINS configs for the synthetic set (baselines/openvins/ov_run)

From OpenVINS's config/euroc_mav (rpng/open_vins 6948812): mono (use_stereo false, max_cameras 1), camera intrinsics
not calibrated online (exact in the synthetic set), extrinsics and time offset calibrated online (OpenVINS default),
track_frequency 31 (every frame), one OpenCV thread; 300 points, FAST threshold 10 (200 / 20 found ~60 features on
the rendered textures and diverged on orbit2 even with the ideal IMU: 30 m ATE; 300 / 10: 16 cm, s = 1.04);
static initialisation with jerk threshold 0.5 (1.5: a gentle start never initialises) and the dynamic initialiser on.
- synth_phone: the DeviceMotion-like IMU (mav0/imu0, 60 Hz Chrome pickup, 0.1 rounding, -88 ms): per-sample noise
  ~0.031 m/s^2 / ~8e-4 rad/s (white + rounding) -> densities 0.004 / 1e-4, inflated x5 for the pickup / stamp errors
  (0.02, 5e-4); random walks 3e-3 / 1e-4; time shift -0.088 s (t_imu = t_cam + shift).
- synth_ideal: the ideal-IMU variants (<seq>__imuideal, 100 Hz, exact): EuRoC's noise values, time shift 0.
