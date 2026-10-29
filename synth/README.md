# WebVIO synthetic dataset

Rendered phone sequences with exact ground truth, for the metric-scale and IMU work (claude/ISSUES.md G25) and the
evaluation. Every sequence is in the **phone recording layout** (record.html), so the AR page, the viewer, the replay
batch scripts and the evaluation tools run on it unchanged; the dev server lists it as dataset `synth`.

## Contents

7 motion scenarios x 6 variants = 42 sequences (plus a clean render of variant 0 of each scenario, `*_clean`),
in three procedurally textured rooms (`living` 6 x 5 m, `bedroom` 4.5 x 4 m, `office` 5 x 4 m), 15-65 s each, 29.97 fps,
480 x 640 portrait.

| Scenario | What | Variants (k = 0..5) |
|---|---|---|
| `slide` | phone resting on a table edge, slides of exactly **1.00 m** between two marks (the user's real test) | 13 / 6 / 4 / 3 / 2 / 1.5 s per slide |
| `gesture` | hand-held brisk side-to-side "scale gesture", then looking around | amplitude 0.1-0.3 m, 0.7-1.6 Hz |
| `shake` | hand-held random shaking bursts between calm phases | 1.5-3.5 cm, 2-10 deg RMS |
| `rotate` | standing, turning the phone: pans, tilts, rolls, some full turns | peak 30-240 deg/s |
| `orbit` | walking an arc around a point while looking at it, back and forth | radius 0.8-1.4 m, 0.25-0.6 m/s |
| `walk` | walking through the room with gait bob / sway and head turns | 0.4-1.0 m/s |
| `ar` | an AR session: still, sideways init, look at a floor spot and tap, half circle around it, look away and back, fast turn, return | radius 1.0-1.3 m, turns 120-240 deg/s |

Every sequence starts with >= 2.5 s of stillness. The rooms contain objects of exactly known size (`scene.json`
`known`: a 1 x 1 m checkerboard with 10 cm squares on the floor, tables, desks).

## Files per sequence (`datasets/synth/<name>/`)

| File | Content |
|---|---|
| `mav0/cam0/data/frame_<i>.jpg`, `mav0/cam0/data.csv` | frames (JPEG q92), **reported** stamps (with the measured camera stamp jitter) |
| `mav0/imu0/data.csv` | DeviceMotion-like IMU: 60 Hz Chrome tick stamps, values rounded to 0.1 m/s^2 / 0.1 deg/s, camera-IMU offset -88 ms |
| `accel_gs.csv`, `gyro_gs.csv` | the same readings with the sensor's own stamps (+ the same offset) |
| `variants/imu_ideal.csv` | the true signal at the 100 Hz sensor times, exact stamps (validation) |
| `variants/imu_chrome_sensorstamps.csv` | DeviceMotion values with sensor stamps instead of tick stamps |
| `variants/cam_exact.csv` | frame stamps without jitter |
| `mav0/state_groundtruth_estimate0/data.csv` | body (IMU) ground truth, EuRoC format, 200 Hz: p, q (w x y z), v, gyro / acc bias |
| `gt/camera_dense.csv` | camera poses (Twc, TUM order) at 200 Hz on the true clock |
| `settings.yaml`, `meta.json`, `scene.json`, `touches.csv` | exact intrinsics; every generator parameter (assumptions flagged); room geometry; the AR scenario's tap |
| `validation.json` | the validation ladder's results (synth/validate.py) |

`results/gt/synth_<name>.txt` (TUM, camera) is keyed by the **reported** frame stamps with the pose at the **true**
capture time, so timestamp association adds no speed x jitter error.

## Sensor model

Measured on the Pixel 9 recordings and taken from the Chromium source (claude/RESEARCH.md "Phone sensor model,
measured"): 100 Hz sensor samples (10 ms grid); Chrome's 60 Hz DeviceMotion timer reads the latest sample (intervals of
10 / 20 ms, 1 : 2) and stamps it with the tick time (+-5 ms against the sample time); rounding to 0.1 m/s^2 and 0.1 deg/s;
camera at 29.97 fps with 3.7 ms white stamp jitter; camera-IMU pipeline offset -88 ms.
**Assumed** (typical phone data sheet values; `meta.json` lists them; to be replaced by a still calibration recording):
accelerometer / gyro white noise (0.012 m/s^2, 0.035 deg/s per sample), biases (0.05 m/s^2, 0.11 deg/s per axis, plus
random walk), accelerometer scale factors (0.5 %), the sensor's internal 10 ms averaging, a 2 ms sample availability
latency, camera exposure 12 ms and rolling-shutter readout 20 ms, image noise.

## Rendering

`synth/render.py`: exact ray casting of textured rectangles (no mesh approximation), pinhole camera in the OpenCV
convention, mip-mapped texture sampling (no aliasing on oblique floors), Lambertian ceiling lights, rolling shutter
(each row at its own time), motion blur (sub-samples over the exposure, about one per pixel of image motion),
vignetting, auto exposure, shot + read noise, gamma, JPEG. Frame time = middle row, middle of the exposure.
Textures are drawn by code (noise, shapes, text glyphs, checkerboards): no photographs, free to publish.

## Commands

```
python synth/make_dataset.py --procs 4 --clean        # all 42 + 7 clean (H:/Research/datasets/synth, results/gt)
python synth/make_dataset.py --only slide4,orbit3     # selected sequences
python synth/validate.py H:/Research/datasets/synth/*  # validation ladder
python synth/imu_stats.py <recording dir>              # the real-phone statistics the model is based on
```

Viewer / AR page (dev server): `viewer.html` lists `synth/<name>` with its ground truth;
`ar.html?seq=synth/<name>` replays a sequence as the camera (also on the phone through `adb reverse`).
Replay batches: `PH=/mnt/h/Research/datasets/synth RECS="slide4 ..." replay_arms.sh ...`.
