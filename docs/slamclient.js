// Shared helpers for pages that run the WASM ORB-SLAM3 worker (viewer, AR).

// Request/response wrapper around slam.worker.js (one outstanding request at a time; worker logs go to onLog).
export function createSlamWorker(onLog = () => {}, onProgress = () => {}) {
  const worker = new Worker(new URL('./slam.worker.js', import.meta.url), { type: 'module' });
  let pending = null;
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === 'log') { if (!/^\s*$/.test(m.text)) onLog(m.text); return; }
    if (m.type === 'progress') { onProgress(m); return; }  // init: vocabulary download / build
    if (m.type === 'error') onLog('ERROR ' + m.text);
    if (pending) { const p = pending; pending = null; p(m); }
  };
  return {
    worker,
    request: (msg, transfer = []) => new Promise((res) => { pending = res; worker.postMessage(msg, transfer); }),
    terminate: () => worker.terminate(),
  };
}

// Pinhole intrinsics of a camera stream: the saved calibration (calib.html) if given, else from the horizontal FOV
// (along the sensor long side, portrait streams too) with the centre in the middle and no distortion.
export function intrinsics(w, h, hfovDeg, calib = null) {
  if (calib) return { fx: calib.fx, fy: calib.fy, cx: calib.cx, cy: calib.cy, k1: calib.k1, k2: calib.k2, p1: calib.p1, p2: calib.p2 };
  const f = (Math.max(w, h) / 2) / Math.tan(hfovDeg * Math.PI / 360);
  return { fx: f, fy: f, cx: w / 2, cy: h / 2, k1: 0, k2: 0, p1: 0, p2: 0 };
}

// Camera calibrations saved by calib.html, per stream size, in this browser (localStorage; may be unavailable).
export function loadCalibration(w, h) {
  try {
    const c = JSON.parse(localStorage.getItem(`calib:${w}x${h}`) || 'null');
    return c && c.w === w && c.h === h ? c : null;
  } catch { return null; }
}
export function saveCalibration(c) {
  try { localStorage.setItem(`calib:${c.w}x${c.h}`, JSON.stringify(c)); return true; } catch { return false; }
}
export function savedCalibrations() {
  const out = [];
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k && k.startsWith('calib:')) { const c = JSON.parse(localStorage.getItem(k)); if (c && c.w) out.push(c); }
    }
  } catch { /* storage blocked */ }
  return out;
}

// IMU block for ORB-SLAM3 mono-inertial on a phone (appended to cameraSettings). Body = the device's DeviceMotion axes
// (x right, y up, z out of the screen); the rear camera (x right, y down, z forward) is R_bc = diag(1, -1, -1) in
// portrait, translation ~1 cm (ignored). Noise: typical phone MEMS densities x ~5 (browser filtering / resampling).
export function phoneImuSettings(hz = 100) {
  return `
# IMU (phone, DeviceMotion): web/slamclient.js phoneImuSettings
IMU.T_b_c1: !!opencv-matrix
   rows: 4
   cols: 4
   dt: f
   data: [1.0, 0.0, 0.0, 0.0,
          0.0, -1.0, 0.0, 0.0,
          0.0, 0.0, -1.0, 0.0,
          0.0, 0.0, 0.0, 1.0]
IMU.InsertKFsWhenLost: 0
IMU.NoiseGyro: 6.0e-4
IMU.NoiseAcc: 1.0e-2
IMU.GyroWalk: 2.0e-5
IMU.AccWalk: 3.0e-3
IMU.Frequency: ${hz.toFixed(1)}
`;
}

// ORB-SLAM3 settings for a camera stream: calibrated (calib.html) or pinhole from the horizontal FOV, no distortion.
export function cameraSettings(w, h, hfovDeg, fps, nFeatures = 1000, calib = null) {
  const K = intrinsics(w, h, hfovDeg, calib);
  return `%YAML:1.0
File.version: "1.0"
Camera.type: "PinHole"
Camera1.fx: ${K.fx.toFixed(3)}
Camera1.fy: ${K.fy.toFixed(3)}
Camera1.cx: ${K.cx.toFixed(3)}
Camera1.cy: ${K.cy.toFixed(3)}
Camera1.k1: ${K.k1.toFixed(6)}
Camera1.k2: ${K.k2.toFixed(6)}
Camera1.p1: ${K.p1.toFixed(6)}
Camera1.p2: ${K.p2.toFixed(6)}
Camera.width: ${w}
Camera.height: ${h}
Camera.fps: ${Math.round(fps)}
Camera.RGB: 1
ORBextractor.nFeatures: ${nFeatures}
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
`;
}

// Vocabulary URL: the dev server (web/serve.py) serves /vocab/ORBvoc.bin; the static (GitHub Pages) build ships a
// gzipped copy next to the pages (decompressed by the worker).
const DEV_SERVER = location.pathname.startsWith('/web/');
export const VOCAB_URL = DEV_SERVER ? '/vocab/ORBvoc.bin' : new URL('./vocab/ORBvoc.bin.gz', location.href).href;

// Vocabulary for a descriptor: 'orb' -> ORB-SLAM3's ORBvoc; 'teblid' -> our TEBLID vocabulary (scripts: vocab_train,
// trained on KITTI + TUM-VIE images that are not used for evaluation).
export function vocabUrl(desc = 'orb') {
  if (desc !== 'teblid') return VOCAB_URL;
  return DEV_SERVER ? '/vocab/TEBLIDvoc_own_k10L6.bin' : new URL('./vocab/TEBLIDvoc_own_k10L6.bin.gz', location.href).href;
}

export const STATE = { '-1': 'not ready', 0: 'no images', 1: 'initialising', 2: 'tracking', 3: 'recently lost', 4: 'lost', 5: 'tracking (KLT)' };
