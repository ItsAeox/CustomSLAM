// Shared helpers for pages that run the WASM ORB-SLAM3 worker (viewer, AR).

// Request/response wrapper around slam.worker.js (one outstanding request at a time; worker logs go to onLog).
export function createSlamWorker(onLog = () => {}) {
  const worker = new Worker(new URL('./slam.worker.js', import.meta.url), { type: 'module' });
  let pending = null;
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === 'log') { if (!/^\s*$/.test(m.text)) onLog(m.text); return; }
    if (m.type === 'error') onLog('ERROR ' + m.text);
    if (pending) { const p = pending; pending = null; p(m); }
  };
  return {
    worker,
    request: (msg, transfer = []) => new Promise((res) => { pending = res; worker.postMessage(msg, transfer); }),
    terminate: () => worker.terminate(),
  };
}

// ORB-SLAM3 settings for an uncalibrated camera stream: pinhole from the horizontal FOV, no distortion.
export function cameraSettings(w, h, hfovDeg, fps, nFeatures = 1000) {
  const f = (Math.max(w, h) / 2) / Math.tan(hfovDeg * Math.PI / 360);  // FOV along the sensor long side (portrait streams too)
  return `%YAML:1.0
File.version: "1.0"
Camera.type: "PinHole"
Camera1.fx: ${f.toFixed(3)}
Camera1.fy: ${f.toFixed(3)}
Camera1.cx: ${(w / 2).toFixed(3)}
Camera1.cy: ${(h / 2).toFixed(3)}
Camera1.k1: 0.0
Camera1.k2: 0.0
Camera1.p1: 0.0
Camera1.p2: 0.0
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
export const VOCAB_URL = location.pathname.startsWith('/web/')
  ? '/vocab/ORBvoc.bin' : new URL('./vocab/ORBvoc.bin.gz', location.href).href;

export const STATE = { '-1': 'not ready', 0: 'no images', 1: 'initialising', 2: 'tracking', 3: 'recently lost', 4: 'lost', 5: 'tracking (KLT)' };
