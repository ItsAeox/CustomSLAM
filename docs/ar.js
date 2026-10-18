// Markerless AR in the browser: ORB-SLAM3 (WASM, worker) tracks the camera; the floor is found as a RANSAC plane over
// the local map points (constrained to gravity from DeviceMotion when available); tapping places a cactus on it.
// The background shows the frame the pose was computed for, so the overlay is in sync with the image (no lag).
//
// URL parameters: fov=<horizontal FOV in degrees> (default 69: Pixel 9 main camera, ~25 mm equivalent at 4:3), res=640x480, features=1000,
//                 seq=<dataset>/<name> (test mode: plays a dataset sequence in real time instead of the camera)

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { createSlamWorker, cameraSettings, intrinsics, loadCalibration, phoneImuSettings, savedCalibrations, STATE, vocabUrl } from './slamclient.js';
import { estimateImuCalib } from './imucalib.js';
import { estimateScale } from './imuscale.js';
import { createMiniMap } from './minimap.js';
import { dot, fitPlaneRansac, normalize, rayPlane } from './plane.js';
import { applyOptions, applyPreset, buildOptionsUI, describeOptions, matchPreset, OPTIONS, optionsFromParams, optionsToParams, PRESETS } from './options.js';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
if (params.get('fov')) $('fov').value = params.get('fov');
if (params.get('res')) $('res').value = params.get('res');
// algorithm / settings variant (web/options.js), also in the URL. AR defaults: the hybrid ORB+KLT tracker (2-3x faster;
// at honest real time on the five hardest sequences, 10 seeds, WASM: 22/50 bad vs 27/50 for stock, native 12 vs 12 —
// results/hard10_rt), relocalise 10 s before a new map (stock 3 s), and keep the map while the tree is placed.
const AR_DEFAULTS = { reloc: '10', tracker: 'klt' };
const opts = optionsFromParams(params, AR_DEFAULTS, 'ar');
function ensureImuOffsetValue(v) {  // a measured offset (imucalib.js) need not be one of the listed 20 ms steps
  const d = OPTIONS.find((x) => x.key === 'imudt');
  if (d && !d.values.some(([k]) => k === String(v))) d.values.push([String(v), `${v >= 0 ? '+' : ''}${v} (measured)`]);
}
ensureImuOffsetValue(opts.imudt);
// Metric scale prior (claude/RESEARCH.md "Metric scale"): monocular maps have no scale, but the camera height above the
// detected floor is measured in map units; a phone held at ~1.4 m gives metres per map unit. Shown in the log only.
const PHONE_HEIGHT_M = 1.4;
const metresPerUnit = () => imuScaleM ?? (floor && floor.height > 0 ? PHONE_HEIGHT_M / floor.height : null);
const scaleSource = () => (imuScaleM ? 'accelerometer scale' : `phone at ${PHONE_HEIGHT_M} m`);
// Presets (web/options.js PRESETS) set every option at once; changing a single option shows "custom"
const saveOptsToUrl = () => history.replaceState(null, '', `?${optionsToParams(opts, new URLSearchParams(location.search), AR_DEFAULTS)}`);
const presetKey = () => matchPreset(opts, AR_DEFAULTS, 'ar');
const syncPreset = () => {
  const k = presetKey(), p = PRESETS.find((x) => x.key === k);
  $('preset').value = k;
  $('presetNote').textContent = p ? p.note : 'Your own combination (see Algorithm options).';
};
const renderOpts = () => buildOptionsUI($('opts'), opts, () => { saveOptsToUrl(); syncPreset(); });
for (const p of PRESETS) $('preset').append(new Option(p.label, p.key));
$('preset').append(Object.assign(new Option('custom', 'custom'), { disabled: true }));
$('preset').onchange = () => {
  const p = PRESETS.find((x) => x.key === $('preset').value);
  if (!p) return;
  applyPreset(opts, p, AR_DEFAULTS, 'ar');
  renderOpts(); saveOptsToUrl(); syncPreset();
};
renderOpts(); syncPreset();

const bg = $('bg'), bgx = bg.getContext('2d');
const renderer = new THREE.WebGLRenderer({ canvas: $('gl'), alpha: true, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
const scene = new THREE.Scene();
scene.add(new THREE.HemisphereLight(0xffffff, 0x445544, 2.2));
const sun = new THREE.DirectionalLight(0xffffff, 1.5);
scene.add(sun);
const cam3 = new THREE.PerspectiveCamera(50, 4 / 3, 0.005, 1000);
cam3.matrixAutoUpdate = false;
cam3.matrixWorldAutoUpdate = false;
const FLIP = new THREE.Matrix4().makeScale(1, -1, -1);  // OpenCV camera (x right, y down, z forward) -> three.js camera

let W = 640, H = 480;               // processed image size (pixels)
let K = { fx: 500, fy: 500, cx: 320, cy: 240 };  // pinhole intrinsics (calib.html calibration, else from the FOV)
let calib = null;                   // saved calibration in use, or null
let rect = { x: 0, y: 0, w: 1, h: 1, s: 1 };
let twc = null, state = 0, stats = [0, 0, 0, 0];
let localPts = [];                  // local map points [[x,y,z]]
let kps = null;
let floor = null, floorMesh = null;  // plane {n (up), d, height}
let cactusProto = null, cactus = null;  // the placed object (now a tree: assets/tree.glb)
let objBase = 1;                     // placement size (map units); the Size slider scales it
const sizeFactor = () => 2 ** Number($('size').value);
let downSum = [0, 0, 0], downN = 0;  // gravity direction in the SLAM world frame (sum of unit vectors)
let accLP = null, motionEvents = 0;
let imuN = 0, imuT0 = 0, imuT1 = 0;  // DeviceMotion events with a rotation rate: count, first/last timeStamp (ms)
const imuRateHz = () => (imuN >= 10 && imuT1 > imuT0 ? (imuN - 1) * 1000 / (imuT1 - imuT0) : 60);  // Chrome: ~60 Hz
let camFrames = 0, camDropped = 0, frameClock = 'dataset';
let camLast = [0, 0];  // camera counters at the previous status line (per-interval drop rate)
let camMode = 'bitmap';               // 'webcodecs': the worker reads VideoFrames itself (see slam.worker.js)
let camStream = null, camTrack = null; // MediaStreamTrackProcessor.readable (on a clone of the camera track)
let stopRvfc = () => {};
const decodeTimes = [];               // per frame: gray image preparation in the worker (ms)  // live camera: frames presented / replaced before tracking
const gyroHist = [];                 // [time ms, |rotation rate| deg/s] of the last second (logged on tracking loss)
let inertial = opts.imu === 'on';  // ORB-SLAM3 mono-inertial (experimental); re-read from the options at start()
let rec = false;                   // record gyro + poses for the time-offset estimate (option 'rec'); set at start()
const recPoses = [], recGyro = [];  // with rec: [t, qx, qy, qz, qw, maps, x, y, z] / [t, wx, wy, wz] (last 90 s) for imucalib.js
const recAcc = [];                  // with rec: [t, ax, ay, az] (specific force, device axes; time offset applied) for imuscale.js
let imuScaleM = null;               // metres per map unit from the accelerometer (imuscale.js), when measured
let recCalibAt = 0;
let gyroOn = false;               // gyro rotation prior (option 'gyro', visual-only; or gyro init); set at start()
let holdOn = false;               // option 'hold': show the tree through brief losses with the gyro-rotated last pose
let replay = false;               // a dataset or a phone recording (record.html) is the camera; its IMU, not the phone's
let qDev = [0, 0, 0, 1], qDevT = null;  // integrated device attitude from DeviceMotion rotation rates (x, y, z, w)
let holdRef = null, holdTwc = null;     // last tracked pose with its attitude and time; the held pose while lost
const HOLD_MS = 400;              // eval/deadreckon_sim.py: rotation-only hold is good for blips < ~0.3-0.5 s
const absClock = () => inertial || rec || gyroOn;  // frame times on the IMU's clock (page timeOrigin + ms) instead of camera time
let imuPending = [];
let imuFlags = [0, 0];               // last logged IMU initialisation flags (map stats [4], [5])                 // live IMU samples not yet sent: [t abs s, ax, ay, az (m/s^2), gx, gy, gz (rad/s)] flat
let lastMaps = 0, lastState = 0;
let replayTaps = [], replayT = -1;               // recording taps still to replay: { t (s, sequence clock), u, v (image px) }
let stash = null;          // anchors of a map that went inactive (tracking lost -> new map); restored on map merge
let resetRequested = false;  // Reset button: sent to the worker between frames (one request at a time)
let saveRequested = false;   // Save room button: also between frames
// Instant placement (ARCore-style): a tap before tracking is kept and shown as a marker; once the map and the floor
// exist the tree goes there. The tap direction is carried across the wait by the gyro attitude (qDev).
let pendingTap = null;       // { u, v (image px at the tap), q0 (device attitude then), t (ms) }
let trackingSince = 0;       // when tracking (re)started, ms
let pendingRoom = null;      // a loaded room's tree and floor, restored once the tree model is loaded
let mapWait = -1;            // room map: 1 = loaded, not relocalised into yet; 0 = relocalised; -1 = no room loaded
const tracking = () => state === 2 || state === 5;  // 5: tracked by a KLT frame

function layout() {  // cover-fit the processed image into the window; the GL canvas covers the same rectangle
  const s = Math.max(innerWidth / W, innerHeight / H);
  rect = { s, w: W * s, h: H * s, x: (innerWidth - W * s) / 2, y: (innerHeight - H * s) / 2 };
  for (const c of [bg, $('gl')]) Object.assign(c.style, { left: `${rect.x}px`, top: `${rect.y}px`, width: `${rect.w}px`, height: `${rect.h}px` });
  bg.width = W; bg.height = H;
  renderer.setSize(rect.w, rect.h, false);
  // projection from the intrinsics (principal point off-centre, fx != fy), OpenCV pixels -> GL clip space
  const n = cam3.near, f = cam3.far;
  cam3.projectionMatrix.set(
    2 * K.fx / W, 0, 1 - 2 * K.cx / W, 0,
    0, 2 * K.fy / H, 2 * K.cy / H - 1, 0,
    0, 0, -(f + n) / (f - n), -2 * f * n / (f - n),
    0, 0, -1, 0);
  cam3.projectionMatrixInverse.copy(cam3.projectionMatrix).invert();
}
addEventListener('resize', layout);

// ---------------------------------------------------------------- gravity (DeviceMotion, any rate is fine)
// DeviceMotionEvent.rotationRate -> device-axis rates (x, y, z; rad/s). The W3C spec says alpha = z, beta = x, gamma = y,
// but Chrome on Android reports alpha = x, beta = y, gamma = z: on the Pixel 9 (2026-10-18) the Generic Sensor Gyroscope
// x, y, z equal rotationRate alpha, beta, gamma sample for sample, and the camera rotation fits only that order. Until
// then every phone gyro feature had its axes permuted. ?rr=spec|chrome overrides the choice by browser.
const RR_ORDER = new URLSearchParams(location.search).get('rr') ||
  (/Chrome\//.test(navigator.userAgent) && !/Firefox|FxiOS/.test(navigator.userAgent) ? 'chrome' : 'spec');
function rateXYZ(r) {
  const k = Math.PI / 180;
  return RR_ORDER === 'chrome' ? [r.alpha * k, r.beta * k, r.gamma * k] : [r.beta * k, r.gamma * k, r.alpha * k];
}

function onMotion(e) {
  const a = e.accelerationIncludingGravity;
  if (!a || a.x == null) return;
  const v = [a.x, a.y, a.z];
  accLP = accLP ? accLP.map((x, i) => 0.7 * x + 0.3 * v[i]) : v;
  motionEvents++;
  const r = e.rotationRate;
  if ((inertial || gyroOn) && !replay && r && r.alpha != null && a.x != null) {  // device axes = IMU body; rotationRate alpha/beta/gamma = z/x/y (deg/s)
    const d = Math.PI / 180;
    // IMU time = reported time - the measured offset (option imudt; reported = true + offset, see estimate_time_offset.py)
    imuPending.push((performance.timeOrigin + e.timeStamp) / 1000 - Number(opts.imudt) / 1000, a.x, a.y, a.z, ...rateXYZ(r));
  }
  if (rec && r && r.alpha != null) {  // diagnostic: raw gyro (rad/s, device axes) on the frames' absolute clock
    const d = Math.PI / 180;
    const w = rateXYZ(r);
    tele(`imu ${((performance.timeOrigin + e.timeStamp) / 1000).toFixed(4)} ${w[0].toFixed(5)} ${w[1].toFixed(5)} ${w[2].toFixed(5)}`);
    recGyro.push([(performance.timeOrigin + e.timeStamp) / 1000, ...w]);
    while (recGyro.length && recGyro[0][0] < recGyro[recGyro.length - 1][0] - 90) recGyro.shift();
    recAcc.push([(performance.timeOrigin + e.timeStamp) / 1000 - Number(opts.imudt) / 1000, a.x, a.y, a.z]);
    while (recAcc.length && recAcc[0][0] < recAcc[recAcc.length - 1][0] - 90) recAcc.shift();
  }
  if ((holdOn || pendingTap) && r && r.alpha != null) {  // attitude for the hold / a pending tap: q <- q * exp(w dt), device axes
    if (qDevT != null) {
      const dt = Math.min((e.timeStamp - qDevT) / 1000, 0.1), d = Math.PI / 180;
      const w = rateXYZ(r), th = Math.hypot(...w) * dt;
      if (th > 0) {
        const s = Math.sin(th / 2) / (th / dt), dq = [w[0] * s, w[1] * s, w[2] * s, Math.cos(th / 2)];
        const [x, y, z, ww] = qDev;
        qDev = [ww * dq[0] + x * dq[3] + y * dq[2] - z * dq[1], ww * dq[1] - x * dq[2] + y * dq[3] + z * dq[0],
                ww * dq[2] + x * dq[1] - y * dq[0] + z * dq[3], ww * dq[3] - x * dq[0] - y * dq[1] - z * dq[2]];
      }
    }
    qDevT = e.timeStamp;
  }
  if (r && r.alpha != null) {
    if (!imuN) imuT0 = e.timeStamp;
    imuN++; imuT1 = e.timeStamp;
    const now = performance.now();
    gyroHist.push([now, Math.hypot(r.alpha, r.beta, r.gamma)]);
    while (gyroHist.length && gyroHist[0][0] < now - 1000) gyroHist.shift();
  }
}
const gyroPeak = (ms = 500) => {  // peak rotation speed over the last `ms` (deg/s), or null without a gyroscope
  if (!gyroHist.length) return null;
  const t = performance.now() - ms;
  return gyroHist.reduce((m, [ti, w]) => (ti >= t ? Math.max(m, w) : m), 0);
};
function gravityCam() {  // unit "down" in the camera frame (x right, y down, z forward) for the rear camera
  if (!accLP) return null;
  const th = ((screen.orientation && screen.orientation.angle) ?? window.orientation ?? 0) * Math.PI / 180;
  const c = Math.cos(th), s = Math.sin(th);
  const x = accLP[0] * c - accLP[1] * s, y = accLP[0] * s + accLP[1] * c, z = accLP[2];  // device -> screen axes
  // accelerationIncludingGravity points "up"; screen axes (x right, y up, z to viewer) -> camera (x, -y, -z)
  return normalize([-x, y, z]);
}

// ---------------------------------------------------------------- pose and anchors
function rotOf(m) { return [m[0], m[4], m[8], m[1], m[5], m[9], m[2], m[6], m[10]]; }  // column-major 4x4 -> row-major R
const mulR = (R, v) => [R[0] * v[0] + R[1] * v[1] + R[2] * v[2], R[3] * v[0] + R[4] * v[1] + R[5] * v[2], R[6] * v[0] + R[7] * v[1] + R[8] * v[2]];

// Option 'hold': the last tracked camera pose rotated by the device rotation since (camera centre kept). Body (device)
// rotation dq = q_ref^-1 q_now; camera = R_cb dq R_bc with R_bc = diag(1, -1, -1) (rear camera, portrait: as
// slamclient.js phoneImuSettings); Twc_held = Twc_ref * [R_c 0; 0 1].
function heldPose() {
  const [ax, ay, az, aw] = holdRef.q, [bx, by, bz, bw] = qDev;
  const x = aw * bx - ax * bw - ay * bz + az * by, y = aw * by + ax * bz - ay * bw - az * bx;  // conj(a) * b
  const z = aw * bz - ax * by + ay * bx - az * bw, w = aw * bw + ax * bx + ay * by + az * bz;
  const dq = new THREE.Matrix4().makeRotationFromQuaternion(new THREE.Quaternion(x, y, z, w));
  const F = new THREE.Matrix4().makeScale(1, -1, -1);
  const Rc = F.clone().multiply(dq).multiply(F);
  return new THREE.Matrix4().fromArray(holdRef.twc).multiply(Rc).toArray();
}

function setCameraPose(m) {
  cam3.matrixWorld.fromArray(m).multiply(FLIP);
  cam3.matrixWorldInverse.copy(cam3.matrixWorld).invert();
  sun.position.set(m[12], m[13] - 2, m[14]);  // light from "above" the camera (world -y is roughly up after init)
}

function upDir() { return downN ? normalize(downSum.map((x) => -x)) : null; }

function resetAnchors(reason) {
  tele(`anchors reset: ${reason || 'manual'}`);
  floor = null;
  if (floorMesh) { scene.remove(floorMesh); floorMesh = null; }
  if (cactus) { scene.remove(cactus); cactus = null; }
  downSum = [0, 0, 0]; downN = 0;
  if (reason) hint(reason, 3000);
}

// ORB-SLAM3 started a new map after losing tracking: the cactus/floor live in the old map's coordinates, so hide them
// (not delete). If the user returns and ORB-SLAM3 merges the maps, the merged map uses the old map's coordinates
// (LoopClosing::MergeLocal corrects the new map into it and makes it active again), so the anchors become valid again.
function stashAnchors() {
  if (!cactus && !floor) return;
  if (stash) { if (stash.cactus) scene.remove(stash.cactus); if (stash.floorMesh) scene.remove(stash.floorMesh); }
  stash = { cactus, floor, floorMesh, downSum, downN };
  if (cactus) cactus.visible = false;
  if (floorMesh) floorMesh.visible = false;
  cactus = null; floor = null; floorMesh = null; downSum = [0, 0, 0]; downN = 0;
  tele('anchors stashed (new map after tracking loss)');
  hint('Tracking lost: new map started. Go back to where the tree was to recover it.', 4000);
}
function restoreAnchors() {
  if (!stash) return;
  if (cactus) scene.remove(cactus);
  if (floorMesh) scene.remove(floorMesh);
  ({ cactus, floor, floorMesh, downSum, downN } = stash);
  stash = null;
  if (cactus) cactus.visible = true;
  tele('anchors restored (maps merged)');
  hint('Map recovered: tree restored', 3000);
}

function detectFloor() {
  const up = upDir();
  if (!up || !twc || localPts.length < 30) return;
  const c = [twc[12], twc[13], twc[14]];
  const dists = localPts.map((p) => Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2])).sort((a, b) => a - b);
  const scale = dists[Math.floor(dists.length / 2)];
  // only points below the camera can be floor
  const below = localPts.filter((p) => dot([p[0] - c[0], p[1] - c[1], p[2] - c[2]], up) < -0.2 * scale);
  const pl = fitPlaneRansac(below, { thresh: 0.02 * scale, prior: up, maxAngleDeg: 15, minInliers: 20 });
  if (!pl) return;
  if (dot(pl.n, up) < 0) { pl.n = pl.n.map((x) => -x); pl.d = -pl.d; }
  const height = dot(pl.n, c) + pl.d;  // camera height above the plane
  if (height <= 0) return;
  if (floor && pl.inliers.length < 0.8 * floor.inliers) return;  // keep a better earlier estimate
  if (!floor || Math.abs(height - floor.height) > 0.2 * floor.height) {  // log new / clearly changed floor estimates
    const tilt = Math.acos(Math.min(1, Math.abs(dot(pl.n, up)))) * 180 / Math.PI;
    tele(`floor ${floor ? 'updated' : 'found'}: camera height ${height.toFixed(3)} (map units), ${pl.inliers.length} of ${below.length} points below the camera, tilt to gravity ${tilt.toFixed(1)} deg` +
         ` | scale prior: 1 map unit ~ ${(PHONE_HEIGHT_M / height).toFixed(3)} m if the phone is ${PHONE_HEIGHT_M} m above the floor`);
  }
  floor = { n: pl.n, d: pl.d, height, inliers: pl.inliers.length, centroid: pl.centroid };
  placeFloorMesh();
}
function placeFloorMesh() {
  if (!floorMesh) {
    floorMesh = new THREE.GridHelper(1, 10, 0x22c55e, 0x22c55e);
    floorMesh.material.transparent = true; floorMesh.material.opacity = 0.35;
    scene.add(floorMesh);
  }
  // the grid is fixed in the world: centred on the floor points that support the plane (not under the camera)
  const k = dot(floor.n, floor.centroid) + floor.d;
  floorMesh.position.set(floor.centroid[0] - floor.n[0] * k, floor.centroid[1] - floor.n[1] * k, floor.centroid[2] - floor.n[2] * k);
  floorMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(...floor.n));
  floorMesh.scale.setScalar(floor.height * 2);
}

// ---------------------------------------------------------------- panels and the 3D map view
let mini = null, miniFloor = null, miniTree = null, miniMaps = -1;
function setupPanels() {
  const toggle = (btn, els, labels) => {
    btn.onclick = () => {
      const open = els[0].hidden;
      for (const e of els) e.hidden = !open;
      btn.setAttribute('aria-expanded', String(open));
      if (labels) btn.textContent = open ? labels[1] : labels[0];
    };
  };
  toggle($('menuBtn'), [$('bar')], ['☰', '✕']);
  toggle($('infoBtn'), [$('hud'), $('usage')]);
  toggle($('miniTab'), [$('miniOpts')], ['3D view ▴', '3D view ▾']);
  mini = createMiniMap($('mini'));
  const ticks = { mvPoints: 'points', mvKF: 'keyframes', mvGraph: 'graph', mvTraj: 'trajectory', mvGrid: 'grid', mvTree: 'tree', mvFollow: 'follow' };
  for (const [id, key] of Object.entries(ticks)) {
    mini.show[key] = $(id).checked;
    $(id).onchange = () => { mini.show[key] = $(id).checked; if (key === 'follow' && $(id).checked) mini.resetView(); };
  }
  $('mvShow').onchange = () => { $('mini').hidden = !$('mvShow').checked; mini.setVisible($('mvShow').checked); };
}
function updateMini(r) {  // per frame: map (every 10th frame), camera, floor, tree, trajectory reset on a map change
  if (!mini) return;
  if (r.viz && r.viz.mps) mini.updateMap(r.viz);
  if (tracking() && twc) mini.updateCamera(twc);
  if (floor !== miniFloor) { miniFloor = floor; mini.setFloor(floor); }
  if (cactus !== miniTree) { miniTree = cactus; mini.setTree(cactus ? cactus.position.toArray() : null, objBase * sizeFactor()); }
  if (stats[3] !== miniMaps) { miniMaps = stats[3]; mini.clearTrajectory(); }
}

// ---------------------------------------------------------------- room map save and reload
// One saved room per browser (IndexedDB): the SLAM map (slam/orbslam3/src/WebMapIO.cc format) with the tree and floor
// in its coordinates. Loading it at the start makes ORB-SLAM3 relocalise into the saved map, so the tree comes back
// where it was. The map only fits the same camera (size, intrinsics) and vocabulary (checked here and in WebMapIO.cc).
const ROOM_DB = 'webvio-rooms', ROOM_STORE = 'rooms', ROOM_KEY = 'room';
function roomDb() {
  return new Promise((res, rej) => {
    const rq = indexedDB.open(ROOM_DB, 1);
    rq.onupgradeneeded = () => rq.result.createObjectStore(ROOM_STORE);
    rq.onsuccess = () => res(rq.result);
    rq.onerror = () => rej(rq.error);
  });
}
async function roomGet() {
  try {
    const db = await roomDb();
    return await new Promise((res, rej) => {
      const rq = db.transaction(ROOM_STORE).objectStore(ROOM_STORE).get(ROOM_KEY);
      rq.onsuccess = () => res(rq.result || null);
      rq.onerror = () => rej(rq.error);
    });
  } catch { return null; }  // storage blocked (private window)
}
// gzip in the browser (CompressionStream): the map's floats compress little, but its integers and empty slots do
const gzip = (buf, how) => new Response(new Blob([buf]).stream().pipeThrough(how === 'c' ? new CompressionStream('gzip') : new DecompressionStream('gzip'))).arrayBuffer();
async function roomPut(room) {
  const db = await roomDb();
  return new Promise((res, rej) => {
    const tx = db.transaction(ROOM_STORE, 'readwrite');
    tx.objectStore(ROOM_STORE).put(room, ROOM_KEY);
    tx.oncomplete = () => res(true);
    tx.onerror = () => rej(tx.error);
  });
}
function restoreRoom(room) {  // the saved tree and floor (hidden until tracking relocalises into the loaded map)
  if (room.floor) { floor = room.floor; placeFloorMesh(); floorMesh.visible = false; }
  downSum = room.downSum || [0, 0, 0]; downN = room.downN || 0;
  if (room.tree) {
    cactus = cactusProto.clone();
    cactus.position.fromArray(room.tree.p);
    cactus.quaternion.fromArray(room.tree.q);
    objBase = room.tree.base;
    $('size').value = room.tree.size;
    $('sizeVal').textContent = `${sizeFactor().toFixed(2)}×`;
    cactus.scale.setScalar(objBase * sizeFactor());
    cactus.visible = false;
    scene.add(cactus);
    $('sizeRow').style.visibility = 'visible';
  }
  tele(`room restored: tree ${room.tree ? `at ${room.tree.p.map((x) => x.toFixed(4)).join(' ')}` : 'none'}, floor ${room.floor ? 'yes' : 'no'}`);
}

function quatMulA(a, b) {
  return [a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1], a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
          a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3], a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]];
}
function quatRot(q, v) {  // rotate v by unit quaternion q (x, y, z, w)
  const r = quatMulA(quatMulA(q, [v[0], v[1], v[2], 0]), [-q[0], -q[1], -q[2], q[3]]);
  return [r[0], r[1], r[2]];
}
// image pixel where the pending tap's direction is now (gyro rotation since the tap; rear camera = device diag(1,-1,-1))
function pendingPixel() {
  const d0 = [(pendingTap.u - K.cx) / K.fx, -(pendingTap.v - K.cy) / K.fy, -1];  // camera ray in device axes
  const qRel = quatMulA([-pendingTap.q0[0], -pendingTap.q0[1], -pendingTap.q0[2], pendingTap.q0[3]], qDev);
  const dn = quatRot([-qRel[0], -qRel[1], -qRel[2], qRel[3]], d0);  // the same world direction in today's device axes
  const c = [dn[0], -dn[1], -dn[2]];                                // back to camera axes
  if (c[2] <= 0.05) return null;                                    // behind the camera now
  return [K.fx * c[0] / c[2] + K.cx, K.fy * c[1] / c[2] + K.cy];
}
function showGhost() {  // marker at the pending tap (follows the gyro until the tree can be placed)
  const g = $('ghost');
  const px = pendingTap && pendingPixel();
  if (!px || px[0] < 0 || px[1] < 0 || px[0] >= W || px[1] >= H) { g.hidden = true; return; }
  g.hidden = false;
  g.style.left = `${rect.x + px[0] * rect.s}px`;
  g.style.top = `${rect.y + px[1] * rect.s}px`;
}
function resolvePendingTap() {  // per frame: place the pending tree once tracking + a floor (or 3 s of tracking) exist
  if (!pendingTap) return;
  if (performance.now() - pendingTap.t > 30000) {
    tele('instant placement: gave up after 30 s without a map');
    pendingTap = null; showGhost(); hint('Could not place the tree yet: move the phone slowly sideways, then tap again.', 3000);
    return;
  }
  showGhost();
  if (!tracking() || !(floor || performance.now() - trackingSince > 3000)) return;
  const px = pendingPixel();
  if (!px || px[0] < 0 || px[1] < 0 || px[0] >= W || px[1] >= H) return;  // the spot is out of view: wait
  const tap = pendingTap;
  pendingTap = null;
  showGhost();
  tele(`instant placement: tap from ${((performance.now() - tap.t) / 1000).toFixed(1)} s ago at ${tap.u.toFixed(0)}, ${tap.v.toFixed(0)} px -> now ${px[0].toFixed(0)}, ${px[1].toFixed(0)} px`);
  placeCactus(rect.x + px[0] * rect.s, rect.y + px[1] * rect.s);
}

function placeCactus(clientX, clientY) {
  if (!twc || !tracking()) {  // no map yet: remember the tap (instant placement)
    if (cactus || !rect) return;
    pendingTap = { u: (clientX - rect.x) / rect.s, v: (clientY - rect.y) / rect.s, q0: qDev.slice(), t: performance.now() };
    qDevT = null;  // start integrating from this tap
    tele(`instant placement: tap kept at ${pendingTap.u.toFixed(0)}, ${pendingTap.v.toFixed(0)} px (not tracking yet)`);
    hint('The tree will go here once tracking starts. Move the phone slowly sideways.', 4000);
    showGhost();
    return;
  }
  if (inertial && !(stats[4] && stats[5])) {  // the IMU initialisation rotates and rescales the map: place afterwards
    hint(stats[4] ? 'IMU almost ready (refining). Keep moving a few more seconds.' : 'IMU initialising: move the phone around (walk, turn).', 2500);
    return;
  }
  const u = (clientX - rect.x) / rect.s, v = (clientY - rect.y) / rect.s;  // image pixel
  const dir = normalize(mulR(rotOf(twc), [(u - K.cx) / K.fx, (v - K.cy) / K.fy, 1]));  // (lens distortion ignored)
  const o = [twc[12], twc[13], twc[14]];
  let plane = floor, size;
  if (plane) {
    size = 0.6 * plane.height;  // tree ~60% of the camera height above the floor (then the Size slider)
  } else {  // no gravity: fit a plane to the local map points that project near the tap
    const R = rotOf(twc);
    const near = [], depths = [];
    for (const p of localPts) {
      const d = [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
      const pc = [R[0] * d[0] + R[3] * d[1] + R[6] * d[2], R[1] * d[0] + R[4] * d[1] + R[7] * d[2],
                  R[2] * d[0] + R[5] * d[1] + R[8] * d[2]];  // camera frame: R^T (p - o)
      if (pc[2] <= 0) continue;
      const pu = K.fx * pc[0] / pc[2] + K.cx, pv = K.fy * pc[1] / pc[2] + K.cy;
      if (Math.hypot(pu - u, pv - v) < 0.15 * W) { near.push(p); depths.push(pc[2]); }
    }
    if (near.length < 12) { hint('Not enough map points there. Move a bit and tap again.', 2500); tele(`tap rejected: ${near.length} map points near the tap`); return; }
    depths.sort((a, b) => a - b);
    const zMed = depths[depths.length >> 1];
    const pl = fitPlaneRansac(near, { thresh: 0.03 * zMed, minInliers: 10 });
    if (!pl) { hint('No flat surface found there. Tap a flat, textured area.', 2500); tele(`tap rejected: no plane in ${near.length} points`); return; }
    if (dot(pl.n, [o[0] - pl.centroid[0], o[1] - pl.centroid[1], o[2] - pl.centroid[2]]) < 0) { pl.n = pl.n.map((x) => -x); pl.d = -pl.d; }
    plane = { n: pl.n, d: pl.d };
    size = 0.35 * zMed;
    const hitTest = rayPlane(o, dir, plane);
    const depth = hitTest ? dot([hitTest[0] - o[0], hitTest[1] - o[1], hitTest[2] - o[2]], dir) : 0;
    if (!hitTest || depth < 0.4 * zMed || depth > 2.5 * zMed) {
      hint('That surface is not usable. Tap a flat, textured area.', 2500);
      tele(`tap rejected: plane hit at depth ${depth.toFixed(2)} vs median point depth ${zMed.toFixed(2)}`);
      return;
    }
  }
  const hit = rayPlane(o, dir, plane);
  if (!hit) { hint('Tap on the floor (below the horizon).', 2500); tele('tap rejected: the ray misses the floor plane'); return; }
  if (!cactusProto) return;
  if (cactus) scene.remove(cactus);
  cactus = cactusProto.clone();
  cactus.position.set(...hit);
  cactus.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(...plane.n));
  objBase = size;
  cactus.scale.setScalar(size * sizeFactor());  // cactusProto is normalised to height 1, base at the origin
  scene.add(cactus);
  $('sizeRow').style.visibility = 'visible';
  tele(`tree placed: ${floor ? 'floor plane' : 'tap plane'}, distance ${Math.hypot(hit[0] - o[0], hit[1] - o[1], hit[2] - o[2]).toFixed(2)}, size ${size.toFixed(3)} x ${sizeFactor().toFixed(2)} (map units)` +
       ` | at ${hit.map((x) => x.toFixed(4)).join(' ')} | camera ${o.map((x) => x.toFixed(4)).join(' ')}${floor ? ` | floor height ${floor.height.toFixed(4)}` : ''} | map ${stats[3]}` +
       (metresPerUnit() ? ` | tree ~ ${(size * sizeFactor() * metresPerUnit()).toFixed(2)} m tall (${scaleSource()})` : ''));
  hint('');
}

// ---------------------------------------------------------------- drawing
let hintText = '', hintHold = 0;
function hint(s, holdMs = 0) {  // holdMs: keep this message instead of the per-frame status for a while
  if (!holdMs && performance.now() < hintHold) return;
  if (holdMs) hintHold = performance.now() + holdMs;
  hintText = s; $('hint').textContent = s; $('hint').hidden = !s;
}

function drawFrame(bmp) {
  bgx.drawImage(bmp, 0, 0);
  bmp.close();
  if (kps && tracking()) {
    bgx.fillStyle = '#22c55e';
    for (let i = 0; i < kps.length; i += 3) if (kps[i + 2] === 1) bgx.fillRect(kps[i] - 1.5, kps[i + 1] - 1.5, 3, 3);
  }
  if (holdTwc) setCameraPose(holdTwc);
  else if (twc) setCameraPose(twc);
  if (floorMesh) floorMesh.visible = !cactus && tracking();
  if (cactus) cactus.visible = tracking() || !!holdTwc;  // hidden while lost (stale pose), unless held by the gyro
  renderer.render(scene, cam3);
}

// ---------------------------------------------------------------- main loop
let resChoice = '';  // the start screen's resolution choice (key of the stored IMU calibration)
async function start() {
  $('go').disabled = true;
  resChoice = $('res').value;
  inertial = opts.imu === 'on';  // the options may have changed since the page loaded
  rec = opts.rec === 'on';
  replay = !!params.get('seq');
  gyroOn = (opts.gyro === 'on' || opts.init === 'gyro') && !inertial;  // dataset / recording: its own IMU (seq branch)
  holdOn = opts.hold === 'gyro' && (!params.get('seq') || params.get('holdtest') === '1');  // holdtest: run the code on a dataset (no gyro: held = last pose)
  try {
    if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
      await DeviceMotionEvent.requestPermission().catch(() => {});  // iOS
    }
    addEventListener('devicemotion', onMotion);
    const seq = params.get('seq');  // test mode: a dataset sequence played as the "camera" (desktop testing)
    let nextBitmap, settingsText = null, settingsUrl = null, clahe = false, fps = 30;
    const hfov = Number($('fov').value);
    if (seq) {
      const [dataset, name] = seq.split('/');
      const info = (await (await fetch('/api/sequences')).json()).find((s) => s.dataset === dataset && s.name === name);
      const man = await (await fetch(`/api/sequence?dataset=${dataset}&name=${encodeURIComponent(name)}`)).json();
      // gravity from the dataset IMU (EuRoC / TUM-VI): the accelerometer averaged over +-0.25 s around the frame,
      // rotated into the camera frame (R_cb). At rest it is the reaction to gravity, so "down" = -a. noimu=1: none.
      const imu = params.get('noimu') ? null : await fetch(`/api/imu?dataset=${dataset}&name=${encodeURIComponent(name)}`)
        .then((r) => (r.ok ? r.json() : null)).catch(() => null);
      const imuRaw = (inertial || gyroOn) ? await fetch(`/api/imu?dataset=${dataset}&name=${encodeURIComponent(name)}&raw=1`)
        .then((r) => (r.ok ? r.json() : null)).catch(() => null) : null;  // gyro presets: the gyro prior needs IMU.T_b_c1 in the settings (recordings have it)
      // recordings (record.html): the taps, replayed at their time and image position (same tree placement for every preset)
      if (dataset === 'phone') {
        const txt = await fetch(`/datasets/phone/${encodeURIComponent(name)}/touches.csv`).then((r) => (r.ok ? r.text() : '')).catch(() => '');
        replayTaps = txt.split(/\r?\n/).filter((l) => l && !l.startsWith('#')).map((l) => {
          const [t, u, v] = l.split(','); return { t: Number(BigInt(t) - BigInt(Math.round(man.tBase * 1e9))) * 1e-9, u: Number(u), v: Number(v) };
        });
        tele(`recording ${name}: ${man.frames.length} frames, ${replayTaps.length} taps to replay`);
      }
      let kRaw = 0;
      const imuUpTo = (t) => {  // raw dataset IMU samples up to t (inertial test mode)
        const out = [];
        while (imuRaw && kRaw < imuRaw.t.length && imuRaw.t[kRaw] <= t) {
          const a = imuRaw.acc[kRaw], g = imuRaw.gyro[kRaw];
          out.push(imuRaw.t[kRaw], a[0], a[1], a[2], g[0], g[1], g[2]); kRaw++;
        }
        return out;
      };
      let ki = 0;
      const imuDown = (t) => {
        if (!imu) return null;
        while (ki + 1 < imu.t.length && imu.t[ki] < t - 0.25) ki++;
        const a = [0, 0, 0];
        let n = 0;
        for (let k = ki; k < imu.t.length && imu.t[k] <= t + 0.25; k++) { a[0] += imu.acc[k][0]; a[1] += imu.acc[k][1]; a[2] += imu.acc[k][2]; n++; }
        if (!n) return null;
        const R = imu.R_cb;
        return normalize([0, 1, 2].map((i) => -(R[i][0] * a[0] + R[i][1] * a[1] + R[i][2] * a[2])));
      };
      settingsUrl = inertial ? info.settings.replace('_mono.yaml', '_mono_inertial.yaml') : info.settings;  // ORB-SLAM3's own
      clahe = info.clahe;
      const yaml = await (await fetch(settingsUrl)).text();
      settingsText = yaml;
      const num = (key) => Number((new RegExp(`Camera1\\.${key}:\\s*([\\d.eE+-]+)`).exec(yaml) || [])[1] || 0);
      const fx = num('fx');
      let i = 0;
      // cover=a-b: frames from a to b seconds (sequence time) are black, as if the camera were covered (loss test)
      const cover = /^([\d.]+)-([\d.]+)$/.test(params.get('cover') || '') ? params.get('cover').split('-').map(Number) : null;
      let blank = null, seqEnded = false;
      const t0 = performance.now();
      nextBitmap = async () => {  // real-time playback; frames that are already late are skipped
        const now = (performance.now() - t0) / 1000;
        if (i >= man.frames.length) {  // end of the sequence: keep showing the last frame (was: undefined frame, TypeError)
          i = man.frames.length - 1;
          if (!seqEnded) { seqEnded = true; tele(`dataset sequence ended (${man.frames.length} frames)`); }
          hint('End of the dataset sequence.', 60000);
          await new Promise((res) => setTimeout(res, 100));  // do not spin on the last frame
        }
        while (i + 1 < man.frames.length && man.frames[i + 1].t <= now) i++;
        if (man.frames[i].t > now) await new Promise((res) => setTimeout(res, (man.frames[i].t - now) * 1000));
        const f = man.frames[Math.min(i++, man.frames.length - 1)];
        if (cover && f.t - man.frames[0].t >= cover[0] && f.t - man.frames[0].t < cover[1]) {  // camera covered
          if (!blank) { blank = new OffscreenCanvas(W, H); blank.getContext('2d').fillRect(0, 0, W, H); }
          return { bitmap: await createImageBitmap(blank), t: f.t, down: imuDown(f.t), imu: imuUpTo(f.t) };
        }
        return { bitmap: await createImageBitmap(await (await fetch(f.url)).blob()), t: f.t, down: imuDown(f.t), imu: imuUpTo(f.t) };
      };
      const first = await nextBitmap();
      W = first.bitmap.width; H = first.bitmap.height;
      first.bitmap.close();
      K = fx ? { fx, fy: num('fy'), cx: num('cx'), cy: num('cy') } : intrinsics(W, H, hfov);  // original image size (ORB-SLAM3 rescales internally)
    } else {
      const [rw, rh] = $('res').value.split('x').map(Number);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: rw }, height: { ideal: rh }, frameRate: { ideal: 30 } },
        audio: false,
      });
      const video = Object.assign(document.createElement('video'), { srcObject: stream, muted: true, playsInline: true });
      await video.play();
      W = video.videoWidth; H = video.videoHeight;
      calib = params.get('calib') === '0' ? null : loadCalibration(W, H);  // calib.html, this browser, this stream size
      K = intrinsics(W, H, hfov, calib);  // without a calibration: the FOV is along the sensor long side (portrait streams too)
      fps = stream.getVideoTracks()[0].getSettings().frameRate || 30;
      settingsText = cameraSettings(W, H, hfov, fps, Number(params.get('features') || 1000), calib);  // + IMU block at init
      const tc = performance.now();
      if ('MediaStreamTrackProcessor' in window && params.get('webcodecs') !== '0') {
        camTrack = stream.getVideoTracks()[0].clone();  // its own track: the <video> element keeps the original
        camStream = new MediaStreamTrackProcessor({ track: camTrack }).readable;
      }
      const grab = async () => ({ bitmap: await createImageBitmap(video),
                                  t: absClock() ? (performance.timeOrigin + performance.now()) / 1000 : (performance.now() - tc) / 1000 });
      nextBitmap = grab;
      frameClock = 'grab time (performance.now)';
      if ('requestVideoFrameCallback' in HTMLVideoElement.prototype && params.get('rvfc') !== '0') {
        // Camera frames with their own timestamps (metadata.mediaTime): every presented frame is grabbed when it
        // arrives and the tracker takes the newest one, so the frame interval seen by SLAM is the camera's, not the
        // JS loop's, and frames that arrive while tracking runs are counted as dropped. The video element is kept in
        // the page (1 px) so that the browser presents its frames; if no callback arrives within 400 ms, the page
        // falls back to grabbing frames on demand (grab time).
        Object.assign(video.style, { position: 'fixed', left: '0', top: '0', width: '1px', height: '1px', opacity: '0.01', pointerEvents: 'none' });
        document.body.append(video);
        let latest = null, waiter = null, active = true;
        const onFrame = (now, meta) => {
          if (!active) return;
          camFrames++;
          createImageBitmap(video).then((bmp) => {
            if (!active) { bmp.close(); return; }
            if (latest) { latest.bitmap.close(); camDropped++; }  // replaced before the tracker took it
            latest = { bitmap: bmp, t: absClock() ? (performance.timeOrigin + now) / 1000 : meta.mediaTime };
            if (waiter) { const w = waiter; waiter = null; w(); }
          });
          video.requestVideoFrameCallback(onFrame);
        };
        video.requestVideoFrameCallback(onFrame);
        stopRvfc = () => { active = false; if (latest) { latest.bitmap.close(); latest = null; } };
        frameClock = 'camera (requestVideoFrameCallback)';
        nextBitmap = async () => {
          if (!active) return grab();
          if (!latest) {
            const ok = await new Promise((res) => { waiter = () => res(true); setTimeout(() => res(false), 400); });
            if (!ok && !latest && camFrames === 0) {  // the browser never presented a frame: use on-demand grabs
              active = false; waiter = null;
              frameClock = 'grab time (no requestVideoFrameCallback frames)';
              tele(`frame clock fallback: ${frameClock}`);
              return grab();
            }
            if (!latest) await new Promise((res) => { waiter = res; });  // camera paused (page hidden): wait
          }
          const f = latest; latest = null;
          return f;
        };
      }
    }
    layout();
    $('start').hidden = true; $('topRow').hidden = false;
    $('menuBtn').hidden = $('infoBtn').hidden = $('miniWrap').hidden = false;  // panels start collapsed
    $('usage').hidden = true;
    setupPanels();
    // Back / Options return to the start screen (options kept in the URL): reload ends the camera and the worker
    history.pushState({ running: true }, '');
    addEventListener('popstate', () => location.reload());
    $('btnOpts').onclick = () => location.reload();
    hint('Loading SLAM…');
    new GLTFLoader().load('./assets/tree.glb', (g) => {  // normalise: height 1, base centre at the origin
      const box = new THREE.Box3().setFromObject(g.scene);
      const size = box.getSize(new THREE.Vector3()), centre = box.getCenter(new THREE.Vector3());
      g.scene.position.set(-centre.x, -box.min.y, -centre.z);
      const wrap = new THREE.Group();
      wrap.add(g.scene);
      wrap.scale.setScalar(1 / (size.y || 1));
      cactusProto = new THREE.Group();
      cactusProto.add(wrap);
    });

    tele(`start: ${navigator.userAgent} | cores ${navigator.hardwareConcurrency} | image ${W}x${H} | ` +
         `fx ${K.fx.toFixed(1)} fy ${K.fy.toFixed(1)} cx ${K.cx.toFixed(1)} cy ${K.cy.toFixed(1)} px | ` +
         (seq ? 'dataset intrinsics' : calib ? `calibrated ${calib.date.slice(0, 10)} (rms ${calib.rms.toFixed(2)} px, k1 ${calib.k1.toFixed(3)} k2 ${calib.k2.toFixed(3)})` : `hfov ${hfov} (assumed)`) +
         ` | preset ${presetKey()} | options ${describeOptions(opts)} | source ${seq || 'camera'} | frame clock ${frameClock} | gyro axes: rotationRate order ${RR_ORDER}`);
    let lastSlamLine = '', slamRepeats = 0;
    // loading progress: vocabulary download %, then building; a stall (no data for 15 s) is reported
    let lastProgress = performance.now(), loadingDone = false;
    const onProgress = (p) => {
      lastProgress = performance.now();
      if (p.stage === 'download') {
        const mb = (x) => (x / 1048576).toFixed(1);
        hint(p.total ? `Loading SLAM… ${Math.min(99, Math.floor(100 * p.loaded / p.total))}% (${mb(p.loaded)} / ${mb(p.total)} MB)` : `Loading SLAM… ${mb(p.loaded)} MB`);
      } else {
        hint('Loading SLAM… building the vocabulary (a few seconds)');
      }
    };
    const stallTimer = setInterval(() => {
      if (!loadingDone && performance.now() - lastProgress > 15000) {
        hint('Loading seems stalled: no data for 15 s. Check the connection, then reload the page.');
        tele('loading stalled (no vocabulary data for 15 s)');
        lastProgress = performance.now();
      }
    }, 3000);
    const slam = createSlamWorker((s) => {
      console.log(s);
      if (!/Web config|ERROR|Map|map|Loading|loaded|[Rr]eloc|[Ll]ost|[Rr]eset|[Mm]erg|[Ll]oop|Fail|Not enough|initiali|IMU|scale too small|[Tt]imestamp|VIBA|Bundle/.test(s)) return;
      if (s === lastSlamLine) { slamRepeats++; return; }  // e.g. "Lost for a short time" on every frame
      if (slamRepeats) tele(`slam: (previous line ${slamRepeats} more times)`);
      lastSlamLine = s; slamRepeats = 0;
      tele(`slam: ${s}`);
    }, onProgress);
    if ((inertial || gyroOn) && !seq) {  // phone IMU block (+ T_b_c1 for the gyro prior) with the measured DeviceMotion rate
      const hz = imuRateHz();
      settingsText += phoneImuSettings(hz);
      tele(`IMU: DeviceMotion ${hz.toFixed(1)} Hz` + (imuN >= 10 ? ` (measured over ${imuN} events)` : ' (assumed: too few events yet)'));
    }
    let room = $('room').value === 'saved' ? await roomGet() : null;
    if (room) {  // the saved map fits only the same camera and vocabulary (WebMapIO.cc checks again)
      const same = room.W === W && room.H === H && room.desc === opts.desc && room.voc === opts.voc && !inertial &&
                   ['fx', 'fy', 'cx', 'cy'].every((k) => Math.abs(room.K[k] - K[k]) <= 0.01 * Math.abs(room.K[k]));
      if (!same) {
        tele(`room not loaded: saved for ${room.W}x${room.H} fx ${room.K.fx.toFixed(1)} ${room.desc}/${room.voc}, now ${W}x${H} fx ${K.fx.toFixed(1)} ${opts.desc}/${opts.voc}${inertial ? ' (IMU on: visual-only maps)' : ''}`);
        hint('Saved room not loaded: it was saved with another camera setting, vocabulary or IMU mode. Starting a new room.', 5000);
        room = null;
      }
    }
    const tInit = performance.now();
    const ready = await slam.request({ type: 'init', vocabUrl: vocabUrl(opts.desc, opts.voc), clahe, seed: 1, prof: params.get('prof') === '1', inertial,
                                       settingsText: withTracker(settingsText), mapBytes: room ? (room.gz ? await gzip(room.map, 'd') : room.map) : null });
    if (room) {
      tele(`room map ${ready.mapLoaded ? 'loaded' : 'NOT loaded'} (saved ${room.date}, ${(room.map.byteLength / 1048576).toFixed(1)} MB, preset ${room.preset})`);
      if (ready.mapLoaded) pendingRoom = room;
      else hint('The saved room could not be loaded (see Log). Starting a new room.', 5000);
    }
    loadingDone = true; clearInterval(stallTimer);
    if (camStream) {  // WebCodecs: frames go straight to the worker; use it only if they match the stream we calibrated for
      const c = await slam.request({ type: 'camera', readable: camStream }, [camStream]);
      const why = !c.ok ? 'no frames' : (c.rotation ? `rotated ${c.rotation} deg` : (c.visibleW !== W || c.visibleH !== H ? `size ${c.visibleW}x${c.visibleH} vs ${W}x${H}` : ''));
      if (!why) {
        camMode = 'webcodecs'; stopRvfc();
        frameClock = 'camera (WebCodecs VideoFrame timestamps)';
        tele(`camera: WebCodecs ${c.format} ${c.visibleW}x${c.visibleH} (coded ${c.codedW}x${c.codedH})`);
      } else {
        await slam.request({ type: 'cameraStop' });
        if (camTrack) camTrack.stop();
        tele(`camera: WebCodecs not used (${why}, format ${c.format}); frames via ${frameClock}`);
      }
    }
    if (ready.type === 'error') throw new Error(ready.text);
    tele(`init ${(performance.now() - tInit).toFixed(0)} ms (vocabulary ${(ready.vocabBytes / 1e6).toFixed(1)} MB), wasm heap ${(ready.heap / 1048576).toFixed(0)} MiB`);
    $('btnReset').onclick = () => {  // applied between frames: the worker handles one request at a time
      resetRequested = true;
      if (stash) { if (stash.cactus) scene.remove(stash.cactus); if (stash.floorMesh) scene.remove(stash.floorMesh); stash = null; }
      resetAnchors('Map reset. Move the phone slowly sideways.');
    };
    $('btnSaveRoom').onclick = () => { saveRequested = true; hint('Saving the room…', 3000); };
    $('btnClear').onclick = () => { if (cactus) { scene.remove(cactus); cactus = null; $('sizeRow').style.visibility = 'hidden'; tele('tree removed'); } };
    let sizeLog = 0;
    $('size').oninput = () => {  // resize the placed tree (2^value: 0.25x .. 4x of the placement size)
      $('sizeVal').textContent = `${sizeFactor().toFixed(2)}×`;
      if (cactus) cactus.scale.setScalar(objBase * sizeFactor());
      clearTimeout(sizeLog); sizeLog = setTimeout(() => tele(`tree size x${sizeFactor().toFixed(2)}`), 800);
    };
    $('stage').addEventListener('pointerdown', (e) => placeCactus(e.clientX, e.clientY));

    const t0 = performance.now();
    const times = [], trackTimes = [];
    let n = 0, keepMapOn = false, prevStats = stats, prevTrackMs = 0, lostSince = 0;
    camLast = [camFrames, camDropped];  // frames that arrived while SLAM was loading do not count as dropped
    // Watchdog: the background is the frame SLAM returned, so a stuck worker looks like a frozen camera. Say so.
    // Known case: IMU on, the mapping thread hangs in the IMU initialisation and a map reset waits for it (ISSUES G13).
    let lastReply = performance.now(), stuck = false;
    setInterval(() => {
      if (stuck || performance.now() - lastReply < 4000) return;
      stuck = true;
      hint('SLAM stopped responding (known IMU bug): the picture is frozen, not the camera. Press Back or Options to restart.', 1e9);
      tele(`watchdog: no reply from SLAM for 4 s (worker stuck${inertial ? ', IMU on' : ''}), state ${STATE[state]}, KFs ${stats[0]}`);
    }, 1000);
    for (;;) {
      if (resetRequested) {
        resetRequested = false;
        await slam.request({ type: 'reset' });
        tele('map reset requested');
      }
      if (pendingRoom && cactusProto) { restoreRoom(pendingRoom); pendingRoom = null; }
      if (saveRequested) {
        saveRequested = false;
        const sv = await slam.request({ type: 'saveMap' });
        if (sv.type === 'mapSaved' && sv.ok) {
          const tz = performance.now();
          const packed = await gzip(sv.bytes, 'c');
          tele(`room map gzip: ${(sv.bytes.byteLength / 1048576).toFixed(1)} -> ${(packed.byteLength / 1048576).toFixed(1)} MB (${(performance.now() - tz).toFixed(0)} ms)`);
          const room = { v: 2, gz: true, date: new Date().toISOString(), map: packed, raw: sv.bytes.byteLength, W, H, K: { fx: K.fx, fy: K.fy, cx: K.cx, cy: K.cy },
                         desc: opts.desc, voc: opts.voc, preset: presetKey(),
                         tree: cactus ? { p: cactus.position.toArray(), q: cactus.quaternion.toArray(), base: objBase, size: Number($('size').value) } : null,
                         floor, downSum, downN, keyframes: stats[0], points: stats[1] };
          try {
            await roomPut(room);
            tele(`room saved: ${(sv.bytes.byteLength / 1048576).toFixed(1)} MB, ${stats[0]} KFs, ${stats[1]} points, tree ${cactus ? 'yes' : 'no'} (${sv.ms.toFixed(0)} ms)`);
            hint(`Room saved (${(sv.bytes.byteLength / 1048576).toFixed(1)} MB). Choose "Saved room" at the start to reload it.`, 4000);
          } catch (e) {
            tele(`room save failed (storage): ${e}`);
            hint('Saving failed: browser storage is full or blocked.', 4000);
          }
        } else {
          tele(`room save failed: ${sv.text || 'see the slam log'}`);
          hint('Saving failed (see Log).', 4000);
        }
      }
      const keep = opts.anchor === 'keep' && !!cactus;  // keep map: a loss relocalises instead of a new map (WebConfig.h)
      if (keep !== keepMapOn) {
        const k = await slam.request({ type: 'keepMap', on: keep });
        keepMapOn = keep;
        tele(k.type === 'error' ? `keep map failed: ${k.text}` : `keep map ${keep ? 'on (tree placed)' : 'off'}`);
      }
      const tf = performance.now();
      let r, down = null;
      if (camMode === 'webcodecs') {
        const imu = (inertial || gyroOn) ? new Float64Array(imuPending) : null; imuPending = [];
        r = await slam.request({ type: 'trackCamera', viz: true, withMap: n % 10 === 0, absClock: absClock(), imu });
        if (r.cam) { camFrames = r.cam[0]; camDropped = r.cam[1]; }
      } else {
        const f = await nextBitmap();  // f.down: dataset IMU gravity (camera frame), test mode only
        down = f.down;
        if (down) motionEvents++;
        let imu = null;
        if (inertial || gyroOn) { imu = new Float64Array(f.imu || imuPending); if (!f.imu) imuPending = []; }
        r = await slam.request({ type: 'bitmap', bitmap: f.bitmap, t: f.t, viz: true, withMap: n % 10 === 0, imu }, [f.bitmap]);
        replayT = f.t;
      }
      if (r.decodeMs != null) { decodeTimes.push(r.decodeMs); if (decodeTimes.length > 30) decodeTimes.shift(); }
      if (r.type !== 'pose') throw new Error(r.text || 'tracking error');
      n++;
      lastReply = performance.now();
      if (stuck) { stuck = false; hintHold = 0; hint(''); tele('watchdog: SLAM responding again'); }
      if (r.state !== state) {  // state transition: log what the tracker saw just before (loss diagnosis)
        const g = gyroPeak();
        const lost = r.state === 3 || r.state === 4;
        if (lost && !lostSince) lostSince = performance.now();
        tele(`state ${STATE[state]} -> ${STATE[r.state]} | before: matches ${prevStats[2]}, track ${prevTrackMs.toFixed(0)} ms | ` +
             `now: KFs ${r.viz.stats[0]} matches ${r.viz.stats[2]} maps ${r.viz.stats[3]} | gyro peak ${g == null ? 'n/a' : g.toFixed(0) + ' deg/s'}` +
             (!lost && lostSince ? ` | was lost ${((performance.now() - lostSince) / 1000).toFixed(1)} s` : ''));
        if (!lost) lostSince = 0;
      }
      prevStats = r.viz.stats; prevTrackMs = r.trackMs;
      if (r.mapWait !== mapWait) {
        if (r.mapWait === 0 && (mapWait === 1 || (mapWait === -1 && n <= 1))) tele(r.state === 2 || r.state === 5 ? 'room: relocalised into the saved map' : 'room: gave up waiting, new map');
        mapWait = r.mapWait;
      }
      state = r.state; stats = r.viz.stats; kps = r.viz.kps;
      if (lastMaps && stats[3] > lastMaps) stashAnchors();              // lost -> ORB-SLAM3 started a new map
      else if (lastMaps && stats[3] < lastMaps) restoreAnchors();       // maps merged back into the old one
      else if (lastState >= 2 && state === 1 && !stash) resetAnchors('Tracking restarted. Move the phone slowly sideways.');
      if (inertial && (stats[4] !== imuFlags[0] || stats[5] !== imuFlags[1])) {
        tele(`IMU: initialised ${stats[4] ? 'yes' : 'no'}, refined (VIBA2) ${stats[5] ? 'yes' : 'no'}`);
        imuFlags = [stats[4], stats[5]];
      }
      lastState = state; lastMaps = stats[3];
      holdTwc = null;
      if (holdOn && !tracking() && holdRef && performance.now() - holdRef.wall < HOLD_MS) holdTwc = heldPose();
      if (tracking()) {
        twc = r.twc;
        if (holdOn) holdRef = { twc: twc.slice(), q: qDev.slice(), wall: performance.now() };
        if (rec) {  // diagnostic: camera pose (Twc, TUM order) per tracked frame -> eval/estimate_time_offset.py --ar-log
          const q = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().fromArray(twc));
          tele(`pose ${r.t.toFixed(4)} ${twc[12].toFixed(4)} ${twc[13].toFixed(4)} ${twc[14].toFixed(4)} ` +
               `${q.x.toFixed(6)} ${q.y.toFixed(6)} ${q.z.toFixed(6)} ${q.w.toFixed(6)} ${stats[3]}`);  // + map count
          recPoses.push([r.t, q.x, q.y, q.z, q.w, stats[3], twc[12], twc[13], twc[14]]);
          while (recPoses.length && recPoses[0][0] < r.t - 90) recPoses.shift();
          if (performance.now() - recCalibAt > 10000 && recPoses.length > 50 && r.t - recPoses[0][0] > 20) {
            recCalibAt = performance.now();
            const c = estimateImuCalib(recPoses, recGyro);  // web/imucalib.js (= eval/estimate_time_offset.py --focal)
            if (c) {
              // focal ratio = f_true / f_assumed -> the horizontal FOV that would match (uncalibrated phone: the FOV box)
              const fovNow = Number($('fov').value), fovFit = 2 * Math.atan(Math.tan(fovNow * Math.PI / 360) / c.focalRatio) * 180 / Math.PI;
              const f = Number.isFinite(c.focalRatio) ? `focal x ${c.focalRatio.toFixed(3)} (roll ${c.rollRatio.toFixed(3)}, ${c.focalSamples} samples)` +
                (calib ? '' : ` -> FOV ${fovFit.toFixed(1)}° instead of ${fovNow}°`) : 'focal: rotate more';
              tele(`imu calib (last ${(r.t - recPoses[0][0]).toFixed(0)} s): offset ${c.offsetMs >= 0 ? '+' : ''}${c.offsetMs} ms (cost ${c.cost.toFixed(4)}, ${c.samples} rates), ${f}`);
              if (!replay) {  // kept for the start screen's "Apply" (per resolution choice)
                try {
                  localStorage.setItem(`imucalib:${resChoice}`, JSON.stringify({ offsetMs: c.offsetMs, focalRatio: c.focalRatio,
                    fovNow, fovFit: Number.isFinite(c.focalRatio) && !calib ? fovFit : null, date: new Date().toISOString(), samples: c.samples }));
                } catch { /* storage blocked */ }
              }
              hint(`IMU time offset ${c.offsetMs >= 0 ? '+' : ''}${c.offsetMs} ms · ${f}`, 4000);
            }
            const sc = replay ? null : estimateScale(recPoses, recAcc);  // live phone only (replays have no device accelerometer)
            if (sc) {
              imuScaleM = sc.scale;
              tele(`imu scale (last ${sc.span.toFixed(0)} s): 1 map unit = ${sc.scale.toFixed(3)} m (${sc.windows} windows, |g| ${sc.gLinear.toFixed(2)} from the linear step, ` +
                   `bias ${sc.bias.toFixed(2)} m/s2, rms ${sc.rms.toFixed(3)} m/s)` + (floor ? ` -> camera ${(floor.height * sc.scale).toFixed(2)} m above the floor` : ''));
            }
          }
        }
        const g = down || gravityCam();
        if (g) { const gw = mulR(rotOf(twc), g); downSum = downSum.map((x, i) => x + gw[i]); downN++; }
      }
      if (r.viz.mps) {
        localPts = [];
        for (let i = 0; i < r.viz.mps.length; i += 4) if (r.viz.mps[i + 3] > 0) localPts.push([r.viz.mps[i], r.viz.mps[i + 1], r.viz.mps[i + 2]]);
        if (tracking()) detectFloor();
      }
      if (tracking() && !trackingSince) trackingSince = performance.now();
      if (!tracking()) trackingSince = 0;
      resolvePendingTap();  // instant placement
      updateMini(r);
      drawFrame(r.bmp);
      while (replayTaps.length && replayTaps[0].t <= replayT) {  // recorded tap: same image point as on the phone
        const p = replayTaps.shift();
        tele(`replayed tap at ${p.u.toFixed(0)}, ${p.v.toFixed(0)} px (t ${p.t.toFixed(2)} s): ${tracking() ? 'tracking' : 'not tracking, skipped'}`);
        placeCactus(rect.x + p.u * rect.s, rect.y + p.v * rect.s);
      }
      times.push(performance.now() - tf);
      if (times.length > 30) times.shift();
      if (state === 1) hint('Move the phone slowly sideways to start tracking');
      else if ((state === 3 || state === 4) && mapWait === 1) hint('Saved room: point the camera at the area you saved (where the tree was)');
      else if (state === 3 || state === 4) hint(keepMapOn ? 'Tracking lost: point the camera back at where the tree was' : 'Tracking lost: move back towards where you were');
      else if (tracking() && inertial && !(stats[4] && stats[5])) hint(stats[4] ? 'IMU refining… keep moving' : 'IMU initialising: move the phone around');
      else if (tracking() && !cactus) hint(floor ? 'Tap the floor to place the tree' : (downN ? 'Point at the floor…' : 'Tap a flat surface to place the tree'));
      else if (tracking() && cactus && hintText) hint('');
      if (n % 5 === 0) {
        const fpsNow = 1000 / (times.reduce((a, b) => a + b, 0) / times.length);
        const mhz = motionEvents / ((performance.now() - t0) / 1000);
        $('hud').innerHTML = `<b>${STATE[state]}</b> · ${(opts.tracker === "klt" && !inertial) ? "hybrid" : "stock"} · ${fpsNow.toFixed(1)} fps · track ${r.trackMs.toFixed(0)} ms · ` +
          `KFs ${stats[0]} · points ${stats[2]} · floor ${floor ? 'yes' : 'no'} · gravity ${mhz ? mhz.toFixed(0) + ' Hz' : 'n/a'} · ${W}×${H} ${seq ? 'dataset camera' : calib ? 'calibrated' : `${hfov}° FOV (assumed)`}`;
      }
      trackTimes.push(r.trackMs);
      if (n % 15 === 0) {  // resource panel (top right): memory, map size, tracking load, threads
        const avg = times.reduce((a, b) => a + b, 0) / Math.max(1, times.length);
        const js = performance.memory ? performance.memory.usedJSHeapSize / 1048576 : null;
        const trk = trackTimes.length ? trackTimes.reduce((a, b) => a + b, 0) / trackTimes.length : r.trackMs;
        $('usage').innerHTML = `WASM ${(r.heap / 1048576).toFixed(0)} MB` + (js != null ? ` · JS ${js.toFixed(0)} MB` : '') +
          `<br>map ${stats[1]} pts · ${stats[0]} KFs · ${stats[3]} map${stats[3] === 1 ? '' : 's'}` +
          `<br>tracking ${Math.min(100, 100 * trk / Math.max(avg, 1)).toFixed(0)}% busy · ${navigator.hardwareConcurrency || '?'} cores, SLAM threads 3`;
      }
      if (n % 300 === 0) {  // per-stage profile (?prof=1 loads the profiling build; KLT counters always)
        const pr = await slam.request({ type: 'profile' });
        const p = pr.p || [], f = (x) => (x == null ? '-' : x.toFixed(1));
        let line = `profile: KLT frames ${p[0]} ORB frames ${p[1]} | KLT ms ${f(p[2])} | LK pyramid ms/frame ${f(p[3])}`;
        if (p.length > 4) {
          line += ` | tracking (ms, mean of ${p[9]} frames): ORB ${f(p[4])} pose ${f(p[5])} local map ${f(p[6])} new KF ${f(p[7])} total ${f(p[8])}` +
                  ` | mapping (ms per KF, ${p[16]} KFs): insert ${f(p[10])} MP culling ${f(p[11])} MP creation ${f(p[12])} local BA ${f(p[13])} KF culling ${f(p[14])} total ${f(p[15])}`;
        }
        tele(line + ` | page: gray ${decodeTimes.length ? f([...decodeTimes].sort((x, y) => x - y)[decodeTimes.length >> 1]) : '-'} ms (${camMode})`);
      }
      if (n % 30 === 0) {
        const s = [...trackTimes].sort((a, b) => a - b);
        const fpsNow = 1000 / (times.reduce((a, b) => a + b, 0) / times.length);
        tele(`frame ${n}: ${STATE[state]} | ${fpsNow.toFixed(1)} fps | track ms med ${s[s.length >> 1].toFixed(1)} p95 ${s[Math.floor(s.length * 0.95)].toFixed(1)} | ` +
             `gray ms ${decodeTimes.length ? [...decodeTimes].sort((x, y) => x - y)[decodeTimes.length >> 1].toFixed(1) : '-'} (${camMode}) | ` +
             `KFs ${stats[0]} MPs ${stats[1]} matches ${stats[2]} maps ${stats[3]} | wasm heap ${(r.heap / 1048576).toFixed(0)} MB | floor ${floor ? `yes (cam height ${floor.height.toFixed(2)}, ${floor.inliers} inl)` : 'no'} | ` +
             `gravity ${(motionEvents / ((performance.now() - t0) / 1000)).toFixed(0)} Hz | tree ${cactus ? 'placed' : 'no'}` +
             (camFrames ? ` | camera ${camFrames} frames, ${(100 * (camDropped - camLast[1]) / Math.max(1, camFrames - camLast[0])).toFixed(0)}% dropped (last 30 frames)` : ''));
        camLast = [camFrames, camDropped];
        trackTimes.length = 0;
      }
    }
  } catch (e) {
    $('start').hidden = false; $('go').disabled = false;
    $('err').textContent = String(e.message || e);
    tele(`ERROR ${e.stack || e}`);
  }
}

// Session log: kept in memory (Log button: view, copy, download — for tests on GitHub Pages) and, when served by
// web/serve.py on localhost, appended to H:\Research\results\browser\ar_<start time>.log every 5 s.
const logLines = [], LOG_MAX = 20000;
let teleSent = 0;  // logLines[0 .. teleSent) were sent to the dev server
const teleName = `ar_${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}.log`;
function tele(line) {
  logLines.push(`${new Date().toISOString().slice(11, 23)} ${line}`);
  if (logLines.length > LOG_MAX) { logLines.splice(0, 1000); teleSent = Math.max(0, teleSent - 1000); }
}
setInterval(() => {
  if (teleSent >= logLines.length || location.hostname !== 'localhost') return;
  // sendBeacon: fire-and-forget, does not hold a connection (a fetch over adb reverse could stall); only new lines
  // (a beacon body is limited to 64 KB)
  if (navigator.sendBeacon(`/api/result?name=${teleName}&append=1`, logLines.slice(teleSent).join('\n') + '\n')) teleSent = logLines.length;
}, 5000);
const logText = () => `${location.href}\n${logLines.join('\n')}\n`;
$('btnLog').onclick = () => { $('logText').textContent = logText(); $('log').hidden = false; $('logText').scrollTop = 1e9; };
$('logClose').onclick = () => { $('log').hidden = true; };
$('logCopy').onclick = async () => {
  try { await navigator.clipboard.writeText(logText()); $('logCopy').textContent = 'Copied'; } catch (e) { $('logCopy').textContent = 'Copy failed'; }
  setTimeout(() => { $('logCopy').textContent = 'Copy'; }, 2000);
};
$('logSave').onclick = () => {
  const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(new Blob([logText()], { type: 'text/plain' })), download: teleName });
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
};
addEventListener('error', (e) => tele(`ERROR ${e.message} @ ${e.filename}:${e.lineno}`));
addEventListener('unhandledrejection', (e) => tele(`ERROR (promise) ${e.reason && (e.reason.stack || e.reason)}`));

// tracker=klt: ORB-SLAM3 with KLT tracking frames between ORB frames (Web.KLT.*, include/WebConfig.h)
function withTracker(yaml) {  // applies the selected options (web/options.js)
  // IMU on: stock tracking. KLT frames do not carry IMU preintegration yet; with them, mono-inertial never gets past the
  // early resets (EuRoC MH01 in the dataset test mode), while stock tracking initialises and refines the IMU.
  if (inertial && opts.tracker === 'klt') {
    tele('IMU on: hybrid tracker disabled (KLT frames are not IMU-aware yet), ORB on every frame');
    return applyOptions(yaml, { ...opts, tracker: 'stock' });
  }
  return applyOptions(yaml, opts);
}

$('go').onclick = start;
function showImuCalib() {  // start screen: the last IMU offset / focal measured with "Record gyro + poses" at this resolution
  let c = null;
  try { c = JSON.parse(localStorage.getItem(`imucalib:${$('res').value}`) || 'null'); } catch { /* storage blocked */ }
  if (!c) { $('imuNote').textContent = ''; return; }
  $('imuNote').innerHTML = `Measured ${new Date(c.date).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })} (Record gyro + poses): IMU offset ${c.offsetMs >= 0 ? '+' : ''}${c.offsetMs} ms` +
    (c.fovFit ? `, FOV ${c.fovFit.toFixed(1)}° (was ${c.fovNow}°)` : '') + ' <button id="imuApply">Apply</button>';
  $('imuApply').onclick = () => {
    ensureImuOffsetValue(c.offsetMs);
    opts.imudt = String(c.offsetMs);
    if (c.fovFit) $('fov').value = c.fovFit.toFixed(1);
    renderOpts(); saveOptsToUrl(); syncPreset();
    $('imuNote').textContent = `Applied: IMU offset ${c.offsetMs >= 0 ? '+' : ''}${c.offsetMs} ms` + (c.fovFit ? `, FOV ${c.fovFit.toFixed(1)}°` : '') +
      ' (offset: IMU and gyro presets; FOV: when the camera is not calibrated)';
  };
}
$('res').addEventListener('change', showImuCalib);
showImuCalib();
roomGet().then((room) => {  // start screen: offer the saved room
  if (!room) { $('roomNote').textContent = 'No saved room yet: use "Save room" while tracking.'; return; }
  const o = document.createElement('option');
  o.value = 'saved';
  o.textContent = `Saved room (${new Date(room.date).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })})`;
  $('room').append(o);
  $('roomNote').textContent = `Saved: ${(room.map.byteLength / 1048576).toFixed(1)} MB, ${room.keyframes} keyframes, ${room.tree ? 'with a tree' : 'no tree'}, ` +
    `${room.W}×${room.H}, preset ${room.preset}. Same camera setting and vocabulary needed.`;
});
{  // camera model shown on the start screen (the calibration is chosen by stream size once the camera runs)
  const saved = savedCalibrations();
  $('calibNote').innerHTML = saved.length
    ? `Camera calibrated for ${saved.map((c) => `${c.w}×${c.h}`).join(', ')} (used when the stream has that size). <a href="./calib.html" style="color:#5bd18b">Recalibrate</a>`
    : `Camera not calibrated: the field of view below is assumed. <a href="./calib.html" style="color:#5bd18b">Calibrate (chessboard)</a>`;
}
// debugging handle (console): scene graph, camera, anchors
window.__ar = { scene, cam3, renderer,
                snapshot: (quality = 0.8) => {  // JPEG data URL of the camera image with the 3D overlay
                  renderer.render(scene, cam3);
                  const c = Object.assign(document.createElement('canvas'), { width: W, height: H });
                  const g = c.getContext('2d');
                  g.drawImage(bg, 0, 0);
                  g.drawImage($('gl'), 0, 0, W, H);
                  return c.toDataURL('image/jpeg', quality);
                }, get cactus() { return cactus; }, get floor() { return floor; }, get twc() { return twc; },
                get rect() { return rect; }, get K() { return K; } };
if (!crossOriginIsolated) $('err').textContent = 'Not cross-origin isolated: open through web/serve.py (WASM threads need COOP/COEP).';
