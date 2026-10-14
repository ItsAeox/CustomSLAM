// SLAM viewer: runs ORB-SLAM3 (WebAssembly, in slam.worker.js) over a dataset sequence and shows, live,
//  - an ORB-SLAM3-style 3D map view (map points, local map, keyframes, covisibility graph, current camera,
//    live trajectory, Sim(3)-aligned ground truth), rotatable with the mouse;
//  - the current image with ORB-SLAM3's frame overlay (tracked map points, init matches, status bar);
//  - tracking time, throughput and live accuracy against ground truth.
//
// URL parameters (optional, for scripted runs): seq=<dataset>/<name>, seed, pacing=rt|live|half|fast, max, autorun=1

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { alignSim3, nearest, parseTum, rotationRmse } from './align.js';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const STATE = { '-1': 'SYSTEM NOT READY', 0: 'NO IMAGES YET', 1: 'NOT INITIALIZED', 2: 'OK', 3: 'RECENTLY LOST', 4: 'LOST', 5: 'OK (KLT)' };
const logEl = $('log');
const log = (s) => { logEl.textContent += s + '\n'; logEl.scrollTop = logEl.scrollHeight; };

// ------------------------------------------------------------------------------------------------ 3D view
const view = $('view3d');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
view.prepend(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);
const camera = new THREE.PerspectiveCamera(50, 1, 0.001, 2000);
camera.up.set(0, -1, 0);  // ORB-SLAM3 world: y down, camera looks along +z
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
function resetView() {  // ORB-SLAM3's default viewpoint (ViewpointX 0, Y -0.7, Z -1.8)
  const c = followTarget.clone();
  camera.position.set(c.x, c.y - 0.7, c.z - 1.8);
  controls.target.copy(c);
  controls.update();
}
const followTarget = new THREE.Vector3();
new ResizeObserver(() => {
  const w = view.clientWidth, h = view.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
}).observe(view);

const pxr = renderer.getPixelRatio();
const mpGeom = new THREE.BufferGeometry();
const mpCloud = new THREE.Points(mpGeom, new THREE.PointsMaterial({ size: 2 * pxr, sizeAttenuation: false, vertexColors: true }));
const kfLines = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x0000ff }));
const graphLines = new THREE.LineSegments(new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({ color: 0x00b000, transparent: true, opacity: 0.6 }));
const camLines = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x00b000 }));
const trajLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xff8800 }));
const gtLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x888888 }));
scene.add(mpCloud, kfLines, graphLines, camLines, trajLine, gtLine);
camLines.frustumCulled = trajLine.frustumCulled = gtLine.frustumCulled = false;
mpCloud.frustumCulled = kfLines.frustumCulled = graphLines.frustumCulled = false;
resetView();

function setPositions(obj, arr, count) {
  obj.geometry.setAttribute('position', new THREE.BufferAttribute(arr, 3));
  obj.geometry.setDrawRange(0, count ?? arr.length / 3);
}

// Camera frustum as in MapDrawer::DrawKeyFrames: w, h = 0.75 w, z = 0.6 w; 8 segments
const FRUSTUM = (w) => {
  const h = w * 0.75, z = w * 0.6;
  const P = [[0, 0, 0], [w, h, z], [w, -h, z], [-w, -h, z], [-w, h, z]];
  return [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]].flatMap(([a, b]) => [P[a], P[b]]);
};
function frustumInto(out, o, R, t, verts) {  // R row-major 9, t 3
  for (const p of verts) {
    out[o++] = R[0] * p[0] + R[1] * p[1] + R[2] * p[2] + t[0];
    out[o++] = R[3] * p[0] + R[4] * p[1] + R[5] * p[2] + t[1];
    out[o++] = R[6] * p[0] + R[7] * p[1] + R[8] * p[2] + t[2];
  }
  return o;
}
const KF_VERTS = FRUSTUM(0.05), CAM_VERTS = FRUSTUM(0.08);  // ORB-SLAM3 KeyFrameSize / CameraSize

function updateMap(v) {
  const n = v.mps.length / 4;
  const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    pos[3 * i] = v.mps[4 * i]; pos[3 * i + 1] = v.mps[4 * i + 1]; pos[3 * i + 2] = v.mps[4 * i + 2];
    if (v.mps[4 * i + 3] > 0) col[3 * i] = 1;  // local map: red, others black
  }
  mpGeom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  mpGeom.setAttribute('color', new THREE.BufferAttribute(col, 3));
  const nk = v.kfs.length / 12;
  const kf = new Float32Array(nk * KF_VERTS.length * 3);
  let o = 0;
  for (let i = 0; i < nk; i++) {
    const R = v.kfs.subarray(12 * i, 12 * i + 9), t = v.kfs.subarray(12 * i + 9, 12 * i + 12);
    o = frustumInto(kf, o, R, t, KF_VERTS);
  }
  setPositions(kfLines, kf);
  setPositions(graphLines, v.edges);
}

function updateCamera(twc) {  // column-major 4x4 camera-to-world
  const R = [twc[0], twc[4], twc[8], twc[1], twc[5], twc[9], twc[2], twc[6], twc[10]];
  const t = [twc[12], twc[13], twc[14]];
  const arr = new Float32Array(CAM_VERTS.length * 3);
  frustumInto(arr, 0, R, t, CAM_VERTS);
  setPositions(camLines, arr);
  if ($('optFollow').checked) {  // keep the viewing offset, move with the camera (ORB-SLAM3 "Follow Camera")
    const d = new THREE.Vector3(t[0], t[1], t[2]).sub(controls.target);
    controls.target.add(d);
    camera.position.add(d);
  }
  followTarget.set(t[0], t[1], t[2]);
}

function render() {
  kfLines.visible = $('optKF').checked;
  graphLines.visible = $('optGraph').checked;
  trajLine.visible = $('optTraj').checked;
  gtLine.visible = $('optGT').checked;
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(render);
}
requestAnimationFrame(render);
$('resetView').onclick = resetView;

// ------------------------------------------------------------------------------------------------ frame view
const fcv = $('frame');
const fctx = fcv.getContext('2d');

function drawFrame(bmp, v, state) {
  if (fcv.width !== bmp.width || fcv.height !== bmp.height) { fcv.width = bmp.width; fcv.height = bmp.height; }
  fctx.drawImage(bmp, 0, 0);
  bmp.close();
  const s = Math.max(1, fcv.width / 640);  // marker size relative to image resolution
  const k = v.kps;
  if (state === 1) {  // initialisation: lines from the initial frame's keypoints (FrameDrawer, NOT_INITIALIZED)
    fctx.strokeStyle = '#0f0'; fctx.lineWidth = s;
    fctx.beginPath();
    for (let i = 0; i < v.ini.length; i += 4) { fctx.moveTo(v.ini[i], v.ini[i + 1]); fctx.lineTo(v.ini[i + 2], v.ini[i + 3]); }
    fctx.stroke();
  }
  const r = 5 * s;
  for (let i = 0; i < k.length; i += 3) {
    const x = k[i], y = k[i + 1], f = k[i + 2];
    if (f === 1 || f === 2) {  // map point match: green; visual-odometry match: blue
      fctx.strokeStyle = fctx.fillStyle = f === 1 ? '#0f0' : '#00f';
      fctx.lineWidth = s;
      fctx.strokeRect(x - r, y - r, 2 * r, 2 * r);
      fctx.beginPath(); fctx.arc(x, y, 2 * s, 0, 2 * Math.PI); fctx.fill();
    } else if ($('optAllKp').checked && f === 0) {
      fctx.fillStyle = '#ff0';
      fctx.fillRect(x - s, y - s, 2 * s, 2 * s);
    }
  }
}

function infoText(state, st) {  // FrameDrawer::DrawTextInfo
  if (state === 1) return 'TRYING TO INITIALIZE';
  if (state === 4 || state === 3) return `TRACK LOST. TRYING TO RELOCALIZE   |  Maps: ${st[3]}`;
  if (state === 2 || state === 5) return `SLAM MODE${state === 5 ? " (KLT frame)" : ""} |  Maps: ${st[3]}, KFs: ${st[0]}, MPs: ${st[1]}, Matches: ${st[2]}`;
  return 'WAITING FOR IMAGES';
}

// ------------------------------------------------------------------------------------------------ charts
function spark(canvas, values, { max, ref, color = '#2563eb' } = {}) {
  const c = canvas.getContext('2d');
  const w = canvas.width = canvas.clientWidth * pxr, h = canvas.height = canvas.clientHeight * pxr;
  c.clearRect(0, 0, w, h);
  if (!values.length) return;
  const m = max ?? (Math.max(...values) * 1.1 || 1);
  if (ref !== undefined && ref < m) {
    c.strokeStyle = '#d97706'; c.setLineDash([4 * pxr, 4 * pxr]); c.lineWidth = pxr;
    c.beginPath(); c.moveTo(0, h - ref / m * h); c.lineTo(w, h - ref / m * h); c.stroke(); c.setLineDash([]);
  }
  c.strokeStyle = color; c.lineWidth = 1.5 * pxr;
  c.beginPath();
  values.forEach((v, i) => {
    const x = values.length > 1 ? i / (values.length - 1) * w : 0, y = h - Math.min(v, m) / m * h;
    i ? c.lineTo(x, y) : c.moveTo(x, y);
  });
  c.stroke();
}

// ------------------------------------------------------------------------------------------------ run
let sequences = [];
let run = null;  // { worker, stop, paused, resume }

async function loadSequences() {
  sequences = await (await fetch('/api/sequences')).json();
  const sel = $('seq');
  const groups = {};
  for (const s of sequences) {
    const g = groups[s.dataset] ??= Object.assign(document.createElement('optgroup'), { label: s.dataset.toUpperCase() });
    g.append(new Option(`${s.name}${s.gt ? '' : ' (no GT)'}`, `${s.dataset}/${s.name}`));
  }
  for (const g of Object.values(groups)) sel.append(g);
  const live = Object.assign(document.createElement('optgroup'), { label: 'LIVE' });
  live.append(new Option('Camera (this device)', 'camera/live'));
  sel.prepend(live);
  sel.onchange = () => {
    const cam = sel.value === 'camera/live';
    $('fovLabel').hidden = $('resLabel').hidden = !cam;
    $('pacing').disabled = cam;
  };
  sel.value = sequences.length ? `${sequences[0].dataset}/${sequences[0].name}` : 'camera/live';
  if (params.get('seq')) sel.value = params.get('seq');
  sel.onchange();
  if (params.get('seed')) $('seed').value = params.get('seed');
  if (params.get('pacing')) $('pacing').value = params.get('pacing');
  if (params.get('tracker')) $('tracker').value = params.get('tracker');
}

function setStatus(s) { $('status').textContent = s; }
const fmtLen = (m) => (Math.abs(m) < 1 ? `${(m * 100).toFixed(1)} cm` : `${m.toFixed(3)} m`);
const pct = (a, q) => { if (!a.length) return NaN; const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(q * s.length))]; };

function euRocToTum(text, tBase) {  // ORB-SLAM3 EuRoC-format trajectory (rebased ns) -> TUM rows
  const out = [];
  for (const line of text.split('\n')) {
    const v = line.trim().split(/\s+/).map(Number);
    if (v.length >= 8 && v.every(Number.isFinite)) out.push(`${(v[0] * 1e-9 + tBase).toFixed(9)} ${v.slice(1, 8).join(' ')}`);
  }
  return out.join('\n') + '\n';
}

async function upload(name, text) {
  const r = await fetch(`/api/result?name=${encodeURIComponent(name)}`, { method: 'POST', body: text });
  if (!r.ok) throw new Error(`upload ${name}: ${r.status}`);
}

// ORB-SLAM3 settings for an uncalibrated camera stream: pinhole from the horizontal FOV, no distortion.
function cameraSettings(w, h, hfovDeg, fps) {
  const f = (w / 2) / Math.tan(hfovDeg * Math.PI / 360);
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
ORBextractor.nFeatures: 1000
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

// Tracker choice: the hybrid appends the Web.KLT keys (include/WebConfig.h) to the ORB-SLAM3 settings
function withTracker(yaml) {
  if ($('tracker').value !== 'klt') return yaml;
  return yaml.replace(/\s*$/, '\n') + '\n# web port: KLT tracking frames\nWeb.KLT.Enable: 1\nWeb.KLT.MaxConsecutive: 2\n';
}

// Live camera: process the newest frame whenever the tracker is free (frames arriving meanwhile are dropped).
async function startCamera() {
  const [w, h] = $('camRes').value.split('x').map(Number);
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: 'environment' }, width: { ideal: w }, height: { ideal: h }, frameRate: { ideal: 30 } },
    audio: false,
  });
  const track = stream.getVideoTracks()[0];
  const s = track.getSettings();
  const video = Object.assign(document.createElement('video'), { srcObject: stream, muted: true, playsInline: true });
  await video.play();
  const vw = video.videoWidth, vh = video.videoHeight;
  const settingsText = withTracker(cameraSettings(vw, vh, Number($('fov').value), s.frameRate || 30));
  $('run').disabled = true; $('pause').disabled = true; $('stop').disabled = false;
  logEl.textContent = '';
  $('gtName').textContent = 'live camera (no ground truth)';
  log(`camera ${track.label || ''}: ${vw}×${vh} @ ${s.frameRate ?? '?'} fps, hFOV ${$('fov').value}°`);

  const worker = new Worker('./slam.worker.js', { type: 'module' });
  run = { worker, stop: false, paused: false, resume: null, stream };
  let pending = null;
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === 'log') { if (!/^\s*$/.test(m.text)) log(m.text); return; }
    if (m.type === 'error') { log('ERROR ' + m.text); setStatus('error (see log)'); }
    if (pending) { const p = pending; pending = null; p(m); }
  };
  const request = (msg, transfer = []) => new Promise((res) => { pending = res; worker.postMessage(msg, transfer); });
  setStatus('loading WASM + vocabulary…');
  const ready = await request({ type: 'init', vocabUrl: '/vocab/ORBvoc.bin', settingsText, clahe: false, seed: Number($('seed').value) });
  if (ready.type === 'error') return finishUi();

  const trackMs = [];
  const traj = new Float32Array(3 * 100000);
  let trajN = 0, n = 0, tracked = 0, mapEvery = 5, lastMaps = 0, lastState = 0;
  const t0 = performance.now();
  while (!run.stop) {
    const bitmap = await createImageBitmap(video);
    const t = (performance.now() - t0) / 1000;
    const r = await request({ type: 'bitmap', bitmap, t, viz: true, withMap: n % mapEvery === 0 }, [bitmap]);
    if (r.type !== 'pose') break;
    n++;
    const st = r.viz.stats;
    if ((lastState >= 2 && r.state === 1) || st[3] !== lastMaps) trajN = 0;
    lastState = r.state; lastMaps = st[3];
    trackMs.push(r.trackMs);
    drawFrame(r.bmp, r.viz, r.state);
    $('infobar').textContent = infoText(r.state, st);
    if (r.viz.mps) { updateMap(r.viz); if (r.viz.ms > 8 && mapEvery < 30) mapEvery += 5; }
    if (r.state === 2 || r.state === 5) {
      tracked++;
      const m = r.twc;
      if (trajN < 100000) traj.set([m[12], m[13], m[14]], 3 * trajN++);
      setPositions(trajLine, traj, trajN);
      updateCamera(m);
    }
    if (n % 5 === 0) {
      const wallS = (performance.now() - t0) / 1000;
      $('sFrame').textContent = `${n} processed (${wallS.toFixed(0)} s)`;
      $('sState').textContent = STATE[r.state];
      $('sState').className = r.state === 2 || r.state === 5 ? 'state-ok' : r.state === 1 || r.state === 3 ? 'state-warn' : 'state-bad';
      $('sTrack').textContent = `${r.trackMs.toFixed(1)} / ${pct(trackMs, 0.5).toFixed(1)} / ${pct(trackMs, 0.95).toFixed(1)} ms`;
      $('sFps').textContent = `${(n / wallS).toFixed(1)} fps processed (camera ${s.frameRate ?? '?'} fps)`;
      $('sMap').textContent = `${st[0]} / ${st[1]} / ${st[3]}`;
      $('sCov').textContent = `${tracked} / ${n} (${(100 * tracked / n).toFixed(0)}%)`;
      spark($('chartMs'), trackMs.slice(-300), { max: Math.max(66, pct(trackMs.slice(-300), 0.98) * 1.1), ref: 33.3 });
      setStatus('live');
    }
  }
  stream.getTracks().forEach((tr) => tr.stop());
  setStatus(`stopped: ${n} frames, median track ${pct(trackMs, 0.5).toFixed(1)} ms`);
  finishUi();
}

async function start() {
  if ($('seq').value === 'camera/live') return startCamera();
  const [dataset, name] = $('seq').value.split('/');
  const info = sequences.find((s) => s.dataset === dataset && s.name === name);
  if (!info) return;
  const seed = Number($('seed').value);
  const pacing = $('pacing').value;
  const maxFrames = Number(params.get('max') || 0);
  $('run').disabled = true; $('pause').disabled = false; $('stop').disabled = false;
  logEl.textContent = '';
  for (const id of ['aAte', 'aRot', 'aScale', 'aN', 'aFinal']) $(id).textContent = '—';
  setPositions(trajLine, new Float32Array(3), 0); setPositions(gtLine, new Float32Array(3), 0);
  mpGeom.deleteAttribute('position'); setPositions(kfLines, new Float32Array(0)); setPositions(graphLines, new Float32Array(0));

  const manifest = await (await fetch(`/api/sequence?dataset=${dataset}&name=${encodeURIComponent(name)}`)).json();
  const frames = maxFrames > 0 ? manifest.frames.slice(0, maxFrames) : manifest.frames;
  const gt = info.gt ? parseTum(await (await fetch(info.gt)).text()) : null;
  $('gtName').textContent = gt ? `${gt.t.length} GT poses` : 'no ground truth';
  log(`${dataset}/${name}: ${frames.length} frames, settings ${info.settings}, seed ${seed}, pacing ${pacing}`);

  const worker = new Worker('./slam.worker.js', { type: 'module' });
  run = { worker, stop: false, paused: false, resume: null };
  let pending = null;
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === 'log') { if (!/^\s*$/.test(m.text)) log(m.text); return; }
    if (m.type === 'error') { log('ERROR ' + m.text); setStatus('error (see log)'); }
    if (pending) { const p = pending; pending = null; p(m); }
  };
  const request = (msg) => new Promise((res) => { pending = res; worker.postMessage(msg); });

  setStatus('loading WASM + vocabulary…');
  const settingsText = withTracker(await (await fetch(info.settings)).text());
  log(`tracker: ${$('tracker').value === 'klt' ? 'hybrid ORB + KLT' : 'ORB-SLAM3 stock'}`);
  const ready = await request({ type: 'init', vocabUrl: '/vocab/ORBvoc.bin', settingsText, clahe: info.clahe, seed });
  if (ready.type === 'error') return finishUi();
  log(`init ${ready.ms.toFixed(0)} ms, ${navigator.hardwareConcurrency} hardware threads`);

  // live state
  const trackMs = [], ateHist = [];
  const live = [];          // { t (abs), p [3], R [9] } for tracked frames of the current map segment
  const liveAll = ['# t tx ty tz qx qy qz qw  (live poses, viewer)'];
  const timing = ['idx,t,state,track_ms,decode_ms'];
  let tracked = 0, dropped = 0, segment = 0, lastState = 0, lastMaps = 0, mapEvery = 5, align = null;
  const traj = new Float32Array(frames.length * 3);
  let trajN = 0;
  const wall0 = performance.now();
  let pausedMs = 0;
  const speed = pacing === 'half' ? 0.5 : 1;

  for (let i = 0; i < frames.length && !run.stop; i++) {
    if (run.paused) {
      const p0 = performance.now();
      await new Promise((res) => { run.resume = res; });
      pausedMs += performance.now() - p0;
    }
    const elapsed = (performance.now() - wall0 - pausedMs) / 1000 * speed;  // dataset seconds since start
    if (pacing === 'live') {  // emulate a camera: skip frames that are already in the past
      while (i + 1 < frames.length && frames[i + 1].t <= elapsed) { i++; dropped++; }
    } else if (pacing !== 'fast' && frames[i].t > elapsed) {
      await new Promise((res) => setTimeout(res, (frames[i].t - elapsed) / speed * 1000));
    }
    const f = frames[i];
    const withMap = i % mapEvery === 0;
    const r = await request({ type: 'frame', idx: f.idx, t: f.t, url: f.url, viz: true, withMap });
    if (r.type !== 'pose') break;
    const st = r.viz.stats;
    // a new map segment starts after a reset or when the atlas gets a new map
    if ((lastState >= 2 && r.state === 1) || st[3] !== lastMaps) {
      if (live.length) { segment++; log(`map segment ${segment} (state ${STATE[r.state]}, maps ${st[3]})`); }
      live.length = 0; trajN = 0; align = null;
    }
    lastState = r.state; lastMaps = st[3];
    trackMs.push(r.trackMs);
    timing.push(`${f.idx},${f.t.toFixed(6)},${r.state},${r.trackMs.toFixed(3)},${r.decodeMs.toFixed(3)}`);

    drawFrame(r.bmp, r.viz, r.state);
    $('infobar').textContent = infoText(r.state, st);
    if (r.viz.mps) { updateMap(r.viz); if (r.viz.ms > 8 && mapEvery < 30) mapEvery += 5; }

    if (r.state === 2 || r.state === 5) {
      tracked++;
      const m = r.twc;
      const p = [m[12], m[13], m[14]], R = [m[0], m[4], m[8], m[1], m[5], m[9], m[2], m[6], m[10]];
      const tAbs = f.t + manifest.tBase;
      live.push({ t: tAbs, p, R });
      liveAll.push(`${tAbs.toFixed(9)} ${p.join(' ')} ${quatString(R)}`);
      traj.set(p, 3 * trajN++);
      setPositions(trajLine, traj, trajN);
      updateCamera(m);
    }

    // live accuracy every 10 frames: Sim(3) alignment of the current segment's live poses to GT
    if (gt && i % 10 === 0 && live.length >= 30) {
      const est = [], ref = [], estR = [], refR = [];
      for (const q of live) {
        const j = nearest(gt.t, q.t);
        if (j >= 0) { est.push(q.p); ref.push(gt.p[j]); estR.push(q.R); refR.push(gt.R[j]); }
      }
      align = alignSim3(est, ref, true);
      if (align) {
        const rot = rotationRmse(align.R, estR, refR);
        ateHist.push(align.rmse);
        $('aAte').textContent = fmtLen(align.rmse);
        $('aAte').className = align.rmse < 0.05 ? 'state-ok' : align.rmse < 0.2 ? 'state-warn' : 'state-bad';
        $('aRot').textContent = `${rot.toFixed(2)}°`;
        $('aScale').textContent = align.s.toFixed(3);
        $('aN').textContent = `${est.length}${segment ? ` (map segment ${segment})` : ''}`;
        $('ateMax').textContent = `max ${fmtLen(Math.max(...ateHist))}`;
        spark($('chartAte'), ateHist.slice(-300), { color: '#ff8800' });
        drawGt(gt, align, f.t + manifest.tBase);
      }
    }

    if (i % 5 === 0 || i === frames.length - 1) {
      const wallS = (performance.now() - wall0 - pausedMs) / 1000;
      $('sFrame').textContent = `${i + 1} / ${frames.length}  (t = ${f.t.toFixed(1)} s${dropped ? `, ${dropped} dropped` : ''})`;
      $('sState').textContent = STATE[r.state];
      $('sState').className = r.state === 2 || r.state === 5 ? 'state-ok' : r.state === 1 || r.state === 3 ? 'state-warn' : 'state-bad';
      $('sTrack').textContent = `${r.trackMs.toFixed(1)} / ${pct(trackMs, 0.5).toFixed(1)} / ${pct(trackMs, 0.95).toFixed(1)} ms`;
      $('sFps').textContent = `${((i + 1 - dropped) / wallS).toFixed(1)} fps processed, ${(f.t / wallS).toFixed(2)}× real time`;
      $('sMap').textContent = `${st[0]} / ${st[1]} / ${st[3]}`;
      $('sCov').textContent = `${tracked} / ${i + 1 - dropped} (${(100 * tracked / (i + 1 - dropped)).toFixed(0)}%)`;
      spark($('chartMs'), trackMs.slice(-300), { max: Math.max(66, pct(trackMs.slice(-300), 0.98) * 1.1), ref: 33.3 });
      setStatus(run.paused ? 'paused' : 'running');
    }
  }

  setStatus('finishing (ORB-SLAM3 final optimisation + save)…');
  const fin = await request({ type: 'finish' });
  if (fin.type === 'finished') {
    const finalTum = euRocToTum(fin.frames, manifest.tBase);
    if (gt) {
      const est = [], ref = [];
      for (const line of finalTum.split('\n')) {
        const v = line.split(' ').map(Number);
        if (v.length < 8) continue;
        const j = nearest(gt.t, v[0]);
        if (j >= 0) { est.push([v[1], v[2], v[3]]); ref.push(gt.p[j]); }
      }
      const a = alignSim3(est, ref, true);
      $('aFinal').textContent = a ? `${fmtLen(a.rmse)} (s = ${a.s.toFixed(3)}, ${est.length} poses)` : '—';
    }
    if ($('save').checked) {
      const base = `${name}_viewer_seed${seed}`;
      await upload(`${base}_final.txt`, '# t tx ty tz qx qy qz qw\n' + finalTum);
      await upload(`${base}_live.txt`, liveAll.join('\n') + '\n');
      await upload(`${base}_timing.csv`, timing.join('\n') + '\n');
      log(`saved results/browser/${base}_{final,live,timing}`);
    }
  }
  setStatus(run.stop ? 'stopped' : `done: ${tracked} tracked frames, median track ${pct(trackMs, 0.5).toFixed(1)} ms`);
  document.title = 'DONE SLAM Viewer';
  finishUi();
}

function drawGt(gt, a, tNow) {  // GT up to now, mapped into the SLAM frame: p = R^T (g - t) / s
  let n = 0;
  while (n < gt.t.length && gt.t[n] <= tNow) n++;
  const arr = new Float32Array(Math.max(n, 1) * 3);
  const R = a.R;
  for (let i = 0; i < n; i++) {
    const d = [gt.p[i][0] - a.t[0], gt.p[i][1] - a.t[1], gt.p[i][2] - a.t[2]];
    arr[3 * i] = (R[0] * d[0] + R[3] * d[1] + R[6] * d[2]) / a.s;
    arr[3 * i + 1] = (R[1] * d[0] + R[4] * d[1] + R[7] * d[2]) / a.s;
    arr[3 * i + 2] = (R[2] * d[0] + R[5] * d[1] + R[8] * d[2]) / a.s;
  }
  setPositions(gtLine, arr, n);
}

function quatString(R) {  // row-major rotation -> "qx qy qz qw"
  const [r00, r01, r02, r10, r11, r12, r20, r21, r22] = R;
  const tr = r00 + r11 + r22;
  let x, y, z, w;
  if (tr > 0) { const S = Math.sqrt(tr + 1) * 2; w = 0.25 * S; x = (r21 - r12) / S; y = (r02 - r20) / S; z = (r10 - r01) / S; }
  else if (r00 > r11 && r00 > r22) { const S = Math.sqrt(1 + r00 - r11 - r22) * 2; w = (r21 - r12) / S; x = 0.25 * S; y = (r01 + r10) / S; z = (r02 + r20) / S; }
  else if (r11 > r22) { const S = Math.sqrt(1 + r11 - r00 - r22) * 2; w = (r02 - r20) / S; x = (r01 + r10) / S; y = 0.25 * S; z = (r12 + r21) / S; }
  else { const S = Math.sqrt(1 + r22 - r00 - r11) * 2; w = (r10 - r01) / S; x = (r02 + r20) / S; y = (r12 + r21) / S; z = 0.25 * S; }
  return `${x} ${y} ${z} ${w}`;
}

function finishUi() {
  $('run').disabled = false; $('pause').disabled = true; $('stop').disabled = true;
  if (run) {
    run.worker.terminate();
    if (run.stream) run.stream.getTracks().forEach((tr) => tr.stop());
    run = null;
  }
}

$('run').onclick = () => start().catch((e) => { log('ERROR ' + (e.stack || e)); setStatus('error'); finishUi(); });
$('pause').onclick = () => {
  if (!run) return;
  run.paused = !run.paused;
  $('pause').textContent = run.paused ? 'Resume' : 'Pause';
  if (!run.paused && run.resume) { run.resume(); run.resume = null; }
  setStatus(run.paused ? 'paused' : 'running');
};
$('stop').onclick = () => {
  if (!run) return;
  run.stop = true;
  if (run.paused && run.resume) { run.paused = false; run.resume(); }
};

if (!crossOriginIsolated) {
  setStatus('Not cross-origin isolated: open this page through web/serve.py (WASM threads need COOP/COEP).');
  $('run').disabled = true;
} else {
  loadSequences().then(() => { if (params.get('autorun') === '1') $('run').click(); })
    .catch((e) => log('ERROR loading sequences: ' + e));
}
