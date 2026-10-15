// Markerless AR in the browser: ORB-SLAM3 (WASM, worker) tracks the camera; the floor is found as a RANSAC plane over
// the local map points (constrained to gravity from DeviceMotion when available); tapping places a cactus on it.
// The background shows the frame the pose was computed for, so the overlay is in sync with the image (no lag).
//
// URL parameters: fov=<horizontal FOV in degrees> (default 69: Pixel 9 main camera, ~25 mm equivalent at 4:3), res=640x480, features=1000,
//                 seq=<dataset>/<name> (test mode: plays a dataset sequence in real time instead of the camera)

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { createSlamWorker, cameraSettings, STATE, vocabUrl } from './slamclient.js';
import { dot, fitPlaneRansac, normalize, rayPlane } from './plane.js';
import { applyOptions, buildOptionsUI, describeOptions, optionsFromParams, optionsToParams } from './options.js';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
if (params.get('fov')) $('fov').value = params.get('fov');
if (params.get('res')) $('res').value = params.get('res');
// algorithm / settings variant (web/options.js), also in the URL. AR: relocalise for 10 s before a new map (stock 3 s)
const opts = optionsFromParams(params, { reloc: '10' });
buildOptionsUI($('opts'), opts, () => history.replaceState(null, '', `?${optionsToParams(opts, new URLSearchParams(location.search))}`));

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

let W = 640, H = 480, F = 500;      // processed image size and focal length (pixels)
let rect = { x: 0, y: 0, w: 1, h: 1, s: 1 };
let twc = null, state = 0, stats = [0, 0, 0, 0];
let localPts = [];                  // local map points [[x,y,z]]
let kps = null;
let showPoints = true;
let floor = null, floorMesh = null;  // plane {n (up), d, height}
let cactusProto = null, cactus = null;
let downSum = [0, 0, 0], downN = 0;  // gravity direction in the SLAM world frame (sum of unit vectors)
let accLP = null, motionEvents = 0;
let lastMaps = 0, lastState = 0;
let stash = null;          // anchors of a map that went inactive (tracking lost -> new map); restored on map merge
let resetRequested = false;  // Reset button: sent to the worker between frames (one request at a time)
const tracking = () => state === 2 || state === 5;  // 5: tracked by a KLT frame

function layout() {  // cover-fit the processed image into the window; the GL canvas covers the same rectangle
  const s = Math.max(innerWidth / W, innerHeight / H);
  rect = { s, w: W * s, h: H * s, x: (innerWidth - W * s) / 2, y: (innerHeight - H * s) / 2 };
  for (const c of [bg, $('gl')]) Object.assign(c.style, { left: `${rect.x}px`, top: `${rect.y}px`, width: `${rect.w}px`, height: `${rect.h}px` });
  bg.width = W; bg.height = H;
  renderer.setSize(rect.w, rect.h, false);
  cam3.fov = 2 * Math.atan(H / 2 / F) * 180 / Math.PI;
  cam3.aspect = W / H;
  cam3.updateProjectionMatrix();
}
addEventListener('resize', layout);

// ---------------------------------------------------------------- gravity (DeviceMotion, any rate is fine)
function onMotion(e) {
  const a = e.accelerationIncludingGravity;
  if (!a || a.x == null) return;
  const v = [a.x, a.y, a.z];
  accLP = accLP ? accLP.map((x, i) => 0.7 * x + 0.3 * v[i]) : v;
  motionEvents++;
}
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
  hint('Tracking lost: new map started. Go back to where the cactus was to recover it.', 4000);
}
function restoreAnchors() {
  if (!stash) return;
  if (cactus) scene.remove(cactus);
  if (floorMesh) scene.remove(floorMesh);
  ({ cactus, floor, floorMesh, downSum, downN } = stash);
  stash = null;
  if (cactus) cactus.visible = true;
  tele('anchors restored (maps merged)');
  hint('Map recovered: cactus restored', 3000);
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
  floor = { n: pl.n, d: pl.d, height, inliers: pl.inliers.length, centroid: pl.centroid };
  if (!floorMesh) {
    floorMesh = new THREE.GridHelper(1, 10, 0x22c55e, 0x22c55e);
    floorMesh.material.transparent = true; floorMesh.material.opacity = 0.35;
    scene.add(floorMesh);
  }
  // the grid is fixed in the world: centred on the floor points that support the plane (not under the camera)
  const k = dot(pl.n, pl.centroid) + pl.d;
  floorMesh.position.set(pl.centroid[0] - pl.n[0] * k, pl.centroid[1] - pl.n[1] * k, pl.centroid[2] - pl.n[2] * k);
  floorMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(...floor.n));
  floorMesh.scale.setScalar(height * 2);
}

function placeCactus(clientX, clientY) {
  if (!twc || !tracking()) return;
  const u = (clientX - rect.x) / rect.s, v = (clientY - rect.y) / rect.s;  // image pixel
  const dir = normalize(mulR(rotOf(twc), [(u - W / 2) / F, (v - H / 2) / F, 1]));
  const o = [twc[12], twc[13], twc[14]];
  let plane = floor, size;
  if (plane) {
    size = 0.25 * plane.height;  // ~25% of the camera height above the floor
  } else {  // no gravity: fit a plane to the local map points that project near the tap
    const R = rotOf(twc);
    const near = [], depths = [];
    for (const p of localPts) {
      const d = [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
      const pc = [R[0] * d[0] + R[3] * d[1] + R[6] * d[2], R[1] * d[0] + R[4] * d[1] + R[7] * d[2],
                  R[2] * d[0] + R[5] * d[1] + R[8] * d[2]];  // camera frame: R^T (p - o)
      if (pc[2] <= 0) continue;
      const pu = F * pc[0] / pc[2] + W / 2, pv = F * pc[1] / pc[2] + H / 2;
      if (Math.hypot(pu - u, pv - v) < 0.15 * W) { near.push(p); depths.push(pc[2]); }
    }
    if (near.length < 12) { hint('Not enough map points there. Move a bit and tap again.', 2500); return; }
    depths.sort((a, b) => a - b);
    const zMed = depths[depths.length >> 1];
    const pl = fitPlaneRansac(near, { thresh: 0.03 * zMed, minInliers: 10 });
    if (!pl) { hint('No flat surface found there. Tap a flat, textured area.', 2500); return; }
    if (dot(pl.n, [o[0] - pl.centroid[0], o[1] - pl.centroid[1], o[2] - pl.centroid[2]]) < 0) { pl.n = pl.n.map((x) => -x); pl.d = -pl.d; }
    plane = { n: pl.n, d: pl.d };
    size = 0.15 * zMed;
    const hitTest = rayPlane(o, dir, plane);
    const depth = hitTest ? dot([hitTest[0] - o[0], hitTest[1] - o[1], hitTest[2] - o[2]], dir) : 0;
    if (!hitTest || depth < 0.4 * zMed || depth > 2.5 * zMed) { hint('That surface is not usable. Tap a flat, textured area.', 2500); return; }
  }
  const hit = rayPlane(o, dir, plane);
  if (!hit || !cactusProto) return;
  if (cactus) scene.remove(cactus);
  cactus = cactusProto.clone();
  cactus.position.set(...hit);
  cactus.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(...plane.n));
  cactus.scale.setScalar(size);  // cactusProto is normalised to height 1 with its base at the origin
  scene.add(cactus);
  tele(`cactus placed: ${floor ? 'floor plane' : 'tap plane'}, distance ${Math.hypot(hit[0] - o[0], hit[1] - o[1], hit[2] - o[2]).toFixed(2)}, size ${size.toFixed(3)} (map units)`);
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
  if (showPoints && kps && tracking()) {
    bgx.fillStyle = '#22c55e';
    for (let i = 0; i < kps.length; i += 3) if (kps[i + 2] === 1) bgx.fillRect(kps[i] - 1.5, kps[i + 1] - 1.5, 3, 3);
  }
  if (twc) setCameraPose(twc);
  if (floorMesh) floorMesh.visible = !cactus && tracking();
  if (cactus) cactus.visible = tracking();  // hidden while lost: the last pose is stale
  renderer.render(scene, cam3);
}

// ---------------------------------------------------------------- main loop
async function start() {
  $('go').disabled = true;
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
      settingsUrl = info.settings; clahe = info.clahe;
      const yaml = await (await fetch(settingsUrl)).text();
      settingsText = yaml;
      F = Number((/Camera1\.fx:\s*([\d.eE+-]+)/.exec(yaml) || [])[1] || 0);
      let i = 0;
      const t0 = performance.now();
      nextBitmap = async () => {  // real-time playback; frames that are already late are skipped
        const now = (performance.now() - t0) / 1000;
        while (i + 1 < man.frames.length && man.frames[i + 1].t <= now) i++;
        if (man.frames[i].t > now) await new Promise((res) => setTimeout(res, (man.frames[i].t - now) * 1000));
        const f = man.frames[Math.min(i++, man.frames.length - 1)];
        return { bitmap: await createImageBitmap(await (await fetch(f.url)).blob()), t: f.t };
      };
      const first = await nextBitmap();
      W = first.bitmap.width; H = first.bitmap.height;
      first.bitmap.close();
      if (!F) F = (Math.max(W, H) / 2) / Math.tan(hfov * Math.PI / 360);
    } else {
      const [rw, rh] = $('res').value.split('x').map(Number);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: rw }, height: { ideal: rh }, frameRate: { ideal: 30 } },
        audio: false,
      });
      const video = Object.assign(document.createElement('video'), { srcObject: stream, muted: true, playsInline: true });
      await video.play();
      W = video.videoWidth; H = video.videoHeight;
      F = (Math.max(W, H) / 2) / Math.tan(hfov * Math.PI / 360);  // the FOV is along the sensor long side (portrait streams too)
      fps = stream.getVideoTracks()[0].getSettings().frameRate || 30;
      settingsText = cameraSettings(W, H, hfov, fps, Number(params.get('features') || 1000));
      const tc = performance.now();
      nextBitmap = async () => ({ bitmap: await createImageBitmap(video), t: (performance.now() - tc) / 1000 });
    }
    layout();
    $('start').hidden = true; $('hud').hidden = $('bar').hidden = false;
    hint('Loading SLAM…');
    new GLTFLoader().load('./assets/cactus.glb', (g) => {  // normalise: height 1, base centre at the origin
      const box = new THREE.Box3().setFromObject(g.scene);
      const size = box.getSize(new THREE.Vector3()), centre = box.getCenter(new THREE.Vector3());
      g.scene.position.set(-centre.x, -box.min.y, -centre.z);
      const wrap = new THREE.Group();
      wrap.add(g.scene);
      wrap.scale.setScalar(1 / (size.y || 1));
      cactusProto = new THREE.Group();
      cactusProto.add(wrap);
    });

    tele(`start: ${navigator.userAgent} | cores ${navigator.hardwareConcurrency} | image ${W}x${H} | f ${F.toFixed(1)} px | ` +
         `hfov ${hfov} | options ${describeOptions(opts)} | source ${seq || 'camera'}`);
    const slam = createSlamWorker((s) => { console.log(s); if (/Web config|ERROR|Map|map|Loading|loaded/.test(s)) tele(`slam: ${s}`); });
    const tInit = performance.now();
    const ready = await slam.request({ type: 'init', vocabUrl: vocabUrl(opts.desc), clahe, seed: 1,
                                       settingsText: withTracker(settingsText) });
    if (ready.type === 'error') throw new Error(ready.text);
    tele(`init ${(performance.now() - tInit).toFixed(0)} ms (vocabulary ${(ready.vocabBytes / 1e6).toFixed(1)} MB), wasm heap ${(ready.heap / 1048576).toFixed(0)} MiB`);
    $('btnReset').onclick = () => {  // applied between frames: the worker handles one request at a time
      resetRequested = true;
      if (stash) { if (stash.cactus) scene.remove(stash.cactus); if (stash.floorMesh) scene.remove(stash.floorMesh); stash = null; }
      resetAnchors('Map reset. Move the phone slowly sideways.');
    };
    $('btnPoints').onclick = () => { showPoints = !showPoints; };
    $('btnClear').onclick = () => { if (cactus) { scene.remove(cactus); cactus = null; } };
    $('stage').addEventListener('pointerdown', (e) => placeCactus(e.clientX, e.clientY));

    const t0 = performance.now();
    const times = [], trackTimes = [];
    let n = 0;
    for (;;) {
      if (resetRequested) {
        resetRequested = false;
        await slam.request({ type: 'reset' });
        tele('map reset requested');
      }
      const tf = performance.now();
      const { bitmap, t } = await nextBitmap();
      const r = await slam.request({ type: 'bitmap', bitmap, t, viz: true, withMap: n % 10 === 0 }, [bitmap]);
      if (r.type !== 'pose') throw new Error(r.text || 'tracking error');
      n++;
      state = r.state; stats = r.viz.stats; kps = r.viz.kps;
      if (lastMaps && stats[3] > lastMaps) stashAnchors();              // lost -> ORB-SLAM3 started a new map
      else if (lastMaps && stats[3] < lastMaps) restoreAnchors();       // maps merged back into the old one
      else if (lastState >= 2 && state === 1 && !stash) resetAnchors('Tracking restarted. Move the phone slowly sideways.');
      lastState = state; lastMaps = stats[3];
      if (tracking()) {
        twc = r.twc;
        const g = gravityCam();
        if (g) { const gw = mulR(rotOf(twc), g); downSum = downSum.map((x, i) => x + gw[i]); downN++; }
      }
      if (r.viz.mps) {
        localPts = [];
        for (let i = 0; i < r.viz.mps.length; i += 4) if (r.viz.mps[i + 3] > 0) localPts.push([r.viz.mps[i], r.viz.mps[i + 1], r.viz.mps[i + 2]]);
        if (tracking()) detectFloor();
      }
      drawFrame(r.bmp);
      times.push(performance.now() - tf);
      if (times.length > 30) times.shift();
      if (state === 1) hint('Move the phone slowly sideways to start tracking');
      else if (state === 3 || state === 4) hint('Tracking lost: move back towards where you were');
      else if (tracking() && !cactus) hint(floor ? 'Tap the floor to place the cactus' : (downN ? 'Point at the floor…' : 'Tap a flat surface to place the cactus'));
      else if (tracking() && cactus && hintText) hint('');
      if (n % 5 === 0) {
        const fpsNow = 1000 / (times.reduce((a, b) => a + b, 0) / times.length);
        const mhz = motionEvents / ((performance.now() - t0) / 1000);
        $('hud').innerHTML = `<b>${STATE[state]}</b> · ${fpsNow.toFixed(1)} fps · track ${r.trackMs.toFixed(0)} ms · ` +
          `KFs ${stats[0]} · points ${stats[2]} · floor ${floor ? 'yes' : 'no'} · gravity ${mhz ? mhz.toFixed(0) + ' Hz' : 'n/a'} · ${W}×${H}`;
      }
      trackTimes.push(r.trackMs);
      if (n % 30 === 0) {
        const s = [...trackTimes].sort((a, b) => a - b);
        const fpsNow = 1000 / (times.reduce((a, b) => a + b, 0) / times.length);
        tele(`frame ${n}: ${STATE[state]} | ${fpsNow.toFixed(1)} fps | track ms med ${s[s.length >> 1].toFixed(1)} p95 ${s[Math.floor(s.length * 0.95)].toFixed(1)} | ` +
             `KFs ${stats[0]} MPs ${stats[1]} matches ${stats[2]} maps ${stats[3]} | floor ${floor ? `yes (cam height ${floor.height.toFixed(2)}, ${floor.inliers} inl)` : 'no'} | ` +
             `gravity ${(motionEvents / ((performance.now() - t0) / 1000)).toFixed(0)} Hz | cactus ${cactus ? 'placed' : 'no'}`);
        trackTimes.length = 0;
      }
    }
  } catch (e) {
    $('start').hidden = false; $('go').disabled = false;
    $('err').textContent = String(e.message || e);
    tele(`ERROR ${e.stack || e}`);
  }
}

// Telemetry for remote testing: status lines are POSTed to the dev server every 5 s
// (H:\Research\results\browser\ar_<start time>.log). Only when served by web/serve.py on localhost.
const teleLines = [];
const teleName = `ar_${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}.log`;
function tele(line) { teleLines.push(`${new Date().toISOString().slice(11, 19)} ${line}`); }
setInterval(() => {
  if (!teleLines.length || location.hostname !== 'localhost') return;
  // sendBeacon: fire-and-forget, does not hold a connection (a fetch over adb reverse could stall)
  navigator.sendBeacon(`/api/result?name=${teleName}`, teleLines.join('\n') + '\n');
}, 5000);
addEventListener('error', (e) => tele(`ERROR ${e.message} @ ${e.filename}:${e.lineno}`));
addEventListener('unhandledrejection', (e) => tele(`ERROR (promise) ${e.reason && (e.reason.stack || e.reason)}`));

// tracker=klt: ORB-SLAM3 with KLT tracking frames between ORB frames (Web.KLT.*, include/WebConfig.h)
function withTracker(yaml) {  // applies the selected options (web/options.js)
  return applyOptions(yaml, opts);
}

$('go').onclick = start;
// debugging handle (console): scene graph, camera, anchors
window.__ar = { scene, cam3, renderer, get cactus() { return cactus; }, get floor() { return floor; }, get twc() { return twc; },
                get rect() { return rect; }, get F() { return F; } };
if (!crossOriginIsolated) $('err').textContent = 'Not cross-origin isolated: open through web/serve.py (WASM threads need COOP/COEP).';
