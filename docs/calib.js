// Camera calibration for the AR page: chessboard views from the phone's camera stream (same resolution request as
// ar.html), corners and OpenCV calibrateCamera in the WASM worker (bindings.cc: Calibrator). The result (pinhole +
// k1 k2 p1 p2, ORB-SLAM3's "PinHole" model) is saved in this browser's localStorage for the stream size; ar.html
// uses it instead of the assumed field of view.
//
// URL parameters: board=<cols>x<rows> shows a chessboard to film (open on a laptop / monitor), cols=, rows=, res=

import { createSlamWorker, saveCalibration, savedCalibrations } from './slamclient.js';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
if (params.get('cols')) $('cols').value = params.get('cols');
if (params.get('rows')) $('rows').value = params.get('rows');
if (params.get('res')) $('res').value = params.get('res');

const TARGET = 20, MIN_VIEWS = 10;
const THR = [0.12, 0.12, 0.08, 0.12, 0.12];  // "new view": centre x, y (image fraction), size, tilt left/right, top/bottom
const GRID = [6, 8];                          // coverage cells (columns, rows) of the image

// ---------------------------------------------------------------- chessboard display (on another screen)
function showBoard(cols, rows) {
  const c = $('board');
  c.hidden = false;
  $('start').hidden = true;
  const draw = () => {
    const dpr = devicePixelRatio || 1;
    c.width = Math.round(innerWidth * dpr); c.height = Math.round(innerHeight * dpr);
    c.style.width = `${innerWidth}px`; c.style.height = `${innerHeight}px`;
    const g = c.getContext('2d');
    g.fillStyle = '#fff'; g.fillRect(0, 0, c.width, c.height);
    const nx = cols + 1, ny = rows + 1;  // squares; a white margin of at least one square (needed by the detector)
    const sq = Math.floor(Math.min(c.width / (nx + 2), c.height / (ny + 2)));
    const x0 = Math.floor((c.width - nx * sq) / 2), y0 = Math.floor((c.height - ny * sq) / 2);
    g.fillStyle = '#000';
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) if ((i + j) % 2 === 0) g.fillRect(x0 + i * sq, y0 + j * sq, sq, sq);
  };
  draw();
  addEventListener('resize', draw);
  c.onclick = () => { if (document.documentElement.requestFullscreen) document.documentElement.requestFullscreen().catch(() => {}); };
}
const boardParam = /^(\d+)x(\d+)$/.exec(params.get('board') || '');
if (boardParam) showBoard(Number(boardParam[1]), Number(boardParam[2]));
$('showBoard').onclick = () => {
  const cols = Number($('cols').value), rows = Number($('rows').value);
  history.replaceState(null, '', `?board=${cols}x${rows}`);
  showBoard(cols, rows);
};

{  // calibrations already saved in this browser
  const saved = savedCalibrations();
  $('saved').textContent = saved.length
    ? `Saved: ${saved.map((c) => `${c.w}×${c.h} (${c.date.slice(0, 10)}, ${c.rms.toFixed(2)} px)`).join(', ')}. A new calibration replaces the one for its size.`
    : '';
}

// ---------------------------------------------------------------- capture
const view = $('view'), vx = view.getContext('2d');
let W = 640, H = 480, cols = 9, rows = 6, rect = { x: 0, y: 0, s: 1 };
let slam = null, stream = null, running = false;
let caps = [], prev = null, coverage = null, nViews = 0, result = null;
let solveRequested = false, restartRequested = false;  // buttons: handled between frames (one worker request at a time)

function layout() {  // cover-fit the camera image into the window
  const s = Math.max(innerWidth / W, innerHeight / H);
  rect = { s, x: (innerWidth - W * s) / 2, y: (innerHeight - H * s) / 2 };
  Object.assign(view.style, { left: `${rect.x}px`, top: `${rect.y}px`, width: `${W * s}px`, height: `${H * s}px` });
  view.width = W; view.height = H;
}
addEventListener('resize', () => { if (running) layout(); });

const pt = (c, i) => [c[2 * i], c[2 * i + 1]];
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
function features(c) {  // board pose in the image: centre, size, perspective tilt (both axes)
  const n = c.length / 2;
  let sx = 0, sy = 0;
  for (let i = 0; i < n; i++) { sx += c[2 * i]; sy += c[2 * i + 1]; }
  const p00 = pt(c, 0), p10 = pt(c, cols - 1), p01 = pt(c, (rows - 1) * cols), p11 = pt(c, n - 1);
  const quad = [p00, p10, p11, p01];
  let area = 0;
  for (let i = 0; i < 4; i++) { const a = quad[i], b = quad[(i + 1) % 4]; area += a[0] * b[1] - b[0] * a[1]; }
  return [sx / n / W, sy / n / H, Math.sqrt(Math.abs(area) / 2) / Math.max(W, H),
          Math.log(dist(p00, p01) / dist(p10, p11)), Math.log(dist(p00, p10) / dist(p01, p11))];
}
const isNew = (f) => caps.every((g) => f.some((v, i) => Math.abs(v - g[i]) > THR[i]));
function still(c) {  // the board barely moved since the previous frame: little motion blur
  if (!prev || prev.length !== c.length) return false;
  let d = 0;
  for (let i = 0; i < c.length; i += 2) d += Math.hypot(c[i] - prev[i], c[i + 1] - prev[i + 1]);
  return d / (c.length / 2) < 1.5;
}

function draw(bmp, c, flash) {
  vx.drawImage(bmp, 0, 0);
  bmp.close();
  const cw = W / GRID[0], ch = H / GRID[1];  // coverage: cells that corners of captured views fell into
  vx.fillStyle = 'rgba(34,197,94,0.22)';
  for (let j = 0; j < GRID[1]; j++) for (let i = 0; i < GRID[0]; i++) if (coverage[j * GRID[0] + i]) vx.fillRect(i * cw, j * ch, cw, ch);
  if (c.length) {
    vx.lineWidth = 2;
    for (let r = 0; r < rows; r++) {
      vx.strokeStyle = `hsl(${(r * 360) / rows}, 90%, 55%)`;
      vx.beginPath();
      for (let k = 0; k < cols; k++) { const [x, y] = pt(c, r * cols + k); if (k) vx.lineTo(x, y); else vx.moveTo(x, y); }
      vx.stroke();
    }
  }
  if (flash) { vx.fillStyle = 'rgba(255,255,255,0.35)'; vx.fillRect(0, 0, W, H); }
}

async function start() {
  $('go').disabled = true;
  $('err').textContent = '';
  try {
    cols = Number($('cols').value); rows = Number($('rows').value);
    const [rw, rh] = $('res').value.split('x').map(Number);
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: rw }, height: { ideal: rh }, frameRate: { ideal: 30 } },
      audio: false,
    });
    const video = Object.assign(document.createElement('video'), { srcObject: stream, muted: true, playsInline: true });
    await video.play();
    W = video.videoWidth; H = video.videoHeight;
    layout();
    if (!slam) {
      slam = createSlamWorker((s) => console.log(s));
      const r = await slam.request({ type: 'calibInit' });
      if (r.type === 'error') throw new Error(r.text);
    }
    await slam.request({ type: 'calibClear' });
    caps = []; prev = null; nViews = 0; coverage = new Uint8Array(GRID[0] * GRID[1]);
    $('start').hidden = true; $('hud').hidden = $('bar').hidden = false;
    running = true; solveRequested = restartRequested = false;
    let ms = 0;
    while (running) {
      if (restartRequested) {
        restartRequested = false;
        await slam.request({ type: 'calibClear' });
        caps = []; prev = null; nViews = 0; coverage.fill(0);
      }
      const bitmap = await createImageBitmap(video);
      const r = await slam.request({ type: 'calibFrame', bitmap, cols, rows }, [bitmap]);
      if (r.type !== 'corners') throw new Error(r.text || 'detection error');
      ms = 0.8 * ms + 0.2 * r.ms;
      const c = r.corners;
      let hint = 'Point the camera at the chessboard (the whole board must be visible)';
      let flash = false;
      if (c.length) {
        const f = features(c);
        if (!isNew(f)) hint = 'Move to a new angle, distance or part of the picture';
        else if (!still(c)) hint = 'Hold still…';
        else {
          nViews = (await slam.request({ type: 'calibAdd' })).n;
          caps.push(f);
          for (let i = 0; i < c.length; i += 2) {
            const gx = Math.min(GRID[0] - 1, Math.floor(c[i] / W * GRID[0])), gy = Math.min(GRID[1] - 1, Math.floor(c[i + 1] / H * GRID[1]));
            coverage[gy * GRID[0] + gx] = 1;
          }
          flash = true;
          hint = 'Captured';
        }
      }
      prev = c.length ? c : null;
      draw(r.bmp, c, flash);
      const cov = coverage.reduce((a, b) => a + b, 0) / coverage.length;
      $('hud').innerHTML = `<b>${nViews} / ${TARGET} views</b> · picture covered ${(cov * 100).toFixed(0)}% · ` +
        `detect ${ms.toFixed(0)} ms · ${W}×${H}<br>${hint}${nViews >= MIN_VIEWS && cov < 0.6 ? ' · <span class="warn">cover the edges and corners too</span>' : ''}`;
      $('btnSolve').disabled = nViews < MIN_VIEWS;
      if (nViews >= TARGET || solveRequested) await solve();
    }
  } catch (e) {
    running = false;
    $('start').hidden = false; $('hud').hidden = $('bar').hidden = true;
    $('err').textContent = String(e.message || e);
  } finally {
    $('go').disabled = false;
  }
}

async function solve() {
  running = false;
  $('hud').innerHTML = '<b>Calibrating…</b>';
  let r = (await slam.request({ type: 'calibSolve', cols, rows })).r;
  if (r && r.length > 9) {  // drop views that fit much worse than the rest (blur, a misdetection), then solve again
    const per = r.slice(9), med = [...per].sort((a, b) => a - b)[per.length >> 1];
    const bad = per.map((e, i) => [e, i]).filter(([e]) => e > Math.max(3 * med, 1.0)).map(([, i]) => i);
    if (bad.length && per.length - bad.length >= MIN_VIEWS) {
      for (const i of bad.sort((a, b) => b - a)) await slam.request({ type: 'calibRemove', index: i });
      r = (await slam.request({ type: 'calibSolve', cols, rows })).r;
    }
  }
  if (stream) stream.getTracks().forEach((t) => t.stop());
  $('hud').hidden = $('bar').hidden = true;
  if (!r || r.length < 9) { $('start').hidden = false; $('err').textContent = 'Calibration failed: too few views.'; return; }
  const [rms, fx, fy, cx, cy, k1, k2, p1, p2] = r;
  result = { w: W, h: H, fx, fy, cx, cy, k1, k2, p1, p2, rms, views: r.length - 9, board: `${cols}x${rows}`,
             date: new Date().toISOString(), ua: navigator.userAgent };
  const long = Math.max(W, H), fLong = W >= H ? fx : fy;
  const hfov = 2 * Math.atan(long / 2 / fLong) * 180 / Math.PI;
  const warnings = [];
  if (rms > 1.0) warnings.push(`the reprojection error is high (${rms.toFixed(2)} px): recalibrate with steadier views`);
  if (Math.abs(fx / fy - 1) > 0.03) warnings.push(`fx and fy differ by ${(Math.abs(fx / fy - 1) * 100).toFixed(1)}%`);
  if (Math.abs(cx - W / 2) > 0.1 * W || Math.abs(cy - H / 2) > 0.1 * H) warnings.push('the image centre is far from the middle');
  const ok = saveCalibration(result);
  const row = (k, v) => `<tr><td>${k}</td><td>${v}</td></tr>`;
  $('resTable').innerHTML = row('Image', `${W}×${H}`) + row('Views', result.views) + row('Reprojection error (RMS)', `${rms.toFixed(3)} px`) +
    row('fx, fy', `${fx.toFixed(1)}, ${fy.toFixed(1)} px`) + row('cx, cy', `${cx.toFixed(1)}, ${cy.toFixed(1)} px`) +
    row('k1, k2', `${k1.toFixed(4)}, ${k2.toFixed(4)}`) + row('p1, p2', `${p1.toFixed(5)}, ${p2.toFixed(5)}`) +
    row('Field of view (long side)', `${hfov.toFixed(1)}° (AR assumed 69°)`);
  $('resNote').innerHTML = (ok ? `Saved in this browser for ${W}×${H}: the AR page uses it automatically.` : '<span class="warn">Could not save (storage blocked).</span>') +
    (warnings.length ? `<br><span class="warn">Check: ${warnings.join('; ')}.</span>` : '');
  $('result').hidden = false;
}

// ---------------------------------------------------------------- self-test (?selftest): synthetic views, known camera
// A chessboard rendered in exact perspective (each square is a projected quadrilateral) with known intrinsics and no
// distortion goes through the same worker detection + calibration; the recovered intrinsics must match.
async function selfTest() {
  $('start').hidden = true; $('hud').hidden = false;
  W = 480; H = 640; cols = 9; rows = 6;
  const truth = { fx: 520, fy: 520, cx: 236, cy: 327 };
  layout();
  const cv = new OffscreenCanvas(W, H), g = cv.getContext('2d');
  slam = createSlamWorker((s) => console.log(s));
  await slam.request({ type: 'calibInit' });
  await slam.request({ type: 'calibClear' });
  let seed = 7;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);
  const rot = (ax, ay, az) => {  // R = Rz Ry Rx
    const [cx, sx, cy, sy, cz, sz] = [Math.cos(ax), Math.sin(ax), Math.cos(ay), Math.sin(ay), Math.cos(az), Math.sin(az)];
    return [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx, sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx, -sy, cy * sx, cy * cx];
  };
  let found = 0;
  for (let v = 0; v < 40 && found < 16; v++) {
    const R = rot((rnd() - 0.5) * 1.1, (rnd() - 0.5) * 1.1, (rnd() - 0.5) * 0.6);
    const z = 9 + rnd() * 9;
    const t = [(rnd() - 0.5) * z * 0.5 - 4, (rnd() - 0.5) * z * 0.7 - 2.5, z];
    const proj = (x, y) => {  // board point (x, y, 0) -> pixel
      const X = R[0] * x + R[1] * y + t[0], Y = R[3] * x + R[4] * y + t[1], Z = R[6] * x + R[7] * y + t[2];
      return [truth.fx * X / Z + truth.cx, truth.fy * Y / Z + truth.cy];
    };
    const quad = (x0, y0, x1, y1) => {
      const ps = [proj(x0, y0), proj(x1, y0), proj(x1, y1), proj(x0, y1)];
      g.beginPath(); g.moveTo(...ps[0]); for (let i = 1; i < 4; i++) g.lineTo(...ps[i]); g.closePath(); g.fill();
    };
    g.fillStyle = '#556'; g.fillRect(0, 0, W, H);
    g.fillStyle = '#fff'; quad(-2, -2, cols + 1, rows + 1);  // white margin
    g.fillStyle = '#000';
    for (let j = -1; j < rows; j++) for (let i = -1; i < cols; i++) if ((i + j) % 2 === 0) quad(i, j, i + 1, j + 1);
    const r = await slam.request({ type: 'calibFrame', bitmap: cv.transferToImageBitmap(), cols, rows });
    if (r.corners.length) { found = (await slam.request({ type: 'calibAdd' })).n; }
    coverage = new Uint8Array(GRID[0] * GRID[1]);
    draw(r.bmp, r.corners, false);
    $('hud').innerHTML = `<b>self-test</b> view ${v + 1}: ${r.corners.length ? 'found' : 'not found'} (${r.ms.toFixed(0)} ms) · ${found} views`;
  }
  const r = (await slam.request({ type: 'calibSolve', cols, rows })).r;
  const [rms, fx, fy, cx, cy, k1, k2] = r;
  const err = Math.max(Math.abs(fx - truth.fx), Math.abs(fy - truth.fy), Math.abs(cx - truth.cx), Math.abs(cy - truth.cy));
  const pass = found >= 10 && rms < 0.5 && err < 0.02 * truth.fx && Math.abs(k1) < 0.05 && Math.abs(k2) < 0.2;
  const text = `selftest ${pass ? 'PASS' : 'FAIL'}: ${found} views, rms ${rms.toFixed(3)} px, fx ${fx.toFixed(1)} fy ${fy.toFixed(1)} ` +
               `cx ${cx.toFixed(1)} cy ${cy.toFixed(1)} (truth 520 520 236 327), k1 ${k1.toFixed(4)} k2 ${k2.toFixed(4)}`;
  $('hud').innerHTML = `<b>${text}</b>`;
  console.log(text);
  window.__selftest = { pass, text, r };
}
if (params.has('selftest')) selfTest().catch((e) => { $('hud').textContent = `selftest ERROR ${e.message || e}`; window.__selftest = { pass: false, text: String(e) }; });

$('go').onclick = start;
$('btnSolve').onclick = () => { solveRequested = true; };
$('btnRestart').onclick = () => { restartRequested = true; };
$('btnAgain').onclick = () => { $('result').hidden = true; start(); };
$('btnCopy').onclick = async () => {
  try { await navigator.clipboard.writeText(JSON.stringify(result, null, 1)); $('btnCopy').textContent = 'Copied'; } catch { $('btnCopy').textContent = 'Copy failed'; }
  setTimeout(() => { $('btnCopy').textContent = 'Copy'; }, 2000);
};
if (!crossOriginIsolated) $('err').textContent = 'Not cross-origin isolated: open through web/serve.py or the GitHub Pages site.';
