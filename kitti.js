import { initRenderer, drawFrame, drawPoints, updateHUDText, drawPathXZ, drawAttitude } from './renderer.js';

// ===== utilities ============================================================
function ensureLogEl() {
  const d = document.getElementById('log');
  return d;
}
const _logEl = ensureLogEl();
function logMsg(...args) {
  const line = args.map(x => (typeof x === 'object' ? JSON.stringify(x) : String(x))).join(' ');
  console.log(...args);
  _logEl.textContent += line + '\n';
  _logEl.scrollTop = _logEl.scrollHeight;
}

function byName(a, b) {
  return a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' });
}

function findFile(fileList, suffix) {
  const s = suffix.replace(/\\/g, '/');
  for (const f of fileList) {
    const p = (f.webkitRelativePath || f.name).replace(/\\/g, '/');
    if (p.endsWith(s)) return f;
  }
  return null;
}

function findFilesUnder(fileList, dirPrefix, exts) {
  const pref = dirPrefix.replace(/\\/g, '/').replace(/\/+$/,'') + '/';
  const out = [];
  for (const f of fileList) {
    const p = (f.webkitRelativePath || f.name).replace(/\\/g, '/');
    if (!p.includes(pref)) continue;
    const lower = p.toLowerCase();
    if (exts.some(e => lower.endsWith(e))) out.push(f);
  }
  out.sort((a,b)=>{
    const pa = (a.webkitRelativePath || a.name).replace(/\\/g,'/');
    const pb = (b.webkitRelativePath || b.name).replace(/\\/g,'/');
    return pa.localeCompare(pb, undefined, { numeric:true, sensitivity:'base' });
  });
  return out;
}

async function readText(file) {
  return await file.text();
}

function parseTimestamps(txt) {
  // KITTI raw timestamps: one per line, like "2011-09-26 13:02:39.123456789"
  // We convert to seconds relative to first.
  const lines = txt.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  if (!lines.length) return [];
  const t0 = Date.parse(lines[0].replace(' ', 'T') + 'Z');
  // Date.parse loses sub-ms. We'll keep relative using the string fractional part.
  function toSec(line) {
    const [datePart, timePart] = line.split(' ');
    const [hhmmss, frac=''] = timePart.split('.');
    const baseMs = Date.parse(`${datePart}T${hhmmss}Z`);
    const fracSec = frac ? Number('0.' + frac) : 0;
    return (baseMs - t0) * 1e-3 + fracSec;
  }
  const arr = lines.map(toSec);
  // normalize to 0
  const first = arr[0];
  return arr.map(x => x - first);
}

function parseOxtsLine(line) {
  // KITTI raw oxts/data line is 30 values.
  // We take ax ay az and wx wy wz (both in body frame) from indices:
  // ax ay az: 11,12,13
  // wx wy wz: 17,18,19
  // (Using the standard KITTI raw OXTS spec.)
  const v = line.trim().split(/\s+/).map(Number);
  if (v.length < 20 || v.some(x => !Number.isFinite(x))) return null;
  return {
    ax: v[11], ay: v[12], az: v[13],
    wx: v[17], wy: v[18], wz: v[19],
  };
}

// ===== WASM setup ===========================================================
async function loadWasm() {
  const ts = Date.now();
  const { default: createModule } = await import(`./vio_wasm.js?v=${ts}`);
  const Module = await createModule({ locateFile: (p)=> p.endsWith('.wasm') ? `./vio_wasm.wasm?v=${ts}` : p });
  return Module;
}

function allocGrayHeap(Module, nBytes) {
  if (!Module.HEAPU8 || !Module._malloc) return { ptr: 0, view: null };
  const ptr = Module._malloc(nBytes);
  const view = new Uint8Array(Module.HEAPU8.buffer, ptr, nBytes);
  return { ptr, view };
}

// ===== UI state =============================================================
const els = {
  dirPick: document.getElementById('dirPick'),
  btnScan: document.getElementById('btnScan'),
  btnLoad: document.getElementById('btnLoad'),
  btnRun: document.getElementById('btnRun'),
  btnStop: document.getElementById('btnStop'),
  btnExport: document.getElementById('btnExport'),
  imgDir: document.getElementById('imgDir'),
  oxtsDir: document.getElementById('oxtsDir'),
  seqInfo: document.getElementById('seqInfo'),
  speed: document.getElementById('speed'),
  skip: document.getElementById('skip'),
  maxFrames: document.getElementById('maxFrames'),
};

let fileList = [];
let seq = null;
let Module = null;

let stopFlag = false;
let recorded = []; // per-frame telemetry for export

// ===== boot renderer and wasm ==============================================
const canvas = document.getElementById('view');
await initRenderer(canvas);
Module = await loadWasm();
window.Module = Module;
logMsg('WASM ready:', {
  feedFramePtr: !!Module.feedFramePtr,
  feedFrameJS: !!Module.feedFrameJS,
  feedImuSample: !!Module.feedImuSample,
  getTwc: !!Module.getTwc,
  getYPR: !!Module.getYPR,
});

// Default: you said you want KLT only going forward
try { Module.setTrackerType?.(0); } catch {}

// ===== interactions =========================================================
els.dirPick.addEventListener('change', () => {
  fileList = Array.from(els.dirPick.files || []);
  els.btnScan.disabled = fileList.length === 0;
  els.btnLoad.disabled = true;
  els.btnRun.disabled = true;
  els.btnExport.disabled = true;
  seq = null;
  recorded = [];
  updateHUDText('');
  logMsg(`Picked ${fileList.length} files`);
});

els.btnScan.addEventListener('click', () => {
  if (!fileList.length) return;

  // Quick presence checks
  const imgDir = els.imgDir.value.trim();
  const oxtsDir = els.oxtsDir.value.trim();

  const imgs = findFilesUnder(fileList, imgDir, ['.png', '.jpg', '.jpeg']);
  const oxtsTxt = findFilesUnder(fileList, oxtsDir, ['.txt']);

  logMsg('Scan:', {
    imgDir, imgCount: imgs.length,
    oxtsDir, oxtsCount: oxtsTxt.length,
  });

  els.btnLoad.disabled = imgs.length === 0;
  els.seqInfo.textContent = imgs.length ? `Found ${imgs.length} images. Found ${oxtsTxt.length} OXTS files.` : 'No images found. Check Image dir path.';
});

els.btnLoad.addEventListener('click', async () => {
  if (!fileList.length) return;

  const imgDir = els.imgDir.value.trim();
  const oxtsDir = els.oxtsDir.value.trim();

  const images = findFilesUnder(fileList, imgDir, ['.png', '.jpg', '.jpeg']);
  if (!images.length) {
    els.seqInfo.textContent = 'No images found. Fix Image dir path.';
    return;
  }

  // timestamps.txt live one level above the "data" folder
  const imgBase  = imgDir.replace(/\/+$/,'').replace(/\/data$/,'');   // "image_02"
  const oxtsBase = oxtsDir.replace(/\/+$/,'').replace(/\/data$/,'');  // "oxts"

  const tsImgFile  = findFile(fileList, `${imgBase}/timestamps.txt`);
  const tsOxtsFile = findFile(fileList, `${oxtsBase}/timestamps.txt`);

  let imageTS = [];
  if (tsImgFile) {
    imageTS = parseTimestamps(await readText(tsImgFile));
  } else if (tsOxtsFile) {
    imageTS = parseTimestamps(await readText(tsOxtsFile));
    logMsg('Using OXTS timestamps for images (image timestamps missing).');
  } else {
    imageTS = images.map((_, i) => i * 0.1);
    logMsg('No image/oxts timestamps.txt found; using synthetic 10Hz timestamps');
  }

  let oxtsTS = [];
  if (tsOxtsFile) {
    oxtsTS = parseTimestamps(await readText(tsOxtsFile));
  } else {
    logMsg('No OXTS timestamps.txt found; IMU will be aligned by index (less accurate).');
  }

  // --- OXTS samples (IMU) ---
  const oxtsFiles = findFilesUnder(fileList, oxtsDir, ['.txt']);
  oxtsFiles.sort(byName);

  let oxts = [];
  if (oxtsFiles.length) {
    oxts = await Promise.all(oxtsFiles.map(async f => {
      const line = (await readText(f)).trim().split(/\r?\n/)[0] || '';
      return parseOxtsLine(line);
    }));
  }

  // --- Build an IMU stream (time, accel, gyro) ---
  // If oxtsTS exists, we use it. Otherwise assume 100 Hz (0.01s) as fallback.
  const imuStream = [];
  {
    const M = Math.min(oxts.length, oxtsFiles.length, (oxtsTS.length ? oxtsTS.length : oxts.length));
    for (let k = 0; k < M; k++) {
      const s = oxts[k];
      if (!s) continue;
      const t = (oxtsTS.length ? oxtsTS[k] : (k * 0.01));
      imuStream.push({ t, ax: s.ax, ay: s.ay, az: s.az, wx: s.wx, wy: s.wy, wz: s.wz });
    }
  }

  // --- Clip to common length for images ---
  const N = Math.min(images.length, imageTS.length || images.length);

  seq = { images, imageTS, oxts, oxtsTS, imuStream, N };

  // Init canvas to first frame size
  const bmp = await createImageBitmap(images[0]);
  canvas.width = bmp.width;
  canvas.height = bmp.height;
  bmp.close?.();

  // ---- Load KITTI calib texts (image_02 left camera) and pass into WASM ----
  // Expect these files somewhere in the picked folder root (or sequence root):
  const cam2cam = findFile(fileList, 'calib_cam_to_cam.txt');
  const velo2cam = findFile(fileList, 'calib_velo_to_cam.txt');
  const imu2velo = findFile(fileList, 'calib_imu_to_velo.txt');

  if (cam2cam && velo2cam && imu2velo && Module.setKittiCalibFromTexts) {
    const cam2camTxt = await readText(cam2cam);
    const velo2camTxt = await readText(velo2cam);
    const imu2veloTxt = await readText(imu2velo);

    const ok = Module.setKittiCalibFromTexts(cam2camTxt, velo2camTxt, imu2veloTxt);
    logMsg('KITTI calib applied:', ok ? 'YES' : 'NO');
  } else {
    logMsg('KITTI calib files not found or binding missing; using fallback intrinsics');
  }

  // ---- Init System ----
  // If KITTI calib was applied, System now has correct fx/fy/cx/cy internally.
  // But init still needs image size, so we call it with any intrinsics; they get overwritten by calib anyway.
  const FOVY = 45;
  const fy = canvas.height / (2 * Math.tan((FOVY * Math.PI/180) / 2));
  const fx = fy * (canvas.width / canvas.height);
  const cx = canvas.width * 0.5;
  const cy = canvas.height * 0.5;
  Module.initSystem(canvas.width, canvas.height, fx, fy, cx, cy);

  els.seqInfo.textContent = `Loaded: ${N} frames (${canvas.width}x${canvas.height}).`;
  els.btnRun.disabled = false;
  els.btnExport.disabled = true;
  recorded = [];

  logMsg('Sequence loaded:', { N, w: canvas.width, h: canvas.height, hasOxts: !!oxts.length });
});

els.btnRun.addEventListener('click', async () => {
  if (!seq) return;
  stopFlag = false;
  els.btnStop.disabled = false;
  els.btnRun.disabled = true;
  els.btnExport.disabled = true;
  recorded = [];

  const speed = Math.max(0.05, Number(els.speed.value || 1));
  const skip = Math.max(0, Number(els.skip.value || 0) | 0);
  const maxFrames = Math.max(0, Number(els.maxFrames.value || 0) | 0);

  // 2D ctx for decoding images -> gray
  const off = new OffscreenCanvas(canvas.width, canvas.height);
  const ctx = off.getContext('2d', { willReadFrequently: true });

  const grayN = canvas.width * canvas.height;
  const { ptr: wasmPtr, view: wasmView } = allocGrayHeap(Module, grayN);
  const gray = new Uint8Array(grayN);

  const start = performance.now();
  let lastUI = performance.now();

  const N = seq.N;
  const startIdx = Math.min(N-1, skip);
  const limit = (maxFrames > 0) ? Math.min(N, startIdx + maxFrames) : N;

  // helper: feed ALL IMU samples in (tPrev, tNow]
  // This is required for tight VIO: preintegration needs the high-rate stream.
  let imuIdx = 0;

  function feedImuWindow(tPrev, tNow) {
    if (!Module.feedImuSample) return;
  
    const S = seq.imuStream || [];
    if (!S.length) {
      Module.feedImuSample(tNow, 0,0,0, 0,0,0);
      return;
    }
  
    while (imuIdx < S.length && S[imuIdx].t < tPrev) imuIdx++;
  
    let fed = 0;
    while (imuIdx < S.length && S[imuIdx].t <= tNow) {
      const s = S[imuIdx++];
      Module.feedImuSample(s.t, s.ax, s.ay, s.az, s.wx, s.wy, s.wz);
      fed++;
    }
  
    // If no sample fell inside the window, feed the closest previous sample once.
    if (fed === 0) {
      const k = Math.max(0, imuIdx - 1);
      const s = S[k];
      Module.feedImuSample(tNow, s.ax, s.ay, s.az, s.wx, s.wy, s.wz);
    }
  }  

  for (let i = startIdx; i < limit; i++) {
    if (stopFlag) break;

    const tRel = seq.imageTS[i] ?? (i * 0.1);
    // Map dataset time to an arbitrary "session" time base (seconds)
    const tNow = tRel;

    // Decode image
    const bmp = await createImageBitmap(seq.images[i]);

    // 1) Draw to visible canvas (so you see the sequence)
    drawFrame(bmp, canvas.width, canvas.height);
    
    // 2) Draw to offscreen canvas (so we can extract pixels for WASM)
    ctx.drawImage(bmp, 0, 0);
    
    // Now we can release it
    bmp.close?.();
    
    // RGBA -> gray
    const img = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let j = 0;
    for (let k = 0; k < img.length; k += 4) {
      gray[j++] = (77*img[k] + 150*img[k+1] + 29*img[k+2]) >> 8;
    }

    // Feed IMU window (tight VIO needs multiple samples between frames)
    const tPrev = (i > startIdx) ? (seq.imageTS[i - 1] ?? (tNow - 0.1)) : (tNow - 0.1);
    feedImuWindow(tPrev, tNow);

    // Feed frame to WASM (prefer ptr path)
    if (wasmView && wasmPtr && Module.feedFramePtr) {
      wasmView.set(gray);
      Module.feedFramePtr(wasmPtr, tNow, canvas.width, canvas.height, false);
    } else {
      Module.feedFrameJS(gray, tNow, canvas.width, canvas.height, false);
    }

    // Draw points
    let pts = [];
    try {
      const arr = Module.getPoints2D?.();
      if (arr && arr.length) pts = arr;
    } catch {}
    drawPoints(pts, canvas.width, canvas.height);

    // Attitude + path overlays (optional)
    try {
      const ypr = Module.getYPR?.();
      if (ypr && ypr.length === 3) {
        const RAD2DEG = 180 / Math.PI;
        drawAttitude(Number(ypr[0]) * RAD2DEG, Number(ypr[1]) * RAD2DEG, Number(ypr[2]) * RAD2DEG, canvas.width, canvas.height);
      }
    } catch {}

    try {
      const pathXZ = Module.getPathXZ?.();
      if (pathXZ && pathXZ.length) drawPathXZ(pathXZ, canvas.width, canvas.height);
    } catch {}

    // Record pose
    let twc = [0,0,0];
    let ypr = [0,0,0];
    try { twc = Module.getTwc?.() || twc; } catch {}
    try { ypr = Module.getYPR?.() || ypr; } catch {}

    // Convert from VO world (OpenCV-ish: x right, y down, z forward)
    // to an ENU-like export where:
    //   x = east-ish  (use VO x)
    //   y = north-ish (use VO z)
    //   z = up-ish    (use -VO y)
    const x = Number(twc[1]);
    const y = -Number(twc[2]);
    const z = Number(twc[0]);
    
    recorded.push({
      frame: i,
      t: tNow,
      x, y, z,
      yaw: Number(ypr[0]), pitch: Number(ypr[1]), roll: Number(ypr[2]),
    });

    // HUD update (throttled)
    const now = performance.now();
    if (now - lastUI > 80) {
      const wasmTotal = Number(Module.getLastTotalMS?.() ?? 0);
      const wasmKLT = Number(Module.getLastKltMS?.() ?? 0);
      const imuUsed = Number(Module.getImuUsedThisFrame?.() ?? 0);
      const imuHz = Number(Module.getImuHz?.() ?? 0);
      const mps = Number(Module.getNumMPs?.() ?? 0);
      const kps = Number(Module.getNumKFs?.() ?? 0);
      updateHUDText([
        `Frame      ${i}/${limit-1}`,
        `t (s)      ${tNow.toFixed(3)}`,
        `WASM total ${wasmTotal.toFixed(2)} ms`,
        `WASM KLT   ${wasmKLT.toFixed(2)} ms`,
        `IMU used   ${imuUsed ? 'YES' : 'NO'}`,
        `IMU Hz     ${imuHz ? imuHz.toFixed(1) : 'NA'}`,
        `Pos        ${Number(twc[0]).toFixed(3)} ${Number(twc[1]).toFixed(3)} ${Number(twc[2]).toFixed(3)}`,
        'MPs        ' + mps,
        'KFs        ' + kps,
      ].join('\n'));
      lastUI = now;
    }

    // Playback timing (dataset dt scaled by speed)
    // We approximate sleep so the browser remains responsive.
    const nextT = seq.imageTS[i+1] ?? (tNow + 0.1);
    const dtDataset = Math.max(0, nextT - tNow);
    const dtMs = (dtDataset / speed) * 1000;
    if (dtMs > 1) await new Promise(r => setTimeout(r, Math.min(25, dtMs)));
  }

  els.btnStop.disabled = true;
  els.btnRun.disabled = false;
  els.btnExport.disabled = recorded.length === 0;

  logMsg(`Run finished. Recorded ${recorded.length} frames.`);
});

els.btnStop.addEventListener('click', () => {
  stopFlag = true;
  els.btnStop.disabled = true;
  els.btnRun.disabled = false;
  logMsg('Stop requested.');
});

els.btnExport.addEventListener('click', () => {
  if (!recorded.length) return;

  // Keep tiny numbers alive: use scientific notation with lots of digits.
  function fmt(x) {
    return Number.isFinite(x) ? Number(x).toExponential(16) : 'nan';
  }
  // (A) Export main pose CSV
  const lines = ['frame,t,x,y,z,yaw_rad,pitch_rad,roll_rad'];
  for (const r of recorded) {
    lines.push([
      r.frame,
      fmt(r.t),
      fmt(r.x), fmt(r.y), fmt(r.z),
      fmt(r.yaw), fmt(r.pitch), fmt(r.roll),
    ].join(','));
  }

  // Download main CSV
  {
    const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'kitti_poses.csv';
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  logMsg('Exported kitti_poses.csv');
});

