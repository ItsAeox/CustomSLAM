import { initRenderer, drawFrame, drawPoints, updateHUDText, drawPathXZ, setPathViewConfig, setPointColor } from './renderer.js';
import { loadTumviSequence } from './tumvi.js';

// ===== utilities ============================================================
// function ensureLogEl() {
//   const d = document.getElementById('log');
//   return d;
// }
// const _logEl = ensureLogEl();
function logMsg(...args) {
  const line = args.map(x => (typeof x === 'object' ? JSON.stringify(x) : String(x))).join(' ');
  console.log(...args);
  // _logEl.textContent += line + '\n';
  // _logEl.scrollTop = _logEl.scrollHeight;
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

function parseTimestamps(text) {
  // Return absolute timestamps in seconds (do NOT normalize here).
  // KITTI timestamp lines look like: "2011-09-26 13:02:34.123456789"
  const lines = text.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
  const out = [];
  for (const s of lines) {
    // split date + time
    const parts = s.split(/\s+/);
    if (parts.length < 2) continue;
    const dateStr = parts[0];
    const timeStr = parts[1];

    // timeStr may have nanoseconds: HH:MM:SS.NNNNNNNNN
    const [hms, fracStrRaw = "0"] = timeStr.split(".");
    const [hh, mm, ss] = hms.split(":").map(Number);

    // Build a Date in UTC-like way using the date part, then add h/m/s
    // KITTI timestamps are local time, but we only need consistent relative deltas.
    const base = new Date(dateStr + "T00:00:00");
    const sec = hh * 3600 + mm * 60 + ss;

    // keep sub-second precision
    const frac = Number("0." + fracStrRaw.replace(/[^\d]/g, "").slice(0, 9).padEnd(9, "0"));

    out.push(base.getTime() / 1000 + sec + frac);
  }
  return out;
}


function parseOxtsLine(line) {
  // KITTI raw OXTS/data format:
  //  0 lat  1 lon  2 alt
  //  3 roll 4 pitch 5 yaw
  //  6 vn   7 ve   8 vf   9 vl  10 vu
  // 11 ax  12 ay  13 az
  // 14 af  15 al  16 au
  // 17 wx  18 wy  19 wz
  // 20 wf  21 wl  22 wu
  //
  // For this debug VIO path, use VEHICLE-ALIGNED channels:
  //   accel = af, al, au
  //   gyro  = wf, wl, wu
  // These are much safer than raw ax/ay/az, wx/wy/wz for your current pipeline.
  const v = line.trim().split(/\s+/).map(Number);
  if (v.length < 23 || v.some(x => !Number.isFinite(x))) return null;

  return {
    // vehicle-frame velocity (optional, useful later)
    vf: v[8],
    vl: v[9],
    vu: v[10],

    // vehicle-frame acceleration
    ax: v[14],
    ay: v[15],
    az: v[16],

    // vehicle-frame angular rates
    wx: v[20],
    wy: v[21],
    wz: v[22],

    // keep orientation too for debugging if needed later
    roll: v[3],
    pitch: v[4],
    yaw: v[5],
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
  btnExportPerf: document.getElementById('btnExportPerf'),
  imgDir: document.getElementById('imgDir'),
  oxtsDir: document.getElementById('oxtsDir'),
  seqInfo: document.getElementById('seqInfo'),
  speed: document.getElementById('speed'),
  skip: document.getElementById('skip'),
  maxFrames: document.getElementById('maxFrames'),
  dataset: document.getElementById('dataset'),
  leftDir: document.getElementById('leftDir'),
  rightDir: document.getElementById('rightDir'),
};

let fileList = [];
let seq = null;
let Module = null;

let stopFlag = false;
let recorded = [];     // pose CSV
let recordedPerf = []; // performance CSV

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
  els.btnExportPerf.disabled = true;
  seq = null;
  recorded = [];
  recordedPerf = [];
  updateHUDText('');
  logMsg(`Picked ${fileList.length} files`);
});

els.btnScan.addEventListener('click', () => {
  if (!fileList.length) return;

  const ds = els.dataset?.value || 'kitti';

  if (ds === 'tumvi') {
    const leftDir = (els.leftDir?.value || 'left_images').trim();

    const imgs = findFilesUnder(fileList, leftDir, ['.png', '.jpg', '.jpeg']);
    const ts = findFile(fileList, `${leftDir}/image_timestamps_left.txt`);
    const imu = findFile(fileList, `imu_data.txt`);
    const cam = findFile(fileList, `camera-calibration.json`);

    logMsg('Scan (TUM-VI):', {
      leftDir,
      imgCount: imgs.length,
      hasTimestamps: !!ts,
      hasImu: !!imu,
      hasCalib: !!cam,
    });

    els.btnLoad.disabled = imgs.length === 0 || !ts;
    els.seqInfo.textContent = imgs.length
      ? `Found ${imgs.length} left images. timestamps: ${ts ? 'YES' : 'NO'} imu: ${imu ? 'YES' : 'NO'} calib: ${cam ? 'YES' : 'NO'}`
      : 'No images found. Check Left dir path.';
    return;
  }

  // ---- KITTI scan (original) ----
  const imgDir = els.imgDir.value.trim();
  const oxtsDir = els.oxtsDir.value.trim();

  const imgs = findFilesUnder(fileList, imgDir, ['.png', '.jpg', '.jpeg']);
  const oxtsTxt = findFilesUnder(fileList, oxtsDir, ['.txt']);

  logMsg('Scan (KITTI):', {
    imgDir, imgCount: imgs.length,
    oxtsDir, oxtsCount: oxtsTxt.length,
  });

  els.btnLoad.disabled = imgs.length === 0;
  els.seqInfo.textContent = imgs.length ? `Found ${imgs.length} images. Found ${oxtsTxt.length} OXTS files.` : 'No images found. Check Image dir path.';
});


els.btnLoad.addEventListener('click', async () => {
  if (!fileList.length) return;

  const ds = els.dataset?.value || 'kitti';
  if (ds === 'tumvi') {
    const leftDir  = (els.leftDir?.value || 'left_images').trim();
    const rightDir = (els.rightDir?.value || 'right_images').trim();

    seq = await loadTumviSequence(fileList, { leftDir, rightDir, Module });

    // Init canvas to first frame size (same as KITTI path)
    const bmp = await createImageBitmap(seq.leftImgs[0]);
    canvas.width = bmp.width;
    canvas.height = bmp.height;
    bmp.close?.();

    // Pick a pinhole intrinsics guess from calibration json if present
    // IMPORTANT: KB4 distortion is ignored for now (will hurt edges).
    let fx, fy, cx, cy;
    let cam = null; // <-- ADD THIS
    
    // camera-calibration.json is shaped like { value0: {...} }, so normalize it:
    const calib = seq.calib?.value0 ?? seq.calib;

    if (calib && Array.isArray(calib.intrinsics) && Array.isArray(calib.resolution) && calib.intrinsics.length) {
      const W = canvas.width, H = canvas.height;

      // pick cam index by resolution match (your left/right are 1024x1024)
      const camIdx = 0; // left camera

      cam = calib.intrinsics[camIdx];
      const intr = cam?.intrinsics || {};

      fx = intr.fx; fy = intr.fy; cx = intr.cx; cy = intr.cy;

      logMsg('Using calib intrinsics (pinhole approx):', {
        camIdx, W, H, camera_type: cam?.camera_type, fx, fy, cx, cy
      });
    } else {
      // fallback: your existing fovy-based guess
      const FOVY = 45;
      fy = canvas.height / (2 * Math.tan((FOVY * Math.PI/180) / 2));
      fx = fy * (canvas.width / canvas.height);
      cx = canvas.width * 0.5;
      cy = canvas.height * 0.5;
      logMsg('No usable camera calibration found; using FOV guess intrinsics.');
    }

    // --- KB4 fisheye: pass k1..k4 into WASM before initSystem() ---
    try {
      // Default off unless we detect KB4
      Module.setUseFisheye?.(false);

      const ct = String(cam?.camera_type || '').toLowerCase();

      // Common shapes you might see:
      //  - cam.distortion_parameters = [k1,k2,k3,k4]
      //  - cam.distortion = { parameters:[...] }
      const intrObj = cam?.intrinsics || {};

      // Accept multiple possible shapes, including TUM-VI's: intrinsics.k1..k4
      const dp =
        (Array.isArray(cam?.distortion_parameters) ? cam.distortion_parameters : null) ||
        (Array.isArray(cam?.distortion?.parameters) ? cam.distortion.parameters : null) ||
        (Number.isFinite(intrObj.k1) && Number.isFinite(intrObj.k2) &&
         Number.isFinite(intrObj.k3) && Number.isFinite(intrObj.k4)
           ? [intrObj.k1, intrObj.k2, intrObj.k3, intrObj.k4]
           : null);      

      // Only enable if it's actually KB4 and we have 4 params
      if ((ct.includes('kb4') || ct.includes('fisheye')) && dp && dp.length >= 4) {
        const k1 = Number(dp[0]), k2 = Number(dp[1]), k3 = Number(dp[2]), k4 = Number(dp[3]);
        if ([k1,k2,k3,k4].every(Number.isFinite) && Module.setKb4Distortion) {
          Module.setKb4Distortion(k1, k2, k3, k4);
          Module.setUseFisheye?.(true);
          logMsg('KB4 distortion enabled:', { k1, k2, k3, k4 });
        } else {
          logMsg('KB4 distortion found but invalid params; fisheye disabled.', { dp });
        }
      } else {
        logMsg('No KB4 distortion in calib for selected camera; fisheye disabled.', { camera_type: cam?.camera_type });
      }
    } catch (e) {
      logMsg('KB4 setup error (ignored):', String(e));
    }

    Module.setAccelIsSpecificForce?.(true);   // TUM-VI IMU behaves like normal VIO specific force
    setPathViewConfig(5, 5, '5m x 5m');
    setPointColor('#4da6ff'); // soft blue for grayscale scenes
    Module.initSystem(canvas.width, canvas.height, fx, fy, cx, cy);
    els.seqInfo.textContent = `Loaded TUM-VI: ${seq.N} frames (${canvas.width}x${canvas.height}).`;
    els.btnRun.disabled = false;
    els.btnExport.disabled = true;
    els.btnExportPerf.disabled = true;
    recorded = [];
    recordedPerf = [];

    logMsg('Sequence loaded (TUM-VI):', { N: seq.N, hasImu: !!seq.imuStream?.length, hasMocap: !!seq.mocap?.length });
    return;
  }
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
    // IMPORTANT: rebase BOTH streams to the SAME t0
    // (image timestamps and oxts timestamps can start at slightly different absolute times)
    const tBase = Math.min(imageTS[0], oxtsTS[0]);
    imageTS = imageTS.map(t => t - tBase);
    oxtsTS  = oxtsTS.map(t => t - tBase);
  } else {
    logMsg('No OXTS timestamps.txt found; IMU will be aligned by index (less accurate).');
  }

  // --- OXTS samples (IMU) ---
  const oxtsFiles = findFilesUnder(fileList, oxtsDir, ['.txt']);
  oxtsFiles.sort(byName);

  let oxts = [];
  if (oxtsFiles.length) {
    for (const f of oxtsFiles) {
      try {
        const txt = await readText(f);
        const line = txt.trim().split(/\r?\n/)[0] || '';
        oxts.push(parseOxtsLine(line));
      } catch (e) {
        const p = (f.webkitRelativePath || f.name || '[unknown]').replace(/\\/g, '/');
        console.error('Failed to read OXTS file:', p, e);
        throw e;
      }
    }
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
      imuStream.push({
        t,
        ax: s.ax, ay: s.ay, az: s.az,
        wx: s.wx, wy: s.wy, wz: s.wz,
        vf: s.vf, vl: s.vl, vu: s.vu,
      });
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
  try { Module.setUseFisheye?.(false); } catch {}
  Module.setAccelIsSpecificForce?.(true);    // KITTI OXTS works better with standard VIO specific-force handling
  setPathViewConfig(500, 500, '500m x 500m');
  setPointColor('#ff0000ff');
  Module.initSystem(canvas.width, canvas.height, fx, fy, cx, cy);

  els.seqInfo.textContent = `Loaded: ${N} frames (${canvas.width}x${canvas.height}).`;
  els.btnRun.disabled = false;
  els.btnExport.disabled = true;
  els.btnExportPerf.disabled = true;
  recorded = [];
  recordedPerf = [];

  logMsg('Sequence loaded:', { N, w: canvas.width, h: canvas.height, hasOxts: !!oxts.length });
});

els.btnRun.addEventListener('click', async () => {
  if (!seq) return;
  stopFlag = false;
  els.btnStop.disabled = false;
  els.btnRun.disabled = true;
  els.btnExport.disabled = true;
  els.btnExportPerf.disabled = true;
  recorded = [];
  recordedPerf = [];

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
  let prevTsForRates = null;
  let prevTwcForRates = null;
  let prevYprForRates = null;

  const N = seq.N;
  const startIdx = Math.min(N-1, skip);
  const limit = (maxFrames > 0) ? Math.min(N, startIdx + maxFrames) : N;

  // helper: feed ALL IMU samples in (tPrev, tNow]
  // This is required for tight VIO: preintegration needs the high-rate stream.
  let imuIdx = 0;

  function feedImuWindow(tPrev, tNow) {
    if (!Module.feedImuSample) return;

    const S = seq.imuStream || [];
    if (!S.length) return;

    while (imuIdx < S.length && S[imuIdx].t < tPrev) imuIdx++;

    while (imuIdx < S.length && S[imuIdx].t <= tNow) {
      const s = S[imuIdx++];
      Module.feedImuSample(s.t, s.ax, s.ay, s.az, s.wx, s.wy, s.wz);
    }
  }

  for (let i = startIdx; i < limit; i++) {
    if (stopFlag) break;

    const tRel = seq.imageTS[i] ?? (i * 0.1);
    // Map dataset time to an arbitrary "session" time base (seconds)
    const tNow = tRel;

    // Decode image
    const ds = els.dataset?.value || 'kitti';
    const imgFile = (ds === 'tumvi') ? seq.leftImgs[i] : seq.images[i];
    const bmp = await createImageBitmap(imgFile);
    
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

    // // Attitude + path overlays (optional)
    // try {
    //   const ypr = Module.getYPR?.();
    //   if (ypr && ypr.length === 3) {
    //     const RAD2DEG = 180 / Math.PI;
    //     drawAttitude(Number(ypr[0]) * RAD2DEG, Number(ypr[1]) * RAD2DEG, Number(ypr[2]) * RAD2DEG, canvas.width, canvas.height);
    //   }
    // } catch {}

    try {
      const pathXZ = Module.getPathXZ?.();
      if (pathXZ && pathXZ.length) drawPathXZ(pathXZ, canvas.width, canvas.height);
    } catch {}

    // Record pose
    let twc = [0,0,0];
    let ypr = [0,0,0];
    try { twc = Module.getTwc?.() || twc; } catch {}
    try { ypr = Module.getYPR?.() || ypr; } catch {}

    const x = Number(twc[1]);
    const y = -Number(twc[2]);
    const z = Number(twc[0]);
    
    recorded.push({
      frame: i,
      t: tNow,
      x, y, z,
      yaw: Number(ypr[0]), pitch: Number(ypr[1]), roll: Number(ypr[2]),
    });

    recordedPerf.push({
      frame: i,
      t: tNow,
      num_kfs: Number(Module.getNumKFs?.() ?? 0),
      num_mps: Number(Module.getNumMPs?.() ?? 0),
      num_keypoints: Number(Module.getNumKeypoints?.() ?? 0),
      wasm_klt_ms: Number(Module.getLastKltMS?.() ?? 0),
      wasm_total_ms: Number(Module.getLastTotalMS?.() ?? 0),
      wasm_imu_ms: Number(Module.getLastImuMS?.() ?? 0),
      wasm_seed_ms: Number(Module.getLastSeedMS?.() ?? 0),
      imu_hz: Number(Module.getImuHz?.() ?? 0),
    });

    // HUD update (throttled)
    const now = performance.now();
    if (now - lastUI > 80) {
      const tNow = Number(seq.imageTS[i] ?? (i * 0.1)); // or whatever timestamp you use
      let vStr = 'V (m/s)    NA';
      let wStr = 'W (deg/s)  NA';

      if (prevTsForRates !== null) {
        const dt = Math.max(1e-6, tNow - prevTsForRates);

        const vx = (Number(twc[0]) - prevTwcForRates[0]) / dt;
        const vy = (Number(twc[1]) - prevTwcForRates[1]) / dt;
        const vz = (Number(twc[2]) - prevTwcForRates[2]) / dt;
        vStr = `V (m/s)    ${vx.toFixed(2)} ${vy.toFixed(2)} ${vz.toFixed(2)}`;

        const wy = (Number(ypr[0]) - prevYprForRates[0]) * 180/Math.PI / dt;
        const wp = (Number(ypr[1]) - prevYprForRates[1]) * 180/Math.PI / dt;
        const wr = (Number(ypr[2]) - prevYprForRates[2]) * 180/Math.PI / dt;
        wStr = `W (deg/s)  ${wy.toFixed(1)} ${wp.toFixed(1)} ${wr.toFixed(1)}`;
      }

      prevTsForRates = tNow;
      prevTwcForRates = [Number(twc[0]), Number(twc[1]), Number(twc[2])];
      prevYprForRates = [Number(ypr[0]), Number(ypr[1]), Number(ypr[2])];

      const wasmTotal = Number(Module.getLastTotalMS?.() ?? 0);
      const wasmKLT = Number(Module.getLastKltMS?.() ?? 0);
      // const imuUsed = Number(Module.getImuUsedThisFrame?.() ?? 0);
      const imuHz = Number(Module.getImuHz?.() ?? 0);
      const mps = Number(Module.getNumMPs?.() ?? 0);
      const kps = Number(Module.getNumKFs?.() ?? 0);
      // const orbDescIn = Number(Module.getOrbDescInputPtsThisFrame?.() ?? 0);
      // const orbDescOut = Number(Module.getOrbDescRowsThisFrame?.() ?? 0);
      // const orbFullDetect = Number(Module.getOrbFullDetectCountThisFrame?.() ?? 0);
      updateHUDText([
        `Frame      ${i}/${limit-1}`,
        `t (s)      ${tNow.toFixed(3)}`,
        `WASM total ${wasmTotal.toFixed(2)} ms`,
        // `WASM KLT   ${wasmKLT.toFixed(2)} ms`,
        // `IMU used   YES`,
        `IMU Hz     ${imuHz ? imuHz.toFixed(1) : 'NA'}`,
        // `Pos        ${Number(twc[0]).toFixed(3)} ${Number(twc[1]).toFixed(3)} ${Number(twc[2]).toFixed(3)}`,
        'MPs        ' + mps,
        'KFs        ' + kps,
        // `IMU Δang   ${(Number(Module.getImuDeltaAngleDeg?.() ?? 0)).toFixed(1)} deg`, 
        // `ORB desc in ${orbDescIn}`,
        // `ORB desc out ${orbDescOut}`,
        // `ORB full det ${orbFullDetect}`,
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
  els.btnExportPerf.disabled = recordedPerf.length === 0;

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
    const ds = els.dataset?.value || 'kitti';
    a.download = (ds === 'tumvi') ? 'tumvi_poses.csv' : 'kitti_poses.csv';
        document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  logMsg('Exported kitti_poses.csv');
});

els.btnExportPerf.addEventListener('click', () => {
  if (!recordedPerf.length) return;

  function fmt(x) {
    return Number.isFinite(x) ? Number(x).toExponential(16) : 'nan';
  }

  const lines = [
    'frame,t,num_kfs,num_mps,num_keypoints,wasm_klt_ms,wasm_total_ms,wasm_imu_ms,wasm_seed_ms,imu_hz'
  ];

  for (const r of recordedPerf) {
    lines.push([
      r.frame,
      fmt(r.t),
      fmt(r.num_kfs),
      fmt(r.num_mps),
      fmt(r.num_keypoints),
      fmt(r.wasm_klt_ms),
      fmt(r.wasm_total_ms),
      fmt(r.wasm_imu_ms),
      fmt(r.wasm_seed_ms),
      fmt(r.imu_hz),
    ].join(','));
  }

  const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;

  const ds = els.dataset?.value || 'kitti';
  a.download = (ds === 'tumvi') ? 'tumvi_performance.csv' : 'kitti_performance.csv';

  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);

  logMsg(`Exported ${a.download}`);
});

