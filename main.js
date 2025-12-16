import { initRenderer, drawPoints, updateHUDText, drawPathXZ, drawAttitude } from './renderer.js';

function ensureLogEl() {
  let d = document.getElementById('log');
  if (!d) {
    d = document.createElement('div');
    d.id = 'log';
    d.style.cssText =
      'position:fixed;left:8px;bottom:48px;max-width:92%;max-height:40%;' +
      'overflow:auto;background:#000a;color:#0f0;font:12px monospace;' +
      'padding:6px;white-space:pre-wrap;z-index:9999;';
    document.body.appendChild(d);
  }
  return d;
}
const _logEl = ensureLogEl();
function logMsg(...args) {
  const line = args.map(x => (typeof x === 'object' ? JSON.stringify(x) : String(x))).join(' ');
  console.log(...args);
  _logEl.textContent += line + '\n';
  _logEl.scrollTop = _logEl.scrollHeight;
}

// Camera helpers
async function getCameraStream() {
  const md = navigator.mediaDevices;
  if (!md?.getUserMedia) throw new Error('getUserMedia not supported');

  try {
    const tmp = await md.getUserMedia({ video: true, audio: false });
    tmp.getTracks().forEach(t => t.stop());
  } catch (_) {}

  const devices = (await md.enumerateDevices()).filter(d => d.kind === 'videoinput');
  const back = devices.find(d => /back|rear|environment/i.test(d.label || '')) || devices[0];

  const constraints = {
    video: back
      ? { deviceId: { exact: back.deviceId }, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30, max: 30 } }
      : { facingMode: { ideal: 'environment' }, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30, max: 30 } },
    audio: false
  };  
  return md.getUserMedia(constraints);
}

function fitRectContain(frameW, frameH, viewW, viewH) {
  const scale = Math.min(viewW / frameW, viewH / frameH);
  const cssW = Math.round(frameW * scale);
  const cssH = Math.round(frameH * scale);
  const left = Math.round((viewW - cssW) / 2);
  const top  = Math.round((viewH - cssH) / 2);
  return { cssW, cssH, left, top, scale };
}
function layoutVideoAndCanvas(bgVideo, canvas, frameW, frameH) {
  const vw = window.innerWidth, vh = window.innerHeight;
  const r = fitRectContain(frameW, frameH, vw, vh);
  canvas.style.left = r.left + "px";
  canvas.style.top  = r.top  + "px";
  canvas.style.width  = r.cssW + "px";
  canvas.style.height = r.cssH + "px";
}

(async function boot() {
  const canvas  = document.getElementById('view');
  const bgVideo = document.getElementById('bgVideo');

  await initRenderer(canvas); // sets up 2D drawing ctx

  // Camera stream to <video> behind the canvas
  const stream = await getCameraStream();
  bgVideo.setAttribute('playsinline',''); bgVideo.setAttribute('muted','');
  bgVideo.muted = true; bgVideo.autoplay = true;
  bgVideo.srcObject = stream;
  await bgVideo.play().catch(()=>{});
  bgVideo.addEventListener('canplay', ()=>logMsg('camera video ready', bgVideo.videoWidth, 'x', bgVideo.videoHeight), { once:true });

  // Hidden true-pixel video 
  const video = document.createElement('video');
  video.setAttribute('playsinline',''); video.setAttribute('muted','');
  video.muted = true; video.autoplay = true;
  video.srcObject = stream;
  await video.play();
  if (video.readyState < 2) await new Promise(r => (video.onloadedmetadata = r));

  // Force a small camera mode to speed everything up
  const track = stream.getVideoTracks()[0];
  try {
    await track.applyConstraints({
      width:     { exact: 640 },
      height:    { exact: 480 },
      frameRate: { ideal: 30, max: 30 }
    });
  } catch (e) {
    logMsg('applyConstraints failed (will use device size):', e?.message || e);
  }
  
  // Prefer track settings 
  const s = track.getSettings();
  const W = s.width  || video.videoWidth  || 640;
  const H = s.height || video.videoHeight || 480;
  

  // Backing store resolution = camera pixels
  canvas.width = W;
  canvas.height = H;
  layoutVideoAndCanvas(bgVideo, canvas, W, H);
  window.addEventListener('resize', () => layoutVideoAndCanvas(bgVideo, canvas, W, H));

  // Cache-busted WASM glue
  const ts = Date.now();
  const { default: createModule } = await import(`./vio_wasm.js?v=${ts}`);
  const Module = await createModule({
    locateFile: (p)=> p.endsWith('.wasm') ? `./vio_wasm.wasm?v=${ts}` : p
  });
  logMsg(
    'WASM exports:',
    '_malloc=', !!Module._malloc,
    'HEAPU8=', !!Module.HEAPU8,
    'feedFramePtr=', !!Module.feedFramePtr
  );  
  window.Module = Module;

    // ================= IMU -> WASM =================
  function startImuFeed(Module) {
    if (!Module?.feedImuSample) {
      console.warn('feedImuSample export missing');
      return;
    }

    // Some browsers (iOS Safari) require permission
    async function ensurePermission() {
      try {
        if (typeof DeviceMotionEvent !== 'undefined' &&
            typeof DeviceMotionEvent.requestPermission === 'function') {
          const res = await DeviceMotionEvent.requestPermission();
          if (res !== 'granted') throw new Error('DeviceMotion permission denied');
        }
      } catch (e) {
        console.warn('IMU permission error:', e?.message || e);
      }
    }

    // Keep latest accel; gyro comes with rotationRate
    let lastAcc = { x: 0, y: 0, z: 0 };

    window.addEventListener('devicemotion', (e) => {
      // e.timeStamp is ms since page start (same clock family as performance.now())
      const ts = performance.now() * 1e-3;

      const a = e.accelerationIncludingGravity || e.acceleration;
      if (a) {
        // DeviceMotion uses m/s^2 (usually). Keep raw.
        lastAcc = {
          x: Number(a.x || 0),
          y: Number(a.y || 0),
          z: Number(a.z || 0),
        };
      }

      const r = e.rotationRate;
      if (r) {
        // rotationRate is usually in deg/s -> convert to rad/s
        const DEG2RAD = Math.PI / 180.0;

        // Common mapping: alpha=z, beta=x, gamma=y (device frame)
        const gx = Number(r.beta  || 0) * DEG2RAD;
        const gy = Number(r.gamma || 0) * DEG2RAD;
        const gz = Number(r.alpha || 0) * DEG2RAD;

        Module.feedImuSample(
          ts,
          lastAcc.x, lastAcc.y, lastAcc.z,
          gx, gy, gz
        );
      }
    }, { passive: true });

    ensurePermission();
  }

  // Call once after Module is ready:
  startImuFeed(Module);

  // Intrinsics (tunable FOV)
  const FOVY = 45;
  const fy = H / (2 * Math.tan((FOVY * Math.PI/180) / 2));
  const fx = fy * (W / H);
  const cx = W * 0.5, cy = H * 0.5;
  Module.initSystem(W, H, fx, fy, cx, cy);  

  // Default to KLT (0); change if you want to default to ORB
  try {
    Module.setTrackerType?.(2);
    Module.setHybridEveryN?.(4);
  } catch {}

// ------------- WebCodecs first
let fps = 0, lastTick = performance.now();

const useWebCodecs =
  ('MediaStreamTrackProcessor' in window) && ('VideoFrame' in window);
let ingestPath = useWebCodecs ? 'WebCodecs' : 'Canvas';

logMsg(
  'WebCodecs support:',
  'VideoFrame=', !!window.VideoFrame,
  'MediaStreamTrackProcessor=', !!window.MediaStreamTrackProcessor,
  'SecureContext=', window.isSecureContext
);

if (useWebCodecs) {
  const track = stream.getVideoTracks()[0];
  const processor = new MediaStreamTrackProcessor({ track });
  const reader = processor.readable.getReader();

  // Mutable dims taken from the *first* frame
  let curW = 0, curH = 0;
  let ySize = 0;
  let rgba = null;   // Uint8Array length = curW * curH * 4
  let yBuf = null;   // Uint8Array length = curW * curH (grayscale Y)
  let wasmPtr = 0, wasmView = null;  

  // Debounce reinit: only reinit if a new size appears in N consecutive frames
  let pendingW = 0, pendingH = 0, mismatchCount = 0;
  const REINIT_AFTER = 3; // frames

  function allocForSize(newW, newH) {
    curW = newW; curH = newH;
    ySize = curW * curH;
    rgba  = new Uint8Array(ySize * 4);
    yBuf  = new Uint8Array(ySize);    

    // Canvas + layout
    canvas.width = curW;
    canvas.height = curH;
    layoutVideoAndCanvas(bgVideo, canvas, curW, curH);

    // Re-init tracker with exact size
    const FOVY = 45;
    const fy = newH / (2 * Math.tan((FOVY * Math.PI/180) / 2));
    const fx = fy * (newW / newH);
    const cx = newW * 0.5, cy = newH * 0.5;
    Module.initSystem(newW, newH, fx, fy, cx, cy);    

    // Optional zero-copy heap path
    wasmPtr  = 0; wasmView = null;
    if (Module.HEAPU8 && Module._malloc && Module.feedFramePtr) {
      try {
        wasmPtr  = Module._malloc(ySize);
        wasmView = new Uint8Array(Module.HEAPU8.buffer, wasmPtr, ySize);
      } catch (_) {
        wasmPtr = 0; wasmView = null;
      }
    }
    logMsg(`(re)init for ${curW}x${curH} | zeroCopy=${!!wasmView}`);
  }

  let fps = 0, lastTick = performance.now();
  let frameIdx = 0;

  (function pull(){
    reader.read().then(async ({ value: frame, done }) => {
      if (done) return;

      const now = performance.now();
      const dt = Math.max(1, now - lastTick);
      fps = 0.9 * fps + 0.1 * (1000 / dt);
      lastTick = now;

      // Always ensure we schedule the next read
      try {
        // Determine the actual frame size
        // (use displayWidth/Height if present; else coded)
        const fw = (frame.displayWidth  || frame.codedWidth)  | 0;
        const fh = (frame.displayHeight || frame.codedHeight) | 0;

        // I420 requires even dims
        const adjW = fw & ~1, adjH = fh & ~1;

        if (curW === 0 || curH === 0) {
          // First frame: initialize from frame size
          allocForSize(adjW, adjH);
        } else if (adjW !== curW || adjH !== curH) {
          // Size changed — debounce to avoid flip-flop
          if (pendingW !== adjW || pendingH !== adjH) {
            pendingW = adjW; pendingH = adjH; mismatchCount = 1;
          } else {
            mismatchCount++;
          }
          if (mismatchCount >= REINIT_AFTER) {
            allocForSize(pendingW, pendingH);
            mismatchCount = 0;
          }
        } else {
          // size matches — clear debounce
          mismatchCount = 0;
        }

        const grayStart = performance.now();

        // RGBA is universally supported for copyTo on mobile Chrome
        await frame.copyTo(rgba, { format: 'RGBA' });
        frame.close();

        // RGBA -> 8-bit luma (Y ≈ 0.299R + 0.587G + 0.114B)
        // Integer form: (77*R + 150*G + 29*B) >> 8
        {
          const px = rgba;
          const Y  = yBuf;
          let j = 0;
          for (let i = 0; i < px.length; i += 4) {
            Y[j++] = (77*px[i] + 150*px[i+1] + 29*px[i+2]) >> 8;
          }
        }

        const tBeforeFeed = performance.now();
        if (wasmView && wasmPtr) {
          // zero-copy: write Y into WASM heap then call pointer API
          wasmView.set(yBuf);
          Module.feedFramePtr(wasmPtr, tBeforeFeed * 1e-3, curW, curH, false);
        } else {
          // fallback: pass JS Y buffer directly
          Module.feedFrameJS(yBuf, tBeforeFeed * 1e-3, curW, curH, false);
        }
        const tAfterFeed = performance.now();

        // Pull points & draw
        let pts = [];
        let kps=0, st=0, wasmTotal=0, wasmKLT=0, wasmSeed=0;
        try {
          const arr = Module.getPoints2D(); if (arr && arr.length) pts = arr;
          kps       = Module.getNumKeypoints?.() || 0;
          st        = Module.getTrackState?.()   || 0;
          wasmTotal = Module.getLastTotalMS?.()  || 0;
          wasmKLT   = Module.getLastKltMS?.()    || 0;
          wasmSeed  = Module.getLastSeedMS?.()   || 0;
        } catch {}

        drawPoints(pts, curW, curH);

        // --- Orientation overlay (yaw/pitch/roll in degrees) ---
        try {
          const ypr = Module.getYPR?.(); // Float32Array or JS array [yawDeg, pitchDeg, rollDeg]
          if (ypr && ypr.length === 3) {
            const RAD2DEG = 180 / Math.PI;
            const yawDeg   = Number(ypr[0]) * RAD2DEG;
            const pitchDeg = Number(ypr[1]) * RAD2DEG;
            const rollDeg  = Number(ypr[2]) * RAD2DEG;
            drawAttitude(yawDeg, pitchDeg, -rollDeg, curW, curH);
          }
        } catch {}
        
        // --- Top-down XZ path (optional – keep this if you like the inset) ---
        try {
          const pathXZ = Module.getPathXZ?.();
          if (pathXZ && pathXZ.length) {
            drawPathXZ(pathXZ, curW, curH);
          }
        } catch {}
      

        // HUD — always update (no silent '...')
        const jsT1 = performance.now();
        const jsGrayMS = (tBeforeFeed - grayStart).toFixed(2);
        const jsFeedMS = (tAfterFeed  - tBeforeFeed).toFixed(2);
        const jsDrawMS = (jsT1 - tAfterFeed).toFixed(2);
        // const N = Number(Module.getHybridEveryN?.() ?? 8);
        const orbMS = Number(Module.getLastOrbMS?.() ?? 0); 

        // E/H gate telemetry
        // const ehModel = Number(Module.getEHModel?.() ?? 0);   // 0=NONE,1=E,2=H
        // const ehE     = Number(Module.getEHInliersE?.() ?? 0);
        // const ehH     = Number(Module.getEHInliersH?.() ?? 0);
        // const ehPar   = Number(Module.getEHParallaxDeg?.() ?? 0);
        // const ehTag   = ehModel === 1 ? 'E' : (ehModel === 2 ? 'H' : '-');

        //Mappoints and Keyframes
        const kfs = Number(Module.getNumKFs?.() ?? 0);
        const mps = Number(Module.getNumMPs?.() ?? 0);

        const ranOrbThisFrame  = Number(Module.getRanOrbThisFrame?.() ?? 0);  // 0/1
        const perMode = `KLT ${wasmKLT.toFixed(2)} ms + ORBkey ${orbMS.toFixed(2)} ms`;  
        const imuMS = Number(Module.getLastImuMS?.() ?? 0);    
        const imuUsed = Number(Module.getImuUsedThisFrame?.() ?? 0);
        const imuHz   = Number(Module.getImuHz?.() ?? 0);
        const imuUsedCount = Number(Module.getImuUsedCount?.() ?? 0);
        const imuBuf = Number(Module.getImuBufSize?.() ?? 0);
        // --- gyro-only delta rotation debug (per video frame) ---
        let dYPR = null, dRod = null, dAng = 0;
        try {
          dYPR = Module.getImuDeltaYPR?.();      // [dyaw, dpitch, droll] in radians
          dRod = Module.getImuDeltaRod?.();      // [rx, ry, rz] in radians (axis*angle)
          dAng = Number(Module.getImuDeltaAngleDeg?.() ?? 0);
        } catch {}
        const RAD2DEG = 180 / Math.PI;

        const dyaw   = dYPR ? (Number(dYPR[0]) * RAD2DEG) : 0;
        const dpitch = dYPR ? (Number(dYPR[1]) * RAD2DEG) : 0;
        const droll  = dYPR ? (Number(dYPR[2]) * RAD2DEG) : 0;
        const fmt = (label, value) => `${String(label).padEnd(14)} ${String(value ?? 'NA')}`;

        updateHUDText([
          fmt('FPS', fps?.toFixed?.(1) ?? 'NA'),
          fmt('Keypoints', kps ?? 'NA'),
          fmt('Mode', ranOrbThisFrame ? 'ORB' : 'KLT'),
          fmt('JS gray ms', jsGrayMS ?? 'NA'),
          fmt('JS feed ms', jsFeedMS ?? 'NA'),
          fmt('JS draw ms', jsDrawMS ?? 'NA'),
          fmt('WASM total ms', wasmTotal?.toFixed?.(2) ?? 'NA'),
          fmt('Per-mode', perMode ?? 'NA'),
          fmt('MapPoints', mps ?? 'NA'),
          fmt('KeyFrames', kfs ?? 'NA'),
          fmt('IMU ms', imuMS?.toFixed?.(2) ?? 'NA'),
          fmt('IMU used', imuUsed ? `YES (${imuUsedCount})` : 'NO'),
          fmt('IMU Hz', imuHz ? imuHz.toFixed(1) : 'NA'),
          fmt('IMU buf', imuBuf ?? 'NA'),
          fmt('Gyro Δ angle', dAng.toFixed(3) + '°'),
          fmt('Gyro Δ YPR', `${dyaw.toFixed(3)} ${dpitch.toFixed(3)} ${droll.toFixed(3)} deg`),
          fmt('Ingest', ingestPath ?? 'NA'),
        ].join('\n'));               

      } catch (e) {
        // Surface any exception into the log AND HUD, so we see it
        logMsg('WebCodecs frame error:', e?.message || e);
        updateHUDText(`error: ${e?.message || e}`);
      } finally {
        pull();
      }
    }).catch(err => {
      console.error('WebCodecs reader error:', err);
      logMsg('WebCodecs reader error:', err?.message || err);
      // Don't loop if the reader failed hard
    });
  })();
} else {
  //------------------- Canvas if webcodecs fail
  let off, ctx;
  if ('OffscreenCanvas' in window) {
    off = new OffscreenCanvas(W, H);
    ctx = off.getContext('2d', { willReadFrequently: true });
  } else {
    off = document.createElement('canvas'); off.width = W; off.height = H;
    ctx = off.getContext('2d', { willReadFrequently: true });
  }

  // Preallocate grayscale buffer once and reuse
  const grayBytes = W * H;
  const gray = new Uint8Array(grayBytes); // reused each frame

  function loop() {
    const now = performance.now();
    const dt = Math.max(1, now - lastTick);
    fps = 0.9 * fps + 0.1 * (1000 / dt);
    lastTick = now;

    // Grab current RGBA frame
    const grayStart = performance.now();
    ctx.drawImage(video, 0, 0, W, H);
    const img = ctx.getImageData(0, 0, W, H).data; // Uint8ClampedArray RGBA

    // Convert RGBA -> GRAY into preallocated 'gray'
    // integer luma: Y ≈ (77*R + 150*G + 29*B) >> 8
    let j = 0;
    for (let i = 0; i < img.length; i += 4) {
      gray[j++] = (77*img[i] + 150*img[i+1] + 29*img[i+2]) >> 8;
    }

    // Push GRAY to WASM (timestamp in seconds) with isRGBA = false\
    const tBeforeFeed = performance.now();
    Module.feedFrameJS(gray, now * 1e-3, W, H, false);
    const tAfterFeed  = performance.now();

    // Pull 2D points & draw
    let pts = [];
    try {
      const arr = Module.getPoints2D();
      if (arr && typeof arr.length === 'number' && arr.length > 0) {
        pts = arr;
      }
    } catch {}

    drawPoints(pts, W, H);

    // pull WASM timings (ms)
    let wasmTotal=0, wasmKLT=0, wasmSeed=0;
    try {
      wasmTotal = Module.getLastTotalMS?.() || 0;
      wasmKLT   = Module.getLastKltMS?.()   || 0;
      wasmSeed  = Module.getLastSeedMS?.()  || 0;
    } catch {}

    const jsT1 = performance.now();
    const jsGrayMS = (tBeforeFeed - grayStart).toFixed(2);   // use grayStart
    const jsFeedMS = (tAfterFeed  - tBeforeFeed).toFixed(2);
    const jsRestMS = (jsT1 - tAfterFeed).toFixed(2);
    
    // HUD
    const orbMS = Module.getLastOrbMS?.() || 0;
    const perMode = `KLT ${wasmKLT.toFixed(2)} ms + ORBkey ${orbMS.toFixed(2)} ms`;
  
    // E/H gate telemetry
    const ehModel = Number(Module.getEHModel?.() ?? 0);
    const ehE     = Number(Module.getEHInliersE?.() ?? 0);
    const ehH     = Number(Module.getEHInliersH?.() ?? 0);
    const ehPar   = Number(Module.getEHParallaxDeg?.() ?? 0);
    const ehTag   = ehModel === 1 ? 'E' : (ehModel === 2 ? 'H' : '-');

    updateHUDText(
      `FPS ${fps.toFixed(1)} | ` +
      `JS gray ${jsGrayMS} ms, feed ${jsFeedMS} ms, rest ${jsRestMS} ms | ` +
      `WASM total ${wasmTotal.toFixed(2)} ms (${perMode}) | ` +
      `EH ${ehTag} E:${ehE} H:${ehH} par:${ehPar.toFixed(1)}° | ingest ${ingestPath}`
    );
   
    requestAnimationFrame(loop);
  }
    requestAnimationFrame(loop);
  }
  logMsg(`Ready ${W}x${H}`);
})().catch(e => {
  console.error(e);
  const d = document.getElementById('log');
  d.textContent += '\nError: ' + (e?.message || String(e));
});
