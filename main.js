import { initRenderer, drawPoints, updateHUDText } from './renderer.js';

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

  // Intrinsics 
  const fovYdeg = 60;
  const fy = H / (2 * Math.tan((fovYdeg * Math.PI/180) / 2));
  const fx = fy * (W / H);
  const cx = W * 0.5, cy = H * 0.5;
  Module.initSystem(W, H, fx, fy, cx, cy);

  // Default to KLT (0); change if you want to default to ORB
  try {
    Module.setTrackerType?.(0);
    Module.setHybridEveryN?.(8);
  } catch {}
  
  const sel = document.getElementById('trackerSel');
  const nEl = document.getElementById('hybridN');
  
  if (sel) {
    sel.value = String(Module.getTrackerType?.() || 0);
    sel.addEventListener('change', () => {
      const t = parseInt(sel.value, 10) || 0;
      Module.setTrackerType?.(t);
    });
  }
  if (nEl) {
    // Initialize from WASM if available
    const curN = Module.getHybridEveryN?.() || 8;
    nEl.value = String(curN);
    nEl.addEventListener('change', () => {
      const v = Math.max(1, parseInt(nEl.value, 10) || 8);
      Module.setHybridEveryN?.(v);
    });
  }  

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
    const fovYdeg = 60;
    const fy = curH / (2 * Math.tan((fovYdeg * Math.PI/180) / 2));
    const fx = fy * (curW / curH);
    const cx = curW * 0.5, cy = curH * 0.5;
    Module.initSystem(curW, curH, fx, fy, cx, cy);

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

        // (Debug) if we still get 0 features, tell us every 30 frames
        if ((++frameIdx % 30) === 0 && kps === 0) {
          logMsg(`DEBUG kps=0 | state=${st} | coded=${curW}x${curH}`);
        }

        // HUD — always update (no silent '...')
        const jsT1 = performance.now();
        const jsGrayMS = (tBeforeFeed - grayStart).toFixed(2);
        const jsFeedMS = (tAfterFeed  - tBeforeFeed).toFixed(2);
        const jsDrawMS = (jsT1 - tAfterFeed).toFixed(2);
        const modeVal = sel ? sel.value : '0';
        const N = Number(Module.getHybridEveryN?.() ?? 8);
        let tracker =
          modeVal === '1' ? 'ORB' :
          modeVal === '2' ? `HYBRID (N=${(Module.getHybridEveryN?.()||8)})` :
          'KLT';
        const orbMS = Number(Module.getLastOrbMS?.() ?? 0); 
        
        let hybInfo = '';
        if (modeVal === '2') {
          const mod  = Number(Module.getHybridFrameMod?.() ?? 0);   // 0..N-1
          const orbKF= Number(Module.getOrbKFCount?.()   ?? 0);     // 0,1,2,...
          const ran  = Number(Module.getRanOrbThisFrame?.() ?? 0);  // 0/1
          // Example: HYB mod 3/8 KF#5 ran:0
          hybInfo = ` | HYB mod ${mod}/${N} KF#${orbKF} ran:${ran}`;
        }
        
        const perMode =
        modeVal === '0'
          ? `KLT ${wasmKLT.toFixed(2)} ms, seed ${wasmSeed.toFixed(2)}`
          : modeVal === '1'
          ? `ORB ${orbMS.toFixed(2)} ms`
          : `KLT ${wasmKLT.toFixed(2)} ms + ORBkey ${orbMS.toFixed(2)} ms${hybInfo}`;      
        
        updateHUDText(
          `FPS ${fps.toFixed(1)} | kps ${kps} | ` +
          `JS gray ${jsGrayMS} ms, feed ${jsFeedMS} ms, draw ${jsDrawMS} ms | ` +
          `WASM total ${wasmTotal.toFixed(2)} ms (${perMode}) | ingest ${ingestPath}`
        );

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
    const modeVal = sel ? sel.value : '0';
    let tracker =
      modeVal === '1' ? 'ORB' :
      modeVal === '2' ? `HYBRID (N=${(Module.getHybridEveryN?.()||8)})` :
      'KLT';
    const orbMS = Module.getLastOrbMS?.() || 0;
    
    
    const perMode =
    modeVal === '0'
      ? `KLT ${wasmKLT.toFixed(2)} ms, seed ${wasmSeed.toFixed(2)}`
      : modeVal === '1'
      ? `ORB ${orbMS.toFixed(2)} ms`
      : `KLT ${wasmKLT.toFixed(2)} ms + ORBkey ${orbMS.toFixed(2)} ms`;
  
    
    updateHUDText(
      `FPS ${fps.toFixed(1)} | kps ${kps} | ` +
      `JS gray ${jsGrayMS} ms, feed ${jsFeedMS} ms, draw ${jsDrawMS} ms | ` +
      `WASM total ${wasmTotal.toFixed(2)} ms (${perMode}) | ingest ${ingestPath}`
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
