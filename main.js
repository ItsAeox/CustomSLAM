// Minimal front-end to: (1) get camera stream, (2) push frames into WASM,
// (3) draw tiny white dots on top of the live video.

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
    video: back ? { deviceId: { exact: back.deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
                : { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
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

  // Hidden true-pixel video (avoid CSS scaling when we read pixels)
  const video = document.createElement('video');
  video.setAttribute('playsinline',''); video.setAttribute('muted','');
  video.muted = true; video.autoplay = true;
  video.srcObject = stream;
  await video.play();
  if (video.readyState < 2) await new Promise(r => (video.onloadedmetadata = r));

  const W = video.videoWidth  || 1280;
  const H = video.videoHeight || 720;

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
  window.Module = Module;

  // Intrinsics (rough; not used by tracker, but kept for API symmetry)
  const fovYdeg = 60;
  const fy = H / (2 * Math.tan((fovYdeg * Math.PI/180) / 2));
  const fx = fy * (W / H);
  const cx = W * 0.5, cy = H * 0.5;
  Module.initSystem(W, H, fx, fy, cx, cy);

  // Offscreen for RGBA reads
  let off, ctx;
  if ('OffscreenCanvas' in window) {
    off = new OffscreenCanvas(W, H);
    ctx = off.getContext('2d', { willReadFrequently: true });
  } else {
    off = document.createElement('canvas'); off.width = W; off.height = H;
    ctx = off.getContext('2d', { willReadFrequently: true });
  }

  // HUD
  let fps = 0, lastTick = performance.now();

  async function loop() {
    const now = performance.now();
    const dt = Math.max(1, now - lastTick);
    fps = 0.9 * fps + 0.1 * (1000 / dt);
    lastTick = now;

    // Grab current RGBA frame
    ctx.drawImage(video, 0, 0, W, H);
    const img = ctx.getImageData(0, 0, W, H);
    const u8 = new Uint8Array(img.data.buffer);

    // Push to WASM (timestamp in seconds)
    Module.feedFrameJS(u8, now * 1e-3, W, H, true);

    // Pull 2D points & draw
    let pts = [];
    try {
      const arr = Module.getPoints2D(); // JS array of numbers
      if (Array.isArray(arr) && arr.length > 0) pts = arr;
    } catch (e) { /* ignore */ }

    drawPoints(pts, W, H); // draws small white dots

    // HUD line
    try {
      const kps = Module.getNumKeypoints?.() || 0;
      const st  = Module.getTrackState?.() || 0;
      updateHUDText(`FPS ${fps.toFixed(1)} | kps ${kps} | state ${st}`);
    } catch {}

    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);

  logMsg(`Ready ${W}x${H}`);
})().catch(e => {
  console.error(e);
  const d = document.getElementById('log');
  d.textContent += '\nError: ' + (e?.message || String(e));
});
