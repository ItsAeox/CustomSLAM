import createVioModule from './vio_wasm.js';
import {
  initRenderer,
  configureCameraFromIntrinsics,
  updateFromPose_Twc_threeBasis
} from './renderer.js';

const log = (...a)=>{ const el=document.getElementById('log'); el.textContent=a.join(' '); console.log(...a); };

async function getCameraStream() {
  const nav = navigator;
  if (nav.mediaDevices?.getUserMedia) {
    return nav.mediaDevices.getUserMedia({
      video: { facingMode:'environment', width:{ideal:1280}, height:{ideal:720} },
      audio: false
    });
  }
  const legacy = nav.getUserMedia || nav.webkitGetUserMedia || nav.mozGetUserMedia || nav.msGetUserMedia;
  if (legacy) return new Promise((res, rej)=>legacy.call(nav, {video:true,audio:false}, res, rej));
  throw new Error(`getUserMedia not supported. Secure:${window.isSecureContext} Host:${location.host}`);
}

// HUD
const hudEl = (() => {
  let el = document.getElementById('hud');
  if (!el) {
    el = document.createElement('div');
    el.id = 'hud';
    el.style.cssText = 'position:fixed;right:8px;top:8px;color:#0f0;font:12px monospace;z-index:2;text-align:right';
    document.body.appendChild(el);
  }
  return el;
})();
let fps = 0, lastTick = performance.now();
let lastPose = null;
function updateHUD(extra='') {
  const p = lastPose;
  const poseStr = p ? `[${Array.from(p).slice(0,4).map(v=>(+v).toFixed(3)).join(', ')} ...]` : 'none';
  hudEl.textContent = `FPS: ${fps.toFixed(1)} | pose: ${p ? 'ok' : 'none'} ${poseStr}${extra}`;
}

(async function boot() {
  try {
    const canvas = document.getElementById('view');
    const bgVideo = document.getElementById('bgVideo');

    // Init three.js (camera intrinsics will be configured after video is ready)
    await initRenderer(canvas);

    // Camera stream
    const stream = await getCameraStream();
    // Show camera behind WebGL
    bgVideo.srcObject = stream;
    bgVideo.play?.();

    // A hidden/secondary <video> used for pixel reads (avoids CSS scaling issues)
    const video = document.createElement('video');
    video.playsInline = true; video.autoplay = true; video.muted = true;
    video.srcObject = stream;
    await video.play();
    if (video.readyState < 2) {
      await new Promise(r => (video.onloadedmetadata = r));
    }

    const W = video.videoWidth  || 1280;
    const H = video.videoHeight || 720;

    // Match elements to pixel size to avoid stretch
    canvas.width = W; canvas.height = H;
    canvas.style.width  = `${W}px`;
    canvas.style.height = `${H}px`;
    bgVideo.width = W; bgVideo.height = H;
    bgVideo.style.width  = `${W}px`;
    bgVideo.style.height = `${H}px`;

    // Load WASM
    const Module = await createVioModule();
    window.Module = Module; // for console debugging

    // --- Intrinsics ---
    // If you have calibrated values, use them here.
    // For now we keep your 60° guess, but three.js camera is configured to match it exactly.
    const fovDeg = 60;
    const fx = W / (2 * Math.tan((fovDeg * Math.PI/180) / 2));
    const fy = fx;
    const cx = W * 0.5;
    const cy = H * 0.5;

    // Configure three.js camera with the SAME intrinsics
    configureCameraFromIntrinsics({ fx, fy, cx, cy, width: W, height: H });

    // Init SLAM with the SAME intrinsics
    Module.initSystem(W, H, fx, fy, cx, cy);

    // Offscreen for RGBA readback
    let off, ctx;
    if ('OffscreenCanvas' in window) {
      off = new OffscreenCanvas(W, H);
      ctx = off.getContext('2d', { willReadFrequently: true });
    } else {
      off = document.createElement('canvas'); off.width = W; off.height = H;
      ctx = off.getContext('2d', { willReadFrequently: true });
    }

    async function loop() {
      // timing & FPS
      const now = performance.now();
      const dt = Math.max(1, now - lastTick);
      fps = 0.9 * fps + 0.1 * (1000 / dt);
      lastTick = now;

      // Feed current frame to WASM (RGBA)
      ctx.drawImage(video, 0, 0, W, H);
      const img = ctx.getImageData(0, 0, W, H);
      const u8 = new Uint8Array(img.data.buffer);
      Module.feedFrameJS(u8, now * 1e-3, W, H, true);

      // Get pose: world-from-camera in three.js basis (column-major 4x4)
      const pose = Module.getPose();
      if (pose && pose.length === 16) {
        lastPose = pose; // keep as Float64 from Embind; three.js accepts number[]
        updateFromPose_Twc_threeBasis(pose);
      } else if (lastPose) {
        updateFromPose_Twc_threeBasis(lastPose); // hold last good pose to avoid flicker
      }

      // HUD counters (optional)
      let extra = '';
      try {
        const kps = Module.getNumKeypoints?.();
        const inl = Module.getNumInliers?.();
        const st  = Module.getTrackState?.();
        if (typeof kps === 'number' && typeof inl === 'number' && typeof st === 'number') {
          extra = ` | kps:${kps} inl:${inl} state:${st}`;
        }
      } catch (_) {}
      updateHUD(extra);

      requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);

    // iOS motion permission placeholder (IMU fusion later)
    if (window.DeviceMotionEvent?.requestPermission) {
      document.body.addEventListener('click', async () => {
        try { await DeviceMotionEvent.requestPermission(); } catch {}
      }, { once: true });
    }

    log(`Ready ${W}x${H}`);
  } catch (e) {
    console.error(e);
    log('Error:', e?.message || e);
  }
})();
