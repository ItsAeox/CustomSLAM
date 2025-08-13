import createVioModule from './vio_wasm.js';
import {
  initRenderer,
  configureCameraFromIntrinsics,
  updateFromPose_Twc_threeBasis,
  placeAnchorAtScreen,
  initPointCloud,          
  updatePointCloud,        
  setPointCloudVisible,    
  debugWobbleCamera,
} from './renderer.js';

// ----- onscreen logger so you can see logs on phone -----
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

// ----- make sure video is behind canvas & doesn’t eat taps -----
(function injectOverlayCSS(){
  const css = `
    #bgVideo{position:fixed;left:0;top:0;z-index:0;pointer-events:none;}
    #view{position:fixed;left:0;top:0;z-index:10;touch-action:manipulation;}
    #hud{z-index:20;}
    #placeBtn{position:fixed;right:8px;bottom:8px;z-index:20;}
  `;
  const s = document.createElement('style'); s.textContent = css; document.head.appendChild(s);
})();

// ---- HUD ----
const hudEl = (() => {
  let el = document.getElementById('hud');
  if (!el) {
    el = document.createElement('div');
    el.id = 'hud';
    el.style.cssText = 'position:fixed;right:8px;top:8px;color:#0f0;font:12px monospace;z-index:20;text-align:right';
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

// ---- robust camera getter ----
async function getCameraStream() {
  const md = navigator.mediaDevices;
  if (!md?.getUserMedia) throw new Error('getUserMedia not supported');

  // Probe once to unlock labels on some phones
  try {
    const tmp = await md.getUserMedia({ video: true, audio: false });
    tmp.getTracks().forEach(t => t.stop());
  } catch (e) {
    // continue; some browsers allow enumerate without this
  }

  const devices = (await md.enumerateDevices()).filter(d => d.kind === 'videoinput');
  const back = devices.find(d => /back|rear|environment/i.test(d.label || '')) || devices[0];

  const constraints = {
    video: back ? { deviceId: { exact: back.deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
                : { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false
  };
  return md.getUserMedia(constraints);
}

(async function boot() {
  try {
    const canvas = document.getElementById('view');
    const bgVideo = document.getElementById('bgVideo');

    // Optional place button (fallback)
    let btn = document.getElementById('placeBtn');
    if (!btn) {
      btn = document.createElement('button');
      btn.id = 'placeBtn';
      btn.textContent = 'Place Cactus';
      document.body.appendChild(btn);
    }

    await initRenderer(canvas);
    initPointCloud(2000, 3); 
    setPointCloudVisible(true);

    // Tap / click to place the cactus
    function onTap(ev) {
      const t = ev.changedTouches ? ev.changedTouches[0] : ev;
      logMsg('tap at', t.clientX, t.clientY);
      placeAnchorAtScreen(t.clientX, t.clientY);
      ev.preventDefault();
    }
    canvas.addEventListener('click', onTap, { passive: false });
    canvas.addEventListener('touchend', onTap, { passive: false });
    btn.addEventListener('click', () => {
      const r = canvas.getBoundingClientRect();
      const cx = r.left + r.width/2, cy = r.top + r.height/2;
      logMsg('button place center', cx, cy);
      placeAnchorAtScreen(cx, cy);
    });

    // Camera stream
    const stream = await getCameraStream();
    // Show camera behind WebGL (ensure autoplay/muted/inline)
    bgVideo.setAttribute('playsinline',''); bgVideo.setAttribute('muted','');
    bgVideo.muted = true; bgVideo.autoplay = true;
    bgVideo.srcObject = stream;
    await bgVideo.play().catch(()=>{});
    bgVideo.addEventListener('canplay', ()=>logMsg('bgVideo canplay; size=', bgVideo.videoWidth, 'x', bgVideo.videoHeight), { once:true });
    logMsg('bgVideo readyState=', bgVideo.readyState);

    // Hidden <video> for pixel reads (avoids CSS scaling)
    const video = document.createElement('video');
    video.setAttribute('playsinline',''); video.setAttribute('muted','');
    video.muted = true; video.autoplay = true;
    video.srcObject = stream;
    await video.play();
    if (video.readyState < 2) await new Promise(r => (video.onloadedmetadata = r));

    // 1) Video size known
    const W = video.videoWidth  || 1280;
    const H = video.videoHeight || 720;

    // 2) Match elements to pixel size
    canvas.width = W; canvas.height = H;
    canvas.style.width  = `${W}px`; canvas.style.height = `${H}px`;
    bgVideo.width = W; bgVideo.height = H;
    bgVideo.style.width  = `${W}px`; bgVideo.style.height = `${H}px`;

    // 3) Intrinsics (compute BEFORE initSystem)
    const fovDeg = 60;
    const fx = W / (2 * Math.tan((fovDeg * Math.PI/180) / 2));
    const fy = fx;
    const cx = W * 0.5;
    const cy = H * 0.5;

    // 4) Load fresh WASM glue + .wasm (cache-busted)
    const ts = Date.now();
    const { default: createModule } = await import(`./vio_wasm.js?v=${ts}`);
    const Module = await createModule({
      locateFile: (path) => path.endsWith('.wasm')
        ? `./vio_wasm.wasm?v=${ts}`
        : path
    });
    window.Module = Module;

    // Sanity: do we have the bindings we expect?
    logMsg('Module keys:', Object.keys(Module).slice(0,20).join(', '));

    // 5) Configure three.js camera FIRST
    configureCameraFromIntrinsics({ fx, fy, cx, cy, width: W, height: H });

    // 6) Now init SLAM with SAME intrinsics
    Module.initSystem(W, H, fx, fy, cx, cy);

    // optional: immediate probe
    const pv = Module.getPose?.();
    if (pv && typeof pv.size === 'function') {
      logMsg('After init, pose size =', pv.size());
      pv.delete?.();
    } else {
      logMsg('After init, pose is not a Vector; type =', typeof pv);
    }

    // Offscreen for RGBA
    let off, ctx;
    if ('OffscreenCanvas' in window) {
      off = new OffscreenCanvas(W, H);
      ctx = off.getContext('2d', { willReadFrequently: true });
    } else {
      off = document.createElement('canvas'); off.width = W; off.height = H;
      ctx = off.getContext('2d', { willReadFrequently: true });
    }

    let lastPC = 0;

    async function loop() {
      //debugWobbleCamera();
      // FPS
      const now = performance.now();
      const dt = Math.max(1, now - lastTick);
      fps = 0.9 * fps + 0.1 * (1000 / dt);
      lastTick = now;

      // Feed current frame (RGBA) to WASM
      ctx.drawImage(video, 0, 0, W, H);
      const img = ctx.getImageData(0, 0, W, H);
      const u8 = new Uint8Array(img.data.buffer);
      Module.feedFrameJS(u8, now * 1e-3, W, H, true);

      try {
        const pv = Module.getPose?.();         // Embind VectorDouble
        if (pv && typeof pv.size === 'function') {
          const n = pv.size();
          if (n === 16) {
            const pose = new Float64Array(16);
            for (let i = 0; i < 16; i++) pose[i] = pv.get(i);
            pv.delete?.();                     // free Embind vector

            lastPose = pose;
            const tx = pose[12], ty = pose[13], tz = pose[14];
            if ((performance.now()|0) % 500 < 16) logMsg('Twc t=', tx.toFixed(3), ty.toFixed(3), tz.toFixed(3));
            updateFromPose_Twc_threeBasis(pose);
          } else {
            pv.delete?.();
            if (lastPose) updateFromPose_Twc_threeBasis(lastPose);
          }
        } else if (lastPose) {
          updateFromPose_Twc_threeBasis(lastPose);
        }
      } catch (e) {
        logMsg('getPose error:', e?.message || e);
      }

      // HUD extras
      let extra = '';
      try {
        const kps = Module.getNumKeypoints?.();
        const inl = Module.getNumInliers?.();
        const st  = Module.getTrackState?.();
        const pts = Module.getNumMapPoints?.();
        if (typeof kps === 'number' && typeof inl === 'number' && typeof st === 'number' && typeof pts === 'number') {
          extra = ` | kps:${kps} inl:${inl} state:${st} pts:${pts}`;
        }
      } catch {}
      updateHUD(extra);

            // --- point cloud update (every ~400 ms) ---
      const now2 = performance.now();
      if (now2 - lastPC > 400 && Module.getPointsXYZ) {
        try {
          const vec = Module.getPointsXYZ(3000);   // up to N points (3*N doubles)
          if (vec && typeof vec.size === 'function') {
            const n = vec.size();
            if (n > 0) {
              // Pull into a typed array for faster transfer
              const tmp = new Float32Array(n);
              for (let i = 0; i < n; ++i) tmp[i] = vec.get(i);
              updatePointCloud(tmp);
            }
            vec.delete?.(); // free Embind vector
          }
        } catch (e) {
          logMsg('getPointsXYZ error:', e?.message || e);
        }
        lastPC = now2;
      }

      requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);

    // ---- IMU streaming to WASM: Generic Sensor API (Android) + fallback (iOS) ----
    let imuHz = 0;
    let _imuCount = 0, _imuLast = performance.now();

    async function startIMU() {
      const hasGenericSensors =
        'Gyroscope' in window && 'Accelerometer' in window;

      if (hasGenericSensors) {
        // These are in SI units: Gyro = rad/s, Accel = m/s^2
        const gyro = new Gyroscope({ frequency: 200 });
        const accel = new Accelerometer({ frequency: 200 });

        let gx=0, gy=0, gz=0;
        let ax=0, ay=0, az=0;

        gyro.addEventListener('reading', () => {
          // Gyro event rate is high; use this as the driver
          gx = gyro.x || 0; gy = gyro.y || 0; gz = gyro.z || 0;

          const now = performance.now();
          if (window.Module?.feedIMU) {
            Module.feedIMU(now * 1e-3, ax, ay, az, gx, gy, gz);
          }

          _imuCount++;
          const dt = now - _imuLast;
          if (dt >= 500) {
            imuHz = (_imuCount * 1000) / dt;
            _imuCount = 0;
            _imuLast = now;
          }
        });

        accel.addEventListener('reading', () => {
          ax = accel.x || 0; ay = accel.y || 0; az = accel.z || 0;
        });

        try {
          gyro.start();
          accel.start();
          logMsg('Generic Sensor API started (gyro+accel)');
          return;
        } catch (e) {
          logMsg('Generic Sensor start failed, falling back:', e?.message || e);
          // Fall through to DeviceMotion below
        }
      }

      // ---- Fallback: DeviceMotion (iOS / older Android) ----
      const deg2rad = Math.PI / 180;
      let firstEventLogged = false;

      window.addEventListener('devicemotion', (e) => {
        const now = performance.now();
        const acc = e.acceleration || e.accelerationIncludingGravity || {x:0,y:0,z:0};
        const rr  = e.rotationRate || {alpha:0,beta:0,gamma:0};

        const ax = acc.x || 0, ay = acc.y || 0, az = acc.z || 0;
        const gx = (rr.beta  || 0) * deg2rad;  // x
        const gy = (rr.gamma || 0) * deg2rad;  // y
        const gz = (rr.alpha || 0) * deg2rad;  // z

        if (window.Module?.feedIMU) {
          Module.feedIMU(now * 1e-3, ax, ay, az, gx, gy, gz);
        }

        if (!firstEventLogged) {
          firstEventLogged = true;
          logMsg('devicemotion active (fallback)');
        }

        _imuCount++;
        const dt = now - _imuLast;
        if (dt >= 500) {
          imuHz = (_imuCount * 1000) / dt;
          _imuCount = 0;
          _imuLast  = now;
        }
      }, { passive: true });

      logMsg('DeviceMotion listener attached (fallback)');
    }

    // Permission flow: iOS needs a tap; Android typically doesn’t
    if (window.DeviceMotionEvent?.requestPermission) {
      document.body.addEventListener('click', async () => {
        try {
          const s = await DeviceMotionEvent.requestPermission();
          logMsg('motion perm:', s);
          await startIMU();
        } catch (e) {
          logMsg('motion perm error:', e?.message || e);
        }
      }, { once: true });
    } else {
      // Android path
      startIMU();
    }

    // Patch HUD to show IMU Hz
    const _origUpdateHUD = updateHUD;
    updateHUD = function(extra='') {
      _origUpdateHUD(`${extra} | imu:${imuHz.toFixed(0)}Hz`);
    };

    logMsg(`Ready ${W}x${H}`);
  } catch (e) {
    console.error(e);
    logMsg('Error:', e?.message || e);
  }
})();

