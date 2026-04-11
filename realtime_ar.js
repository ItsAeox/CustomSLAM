import * as THREE from 'https://esm.sh/three@0.160.0';
import { GLTFLoader } from 'https://esm.sh/three@0.160.0/examples/jsm/loaders/GLTFLoader.js';

function rad(x) { return x * Math.PI / 180; }

function estimateIntrinsics(width, height, fovYDeg = 60) {
  const fy = height / (2 * Math.tan((fovYDeg * Math.PI / 180) / 2));
  const fx = fy * (width / height);
  const cx = width * 0.5;
  const cy = height * 0.5;
  return { fx, fy, cx, cy };
}

// OpenCV world -> Three world helper
function cvWorldToThreeVec3(v) {
  return new THREE.Vector3(v.x, -v.y, -v.z);
}

function mat3RowMajorToThreeMatrix4(r) {
  const B = new THREE.Matrix4().set(
    1, 0, 0, 0,
    0,-1, 0, 0,
    0, 0,-1, 0,
    0, 0, 0, 1
  );

  const Rcv = new THREE.Matrix4().set(
    r[0], r[1], r[2], 0,
    r[3], r[4], r[5], 0,
    r[6], r[7], r[8], 0,
    0,    0,    0,    1
  );

  const out = new THREE.Matrix4();
  out.multiplyMatrices(B, Rcv);
  out.multiply(B);
  return out;
}

function applyCameraPose(camera, twc, rwc) {
  const tThree = cvWorldToThreeVec3({ x: twc[0], y: twc[1], z: twc[2] });
  const Rthree = mat3RowMajorToThreeMatrix4(rwc);

  const pose = new THREE.Matrix4();
  pose.copy(Rthree);
  pose.setPosition(tThree);

  camera.matrixAutoUpdate = false;
  camera.matrix.copy(pose);
  camera.matrix.decompose(camera.position, camera.quaternion, camera.scale);
  camera.matrixWorld.copy(camera.matrix);
  camera.matrixWorldInverse.copy(camera.matrixWorld).invert();
}

function getForwardWorldFromRwc(rwc) {
  return {
    x: rwc[2],
    y: rwc[5],
    z: rwc[8],
  };
}

function addVec(a, b) {
  return { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z };
}

function scaleVec(v, s) {
  return { x: v.x * s, y: v.y * s, z: v.z * s };
}

function fitElementToViewport(el, imgW, imgH) {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const s = Math.min(vw / imgW, vh / imgH);

    const cssW = Math.round(imgW * s);
    const cssH = Math.round(imgH * s);
    const left = Math.round((vw - cssW) * 0.5);
    const top  = Math.round((vh - cssH) * 0.5);

    el.style.position = 'fixed';
    el.style.left = `${left}px`;
    el.style.top = `${top}px`;
    el.style.width = `${cssW}px`;
    el.style.height = `${cssH}px`;
}

export class RealtimeARSession {
  constructor({ Module, overlayCanvas, hudEl, logEl, rtInfoEl, placeBtn }) {
    this.Module = Module;
    this.overlayCanvas = overlayCanvas;
    this.hudEl = hudEl;
    this.logEl = logEl;
    this.rtInfoEl = rtInfoEl;
    this.placeBtn = placeBtn;
    this.lastHudMs = 0;
    this.framesFed = 0;
    this.imuEvents = 0;
    this.isProcessingFrame = false;

    this.video = document.getElementById('bgVideo');
    this.arCanvas = document.getElementById('arView');

    this.running = false;
    this.stream = null;
    this.reader = null;
    this.trackProcessor = null;
    this.imuCleanup = [];
    this.balloon = null;
    this.balloonAnchorCv = null;

    this.frameWidth = 0;
    this.frameHeight = 0;
    this.wasmGrayPtr = 0;
    this.wasmGrayView = null;
    this.overlayCtx = this.overlayCanvas.getContext('2d', { alpha: true });

    this.loader = new GLTFLoader();

    this.scene = new THREE.Scene();
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.arCanvas,
      alpha: true,
      antialias: true,
    });
    this.renderer.setPixelRatio(window.devicePixelRatio || 1);
    this.renderer.setSize(window.innerWidth, window.innerHeight, false);

    this.camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.01, 100);

    const hemi = new THREE.HemisphereLight(0xffffff, 0x404040, 1.15);
    this.scene.add(hemi);

    const dir = new THREE.DirectionalLight(0xffffff, 1.0);
    dir.position.set(1, 2, 1);
    this.scene.add(dir);

    this.placeBtn.addEventListener('click', () => this.placeBalloon());

    window.addEventListener('resize', () => {
      this.renderer.setSize(window.innerWidth, window.innerHeight, false);
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();

      if (this.frameWidth && this.frameHeight) {
        fitElementToViewport(this.video, this.frameWidth, this.frameHeight);
        fitElementToViewport(this.overlayCanvas, this.frameWidth, this.frameHeight);
        fitElementToViewport(this.arCanvas, this.frameWidth, this.frameHeight);
      }
    });
  }

  log(...args) {
    console.log('[realtime]', ...args);
    if (this.logEl) {
      this.logEl.textContent += args.join(' ') + '\n';
      this.logEl.scrollTop = this.logEl.scrollHeight;
    }
  }

  async start() {
    if (this.running) return;

    if (!('MediaStreamTrackProcessor' in window)) {
      throw new Error('WebCodecs camera pipeline is not available in this browser.');
    }

    await this.requestMotionPermissionIfNeeded();

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: 'environment' },
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30, max: 60 }
      }
    });

    this.video.srcObject = this.stream;
    this.video.style.display = '';
    await this.video.play();

    const [track] = this.stream.getVideoTracks();
    const settings = track.getSettings();

    // Do not trust settings/video dimensions yet for processing.
    // We will adopt the first real VideoFrame size in handleVideoFrame().
    this.frameWidth = 0;
    this.frameHeight = 0;

    this.Module.setAccelIsSpecificForce?.(true);

    this.trackProcessor = new MediaStreamTrackProcessor({ track });
    this.reader = this.trackProcessor.readable.getReader();

    this.camera.fov = 60;
    this.camera.aspect = window.innerWidth / Math.max(1, window.innerHeight);
    this.camera.updateProjectionMatrix();

    await this.loadBalloon();
    this.attachIMUListeners();

    this.running = true;
    this.placeBtn.style.display = '';
    this.placeBtn.disabled = true;
    this.rtInfoEl.textContent = 'Realtime running. Move device to initialize.';
    this.log('Realtime WebCodecs path started:', this.frameWidth, 'x', this.frameHeight);

    this.processFrames();
    this.renderLoop();
  }

  stop() {
    this.running = false;

    for (const fn of this.imuCleanup) fn();
    this.imuCleanup = [];

    if (this.reader) {
      this.reader.cancel().catch(() => {});
      this.reader = null;
    }

    if (this.stream) {
      for (const tr of this.stream.getTracks()) tr.stop();
      this.stream = null;
    }

    if (this.wasmGrayPtr) {
      this.Module._free(this.wasmGrayPtr);
      this.wasmGrayPtr = 0;
      this.wasmGrayView = null;
    }

    this.video.srcObject = null;
    this.video.style.display = 'none';
    this.placeBtn.disabled = true;
    this.rtInfoEl.textContent = 'Realtime stopped.';
    this.clearOverlay();
  }

  clearOverlay() {
    this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
  }

  async requestMotionPermissionIfNeeded() {
    if (typeof DeviceMotionEvent !== 'undefined' &&
        typeof DeviceMotionEvent.requestPermission === 'function') {
      const res = await DeviceMotionEvent.requestPermission();
      if (res !== 'granted') {
        throw new Error('Device motion permission denied.');
      }
    }
  }

  attachIMUListeners() {
    const motionHandler = (ev) => {
      const ts = performance.now() * 1e-3;

      const acc = ev.accelerationIncludingGravity || ev.acceleration;
      const rr = ev.rotationRate;
      if (!acc || !rr) return;

      // First-pass browser mapping. May need sign/axis tuning per device.
      const gx = rad(rr.beta  || 0);
      const gy = rad(rr.gamma || 0);
      const gz = rad(rr.alpha || 0);

      const ax = acc.x || 0;
      const ay = acc.y || 0;
      const az = acc.z || 0;

      this.imuEvents++;
      this.Module.feedImuSample?.(ts, ax, ay, az, gx, gy, gz);
    };

    window.addEventListener('devicemotion', motionHandler, true);
    this.imuCleanup.push(() => {
      window.removeEventListener('devicemotion', motionHandler, true);
    });
  }

  async loadBalloon() {
    return new Promise((resolve, reject) => {
      this.loader.load(
        './colorful_balloons.glb',
        (gltf) => {
          this.balloon = gltf.scene;
          this.balloon.visible = false;

          const box = new THREE.Box3().setFromObject(this.balloon);
          const size = new THREE.Vector3();
          box.getSize(size);
          const maxDim = Math.max(size.x, size.y, size.z, 1e-6);

          const s = 0.18 / maxDim;
          this.balloon.scale.setScalar(s);

          this.scene.add(this.balloon);
          this.log('Loaded colorful_balloons.glb');
          resolve();
        },
        undefined,
        reject
      );
    });
  }

  async processFrames() {
    while (this.running && this.reader) {
      const { value: frame, done } = await this.reader.read();
      if (done || !frame) break;

      if (this.isProcessingFrame) {
        frame.close();
        continue; // drop stale frame
      }

      this.isProcessingFrame = true;
      try {
        await this.handleVideoFrame(frame);
      } catch (e) {
        console.error(e);
        this.rtInfoEl.textContent = `Frame processing failed: ${e?.message || e}`;
      } finally {
        this.isProcessingFrame = false;
        frame.close();
      }
    }
  }

  async handleVideoFrame(frame) {
    const ts = (frame.timestamp != null)
      ? frame.timestamp * 1e-6
      : performance.now() * 1e-3;
  
      const W = frame.codedWidth || frame.displayWidth || this.frameWidth;
      const H = frame.codedHeight || frame.displayHeight || this.frameHeight;
    
      // Adopt the first actual frame size as ground truth.
      if (!this.frameWidth || !this.frameHeight) {
        this.frameWidth = W;
        this.frameHeight = H;
    
        this.overlayCanvas.width = W;
        this.overlayCanvas.height = H;
        this.arCanvas.width = W;
        this.arCanvas.height = H;
    
        fitElementToViewport(this.video, W, H);
        fitElementToViewport(this.overlayCanvas, W, H);
        fitElementToViewport(this.arCanvas, W, H);
    
        const intr = estimateIntrinsics(W, H, 60);
        this.Module.initSystem(W, H, intr.fx, intr.fy, intr.cx, intr.cy);
    
        const grayBytes = W * H;
        if (this.wasmGrayPtr) this.Module._free(this.wasmGrayPtr);
        this.wasmGrayPtr = this.Module._malloc(grayBytes);
        this.wasmGrayView = new Uint8Array(this.Module.HEAPU8.buffer, this.wasmGrayPtr, grayBytes);
    
        this.camera.aspect = W / H;
        this.camera.updateProjectionMatrix();
    
        this.log('Adopted first frame size:', W, 'x', H);
      }
    
      // If later frames differ, log it and skip them.
      if (W !== this.frameWidth || H !== this.frameHeight) {
        this.rtInfoEl.textContent = `Frame size mismatch: got ${W}x${H}, expected ${this.frameWidth}x${this.frameHeight}`;
        return;
      }
  
    // Ask the frame how much space it needs in its OWN native format.
    const total = frame.allocationSize();
  
    if (!this.nativeBuffer || this.nativeBuffer.length !== total) {
      this.nativeBuffer = new Uint8Array(total);
    }
  
    // IMPORTANT: no explicit format here.
    const layout = await frame.copyTo(this.nativeBuffer);
  
    const fmt = frame.format || '';
    this.lastFrameFormat = fmt;
  
    // We want the Y plane directly when native format is planar 4:2:0.
    // Common values include "I420". Some browsers may expose other YUV variants.
    if (fmt === 'I420' && layout && layout.length >= 1) {
      const yPlane = layout[0];
      const yOffset = yPlane.offset;
      const yStride = yPlane.stride;
  
      // Fast path if tightly packed
      if (yStride === W) {
        this.wasmGrayView.set(this.nativeBuffer.subarray(yOffset, yOffset + W * H));
      } else {
        // Row-by-row copy if stride has padding
        for (let r = 0; r < H; r++) {
          const srcStart = yOffset + r * yStride;
          const srcEnd = srcStart + W;
          const dstStart = r * W;
          this.wasmGrayView.set(this.nativeBuffer.subarray(srcStart, srcEnd), dstStart);
        }
      }
  
      this.Module.feedFramePtr(this.wasmGrayPtr, ts, W, H, false);
      this.framesFed = (this.framesFed || 0) + 1;
    } else {
      // Native format is not directly usable as grayscale Y plane.
      // Surface it clearly so we know what browser is giving us.
      this.rtInfoEl.textContent = `Unsupported native frame format: ${fmt || 'unknown'}`;
      return;
    }
  
    this.drawTrackingOverlay();
    this.updateHud(ts);
  }
  drawTrackingOverlay() {
    this.clearOverlay();

    let pts = null;
    try {
      pts = this.Module.getPoints2D?.();
    } catch {}

    if (!pts || !pts.length) return;

    const ctx = this.overlayCtx;
    ctx.fillStyle = '#ffe658ff';

    for (let i = 0; i < pts.length; i += 2) {
      const x = pts[i];
      const y = pts[i + 1];
      ctx.fillRect(x - 1, y - 1, 3, 3);
    }
  }

  updateHud(ts) {
    const nowMs = performance.now();
    if (nowMs - this.lastHudMs < 100) return;
    this.lastHudMs = nowMs;

    const mapReady = !!this.Module.getMapInitialized?.();
    const metricReady = !!this.Module.getMetricReady?.();

    this.placeBtn.disabled = !(mapReady && metricReady);

    const twc = this.Module.getTwc?.() || [0, 0, 0];
    const ypr = this.Module.getYPR?.() || [0, 0, 0];
    const numPts = Number(this.Module.getNumKeypoints?.() ?? 0);

    this.hudEl.textContent = [
      `Mode       realtime`,
      `t (s)      ${ts.toFixed(3)}`,
      `Fmt        ${this.lastFrameFormat || 'unknown'}`,
      `Frames fed ${this.framesFed || 0}`,
      `IMU evts   ${this.imuEvents || 0}`,
      `Track      ${Number(this.Module.getTrackState?.() ?? -1)}`,
      `Map ready  ${mapReady ? 'YES' : 'NO'}`,
      `Metric     ${metricReady ? 'YES' : 'NO'}`,
      `Total ms   ${(Number(this.Module.getLastTotalMS?.() ?? 0)).toFixed(2)}`,
      `KLT ms     ${(Number(this.Module.getLastKltMS?.() ?? 0)).toFixed(2)}`,
      `IMU Hz     ${(Number(this.Module.getImuHz?.() ?? 0)).toFixed(1)}`,
      `IMU meas   ${Number(this.Module.getImuBufSize?.() ?? 0)}`,
      `Keypoints  ${numPts}`,
      `Pos        ${Number(twc[0]).toFixed(3)} ${Number(twc[1]).toFixed(3)} ${Number(twc[2]).toFixed(3)}`,
      `YPR        ${Number(ypr[0]).toFixed(2)} ${Number(ypr[1]).toFixed(2)} ${Number(ypr[2]).toFixed(2)}`,
      `MPs        ${Number(this.Module.getNumMPs?.() ?? 0)}`,
      `KFs        ${Number(this.Module.getNumKFs?.() ?? 0)}`
    ].join('\n');

    this.rtInfoEl.textContent = metricReady
      ? 'Metric scale ready. You can place the balloon.'
      : 'Realtime running. Move device to initialize.';
  }

  placeBalloon() {
    if (!this.balloon) return;

    const metricReady = !!this.Module.getMetricReady?.();
    const mapReady = !!this.Module.getMapInitialized?.();
    if (!metricReady || !mapReady) return;

    const twc = this.Module.getTwc?.();
    const rwc = this.Module.getRwc?.();
    if (!twc || !rwc || rwc.length !== 9) return;

    const t = { x: Number(twc[0]), y: Number(twc[1]), z: Number(twc[2]) };
    const f = getForwardWorldFromRwc(Array.from(rwc).map(Number));

    this.balloonAnchorCv = addVec(t, scaleVec(f, 0.10));

    const p3 = cvWorldToThreeVec3(this.balloonAnchorCv);
    this.balloon.position.copy(p3);
    this.balloon.visible = true;

    this.log('Placed balloon 10 cm in front of camera.');
  }

  renderLoop = () => {
    if (!this.running) return;

    try {
      const twc = this.Module.getTwc?.();
      const rwc = this.Module.getRwc?.();

      if (twc && rwc && rwc.length === 9) {
        applyCameraPose(
          this.camera,
          Array.from(twc).map(Number),
          Array.from(rwc).map(Number)
        );
      }

      this.renderer.render(this.scene, this.camera);
    } catch (e) {
      console.error(e);
    }

    requestAnimationFrame(this.renderLoop);
  };
}