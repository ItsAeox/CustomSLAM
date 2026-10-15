// Web Worker hosting the ORB-SLAM3 WebAssembly module (web/build/orbslam3.js).
// ORB-SLAM3's own threads (local mapping, loop closing) become nested workers via emscripten pthreads.
//
// Messages in:   {type:'init', vocabUrl, settingsUrl | settingsText, clahe, seed}
//                {type:'bitmap', bitmap, t[, viz, withMap]}   (live camera frame as an ImageBitmap)
//                {type:'frame', idx, t, url[, viz, withMap]}  (worker fetches + decodes the image; viz: viewer data
//                                                              + the decoded ImageBitmap in the reply)
//                {type:'finish'}
//                {type:'keepMap', on}                          (AR: relocalise instead of a new map while anchored)
//                {type:'calibInit'} {type:'calibFrame', bitmap, cols, rows} {type:'calibAdd'}
//                {type:'calibRemove', index} {type:'calibClear'} {type:'calibSolve', cols, rows}   (calib.html)
// Messages out:  {type:'log', text} {type:'ready', ms} {type:'pose', idx, t, state, twc, trackMs, decodeMs}
//                {type:'finished', frames, keyframes}  (ORB-SLAM3 final trajectories, EuRoC format)
import createOrbSlam3 from './build/orbslam3.js';

let M = null;
let slam = null;
let cal = null;  // camera calibration (calib.html)

// WebCodecs camera (AR page): the page hands over the camera's frame stream (MediaStreamTrackProcessor.readable). A reader
// keeps only the newest VideoFrame (older ones are closed and counted as dropped). For tracking, the luma (Y) plane of
// NV12 / I420 frames is the grayscale image: it is copied into the wasm frame buffer directly (no canvas, no RGBA).
let camReader = null, camLatest = null, camWaiter = null, camFrames = 0, camDropped = 0, camScratch = null;
async function camLoop() {
  for (;;) {
    let r;
    try { r = await camReader.read(); } catch { return; }
    if (r.done) return;
    camFrames++;
    if (camLatest) { camLatest.close(); camDropped++; }
    camLatest = r.value;
    if (camWaiter) { const w = camWaiter; camWaiter = null; w(); }
  }
}
const nextCamFrame = () => (camLatest ? Promise.resolve() : new Promise((res) => { camWaiter = res; }));
const LUMA_FIRST = new Set(['NV12', 'I420', 'I420A', 'I422', 'I444', 'NV12A']);

// gray image of a VideoFrame into the wasm frame buffer; returns { w, h }
async function frameToGray(f) {
  const w = f.visibleRect.width, h = f.visibleRect.height;
  const size = f.allocationSize();
  if (!camScratch || camScratch.length < size) camScratch = new Uint8Array(size);
  const layout = await f.copyTo(camScratch);
  const ptr = slam.frameBuffer(w, h);
  const dst = M.HEAPU8.subarray(ptr, ptr + w * h);
  if (LUMA_FIRST.has(f.format)) {
    const { offset, stride } = layout[0];
    if (stride === w) dst.set(camScratch.subarray(offset, offset + w * h));
    else for (let y = 0; y < h; y++) dst.set(camScratch.subarray(offset + y * stride, offset + y * stride + w), y * w);
  } else {  // RGBA / RGBX / BGRA / BGRX
    const bgr = f.format.startsWith('BGR');
    const { offset, stride } = layout[0];
    for (let y = 0, j = 0; y < h; y++) {
      for (let x = 0, i = offset + y * stride; x < w; x++, i += 4, j++) {
        const r = camScratch[bgr ? i + 2 : i], g = camScratch[i + 1], b = camScratch[bgr ? i : i + 2];
        dst[j] = (r * 4899 + g * 9617 + b * 1868 + 8192) >> 14;
      }
    }
  }
  return { w, h };
}
let canvas = null;
let ctx = null;

const log = (text) => postMessage({ type: 'log', text });

// OpenCV's RGB->gray fixed-point formula (cvtColor, 14-bit): exact for gray images (R=G=B).
function toGray(rgba, gray) {
  for (let i = 0, j = 0; j < gray.length; i += 4, j++) {
    gray[j] = (rgba[i] * 4899 + rgba[i + 1] * 9617 + rgba[i + 2] * 1868 + 8192) >> 14;
  }
}

// keepBitmap: return the decoded ImageBitmap too (viewer display; transferred to the page, which closes it)
async function decode(url, keepBitmap = false) {
  const blob = await (await fetch(url)).blob();
  const bmp = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  if (!canvas || canvas.width !== bmp.width || canvas.height !== bmp.height) {
    canvas = new OffscreenCanvas(bmp.width, bmp.height);
    ctx = canvas.getContext('2d', { willReadFrequently: true });
  }
  ctx.drawImage(bmp, 0, 0);
  if (!keepBitmap) bmp.close();
  return { img: ctx.getImageData(0, 0, canvas.width, canvas.height), bmp: keepBitmap ? bmp : null };
}

// Viewer payload (msg.viz): current-frame keypoints / init matches every frame; the map (points, keyframes,
// graph) when msg.withMap. Typed arrays from the bindings are heap views, so they are copied (slice) here.
function vizData(withMap) {
  const v = { kps: slam.frameKeyPoints().slice(), ini: slam.initMatches().slice(), stats: Array.from(slam.mapStats()) };
  if (withMap) {
    v.mps = slam.mapPoints().slice();
    v.kfs = slam.keyFrames().slice();
    v.edges = slam.graphEdges().slice();
  }
  return v;
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'init') {
      const t0 = performance.now();
      M = await createOrbSlam3({ print: log, printErr: log });
      // download progress: bytes received (compressed, for .gz) vs Content-Length, posted ~10x per second
      const counted = (r) => {
        const total = Number(r.headers.get('Content-Length')) || 0;
        let loaded = 0, last = 0;
        const body = r.body.pipeThrough(new TransformStream({
          transform(chunk, ctl) {
            loaded += chunk.byteLength;
            const now = performance.now();
            if (now - last > 100) { last = now; postMessage({ type: 'progress', stage: 'download', loaded, total }); }
            ctl.enqueue(chunk);
          },
        }));
        return msg.vocabUrl.endsWith('.gz') ? body.pipeThrough(new DecompressionStream('gzip')) : body;
      };
      const [voc, settings] = await Promise.all([
        fetch(msg.vocabUrl).then((r) => new Response(counted(r)).arrayBuffer()),  // Pages ships ORBvoc.bin.gz
        msg.settingsText ?? fetch(msg.settingsUrl).then((r) => r.text()),  // settingsText: generated (live camera)
      ]);
      postMessage({ type: 'progress', stage: 'build', loaded: voc.byteLength, total: voc.byteLength });
      M.FS.writeFile('/ORBvoc.bin', new Uint8Array(voc));
      M.FS.writeFile('/settings.yaml', settings);
      slam = new M.WebSlam();
      slam.init('/ORBvoc.bin', '/settings.yaml', !!msg.clahe, msg.seed ?? -1);
      M.FS.unlink('/ORBvoc.bin');  // the vocabulary is in the tree now; free the 48 MB file copy
      postMessage({ type: 'ready', ms: performance.now() - t0, vocabBytes: voc.byteLength, heap: M.HEAPU8.length });
    } else if (msg.type === 'frame') {
      const td = performance.now();
      const { img, bmp } = await decode(msg.url, !!msg.viz);
      const w = img.width, h = img.height;
      const ptr = slam.frameBuffer(w, h);
      toGray(img.data, M.HEAPU8.subarray(ptr, ptr + w * h));  // HEAPU8 re-read: memory may have grown
      const decodeMs = performance.now() - td;
      const state = slam.track(msg.t);
      const out = { type: 'pose', idx: msg.idx, t: msg.t, state, twc: Array.from(slam.poseTwc()),
                    trackMs: slam.lastTrackMs(), decodeMs, heap: M.HEAPU8.length };
      if (msg.viz) {
        const tv = performance.now();
        out.viz = vizData(!!msg.withMap);
        out.viz.ms = performance.now() - tv;
        out.bmp = bmp;
        const transfer = [bmp, out.viz.kps.buffer, out.viz.ini.buffer];
        if (out.viz.mps) transfer.push(out.viz.mps.buffer, out.viz.kfs.buffer, out.viz.edges.buffer);
        postMessage(out, transfer);
      } else {
        postMessage(out);
      }
    } else if (msg.type === 'bitmap') {  // live camera: {bitmap, t[, viz, withMap]}; the bitmap is sent back for display
      const bmp = msg.bitmap;
      if (!canvas || canvas.width !== bmp.width || canvas.height !== bmp.height) {
        canvas = new OffscreenCanvas(bmp.width, bmp.height);
        ctx = canvas.getContext('2d', { willReadFrequently: true });
      }
      const td = performance.now();
      ctx.drawImage(bmp, 0, 0);
      const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const ptr = slam.frameBuffer(img.width, img.height);
      toGray(img.data, M.HEAPU8.subarray(ptr, ptr + img.width * img.height));
      const decodeMs = performance.now() - td;
      const state = slam.track(msg.t);
      const out = { type: 'pose', t: msg.t, state, twc: Array.from(slam.poseTwc()), trackMs: slam.lastTrackMs(),
                    decodeMs, heap: M.HEAPU8.length, bmp };
      const transfer = [bmp];
      if (msg.viz) {
        out.viz = vizData(!!msg.withMap);
        transfer.push(out.viz.kps.buffer, out.viz.ini.buffer);
        if (out.viz.mps) transfer.push(out.viz.mps.buffer, out.viz.kfs.buffer, out.viz.edges.buffer);
      }
      postMessage(out, transfer);
    } else if (msg.type === 'camera') {  // {readable}: WebCodecs camera frames; replies with the first frame's properties
      camReader = msg.readable.getReader();
      camLoop();
      const first = await Promise.race([nextCamFrame().then(() => true), new Promise((res) => setTimeout(() => res(false), 3000))]);
      const f = camLatest;
      postMessage({ type: 'camera', ok: first && !!f, format: f && f.format, codedW: f && f.codedWidth, codedH: f && f.codedHeight,
                    displayW: f && f.displayWidth, displayH: f && f.displayHeight, visibleW: f && f.visibleRect.width,
                    visibleH: f && f.visibleRect.height, rotation: f && 'rotation' in f ? f.rotation : null });
    } else if (msg.type === 'cameraStop') {
      if (camReader) { camReader.cancel().catch(() => {}); camReader = null; }
      if (camLatest) { camLatest.close(); camLatest = null; }
      postMessage({ type: 'cameraStop' });
    } else if (msg.type === 'trackCamera') {  // {viz, withMap}: track the newest camera frame; the frame is sent back
      await nextCamFrame();
      const f = camLatest; camLatest = null;
      const td = performance.now();
      await frameToGray(f);
      const decodeMs = performance.now() - td;
      const t = f.timestamp / 1e6;  // capture timestamp (microseconds) -> seconds
      const state = slam.track(t);
      const out = { type: 'pose', t, state, twc: Array.from(slam.poseTwc()), trackMs: slam.lastTrackMs(), decodeMs,
                    heap: M.HEAPU8.length, bmp: f, cam: [camFrames, camDropped] };
      const transfer = [f];
      if (msg.viz) {
        out.viz = vizData(!!msg.withMap);
        transfer.push(out.viz.kps.buffer, out.viz.ini.buffer);
        if (out.viz.mps) transfer.push(out.viz.mps.buffer, out.viz.kfs.buffer, out.viz.edges.buffer);
      }
      postMessage(out, transfer);
    } else if (msg.type === 'reset') {  // ORB-SLAM3 ResetActiveMap (applied on the next frame)
      slam.reset();
      postMessage({ type: 'reset' });
    } else if (msg.type === 'calibInit') {  // camera calibration (calib.html): the module without SLAM / vocabulary
      M = M || await createOrbSlam3({ print: log, printErr: log });
      cal = new M.Calibrator();
      postMessage({ type: 'calibReady' });
    } else if (msg.type === 'calibFrame') {  // {bitmap, cols, rows}: chessboard corners (x, y)*; the bitmap is sent back
      const bmp = msg.bitmap;
      if (!canvas || canvas.width !== bmp.width || canvas.height !== bmp.height) {
        canvas = new OffscreenCanvas(bmp.width, bmp.height);
        ctx = canvas.getContext('2d', { willReadFrequently: true });
      }
      ctx.drawImage(bmp, 0, 0);
      const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const ptr = cal.frameBuffer(img.width, img.height);
      toGray(img.data, M.HEAPU8.subarray(ptr, ptr + img.width * img.height));
      const t0 = performance.now();
      const corners = cal.detect(msg.cols, msg.rows).slice();
      postMessage({ type: 'corners', corners, ms: performance.now() - t0, bmp }, [bmp, corners.buffer]);
    } else if (msg.type === 'calibAdd') {
      postMessage({ type: 'views', n: cal.addView() });
    } else if (msg.type === 'calibRemove') {
      postMessage({ type: 'views', n: cal.removeView(msg.index) });
    } else if (msg.type === 'calibClear') {
      cal.clear();
      postMessage({ type: 'views', n: 0 });
    } else if (msg.type === 'calibSolve') {  // -> [rms, fx, fy, cx, cy, k1, k2, p1, p2, per-view rms...]
      postMessage({ type: 'calib', r: Array.from(cal.solve(msg.cols, msg.rows)) });
    } else if (msg.type === 'keepMap') {  // AR anchor placed/removed: relocalise in the map instead of replacing it
      slam.setKeepMap(!!msg.on);
      postMessage({ type: 'keepMap', on: !!msg.on });
    } else if (msg.type === 'finish') {
      slam.finishAndSave('/f.txt', '/kf.txt');
      const stats = {};
      for (const f of ['TrackingTimeStats.txt', 'LocalMapTimeStats.txt', 'LBA_Stats.txt', 'ExecMean.txt']) {
        try { stats[f] = M.FS.readFile('/' + f, { encoding: 'utf8' }); } catch { /* not a profiling build */ }
      }
      postMessage({ type: 'finished', frames: M.FS.readFile('/f.txt', { encoding: 'utf8' }),
                    keyframes: M.FS.readFile('/kf.txt', { encoding: 'utf8' }), stats });
    }
  } catch (err) {
    postMessage({ type: 'error', text: String(err && err.stack || err) });
  }
};
