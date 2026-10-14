// Web Worker hosting the ORB-SLAM3 WebAssembly module (web/build/orbslam3.js).
// ORB-SLAM3's own threads (local mapping, loop closing) become nested workers via emscripten pthreads.
//
// Messages in:   {type:'init', vocabUrl, settingsUrl | settingsText, clahe, seed}
//                {type:'bitmap', bitmap, t[, viz, withMap]}   (live camera frame as an ImageBitmap)
//                {type:'frame', idx, t, url[, viz, withMap]}  (worker fetches + decodes the image; viz: viewer data
//                                                              + the decoded ImageBitmap in the reply)
//                {type:'finish'}
// Messages out:  {type:'log', text} {type:'ready', ms} {type:'pose', idx, t, state, twc, trackMs, decodeMs}
//                {type:'finished', frames, keyframes}  (ORB-SLAM3 final trajectories, EuRoC format)
import createOrbSlam3 from './build/orbslam3.js';

let M = null;
let slam = null;
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
      const [voc, settings] = await Promise.all([
        fetch(msg.vocabUrl).then((r) => r.arrayBuffer()),
        msg.settingsText ?? fetch(msg.settingsUrl).then((r) => r.text()),  // settingsText: generated (live camera)
      ]);
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
