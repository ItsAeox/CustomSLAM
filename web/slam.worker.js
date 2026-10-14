// Web Worker hosting the ORB-SLAM3 WebAssembly module (web/build/orbslam3.js).
// ORB-SLAM3's own threads (local mapping, loop closing) become nested workers via emscripten pthreads.
//
// Messages in:   {type:'init', vocabUrl, settingsUrl, clahe}
//                {type:'frame', idx, t, url}            (worker fetches + decodes the image)
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

async function decode(url) {
  const blob = await (await fetch(url)).blob();
  const bmp = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  if (!canvas || canvas.width !== bmp.width || canvas.height !== bmp.height) {
    canvas = new OffscreenCanvas(bmp.width, bmp.height);
    ctx = canvas.getContext('2d', { willReadFrequently: true });
  }
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'init') {
      const t0 = performance.now();
      M = await createOrbSlam3({ print: log, printErr: log });
      const [voc, settings] = await Promise.all([
        fetch(msg.vocabUrl).then((r) => r.arrayBuffer()),
        fetch(msg.settingsUrl).then((r) => r.text()),
      ]);
      M.FS.writeFile('/ORBvoc.bin', new Uint8Array(voc));
      M.FS.writeFile('/settings.yaml', settings);
      slam = new M.WebSlam();
      slam.init('/ORBvoc.bin', '/settings.yaml', !!msg.clahe);
      postMessage({ type: 'ready', ms: performance.now() - t0, vocabBytes: voc.byteLength });
    } else if (msg.type === 'frame') {
      const td = performance.now();
      const img = await decode(msg.url);
      const w = img.width, h = img.height;
      const ptr = slam.frameBuffer(w, h);
      toGray(img.data, M.HEAPU8.subarray(ptr, ptr + w * h));  // HEAPU8 re-read: memory may have grown
      const decodeMs = performance.now() - td;
      const state = slam.track(msg.t);
      postMessage({ type: 'pose', idx: msg.idx, t: msg.t, state, twc: Array.from(slam.poseTwc()),
                    trackMs: slam.lastTrackMs(), decodeMs });
    } else if (msg.type === 'finish') {
      slam.finishAndSave('/f.txt', '/kf.txt');
      postMessage({ type: 'finished', frames: M.FS.readFile('/f.txt', { encoding: 'utf8' }),
                    keyframes: M.FS.readFile('/kf.txt', { encoding: 'utf8' }) });
    }
  } catch (err) {
    postMessage({ type: 'error', text: String(err && err.stack || err) });
  }
};
