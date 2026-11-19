// AlvaAR baseline driver (github.com/alanross/AlvaAR, the released bundle dist/alva_ar.js, used unmodified).
//
// Same command line and output files as baselines/orbslam3/orbslam3_run.cc, so scripts/seed_sweep.sh runs it with
//   FLAVOURS=wasm WASM_JS=<this file> EXTRA="--rt-wall" (or "--live --rate-scale R")
//
// Usage:
//   node alvaar_run.mjs --dataset euroc|tumvi --seq <dir> --settings <yaml> --out <dir> (--rt-wall | --live)
//                       [--rate-scale R] [--seed N] [--alva dist/alva_ar.js] [--mode mono] [--vocab ignored]
//
// Outputs (in --out):
//   traj_tum.txt          online poses (Twc) of the longest tracked segment, TUM format, absolute dataset time (s)
//   traj_seg<i>_tum.txt   every segment: AlvaAR has no relocalisation; on tracking loss it resets and starts a new map
//                         in a new world frame (status 2), so each reset starts a segment
//   frames.txt            per processed frame: index, time, status (1 tracked, 2 reset, 3 initialising), ms
//   track_times.txt       per processed frame: index, findCameraPose wall time (ms), status
//   run_info.txt          the keys of orbslam3_run's run_info.txt, plus resets / segments / tracked frames
//
// AlvaAR as released:
//  - findCameraPose() takes no timestamp: the frame time is the wall clock (ms) and the motion model divides by its
//    gap, so it only runs on a wall-clock schedule (--rt-wall or --live; an unpaced replay is refused).
//  - The JS wrapper (AlvaAR class) derives the intrinsics from a 45 degree field of view and keeps typed-array views
//    that go stale when the WASM heap grows. This driver uses the module's System class directly: configure() with
//    the calibration from the settings file (pinhole + radtan; AlvaAR undistorts keypoints), and fresh heap views.
//  - It has no seed (RANSAC is seeded from std::random_device); --seed is only the run index.
//
// Frames are decoded on a worker thread (PNG: node:zlib, byte-exact with OpenCV; JPEG: jpeg_gray.mjs, OpenCV 4.10's
// imdecode compiled to WebAssembly, the call WebVIO's runner makes; build_jpeg_gray.sh) with orbslam3_run's
// FrameSource pacing: --rt-wall plays
// every frame, not before its time on the wall clock; --live gives the tracker the newest arrived frame (late frames
// are dropped). --rate-scale multiplies the frame period (0.5 = frames arrive twice as fast = a device half as fast).

import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ALVA_COMMIT = '7796af500ee92001ac2a9888363ff64d7a3bee75';
const nowMs = () => performance.timeOrigin + performance.now();  // absolute, comparable across threads

// ---------------------------------------------------------------- PNG (grayscale 8/16 bit) -> RGBA
// 16-bit samples keep the high byte, as OpenCV's IMREAD_GRAYSCALE (libpng strip_16).
function decodePngGray(buf, out) {
  if (buf.readUInt32BE(0) !== 0x89504e47) throw new Error('not a PNG');
  let off = 8, w = 0, h = 0, depth = 0, ctype = 0;
  const idat = [];
  while (off < buf.length) {
    const len = buf.readUInt32BE(off), type = buf.toString('latin1', off + 4, off + 8);
    if (type === 'IHDR') {
      w = buf.readUInt32BE(off + 8); h = buf.readUInt32BE(off + 12);
      depth = buf[off + 16]; ctype = buf[off + 17];
      if (buf[off + 20] !== 0) throw new Error('interlaced PNG not supported');
    } else if (type === 'IDAT') idat.push(buf.subarray(off + 8, off + 8 + len));
    else if (type === 'IEND') break;
    off += 12 + len;
  }
  if (ctype !== 0 || (depth !== 8 && depth !== 16)) throw new Error(`PNG colour type ${ctype} / depth ${depth} not supported`);
  const bpp = depth / 8, stride = w * bpp;
  const raw = zlib.inflateSync(Buffer.concat(idat));
  const cur = new Uint8Array(stride), prev = new Uint8Array(stride);
  for (let y = 0; y < h; ++y) {
    const p = y * (stride + 1), f = raw[p];
    for (let i = 0; i < stride; ++i) {
      const x = raw[p + 1 + i], a = i >= bpp ? cur[i - bpp] : 0, b = prev[i], c = i >= bpp ? prev[i - bpp] : 0;
      let v;
      switch (f) {
        case 0: v = x; break;
        case 1: v = x + a; break;
        case 2: v = x + b; break;
        case 3: v = x + ((a + b) >> 1); break;
        case 4: {
          const pa = Math.abs(b - c), pb = Math.abs(a - c), pc = Math.abs(a + b - 2 * c);
          v = x + (pa <= pb && pa <= pc ? a : pb <= pc ? b : c);
          break;
        }
        default: throw new Error(`bad PNG filter ${f}`);
      }
      cur[i] = v & 255;
    }
    const o = y * w * 4;
    for (let i = 0; i < w; ++i) {
      const g = cur[i * bpp];  // high byte first (PNG is big-endian)
      out[o + 4 * i] = out[o + 4 * i + 1] = out[o + 4 * i + 2] = g;
      out[o + 4 * i + 3] = 255;
    }
    prev.set(cur);
  }
  return { w, h };
}

// ---------------------------------------------------------------- JPEG -> RGBA (gray), via jpeg_gray.mjs
async function makeJpegDecoder() {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const { default: create } = await import(pathToFileURL(path.join(here, 'jpeg_gray.mjs')).href);
  const J = await create();
  const whPtr = J._malloc(8);
  let inPtr = 0, inCap = 0, outPtr = 0, outCap = 0;
  return (buf, out) => {
    if (buf.length > inCap) { if (inPtr) J._free(inPtr); inCap = 2 * buf.length; inPtr = J._malloc(inCap); }
    if (out.length / 4 > outCap) { if (outPtr) J._free(outPtr); outCap = out.length / 4; outPtr = J._malloc(outCap); }
    J.HEAPU8.set(buf, inPtr);  // J.HEAPU8 is re-read after every call: the heap may have grown
    const rc = J._decode_gray(inPtr, buf.length, outPtr, outCap, whPtr);
    if (rc !== 0) throw new Error(`JPEG decode failed (${rc})`);
    const w = J.HEAP32[whPtr >> 2], h = J.HEAP32[(whPtr >> 2) + 1], g = J.HEAPU8.subarray(outPtr, outPtr + w * h);
    for (let i = 0; i < w * h; ++i) { out[4 * i] = out[4 * i + 1] = out[4 * i + 2] = g[i]; out[4 * i + 3] = 255; }
    return { w, h };
  };
}

// ---------------------------------------------------------------- frame source (shared memory, 3 image slots)
// ctl: 0 readyIdx, 1 readySlot, 2 taken, 3 busySlot, 4 done, 5 lock, 6 change counter, 7 sleep dummy
const C_READY = 0, C_RSLOT = 1, C_TAKEN = 2, C_BUSY = 3, C_DONE = 4, C_LOCK = 5, C_SEQ = 6, C_SLEEP = 7;
const lock = (c) => { while (Atomics.compareExchange(c, C_LOCK, 0, 1) !== 0); };
const unlock = (c) => { Atomics.store(c, C_LOCK, 0); };
const bump = (c) => { Atomics.add(c, C_SEQ, 1); Atomics.notify(c, C_SEQ); };
const sleepMs = (c, ms) => { if (ms > 0) Atomics.wait(c, C_SLEEP, 0, ms); };

async function frameWorker() {
  const { images, t, rate, camera, wall0, k0, sab, slotBytes, width, height } = workerData;
  const jpeg = images.some((f) => /\.jpe?g$/i.test(f)) ? await makeJpegDecoder() : null;
  const decode = (file, out) => {
    const { w, h } = (jpeg && /\.jpe?g$/i.test(file) ? jpeg : decodePngGray)(fs.readFileSync(file), out);
    if (w !== width || h !== height) throw new Error(`${file}: ${w}x${h}, settings say ${width}x${height}`);
  };
  const ctl = new Int32Array(sab, 0, 8);
  const slots = [0, 1, 2].map((s) => new Uint8ClampedArray(sab, 64 + s * slotBytes, slotBytes));
  const arrival = (k) => rate * (t[k] - t[k0]) * 1000;  // ms after start
  const now = () => nowMs() - wall0;
  const n = images.length;
  let k = k0, loaded = k0 - 1;
  const publish = (idx) => {
    lock(ctl);
    const r = Atomics.load(ctl, C_RSLOT), b = Atomics.load(ctl, C_BUSY);
    let s = 0;
    while (s === r || s === b) ++s;
    unlock(ctl);
    // slot s is neither the ready one nor the one being copied; only this thread changes readySlot
    decode(images[idx], slots[s]);
    lock(ctl);
    Atomics.store(ctl, C_RSLOT, s);
    Atomics.store(ctl, C_READY, idx);
    unlock(ctl);
    loaded = idx;
    bump(ctl);
  };
  while (true) {
    if (camera) {  // newest frame that has arrived (wait for the next one if none is new)
      const tn = now();
      while (k + 1 < n && arrival(k + 1) <= tn) ++k;
      if (k <= loaded) {
        if (k + 1 >= n) break;
        sleepMs(ctl, Math.max(0, arrival(k + 1) - tn));
        continue;
      }
      publish(k);
    } else {  // in order: wait until the tracker took the previous frame
      for (;;) {
        const seq = Atomics.load(ctl, C_SEQ);
        if (Atomics.load(ctl, C_READY) <= Atomics.load(ctl, C_TAKEN)) break;
        Atomics.wait(ctl, C_SEQ, seq, 100);
      }
      k = loaded + 1;
      if (k >= n) break;
      publish(k);
      if (k + 1 >= n) break;
    }
  }
  Atomics.store(ctl, C_DONE, 1);
  bump(ctl);
}

const isEntry = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (!isMainThread) await frameWorker();
else if (isEntry) await main();
export { decodePngGray, makeJpegDecoder, quat };  // for tests

// ---------------------------------------------------------------- main
function parseArgs(argv) {
  const a = { rateScale: 1, seed: -1, live: false, rtWall: false, clahe: false, mode: 'mono',
              alva: path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../ext/AlvaAR/dist/alva_ar.js') };
  for (let i = 2; i < argv.length; ++i) {
    const k = argv[i], next = () => argv[++i];
    if (k === '--dataset') a.dataset = next();
    else if (k === '--seq') a.seq = next();
    else if (k === '--settings') a.settings = next();
    else if (k === '--out') a.out = next();
    else if (k === '--seed') a.seed = Number(next());
    else if (k === '--rate-scale') a.rateScale = Number(next());
    else if (k === '--live') a.live = true;
    else if (k === '--rt-wall') a.rtWall = true;
    else if (k === '--alva') a.alva = next();
    else if (k === '--mode') a.mode = next();
    else if (k === '--vocab') next();  // AlvaAR has no vocabulary (accepted for seed_sweep.sh)
    else if (k === '--clahe') a.clahe = true;
    else throw new Error(`unknown argument ${k}`);
  }
  if (!a.seq || !a.settings || !a.out) throw new Error('usage: alvaar_run.mjs --dataset D --seq DIR --settings S --out DIR (--rt-wall | --live)');
  if (!a.live && !a.rtWall) throw new Error('AlvaAR stamps frames with the wall clock: use --rt-wall or --live');
  if (a.mode !== 'mono') throw new Error('AlvaAR is monocular only (its IMU entry point ignores the samples)');
  if (a.clahe) throw new Error('--clahe is not supported yet (AlvaAR has CLAHE compiled off)');
  return a;
}

function readSettings(file) {
  const txt = fs.readFileSync(file, 'utf8');
  const get = (key, def) => {
    const m = txt.match(new RegExp(`^${key.replace('.', '\\.')}:\\s*"?([^"\\s#]+)`, 'm'));
    if (m) return m[1];
    if (def === undefined) throw new Error(`${key} missing in ${file}`);
    return def;
  };
  const type = get('Camera.type');
  if (type !== 'PinHole') throw new Error(`camera type ${type}: AlvaAR has only pinhole + radtan`);
  const num = (k, d) => Number(get(k, d));
  return { width: num('Camera.width'), height: num('Camera.height'),
           fx: num('Camera1.fx'), fy: num('Camera1.fy'), cx: num('Camera1.cx'), cy: num('Camera1.cy'),
           k1: num('Camera1.k1', 0), k2: num('Camera1.k2', 0), p1: num('Camera1.p1', 0), p2: num('Camera1.p2', 0) };
}

// EuRoC / TUM-VI ASL layout, as orbslam3_run's loadAsl (every cam0 frame with an image file; integer-ns rebase)
function loadAsl(seq) {
  const cam = path.join(seq, 'mav0', 'cam0');
  const rows = fs.readFileSync(path.join(cam, 'data.csv'), 'utf8').split('\n')
    .map((l) => l.trim()).filter((l) => l && !l.startsWith('#'))
    .map((l) => { const c = l.indexOf(','); return [BigInt(l.slice(0, c)), l.slice(c + 1).trim()]; });
  const ns0 = rows[0][0];
  const images = [], t = [];
  for (const [ns, name] of rows) {
    const p = path.join(cam, 'data', name);
    if (!fs.existsSync(p)) continue;
    images.push(p);
    t.push(Number(ns - ns0) * 1e-9);
  }
  if (images.length < rows.length) console.error(`WARNING: ${rows.length - images.length} frames have no image file; skipped`);
  return { images, t, tBase: Number(ns0) * 1e-9 };
}

// R (row-major in the pose array: R(i,j) = p[4i+j]) -> quaternion x y z w
function quat(p) {
  const r = (i, j) => p[4 * i + j];
  const tr = r(0, 0) + r(1, 1) + r(2, 2);
  let x, y, z, w;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    w = 0.25 * s; x = (r(2, 1) - r(1, 2)) / s; y = (r(0, 2) - r(2, 0)) / s; z = (r(1, 0) - r(0, 1)) / s;
  } else if (r(0, 0) > r(1, 1) && r(0, 0) > r(2, 2)) {
    const s = Math.sqrt(1 + r(0, 0) - r(1, 1) - r(2, 2)) * 2;
    w = (r(2, 1) - r(1, 2)) / s; x = 0.25 * s; y = (r(0, 1) + r(1, 0)) / s; z = (r(0, 2) + r(2, 0)) / s;
  } else if (r(1, 1) > r(2, 2)) {
    const s = Math.sqrt(1 + r(1, 1) - r(0, 0) - r(2, 2)) * 2;
    w = (r(0, 2) - r(2, 0)) / s; x = (r(0, 1) + r(1, 0)) / s; y = 0.25 * s; z = (r(1, 2) + r(2, 1)) / s;
  } else {
    const s = Math.sqrt(1 + r(2, 2) - r(0, 0) - r(1, 1)) * 2;
    w = (r(1, 0) - r(0, 1)) / s; x = (r(0, 2) + r(2, 0)) / s; y = (r(1, 2) + r(2, 1)) / s; z = 0.25 * s;
  }
  return [x, y, z, w];
}

async function main() {
  const a = parseArgs(process.argv);
  fs.mkdirSync(a.out, { recursive: true });
  const cam = readSettings(a.settings);
  const d = loadAsl(a.seq);
  const N = d.images.length;

  const bundle = fs.readFileSync(a.alva);
  const bundleSha = createHash('sha256').update(bundle).digest('hex');
  const { AlvaAR } = await import(pathToFileURL(path.resolve(a.alva)).href);
  // The wrapper builds the module; its System (FOV intrinsics) is replaced by one configured with the calibration.
  const wrapper = await AlvaAR.Initialize(cam.width, cam.height);
  const M = wrapper.wasm.module;
  wrapper.system.delete();
  const sys = new M.System();
  sys.configure(cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy, cam.k1, cam.k2, cam.p1, cam.p2);
  const imgBytes = cam.width * cam.height * 4;
  const imgPtr = M._malloc(imgBytes), posePtr = M._malloc(16 * 4);

  // frame source
  const slotBytes = Math.ceil(imgBytes / 64) * 64;
  const sab = new SharedArrayBuffer(64 + 3 * slotBytes);
  const ctl = new Int32Array(sab, 0, 8);
  ctl.set([-1, -1, -1, -1, 0, 0, 0, 0]);
  const slots = [0, 1, 2].map((s) => new Uint8ClampedArray(sab, 64 + s * slotBytes, slotBytes));
  const wall0 = nowMs();
  const worker = new Worker(fileURLToPath(import.meta.url),
    { workerData: { images: d.images, t: d.t, rate: a.rateScale, camera: a.live, wall0, k0: 0, sab, slotBytes,
                    width: cam.width, height: cam.height } });
  let workerError = null;
  worker.on('error', (e) => { workerError = e; });
  const arrival = (k) => a.rateScale * (d.t[k] - d.t[0]) * 1000;
  const now = () => nowMs() - wall0;
  const next = () => {  // blocks until a frame newer than the last one taken is ready; its index, or -1 at the end
    for (;;) {
      const seq = Atomics.load(ctl, C_SEQ);
      lock(ctl);
      const ready = Atomics.load(ctl, C_READY), taken = Atomics.load(ctl, C_TAKEN);
      if (ready > taken) {
        const s = Atomics.load(ctl, C_RSLOT);
        Atomics.store(ctl, C_BUSY, s);
        Atomics.store(ctl, C_TAKEN, ready);
        unlock(ctl);
        M.HEAPU8.set(slots[s].subarray(0, imgBytes), imgPtr);  // fresh heap view (the heap may have grown)
        lock(ctl); Atomics.store(ctl, C_BUSY, -1); unlock(ctl);
        bump(ctl);
        return ready;
      }
      unlock(ctl);
      if (Atomics.load(ctl, C_DONE)) return -1;
      if (workerError) throw workerError;
      Atomics.wait(ctl, C_SEQ, seq, 100);
    }
  };

  const fmt = (x) => x.toPrecision(17);
  const framesF = fs.openSync(path.join(a.out, 'frames.txt'), 'w');
  fs.writeSync(framesF, '# index t_rel status ms (status: 1 tracked, 2 reset, 3 initialising)\n');
  const timesF = fs.openSync(path.join(a.out, 'track_times.txt'), 'w');
  const segs = [];  // { file, poses }
  let seg = null, resets = 0, processed = 0, dropped = 0, lastIdx = -1, tracked = 0, error = '';
  const newSeg = () => {
    const file = path.join(a.out, `traj_seg${segs.length}_tum.txt`);
    seg = { file, fd: fs.openSync(file, 'w'), poses: 0 };
    fs.writeSync(seg.fd, '# t tx ty tz qx qy qz qw\n');
    segs.push(seg);
  };
  try {
    for (;;) {
      const k = next();
      if (k < 0) break;
      dropped += k - lastIdx - 1;
      if (a.rtWall) sleepMs(ctl, arrival(k) - now());  // not before the frame's time (no sleep when behind)
      lastIdx = k;
      ++processed;
      const t0 = performance.now();
      const status = sys.findCameraPose(imgPtr, posePtr);
      const ms = performance.now() - t0;
      fs.writeSync(timesF, `${k} ${ms.toFixed(3)} ${status}\n`);  // orbslam3_run's columns: index, ms, state
      fs.writeSync(framesF, `${k} ${d.t[k].toFixed(6)} ${status} ${ms.toFixed(3)}\n`);
      if (status === 1) {
        if (!seg) newSeg();
        const p = M.HEAPF32.slice(posePtr / 4, posePtr / 4 + 16);
        const [qx, qy, qz, qw] = quat(p);
        fs.writeSync(seg.fd, `${fmt(d.t[k] + d.tBase)} ${p[12]} ${p[13]} ${p[14]} ${qx} ${qy} ${qz} ${qw}\n`);
        ++seg.poses; ++tracked;
      } else if (status === 2) {
        ++resets;
        if (seg) { fs.closeSync(seg.fd); seg = null; }
      }
    }
  } catch (e) {  // an abort / exit() inside AlvaAR is a result: keep the trajectory so far
    error = String(e && e.message ? e.message : e).replace(/\s+/g, ' ');
    console.error('AlvaAR stopped:', error);
  }
  const wallS = now() / 1000;
  if (seg) fs.closeSync(seg.fd);
  fs.closeSync(framesF); fs.closeSync(timesF);
  await worker.terminate();

  // traj_tum.txt = the longest segment (the counterpart of ORB-SLAM3's "largest map")
  let best = null;
  for (const s of segs) if (s.poses > 0 && (!best || s.poses > best.poses)) best = s;
  if (best) fs.copyFileSync(best.file, path.join(a.out, 'traj_tum.txt'));

  const seqS = processed > 0 ? (lastIdx >= 0 ? d.t[lastIdx] - d.t[0] : 0) : 0;
  const info = {
    dataset: a.dataset, mode: a.mode, seq: a.seq, settings: a.settings, clahe: 0, seed: a.seed, realtime: 0,
    rate_scale: a.rateScale, live: a.live ? 1 : 0, rt_wall: a.rtWall ? 1 : 0, start_frame: 0, frames: N,
    processed, dropped, wall_s: wallS, seq_s: seqS, effective_rate: wallS > 0 ? seqS / wallS : 0,
    tracked_frames: tracked, resets, segments: segs.filter((s) => s.poses > 0).length,
    longest_segment: best ? best.poses : 0, mem_mb: M.HEAPU8.length / 1048576, t_base: d.tBase,
    system: 'alvaar', alvaar_commit: ALVA_COMMIT, bundle: a.alva, bundle_sha256: bundleSha, node: process.version,
    intrinsics: `${cam.fx} ${cam.fy} ${cam.cx} ${cam.cy} ${cam.k1} ${cam.k2} ${cam.p1} ${cam.p2}`,
    error: error || 'none',
  };
  fs.writeFileSync(path.join(a.out, 'run_info.txt'), Object.entries(info).map(([k, v]) => `${k} ${v}`).join('\n') + '\n');
  console.log(`frames ${N} processed ${processed} dropped ${dropped} tracked ${tracked} resets ${resets} ` +
              `longest ${best ? best.poses : 0} wall ${wallS.toFixed(1)} s${error ? ' ERROR ' + error : ''}`);
  process.exit(0);
}
