// gpuorb.html worker: WebGPU ORB (gpuorb.js) against ORB-SLAM3's own extractor (the WASM module's OrbGpuHost) on frames
// of a phone recording: keypoints (position, level, angle, response, size, order) and descriptors must be identical;
// also times both paths. Frames are decoded before the timed loop. refEvery = N: the CPU reference (and the comparison)
// only on every Nth frame, so the GPU path runs nearly back to back as in tracking. Messages out: { line } / { done } /
// { error }.
import { createGpuOrb } from './gpuorb.js';

const W = 480, H = 640;
const post = (line) => postMessage({ line });
const stats = (v) => {
  if (!v.length) return 'n/a';
  const s = [...v].sort((a, b) => a - b), q = (p) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  return `median ${q(0.5).toFixed(2)} ms, p95 ${q(0.95).toFixed(2)}`;
};

async function gray(url) {
  const bmp = await createImageBitmap(await (await fetch(url)).blob());
  if (bmp.width !== W || bmp.height !== H) throw new Error(`${url}: ${bmp.width}x${bmp.height}, expected ${W}x${H}`);
  const c = new OffscreenCanvas(W, H), x = c.getContext('2d');
  x.drawImage(bmp, 0, 0);
  bmp.close();
  const rgba = x.getImageData(0, 0, W, H).data, g = new Uint8Array(W * H);
  for (let i = 0; i < g.length; i++) g[i] = (rgba[4 * i] * 77 + rgba[4 * i + 1] * 150 + rgba[4 * i + 2] * 29) >> 8;
  return g;
}

onmessage = async (e) => {
  try {
    const { frames, build, nf = 1000, levels = 8, refEvery = 1 } = e.data;
    const createOrbSlam3 = (await import(build)).default;
    const M = await createOrbSlam3({ print: post, printErr: post });
    const host = new M.OrbGpuHost(nf, 1.2, levels, 20, 7, W, H);
    const consts = new Int32Array(host.consts());  // copy out of the module heap
    if (!navigator.gpu) throw new Error('WebGPU not available here');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    const info = adapter.info || {};
    post(`adapter: ${info.vendor || '?'} | ${info.architecture || '?'}; ORB ${nf} features, ${levels} levels; ${consts.length} constants (${(consts.length * 4 / 1024).toFixed(0)} KB); CPU reference every ${refEvery} frame(s)`);
    const device = await adapter.requestDevice();
    const g = await createGpuOrb(device, consts, W, H);
    const imgs = [];
    for (const f of frames) imgs.push(await gray(f));
    const tCpu = [], tGpu = [], t1 = [], tHost = [], t2 = [];
    let exact = 0, checked = 0, kpDiff = 0, descDiff = 0, bits = 0, kpTotal = 0, fallbacks = 0;
    for (let f = 0; f < imgs.length; f++) {
      const img = imgs[f], ref = f % refEvery === 0;
      M.HEAPU8.set(img, host.frameBuffer());
      let t = performance.now();
      if (ref) { host.reference(); tCpu.push(performance.now() - t); }
      t = performance.now();
      const c = await g.pass1(img);
      const ta = performance.now();
      new Uint32Array(M.HEAPU8.buffer, host.candBuffer(c.n), c.n * 4).set(c.cand);
      const k = c.total > c.n ? -1 : host.distribute(c.n);  // overflow: the tracker would use CPU ORB
      if (k < 0) { fallbacks++; post(`${frames[f].split('/').pop()}: overflow (${c.total} candidates) -> CPU ORB`); continue; }
      const p2 = new Uint32Array(host.pass2());
      const tb = performance.now();
      const d = await g.pass2(p2, k);
      const tc = performance.now();
      if (f > 0) { tGpu.push(tc - t); t1.push(ta - t); tHost.push(tb - ta); t2.push(tc - tb); }  // frame 0: warm-up
      if (!ref) continue;
      new Uint32Array(M.HEAPU8.buffer, host.descBuffer(k), k * 8).set(d);
      const r = new Int32Array(host.compare(k));  // keypoints differ, descriptors differ, bits, ref count, GPU count
      checked++; kpDiff += r[0]; descDiff += r[1]; bits += r[2]; kpTotal += r[3];
      exact += r[0] === 0 && r[1] === 0;
      post(`${frames[f].split('/').pop()}: ${c.total} candidates, keypoints ${r[3]} CPU / ${r[4]} GPU, differ ${r[0]}; descriptors differ ${r[1]} (${r[2]} bits); CPU ${tCpu[tCpu.length - 1].toFixed(1)} ms, GPU path ${(tc - t).toFixed(1)} ms`);
    }
    post(`TOTAL ${checked} frames checked, ${exact} exact; keypoints differ ${kpDiff} of ${kpTotal}; descriptors differ ${descDiff} (${bits} bits); overflows ${fallbacks}; second candidate readbacks ${g.extraReadbacks}`);
    post(`CPU ORB (ORBextractor, WASM): ${stats(tCpu)}`);
    post(`GPU path (${tGpu.length} frames): ${stats(tGpu)} = pass 1 ${stats(t1)} + CPU order / octree / angles ${stats(tHost)} + pass 2 ${stats(t2)}`);
    // GPU time per stage (back-to-back submits, so the GPU stays clocked up; cumulative prefixes of pass 1)
    for (const [name, ms] of await g.profile(imgs[imgs.length - 1])) post(`GPU pass 1 ${name}: ${ms.toFixed(2)} ms (submit -> done)`);
    g.destroy();
    device.destroy();
    postMessage({ done: true });
  } catch (err) {
    postMessage({ error: String(err && err.stack || err) });
  }
};
