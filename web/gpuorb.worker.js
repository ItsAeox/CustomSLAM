// gpuorb.html worker: WebGPU ORB (gpuorb.js) against ORB-SLAM3's own extractor (the WASM module's OrbGpuHost) on frames
// of a phone recording, frame by frame: keypoints (position, level, angle, response, size, order) and descriptors must
// be identical; also times both paths. Messages out: { line } / { done } / { error }.
import { createGpuOrb } from './gpuorb.js';

const W = 480, H = 640;
const post = (line) => postMessage({ line });
const stats = (v) => {
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
    const { frames, build } = e.data;
    const createOrbSlam3 = (await import(build)).default;
    const M = await createOrbSlam3({ print: post, printErr: post });
    const host = new M.OrbGpuHost(1000, 1.2, 8, 20, 7, W, H);
    const consts = new Int32Array(host.consts());  // copy out of the module heap
    if (!navigator.gpu) throw new Error('WebGPU not available here');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    const info = adapter.info || {};
    post(`adapter: ${info.vendor || '?'} | ${info.architecture || '?'}; ${consts.length} constants (${(consts.length * 4 / 1024).toFixed(0)} KB)`);
    const device = await adapter.requestDevice();
    const g = await createGpuOrb(device, consts, W, H);
    const tCpu = [], tGpu = [], t1 = [], tHost = [], t2 = [];
    let exact = 0, kpDiff = 0, descDiff = 0, bits = 0, kpTotal = 0;
    for (let f = 0; f < frames.length; f++) {
      const img = await gray(frames[f]);
      M.HEAPU8.set(img, host.frameBuffer());
      let t = performance.now();
      const nRef = host.reference();
      tCpu.push(performance.now() - t);
      t = performance.now();
      const c = await g.pass1(img);
      const ta = performance.now();
      new Uint32Array(M.HEAPU8.buffer, host.candBuffer(c.n), c.n * 4).set(c.cand);
      const k = host.distribute(c.n);
      const p2 = new Uint32Array(host.pass2());
      const tb = performance.now();
      const d = await g.pass2(p2, k);
      const tc = performance.now();
      new Uint32Array(M.HEAPU8.buffer, host.descBuffer(k), k * 8).set(d);
      const r = new Int32Array(host.compare(k));  // keypoints differ, descriptors differ, bits, ref count, GPU count
      if (f > 0) { tGpu.push(tc - t); t1.push(ta - t); tHost.push(tb - ta); t2.push(tc - tb); }  // frame 0: warm-up
      kpDiff += r[0]; descDiff += r[1]; bits += r[2]; kpTotal += r[3];
      exact += r[0] === 0 && r[1] === 0;
      post(`${frames[f].split('/').pop()}: ${c.total} candidates${c.total > c.n ? ' (TRUNCATED)' : ''}, keypoints ${r[3]} CPU / ${r[4]} GPU, differ ${r[0]}; descriptors differ ${r[1]} (${r[2]} bits); CPU ${tCpu[tCpu.length - 1].toFixed(1)} ms, GPU path ${(tc - t).toFixed(1)} ms`);
    }
    post(`TOTAL ${frames.length} frames, ${exact} exact; keypoints differ ${kpDiff} of ${kpTotal}; descriptors differ ${descDiff} (${bits} bits)`);
    post(`CPU ORB (ORBextractor, WASM): ${stats(tCpu)}`);
    post(`GPU path: ${stats(tGpu)} = pass 1 ${stats(t1)} + CPU order / octree / angles ${stats(tHost)} + pass 2 ${stats(t2)}`);
    // GPU time per stage (back-to-back submits, so the GPU stays clocked up; cumulative prefixes of pass 1)
    const last = await gray(frames[frames.length - 1]);
    for (const [name, ms] of await g.profile(last)) post(`GPU pass 1 ${name}: ${ms.toFixed(2)} ms (submit -> done)`);
    // GPU path back to back on one frame (no CPU reference in between: the GPU stays clocked up, as in tracking)
    const tb2 = [];
    for (let i = 0; i < 20; i++) {
      const t0 = performance.now();
      const c = await g.pass1(last);
      new Uint32Array(M.HEAPU8.buffer, host.candBuffer(c.n), c.n * 4).set(c.cand);
      const k = host.distribute(c.n);
      await g.pass2(new Uint32Array(host.pass2()), k);
      if (i >= 3) tb2.push(performance.now() - t0);
    }
    post(`GPU path back to back (one frame, 17 runs): ${stats(tb2)}; second candidate readbacks ${g.extraReadbacks} of ${frames.length + 20}`);
    g.destroy();
    device.destroy();
    postMessage({ done: true });
  } catch (err) {
    postMessage({ error: String(err && err.stack || err) });
  }
};
