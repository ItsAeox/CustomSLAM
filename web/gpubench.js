// WebGPU round-trip benchmark for a GPU ORB front end (is WebGPU ORB worth building? 2026-10-18).
// One 480x640 gray frame per iteration, the GPU work an ORB front end needs: upload, 8-level pyramid (scale 1.2,
// bilinear), FAST-9 (threshold 20) + 3x3 non-maximum suppression on every level, 7x7 Gaussian blur of every level,
// then for up to 2000 keypoints the intensity-centroid orientation (radius 15) and a rotated 256-test binary descriptor;
// count + keypoints + descriptors read back to the CPU (~96 KB). Not ORB-SLAM3's exact features (no keypoint
// spreading, which stays on the CPU, and a hashed test pattern): the point is the cost of the GPU work plus the
// upload / readback round trip. Timed: an empty round trip (4 bytes), the full frame until the readback is mapped
// (what the tracker would wait for), and the full frame until the GPU is done (no readback).
//   runBench(log) -> results object; log(line) receives progress text.

const W0 = 480, H0 = 640, LEVELS = 8, SCALE = 1.2, MAXK = 2000, CAP = 60000, TH = 20;

const WGSL = /* wgsl */ `
struct Level { off: u32, w: u32, h: u32, pad: u32 };
struct Params { src: Level, dst: Level, scale: f32, th: f32, lev: u32, maxk: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> img: array<f32>;
@group(0) @binding(2) var<storage, read_write> score: array<f32>;
@group(0) @binding(3) var<storage, read_write> blur: array<f32>;
@group(0) @binding(4) var<storage, read_write> kps: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> cnt: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> desc: array<u32>;
@group(0) @binding(7) var<storage, read> src8: array<u32>;
@group(0) @binding(8) var<uniform> L: array<vec4<u32>, 8>;

fn px(o: u32, w: u32, h: u32, x: i32, y: i32) -> f32 {
  let cx = u32(clamp(x, 0, i32(w) - 1));
  let cy = u32(clamp(y, 0, i32(h) - 1));
  return img[o + cy * w + cx];
}
fn bpx(o: u32, w: u32, h: u32, x: i32, y: i32) -> f32 {
  let cx = u32(clamp(x, 0, i32(w) - 1));
  let cy = u32(clamp(y, 0, i32(h) - 1));
  return blur[o + cy * w + cx];
}

@compute @workgroup_size(256) fn unpack(@builtin(global_invocation_id) g: vec3<u32>) {
  let n = P.dst.w * P.dst.h;
  if (g.x >= n) { return; }
  let word = src8[g.x >> 2u];
  img[P.dst.off + g.x] = f32((word >> ((g.x & 3u) * 8u)) & 255u);
}

@compute @workgroup_size(16, 16) fn down(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= P.dst.w || g.y >= P.dst.h) { return; }
  let sx = (f32(g.x) + 0.5) * P.scale - 0.5;
  let sy = (f32(g.y) + 0.5) * P.scale - 0.5;
  let x0 = i32(floor(sx));
  let y0 = i32(floor(sy));
  let fx = sx - f32(x0);
  let fy = sy - f32(y0);
  let s = P.src;
  let v = mix(mix(px(s.off, s.w, s.h, x0, y0), px(s.off, s.w, s.h, x0 + 1, y0), fx),
              mix(px(s.off, s.w, s.h, x0, y0 + 1), px(s.off, s.w, s.h, x0 + 1, y0 + 1), fx), fy);
  img[P.dst.off + g.y * P.dst.w + g.x] = v;
}

fn run9(m: u32) -> bool {
  let m2 = m | (m << 16u);
  var r = m2;
  for (var j = 1u; j < 9u; j++) { r = r & (m2 >> j); }
  return (r & 0xFFFFu) != 0u;
}

@compute @workgroup_size(16, 16) fn fast(@builtin(global_invocation_id) g: vec3<u32>) {
  let s = P.src;
  if (g.x < 16u || g.y < 16u || g.x + 16u >= s.w || g.y + 16u >= s.h) { return; }
  var ofs = array<vec2<i32>, 16>(vec2<i32>(0, 3), vec2<i32>(1, 3), vec2<i32>(2, 2), vec2<i32>(3, 1), vec2<i32>(3, 0),
    vec2<i32>(3, -1), vec2<i32>(2, -2), vec2<i32>(1, -3), vec2<i32>(0, -3), vec2<i32>(-1, -3), vec2<i32>(-2, -2),
    vec2<i32>(-3, -1), vec2<i32>(-3, 0), vec2<i32>(-3, 1), vec2<i32>(-2, 2), vec2<i32>(-1, 3));
  let x = i32(g.x);
  let y = i32(g.y);
  let c = img[s.off + g.y * s.w + g.x];
  var bright = 0u;
  var dark = 0u;
  var sc = 0.0;
  for (var k = 0u; k < 16u; k++) {
    let d = img[s.off + u32(y + ofs[k].y) * s.w + u32(x + ofs[k].x)] - c;
    if (d > P.th) { bright = bright | (1u << k); sc = sc + d - P.th; }
    else if (d < -P.th) { dark = dark | (1u << k); sc = sc - d - P.th; }
  }
  score[s.off + g.y * s.w + g.x] = select(0.0, sc, run9(bright) || run9(dark));
}

@compute @workgroup_size(16, 16) fn nms(@builtin(global_invocation_id) g: vec3<u32>) {
  let s = P.src;
  if (g.x < 17u || g.y < 17u || g.x + 17u >= s.w || g.y + 17u >= s.h) { return; }
  let i = s.off + g.y * s.w + g.x;
  let v = score[i];
  if (v <= 0.0) { return; }
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      if (dx == 0 && dy == 0) { continue; }
      let n = score[u32(i32(i) + dy * i32(s.w) + dx)];
      if (n > v || (n == v && (dy < 0 || (dy == 0 && dx < 0)))) { return; }
    }
  }
  let k = atomicAdd(&cnt[0], 1u);
  if (k < ${CAP}u) { kps[k] = vec4<f32>(f32(g.x), f32(g.y), f32(P.lev), v); }
}

@compute @workgroup_size(16, 16) fn gauss(@builtin(global_invocation_id) g: vec3<u32>) {
  let s = P.src;
  if (g.x >= s.w || g.y >= s.h) { return; }
  var acc = 0.0;
  var wsum = 0.0;
  for (var dy = -3; dy <= 3; dy++) {
    for (var dx = -3; dx <= 3; dx++) {
      let w = exp(-f32(dx * dx + dy * dy) / 8.0);
      acc = acc + w * px(s.off, s.w, s.h, i32(g.x) + dx, i32(g.y) + dy);
      wsum = wsum + w;
    }
  }
  blur[s.off + g.y * s.w + g.x] = acc / wsum;
}

fn hash(v: u32) -> u32 {
  var x = v * 747796405u + 2891336453u;
  x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return (x >> 22u) ^ x;
}

@compute @workgroup_size(64) fn describe(@builtin(global_invocation_id) g: vec3<u32>) {
  let n = min(atomicLoad(&cnt[0]), P.maxk);
  if (g.x >= n) { return; }
  let kp = kps[g.x];
  let lv = L[u32(kp.z)];
  let x = i32(kp.x);
  let y = i32(kp.y);
  var m01 = 0.0;
  var m10 = 0.0;
  for (var dy = -15; dy <= 15; dy++) {
    for (var dx = -15; dx <= 15; dx++) {
      if (dx * dx + dy * dy > 225) { continue; }
      let v = px(lv.x, lv.y, lv.z, x + dx, y + dy);
      m10 = m10 + f32(dx) * v;
      m01 = m01 + f32(dy) * v;
    }
  }
  let a = atan2(m01, m10);
  let ca = cos(a);
  let sa = sin(a);
  for (var w = 0u; w < 8u; w++) {
    var bits = 0u;
    for (var b = 0u; b < 32u; b++) {
      let h = hash(w * 32u + b);
      let p = vec4<f32>(f32(h & 31u) - 15.5, f32((h >> 5u) & 31u) - 15.5, f32((h >> 10u) & 31u) - 15.5, f32((h >> 15u) & 31u) - 15.5) * 0.8;
      let u1 = i32(round(ca * p.x - sa * p.y));
      let v1 = i32(round(sa * p.x + ca * p.y));
      let u2 = i32(round(ca * p.z - sa * p.w));
      let v2 = i32(round(sa * p.z + ca * p.w));
      if (bpx(lv.x, lv.y, lv.z, x + u1, y + v1) < bpx(lv.x, lv.y, lv.z, x + u2, y + v2)) { bits = bits | (1u << b); }
    }
    desc[g.x * 8u + w] = bits;
  }
}
`;

function syntheticFrame() {
  // textured scene: blurred random blocks + fine noise (FAST finds thousands of corners, like a room image)
  const a = new Uint8Array(W0 * H0);
  let seed = 12345;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 4294967296);
  const B = 12, cols = Math.ceil(W0 / B), rows = Math.ceil(H0 / B), blocks = new Float32Array(cols * rows);
  for (let i = 0; i < blocks.length; i++) blocks[i] = 40 + 170 * rnd();
  for (let y = 0; y < H0; y++)
    for (let x = 0; x < W0; x++) a[y * W0 + x] = Math.max(0, Math.min(255, blocks[((y / B) | 0) * cols + ((x / B) | 0)] + 18 * (rnd() - 0.5)));
  return a;
}

const stats = (v) => {
  const s = [...v].sort((x, y) => x - y), q = (p) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  return { median: q(0.5), p90: q(0.9), p95: q(0.95), max: s[s.length - 1], mean: v.reduce((x, y) => x + y, 0) / v.length };
};
const fmt = (st) => `median ${st.median.toFixed(2)} ms, p90 ${st.p90.toFixed(2)}, p95 ${st.p95.toFixed(2)}, max ${st.max.toFixed(2)} (mean ${st.mean.toFixed(2)})`;

export async function runBench(log, iters = 150, warm = 20) {
  if (!globalThis.navigator || !navigator.gpu) throw new Error('WebGPU not available here (navigator.gpu missing)');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no WebGPU adapter');
  const info = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
  log(`adapter: ${info.vendor || '?'} | ${info.architecture || '?'} | ${info.description || info.device || '?'}`);
  const device = await adapter.requestDevice();
  device.lost.then((l) => log(`device lost: ${l.message}`));

  // level table
  const lv = [];
  let off = 0, w = W0, h = H0;
  for (let l = 0; l < LEVELS; l++) {
    if (l) { w = Math.round(W0 / SCALE ** l); h = Math.round(H0 / SCALE ** l); }
    lv.push({ off, w, h });
    off += w * h;
  }
  const total = off;
  const S = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
  const buf = (size, usage) => device.createBuffer({ size: Math.ceil(size / 16) * 16, usage });
  const imgB = buf(total * 4, S), scoreB = buf(total * 4, S), blurB = buf(total * 4, S);
  const kpsB = buf(CAP * 16, S | CS), cntB = buf(16, S | CS | CD), descB = buf(MAXK * 32, S | CS);
  const srcB = buf(W0 * H0, S | CD);
  const levB = device.createBuffer({ size: 8 * 16, usage: GPUBufferUsage.UNIFORM | CD });
  device.queue.writeBuffer(levB, 0, new Uint32Array(lv.flatMap((x) => [x.off, x.w, x.h, 0])));
  const RB = 16 + MAXK * 16 + MAXK * 32;
  const rbB = device.createBuffer({ size: RB, usage: GPUBufferUsage.MAP_READ | CD });
  const tinyB = buf(16, CS | CD), rbTiny = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | CD });

  const module = device.createShaderModule({ code: WGSL });
  const ci = await module.getCompilationInfo();
  for (const m of ci.messages) log(`WGSL ${m.type}: ${m.message} (line ${m.lineNum})`);
  const layout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ...[1, 2, 3, 4, 5, 6].map((b) => ({ binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } })),
    { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
  ] });
  const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const pipe = {};
  for (const e of ['unpack', 'down', 'fast', 'nms', 'gauss', 'describe'])
    pipe[e] = await device.createComputePipelineAsync({ layout: pl, compute: { module, entryPoint: e } });
  const group = (src, dst, scale, lev) => {
    const u = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | CD });
    const ab = new ArrayBuffer(48), U = new Uint32Array(ab), F = new Float32Array(ab);
    U.set([src.off, src.w, src.h, 0, dst.off, dst.w, dst.h, 0]);
    F[8] = scale; F[9] = TH; U[10] = lev; U[11] = MAXK;
    device.queue.writeBuffer(u, 0, ab);
    return device.createBindGroup({ layout, entries: [u, imgB, scoreB, blurB, kpsB, cntB, descB, srcB, levB].map((b, i) => ({ binding: i, resource: { buffer: b } })) });
  };
  const gUnpack = group(lv[0], lv[0], 1, 0);
  const gDown = lv.map((x, l) => (l ? group(lv[l - 1], x, lv[l - 1].w / x.w, l) : null));
  const gLev = lv.map((x, l) => group(x, x, 1, l));
  const frame = syntheticFrame();
  const zero = new Uint32Array(4);

  const encodeFrame = () => {
    device.queue.writeBuffer(srcB, 0, frame);
    device.queue.writeBuffer(cntB, 0, zero);
    const enc = device.createCommandEncoder();
    const p = enc.beginComputePass();
    p.setPipeline(pipe.unpack); p.setBindGroup(0, gUnpack); p.dispatchWorkgroups(Math.ceil(W0 * H0 / 256));
    p.setPipeline(pipe.down);
    for (let l = 1; l < LEVELS; l++) { p.setBindGroup(0, gDown[l]); p.dispatchWorkgroups(Math.ceil(lv[l].w / 16), Math.ceil(lv[l].h / 16)); }
    for (const e of ['fast', 'nms', 'gauss']) {
      p.setPipeline(pipe[e]);
      for (let l = 0; l < LEVELS; l++) { p.setBindGroup(0, gLev[l]); p.dispatchWorkgroups(Math.ceil(lv[l].w / 16), Math.ceil(lv[l].h / 16)); }
    }
    p.setPipeline(pipe.describe); p.setBindGroup(0, gLev[0]); p.dispatchWorkgroups(Math.ceil(MAXK / 64));
    p.end();
    return enc;
  };

  // 1) empty round trip: 4 bytes up, copied, mapped
  const tEmpty = [];
  for (let i = 0; i < warm + iters; i++) {
    const t0 = performance.now();
    device.queue.writeBuffer(tinyB, 0, new Uint32Array([i, 0, 0, 0]));
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(tinyB, 0, rbTiny, 0, 16);
    device.queue.submit([enc.finish()]);
    await rbTiny.mapAsync(GPUMapMode.READ);
    rbTiny.unmap();
    if (i >= warm) tEmpty.push(performance.now() - t0);
  }
  log(`empty round trip (4 bytes): ${fmt(stats(tEmpty))}`);

  // 2) full frame, until the readback is mapped (what the tracker would wait for)
  const tFull = [], tSubmit = [];
  let count = 0;
  for (let i = 0; i < warm + iters; i++) {
    const t0 = performance.now();
    const enc = encodeFrame();
    enc.copyBufferToBuffer(cntB, 0, rbB, 0, 16);
    enc.copyBufferToBuffer(kpsB, 0, rbB, 16, MAXK * 16);
    enc.copyBufferToBuffer(descB, 0, rbB, 16 + MAXK * 16, MAXK * 32);
    device.queue.submit([enc.finish()]);
    const t1 = performance.now();
    await rbB.mapAsync(GPUMapMode.READ);
    const r = new Uint32Array(rbB.getMappedRange(0, 16));
    count = r[0];
    const d = new Uint32Array(rbB.getMappedRange(16 + MAXK * 16, MAXK * 32)).slice(0, 8);  // touch the descriptors
    rbB.unmap();
    if (i >= warm) { tFull.push(performance.now() - t0); tSubmit.push(t1 - t0); }
    if (i === warm) log(`keypoints detected (all levels, after NMS): ${count}; descriptors computed: ${Math.min(count, MAXK)}; first descriptor word ${d[0] >>> 0}`);
  }
  log(`full frame + readback (~${(RB / 1024).toFixed(0)} KB): ${fmt(stats(tFull))}`);
  log(`  of which encode + submit on the CPU: ${fmt(stats(tSubmit))}`);

  // 3) full frame, until the GPU is done (no readback)
  const tGpu = [];
  for (let i = 0; i < warm + iters; i++) {
    const t0 = performance.now();
    device.queue.submit([encodeFrame().finish()]);
    await device.queue.onSubmittedWorkDone();
    if (i >= warm) tGpu.push(performance.now() - t0);
  }
  log(`full frame, GPU done (no readback): ${fmt(stats(tGpu))}`);
  device.destroy();
  return { adapter: { vendor: info.vendor, architecture: info.architecture, description: info.description }, empty: stats(tEmpty), full: stats(tFull), submit: stats(tSubmit), gpu: stats(tGpu), keypoints: count };
}
