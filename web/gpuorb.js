// WebGPU ORB (claude/PLAN.md 4c): ORB-SLAM3's monocular features on the GPU with integer steps, meant to be bit-exact
// with ORBextractor. The integer model is slam/port/web/WebOrbGpu.h, checked by slam/port/tests/orb_gpu_model.cc (90/90
// phone frames exact). The CPU side (WASM, OrbGpuHost in slam/port/web/bindings.cc) builds every table with OpenCV's
// own arithmetic, sorts the candidates into ORB-SLAM3's order, runs the unchanged octree and computes the angles.
//   pass 1  frame up; pyramid (resize tables), FAST-9 scores at threshold 7, per-cell "survives at 20" flags, candidates
//           (strict 3x3 maximum inside the cell, threshold 20 or 7) with their intensity-centroid moments; read back
//   blur    the 7-tap fixed-point Gaussian of every level, queued behind the readback (runs while the CPU works)
//   pass 2  chosen keypoints up (level x, y, level, cos, sin from the CPU), rotated BRIEF, descriptors read back
// Usage: const g = await createGpuOrb(device, consts (Int32Array from OrbGpuHost.tables()), width, height);
//        const c = await g.pass1(frame Uint8Array)  -> { n, cand: Uint32Array(4n) [x | y << 16, level | score << 8, m01, m10] }
//        const d = await g.pass2(kp Uint32Array(8k) [x, y, level, f32 bits cos, f32 bits sin, 0, 0, 0], k) -> Uint32Array(8k)
//
// consts layout (Int32, OrbGpuHost::tables): header [0] levels, [1] total pixels P, [2] umax offset, [3] pattern offset,
// [4] taps offset, [5] candidate capacity, [6] keypoint capacity, [7] cells (all levels); per level l at 8 + 16 l: [0] pixel offset, [1] w, [2] h,
// [3] xofs, [4] ia0, [5] ia1, [6] yofs, [7] ib0, [8] ib1 (table offsets), [9] colCell, [10] rowCell (offsets), [11] cell
// columns, [12] first cell, [13] xmax. pix regions: levels [0, P), scores [P, 2P), blur temp [2P, 3P), blurred [3P, 4P).

const WG = 16;

const WGSL = /* wgsl */ `
struct U { lev: u32, n: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> C: array<i32>;
@group(0) @binding(2) var<storage, read_write> pix: array<u32>;
@group(0) @binding(3) var<storage, read_write> at: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> cand: array<u32>;
@group(0) @binding(5) var<storage, read_write> io: array<u32>;

fn lv(l: u32, k: u32) -> i32 { return C[8u + 16u * l + k]; }
fn P() -> u32 { return u32(C[1]); }
fn imgAt(l: u32, x: i32, y: i32) -> i32 { return i32(pix[u32(lv(l, 0u)) + u32(y * lv(l, 1u) + x)]); }
fn scoreAt(l: u32, x: i32, y: i32) -> u32 { return pix[P() + u32(lv(l, 0u)) + u32(y * lv(l, 1u) + x)]; }
fn colCell(l: u32, x: i32) -> i32 { return C[lv(l, 9u) + x]; }
fn rowCell(l: u32, y: i32) -> i32 { return C[lv(l, 10u) + y]; }

@compute @workgroup_size(256) fn unpack(@builtin(global_invocation_id) g: vec3<u32>) {
  let n = u32(lv(0u, 1u) * lv(0u, 2u));
  if (g.x >= n) { return; }
  pix[g.x] = (io[g.x >> 2u] >> ((g.x & 3u) * 8u)) & 255u;
}

// cv::resize INTER_LINEAR, 8-bit: horizontal taps, vertical in OpenCV's SIMD rounding (WebOrbGpu.h applyResize)
@compute @workgroup_size(${WG}, ${WG}) fn resize(@builtin(global_invocation_id) g: vec3<u32>) {
  let l = u.lev;
  let w = lv(l, 1u);
  let h = lv(l, 2u);
  let x = i32(g.x);
  let y = i32(g.y);
  if (x >= w || y >= h) { return; }
  let s = l - 1u;
  let sh = lv(s, 2u);
  let sx = C[lv(l, 3u) + x];
  let a0 = C[lv(l, 4u) + x];
  let a1 = C[lv(l, 5u) + x];
  let full = x < lv(l, 13u);
  let sy0 = clamp(C[lv(l, 6u) + y], 0, sh - 1);
  let sy1 = clamp(C[lv(l, 6u) + y] + 1, 0, sh - 1);
  var h0 = imgAt(s, sx, sy0) * 2048;
  var h1 = imgAt(s, sx, sy1) * 2048;
  if (full) {
    h0 = imgAt(s, sx, sy0) * a0 + imgAt(s, sx + 1, sy0) * a1;
    h1 = imgAt(s, sx, sy1) * a0 + imgAt(s, sx + 1, sy1) * a1;
  }
  let b0 = C[lv(l, 7u) + y];
  let b1 = C[lv(l, 8u) + y];
  let v = ((((h0 >> 4u) * b0) >> 16u) + (((h1 >> 4u) * b1) >> 16u) + 2) >> 2u;
  pix[u32(lv(l, 0u)) + u32(y * w + x)] = u32(clamp(v, 0, 255));
}

fn run9(m: u32) -> bool {
  let m2 = m | (m << 16u);
  var r = m2;
  for (var j = 1u; j < 9u; j++) { r = r & (m2 >> j); }
  return (r & 0xFFFFu) != 0u;
}

// FAST-9 score at threshold 7 (OpenCV FAST_t / cornerScore<16>): strongest 9-arc contrast - 1, 0 if not a corner;
// only inside ORB-SLAM3's cell detection areas (elsewhere 0)
@compute @workgroup_size(${WG}, ${WG}) fn score(@builtin(global_invocation_id) g: vec3<u32>) {
  let l = u.lev;
  let w = lv(l, 1u);
  let h = lv(l, 2u);
  let x = i32(g.x);
  let y = i32(g.y);
  if (x >= w || y >= h) { return; }
  let o = P() + u32(lv(l, 0u)) + u32(y * w + x);
  pix[o] = 0u;
  if (colCell(l, x) < 0 || rowCell(l, y) < 0) { return; }
  var ofs = array<vec2<i32>, 16>(vec2<i32>(0, 3), vec2<i32>(1, 3), vec2<i32>(2, 2), vec2<i32>(3, 1), vec2<i32>(3, 0),
    vec2<i32>(3, -1), vec2<i32>(2, -2), vec2<i32>(1, -3), vec2<i32>(0, -3), vec2<i32>(-1, -3), vec2<i32>(-2, -2),
    vec2<i32>(-3, -1), vec2<i32>(-3, 0), vec2<i32>(-3, 1), vec2<i32>(-2, 2), vec2<i32>(-1, 3));
  let c = imgAt(l, x, y);
  var d: array<i32, 16>;
  var bright = 0u;
  var dark = 0u;
  for (var k = 0u; k < 16u; k++) {
    d[k] = imgAt(l, x + ofs[k].x, y + ofs[k].y) - c;
    if (d[k] > 7) { bright = bright | (1u << k); } else if (d[k] < -7) { dark = dark | (1u << k); }
  }
  if (!(run9(bright) || run9(dark))) { return; }
  var best = 0;
  for (var k = 0u; k < 16u; k++) {
    var mn = 1000;
    var mx = -1000;
    for (var j = 0u; j < 9u; j++) { let v = d[(k + j) & 15u]; mn = min(mn, v); mx = max(mx, v); }
    best = max(best, max(mn, -mx));
  }
  if (best > 7) { pix[o] = u32(best - 1); }
}

// strict 3x3 maximum among the neighbours in the same cell (OpenCV FAST's NMS on the cell ROI)
fn strictMax(l: u32, x: i32, y: i32, v: u32) -> bool {
  let cx = colCell(l, x);
  let cy = rowCell(l, y);
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      if ((dx != 0 || dy != 0) && colCell(l, x + dx) == cx && rowCell(l, y + dy) == cy && scoreAt(l, x + dx, y + dy) >= v) {
        return false;
      }
    }
  }
  return true;
}

@compute @workgroup_size(${WG}, ${WG}) fn flag20(@builtin(global_invocation_id) g: vec3<u32>) {
  let l = u.lev;
  let x = i32(g.x);
  let y = i32(g.y);
  if (x >= lv(l, 1u) || y >= lv(l, 2u)) { return; }
  let cx = colCell(l, x);
  let cy = rowCell(l, y);
  if (cx < 0 || cy < 0) { return; }
  let v = scoreAt(l, x, y);
  if (v >= 20u && strictMax(l, x, y, v)) { atomicOr(&at[16u + u32(lv(l, 12u) + cy * lv(l, 11u) + cx)], 1u); }
}

@compute @workgroup_size(${WG}, ${WG}) fn emit(@builtin(global_invocation_id) g: vec3<u32>) {
  let l = u.lev;
  let x = i32(g.x);
  let y = i32(g.y);
  if (x >= lv(l, 1u) || y >= lv(l, 2u)) { return; }
  let cx = colCell(l, x);
  let cy = rowCell(l, y);
  if (cx < 0 || cy < 0) { return; }
  let v = scoreAt(l, x, y);
  let t = select(7u, 20u, atomicLoad(&at[16u + u32(lv(l, 12u) + cy * lv(l, 11u) + cx)]) != 0u);
  if (v < t || !strictMax(l, x, y, v)) { return; }
  let k = atomicAdd(&at[0], 1u);
  if (k < u32(C[5])) {
    cand[4u * k] = u32(x) | (u32(y) << 16u);
    cand[4u * k + 1u] = l | (v << 8u);
  }
}

// intensity-centroid moments of each candidate (ORBextractor.cc IC_Angle, integer); the angle is computed on the CPU
@compute @workgroup_size(64) fn moments(@builtin(global_invocation_id) g: vec3<u32>) {
  let n = min(atomicLoad(&at[0]), u32(C[5]));
  if (g.x >= n) { return; }
  let p = cand[4u * g.x];
  let l = cand[4u * g.x + 1u] & 255u;
  let x = i32(p & 0xFFFFu);
  let y = i32(p >> 16u);
  var m10 = 0;
  var m01 = 0;
  for (var uu = -15; uu <= 15; uu++) { m10 += uu * imgAt(l, x + uu, y); }
  for (var v = 1; v <= 15; v++) {
    var vsum = 0;
    let d = C[C[2] + v];
    for (var uu = -d; uu <= d; uu++) {
      let plus = imgAt(l, x + uu, y + v);
      let minus = imgAt(l, x + uu, y - v);
      vsum += plus - minus;
      m10 += uu * (plus + minus);
    }
    m01 += v * vsum;
  }
  cand[4u * g.x + 2u] = bitcast<u32>(m01);
  cand[4u * g.x + 3u] = bitcast<u32>(m10);
}

fn refl101(p: i32, n: i32) -> i32 { return select(select(p, 2 * n - p - 2, p >= n), -p, p < 0); }

// OpenCV's bit-exact 8-bit Gaussian (7 taps, sigma 2, reflect-101): rows then columns
@compute @workgroup_size(${WG}, ${WG}) fn blurH(@builtin(global_invocation_id) g: vec3<u32>) {
  let l = u.lev;
  let w = lv(l, 1u);
  let x = i32(g.x);
  let y = i32(g.y);
  if (x >= w || y >= lv(l, 2u)) { return; }
  var s = 0;
  for (var i = 0; i < 7; i++) { s += C[C[4] + i] * imgAt(l, refl101(x + i - 3, w), y); }
  pix[2u * P() + u32(lv(l, 0u)) + u32(y * w + x)] = u32(s);
}

@compute @workgroup_size(${WG}, ${WG}) fn blurV(@builtin(global_invocation_id) g: vec3<u32>) {
  let l = u.lev;
  let w = lv(l, 1u);
  let h = lv(l, 2u);
  let x = i32(g.x);
  let y = i32(g.y);
  if (x >= w || y >= h) { return; }
  var s = 0u;
  for (var j = 0; j < 7; j++) { s += u32(C[C[4] + j]) * pix[2u * P() + u32(lv(l, 0u)) + u32(refl101(y + j - 3, h) * w + x)]; }
  pix[3u * P() + u32(lv(l, 0u)) + u32(y * w + x)] = min((s + 32768u) >> 16u, 255u);
}

// rotated BRIEF (ORBextractor.cc computeOrbDescriptor) on the blurred level; cos / sin from the CPU
@compute @workgroup_size(64) fn describe(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= u.n) { return; }
  let q = 8u * g.x;
  let x = i32(io[q]);
  let y = i32(io[q + 1u]);
  let l = io[q + 2u];
  let a = bitcast<f32>(io[q + 3u]);
  let b = bitcast<f32>(io[q + 4u]);
  let w = lv(l, 1u);
  let base = 3u * P() + u32(lv(l, 0u));
  let pat = C[3];
  let dst = u32(C[6]) * 8u + q;
  for (var wi = 0u; wi < 8u; wi++) {
    var bits = 0u;
    for (var bi = 0u; bi < 4u; bi++) {
      let i = wi * 4u + bi;
      var val = 0u;
      for (var j = 0u; j < 8u; j++) {
        let p0 = pat + i32(2u * (16u * i + 2u * j));
        let p1 = p0 + 2;
        let px0 = f32(C[p0]);
        let py0 = f32(C[p0 + 1]);
        let px1 = f32(C[p1]);
        let py1 = f32(C[p1 + 1]);
        let t0 = pix[base + u32((y + i32(round(px0 * b + py0 * a))) * w + x + i32(round(px0 * a - py0 * b)))];
        let t1 = pix[base + u32((y + i32(round(px1 * b + py1 * a))) * w + x + i32(round(px1 * a - py1 * b)))];
        val = val | (select(0u, 1u, t0 < t1) << j);
      }
      bits = bits | (val << (8u * bi));
    }
    io[dst + wi] = bits;
  }
}
`;

export async function createGpuOrb(device, consts, W, H) {
  const LEVELS = consts[0], P = consts[1], CAPC = consts[5], CAPK = consts[6];
  const lev = (l, k) => consts[8 + 16 * l + k];
  const S = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
  const buf = (size, usage) => device.createBuffer({ size: Math.ceil(size / 16) * 16, usage });
  const constB = buf(consts.byteLength, S | CD);
  device.queue.writeBuffer(constB, 0, consts);
  const nCells = consts[7];
  const pixB = buf(4 * P * 4, S);
  const atB = buf((16 + nCells) * 4, S | CD | CS);
  const candB = buf(CAPC * 16, S | CS);
  const ioWords = Math.max(Math.ceil(W * H / 4), CAPK * 16);
  const ioB = buf(ioWords * 4, S | CD | CS);
  const rb1 = device.createBuffer({ size: 16 + CAPC * 16, usage: GPUBufferUsage.MAP_READ | CD });
  const rb2 = device.createBuffer({ size: CAPK * 32, usage: GPUBufferUsage.MAP_READ | CD });
  const module = device.createShaderModule({ code: WGSL });
  const info = await module.getCompilationInfo();
  const errs = info.messages.filter((m) => m.type === 'error');
  if (errs.length) throw new Error(`WGSL: ${errs.map((m) => `${m.message} (line ${m.lineNum})`).join('; ')}`);
  const layout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    ...[2, 3, 4, 5].map((b) => ({ binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } })),
  ] });
  const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const pipe = {};
  for (const e of ['unpack', 'resize', 'score', 'flag20', 'emit', 'moments', 'blurH', 'blurV', 'describe'])
    pipe[e] = await device.createComputePipelineAsync({ layout: pl, compute: { module, entryPoint: e } });
  const uniform = (lv, n) => {
    const b = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | CD });
    device.queue.writeBuffer(b, 0, new Uint32Array([lv, n, 0, 0]));
    return b;
  };
  const group = (ub) => device.createBindGroup({ layout, entries: [ub, constB, pixB, atB, candB, ioB].map((b, i) => ({ binding: i, resource: { buffer: b } })) });
  const gLev = Array.from({ length: LEVELS }, (_, l) => group(uniform(l, 0)));
  const descU = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | CD });
  const gDesc = group(descU);
  const zeros = new Uint32Array(16 + nCells);
  const perLevel = (p, e) => {
    p.setPipeline(pipe[e]);
    for (let l = e === 'resize' ? 1 : 0; l < LEVELS; l++) {
      p.setBindGroup(0, gLev[l]);
      p.dispatchWorkgroups(Math.ceil(lev(l, 1) / WG), Math.ceil(lev(l, 2) / WG));
    }
  };
  return {
    async pass1(frame) {
      device.queue.writeBuffer(ioB, 0, frame.buffer, frame.byteOffset, Math.ceil(frame.byteLength / 4) * 4);
      device.queue.writeBuffer(atB, 0, zeros);
      const enc = device.createCommandEncoder();
      const p = enc.beginComputePass();
      p.setPipeline(pipe.unpack); p.setBindGroup(0, gLev[0]); p.dispatchWorkgroups(Math.ceil(W * H / 256));
      for (const e of ['resize', 'score', 'flag20', 'emit']) perLevel(p, e);
      p.setPipeline(pipe.moments); p.setBindGroup(0, gLev[0]); p.dispatchWorkgroups(Math.ceil(CAPC / 64));
      p.end();
      enc.copyBufferToBuffer(atB, 0, rb1, 0, 16);
      enc.copyBufferToBuffer(candB, 0, rb1, 16, CAPC * 16);
      device.queue.submit([enc.finish()]);
      const mapped = rb1.mapAsync(GPUMapMode.READ);
      const eb = device.createCommandEncoder();  // blur behind the readback: overlaps the CPU's octree
      const pb = eb.beginComputePass();
      perLevel(pb, 'blurH');
      perLevel(pb, 'blurV');
      pb.end();
      device.queue.submit([eb.finish()]);
      await mapped;
      const n = Math.min(new Uint32Array(rb1.getMappedRange(0, 16))[0], CAPC);
      const total = new Uint32Array(rb1.getMappedRange(0, 16))[0];
      const cand = new Uint32Array(rb1.getMappedRange(16, n * 16)).slice();
      rb1.unmap();
      return { n, total, cand };
    },
    async pass2(kp, k) {
      device.queue.writeBuffer(ioB, 0, kp.buffer, kp.byteOffset, k * 32);
      device.queue.writeBuffer(descU, 0, new Uint32Array([0, k, 0, 0]));
      const enc = device.createCommandEncoder();
      const p = enc.beginComputePass();
      p.setPipeline(pipe.describe); p.setBindGroup(0, gDesc); p.dispatchWorkgroups(Math.ceil(Math.max(k, 1) / 64));
      p.end();
      enc.copyBufferToBuffer(ioB, CAPK * 32, rb2, 0, k * 32);
      device.queue.submit([enc.finish()]);
      await rb2.mapAsync(GPUMapMode.READ);
      const d = new Uint32Array(rb2.getMappedRange(0, k * 32)).slice();
      rb2.unmap();
      return d;
    },
    destroy() { for (const b of [constB, pixB, atB, candB, ioB, rb1, rb2]) b.destroy(); },
  };
}
