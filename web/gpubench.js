// WebGPU round-trip benchmark for a GPU ORB front end (is WebGPU ORB worth building? 2026-10-18).
// One 480x640 gray frame per iteration, the GPU work an ORB front end needs: upload, 8-level pyramid (scale 1.2,
// bilinear), FAST-9 (threshold 20) + 3x3 non-maximum suppression on every level, 7x7 Gaussian blur of every level,
// then for up to 2000 keypoints the intensity-centroid orientation (radius 15) and a rotated 256-test binary descriptor;
// count + keypoints + descriptors read back to the CPU (~96 KB). Not ORB-SLAM3's exact features (no keypoint
// spreading, which stays on the CPU, and a hashed test pattern): the point is the cost of the GPU work plus the
// upload / readback round trip. Timed: an empty round trip (4 bytes), the full frame until the readback is mapped
// (what the tracker would wait for), and the full frame until the GPU is done (no readback).
// Test 4 (two passes, the shape a real GPU ORB needs, because the keypoints are chosen between FAST and the descriptors):
// pass 1 = upload, pyramid, FAST-9 score (largest threshold that still makes a corner, as OpenCV's cornerScore) with
// ORB-SLAM3's per-cell thresholds (20, or 7 in ~35-px cells with no corner at 20), 3x3 non-maximum suppression, blur;
// candidates read back (8 bytes each); the CPU picks the keypoints per level (stand-in: strongest first, where the real
// one runs the octree or SSC in WASM); pass 2 = selected keypoints up, orientation + descriptors, read back. Run on a
// real frame from a phone recording when one is given (frameUrl), else on the synthetic frame.
//   runBench(log, iters, warm, frameUrl) -> results object; log(line) receives progress text.

const W0 = 480, H0 = 640, LEVELS = 8, SCALE = 1.2, MAXK = 2000, CAP = 60000, TH = 20;
const GW = (() => { const w = [-3, -2, -1, 0, 1, 2, 3].map((x) => Math.exp(-x * x / 8)), t = w.reduce((a, b) => a + b);
  return w.map((x) => (x / t).toFixed(7)).join(', '); })();  // 7-tap Gaussian, sigma 2 (ORBextractor's GaussianBlur)
const TH_LO = 7, CAPC = 24576, NF = 1000, CELL = 35;  // ORB-SLAM3 minThFAST, candidate readback cap, nFeatures, cell size
const CELLS0 = 16;  // cnt[0] old keypoint count, cnt[1] candidates, cnt[2] selected keypoints; per-cell max score from 16

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
@group(0) @binding(9) var<uniform> C: array<vec4<u32>, 8>;  // per level: first cell, cell columns | rows << 16, cell w, h

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

// ---- test 4 (two passes) ----
const GW = array<f32, 7>(${GW});
@compute @workgroup_size(16, 16) fn gaussH(@builtin(global_invocation_id) g: vec3<u32>) {  // img -> score (temporary)
  let s = P.src;
  if (g.x >= s.w || g.y >= s.h) { return; }
  var acc = 0.0;
  for (var k = 0; k < 7; k++) { acc = acc + GW[k] * px(s.off, s.w, s.h, i32(g.x) + k - 3, i32(g.y)); }
  score[s.off + g.y * s.w + g.x] = acc;
}
var<workgroup> tileI: array<f32, 484>;  // 16x16 outputs + 3-pixel apron
var<workgroup> tileH: array<f32, 352>;  // horizontal pass: 16 columns x 22 rows
@compute @workgroup_size(16, 16) fn gaussT(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>,
                                         @builtin(workgroup_id) wg: vec3<u32>) {
  let s = P.src;
  let x0 = i32(wg.x * 16u) - 3;
  let y0 = i32(wg.y * 16u) - 3;
  let li = l.y * 16u + l.x;
  for (var k = li; k < 484u; k += 256u) { tileI[k] = px(s.off, s.w, s.h, x0 + i32(k % 22u), y0 + i32(k / 22u)); }
  workgroupBarrier();
  for (var k = li; k < 352u; k += 256u) {
    let tx = k % 16u;
    let ty = k / 16u;
    var acc = 0.0;
    for (var j = 0u; j < 7u; j++) { acc = acc + GW[j] * tileI[ty * 22u + tx + j]; }
    tileH[k] = acc;
  }
  workgroupBarrier();
  if (g.x >= s.w || g.y >= s.h) { return; }
  var acc = 0.0;
  for (var j = 0u; j < 7u; j++) { acc = acc + GW[j] * tileH[(l.y + j) * 16u + l.x]; }
  blur[s.off + g.y * s.w + g.x] = acc;
}

var<workgroup> tileF: array<f32, 484>;
@compute @workgroup_size(16, 16) fn score7T(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>,
                                          @builtin(workgroup_id) wg: vec3<u32>) {
  let s = P.src;
  let x0 = i32(wg.x * 16u) - 3;
  let y0 = i32(wg.y * 16u) - 3;
  let li = l.y * 16u + l.x;
  for (var k = li; k < 484u; k += 256u) { tileF[k] = px(s.off, s.w, s.h, x0 + i32(k % 22u), y0 + i32(k / 22u)); }
  workgroupBarrier();
  if (g.x >= s.w || g.y >= s.h) { return; }
  let i = s.off + g.y * s.w + g.x;
  score[i] = 0.0;
  if (g.x < 19u || g.y < 19u || g.x + 19u >= s.w || g.y + 19u >= s.h) { return; }
  var ofs = array<vec2<i32>, 16>(vec2<i32>(0, 3), vec2<i32>(1, 3), vec2<i32>(2, 2), vec2<i32>(3, 1), vec2<i32>(3, 0),
    vec2<i32>(3, -1), vec2<i32>(2, -2), vec2<i32>(1, -3), vec2<i32>(0, -3), vec2<i32>(-1, -3), vec2<i32>(-2, -2),
    vec2<i32>(-3, -1), vec2<i32>(-3, 0), vec2<i32>(-3, 1), vec2<i32>(-2, 2), vec2<i32>(-1, 3));
  let tc = i32(l.y + 3u) * 22 + i32(l.x + 3u);
  let c = tileF[tc];
  var d: array<f32, 16>;
  var bright = 0u;
  var dark = 0u;
  for (var k = 0u; k < 16u; k++) {
    d[k] = tileF[tc + ofs[k].y * 22 + ofs[k].x] - c;
    if (d[k] > ${TH_LO}.0) { bright = bright | (1u << k); } else if (d[k] < -${TH_LO}.0) { dark = dark | (1u << k); }
  }
  if (!(run9(bright) || run9(dark))) { return; }
  var best = 0.0;
  for (var k = 0u; k < 16u; k++) {
    var mn = 1e9;
    var mx = -1e9;
    for (var j = 0u; j < 9u; j++) { let v = d[(k + j) & 15u]; mn = min(mn, v); mx = max(mx, v); }
    best = max(best, max(mn, -mx));
  }
  score[i] = best;
  if (best > P.th) { atomicMax(&cnt[${CELLS0}u + cellOf(g.x, g.y)], u32(best)); }  // only "has a corner at 20" matters
}

@compute @workgroup_size(16, 16) fn gaussV(@builtin(global_invocation_id) g: vec3<u32>) {  // score -> blur
  let s = P.src;
  if (g.x >= s.w || g.y >= s.h) { return; }
  var acc = 0.0;
  for (var k = 0; k < 7; k++) {
    let y = u32(clamp(i32(g.y) + k - 3, 0, i32(s.h) - 1));
    acc = acc + GW[k] * score[s.off + y * s.w + g.x];
  }
  blur[s.off + g.y * s.w + g.x] = acc;
}

fn cellOf(x: u32, y: u32) -> u32 {  // ORB-SLAM3 cells: detection area [19, w - 19), cells of ~35 px
  let c = C[P.lev];
  let nc = c.y & 0xFFFFu;
  let nr = c.y >> 16u;
  return c.x + min((y - 19u) / c.w, nr - 1u) * nc + min((x - 19u) / c.z, nc - 1u);
}

@compute @workgroup_size(16, 16) fn score7(@builtin(global_invocation_id) g: vec3<u32>) {
  let s = P.src;
  if (g.x >= s.w || g.y >= s.h) { return; }
  let i = s.off + g.y * s.w + g.x;
  score[i] = 0.0;
  if (g.x < 19u || g.y < 19u || g.x + 19u >= s.w || g.y + 19u >= s.h) { return; }
  var ofs = array<vec2<i32>, 16>(vec2<i32>(0, 3), vec2<i32>(1, 3), vec2<i32>(2, 2), vec2<i32>(3, 1), vec2<i32>(3, 0),
    vec2<i32>(3, -1), vec2<i32>(2, -2), vec2<i32>(1, -3), vec2<i32>(0, -3), vec2<i32>(-1, -3), vec2<i32>(-2, -2),
    vec2<i32>(-3, -1), vec2<i32>(-3, 0), vec2<i32>(-3, 1), vec2<i32>(-2, 2), vec2<i32>(-1, 3));
  let c = img[i];
  var d: array<f32, 16>;
  var bright = 0u;
  var dark = 0u;
  for (var k = 0u; k < 16u; k++) {
    d[k] = img[s.off + u32(i32(g.y) + ofs[k].y) * s.w + u32(i32(g.x) + ofs[k].x)] - c;
    if (d[k] > ${TH_LO}.0) { bright = bright | (1u << k); } else if (d[k] < -${TH_LO}.0) { dark = dark | (1u << k); }
  }
  if (!(run9(bright) || run9(dark))) { return; }
  var best = 0.0;  // largest t with 9 contiguous pixels all > c + t (or all < c - t)
  for (var k = 0u; k < 16u; k++) {
    var mn = 1e9;
    var mx = -1e9;
    for (var j = 0u; j < 9u; j++) { let v = d[(k + j) & 15u]; mn = min(mn, v); mx = max(mx, v); }
    best = max(best, max(mn, -mx));
  }
  score[i] = best;
  atomicMax(&cnt[${CELLS0}u + cellOf(g.x, g.y)], u32(best));
}

@compute @workgroup_size(16, 16) fn cand(@builtin(global_invocation_id) g: vec3<u32>) {
  let s = P.src;
  if (g.x < 19u || g.y < 19u || g.x + 19u >= s.w || g.y + 19u >= s.h) { return; }
  let i = s.off + g.y * s.w + g.x;
  let v = score[i];
  let th = select(${TH_LO}.0, P.th, f32(atomicLoad(&cnt[${CELLS0}u + cellOf(g.x, g.y)])) > P.th);  // 20, or 7 in empty cells
  if (v <= th) { return; }
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      if (dx == 0 && dy == 0) { continue; }
      let n = score[u32(i32(i) + dy * i32(s.w) + dx)];
      if (n > v || (n == v && (dy < 0 || (dy == 0 && dx < 0)))) { return; }
    }
  }
  let k = atomicAdd(&cnt[1], 1u);
  if (k < ${CAPC}u) { desc[k] = g.x | (g.y << 9u) | (P.lev << 19u) | (min(u32(v), 255u) << 22u); }
}

@compute @workgroup_size(64) fn describe2(@builtin(global_invocation_id) g: vec3<u32>) {
  let n = atomicLoad(&cnt[2]);
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
  kps[g.x].w = a;
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

async function loadFrame(url) {
  // a phone-recording JPEG (gray, 480x640) as 8-bit luma; works in a worker (createImageBitmap + OffscreenCanvas)
  const bmp = await createImageBitmap(await (await fetch(url)).blob());
  if (bmp.width !== W0 || bmp.height !== H0) throw new Error(`frame is ${bmp.width}x${bmp.height}, expected ${W0}x${H0}`);
  const cv = new OffscreenCanvas(W0, H0), ctx = cv.getContext('2d');
  ctx.drawImage(bmp, 0, 0);
  const rgba = ctx.getImageData(0, 0, W0, H0).data, a = new Uint8Array(W0 * H0);
  for (let i = 0; i < a.length; i++) a[i] = (rgba[4 * i] * 77 + rgba[4 * i + 1] * 150 + rgba[4 * i + 2] * 29) >> 8;
  return a;
}

function featuresPerLevel() {  // ORBextractor: geometric split of nFeatures over the levels, the rest to the top level
  const f = 1 / SCALE, out = [];
  let per = NF * (1 - f) / (1 - f ** LEVELS), sum = 0;
  for (let l = 0; l < LEVELS - 1; l++) { out.push(Math.round(per)); sum += out[l]; per *= f; }
  out.push(Math.max(NF - sum, 0));
  return out;
}

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

export async function runBench(log, iters = 150, warm = 20, frameUrl = null) {
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
  const kpsB = buf(CAP * 16, S | CS | CD), cntB = buf((CELLS0 + 4096) * 4, S | CS | CD), descB = buf(Math.max(MAXK * 32, CAPC * 8), S | CS);
  const srcB = buf(W0 * H0, S | CD);
  const levB = device.createBuffer({ size: 8 * 16, usage: GPUBufferUsage.UNIFORM | CD });
  device.queue.writeBuffer(levB, 0, new Uint32Array(lv.flatMap((x) => [x.off, x.w, x.h, 0])));
  let ncell = 0;
  const cells = lv.map((x) => {  // ORBextractor::ComputeKeyPointsOctTree grid: width = cols - 32, nCols = width / 35
    const wd = x.w - 32, ht = x.h - 32, nc = Math.max(1, Math.floor(wd / CELL)), nr = Math.max(1, Math.floor(ht / CELL));
    const c = [ncell, nc | (nr << 16), Math.ceil(wd / nc), Math.ceil(ht / nr)];
    ncell += nc * nr;
    return c;
  });
  const cellB = device.createBuffer({ size: 8 * 16, usage: GPUBufferUsage.UNIFORM | CD });
  device.queue.writeBuffer(cellB, 0, new Uint32Array(cells.flat()));
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
    { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
  ] });
  const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const pipe = {};
  for (const e of ['unpack', 'down', 'fast', 'nms', 'gauss', 'describe', 'score7', 'cand', 'describe2', 'gaussH', 'gaussV', 'gaussT', 'score7T'])
    pipe[e] = await device.createComputePipelineAsync({ layout: pl, compute: { module, entryPoint: e } });
  const group = (src, dst, scale, lev) => {
    const u = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | CD });
    const ab = new ArrayBuffer(48), U = new Uint32Array(ab), F = new Float32Array(ab);
    U.set([src.off, src.w, src.h, 0, dst.off, dst.w, dst.h, 0]);
    F[8] = scale; F[9] = TH; U[10] = lev; U[11] = MAXK;
    device.queue.writeBuffer(u, 0, ab);
    return device.createBindGroup({ layout, entries: [u, imgB, scoreB, blurB, kpsB, cntB, descB, srcB, levB, cellB].map((b, i) => ({ binding: i, resource: { buffer: b } })) });
  };
  const gUnpack = group(lv[0], lv[0], 1, 0);
  const gDown = lv.map((x, l) => (l ? group(lv[l - 1], x, lv[l - 1].w / x.w, l) : null));
  const gLev = lv.map((x, l) => group(x, x, 1, l));
  let frame = syntheticFrame(), frameName = 'synthetic frame';
  if (frameUrl) {
    try { frame = await loadFrame(frameUrl); frameName = frameUrl.split('/').slice(-4).join('/'); }
    catch (err) { log(`could not load ${frameUrl} (${err.message || err}); synthetic frame instead`); }
  }
  log(`frame: ${frameName}`);
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

  // 4) two passes: candidates back, keypoints chosen on the CPU, orientation + descriptors for those
  const nPer = featuresPerLevel();
  const RB1 = 16 + CAPC * 4, RB2 = NF * 2 * 16 + NF * 2 * 32;
  const rb1 = device.createBuffer({ size: RB1, usage: GPUBufferUsage.MAP_READ | CD });
  const rb2 = device.createBuffer({ size: RB2, usage: GPUBufferUsage.MAP_READ | CD });
  const zeroCells = new Uint32Array(CELLS0 + ncell);
  const t1s = [], tCpu = [], t2s = [], tAll = [];
  let nCand = 0, nSel = 0, perLevel = [];
  const STAGES = ['pyramid', 'FAST score', 'per-cell threshold + NMS', 'blur'];
  let KS = [];  // kernels after the pyramid, one per stage (set per FAST variant below)
  const encodePass1 = (upto = STAGES.length) => {
    device.queue.writeBuffer(srcB, 0, frame);
    device.queue.writeBuffer(cntB, 0, zeroCells);
    const enc = device.createCommandEncoder();
    const p = enc.beginComputePass();
    p.setPipeline(pipe.unpack); p.setBindGroup(0, gUnpack); p.dispatchWorkgroups(Math.ceil(W0 * H0 / 256));
    p.setPipeline(pipe.down);
    for (let l = 1; l < LEVELS; l++) { p.setBindGroup(0, gDown[l]); p.dispatchWorkgroups(Math.ceil(lv[l].w / 16), Math.ceil(lv[l].h / 16)); }
    for (const e of KS.slice(0, upto - 1)) {
      p.setPipeline(pipe[e]);
      for (let l = 0; l < LEVELS; l++) { p.setBindGroup(0, gLev[l]); p.dispatchWorkgroups(Math.ceil(lv[l].w / 16), Math.ceil(lv[l].h / 16)); }
    }
    p.end();
    return enc;
  };
  const cnt8 = new Uint32Array(LEVELS * 256);
  const encodeBlur = () => {
    const enc = device.createCommandEncoder();
    const p = enc.beginComputePass();
    p.setPipeline(pipe.gaussT);
    for (let l = 0; l < LEVELS; l++) { p.setBindGroup(0, gLev[l]); p.dispatchWorkgroups(Math.ceil(lv[l].w / 16), Math.ceil(lv[l].h / 16)); }
    p.end();
    return enc;
  };
  const VARIANTS = [['score7', 'FAST score from global memory'], ['score7T', 'FAST score from a workgroup tile']];
  for (const [fastK, label] of VARIANTS) {
  KS = [fastK, 'cand', 'gaussT'];
  for (const a of [t1s, tCpu, t2s, tAll]) a.length = 0;
  log(`--- ${label} (blur from a workgroup tile)`);
  for (let i = 0; i < warm + iters; i++) {
    const t0 = performance.now();
    let enc = encodePass1(3);  // pyramid, FAST score, threshold + NMS
    enc.copyBufferToBuffer(cntB, 0, rb1, 0, 16);
    enc.copyBufferToBuffer(descB, 0, rb1, 16, CAPC * 4);
    device.queue.submit([enc.finish()]);
    const mapped = rb1.mapAsync(GPUMapMode.READ);
    device.queue.submit([encodeBlur().finish()]);  // after the map request: runs while the CPU chooses the keypoints
    await mapped;
    const ta = performance.now();
    nCand = new Uint32Array(rb1.getMappedRange(0, 16))[1];
    const n = Math.min(nCand, CAPC), cw = new Uint32Array(rb1.getMappedRange(16, n * 4));
    // stand-in for the octree / SSC (which run in WASM: ORB-PROF "spread"): per level, strongest first, up to that
    // level's share of nFeatures; a counting sort on the 8-bit score (linear time)
    cnt8.fill(0);
    for (let k = 0; k < n; k++) cnt8[((cw[k] >>> 19) & 7) * 256 + (cw[k] >>> 22)]++;
    const thr = new Int32Array(LEVELS), take = new Int32Array(LEVELS);  // per level: lowest score taken, how many at it
    perLevel = [];
    for (let l = 0; l < LEVELS; l++) {
      let left = nPer[l], tot = 0, sc = 255;
      for (let q = 0; q < 256; q++) tot += cnt8[l * 256 + q];
      for (; sc >= 0 && left > 0; sc--) { const c = cnt8[l * 256 + sc]; if (c >= left) { take[l] = left; left = 0; break; } left -= c; }
      thr[l] = left > 0 ? -1 : sc;
      perLevel.push(`${tot}->${Math.min(tot, nPer[l])}`);
    }
    const sel = new Float32Array(NF * 2 * 4);
    nSel = 0;
    for (let k = 0; k < n; k++) {
      const w = cw[k], l = (w >>> 19) & 7, sc = w >>> 22;
      if (sc > thr[l] || (sc === thr[l] && take[l]-- > 0)) {
        sel[nSel * 4] = w & 511; sel[nSel * 4 + 1] = (w >>> 9) & 1023; sel[nSel * 4 + 2] = l; nSel++;
        if (nSel >= NF * 2) break;
      }
    }
    rb1.unmap();
    const tb = performance.now();
    device.queue.writeBuffer(kpsB, 0, sel, 0, nSel * 4);
    device.queue.writeBuffer(cntB, 8, new Uint32Array([nSel]));
    enc = device.createCommandEncoder();
    const p = enc.beginComputePass();
    p.setPipeline(pipe.describe2); p.setBindGroup(0, gLev[0]); p.dispatchWorkgroups(Math.ceil(Math.max(nSel, 1) / 64));
    p.end();
    enc.copyBufferToBuffer(kpsB, 0, rb2, 0, NF * 2 * 16);
    enc.copyBufferToBuffer(descB, 0, rb2, NF * 2 * 16, NF * 2 * 32);
    device.queue.submit([enc.finish()]);
    await rb2.mapAsync(GPUMapMode.READ);
    const dd = new Uint32Array(rb2.getMappedRange(NF * 2 * 16, 32)).slice(0, 8);  // touch the descriptors
    rb2.unmap();
    const tc = performance.now();
    if (i >= warm) { t1s.push(ta - t0); tCpu.push(tb - ta); t2s.push(tc - tb); tAll.push(tc - t0); }
    if (i === warm) log(`two passes: ${nCand} candidates (${nCand > CAPC ? `TRUNCATED to ${CAPC}` : 'all read back'}), ${nSel} keypoints chosen; per level candidates->kept ${perLevel.join(' ')}; first descriptor word ${dd[0] >>> 0}`);
  }
  log(`two passes, total: ${fmt(stats(tAll))}`);
  log(`  pass 1 (upload, pyramid, FAST score, per-cell threshold, NMS, ${(RB1 / 1024).toFixed(0)} KB candidates back; the blur is queued behind it): ${fmt(stats(t1s))}`);
  log(`  CPU keypoint choice (JS stand-in for the octree / SSC): ${fmt(stats(tCpu))}`);
  log(`  pass 2 (keypoints up, waits for the blur, orientation + descriptors, ${(RB2 / 1024).toFixed(0)} KB back): ${fmt(stats(t2s))}`);

  // 5) where pass 1's GPU time goes: the stages up to each one, until the GPU is done (no readback)
  for (let upto = 1; upto <= STAGES.length; upto++) {
    const tt = [];
    for (let i = 0; i < warm + 40; i++) {
      const t0 = performance.now();
      device.queue.submit([encodePass1(upto).finish()]);
      await device.queue.onSubmittedWorkDone();
      if (i >= warm) tt.push(performance.now() - t0);
    }
    log(`pass 1 up to ${STAGES[upto - 1]}, GPU done: ${fmt(stats(tt))}`);
  }
  }  // FAST variants
  device.destroy();
  return { adapter: { vendor: info.vendor, architecture: info.architecture, description: info.description }, empty: stats(tEmpty), full: stats(tFull), submit: stats(tSubmit), gpu: stats(tGpu), keypoints: count,
           twoPass: { total: stats(tAll), pass1: stats(t1s), cpu: stats(tCpu), pass2: stats(t2s), candidates: nCand, chosen: nSel, frame: frameName } };
}
