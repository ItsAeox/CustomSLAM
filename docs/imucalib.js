// Camera–IMU time offset and focal-length error measured on the phone (option "Record gyro + poses"), the browser
// version of eval/estimate_time_offset.py --focal (same method, tested against it in scripts/imucalib_check.mjs).
//   offset: the gyro rate (device axes -> camera axes) averaged over each frame interval is compared with the camera's
//           visual angular rate for lags of -200..+200 ms; Huber cost (a few bad pose pairs must not decide)
//   focal:  with an assumed focal f_a, rotations about the image x / y axes come out scaled by f_true / f_a, roll does
//           not (Karpenko et al. 2011) -> least-squares rate ratio at the best lag
// Poses: [t, qx, qy, qz, qw, maps] (Twc, frame time on the IMU clock); gyro: [t, wx, wy, wz] (rad/s, device axes).

const HUBER = 0.5;  // rad/s

function quatMul(a, b) {
  return [a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1], a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
          a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3], a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]];
}
function rotvec(q) {  // unit quaternion -> rotation vector
  let [x, y, z, w] = q;
  if (w < 0) { x = -x; y = -y; z = -z; w = -w; }
  const s = Math.hypot(x, y, z);
  if (s < 1e-12) return [2 * x, 2 * y, 2 * z];
  const ang = 2 * Math.atan2(s, w);
  return [x / s * ang, y / s * ang, z / s * ang];
}

// visual angular rate (camera frame) at the midpoints of consecutive poses
function cameraRates(poses) {
  const out = [];
  for (let k = 0; k + 1 < poses.length; k++) {
    const a = poses[k], b = poses[k + 1], dt = b[0] - a[0];
    if (!(dt > 1e-3 && dt < 0.2) || a[5] !== b[5]) continue;  // gaps (lost, new map) or a map change
    const rel = quatMul([-a[1], -a[2], -a[3], a[4]], [b[1], b[2], b[3], b[4]]);  // R_a^T R_b
    const v = rotvec(rel);
    out.push([0.5 * (a[0] + b[0]), v[0] / dt, v[1] / dt, v[2] / dt]);
  }
  return out;
}

// Rcb: device -> camera rotation (row-major 3x3); phone rear camera, portrait: diag(1, -1, -1)
export function estimateImuCalib(poses, gyro, { Rcb = [1, 0, 0, 0, -1, 0, 0, 0, -1], maxMs = 200, stepMs = 1 } = {}) {
  if (poses.length < 50 || gyro.length < 50) return null;
  const cam = cameraRates(poses);
  if (cam.length < 50) return null;
  const n = gyro.length, ti = new Float64Array(n), cs = [new Float64Array(n + 1), new Float64Array(n + 1), new Float64Array(n + 1)];
  for (let i = 0; i < n; i++) ti[i] = gyro[i][0];
  for (let i = 0; i < n; i++) {  // integral of the camera-frame rate (np.gradient-style time steps)
    const g = gyro[i], w = [Rcb[0] * g[1] + Rcb[1] * g[2] + Rcb[2] * g[3], Rcb[3] * g[1] + Rcb[4] * g[2] + Rcb[5] * g[3],
                             Rcb[6] * g[1] + Rcb[7] * g[2] + Rcb[8] * g[3]];
    const dt = n === 1 ? 0 : i === 0 ? ti[1] - ti[0] : i === n - 1 ? ti[n - 1] - ti[n - 2] : 0.5 * (ti[i + 1] - ti[i - 1]);
    for (let c = 0; c < 3; c++) cs[c][i + 1] = cs[c][i] + w[c] * dt;
  }
  const dts = [];
  for (let k = 1; k < poses.length; k++) if (poses[k][0] > poses[k - 1][0]) dts.push(poses[k][0] - poses[k - 1][0]);
  dts.sort((a, b) => a - b);
  const period = dts[dts.length >> 1];
  const lower = (x) => { let lo = 0, hi = n; while (lo < hi) { const m = (lo + hi) >> 1; if (ti[m] < x) lo = m + 1; else hi = m; } return lo; };
  const gyroAt = (lag) => {  // mean gyro rate over each frame interval at this lag, with the matching visual rates
    const g = [], v = [];
    for (const [tc, ...wc] of cam) {
      const lo = tc - period / 2 + lag, hi = tc + period / 2 + lag;
      if (lo < ti[0] || hi > ti[n - 1]) continue;
      const il = lower(lo), ih = lower(hi), d = ti[Math.min(ih, n - 1)] - ti[il] + 1e-9;
      g.push([0, 1, 2].map((c) => (cs[c][ih] - cs[c][il]) / d));
      v.push(wc);
    }
    return [g, v];
  };
  let best = null;
  for (let ms = -maxMs; ms <= maxMs + 1e-9; ms += stepMs) {
    const [g, v] = gyroAt(ms / 1000);
    if (g.length < 50) continue;
    let err = 0;
    for (let k = 0; k < g.length; k++) {
      const r = Math.hypot(g[k][0] - v[k][0], g[k][1] - v[k][1], g[k][2] - v[k][2]);
      err += r <= HUBER ? 0.5 * r * r : HUBER * (r - 0.5 * HUBER);
    }
    err /= g.length;
    if (!best || err < best.cost) best = { offsetMs: ms, cost: err, samples: g.length };
  }
  if (!best) return null;
  const [g, v] = gyroAt(best.offsetMs / 1000);
  const ratio = (gs, vs) => {  // least squares, |rate| > 0.2 rad/s, refit without the 10 % largest residuals
    const m = gs.map((x, i) => [x, vs[i]]).filter(([x]) => Math.abs(x) > 0.2);
    if (m.length < 30) return { k: NaN, n: m.length };
    const fit = (p) => p.reduce((s, [x, y]) => s + x * y, 0) / p.reduce((s, [x]) => s + x * x, 0);
    const k0 = fit(m), res = m.map(([x, y]) => Math.abs(y - k0 * x)), cut = [...res].sort((a, b) => a - b)[Math.floor(0.9 * (res.length - 1))];
    const keep = m.filter((_, i) => res[i] <= cut);
    return { k: fit(keep), n: keep.length };
  };
  const xy = ratio([...g.map((x) => x[0]), ...g.map((x) => x[1])], [...v.map((x) => x[0]), ...v.map((x) => x[1])]);
  const z = ratio(g.map((x) => x[2]), v.map((x) => x[2]));
  return { ...best, periodMs: period * 1000, focalRatio: xy.k, focalSamples: xy.n, rollRatio: z.k, rollSamples: z.n };
}
