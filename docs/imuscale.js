// Metric scale of the monocular map from the accelerometer (option "Record gyro + poses"), the browser version of
// eval/scale_from_accel.py (same method; checked against it by scripts/imuscale_check.mjs). Over windows of the
// trajectory the specific force, rotated into the map frame with the visual rotation, integrates to the velocity change
// minus gravity:  s (v1 - v0) = sum R_wb (f - b) dt + g T. Unknowns: s (metres per map unit), gravity g in the map
// frame, accelerometer bias b (device axes). Linear least squares, then a few Gauss-Newton steps with |g| = 9.81 and
// Huber weights. The camera-IMU lever arm is ignored.
// Poses: [t, qx, qy, qz, qw, maps, x, y, z] (Twc, frame time on the IMU clock; as recorded for imucalib.js plus the
// position); acc: [t, ax, ay, az] (m/s^2, accelerationIncludingGravity = specific force, device axes).

const G = 9.81;

function quatToR([x, y, z, w]) {  // row-major 3x3
  return [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
          2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
          2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)];
}
const mul3 = (A, B) => { const C = new Array(9).fill(0); for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) C[3 * i + j] += A[3 * i + k] * B[3 * k + j]; return C; };
function nlerp(a, b, u) {  // quaternion interpolation (short steps between frames)
  const d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3], sg = d < 0 ? -1 : 1;
  const q = [0, 1, 2, 3].map((i) => (1 - u) * a[i] + u * sg * b[i]), n = Math.hypot(...q);
  return q.map((v) => v / n);
}
function solve(M, v) {  // Gaussian elimination with partial pivoting (small dense systems)
  const n = v.length, A = M.map((r, i) => [...r, v[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    if (Math.abs(A[p][c]) < 1e-12) return null;
    [A[c], A[p]] = [A[p], A[c]];
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = A[r][c] / A[c][c];
      for (let k = c; k <= n; k++) A[r][k] -= f * A[c][k];
    }
  }
  return A.map((r, i) => r[n] / r[i]);
}

// Rcb: device -> camera rotation (row-major); the phone's rear camera in portrait is diag(1, -1, -1)
export function estimateScale(poses, acc, { Rcb = [1, 0, 0, 0, -1, 0, 0, 0, -1], window = 2.0, maxGap = 0.2 } = {}) {
  if (poses.length < 50 || acc.length < 50) return null;
  // segments without gaps or map changes
  const segs = [];
  let s0 = 0;
  for (let i = 1; i <= poses.length; i++) {
    if (i === poses.length || poses[i][0] - poses[i - 1][0] > maxGap || poses[i][5] !== poses[i - 1][5]) {
      if (i - s0 > 10) segs.push(poses.slice(s0, i));
      s0 = i;
    }
  }
  const rows = [], rhs = [];
  let ai = 0;
  for (const P of segs) {
    if (P[P.length - 1][0] - P[0][0] < 2 * window) continue;
    const k = 2, n = P.length;
    const ps = P.map((_, i) => { const lo = Math.max(0, i - k), hi = Math.min(n - 1, i + k); const m = [0, 0, 0];
      for (let j = lo; j <= hi; j++) for (let d = 0; d < 3; d++) m[d] += P[j][6 + d] / (hi - lo + 1); return m; });
    const v = ps.map((_, i) => { const a = Math.max(0, i - 1), b = Math.min(n - 1, i + 1), dt = P[b][0] - P[a][0];
      return [0, 1, 2].map((d) => (ps[b][d] - ps[a][d]) / dt); });
    let i0 = k;
    for (;;) {
      let i1 = i0; while (i1 < n && P[i1][0] < P[i0][0] + window) i1++;
      if (i1 >= n - k) break;
      const t0 = P[i0][0], t1 = P[i1][0];
      while (ai > 0 && acc[ai][0] > t0) ai--;
      while (ai < acc.length && acc[ai][0] < t0) ai++;
      const Af = [0, 0, 0], AR = new Array(9).fill(0);
      let T = 0, cnt = 0, pi = i0;
      for (let a = ai; a < acc.length && acc[a][0] <= t1; a++) {
        const ta = acc[a][0], dt = a + 1 < acc.length ? Math.min(acc[a + 1][0] - ta, 0.1) : 0;
        while (pi + 1 < n && P[pi + 1][0] < ta) pi++;
        const pa = P[pi], pb = P[Math.min(pi + 1, n - 1)], u = pb[0] > pa[0] ? Math.min(1, Math.max(0, (ta - pa[0]) / (pb[0] - pa[0]))) : 0;
        const Rwb = mul3(quatToR(nlerp(pa.slice(1, 5), pb.slice(1, 5), u)), Rcb);  // R_wc R_cb
        for (let r = 0; r < 3; r++) {
          Af[r] += (Rwb[3 * r] * acc[a][1] + Rwb[3 * r + 1] * acc[a][2] + Rwb[3 * r + 2] * acc[a][3]) * dt;
          for (let c = 0; c < 3; c++) AR[3 * r + c] += Rwb[3 * r + c] * dt;
        }
        T += dt; cnt++;
      }
      if (cnt >= 5) {
        for (let r = 0; r < 3; r++) {
          rows.push([v[i1][r] - v[i0][r], r === 0 ? -T : 0, r === 1 ? -T : 0, r === 2 ? -T : 0, AR[3 * r], AR[3 * r + 1], AR[3 * r + 2]]);
          rhs.push(Af[r]);
        }
      }
      let next = i0 + 1; while (next < n && P[next][0] < P[i0][0] + window / 2) next++;
      i0 = next;
    }
  }
  const nWin = rows.length / 3;
  if (nWin < 10) return null;
  const normal = (w, extra) => {  // weighted normal equations (+ the |g| constraint row)
    const N = Array.from({ length: 7 }, () => new Array(7).fill(0)), b = new Array(7).fill(0);
    rows.forEach((r, i) => { for (let a = 0; a < 7; a++) { b[a] += w[i] * r[a] * rhs[i]; for (let c = 0; c < 7; c++) N[a][c] += w[i] * r[a] * r[c]; } });
    if (extra) { const [J, res, wc] = extra; for (let a = 0; a < 7; a++) { b[a] += wc * J[a] * res; for (let c = 0; c < 7; c++) N[a][c] += wc * J[a] * J[c]; } }
    return [N, b];
  };
  let x = solve(...normal(rows.map(() => 1)));
  if (!x) return null;
  const gLin = Math.hypot(x[1], x[2], x[3]);
  if (gLin > 1e-6) for (let d = 1; d <= 3; d++) x[d] *= G / gLin;
  if (x[0] <= 0) x[0] = Math.abs(x[0]) + 1e-3;
  for (let it = 0; it < 15; it++) {  // Gauss-Newton on r = A x - b (Huber 0.1) and 10 (|g| - 9.81)
    const w = rows.map((r, i) => { const e = Math.abs(r.reduce((s, a, j) => s + a * x[j], 0) - rhs[i]); return e <= 0.1 ? 1 : 0.1 / e; });
    const gn = Math.hypot(x[1], x[2], x[3]);
    const J = [0, x[1] / gn, x[2] / gn, x[3] / gn, 0, 0, 0];
    // linearised: J dx = -(gn - G)  ->  normal equations for the full x: J x = J x0 - (gn - G)
    const target = J.reduce((s, a, j) => s + a * x[j], 0) - (gn - G);
    const nx = solve(...normal(w, [J, target, 100]));
    if (!nx) break;
    const step = Math.hypot(...nx.map((v, i) => v - x[i]));
    x = nx;
    if (step < 1e-6) break;
  }
  const res = rows.map((r, i) => r.reduce((s, a, j) => s + a * x[j], 0) - rhs[i]);
  return { scale: x[0], gLinear: gLin, bias: Math.hypot(x[4], x[5], x[6]), windows: nWin,
           rms: Math.sqrt(res.reduce((s, e) => s + e * e, 0) / res.length), span: poses[poses.length - 1][0] - poses[0][0] };
}
