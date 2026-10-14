// Trajectory alignment and ATE in the browser (live accuracy in the viewer).
// Closed-form Sim(3)/SE(3) alignment: Horn (1987) quaternion method for the rotation, Umeyama (1991) scale.
// Matches eval/evaluate.py (evo) up to floating-point and timestamp-association details; the offline evaluation
// remains the reference for reported numbers.

// Symmetric 4x4 eigen-decomposition (cyclic Jacobi). Returns the eigenvector of the largest eigenvalue.
function maxEigenvector4(A) {
  const a = A.map((r) => r.slice());
  const v = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
  for (let sweep = 0; sweep < 50; sweep++) {
    let off = 0;
    for (let p = 0; p < 4; p++) for (let q = p + 1; q < 4; q++) off += a[p][q] * a[p][q];
    if (off < 1e-30) break;
    for (let p = 0; p < 4; p++) {
      for (let q = p + 1; q < 4; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < 4; k++) {  // A <- A J
          const akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < 4; k++) {  // A <- J^T A
          const apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < 4; k++) {  // V <- V J
          const vkp = v[k][p], vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq; v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  let best = 0;
  for (let i = 1; i < 4; i++) if (a[i][i] > a[best][best]) best = i;
  return [v[0][best], v[1][best], v[2][best], v[3][best]];
}

export function quatToR(w, x, y, z) {
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ];
}

const mulRv = (R, p) => [R[0] * p[0] + R[1] * p[1] + R[2] * p[2], R[3] * p[0] + R[4] * p[1] + R[5] * p[2],
                         R[6] * p[0] + R[7] * p[1] + R[8] * p[2]];

// Find s, R, t minimising sum |gt_i - (s R est_i + t)|^2. est, gt: arrays of [x, y, z].
// Returns { s, R (row-major 9), t, rmse } or null for fewer than 3 pairs.
export function alignSim3(est, gt, withScale = true) {
  const n = est.length;
  if (n < 3) return null;
  const mx = [0, 0, 0], my = [0, 0, 0];
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) { mx[k] += est[i][k]; my[k] += gt[i][k]; }
  for (let k = 0; k < 3; k++) { mx[k] /= n; my[k] /= n; }
  const S = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];  // S[a][b] = sum x'_a y'_b
  let sx = 0;
  for (let i = 0; i < n; i++) {
    const x = [est[i][0] - mx[0], est[i][1] - mx[1], est[i][2] - mx[2]];
    const y = [gt[i][0] - my[0], gt[i][1] - my[1], gt[i][2] - my[2]];
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) S[a][b] += x[a] * y[b];
    sx += x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
  }
  const [[Sxx, Sxy, Sxz], [Syx, Syy, Syz], [Szx, Szy, Szz]] = S;
  const N = [
    [Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx],
    [Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz],
    [Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy],
    [Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz],
  ];
  const q = maxEigenvector4(N);
  const R = quatToR(q[0], q[1], q[2], q[3]);
  let s = 1;
  if (withScale && sx > 0) {
    let num = 0;
    for (let i = 0; i < n; i++) {
      const rx = mulRv(R, [est[i][0] - mx[0], est[i][1] - mx[1], est[i][2] - mx[2]]);
      num += rx[0] * (gt[i][0] - my[0]) + rx[1] * (gt[i][1] - my[1]) + rx[2] * (gt[i][2] - my[2]);
    }
    s = num / sx;
  }
  const Rm = mulRv(R, mx);
  const t = [my[0] - s * Rm[0], my[1] - s * Rm[1], my[2] - s * Rm[2]];
  let se = 0;
  for (let i = 0; i < n; i++) {
    const p = mulRv(R, est[i]);
    const dx = s * p[0] + t[0] - gt[i][0], dy = s * p[1] + t[1] - gt[i][1], dz = s * p[2] + t[2] - gt[i][2];
    se += dx * dx + dy * dy + dz * dz;
  }
  return { s, R, t, rmse: Math.sqrt(se / n) };
}

// Rotation ATE (deg, RMS) after alignment rotation R: angle of R_gt_i^T R R_est_i. Rotations row-major 9.
export function rotationRmse(R, estR, gtR) {
  let se = 0;
  for (let i = 0; i < estR.length; i++) {
    const A = estR[i], G = gtR[i];
    // M = R * A
    const M = new Array(9);
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++)
      M[r * 3 + c] = R[r * 3] * A[c] + R[r * 3 + 1] * A[3 + c] + R[r * 3 + 2] * A[6 + c];
    // trace(G^T M)
    let tr = 0;
    for (let k = 0; k < 9; k++) tr += G[k] * M[k];
    const ang = Math.acos(Math.max(-1, Math.min(1, (tr - 1) / 2)));
    se += ang * ang;
  }
  return estR.length ? Math.sqrt(se / estR.length) * 180 / Math.PI : NaN;
}

// TUM text -> { t: Float64Array, p: [[x,y,z]], R: [row-major 9] } sorted by time
export function parseTum(text) {
  const rows = [];
  for (const line of text.split('\n')) {
    if (!line.trim() || line.startsWith('#')) continue;
    const v = line.trim().split(/\s+/).map(Number);
    if (v.length >= 8 && v.every(Number.isFinite)) rows.push(v);
  }
  rows.sort((a, b) => a[0] - b[0]);
  return {
    t: Float64Array.from(rows.map((r) => r[0])),
    p: rows.map((r) => [r[1], r[2], r[3]]),
    R: rows.map((r) => quatToR(r[7], r[4], r[5], r[6])),
  };
}

// index of the GT sample nearest to time t, or -1 if further than maxDiff seconds
export function nearest(tArr, t, maxDiff = 0.01) {
  let lo = 0, hi = tArr.length - 1;
  if (hi < 0) return -1;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (tArr[mid] < t) lo = mid + 1; else hi = mid; }
  let best = lo;
  if (lo > 0 && Math.abs(tArr[lo - 1] - t) < Math.abs(tArr[lo] - t)) best = lo - 1;
  return Math.abs(tArr[best] - t) <= maxDiff ? best : -1;
}
