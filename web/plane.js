// Plane detection on SLAM map points for AR anchoring.
// Plane: n·p + d = 0 with |n| = 1. RANSAC on 3-point samples (optionally constrained to a normal prior, e.g. gravity),
// then least-squares refinement on the inliers (smallest-eigenvalue direction of the covariance).

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => Math.hypot(a[0], a[1], a[2]);
export const normalize = (a) => { const n = norm(a) || 1; return [a[0] / n, a[1] / n, a[2] / n]; };

// eigenvector of the smallest eigenvalue of a symmetric 3x3 matrix (cyclic Jacobi)
function smallestEigenvector3(A) {
  const a = A.map((r) => r.slice());
  const v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 50; sweep++) {
    const off = a[0][1] ** 2 + a[0][2] ** 2 + a[1][2] ** 2;
    if (off < 1e-24) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      if (Math.abs(a[p][q]) < 1e-300) continue;
      const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 3; k++) { const x = a[k][p], y = a[k][q]; a[k][p] = c * x - s * y; a[k][q] = s * x + c * y; }
      for (let k = 0; k < 3; k++) { const x = a[p][k], y = a[q][k]; a[p][k] = c * x - s * y; a[q][k] = s * x + c * y; }
      for (let k = 0; k < 3; k++) { const x = v[k][p], y = v[k][q]; v[k][p] = c * x - s * y; v[k][q] = s * x + c * y; }
    }
  }
  let best = 0;
  for (let i = 1; i < 3; i++) if (a[i][i] < a[best][best]) best = i;
  return [v[0][best], v[1][best], v[2][best]];
}

function refine(points, idx) {
  const c = [0, 0, 0];
  for (const i of idx) for (let k = 0; k < 3; k++) c[k] += points[i][k];
  for (let k = 0; k < 3; k++) c[k] /= idx.length;
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const i of idx) {
    const d = sub(points[i], c);
    for (let r = 0; r < 3; r++) for (let s = 0; s < 3; s++) C[r][s] += d[r] * d[s];
  }
  const n = normalize(smallestEigenvector3(C));
  return { n, d: -dot(n, c), centroid: c };
}

// points: [[x,y,z]]; opts: { thresh, iters, prior (unit normal), maxAngleDeg, minInliers }
// Returns { n, d, centroid, inliers: [indices] } or null.
export function fitPlaneRansac(points, { thresh, iters = 300, prior = null, maxAngleDeg = 20, minInliers = 15 } = {}) {
  const N = points.length;
  if (N < Math.max(3, minInliers)) return null;
  const cosMax = Math.cos(maxAngleDeg * Math.PI / 180);
  let best = null, bestCount = 0;
  let seed = 12345;
  const rnd = () => (seed = (seed * 1103515245 + 12345) >>> 0) % N;
  for (let it = 0; it < iters; it++) {
    const a = points[rnd()], b = points[rnd()], c = points[rnd()];
    let n = cross(sub(b, a), sub(c, a));
    if (norm(n) < 1e-9) continue;
    n = normalize(n);
    if (prior && Math.abs(dot(n, prior)) < cosMax) continue;
    const d = -dot(n, a);
    let count = 0;
    for (let i = 0; i < N; i++) if (Math.abs(dot(n, points[i]) + d) < thresh) count++;
    if (count > bestCount) { bestCount = count; best = { n, d }; }
  }
  if (!best || bestCount < minInliers) return null;
  let inl = [];
  for (let i = 0; i < N; i++) if (Math.abs(dot(best.n, points[i]) + best.d) < thresh) inl.push(i);
  let pl = refine(points, inl);
  if (prior && Math.abs(dot(pl.n, prior)) < cosMax) pl = { ...best, centroid: pl.centroid };  // keep the constrained fit
  inl = [];
  for (let i = 0; i < N; i++) if (Math.abs(dot(pl.n, points[i]) + pl.d) < thresh) inl.push(i);
  return { ...pl, inliers: inl };
}

// Intersection of the ray o + t·dir (t > 0) with the plane, or null.
export function rayPlane(o, dir, plane) {
  const den = dot(plane.n, dir);
  if (Math.abs(den) < 1e-9) return null;
  const t = -(dot(plane.n, o) + plane.d) / den;
  return t > 0 ? [o[0] + t * dir[0], o[1] + t * dir[1], o[2] + t * dir[2]] : null;
}

export { dot, cross, sub, norm };
