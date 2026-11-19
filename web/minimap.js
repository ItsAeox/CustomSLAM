// Small 3D view of the map for the AR page (the dataset viewer's 3D view, web/viewer.js, in a corner): map points
// (local map red), keyframes (blue), covisibility graph (green), the live camera (green frustum), its trajectory
// (orange), the detected floor (grid) and the placed tree (green dot). Own WebGL canvas with orbit controls (drag:
// rotate, pinch / wheel: zoom, two fingers: pan), so touches there never reach the AR view. Renders ~15 fps and
// only while visible.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const FRUSTUM = (w) => {  // MapDrawer::DrawKeyFrames: w, h = 0.75 w, z = 0.6 w
  const h = w * 0.75, z = w * 0.6;
  const P = [[0, 0, 0], [w, h, z], [w, -h, z], [-w, -h, z], [-w, h, z]];
  return [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]].flatMap(([a, b]) => [P[a], P[b]]);
};
const KF_VERTS = FRUSTUM(0.05), CAM_VERTS = FRUSTUM(0.08);
function frustumInto(out, o, R, t, verts) {  // R row-major 9, t 3
  for (const p of verts) {
    out[o++] = R[0] * p[0] + R[1] * p[1] + R[2] * p[2] + t[0];
    out[o++] = R[3] * p[0] + R[4] * p[1] + R[5] * p[2] + t[1];
    out[o++] = R[6] * p[0] + R[7] * p[1] + R[8] * p[2] + t[2];
  }
  return o;
}

export function createMiniMap(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  container.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff);
  const camera = new THREE.PerspectiveCamera(50, 1, 0.001, 2000);
  camera.up.set(0, -1, 0);  // ORB-SLAM3 world: y down, the first camera looks along +z
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  new ResizeObserver(() => {
    const w = container.clientWidth, h = container.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / Math.max(h, 1);
    camera.updateProjectionMatrix();
  }).observe(container);

  const pxr = renderer.getPixelRatio();
  const mk = (geomObj) => { geomObj.frustumCulled = false; scene.add(geomObj); return geomObj; };
  const points = mk(new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ size: 1.5 * pxr, sizeAttenuation: false, vertexColors: true })));
  const kfs = mk(new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x0000ff })));
  const graph = mk(new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x00b000, transparent: true, opacity: 0.5 })));
  const cam = mk(new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x00b000 })));
  const traj = mk(new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xff8800 })));
  const tree = mk(new THREE.Mesh(new THREE.SphereGeometry(1, 12, 8), new THREE.MeshBasicMaterial({ color: 0x16a34a, depthTest: false })));
  tree.visible = false;
  let grid = null;
  const trajPts = [];
  const target = new THREE.Vector3();
  const show = { points: true, keyframes: true, graph: false, trajectory: true, grid: true, tree: true, follow: true };

  const setPositions = (obj, arr, count) => {
    obj.geometry.setAttribute('position', new THREE.BufferAttribute(arr, 3));
    obj.geometry.setDrawRange(0, count ?? arr.length / 3);
  };
  function resetView() {  // ORB-SLAM3's default viewpoint, a bit further out for the small window
    camera.position.set(target.x, target.y - 1.0, target.z - 2.4);
    controls.target.copy(target);
    controls.update();
  }
  resetView();

  let last = 0, visible = true;
  function render(t) {
    requestAnimationFrame(render);
    if (!visible || t - last < 60) return;  // ~15 fps
    last = t;
    points.visible = show.points; kfs.visible = show.keyframes; graph.visible = show.graph;
    traj.visible = show.trajectory; if (grid) grid.visible = show.grid;
    tree.visible = tree.userData.placed && show.tree;
    controls.update();
    renderer.render(scene, camera);
  }
  requestAnimationFrame(render);

  return {
    show,
    resetView,
    setVisible(v) { visible = v; },
    updateMap(v) {  // v.mps: [x, y, z, local]*, v.kfs: [R row-major 9, t 3]*, v.edges: segment endpoints
      const n = v.mps.length / 4, pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        pos[3 * i] = v.mps[4 * i]; pos[3 * i + 1] = v.mps[4 * i + 1]; pos[3 * i + 2] = v.mps[4 * i + 2];
        if (v.mps[4 * i + 3] > 0) col[3 * i] = 1;  // local map red, others black
      }
      points.geometry.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      points.geometry.setAttribute('color', new THREE.BufferAttribute(col, 3));
      const nk = v.kfs.length / 12, kf = new Float32Array(nk * KF_VERTS.length * 3);
      let o = 0;
      for (let i = 0; i < nk; i++) o = frustumInto(kf, o, v.kfs.subarray(12 * i, 12 * i + 9), v.kfs.subarray(12 * i + 9, 12 * i + 12), KF_VERTS);
      setPositions(kfs, kf);
      setPositions(graph, v.edges);
    },
    updateCamera(twc) {  // column-major 4x4 camera-to-world
      const R = [twc[0], twc[4], twc[8], twc[1], twc[5], twc[9], twc[2], twc[6], twc[10]], t = [twc[12], twc[13], twc[14]];
      const arr = new Float32Array(CAM_VERTS.length * 3);
      frustumInto(arr, 0, R, t, CAM_VERTS);
      setPositions(cam, arr);
      trajPts.push(t[0], t[1], t[2]);
      if (trajPts.length > 3 * 3000) trajPts.splice(0, 3 * 500);
      setPositions(traj, new Float32Array(trajPts));
      if (show.follow) {  // keep the viewing offset, move with the camera (ORB-SLAM3 "Follow Camera")
        const d = new THREE.Vector3(t[0], t[1], t[2]).sub(controls.target);
        controls.target.add(d);
        camera.position.add(d);
      }
      target.set(t[0], t[1], t[2]);
    },
    clearTrajectory() { trajPts.length = 0; setPositions(traj, new Float32Array(0)); },
    setFloor(f) {  // {n, d, centroid, height} or null: a grid on the floor plane
      if (grid) { scene.remove(grid); grid = null; }
      if (!f) return;
      grid = new THREE.GridHelper(1, 10, 0x22c55e, 0x9ca3af);
      const k = f.n[0] * f.centroid[0] + f.n[1] * f.centroid[1] + f.n[2] * f.centroid[2] + f.d;
      grid.position.set(f.centroid[0] - f.n[0] * k, f.centroid[1] - f.n[1] * k, f.centroid[2] - f.n[2] * k);
      grid.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(...f.n));
      grid.scale.setScalar(f.height * 6);
      grid.frustumCulled = false;
      scene.add(grid);
    },
    setTree(p, size) {  // position [x, y, z] or null
      tree.userData.placed = !!p;
      if (!p) return;
      tree.position.set(p[0], p[1], p[2]);
      tree.scale.setScalar(Math.max(size * 0.35, 0.02));
      tree.renderOrder = 10;  // drawn over the points
    },
  };
}
